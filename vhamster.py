#!/usr/bin/env python3
"""
VHAMSTeR
==============================================================
Virus Host Assignment Model using Sequence Transformers and Reading-frames.
Runs 5-fold deep ensemble inference with per-fold XGBoost stacking 
and applies length-aware continuous vector calibration.
"""

import gc
import json
import multiprocessing
import pathlib
import pickle
import re
import shutil
import sys
import sysconfig
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import click
import joblib
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import xgboost as xgb
from loguru import logger
from peft import PeftModel
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("vhamster")
except Exception:
    try:
        __version__ = (pathlib.Path(__file__).resolve().parent / "VERSION").read_text().strip()
    except FileNotFoundError:
        __version__ = "unknown"

# ── src/ on path ──────────────────────────────────────────────────────────────
_ROOT = pathlib.Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from genomad_markers import extract_genomad_markers, write_annotation_rows
from features import (
    ARCH_FEATURE_NAMES,
    build_xgb1_marker_features,
    build_xgb2_marker_features,
    extract_features_worker,
)
from models import GenomeClassifier, load_model_and_tokenizer, make_collate_fn
from sequences import GenomeDataset, load_fasta_sequences


def _default_model_root() -> pathlib.Path:
    purelib = sysconfig.get_path("purelib")
    if purelib is None:
        return _ROOT / "model"
    return pathlib.Path(purelib) / "vhamster_models_v1.2.0"


_DEFAULT_MODEL_ROOT = _default_model_root()


def _discover_fold_dirs(args: Any) -> List[pathlib.Path]:
    """Resolve fold directories from --fold-dirs or --ensemble-dir."""
    if args.fold_dirs:
        fold_dirs = [pathlib.Path(p).resolve() for p in args.fold_dirs]
        missing = [str(p) for p in fold_dirs if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Fold directory(ies) not found: {missing}")
        return fold_dirs

    if args.ensemble_dir is None:
        raise ValueError("Provide either --ensemble-dir or --fold-dirs.")

    ensemble_dir = pathlib.Path(args.ensemble_dir).resolve()
    if not ensemble_dir.exists():
        raise FileNotFoundError(f"Ensemble directory not found: {ensemble_dir}")

    fold_dirs = [p for p in ensemble_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")]
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* directories found under {ensemble_dir}.")

    def _fold_sort_key(p: pathlib.Path) -> Tuple[int, str]:
        m = re.search(r"fold_(\d+)$", p.name)
        return (int(m.group(1)), p.name) if m else (10**9, p.name)

    return sorted(fold_dirs, key=_fold_sort_key)


def _load_fold_config(fold_dir: pathlib.Path, checkpoint_subdir: str) -> Dict:
    candidates = [
        fold_dir / "config.json",
        fold_dir / checkpoint_subdir / "config.json",
    ]
    for cfg in candidates:
        if cfg.exists():
            with open(cfg, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"No config.json found for fold {fold_dir}. Tried: {[str(c) for c in candidates]}")


def _load_calibration_params(path: Optional[Union[str, pathlib.Path]]) -> Optional[Dict]:
    """Load vector scaling parameters (w0, w1, length_scale) from JSON."""
    if path is None:
        return None
    p = pathlib.Path(str(path))
    if not p.exists():
        logger.warning(f"Calibration JSON not found: {p}. Using uncalibrated logits.")
        return None
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded length-aware vector scaling parameters from {p}")
    return data


def _apply_calibration(logits: torch.Tensor, calib_params: Optional[Dict], chunk_lengths: List[int], num_classes: int) -> np.ndarray:
    """Apply length-aware vector scaling calibration to a set of logits."""
    if calib_params and "classes" in calib_params:
        length_scale = float(calib_params.get("length_scale", 1000.0))
        w0 = torch.tensor([calib_params["classes"].get(str(c), {"w0": 0.0})["w0"] for c in range(num_classes)], dtype=torch.float32)
        w1 = torch.tensor([calib_params["classes"].get(str(c), {"w1": 0.0})["w1"] for c in range(num_classes)], dtype=torch.float32)
        lengths_tensor = torch.tensor(chunk_lengths, dtype=torch.float32).clamp(min=1e-6)
        log_norm_lengths = torch.log(lengths_tensor / length_scale)
        temps = torch.exp(w0.unsqueeze(0) + w1.unsqueeze(0) * log_norm_lengths.unsqueeze(1))

        raw_probs = torch.softmax(logits, dim=-1)
        pred_cls = torch.argmax(logits, dim=-1)
        n = logits.shape[0]
        idx = torch.arange(n)
        top_temp = temps[idx, pred_cls]
        top_raw = raw_probs[idx, pred_cls]
        top_raw_clamp = top_raw.clamp(min=1e-7, max=1.0 - 1e-7)
        bce_logit = torch.log(top_raw_clamp / (1.0 - top_raw_clamp))
        cal_top = torch.sigmoid(bce_logit / top_temp)
        remaining_raw = (1.0 - top_raw).clamp(min=1e-8)
        remaining_cal = (1.0 - cal_top).clamp(min=0.0)
        scale = (remaining_cal / remaining_raw).unsqueeze(1)
        final_probs = raw_probs * scale
        final_probs[idx, pred_cls] = cal_top
        final_probs = final_probs / final_probs.sum(dim=-1, keepdim=True)
        return np.round(final_probs.numpy(), 4)
    else:
        return np.round(torch.softmax(logits, dim=-1).numpy(), 4)


def _reconstruct_hierarchical_probs(xgb1_probs, xgb2_probs, n_fine_classes, prokaryote_idx, euk_fine_indices):
    out = np.zeros((xgb1_probs.shape[0], n_fine_classes), dtype=np.float32)
    out[:, prokaryote_idx] = xgb1_probs[:, 0]
    p_euk = xgb1_probs[:, 1]
    for euk_col, fine_col in enumerate(euk_fine_indices):
        out[:, fine_col] = p_euk * xgb2_probs[:, euk_col]
    return out


def _aggregate_chunks(
    chunk_df: pl.DataFrame,
    class_cols: List[str],
    prok_col_idx: Optional[int],
    label_mapping: Dict[int, str],
) -> pl.DataFrame:
    """Max-score weighted average of per-class probabilities across chunks."""
    genome_col = (
        pl.col("accession")
        .str.replace(r"_chunk\d+_\d+$", "", literal=False)
        .alias("genome")
    )
    df = chunk_df.with_columns(genome_col)

    agg_exprs = [
        ((pl.col(c) * pl.col("confidence")).sum() / pl.col("confidence").sum()).alias(c)
        for c in class_cols
    ]
    genome_df = df.group_by("genome").agg(agg_exprs).sort("genome")

    prob_arr = genome_df.select(class_cols).to_numpy()
    pred_indices = np.argmax(prob_arr, axis=1)
    confidences = np.max(prob_arr, axis=1)
    predicted_hosts = [label_mapping.get(int(i), f"class_{i}") for i in pred_indices]

    genome_df = genome_df.with_columns([
        pl.Series("predicted_host", predicted_hosts),
        pl.Series("confidence", np.round(confidences, 4)),
    ])

    if prok_col_idx is not None:
        prok_col_name = class_cols[prok_col_idx]
        genome_df = genome_df.with_columns([
            pl.col(prok_col_name).alias("prokaryote_score"),
            (pl.lit(1.0) - pl.col(prok_col_name)).alias("eukaryote_score"),
        ])

    lead_cols = ["genome", "predicted_host", "confidence"]
    #rest = [c for c in genome_df.columns if c not in lead_cols]
    return genome_df.select(lead_cols)


def _chunk_sequences(seqs: List[str], accessions: List[str], chunk_size: int, overlap: int) -> Tuple[List[str], List[str]]:
    chunked_seqs, chunked_accs = [], []
    for seq, acc in zip(seqs, accessions):
        seq_len = len(seq)
        if seq_len <= chunk_size:
            chunked_seqs.append(seq)
            chunked_accs.append(acc)
        else:
            start = 0
            while start < seq_len:
                end = min(start + chunk_size, seq_len)
                chunked_seqs.append(seq[start:end])
                chunked_accs.append(f"{acc}_chunk{start}_{end}")
                if end == seq_len:
                    break
                start += chunk_size - overlap
    return chunked_seqs, chunked_accs


def _configure_logging(output_dir: pathlib.Path, prefix: str) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{prefix}.log"
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=False)
    logger.add(log_path, level="DEBUG", enqueue=True, backtrace=False, diagnose=False)
    return log_path


def _run(args: Any) -> None:
    logger.info("=" * 68)
    logger.info(f"VHAMSTeR v{__version__}")
    logger.info("Virus Host Assignment Model using Sequence Transformers and Reading-frame")
    logger.info("=" * 68)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── [1/5] Configuration & Pre-flight ─────────────────────────────────────
    ref_config = _load_fold_config(args.fold_dirs_resolved[0], args.checkpoint_subdir)
    _LABEL_RENAMES = {"Metazoa (All Animals)": "Animal", "Viridiplantae (Plants)": "Plant"}
    label_mapping = {int(k): _LABEL_RENAMES.get(str(v), str(v)) for k, v in ref_config.get("label_mapping", {}).items()}
    num_classes = len(label_mapping) if label_mapping else int(ref_config.get("num_classes", 5))
    model_type = ref_config.get("model_type", "nucleotidetransformer")
    max_length = int(ref_config.get("max_length", 10000))

    calib_params = _load_calibration_params(args.calibration_params)

    # ── [2/5] Sequence Chunking ──────────────────────────────────────────────
    logger.info("[2/5] Loading and chunking sequences")
    seqs, accessions = load_fasta_sequences(str(args.fasta))
    chunked_seqs, chunked_accs = _chunk_sequences(seqs, accessions, args.chunk_size, args.overlap)
    logger.info(f"{len(accessions)} sequence(s) -> {len(chunked_seqs)} chunk(s)")

    # ── [3/5] Global Feature Extraction (PyRodigal & geNomad MMseqs2) ────────
    if args.precomputed_features:
        logger.info("[3/5] Loading precomputed architectural features & geNomad hits")
        feat_dir = pathlib.Path(args.precomputed_features)
        arch_tsv = feat_dir / f"{args.prefix}.arch_features.tsv"
        hits_json = feat_dir / f"{args.prefix}.genomad_hits.json"
        feat_df = pl.read_csv(arch_tsv, separator="\t")
        precomp_accs = feat_df["accession"].to_list()
        features_df_arch = feat_df.drop("accession")
        arch_feature_names = features_df_arch.columns
        with open(hits_json) as _fh:
            genomad_marker_dict = json.load(_fh)
        annotation_rows = []
        gene_pred_src = feat_dir / f"{args.prefix}.gene_predictions.tsv"
        gene_table_path = args.output_dir / f"{args.prefix}.gene_predictions.tsv"
        if gene_pred_src.exists():
            shutil.copy2(gene_pred_src, gene_table_path)
            logger.info(f"      Gene prediction table copied to: {gene_table_path}")
        else:
            logger.warning(f"      Gene prediction table not found in features dir: {gene_pred_src}")
        # still need chunked_seqs for GLM tokenization — re-chunk from FASTA
        seqs, accessions = load_fasta_sequences(str(args.fasta))
        chunked_seqs, chunked_accs = _chunk_sequences(seqs, accessions, args.chunk_size, args.overlap)
        if chunked_accs != precomp_accs:
            raise click.ClickException(
                "Chunk accessions in precomputed features do not match the current FASTA + "
                "--chunk-size/--overlap. Ensure these parameters match the values used with "
                "vhamster-features."
            )
        logger.info(f"      Loaded {len(chunked_accs)} chunk(s) from precomputed features")
    else:
        logger.info("[3/5] Extracting global architectural features & geNomad hits")

        # 3a. PyRodigal
        arch_feature_names = list(ARCH_FEATURE_NAMES) + ["n_genes"]
        n_workers = min(multiprocessing.cpu_count(), len(chunked_seqs), 8)
        args_list = [(s, a, arch_feature_names, args.chunk_size) for s, a in zip(chunked_seqs, chunked_accs)]
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            raw_arch_features = list(tqdm(ex.map(extract_features_worker, args_list), total=len(chunked_seqs), desc="      PyRodigal"))

        raw_arch_array = np.array(raw_arch_features, dtype=np.float32)
        features_df_arch = pl.DataFrame(raw_arch_array, schema=arch_feature_names)

        # 3b. geNomad MMseqs2
        genomad_metadata = args.genomad_db / "genomad_marker_metadata.tsv"
        if not genomad_metadata.exists():
            raise FileNotFoundError(
                f"geNomad metadata file not found at expected location: {genomad_metadata}\n"
                "Ensure you are passing the root directory of a complete geNomad database."
            )

        mmseqs_threads = getattr(args, 'mmseqs_threads', 4)
        logger.info(f"      Running MMseqs2 against geNomad marker database ({mmseqs_threads} threads)...")
        seq_dict = dict(zip(chunked_accs, chunked_seqs))

        marker_hits, _, annotation_rows = extract_genomad_markers(
            sequences=seq_dict,
            genomad_db=args.genomad_db,
            genomad_metadata=genomad_metadata,
            threads=mmseqs_threads,
            return_details=True,
        )
        genomad_marker_dict = marker_hits

        gene_table_path = args.output_dir / f"{args.prefix}.gene_predictions.tsv"
        write_annotation_rows(annotation_rows, gene_table_path)
        logger.info(f"      Gene prediction table written to: {gene_table_path}")

    arch_cols_no_frag = [c for c in arch_feature_names if c != "fragment_size" and c != "n_genes"]
    
    # ── [4/5] Ensemble Inference Loop ────────────────────────────────────────
    logger.info("[4/5] Running ensemble inference across folds")
    accumulated_logits: Optional[torch.Tensor] = None
    per_fold_basic: List[Dict] = []
    per_fold_verbose: List[Dict] = [] if args.verbose else None

    for fold_idx, fold_dir in enumerate(args.fold_dirs_resolved, start=1):
        logger.info(f"--- Fold {fold_idx}/{len(args.fold_dirs_resolved)}: {fold_dir.name} ---")
        fold_config = _load_fold_config(fold_dir, args.checkpoint_subdir)
        
        # 4a. Locate Fold Stacking Artifacts
        xgb_artifacts_dir = fold_dir / "xgb_stacking_artifacts"
        xgb_spec_path = xgb_artifacts_dir / "xgb_stacking_artifacts.json"
        if not xgb_spec_path.exists():
            raise FileNotFoundError(f"Missing XGBoost artifacts in fold {fold_dir.name}: {xgb_spec_path}")

        with open(xgb_spec_path) as f:
            xgb_spec = json.load(f)

        spec_pkl_path = xgb_artifacts_dir / xgb_spec.get("specificity_pkl", "marker_classification.pkl")
        cutoff_pkl_path = xgb_artifacts_dir / xgb_spec.get("cutoff_pkl", "marker_cutoffs.pkl")
        
        with open(spec_pkl_path, "rb") as fh:
            marker_classification = pickle.load(fh)
        with open(cutoff_pkl_path, "rb") as fh:
            cutoff_dict = pickle.load(fh)

        # 4b. Compute Fold-Specific XGBoost Marker Features
        marker_hits = genomad_marker_dict or {}
        xgb1_mf_df = build_xgb1_marker_features(chunked_accs, marker_hits, marker_classification, cutoff_dict, features_df_arch)
        xgb2_mf_df = build_xgb2_marker_features(chunked_accs, marker_hits, marker_classification, cutoff_dict, features_df_arch)

        # 4c. Load & Run Fold XGBoost Models
        xgb1_booster = xgb.Booster()
        xgb2_booster = xgb.Booster()
        xgb1_booster.load_model(str(xgb_artifacts_dir / xgb_spec["xgb1_model"]))
        xgb2_booster.load_model(str(xgb_artifacts_dir / xgb_spec["xgb2_model"]))

        xgb_arch_df = features_df_arch.select(arch_cols_no_frag).fill_null(0.0)

        # Ablate by swapping out gene_desnity feature with nan
        if "gene_density" in xgb_arch_df.columns:
            xgb_arch_df = xgb_arch_df.with_columns(pl.lit(np.nan).alias("gene_density"))
        if "gene_density_fwd" in xgb_arch_df.columns:
            xgb_arch_df = xgb_arch_df.with_columns(pl.lit(np.nan).alias("gene_density_fwd"))
        if "gene_density_rev" in xgb_arch_df.columns:
            xgb_arch_df = xgb_arch_df.with_columns(pl.lit(np.nan).alias("gene_density_rev"))
        

        x_all_df = pl.concat([xgb_arch_df, xgb1_mf_df], how="horizontal")
        x_euk_df = pl.concat([xgb_arch_df, xgb2_mf_df], how="horizontal")

        # Add a combine feature dictionary for this fold 
        unique_euk_cols = [c for c in x_euk_df.columns if c not in x_all_df.columns]
        fold_features_df = pl.concat([x_all_df, x_euk_df.select(unique_euk_cols)], how="horizontal")
        fold_features_dicts = fold_features_df.to_dicts()

        exp_xgb1 = xgb1_booster.feature_names
        exp_xgb2 = xgb2_booster.feature_names
        x_all = x_all_df.select(exp_xgb1).to_numpy() if exp_xgb1 else x_all_df.to_numpy()
        x_euk = x_euk_df.select(exp_xgb2).to_numpy() if exp_xgb2 else x_euk_df.to_numpy()

        raw1 = xgb1_booster.predict(xgb.DMatrix(x_all))
        xgb1_probs = np.column_stack([1.0 - raw1, raw1]) if raw1.ndim == 1 else raw1

        raw2 = xgb2_booster.predict(xgb.DMatrix(x_euk))
        xgb2_probs = np.column_stack([1.0 - raw2, raw2]) if raw2.ndim == 1 else raw2

        xgb_probs = _reconstruct_hierarchical_probs(
            xgb1_probs, xgb2_probs,
            int(xgb_spec["n_fine_classes"]),
            int(xgb_spec["prokaryote_idx"]),
            [int(i) for i in xgb_spec["euk_fine_indices"]],
        )

        # 4d. Build Combined Feature Matrix & Gate Inputs
        gate_marker_dim = int(fold_config.get("gate_marker_dim", 0))
        if gate_marker_dim > 0:
            n_genes_array = features_df_arch.select("n_genes").to_numpy()
            fold_features = np.concatenate([xgb_probs, n_genes_array], axis=1)
        else:
            fold_features = xgb_probs

        raw_feature_dim = int(fold_config.get("raw_feature_dim", 0))
        if raw_feature_dim == (len(arch_cols_no_frag) + len(xgb1_mf_df.columns)):
            raw_gate_features = pl.concat([xgb_arch_df, xgb1_mf_df], how="horizontal").to_numpy()
        else:
            raw_gate_features = xgb_arch_df.to_numpy()

        # 4e. Apply Scaler if present
        scaler_path = xgb_artifacts_dir / "feature_scaler.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            fold_features = scaler.transform(fold_features)

        # 4f. Load Fold Transformer & Predict
        model_artifact_dir = fold_dir / args.checkpoint_subdir
        tokenizer = AutoTokenizer.from_pretrained(fold_config.get("tokenizer", fold_config["model"]), trust_remote_code=True)
        if model_type == "bibert":
            tokenizer.model_max_length = max_length

        base_model, _, _, _, hidden_size = load_model_and_tokenizer(
            model_path=str(fold_dir),
            model_type=model_type,
            pooling=fold_config.get("pooling", "mean"),
            max_length=max_length,
            base_model_name=fold_config.get("model"),
        )

        if (model_artifact_dir / "adapter_config.json").exists():
            base_model = PeftModel.from_pretrained(base_model, model_artifact_dir)

        classifier = GenomeClassifier(
            base_model=base_model,
            num_classes=num_classes,
            hidden_size=hidden_size,
            tokenizer=tokenizer,
            pooling=fold_config.get("pooling", "mean"),
            model_type=model_type,
            dropout=0.0,
            feature_dim=fold_features.shape[1],
            xgb_feature_dim=num_classes,
            gate_marker_dim=gate_marker_dim,
            use_features=True,
            use_glm=bool(fold_config.get("use_glm", True)),
            feature_integration_mode=fold_config.get("feature_integration_mode", "stacking"),
            gate_hidden_dim=int(fold_config.get("gate_hidden_dim", 64)),
            raw_feature_dim=raw_feature_dim,
        ).to(device)

        ckpt_path = model_artifact_dir / "classifier_head.pt"
        if not ckpt_path.exists():
            ckpt_path = fold_dir / "classifier_head.pt"
        ckpt_state = torch.load(ckpt_path, map_location=device)
        classifier.load_state_dict(ckpt_state, strict=False)
        classifier.eval()

        # Build DataLoader
        dataset = GenomeDataset(
            sequences=chunked_seqs,
            labels=[0] * len(chunked_seqs),
            features=fold_features.tolist(),
            accessions=chunked_accs,
            tokenizer=tokenizer,
            max_length=max_length,
            model_type=model_type,
            token_pooling=fold_config.get("pooling", "mean"),
            raw_features=raw_gate_features.tolist(),
        )
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=make_collate_fn(tokenizer, model_type, max_length),
            num_workers=args.num_workers, pin_memory=(device.type == "cuda")
        )

        # Collect Logits
        fold_logits_list = []
        fold_alphas_list = []
        fold_glm_logits_list = [] if args.verbose else None
        fold_xgb_logprobs_list = [] if args.verbose else None
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"      Inference", leave=False):
                inputs = {k: v.to(device) for k, v in batch.items() if k not in {'accession', 'labels'}}
                extra_kwargs = {"return_auxiliary_logits": True} if args.verbose else {}
                if args.fp16 and device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = classifier(**inputs, **extra_kwargs)
                else:
                    outputs = classifier(**inputs, **extra_kwargs)
                if args.verbose:
                    logits = outputs[0]
                    glm_batch = outputs[1]
                    xgb_batch = outputs[2]
                    fold_glm_logits_list.append(glm_batch.cpu() if glm_batch is not None else None)
                    fold_xgb_logprobs_list.append(xgb_batch.cpu() if xgb_batch is not None else None)
                else:
                    logits = outputs
                fold_logits_list.append(logits.cpu())
                if hasattr(classifier, '_last_alpha_batch'):
                    alpha = np.atleast_1d(classifier._last_alpha_batch.flatten())
                    fold_alphas_list.append(alpha)

        fold_logits = torch.cat(fold_logits_list, dim=0)
        fold_alphas = np.round(
            np.concatenate(fold_alphas_list) if fold_alphas_list else np.full(len(chunked_seqs), np.nan),
            4,
        )
        per_fold_basic.append({"fold_name": fold_dir.name, "logits": fold_logits, "alphas": fold_alphas})

        if args.verbose:
            if fold_xgb_logprobs_list and all(x is not None for x in fold_xgb_logprobs_list):
                fold_xgb_probs = np.round(torch.exp(torch.cat(fold_xgb_logprobs_list, dim=0)).numpy(), 4)
            else:
                fold_xgb_probs = None
            if fold_glm_logits_list and all(x is not None for x in fold_glm_logits_list):
                fold_glm_probs = np.round(F.softmax(torch.cat(fold_glm_logits_list, dim=0), dim=-1).numpy(), 4)
            else:
                fold_glm_probs = None
            per_fold_verbose.append({
                "fold_name": fold_dir.name,
                "logits": fold_logits,
                "alphas": fold_alphas,
                "features": fold_features_dicts,
                "xgb_probs": fold_xgb_probs,
                "glm_probs": fold_glm_probs,
            })
        if accumulated_logits is None:
            accumulated_logits = fold_logits.clone()
        else:
            accumulated_logits += fold_logits

        del classifier, base_model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    # ── [5/5] Post-Ensemble Vector Calibration & Aggregation ────────────────
    logger.info("[5/5] Applying length-aware vector calibration & chunk aggregation")
    avg_logits = accumulated_logits / len(args.fold_dirs_resolved)
    chunk_lengths = [len(s) for s in chunked_seqs]

    calibrated_probs = _apply_calibration(avg_logits, calib_params, chunk_lengths, num_classes)
    for fold_data in per_fold_basic:
        fold_data["probs"] = _apply_calibration(fold_data["logits"], calib_params, chunk_lengths, num_classes)
    if args.verbose:
        for fold_data in per_fold_verbose:
            fold_data["probs"] = _apply_calibration(fold_data["logits"], calib_params, chunk_lengths, num_classes)


    class_cols = [label_mapping.get(i, f"class_{i}") for i in range(num_classes)]
    pred_indices = np.argmax(calibrated_probs, axis=1)
    confidences = np.max(calibrated_probs, axis=1)
    predicted_hosts = [label_mapping.get(int(i), f"class_{i}") for i in pred_indices]

    data_dict: Dict = {
        "accession": chunked_accs,
        "predicted_host": predicted_hosts,
        "confidence": confidences.tolist(),
        **{col: calibrated_probs[:, i].tolist() for i, col in enumerate(class_cols)},
    }

    prok_col_idx = next((int(i) for i, n in label_mapping.items() if "prokaryote" in n.lower()), None)
    if prok_col_idx is not None:
        data_dict["prokaryote_score"] = calibrated_probs[:, prok_col_idx].tolist()
        data_dict["eukaryote_score"] = (1.0 - calibrated_probs[:, prok_col_idx]).tolist()

    df_chunks = pl.DataFrame(data_dict).sort("accession")
    df_chunks.write_csv(args.output, separator="\t")

    if args.aggregate_chunks:
        df_genomes = _aggregate_chunks(df_chunks, class_cols, prok_col_idx, label_mapping)
        df_genomes.write_csv(args.genome_output, separator="\t")
        logger.info(f"Genome-level predictions written to: {args.genome_output}")

    fold_rows: List[Dict] = []
    for fold_data in per_fold_basic:
        fold_probs = fold_data["probs"]
        fold_alphas = fold_data["alphas"]
        fold_pred_indices = np.argmax(fold_probs, axis=1)
        fold_confidences = np.max(fold_probs, axis=1)
        fold_predicted_hosts = [label_mapping.get(int(i), f"class_{i}") for i in fold_pred_indices]
        for j, acc in enumerate(chunked_accs):
            row: Dict = {
                "accession": acc,
                "fold": fold_data["fold_name"],
                "predicted_host": fold_predicted_hosts[j],
                "confidence": round(float(fold_confidences[j]), 4),
                "glm_gate_weight": None if np.isnan(fold_alphas[j]) else round(float(fold_alphas[j]), 4),
                **{col: round(float(fold_probs[j, i]), 4) for i, col in enumerate(class_cols)},
            }
            fold_rows.append(row)
    df_folds = pl.DataFrame(fold_rows).sort(["accession", "fold"])
    folds_path = args.output_dir / f"{args.prefix}.folds.tsv"
    df_folds.write_csv(folds_path, separator="\t")
    logger.info(f"Per-fold predictions written to: {folds_path}")

    if args.verbose and per_fold_verbose:
        verbose_rows = []
        for fold_data in per_fold_verbose:
            fold_probs = fold_data["probs"]
            fold_alphas = fold_data["alphas"]
            fold_pred_indices = np.argmax(fold_probs, axis=1)
            fold_confidences = np.max(fold_probs, axis=1)
            fold_predicted_hosts = [label_mapping.get(int(i), f"class_{i}") for i in fold_pred_indices]
            for j, acc in enumerate(chunked_accs):
                row: Dict = {
                    "accession": acc,
                    "fold": fold_data["fold_name"],
                    "predicted_host": fold_predicted_hosts[j],
                    "confidence": float(fold_confidences[j]),
                    "glm_gate_weight": None if np.isnan(fold_alphas[j]) else float(fold_alphas[j]),
                }
                for i, col in enumerate(class_cols):
                    row[col] = float(fold_probs[j, i])

                if fold_data.get("xgb_probs") is not None:
                    for i, col in enumerate(class_cols):
                        row[f"xgb_{col}"] = float(fold_data["xgb_probs"][j, i])
                if fold_data.get("glm_probs") is not None:
                    for i, col in enumerate(class_cols):
                        row[f"glm_{col}"] = float(fold_data["glm_probs"][j, i])
                # inject all architecture and marker features
                row.update(fold_data["features"][j])

                verbose_rows.append(row)

        df_verbose = pl.DataFrame(verbose_rows).sort(["accession", "fold"])
        verbose_path = args.output_dir / f"{args.prefix}.verbose.tsv"
        df_verbose.write_csv(verbose_path, separator="\t")
        logger.info(f"Verbose per-fold predictions written to: {verbose_path}")

    logger.info("Done!")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--fasta", type=click.Path(path_type=pathlib.Path, exists=True, dir_okay=False), required=True, help="Input FASTA file.")
@click.option("--output", type=click.Path(path_type=pathlib.Path, file_okay=False), required=True, help="Output directory for feature files.")
@click.option("--prefix", default="vhamster", show_default=True, help="Base filename prefix for output files.")
@click.option("--genomad-db", type=click.Path(path_type=pathlib.Path), default=None, help="geNomad MMseqs2 DB path.")
@click.option("--ensemble-dir", type=click.Path(path_type=pathlib.Path), default=_DEFAULT_MODEL_ROOT, show_default=True, help="Root directory containing fold_* subdirs (used to locate geNomad DB if --genomad-db not set).")
@click.option("--chunk-size", type=int, default=10000, show_default=True, help="Chunk length in bp (must match value used with vhamster).")
@click.option("--overlap", type=int, default=1000, show_default=True, help="Overlap between chunks in bp (must match value used with vhamster).")
@click.option("--num-workers", type=int, default=4, show_default=True, help="Worker processes for PyRodigal feature extraction.")
@click.option("--mmseqs-threads", type=int, default=None, help="Threads for protein prediction and MMseqs2 search. Defaults to --num-workers when not set, so on a cluster you can just set --num-workers to your CPU count and both steps scale together.")
def features_main(
    fasta: pathlib.Path,
    output: pathlib.Path,
    prefix: str,
    genomad_db: Optional[pathlib.Path],
    ensemble_dir: pathlib.Path,
    chunk_size: int,
    overlap: int,
    num_workers: int,
    mmseqs_threads: Optional[int],
) -> None:
    """Extract architectural and geNomad marker features without running the GLM.

    Produces {prefix}.arch_features.tsv and {prefix}.genomad_hits.json in the
    output directory. Pass the output directory to 'vhamster --precomputed-features'
    to skip this step on the GPU node.
    """
    output_dir = output
    log_path = _configure_logging(output_dir, prefix)
    logger.info(f"VHAMSTeR v{__version__} — feature extraction only")
    logger.info(f"Logging to: {log_path}")

    if genomad_db is None:
        genomad_db = ensemble_dir / "genomad_db"

    # Pre-flight: mmseqs2 + genomad_db (no model files needed here)
    errors: List[str] = []
    if shutil.which("mmseqs") is None:
        errors.append(
            "mmseqs2 binary not found in PATH. Install via conda:\n"
            "    conda install -c bioconda mmseqs2"
        )
    mmseqs_db_type = genomad_db / "genomad_db.dbtype"
    if not mmseqs_db_type.exists():
        errors.append(
            f"geNomad MMseqs2 database not found at {genomad_db / 'genomad_db'}. "
            "Run 'vhamster-install-models' to download it, or provide --genomad-db."
        )
    if errors:
        raise click.ClickException("Pre-flight validation failed:\n  " + "\n  ".join(errors))

    # Chunk sequences
    logger.info("Loading and chunking sequences")
    seqs, accessions = load_fasta_sequences(str(fasta))
    chunked_seqs, chunked_accs = _chunk_sequences(seqs, accessions, chunk_size, overlap)
    logger.info(f"{len(accessions)} sequence(s) -> {len(chunked_seqs)} chunk(s)")

    # PyRodigal architectural features
    logger.info("Extracting architectural features (PyRodigal)")
    arch_feature_names = list(ARCH_FEATURE_NAMES) + ["n_genes"]
    n_workers = min(multiprocessing.cpu_count(), len(chunked_seqs), num_workers)
    args_list = [(s, a, arch_feature_names, chunk_size) for s, a in zip(chunked_seqs, chunked_accs)]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        raw_arch_features = list(tqdm(ex.map(extract_features_worker, args_list), total=len(chunked_seqs), desc="      PyRodigal"))

    features_df_arch = pl.DataFrame(
        np.array(raw_arch_features, dtype=np.float32),
        schema=arch_feature_names,
    ).with_columns(pl.Series("accession", chunked_accs))
    col_order = ["accession"] + arch_feature_names
    arch_tsv = output_dir / f"{prefix}.arch_features.tsv"
    features_df_arch.select(col_order).write_csv(arch_tsv, separator="\t")
    logger.info(f"Architectural features written to: {arch_tsv}")

    # geNomad MMseqs2
    genomad_metadata = genomad_db / "genomad_marker_metadata.tsv"
    if not genomad_metadata.exists():
        raise click.ClickException(
            f"geNomad metadata file not found: {genomad_metadata}"
        )
    effective_mmseqs_threads = mmseqs_threads if mmseqs_threads is not None else num_workers
    logger.info(f"Running MMseqs2 against geNomad marker database ({effective_mmseqs_threads} threads)")
    seq_dict = dict(zip(chunked_accs, chunked_seqs))
    marker_hits, _, annotation_rows = extract_genomad_markers(
        sequences=seq_dict,
        genomad_db=genomad_db,
        genomad_metadata=genomad_metadata,
        threads=effective_mmseqs_threads,
        return_details=True,
    )
    hits_json = output_dir / f"{prefix}.genomad_hits.json"
    with open(hits_json, "w") as fh:
        json.dump(marker_hits, fh)
    logger.info(f"geNomad marker hits written to: {hits_json}")

    gene_table_path = output_dir / f"{prefix}.gene_predictions.tsv"
    write_annotation_rows(annotation_rows, gene_table_path)
    logger.info(f"Gene prediction table written to: {gene_table_path}")
    logger.info("Feature extraction complete.")


def _preflight_checks(args: Any) -> None:
    """Verify required binaries, databases, and model files are present before running."""
    errors: List[str] = []

    if args.precomputed_features:
        # Stage 2: skip mmseqs2/genomad checks; verify precomputed files instead
        feat_dir = pathlib.Path(args.precomputed_features)
        arch_tsv = feat_dir / f"{args.prefix}.arch_features.tsv"
        hits_json = feat_dir / f"{args.prefix}.genomad_hits.json"
        if not arch_tsv.exists():
            errors.append(
                f"Precomputed arch features not found: {arch_tsv}. "
                "Run 'vhamster-features' first."
            )
        if not hits_json.exists():
            errors.append(
                f"Precomputed geNomad hits not found: {hits_json}. "
                "Run 'vhamster-features' first."
            )
    else:
        # Full pipeline: check mmseqs2 binary and genomad database
        if shutil.which("mmseqs") is None:
            errors.append(
                "mmseqs2 binary not found in PATH. Install via conda:\n"
                "    conda install -c bioconda mmseqs2"
            )
        genomad_db = pathlib.Path(args.genomad_db)
        mmseqs_db_type = genomad_db / "genomad_db.dbtype"
        if not mmseqs_db_type.exists():
            errors.append(
                f"geNomad MMseqs2 database not found at {genomad_db / 'genomad_db'}. "
                "Run 'vhamster-install-models' to download it, or provide --genomad-db."
            )

    # Per-fold model files (always required)
    for fold_dir in args.fold_dirs_resolved:
        if not (fold_dir / "config.json").exists():
            errors.append(
                f"Missing config.json in {fold_dir}. Run 'vhamster-install-models'."
            )
        adapter_dir = fold_dir / args.checkpoint_subdir
        if not (adapter_dir / "adapter_config.json").exists():
            errors.append(
                f"Missing adapter_config.json in {adapter_dir}. Run 'vhamster-install-models'."
            )
        xgb_json = fold_dir / "xgb_stacking_artifacts" / "xgb_stacking_artifacts.json"
        if not xgb_json.exists():
            errors.append(
                f"Missing XGBoost artifacts at {xgb_json}. Run 'vhamster-install-models'."
            )

    if errors:
        msg = "\n  ".join(errors)
        raise click.ClickException(f"Pre-flight validation failed:\n  {msg}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--fasta", type=click.Path(path_type=pathlib.Path, exists=True, dir_okay=False), required=True, help="Input FASTA file.")
@click.option("--output", type=click.Path(path_type=pathlib.Path, file_okay=False), required=True, help="Output directory.")
@click.option("--prefix", default="vhamster", show_default=True, help="Base filename prefix for outputs.")
@click.option("-f", "--force", "force", is_flag=True, help="Overwrite output if it exists.")
@click.option("--ensemble-dir", type=click.Path(path_type=pathlib.Path), default=_DEFAULT_MODEL_ROOT, show_default=True, help="Root directory containing fold_* subdirs.")
@click.option("--fold-dirs", type=click.Path(path_type=pathlib.Path), multiple=True, help="Explicit fold directories (overrides --ensemble-dir).")
@click.option("--genomad-db", type=click.Path(path_type=pathlib.Path), default=None, help="geNomad MMseqs2 DB path.")
@click.option("--num-folds", type=int, default=5, show_default=True, help="Number of folds to use.")
@click.option("--fold-index", type=int, default=None, help="Use only one fold by index, e.g. 0..4.")
@click.option("--checkpoint-subdir", default="best_macro_auprc_model", show_default=True, help="Checkpoint subdirectory name.")
@click.option("--calibration-params", type=str, default=str(_DEFAULT_MODEL_ROOT / "length_aware_vector_scaling_anchors_toplabel_5.json"), show_default=True, help="Path to length-aware vector scaling JSON.")
@click.option("--chunk-size", type=int, default=10000, show_default=True, help="Chunk length in bp.")
@click.option("--overlap", type=int, default=1000, show_default=True, help="Overlap between chunks in bp.")
@click.option("--batch-size", type=int, default=16, show_default=True, help="Inference batch size.")
@click.option("--fp16", is_flag=True, help="Use FP16 mixed precision.")
@click.option("--num-workers", type=int, default=4, show_default=True, help="DataLoader workers.")
@click.option("--mmseqs-threads", type=int, default=4, show_default=True, help="Threads for protein prediction and MMseqs2 search (only used when --precomputed-features is not set).")
@click.option("--precomputed-features", type=click.Path(path_type=pathlib.Path), default=None, help="Directory containing {prefix}.arch_features.tsv and {prefix}.genomad_hits.json from vhamster-features. Skips MMseqs2 and PyRodigal.")
@click.option("--aggregate-chunks/--no-aggregate-chunks", default=True, show_default=True, help="Enable/disable genome-level consensus output.")
@click.option("--verbose", is_flag=True, help="Write per-fold predictions and GLM gate weights to {prefix}.verbose.tsv.")
def main(
    fasta: pathlib.Path,
    output: pathlib.Path,
    prefix: str,
    force: bool,
    ensemble_dir: pathlib.Path,
    fold_dirs: Tuple[pathlib.Path, ...],
    genomad_db: pathlib.Path,
    num_folds: int,
    fold_index: Optional[int],
    checkpoint_subdir: str,
    calibration_params: str,
    chunk_size: int,
    overlap: int,
    batch_size: int,
    fp16: bool,
    num_workers: int,
    mmseqs_threads: int,
    precomputed_features: Optional[pathlib.Path],
    aggregate_chunks: bool,
    verbose: bool,
) -> None:
    output_dir = output
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = _configure_logging(output_dir, prefix)

    # Locate geNomad database (only needed when not using precomputed features)
    if genomad_db is None:
        genomad_db = ensemble_dir / "genomad_db"

    if precomputed_features is None and not (genomad_db / "genomad_marker_metadata.tsv").exists():
        raise click.ClickException(
            f"geNomad database not found at {genomad_db}. "
            "Run 'vhamster-install-models' to download it, or provide --genomad-db."
        )

    args = SimpleNamespace(
        fasta=fasta,
        output_dir=output_dir,
        output=output_dir / f"{prefix}.chunks.tsv",
        prefix=prefix,
        force=force,
        ensemble_dir=ensemble_dir,
        fold_dirs=list(fold_dirs) if fold_dirs else None,
        genomad_db=genomad_db,
        num_folds=num_folds,
        fold_index=fold_index,
        checkpoint_subdir=checkpoint_subdir,
        calibration_params=calibration_params,
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=batch_size,
        fp16=fp16,
        num_workers=num_workers,
        mmseqs_threads=mmseqs_threads,
        precomputed_features=precomputed_features,
        aggregate_chunks=aggregate_chunks,
        genome_output=output_dir / f"{prefix}.genomes.tsv",
        verbose=verbose,
    )

    args.fold_dirs_resolved = _discover_fold_dirs(args)
    if args.fold_index is not None:
        if args.fold_index < 0:
            raise click.ClickException("--fold-index must be >= 0.")
        target_name = f"fold_{args.fold_index}"
        selected = next((p for p in args.fold_dirs_resolved if p.name == target_name), None)
        if selected is None:
            available = [p.name for p in args.fold_dirs_resolved]
            raise click.ClickException(
                f"Requested {target_name}, but it was not found. Available folds: {available}"
            )
        args.fold_dirs_resolved = [selected]
    else:
        if args.num_folds > len(args.fold_dirs_resolved):
            raise click.ClickException(
                f"--num-folds={args.num_folds} but only {len(args.fold_dirs_resolved)} fold directories were found."
            )
        args.fold_dirs_resolved = args.fold_dirs_resolved[: args.num_folds]
    args.num_folds = len(args.fold_dirs_resolved)

    # Ensure calibration_params points to ensemble_dir if default site-packages path doesn't exist
    calib_path = pathlib.Path(calibration_params)
    if not calib_path.exists():
        alt_calib = ensemble_dir / "length_aware_vector_scaling_anchors_toplabel_5.json"
        if alt_calib.exists():
            calibration_params = str(alt_calib)

    logger.info(f"Logging to: {log_path}")
    _preflight_checks(args)
    _run(args)


if __name__ == "__main__":
    main()