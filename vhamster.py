#!/usr/bin/env python3
"""
V-HAMSTeR 
==============================================================
Virus Host Assignment Model using Sequence Transformers and Reading-frame.
Run inference using the full 5-fold deep ensemble and apply the joint
temperature T_joint (from calibrate_joint_temperature.py) before converting
logits to probabilities.

Flow
----
1. Discover fold directories from --ensemble-dir or --fold-dirs.
2. Read architecture / label / feature config from fold 0's config.json.
3. Chunk input sequences and extract handcrafted features.
4. For every fold model: load weights → run no-grad inference → get raw logits.
5. Scale each fold's logits by T_joint, apply softmax, then average calibrated
    probability distributions across folds.
6. Write chunk-level TSV output in the same format as predict_genome.py.
7. Aggregate chunks from the same parent genome by mean-pooling calibrated
    class probabilities to produce genome-level consensus predictions.
"""

__version__ = "1.0.0"

import gc
import json
import multiprocessing
import pathlib
import re
import sys
import sysconfig
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple, Any, Union

import click
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
from peft import PeftModel

# ── src/ on path ──────────────────────────────────────────────────────────────
_ROOT = pathlib.Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.features import extract_features_worker
from src.models import GenomeClassifier, make_collate_fn
from src.sequences import GenomeDataset, load_fasta_sequences

# ── local helper functions (self-contained; no external calibration dependency) ─


def _default_model_root() -> pathlib.Path:
    purelib = sysconfig.get_path("purelib")
    if purelib is None:
        return _ROOT / "model"
    return pathlib.Path(purelib) / "vhamster_models"


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
        raise FileNotFoundError(
            f"No fold_* directories found under {ensemble_dir}."
        )

    def _fold_sort_key(p: pathlib.Path) -> Tuple[int, str]:
        m = re.search(r"fold_(\d+)$", p.name)
        if m:
            return int(m.group(1)), p.name
        return 10**9, p.name

    return sorted(fold_dirs, key=_fold_sort_key)


def _load_fold_config(fold_dir: pathlib.Path, checkpoint_subdir: str) -> Dict:
    """Load config.json from fold root or checkpoint subdir."""
    candidates = [
        fold_dir / "config.json",
        fold_dir / checkpoint_subdir / "config.json",
    ]
    for cfg in candidates:
        if cfg.exists():
            with open(cfg, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(
        f"No config.json found for fold {fold_dir}. Tried: {[str(c) for c in candidates]}"
    )


def _build_base_model(model_name: str, model_type: str, max_length: int) -> nn.Module:
    """Build base transformer model for inference."""
    _ = max_length  # kept for API compatibility and future model-specific controls
    if model_type == "modernbert":
        return AutoModel.from_pretrained(model_name, trust_remote_code=True)
    return AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)


def _get_hidden_size(base_model: nn.Module) -> int:
    """Infer hidden size from common config attributes."""
    for attr in ("hidden_size", "d_model", "embed_dim"):
        if hasattr(base_model.config, attr):
            return int(getattr(base_model.config, attr))
    return int(base_model.get_input_embeddings().weight.shape[1])


def _apply_feature_scaling(raw_features: List[List[float]], config: Dict) -> List[List[float]]:
    """Apply training-time feature scaling when scaler stats are available."""
    scaler_mean = config.get("feature_scaler_mean")
    scaler_scale = config.get("feature_scaler_scale")
    if not scaler_mean or not scaler_scale:
        return raw_features

    mean = np.asarray(scaler_mean, dtype=np.float32)
    scale = np.asarray(scaler_scale, dtype=np.float32)
    if mean.ndim != 1 or scale.ndim != 1:
        return raw_features
    if len(raw_features) == 0 or len(raw_features[0]) != mean.shape[0] or scale.shape[0] != mean.shape[0]:
        return raw_features

    arr = np.asarray(raw_features, dtype=np.float32)
    safe_scale = np.where(scale == 0.0, 1.0, scale)
    scaled = (arr - mean) / safe_scale
    return scaled.tolist()


# Keep feature schema consistent with predict_genome.py so ensemble output
# includes the familiar per-chunk gene/RBS columns.
DEFAULT_FEATURE_NAMES: List[str] = [
    "fragment_size",
    "strand_switch_rate",
    "coding_density",
    "leaderless_freq",
    "short_utr_freq",
    "no_rbs_freq",
    "sd_bacteroidetes_rbs_freq",
    "sd_canonical_rbs_freq",
    "tatata_rbs_freq",
    "mean_rbs_score",
    "gene_density",
    "gene_density_fwd",
    "gene_density_rev",
]


# ─────────────────────────────────────────────────────────────────────────────
# Chunk aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_chunks(
    chunk_df: pl.DataFrame,
    class_cols: List[str],
    prok_col_idx: Optional[int],
    label_mapping: Dict[int, str],
) -> pl.DataFrame:
    """Mean-pool per-class probabilities across chunks to produce one row per
    parent genome.  The parent genome ID is the accession with every
    trailing '_chunk<start>_<end>' suffix stripped.

    Mean-pooling over calibrated probability vectors (rather than majority
    vote) is the correct aggregation because it preserves the full
    uncertainty information captured by temperature scaling and the ensemble.
    """
    # Derive the parent genome ID: strip the trailing _chunk<int>_<int> suffix
    # produced by _chunk_sequences(), leaving the original FASTA accession.
    genome_col = (
        pl.col("accession")
        .str.replace(r"_chunk\d+_\d+$", "", literal=False)
        .alias("genome")
    )
    df = chunk_df.with_columns(genome_col)

    # Average per-class probability columns across all chunks of the same genome.
    agg_exprs = [pl.col(c).mean() for c in class_cols]
    genome_df = df.group_by("genome").agg(agg_exprs).sort("genome")

    # Re-derive predicted host and confidence from the averaged probabilities.
    prob_arr = genome_df.select(class_cols).to_numpy()
    pred_indices = np.argmax(prob_arr, axis=1)
    confidences = np.max(prob_arr, axis=1)
    predicted_hosts = [label_mapping.get(int(i), f"class_{i}") for i in pred_indices]

    genome_df = genome_df.with_columns([
        pl.Series("predicted_host", predicted_hosts),
        pl.Series("confidence", confidences),
    ])

    # Prokaryote / eukaryote convenience columns.
    if prok_col_idx is not None:
        prok_col_name = class_cols[prok_col_idx]
        genome_df = genome_df.with_columns([
            pl.col(prok_col_name).alias("prokaryote_score"),
            (pl.lit(1.0) - pl.col(prok_col_name)).alias("eukaryote_score"),
        ])

    # Reorder columns: genome, predicted_host, confidence, class probs, extras.
    lead_cols = ["genome", "predicted_host", "confidence"]
    rest = [c for c in genome_df.columns if c not in lead_cols]
    genome_df = genome_df.select(lead_cols + rest)

    return genome_df


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_temperature(temperature_value: Optional[Union[str, pathlib.Path, float]]) -> float:
    """Load temperature from either a numeric value or a saved .pt file.

    Accepted inputs:
    - float/int-like string, e.g. "1.0", "2.35"
    - pathlib.Path / path string to a torch-saved object with key T_joint or scalar
    """
    if temperature_value is None:
        return 1.0

    # Allow direct scalar input for quick benchmarking.
    if isinstance(temperature_value, (int, float)):
        t = float(temperature_value)
        logger.info(f"Using temperature scalar T = {t:.8f} from CLI")
        return t

    value_str = str(temperature_value)
    try:
        t = float(value_str)
        logger.info(f"Using temperature scalar T = {t:.8f} from CLI")
        return t
    except (TypeError, ValueError):
        pass

    # Fall back to path-based loading.
    temperature_file = pathlib.Path(value_str)
    if not temperature_file.exists():
        logger.warning(f"Temperature file not found: {temperature_file}. Using T=1.0 (no calibration).")
        return 1.0

    state = torch.load(temperature_file, map_location="cpu")
    if isinstance(state, dict):
        t = float(state.get("T_joint", 1.0))
    else:
        t = float(state)
    logger.info(f"Loaded T_joint = {t:.8f} from {temperature_file}")
    return t


def _normalise_label_mapping(mapping: Dict) -> Dict[int, str]:
    """Convert config label mapping keys to int and values to str."""
    return {int(k): str(v) for k, v in (mapping or {}).items()}


def _align_fold_probs_to_reference(
    fold_probs: torch.Tensor,
    fold_label_mapping: Dict[int, str],
    ref_label_mapping: Dict[int, str],
    fold_name: str,
) -> torch.Tensor:
    """Reorder one fold's [N, C] probabilities into reference class-index order.

    This protects ensemble averaging from per-fold class-index drift.
    """
    if not ref_label_mapping or not fold_label_mapping:
        return fold_probs

    ref_order = [ref_label_mapping[i] for i in sorted(ref_label_mapping.keys())]
    fold_order = [fold_label_mapping[i] for i in sorted(fold_label_mapping.keys())]

    if ref_order == fold_order:
        return fold_probs

    fold_name_to_idx = {name: idx for idx, name in fold_label_mapping.items()}
    reorder_idx: List[int] = []
    missing_classes: List[str] = []
    for ref_idx in sorted(ref_label_mapping.keys()):
        class_name = ref_label_mapping[ref_idx]
        if class_name not in fold_name_to_idx:
            missing_classes.append(class_name)
        else:
            reorder_idx.append(fold_name_to_idx[class_name])

    if missing_classes:
        raise ValueError(
            f"Fold {fold_name} is missing reference classes: {missing_classes}. "
            "Cannot safely ensemble across folds with mismatched class sets."
        )

        logger.warning(
            f"{fold_name} label index order differs from reference; "
            "reordering fold probabilities by class name before averaging."
        )
    reorder_tensor = torch.tensor(reorder_idx, dtype=torch.long)
    return fold_probs.index_select(dim=1, index=reorder_tensor)


def _build_fold_classifier(
    fold_dir: pathlib.Path,
    checkpoint_subdir: str,
    device: torch.device,
) -> Tuple[nn.Module, AutoTokenizer, Dict]:
    """Reconstruct one fold's GenomeClassifier from its config + checkpoint.

    Loading path intentionally mirrors predict_genome.py:
    1) inject LoRA adapters (if present)
    2) load classifier_head.pt when available
    3) otherwise fallback to training_state.pt model_state_dict
    4) filter by key+shape and load non-strict
    """
    config = _load_fold_config(fold_dir, checkpoint_subdir)

    model_name = config["model"]
    tokenizer_name = config.get("tokenizer", model_name)
    model_type = config.get("model_type", "nucleotidetransformer")
    max_length = int(config.get("max_length", 10000))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if model_type == "bibert":
        tokenizer.model_max_length = max_length

    base_model = _build_base_model(model_name=model_name, model_type=model_type, max_length=max_length)
    hidden_size = _get_hidden_size(base_model)

    # Match predict_genome.py LoRA loading behavior.
    ckpt_dir = fold_dir / checkpoint_subdir
    if (ckpt_dir / "adapter_config.json").exists():
        logger.info(f"LoRA adapters detected in {ckpt_dir.name}; injecting into base model...")
        base_model = PeftModel.from_pretrained(base_model, ckpt_dir)
    elif (fold_dir / "adapter_config.json").exists():
        logger.info(f"LoRA adapters detected in {fold_dir.name}; injecting into base model...")
        base_model = PeftModel.from_pretrained(base_model, fold_dir)

    label_mapping = config.get("label_mapping", {})
    num_classes = len(label_mapping) if label_mapping else int(config.get("num_classes", 5))
    feature_names = config.get("feature_names") or []
    feature_dim = len(feature_names)

    classifier = GenomeClassifier(
        base_model=base_model,
        num_classes=num_classes,
        hidden_size=hidden_size,
        tokenizer=tokenizer,
        pooling=config.get("pooling", "mean"),
        model_type=model_type,
        dropout=0.0,                              # no dropout at inference
        feature_dim=feature_dim,
        use_features=bool(config.get("use_features", True)),
        use_glm=bool(config.get("use_glm", True)),
        feature_integration_mode=config.get("feature_integration_mode", "concat"),
        stream_weight_init=float(config.get("stream_weight_init", 0.0)),
        use_learnable_aux_loss=bool(config.get("use_learnable_aux_loss", False)),
        gate_hidden_dim=int(config.get("gate_hidden_dim", 64)),
        max_length=max_length,
    )

    # Match predict_genome.py checkpoint selection behavior.
    classifier_head_path = None
    fallback_checkpoint = None
    if (ckpt_dir / "classifier_head.pt").exists():
        classifier_head_path = ckpt_dir / "classifier_head.pt"
    elif (fold_dir / "classifier_head.pt").exists():
        classifier_head_path = fold_dir / "classifier_head.pt"

    possible_paths = [
        ckpt_dir / "training_state.pt",
        fold_dir / "training_state.pt",
        fold_dir / "best_model" / "training_state.pt",
    ]
    for p in possible_paths:
        if p.exists():
            fallback_checkpoint = p
            break

    ckpt_state = None
    if classifier_head_path is not None:
        ckpt_state = torch.load(classifier_head_path, map_location="cpu")
    elif fallback_checkpoint is not None:
        checkpoint = torch.load(fallback_checkpoint, map_location="cpu")
        ckpt_state = checkpoint.get("model_state_dict", checkpoint)
    else:
        raise FileNotFoundError(
            f"No classifier weights found for fold {fold_dir}. Expected classifier_head.pt "
            f"or training_state.pt under {ckpt_dir} / {fold_dir}."
        )

    # Filter and non-strict load to match predict_genome.py behavior.
    model_state = classifier.state_dict()
    filtered_state = {}
    for key, value in ckpt_state.items():
        if key in model_state and getattr(value, "shape", None) == model_state[key].shape:
            filtered_state[key] = value

    classifier.load_state_dict(filtered_state, strict=False)
    logger.info(
        f"Loaded {len(filtered_state)}/{len(ckpt_state)} checkpoint keys for {fold_dir.name}"
    )

    classifier.to(device)
    classifier.eval()
    return classifier, tokenizer, config


def _chunk_sequences(
    seqs: List[str],
    accessions: List[str],
    chunk_size: int,
    overlap: int,
) -> Tuple[List[str], List[str]]:
    """Split sequences into overlapping chunks, same logic as predict_genome.py."""
    chunked_seqs: List[str] = []
    chunked_accs: List[str] = []
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


def _extract_features(
    chunked_seqs: List[str],
    chunked_accs: List[str],
    feature_names: List[str],
    chunk_size: int,
    config: Dict,
) -> Tuple[List[List[float]], Optional[List[List[float]]]]:
    """Parallel feature extraction + scaler normalisation matching training."""
    logger.info(f"{len(feature_names)} features x {len(chunked_seqs)} chunk(s)")
    n_workers = min(multiprocessing.cpu_count(), len(chunked_seqs), 8)
    args_list = [
        (seq, acc, feature_names, chunk_size)
        for seq, acc in zip(chunked_seqs, chunked_accs)
    ]
    chunksize = max(1, len(chunked_seqs) // (n_workers * 4))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        raw_features = list(tqdm(
            ex.map(extract_features_worker, args_list, chunksize=chunksize),
            desc="      Extracting",
            total=len(chunked_seqs),
        ))
    gc.collect()

    # Apply the same StandardScaler that was fitted during training.
    scaled = _apply_feature_scaling([list(f) for f in raw_features], config)
    return raw_features, scaled


@torch.inference_mode()
def _collect_logits(
    classifier: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_fp16: bool,
) -> torch.Tensor:
    """Run inference for one fold; return raw logits [N, C] on CPU."""
    logits_list: List[torch.Tensor] = []
    for batch in tqdm(dataloader, desc="      Inference", leave=False):
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        features = batch.get("features")

        if input_ids is not None:
            input_ids = input_ids.to(device, non_blocking=True)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking=True)
        if features is not None:
            features = features.to(device, non_blocking=True)

        if use_fp16 and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = classifier(input_ids=input_ids, attention_mask=attention_mask, features=features)
        else:
            logits = classifier(input_ids=input_ids, attention_mask=attention_mask, features=features)

        logits_list.append(logits.detach().cpu().float())

    return torch.cat(logits_list, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _configure_logging(output_dir: pathlib.Path, prefix: str) -> pathlib.Path:
    """Configure loguru sinks for console + file in output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{prefix}.log"
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=False)
    logger.add(log_path, level="DEBUG", enqueue=True, backtrace=False, diagnose=False)
    return log_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _run(args: Any) -> None:

    logger.info("=" * 68)
    logger.info(f"V-HAMSTeR v{__version__}")
    logger.info("Virus Host Assignment Model using Sequence Transformers and Reading-frame")
    logger.info("=" * 68)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Validate I/O ──────────────────────────────────────────────────────────
    if not args.fasta.is_file():
        sys.exit(f"ERROR: FASTA file not found: {args.fasta}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.force:
        sys.exit(
            f"ERROR: Output file already exists: {args.output}\n"
            "       Use --force / -f to overwrite."
        )

    if args.aggregate_chunks and args.genome_output is not None:
        args.genome_output.parent.mkdir(parents=True, exist_ok=True)
        if args.genome_output.exists() and not args.force:
            sys.exit(
                f"ERROR: Genome output file already exists: {args.genome_output}\n"
                "       Use --force / -f to overwrite."
            )

    if args.overlap >= args.chunk_size:
        sys.exit("ERROR: --overlap must be smaller than --chunk-size.")

    # ── [1/5] Configuration ───────────────────────────────────────────────────
    logger.info("[1/5] Loading configuration")
    logger.info(f"Fold directories: {[str(p) for p in args.fold_dirs_resolved]}")
    logger.info(f"Device: {device}")

    # All folds share the same training config; use fold 0 as the reference.
    ref_config = _load_fold_config(args.fold_dirs_resolved[0], args.checkpoint_subdir)

    feature_names: List[str] = ref_config.get("feature_names") or DEFAULT_FEATURE_NAMES
    use_features: bool = bool(ref_config.get("use_features", True)) and len(feature_names) > 0
    label_mapping: Dict[int, str] = _normalise_label_mapping(ref_config.get("label_mapping", {}))
    num_classes: int = len(label_mapping) if label_mapping else int(ref_config.get("num_classes", 5))
    model_type: str = ref_config.get("model_type", "nucleotidetransformer")
    max_length: int = int(ref_config.get("max_length", 10000))
    feature_integration_mode: str = ref_config.get("feature_integration_mode", "concat")

    logger.info(f"Classes: {num_classes} ({list(label_mapping.values())})")
    logger.info(f"Architecture: {model_type}")
    logger.info(f"Feature stream: {use_features} ({len(feature_names)} features)")
    logger.info(f"Integration mode: {feature_integration_mode}")

    T_joint: float = _load_temperature(args.temperature_file)
    logger.info(f"Temperature: {T_joint:.8f}")
    if T_joint > 5.0:
        logger.warning(
            "Large temperature detected; probabilities may be very flat (close to uniform)."
        )

    # ── [2/5] Chunking ────────────────────────────────────────────────────────
    logger.info("[2/5] Loading and chunking sequences")
    seqs, accessions = load_fasta_sequences(str(args.fasta))
    chunked_seqs, chunked_accs = _chunk_sequences(seqs, accessions, args.chunk_size, args.overlap)
    logger.info(
        f"{len(accessions)} sequence(s) -> {len(chunked_seqs)} chunk(s) "
        f"(chunk: {args.chunk_size:,} bp, overlap: {args.overlap:,} bp)"
    )

    # ── [3/5] Feature extraction ──────────────────────────────────────────────
    logger.info("[3/5] Extracting gene features")
    raw_features: Optional[List[List[float]]] = None
    scaled_features: Optional[List[List[float]]] = None

    if use_features:
        raw_features, scaled_features = _extract_features(
            chunked_seqs=chunked_seqs,
            chunked_accs=chunked_accs,
            feature_names=feature_names,
            chunk_size=args.chunk_size,
            config=ref_config,
        )
    else:
        logger.info("Feature stream disabled; skipping.")

    # ── [4/5] Ensemble inference ──────────────────────────────────────────────
    logger.info("[4/5] Running ensemble inference")

    # Build one shared tokenizer + dataset/dataloader from the reference config.
    ref_tokenizer = AutoTokenizer.from_pretrained(
        ref_config.get("tokenizer", ref_config["model"]), trust_remote_code=True
    )
    if model_type == "bibert":
        ref_tokenizer.model_max_length = max_length

    dataset = GenomeDataset(
        sequences=chunked_seqs,
        labels=[0] * len(chunked_seqs),        # dummy labels; not used in inference
        accessions=chunked_accs,
        tokenizer=ref_tokenizer,
        max_length=max_length,
        model_type=model_type,
        features=scaled_features,
        token_pooling=ref_config.get("pooling", "mean"),
    )
    collate_fn = make_collate_fn(ref_tokenizer, model_type, max_length)
    dataloader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True

    dataloader = DataLoader(dataset, **dataloader_kwargs)

    # Accumulate per-fold calibrated probabilities then average.
    # Correct deep-ensemble aggregation: scale each fold's logits by T_joint,
    # apply softmax to get that fold's probability distribution, then average
    # the distributions.  Averaging raw logits before softmax would produce an
    # artificially sharper distribution and is inconsistent with how T_joint
    # was fitted (which scaled individual-fold logits during calibration).
    accumulated_probs: Optional[torch.Tensor] = None

    for fold_idx, fold_dir in enumerate(args.fold_dirs_resolved, start=1):
        logger.info(f"Fold {fold_idx}/{args.num_folds}: {fold_dir.name}")
        classifier, _, fold_config = _build_fold_classifier(fold_dir, args.checkpoint_subdir, device)

        # Raw logits from this fold's model [N, C]
        fold_logits = _collect_logits(classifier, dataloader, device, args.fp16)

        # Scale by T_joint and convert to probabilities before accumulating.
        fold_probs = F.softmax(fold_logits / T_joint, dim=-1)  # [N, C]

        # Align fold probabilities to reference class order by class name before
        # averaging; protects against per-fold label index drift.
        fold_label_mapping = _normalise_label_mapping(fold_config.get("label_mapping", {}))
        fold_probs = _align_fold_probs_to_reference(
            fold_probs=fold_probs,
            fold_label_mapping=fold_label_mapping,
            ref_label_mapping=label_mapping,
            fold_name=fold_dir.name,
        )

        if accumulated_probs is None:
            accumulated_probs = fold_probs
        else:
            accumulated_probs += fold_probs

        # Free GPU memory before loading the next fold.
        del classifier
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # Average the probability distributions across folds.
    calibrated_probs: np.ndarray = (accumulated_probs / args.num_folds).numpy()  # [N, C]

    # The DataLoader uses shuffle=False, so accession order matches chunked_accs exactly.
    all_accessions = chunked_accs

    # ── [5/5] Writing output ──────────────────────────────────────────────────
    logger.info("[5/5] Writing results")

    class_cols = [label_mapping.get(i, f"class_{i}") for i in range(num_classes)]
    pred_indices = np.argmax(calibrated_probs, axis=1)
    confidences = np.max(calibrated_probs, axis=1)
    predicted_hosts = [label_mapping.get(int(i), f"class_{i}") for i in pred_indices]

    data_dict: Dict = {
        "accession": all_accessions,
        "predicted_host": predicted_hosts,
        "confidence": confidences.tolist(),
        **{col: calibrated_probs[:, i].tolist() for i, col in enumerate(class_cols)},
    }

    # Convenience prokaryote/eukaryote summary columns (matches predict_genome.py).
    prok_col_idx: Optional[int] = None
    for idx, name in label_mapping.items():
        if "prokaryote" in name.lower():
            prok_col_idx = int(idx)
            break
    if prok_col_idx is not None:
        data_dict["prokaryote_score"] = calibrated_probs[:, prok_col_idx].tolist()
        data_dict["eukaryote_score"] = (1.0 - calibrated_probs[:, prok_col_idx]).tolist()

    # Append raw (unscaled) feature values for interpretability.
    # Use the canonical predict_genome feature schema when possible so columns
    # are stable across single-model and ensemble predictions.
    if use_features and raw_features and len(raw_features) == len(all_accessions):
        acc_to_bp = {acc: len(seq) for acc, seq in zip(chunked_accs, chunked_seqs)}
        raw_feature_map = {fname: i for i, fname in enumerate(feature_names)}
        output_feature_names = [
            f for f in DEFAULT_FEATURE_NAMES if f in raw_feature_map
        ]
        for fname in output_feature_names:
            if fname == "fragment_size":
                data_dict["fragment_size_bp"] = [int(acc_to_bp.get(a, 0)) for a in all_accessions]
            else:
                i = raw_feature_map[fname]
                data_dict[fname] = [float(f[i]) for f in raw_features]

    df = pl.DataFrame(data_dict)
    df = df.sort("accession")
    float_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt in (pl.Float32, pl.Float64)]
    if float_cols:
        df = df.with_columns([pl.col(c).round(4) for c in float_cols])
    df.write_csv(args.output, separator="\t")

    # ── Optional genome-level aggregation ────────────────────────────────────
    genome_df: Optional[pl.DataFrame] = None
    if args.aggregate_chunks:
        logger.info("Aggregating chunk predictions -> genome-level consensus")
        logger.info("Method: mean-pool calibrated class probabilities per parent genome")
        genome_df = _aggregate_chunks(
            chunk_df=df,
            class_cols=class_cols,
            prok_col_idx=prok_col_idx,
            label_mapping=label_mapping,
        )
        genome_float_cols = [
            c for c, dt in zip(genome_df.columns, genome_df.dtypes)
            if dt in (pl.Float32, pl.Float64)
        ]
        if genome_float_cols:
            genome_df = genome_df.with_columns(
                [pl.col(c).round(4) for c in genome_float_cols]
            )
        args.genome_output.parent.mkdir(parents=True, exist_ok=True)
        genome_df.write_csv(args.genome_output, separator="\t")
        logger.info(f"Genome-level output: {args.genome_output}")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 68)
    logger.info("Done!")
    logger.info(f"Sequences processed: {len(accessions)}")
    logger.info(f"Chunks processed: {len(chunked_seqs)}")
    logger.info(f"Folds used: {args.num_folds}")
    logger.info(f"Temperature (T): {T_joint:.6f}")
    logger.info(f"Chunk output: {args.output}")
    if prok_col_idx is not None:
        n_prok = sum(1 for h in predicted_hosts if "prokaryote" in h.lower())
        logger.info(f"Prokaryotic chunks: {n_prok}")
        logger.info(f"Eukaryotic chunks: {len(predicted_hosts) - n_prok}")
    if genome_df is not None:
        genome_preds = genome_df["predicted_host"].to_list()
        logger.info(f"Genome output: {args.genome_output}")
        logger.info("Consensus mode: mean-pooled calibrated probabilities")
        logger.info(f"Genomes predicted: {len(genome_preds)}")
        if prok_col_idx is not None:
            n_prok_g = sum(1 for h in genome_preds if "prokaryote" in h.lower())
            logger.info(f"Prokaryotic genomes: {n_prok_g}")
            logger.info(f"Eukaryotic genomes: {len(genome_preds) - n_prok_g}")
    logger.info("=" * 68)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--fasta", type=click.Path(path_type=pathlib.Path, exists=True, dir_okay=False), required=True, help="Input FASTA file.")
@click.option("--output", type=click.Path(path_type=pathlib.Path, file_okay=False), required=True, help="Output directory.")
@click.option("--prefix", default="vhamster", show_default=True, help="Base filename prefix for outputs.")
@click.option("--force", "force", is_flag=True, help="Overwrite output if it exists.")
@click.option("--ensemble-dir", type=click.Path(path_type=pathlib.Path), default=_DEFAULT_MODEL_ROOT / "best_params_20260331", show_default=True, help="Root directory containing fold_* subdirs.")
@click.option("--fold-dirs", type=click.Path(path_type=pathlib.Path), multiple=True, help="Explicit fold directories (overrides --ensemble-dir).")
@click.option("--num-folds", type=int, default=5, show_default=True, help="Number of folds to use.")
@click.option("--fold-index", type=int, default=None, help="Use only one fold by index, e.g. 0..4.")
@click.option("--checkpoint-subdir", default="best_macro_f1_model", show_default=True, help="Checkpoint subdirectory name.")
@click.option("--temperature-file", type=str, default=str(_DEFAULT_MODEL_ROOT / "joint_temperature.pt"), show_default=True, help="Temperature as scalar (e.g. 1.0) or path to saved T_joint file.")
@click.option("--chunk-size", type=int, default=10000, show_default=True, help="Chunk length in bp.")
@click.option("--overlap", type=int, default=1000, show_default=True, help="Overlap between chunks in bp.")
@click.option("--batch-size", type=int, default=16, show_default=True, help="Inference batch size.")
@click.option("--fp16", is_flag=True, help="Use FP16 mixed precision.")
@click.option("--num-workers", type=int, default=4, show_default=True, help="DataLoader workers.")
@click.option("--aggregate-chunks/--no-aggregate-chunks", default=True, show_default=True, help="Enable/disable genome-level consensus output.")
@click.option("--genome-output", type=click.Path(path_type=pathlib.Path), default=None, help="Path for genome-level TSV output.")
def main(
    fasta: pathlib.Path,
    output: pathlib.Path,
    prefix: str,
    force: bool,
    ensemble_dir: pathlib.Path,
    fold_dirs: Tuple[pathlib.Path, ...],
    num_folds: int,
    fold_index: Optional[int],
    checkpoint_subdir: str,
    temperature_file: str,
    chunk_size: int,
    overlap: int,
    batch_size: int,
    fp16: bool,
    num_workers: int,
    aggregate_chunks: bool,
    genome_output: Optional[pathlib.Path],
) -> None:
    output_dir = output
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = _configure_logging(output_dir, prefix)

    args = SimpleNamespace(
        fasta=fasta,
        output_dir=output_dir,
        output=output_dir / f"{prefix}.chunks.tsv",
        prefix=prefix,
        force=force,
        ensemble_dir=ensemble_dir,
        fold_dirs=list(fold_dirs) if fold_dirs else None,
        num_folds=num_folds,
        fold_index=fold_index,
        checkpoint_subdir=checkpoint_subdir,
        temperature_file=temperature_file,
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=batch_size,
        fp16=fp16,
        num_workers=num_workers,
        aggregate_chunks=aggregate_chunks,
        genome_output=genome_output,
    )

    if args.aggregate_chunks and args.genome_output is None:
        args.genome_output = args.output_dir / f"{args.prefix}.genomes.tsv"

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

    logger.info(f"Logging to: {log_path}")
    _run(args)


if __name__ == "__main__":
    main()
