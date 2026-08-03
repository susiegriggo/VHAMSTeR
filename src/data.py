#!/usr/bin/env python3
"""
modules to handle data
"""

# imports
import pathlib
import pandas as pd
import multiprocessing
import concurrent.futures
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Optional
import torch
from tqdm.auto import tqdm
from loguru import logger
import numpy as np
import random
import sequences
from torch.utils.data import Sampler
from features import ARCH_FEATURE_NAMES_NO_FRAGMENT


class EpochRedrawProkaryoteSampler(Sampler):
    """
    Dynamically undersamples the prokaryotic class at each epoch.
    At each epoch, randomly selects a subset of prokaryotic indices and combines with all other indices.
    """
    def __init__(self, labels, prokaryote_idx, max_prop=0.5, random_seed=42):
        self.labels = np.array(labels)
        self.prokaryote_idx = prokaryote_idx
        self.max_prop = max_prop
        self.random_seed = random_seed
        self.other_indices = np.where(self.labels != self.prokaryote_idx)[0]
        self.prok_indices = np.where(self.labels == self.prokaryote_idx)[0]
        self.n_other = len(self.other_indices)
        self.n_prok_desired = int(self.n_other * self.max_prop / (1 - self.max_prop)) if self.max_prop < 1.0 else len(self.prok_indices)
        self.epoch = 0
        self._lat_indices = None

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        random.seed(self.random_seed + self.epoch)
        if self.n_prok_desired < len(self.prok_indices):
            prok_sampled = random.sample(list(self.prok_indices), self.n_prok_desired)
        else:
            prok_sampled = list(self.prok_indices)
        indices = list(self.other_indices) + prok_sampled
        random.shuffle(indices)
        self._lat_indices = indices
        return iter(indices)

    def __len__(self):
        return len(self.other_indices) + min(self.n_prok_desired, len(self.prok_indices))
    
    def get_category_counts(self): 
        """Return a dictionary of class counts for the most recent sampled indices."""
        if self._lat_indices is None:
            raise ValueError("Sampler has not been iterated yet; no indices available.")
        labels_sampled = self.labels[self._lat_indices]
        unique, counts = np.unique(labels_sampled, return_counts=True)
        return dict(zip(unique, counts))

def load_epoch_fold_data(epoch_dir, use_features=True, xgb_features_filename="xgb_probabilities.tsv"):
    """
    Loads FASTA, labels, precomputed XGBoost probabilities, and raw arch/marker
    features from a given epoch/fold directory.

    The raw features (structural/architectural + marker frequencies) come from the
    ``*features.tsv`` file that is NOT ``xgb_probabilities.tsv``.  They are used as
    gate inputs in the ``GenomeClassifier`` to decouple gating from the model's own
    XGBoost predictions and prevent training-set data leakage.

    Returns:
        seqs, labels, accs, idx_to_label, feature_names, features,
        raw_feature_names, raw_features
        (raw_feature_names and raw_features are None when no raw file is found)
    """
    epoch_dir = pathlib.Path(epoch_dir)
    fasta_file = next(epoch_dir.glob("*.fasta"))
    labels_file = next(epoch_dir.glob("*labels.tsv"))
    features_file = epoch_dir / xgb_features_filename
    if use_features and not features_file.exists():
        raise FileNotFoundError(
            f"Expected precomputed {xgb_features_filename} in {epoch_dir} because use_features=True. "
            "Run hierarchical XGBoost stacking first or disable feature usage explicitly."
        )
    if not features_file.exists():
        features_file = None

    # Locate raw arch/marker features file (any *features.tsv that isn't the XGB probs file)
    _raw_feature_candidates = sorted(
        p for p in epoch_dir.glob("*features.tsv")
        if p.name != xgb_features_filename and p.name != "xgb_probabilities.tsv"
    )
    raw_features_file = _raw_feature_candidates[0] if _raw_feature_candidates else None

    seqs, accs = sequences.load_fasta_sequences(fasta_file)
    label_map = {}
    with open(labels_file) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            acc, label = parts[0], parts[1]
            label_map[acc] = label
    labels = [label_map[a] for a in accs]
    unique_labels = sorted(set(labels))
    idx_to_label = {i: l for i, l in enumerate(unique_labels)}
    label_to_idx = {l: i for i, l in idx_to_label.items()}
    labels = [label_to_idx[l] for l in labels]

    feature_names, features = None, None
    if use_features and features_file is not None:
        df = pd.read_csv(features_file, sep='\t')
        id_col = None
        for candidate in ("accession", "chunk_id", "id"):
            if candidate in df.columns:
                id_col = candidate
                break
        if id_col is None:
            id_col = df.columns[0]

        feature_names = [c for c in df.columns if c != id_col]
        if not feature_names:
            raise ValueError(
                f"Precomputed feature file has no usable feature columns: {features_file}"
            )

        feature_df = df.set_index(id_col)[feature_names]
        feature_df.index = feature_df.index.astype(str)
        feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        aligned = feature_df.reindex(accs).fillna(0.0)
        features = aligned.to_numpy(dtype=np.float32)

    # Load raw biological features for gate input (optional — not required to exist)
    raw_feature_names, raw_features = None, None
    if raw_features_file is not None:
        try:
            raw_df = pd.read_csv(raw_features_file, sep='\t')
            raw_id_col = None
            for candidate in ("accession", "chunk_id", "id"):
                if candidate in raw_df.columns:
                    raw_id_col = candidate
                    break
            if raw_id_col is None:
                raw_id_col = raw_df.columns[0]

            # Strictly use the 12 canonical arch features — in canonical order — as gate input.
            # This guarantees training and inference receive the identical feature vector.
            raw_feature_names = [c for c in ARCH_FEATURE_NAMES_NO_FRAGMENT if c in raw_df.columns]
            if raw_feature_names:
                raw_feat_df = raw_df.set_index(raw_id_col)[raw_feature_names]
                raw_feat_df.index = raw_feat_df.index.astype(str)
                raw_feat_df = raw_feat_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
                raw_aligned = raw_feat_df.reindex(accs).fillna(0.0)
                raw_features = raw_aligned.to_numpy(dtype=np.float32)
            else:
                raw_feature_names = None
        except Exception:
            raw_feature_names, raw_features = None, None

    return seqs, labels, accs, idx_to_label, feature_names, features, raw_feature_names, raw_features

def load_labels(labels_file: pathlib.Path) -> Tuple[Dict[str, str], List[str]]:
    """Load labels from TSV/CSV file."""
    try:
        df = pd.read_csv(labels_file, sep='\t', header=0)
    except:
        df = pd.read_csv(labels_file, header=0)
    
    if len(df.columns) < 2:
        raise ValueError(f"Labels file must have at least 2 columns (accession, label)")
    
    accession_col = df.columns[0]
    label_col = df.columns[1]
    
    label_dict = dict(zip(df[accession_col].astype(str), df[label_col].astype(str)))
    label_names = sorted(df[label_col].unique())
    
    logger.info(f"Loaded {len(label_dict)} labels with {len(label_names)} classes: {label_names}")
    return label_dict, label_names


def load_data(
    fasta_file: pathlib.Path,
    labels_file: pathlib.Path,
    test_size: float = 0.2,
    random_seed: int = 42,
    use_rv: bool = False,
    use_features: bool = True,
    use_class_weights: bool = False,
    features_file: pathlib.Path = None,
    output_dir: Optional[pathlib.Path] = None,
) -> Tuple[List[str], List[int], List[str], List[str], List[int], List[str], Dict[int, str], List[str], List[List[float]], List[List[float]], torch.Tensor]:
    """Load sequences and labels, split into train/val sets."""
    label_dict, label_names = load_labels(labels_file)
    label_to_idx = {name: idx for idx, name in enumerate(label_names)}
    idx_to_label = {idx: name for name, idx in label_to_idx.items()}
    
    sequences = []
    labels = []
    accessions = []
    skipped = []
    
    logger.info(f"Loading sequences from {fasta_file}...")
    for record in tqdm(SeqIO.parse(str(fasta_file), "fasta"), desc="Loading sequences"):
        accession = record.id.split()[0]
        if accession not in label_dict:
            skipped.append(accession)
            continue
        sequences.append(str(record.seq).upper())
        labels.append(label_to_idx[label_dict[accession]])
        accessions.append(accession)
    
    if skipped:
        logger.warning(f"Skipped {len(skipped)} sequences without labels")
    
    if len(sequences) == 0:
        raise ValueError("No sequences with labels found!")
    
    logger.info(f"Loaded {len(sequences)} sequences with labels")
    
    label_counts = pd.Series(labels).value_counts().sort_index()
    logger.info("Class distribution:")
    for idx, count in label_counts.items():
        logger.info(f"  {idx_to_label[idx]}: {count} ({100*count/len(labels):.1f}%)")
    
    if use_class_weights:
        total_samples = len(labels)
        num_classes = len(idx_to_label)
        weights = torch.tensor([total_samples / (num_classes * label_counts[idx]) for idx in sorted(idx_to_label.keys())], dtype=torch.float)
        logger.info(f"Using class weights: {weights}")
    else:
        weights = None
    
    # Feature Extraction
    features_list = None
    feature_names = []
    if use_features:
        if features_file and features_file.is_file():
            logger.info(f"Loading pre-computed features from {features_file}...")
            features_df = pd.read_csv(features_file)
            feature_names = [col for col in features_df.columns if col != 'accession']
            features_dict = {row['accession']: [row[name] for name in feature_names] for _, row in features_df.iterrows()}
            features_list = []
            for acc in accessions:
                if acc not in features_dict:
                    raise ValueError(f"Accession {acc} not found in features file")
                features_list.append(features_dict[acc])
        else:
            # Note: requires sequences.extract_features_worker to be importable here if running standard, 
            # but usually called via fine_tune_glm context.
            # Assuming 'from features import extract_features_worker' logic in calling script or similar
            # Since data.py imports 'sequences', make sure 'extract_features_worker' is available or import from features.py
            from features import extract_features_worker, ARCH_FEATURE_NAMES_NO_FRAGMENT
            
            feature_names = ARCH_FEATURE_NAMES_NO_FRAGMENT
            logger.info(f"Extracting {len(feature_names)} features from {len(sequences)} sequences...")
            
            num_workers = min(multiprocessing.cpu_count(), len(sequences))
            logger.info(f"Using {num_workers} workers for parallel feature extraction...")
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                args_list = [(seq, acc, use_rv, feature_names, 10000) for seq, acc in zip(sequences, accessions)]
                features_list = list(tqdm(
                    executor.map(extract_features_worker, args_list), 
                    desc="Extracting features", 
                    total=len(sequences)
                ))
    else:
        logger.info("Skipping feature extraction (--no-use-features)")
    
    # Split
    if use_features:
        train_seqs, val_seqs, train_labels, val_labels, train_accs, val_accs, train_features, val_features = train_test_split(
            sequences, labels, accessions, features_list,
            test_size=test_size, random_state=random_seed, stratify=labels,
        )
    else:
        train_seqs, val_seqs, train_labels, val_labels, train_accs, val_accs = train_test_split(
            sequences, labels, accessions,
            test_size=test_size, random_state=random_seed, stratify=labels,
        )
        train_features = None
        val_features = None
    
    logger.info(f"Split: {len(train_seqs)} train, {len(val_seqs)} validation")

    return train_seqs, train_labels, train_accs, val_seqs, val_labels, val_accs, idx_to_label, feature_names, train_features, val_features, weights