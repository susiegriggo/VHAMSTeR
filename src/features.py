#!/usr/bin/env python3
"""
modules to precompute features using pyrodigal
"""

# imports
from ast import DictComp

import pyrodigal_gv
from typing import Dict, List, Any, Optional
import numpy as np
import polars as pl


# motifs to detect - following geNomad's classification
SD_CANONICAL_PATTERNS = [
    "3Base/5BMM", "4Base/6BMM", "AGG", "AGGA", "AGGA/GGAG/GAGG",
    "AGGAG", "AGGAG/GGAGG", "AGGAG(G)/GGAGG", "AGGAGG",
    "AGxAG", "AGxAGG/AGGxGG", "GAG", "GAGG", "GAGGA",
    "GGA", "GGA/GAG/AGG", "GGAG", "GGAG/GAGG", "GGAGG", "GGAGGA", "GGxGG"
]

def classify_rbs_motif(motif: str) -> Dict[str, bool]:
    """Classify an RBS motif into geNomad categories."""
    if not motif:
        return {'bacteroidetes': False, 'canonical': False, 'tatata': False}
    
    motif_upper = motif.upper()
    is_bacteroidetes = motif_upper in ['TAA', 'TAAA', 'TAAAA', 'TAAAT', 'TAAAAA', 'TAAAAT']
    is_canonical = motif_upper in SD_CANONICAL_PATTERNS
    is_tatata = motif_upper in ['ATA', 'ATAT', 'ATATA', 'ATATAT', 'TAT', 'TATA', 'TATAT', 'TATATA']
    
    return {
        'bacteroidetes': is_bacteroidetes,
        'canonical': is_canonical,
        'tatata': is_tatata
    }

def predict_orfs_with_rbs(seq: str, min_len: int = 90, meta: bool = True) -> List[Dict[str, Any]]:
    """Run Prodigal on sequence."""
    gene_finder = pyrodigal_gv.ViralGeneFinder(meta=True)
    
    genes = gene_finder.find_genes(bytes(seq.encode()))
    orfs = []
    prev_stop = {}
    prev_strand = None

    for g in genes:
        if len(g.translate()) < min_len:
            prev_stop['+' if g.strand == 1 else '-'] = g.end
            continue

        start0 = g.begin - 1
        end0 = g.end
        strand = '+' if g.strand == 1 else '-'
        leaderless = start0 <= 2
        prev = prev_stop.get(strand, -3)
        short_utr = (start0 - prev) <= 3
        prev_stop[strand] = end0
        strand_switch = (prev_strand is not None and strand != prev_strand)
        prev_strand = strand
        has_rbs = bool(g.rbs_motif)
        rbs_motif = g.rbs_motif.upper() if has_rbs else ""
        rbs_score = float(g.score) if has_rbs else 0.0
        rbs_class = classify_rbs_motif(rbs_motif)

        orfs.append({
            "start": start0, "end": end0, "strand": strand, "length": len(g.translate()),
            "leaderless": leaderless, "short_utr": short_utr, "strand_switch": strand_switch,
            "has_rbs": has_rbs, "rbs_motif": rbs_motif, "rbs_score": rbs_score,
            "is_bacteroidetes_rbs": rbs_class['bacteroidetes'], "is_canonical_rbs": rbs_class['canonical'],
            "is_tatata_rbs": rbs_class['tatata'],
        })
    return orfs

def extract_features(seq: str, genome_name: str, chunk_size: int = 10000) -> Dict[str, Any]:
    """Extract features from a single sequence, including fragment_size as a portion of chunk_size."""
    seq_len = len(seq)
    seq_len_kb = seq_len / 1000.0
    fragment_size = seq_len / chunk_size if chunk_size > 0 else 0.0
    orfs = predict_orfs_with_rbs(seq)
    
    if not orfs:
        return {
            "genome_name": genome_name, "fragment_size": fragment_size, "n_genes": 0,
            "strand_switch_rate": 0.0, "coding_density": 0.0, "leaderless_freq": 0.0,
            "short_utr_freq": 0.0, "no_rbs_freq": 0.0, "sd_bacteroidetes_rbs_freq": 0.0,
            "sd_canonical_rbs_freq": 0.0, "tatata_rbs_freq": 0.0, "mean_rbs_score": 0.0,
            "gene_density": 0.0, "gene_density_fwd": 0.0, "gene_density_rev": 0.0,
            "median_orf_length": 0.0, 'max_orf_length': 0.0
        }
    
    n_switches = sum(o["strand_switch"] for o in orfs)
    strand_switch_rate = n_switches / len(orfs) if len(orfs) > 1 else 0.0
    total_coding_bp = sum(o["end"] - o["start"] for o in orfs)
    coding_density = total_coding_bp / seq_len if seq_len > 0 else 0.0
    leaderless_freq = sum(o["leaderless"] for o in orfs) / len(orfs)
    short_utr_freq = sum(o["short_utr"] for o in orfs) / len(orfs)
    no_rbs_freq = sum(1 for o in orfs if not o["has_rbs"]) / len(orfs)
    sd_bacteroidetes_rbs_freq = sum(o["is_bacteroidetes_rbs"] for o in orfs) / len(orfs)
    sd_canonical_rbs_freq = sum(o["is_canonical_rbs"] for o in orfs) / len(orfs)
    tatata_rbs_freq = sum(o["is_tatata_rbs"] for o in orfs) / len(orfs)
    scores = [o["rbs_score"] for o in orfs if o["has_rbs"]]
    mean_rbs_score = sum(scores) / len(scores) if scores else 0.0
    mean_rbs_score = mean_rbs_score / 100.0  # Rescale to match other features
    gene_density = len(orfs) / seq_len_kb if seq_len_kb > 0 else 0.0
    n_fwd = sum(1 for o in orfs if o["strand"] == '+')
    n_rev = sum(1 for o in orfs if o["strand"] == '-')
    gene_density_fwd = n_fwd / seq_len_kb if seq_len_kb > 0 else 0.0
    gene_density_rev = n_rev / seq_len_kb if seq_len_kb > 0 else 0.0
    orf_lengths = [o["length"] for o in orfs]
    median_orf_length = float(np.median(orf_lengths)) if orf_lengths else 0.0
    max_orf_length = float(np.max(orf_lengths)) if orf_lengths else 0.0
    
    return {
        "genome_name": genome_name, "fragment_size": fragment_size, "n_genes": len(orfs),
        "strand_switch_rate": strand_switch_rate, "coding_density": coding_density,
        "leaderless_freq": leaderless_freq, "short_utr_freq": short_utr_freq,
        "no_rbs_freq": no_rbs_freq, "sd_bacteroidetes_rbs_freq": sd_bacteroidetes_rbs_freq,
        "sd_canonical_rbs_freq": sd_canonical_rbs_freq, "tatata_rbs_freq": tatata_rbs_freq,
        "mean_rbs_score": mean_rbs_score, "gene_density": gene_density,
        "gene_density_fwd": gene_density_fwd, "gene_density_rev": gene_density_rev,
        "median_orf_length": median_orf_length, 
        "max_orf_length": max_orf_length
    }

def extract_features_worker(args):
    """Worker for parallel feature extraction."""
    seq, acc, feature_names, chunk_size = args
    feat_dict = extract_features(seq, acc, chunk_size=chunk_size)
    return [feat_dict[name] for name in feature_names]


# ---------------------------------------------------------------------------
# Feature name constants
# ---------------------------------------------------------------------------

ARCH_FEATURE_NAMES = [
    "fragment_size", "strand_switch_rate", "coding_density",
    "leaderless_freq", "short_utr_freq", "no_rbs_freq",
    "sd_bacteroidetes_rbs_freq", "sd_canonical_rbs_freq", "tatata_rbs_freq",
    "mean_rbs_score", "gene_density", "gene_density_fwd", "gene_density_rev",
]

# fragment_size is excluded as a model input; n_genes is stored alongside arch
# features for use as the marker-frequency denominator.
ARCH_FEATURE_NAMES_NO_FRAGMENT = [f for f in ARCH_FEATURE_NAMES if f != "fragment_size"]

# Marker classification codes used by compute_marker_specificity.py
_PROK_CODE = "K"
_KT_CODE = "KT"
_EUK_CODES = {"A", "F", "P", "T", "E"}

XGB1_MARKER_FEATURE_NAMES = [
    "prok_marker_freq", "euk_marker_freq", 
    "prok_median_bitscore", "euk_median_bitscore",
    "prok_max_bitscore", "euk_max_bitscore", 
]
XGB2_MARKER_FEATURE_NAMES = [
    "protist_marker_freq", "fungi_marker_freq", "animal_marker_freq", "plant_marker_freq", 
    "protist_median_bitscore", "fungi_median_bitscore",
    "animal_median_bitscore", "plant_median_bitscore", 
    "protist_max_bitscore", "fungi_max_bitscore",
    "animal_max_bitscore", "plant_max_bitscore"
]

XGB1_FEATURE_NAMES = ARCH_FEATURE_NAMES_NO_FRAGMENT + XGB1_MARKER_FEATURE_NAMES
XGB2_FEATURE_NAMES = ARCH_FEATURE_NAMES_NO_FRAGMENT + XGB2_MARKER_FEATURE_NAMES


# ---------------------------------------------------------------------------
# Specificity-based marker feature builders
# ---------------------------------------------------------------------------

def _median_or_nan(values: list) -> float:
    return float(np.median(values)) if values else np.nan

def _max_or_nan(values: list) -> float:
    return float(np.max(values)) if values else np.nan


def build_xgb1_marker_features(
    accessions: List[str],
    marker_dict: Dict[str, Dict[str, float]],
    marker_classification: Dict[str, str],
    cutoff_dict: Dict[str, float],
    features_df: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    Build 6 XGB-1 marker features per accession.

    Frequencies are normalised by n_genes (total predicted ORFs for the chunk),
    giving marker counts per gene. Median bitscores are 0 when no markers of
    that class are present.

    Parameters
    ----------
    accessions            : ordered list of sequence/chunk IDs
    marker_dict           : {acc -> {marker_id -> bitscore}}
    marker_classification : {marker_id -> class_code} (from compute_marker_specificity.py)
    cutoff_dict           : {marker_id -> cutoff_value} (from compute_marker_specificity.py)
    features_df           : DataFrame indexed by accession, must contain 'n_genes'.
    """
    rows = []
    for acc in accessions:
        hits = marker_dict.get(acc, {})

        prok_bs: List[float] = []
        euk_bs: List[float] = []
        kt_bs: List[float] = []
        u_bs: List[float] = []
        u_count = 0 # count unknown markers
        valid_hit_count = 0 

        for marker, bitscore in hits.items():
            if marker in marker_classification:
                valid_hit_count += 1
                code = marker_classification[marker]
                
                cutoff = cutoff_dict.get(marker)
                if cutoff is None or cutoff <= 0:
                    raise ValueError(f"Cutoff for marker {marker} is missing")
                score_ratio = bitscore / cutoff

                if code == _PROK_CODE:
                    prok_bs.append(score_ratio)
                elif code == _KT_CODE:
                    kt_bs.append(score_ratio)
                elif code in _EUK_CODES:
                    euk_bs.append(score_ratio)
                elif code == "U":
                    u_bs.append(score_ratio)

        denom = valid_hit_count if valid_hit_count > 0 else 1

        rows.append({
            "prok_marker_freq":      len(prok_bs) / denom,
            "euk_marker_freq":       len(euk_bs)  / denom,
            "prok_median_bitscore":  _median_or_nan(prok_bs),
            "euk_median_bitscore":   _median_or_nan(euk_bs),
            "prok_max_bitscore":  _max_or_nan(prok_bs),
            "euk_max_bitscore":   _max_or_nan(euk_bs),
        })

    return pl.DataFrame(rows, schema=XGB1_MARKER_FEATURE_NAMES)


def build_xgb2_marker_features(
    accessions: List[str],
    marker_dict: Dict[str, Dict[str, float]],
    marker_classification: Dict[str, str],
    cutoff_dict: Dict[str, float],
    features_df: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    Build 8 XGB-2 marker features per accession.

    KT (prokaryote+protist) markers are counted as protist (T) in XGB-2 because
    in the eukaryote-only context both host groups are eukaryotic.

    Parameters
    ----------
    Same as build_xgb1_marker_features.
    """
    rows = []
    for acc in accessions:
        hits = marker_dict.get(acc, {})

        protist_bs: List[float] = []
        fungi_bs:   List[float] = []
        animal_bs:  List[float] = []
        plant_bs:   List[float] = []
        u_bs:       List[float] = []

        valid_hit_count = 0

        for marker, bitscore in hits.items():
            if marker in marker_classification:
                valid_hit_count += 1

                cutoff = cutoff_dict.get(marker)
                if cutoff is None or cutoff <= 0:
                    raise ValueError(f"Cutoff for marker {marker} is missing")
                score_ratio = bitscore / cutoff

                code = marker_classification.get(marker)
                if code in ("T", _KT_CODE):
                    protist_bs.append(score_ratio)
                elif code == "F":
                    fungi_bs.append(score_ratio)
                elif code == "A":
                    animal_bs.append(score_ratio)
                elif code == "P":
                    plant_bs.append(score_ratio)
                elif code == "U":  
                    u_bs.append(score_ratio)

        denom = valid_hit_count if valid_hit_count > 0 else 1

        rows.append({
            "protist_marker_freq":     len(protist_bs) / denom,
            "fungi_marker_freq":       len(fungi_bs)   / denom,
            "animal_marker_freq":      len(animal_bs)  / denom,
            "plant_marker_freq":       len(plant_bs)   / denom,
            "protist_median_bitscore": _median_or_nan(protist_bs),
            "fungi_median_bitscore":   _median_or_nan(fungi_bs),
            "animal_median_bitscore":  _median_or_nan(animal_bs),
            "plant_median_bitscore":   _median_or_nan(plant_bs),
            "protist_max_bitscore":    _max_or_nan(protist_bs),
            "fungi_max_bitscore":      _max_or_nan(fungi_bs),
            "animal_max_bitscore":     _max_or_nan(animal_bs),
            "plant_max_bitscore":      _max_or_nan(plant_bs),
        })

    return pl.DataFrame(rows, schema=XGB2_MARKER_FEATURE_NAMES)

def build_gate_context_features(
    accessions: List[str],
    marker_dict: Dict[str, Dict[str, float]],
    marker_classification: Dict[str, str],
    cutoff_dict: Dict[str, float],
    features_df: Optional[pl.DataFrame] = None,
) -> np.ndarray:
    """
    Builds the 2 strictly curated contextual gate features per accession:
    1. total_specific_freq (Density: valid hits / n_genes)
    2. specific_median_bitscore
    
    Returns an (N, 2) numpy array of float32.
    """
    valid_codes = {'A', 'F', 'P', 'T', 'K', 'KT'}
    rows = []

    # Build accession -> n_genes lookup from polars DataFrame (must have 'accession' and 'n_genes' columns)
    _n_genes_lookup: Dict[str, float] = {}
    if features_df is not None and "accession" in features_df.columns and "n_genes" in features_df.columns:
        _n_genes_lookup = dict(zip(features_df["accession"].to_list(), features_df["n_genes"].to_list()))

    for acc in accessions:
        hits = marker_dict.get(acc, {})
        
        # --- 1. TRUE DENSITY DENOMINATOR (Based on n_genes) ---
        n_genes = 1
        if features_df is not None and _n_genes_lookup:
            n_genes = _n_genes_lookup.get(acc, 1)
        denom = max(1.0, float(n_genes))
        
        # --- 2. Collect targets ---
        spec_bits = []
        for m, bs in hits.items():
            if m in marker_classification:
                code = marker_classification[m]
                if code in valid_codes:
                    cutoff = cutoff_dict.get(m)
                    if cutoff is None or cutoff <= 0:
                        raise ValueError(f"Cutoff for marker {m} is missing")
                    spec_bits.append(bs / cutoff)
                    
        total_spec_freq = len(spec_bits) / denom
        spec_median = float(np.median(spec_bits)) if spec_bits else np.nan
        spec_max = float(np.max(spec_bits)) if spec_bits else np.nan
        
        rows.append([total_spec_freq, spec_median, spec_max])
        
    return np.array(rows, dtype=np.float32)