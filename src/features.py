#!/usr/bin/env python3 
"""
modules to precompute features using pyrodigal 
"""

# imports 
import pyrodigal_gv
from typing import Dict, List, Any


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
            "genome_name": genome_name, "fragment_size": fragment_size,
            "strand_switch_rate": 0.0, "coding_density": 0.0, "leaderless_freq": 0.0,
            "short_utr_freq": 0.0, "no_rbs_freq": 0.0, "sd_bacteroidetes_rbs_freq": 0.0,
            "sd_canonical_rbs_freq": 0.0, "tatata_rbs_freq": 0.0, "mean_rbs_score": 0.0,
            "gene_density": 0.0, "gene_density_fwd": 0.0, "gene_density_rev": 0.0,
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
    
    return {
        "genome_name": genome_name, "fragment_size": fragment_size, "strand_switch_rate": strand_switch_rate,
        "coding_density": coding_density, "leaderless_freq": leaderless_freq, "short_utr_freq": short_utr_freq,
        "no_rbs_freq": no_rbs_freq, "sd_bacteroidetes_rbs_freq": sd_bacteroidetes_rbs_freq,
        "sd_canonical_rbs_freq": sd_canonical_rbs_freq, "tatata_rbs_freq": tatata_rbs_freq,
        "mean_rbs_score": mean_rbs_score, "gene_density": gene_density,
        "gene_density_fwd": gene_density_fwd, "gene_density_rev": gene_density_rev,
    }

def extract_features_worker(args):
    """Worker for parallel feature extraction."""
    # Accepts (seq, acc, feature_names, chunk_size) only
    seq, acc, feature_names, chunk_size = args
    feat_dict = extract_features(seq, acc, chunk_size=chunk_size)
    return [feat_dict[name] for name in feature_names]
