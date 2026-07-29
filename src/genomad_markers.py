#!/usr/bin/env python3
"""
Extraction of geNomad viral marker hits from nucleotide sequences.

Shared module used by both inference (predict_genome.py) and offline
pre-computation (fine_tuning/extract_genomad_features.py).

Pipeline
--------
1. Predict proteins from nucleotide sequences using pyrodigal-gv.
2. Run MMseqs2 easy-search against the geNomad marker database.
3. Filter hits to viral-specific markers (specificity_class == 'VV').
4. Return / serialise a {seq_id -> {marker_id: max_bitscore}} dict.
"""

import concurrent.futures
import io
import pathlib
import pickle
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Tuple
import polars as pl
import pyrodigal_gv
from Bio import SeqIO


# ---------------------------------------------------------------------------
# Viral marker whitelist
# ---------------------------------------------------------------------------

def load_viral_marker_metadata(metadata_tsv: pathlib.Path) -> Tuple[set, Dict[str, Dict[str, str]]]:
    """
    Read genomad_marker_metadata.tsv and return:
      1. a set of VV marker IDs
      2. a metadata mapping keyed by marker ID for viral-only markers
    """
    df = pl.read_csv(metadata_tsv, separator="\t", null_values=["", "NA", "na", "NaN", "nan"], infer_schema_length=0)
    df.columns = [c.strip().lower() for c in df.columns]

    if "marker" not in df.columns:
        raise ValueError(
            f"Expected a 'marker' column in {metadata_tsv}. "
            f"Found columns: {list(df.columns)}"
        )
    if "specificity_class" not in df.columns:
        raise ValueError(
            f"Expected a 'specificity_class' column in {metadata_tsv}. "
            f"Found columns: {list(df.columns)}"
        )

    viral_df = df.filter(pl.col("specificity_class").str.to_uppercase() == "VV")
    whitelist = set(viral_df["marker"].cast(pl.Utf8).to_list())
    metadata_by_marker = {
        str(row["marker"]): {
            str(col): ("" if row[col] is None else str(row[col]))
            for col in viral_df.columns
        }
        for row in viral_df.iter_rows(named=True)
    }
    print(f"[genomad] Loaded {len(whitelist):,} viral-only (VV) markers from {metadata_tsv}")
    return whitelist, metadata_by_marker


# ---------------------------------------------------------------------------
# Protein prediction
# ---------------------------------------------------------------------------

def predict_proteins(
    sequences: Dict[str, str],
    threads: int = 4,
) -> Dict[str, str]:
    """
    Run pyrodigal-gv on a dict of {seq_id: nucleotide_sequence} using a
    ThreadPoolExecutor and return {protein_id: amino_acid_sequence}.
    """
    gene_finder = pyrodigal_gv.ViralGeneFinder(meta=True)

    def process_record(item):
        seq_id, nt_seq = item
        nt_seq = nt_seq.upper()
        if len(nt_seq) < 100:
            return []
        try:
            genes = gene_finder.find_genes(nt_seq.encode())
            return {
                f"{seq_id}_{i}": str(gene.translate())
                for i, gene in enumerate(genes, start=1)
            }
        except Exception as exc:
            print(
                f"[genomad] Warning: pyrodigal-gv failed on {seq_id}: {exc}",
                file=sys.stderr,
            )
            return {}

    proteins: Dict[str, str] = {}
    written = 0
    print(
        f"[genomad] Extracting ORFs from {len(sequences)} sequences "
        f"using {threads} threads..."
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        for protein_map in executor.map(process_record, sequences.items()):
            proteins.update(protein_map)
            written += len(protein_map)

    print(f"[genomad] Predicted {written:,} proteins.")
    return proteins


def predict_protein_records(
    sequences: Dict[str, str],
    threads: int = 4,
) -> Dict[str, Dict[str, Any]]:
    """
    Run pyrodigal-gv on a dict of {seq_id: nucleotide_sequence} and return
    per-protein records including amino-acid sequence and genomic coordinates.
    """
    gene_finder = pyrodigal_gv.ViralGeneFinder(meta=True)

    def process_record(item):
        seq_id, nt_seq = item
        nt_seq = nt_seq.upper()
        if len(nt_seq) < 100:
            return {}
        try:
            genes = gene_finder.find_genes(nt_seq.encode())
            return {
                f"{seq_id}_{i}": {
                    "amino_acid_sequence": str(gene.translate()),
                    "protein_start": int(gene.begin),
                    "protein_end": int(gene.end),
                    "protein_strand": int(gene.strand),
                    "partial_begin": bool(gene.partial_begin),
                    "partial_end": bool(gene.partial_end),
                    "start_type": str(gene.start_type),
                    "rbs_motif": "" if gene.rbs_motif is None else str(gene.rbs_motif),
                    "rbs_spacer": "" if gene.rbs_spacer is None else str(gene.rbs_spacer),
                    "translation_table": int(gene.translation_table),
                }
                for i, gene in enumerate(genes, start=1)
            }
        except Exception as exc:
            print(
                f"[genomad] Warning: pyrodigal-gv failed on {seq_id}: {exc}",
                file=sys.stderr,
            )
            return {}

    protein_records: Dict[str, Dict[str, Any]] = {}
    written = 0
    print(
        f"[genomad] Extracting ORFs from {len(sequences)} sequences "
        f"using {threads} threads..."
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        for record_map in executor.map(process_record, sequences.items()):
            protein_records.update(record_map)
            written += len(record_map)

    print(f"[genomad] Predicted {written:,} proteins.")
    return protein_records


def predict_proteins_to_fasta(
    sequences: Dict[str, str],
    threads: int = 4,
) -> str:
    """
    Run pyrodigal-gv on a dict of {seq_id: nucleotide_sequence} using a
    ThreadPoolExecutor and return the predicted proteins as a FASTA string.
    """
    proteins = predict_proteins(sequences, threads=threads)
    buf = io.StringIO()
    for protein_id, aa_seq in proteins.items():
        buf.write(f">{protein_id}\n{aa_seq}\n")
    return buf.getvalue()


def predict_proteins_from_fasta(
    fasta_path: pathlib.Path,
    protein_fasta_path: pathlib.Path,
    threads: int = 4,
) -> None:
    """
    Run pyrodigal-gv on a FASTA file and write predicted proteins to
    protein_fasta_path.  Convenience wrapper used by the offline script.
    """
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    sequences = {r.id: str(r.seq) for r in records}
    protein_fasta_str = predict_proteins_to_fasta(sequences, threads=threads)
    with open(protein_fasta_path, "w") as fh:
        fh.write(protein_fasta_str)


# ---------------------------------------------------------------------------
# MMseqs2 search
# ---------------------------------------------------------------------------

def run_mmseqs_search(
    query_fasta: pathlib.Path,
    target_db: pathlib.Path,
    result_tsv: pathlib.Path,
    tmp_dir: pathlib.Path,
    threads: int = 4,
    min_seq_id: float = 0.0,
    evalue: float = 1e-3,
) -> None:
    """Run mmseqs easy-search and write results as TSV."""
    format_output = "query,target,evalue,bits,qlen,tlen,qstart,qend,tstart,tend,alnlen"
    cmd = [
        "mmseqs",
        "easy-search",
        str(query_fasta),
        str(target_db),
        str(result_tsv),
        str(tmp_dir),
        "--threads", str(threads),
        "--min-seq-id", str(min_seq_id),
        "-e", str(evalue),
        "--format-output", format_output,
        "-v", "1",
    ]
    print(f"[genomad] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            f"mmseqs easy-search failed with exit code {result.returncode}"
        )


# ---------------------------------------------------------------------------
# Parse MMseqs2 results
# ---------------------------------------------------------------------------

def parse_mmseqs_results(
    result_tsv: pathlib.Path,
    viral_whitelist: set,
    metadata_by_marker: Dict[str, Dict[str, str]],
    protein_records: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, str]]]:
    """
    Parse MMseqs2 TSV output and return:
      1. {parent_seq_id -> {marker_id: max_bitscore}}
      2. one annotation record per retained viral hit

        Expected TSV columns:
            query, target, evalue, bits, qlen, tlen, qstart, qend, tstart, tend, alnlen
    Parent seq ID is recovered by stripping the trailing _<orf_index> suffix.
    When the same marker hits multiple ORFs of the same sequence, the maximum
    bitscore across all ORF hits is retained.
    """
    hits: dict = {}
    annotation_rows: List[Dict[str, str]] = []

    if not result_tsv.exists() or result_tsv.stat().st_size == 0:
        print("[genomad] MMseqs2 produced no hits.")
        return {}, []

    with open(result_tsv) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 11:
                continue
            query_protein_id = parts[0]
            target_marker_id = parts[1]
            evalue = parts[2] if len(parts) > 2 else ""
            try:
                bitscore = float(parts[3])
            except (ValueError, IndexError):
                bitscore = 1.0
            try:
                query_length = int(parts[4])
                target_length = int(parts[5])
                query_start = int(parts[6])
                query_end = int(parts[7])
                target_start = int(parts[8])
                target_end = int(parts[9])
                alignment_length = int(parts[10])
            except (ValueError, IndexError):
                query_length = 0
                target_length = 0
                query_start = 0
                query_end = 0
                target_start = 0
                target_end = 0
                alignment_length = 0

            query_coverage = (alignment_length / query_length) if query_length else 0.0
            target_coverage = (alignment_length / target_length) if target_length else 0.0

            if target_marker_id not in viral_whitelist:
                continue

            parent_id = query_protein_id.rsplit("_", maxsplit=1)[0]
            seq_hits = hits.setdefault(parent_id, {})
            if target_marker_id not in seq_hits or bitscore > seq_hits[target_marker_id]:
                seq_hits[target_marker_id] = bitscore

            protein_record = protein_records.get(query_protein_id, {})
            annotation_row = {
                "parent_seq_id": parent_id,
                "protein_id": query_protein_id,
                "amino_acid_sequence": str(protein_record.get("amino_acid_sequence", "")),
                "protein_start": str(protein_record.get("protein_start", "")),
                "protein_end": str(protein_record.get("protein_end", "")),
                "protein_strand": str(protein_record.get("protein_strand", "")),
                "partial_begin": str(protein_record.get("partial_begin", "")),
                "partial_end": str(protein_record.get("partial_end", "")),
                "start_type": str(protein_record.get("start_type", "")),
                "rbs_motif": str(protein_record.get("rbs_motif", "")),
                "rbs_spacer": str(protein_record.get("rbs_spacer", "")),
                "translation_table": str(protein_record.get("translation_table", "")),
                "marker_id": target_marker_id,
                "evalue": str(evalue),
                "bitscore": str(bitscore),
                "query_length": str(query_length),
                "target_length": str(target_length),
                "query_start": str(query_start),
                "query_end": str(query_end),
                "target_start": str(target_start),
                "target_end": str(target_end),
                "alignment_length": str(alignment_length),
                "query_coverage": str(query_coverage),
                "target_coverage": str(target_coverage),
            }
            for key, value in metadata_by_marker.get(target_marker_id, {}).items():
                annotation_row[f"marker_{key}"] = value
            annotation_rows.append(annotation_row)

    total_hits = sum(len(v) for v in hits.values())
    print(
        f"[genomad] Found {total_hits:,} unique viral marker hits across "
        f"{len(hits):,} sequences."
    )
    return hits, annotation_rows


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def extract_genomad_markers(
    sequences: Dict[str, str],
    genomad_db: pathlib.Path,
    genomad_metadata: pathlib.Path,
    threads: int = 4,
    min_seq_id: float = 0.0,
    evalue: float = 1e-3,
    return_details: bool = False,
) -> Dict[str, Dict[str, float]] | Tuple[Dict[str, Dict[str, float]], Dict[str, str], List[Dict[str, str]]]:
    """
    Given a dict of {seq_id: nucleotide_sequence}, run the full geNomad marker
    extraction pipeline.

    By default, returns {seq_id -> {marker_id: max_bitscore}}.
    If return_details=True, additionally returns predicted amino-acid sequences
    and one annotation record per retained viral hit.

    This is the primary entry point used at inference time.
    """
    viral_whitelist, metadata_by_marker = load_viral_marker_metadata(genomad_metadata)

    with tempfile.TemporaryDirectory(prefix="genomad_extract_") as tmp_str:
        tmp_dir = pathlib.Path(tmp_str)
        protein_fasta = tmp_dir / "proteins.faa"
        result_tsv = tmp_dir / "mmseqs_results.tsv"
        mmseqs_tmp = tmp_dir / "mmseqs_tmp"
        mmseqs_tmp.mkdir()

        # Predict proteins and write to temp file
        protein_records = predict_protein_records(sequences, threads=threads)
        protein_fasta_str = "".join(
            f">{protein_id}\n{record['amino_acid_sequence']}\n"
            for protein_id, record in protein_records.items()
        )
        with open(protein_fasta, "w") as fh:
            fh.write(protein_fasta_str)

        if protein_fasta.stat().st_size == 0:
            print("[genomad] No proteins predicted; returning empty marker dict.")
            if return_details:
                return {}, {}, []
            return {}

        run_mmseqs_search(
            query_fasta=protein_fasta,
            target_db=genomad_db / "genomad_db",
            result_tsv=result_tsv,
            tmp_dir=mmseqs_tmp,
            threads=threads,
            min_seq_id=min_seq_id,
            evalue=evalue,
        )

        marker_hits, annotation_rows = parse_mmseqs_results(
            result_tsv,
            viral_whitelist,
            metadata_by_marker,
            protein_records,
        )
        if return_details:
            protein_sequences = {
                protein_id: str(record["amino_acid_sequence"])
                for protein_id, record in protein_records.items()
            }
            return marker_hits, protein_sequences, annotation_rows
        return marker_hits


def write_annotation_rows(annotation_rows: List[Dict[str, str]], output_tsv: pathlib.Path) -> None:
    """Write per-hit annotations to TSV, including amino-acid sequences and marker metadata."""
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    if annotation_rows:
        pl.DataFrame(annotation_rows).to_csv(output_tsv, separator="\t", index=False)
    else:
        pl.DataFrame(
            columns=[
                "parent_seq_id",
                "protein_id",
                "amino_acid_sequence",
                "protein_start",
                "protein_end",
                "protein_strand",
                "partial_begin",
                "partial_end",
                "start_type",
                "rbs_motif",
                "rbs_spacer",
                "translation_table",
                "marker_id",
                "evalue",
                "bitscore",
                "query_length",
                "target_length",
                "query_start",
                "query_end",
                "target_start",
                "target_end",
                "alignment_length",
                "query_coverage",
                "target_coverage",
            ]
        ).to_csv(output_tsv, separator="\t", index=False)
    print(f"[genomad] Saved hit annotations to {output_tsv}")


def write_proteins_to_fasta(protein_sequences: Dict[str, str], output_fasta: pathlib.Path) -> None:
    """Write predicted amino-acid sequences to FASTA."""
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    with open(output_fasta, "w") as fh:
        for protein_id, aa_seq in protein_sequences.items():
            fh.write(f">{protein_id}\n{aa_seq}\n")
    print(f"[genomad] Saved predicted proteins to {output_fasta}")


def extract_genomad_markers_from_fasta(
    fasta_path: pathlib.Path,
    genomad_db: pathlib.Path,
    genomad_metadata: pathlib.Path,
    output_pkl: pathlib.Path,
    output_proteins_faa: pathlib.Path | None = None,
    output_annotations_tsv: pathlib.Path | None = None,
    threads: int = 4,
    min_seq_id: float = 0.0,
    evalue: float = 1e-3,
) -> Dict[str, Dict[str, float]]:
    """
    Run the full pipeline starting from a FASTA file and save results to a
    pickle.  Used by the offline extraction script.  The pickle stores
    {seq_id -> {marker_id: max_bitscore}}.

    Optional sidecar outputs can also be written:
      - output_proteins_faa: predicted amino acid sequences in FASTA format
      - output_annotations_tsv: per-hit annotations with marker metadata
    """
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    sequences = {r.id: str(r.seq) for r in records}

    marker_hits, protein_sequences, annotation_rows = extract_genomad_markers(
        sequences=sequences,
        genomad_db=genomad_db,
        genomad_metadata=genomad_metadata,
        threads=threads,
        min_seq_id=min_seq_id,
        evalue=evalue,
        return_details=True,
    )

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, "wb") as fh:
        pickle.dump(marker_hits, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[genomad] Saved marker hits to {output_pkl}")

    if output_proteins_faa is not None:
        write_proteins_to_fasta(protein_sequences, output_proteins_faa)

    if output_annotations_tsv is not None:
        write_annotation_rows(annotation_rows, output_annotations_tsv)

    return marker_hits
