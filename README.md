# VHAMSTeR
![](vhamsterlogo.png)


**V**irus **H**ost **A**ssignment **M**odel using **S**equence **T**ransformers and **R**eading-frames

V-HAMSTeR uses a genomic language model to predict the host of a virus as one of: animal, plant, fungi, protist, or prokaryote.

It is designed for viral sequences up to 10 kbp. Longer sequences are split
into 10 kbp chunks, each chunk is scored independently, and the chunk predictions are
mean-pooled to produce a genome-level consensus prediction.

It runs a 5-fold ensemble and writes:
- chunk-level predictions (10kbp)
- genome-level consensus predictions (mean-pooled over chunks)

## Dependencies

### Python packages (installed automatically)

- `click`
- `loguru`
- `torch`
- `transformers`
- `peft`
- `numpy`
- `polars`
- `biopython`
- `pyrodigal-gv`
- `tqdm`
- `scikit-learn`
- `xgboost`
- `joblib`

### External tools (must be installed separately)

- `mmseqs2` — used for geNomad marker search. Install via conda:

```bash
conda install -c bioconda mmseqs2
```

### geNomad database

A geNomad database is required at runtime and must be downloaded separately.
See https://github.com/apcamargo/genomad for instructions. Pass the path to the
database root directory with `--genomad-db`.

## Installation

### pip (editable install from repo)

Clone the repository and install from the repository root:

```bash
git clone https://code.jgi.doe.gov/SusieGrigson/vhamster.git
cd vhamster
pip install -e .
```

This installs all Python dependencies listed in `pyproject.toml` and registers the
`vhamster` and `vhamster-install-models` shell commands.

> **Note:** `mmseqs2` is not a Python package and must be installed via conda (see above).

### Optional GPU support

CUDA is only needed if you want to run on GPU. CPU inference is supported and is
often fast enough unless you are processing a large amount of data.

If you want GPU support, install a PyTorch build that matches your CUDA version.
Visit https://pytorch.org/get-started/locally/ to get the right command for your
setup, for example:

```bash
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

If you already ran `pip install -e .`, installing the appropriate PyTorch build
afterward is fine and will replace the default wheel if needed.

### Conda environment (recommended for HPC)

An `environment.yml` is provided that includes `mmseqs2` from `bioconda`. If you
want GPU support, edit the `pytorch-cuda` version to match your cluster before running:

```bash
conda env create -f environment.yml
conda activate vhamster
```

## Model installation

Model installation is a separate step after installing `vhamster` itself.

Install the pretrained model bundle (v1.2.0) from NERSC:

```bash
vhamster-install-models
```

By default, this installs into an environment-scoped location in the active Python
environment: `site-packages/vhamster_models_v1.2.0`.

If that default location is not writable, install to your own directory instead:

```bash
vhamster-install-models -o /path/to/my_vhamster_models
```

Reinstall if needed:

```bash
vhamster-install-models --force
```

The installer places the fold directories and calibration parameters at:
- `<install_root>/fold_0/` … `<install_root>/fold_4/`
- `<install_root>/length_aware_vector_scaling_anchors_5.json`

If your models are stored elsewhere, pass the path explicitly when running inference:

```bash
vhamster \
  --fasta input.fasta \
  --output results/ \
  --ensemble-dir /path/to/vhamster_models_v1.2.0 \
  --genomad-db /path/to/genomad_db
```

Runtime logs are written to `<output>/<prefix>.log` and also shown in the
terminal.

## Quick-start example

A test genome (accession NC_110914.1) is included in `test_data/`. After
installing `vhamster`, the model bundle, and a geNomad database, run from the
repository root:

```bash
vhamster \
  --fasta test_data/escherichia_phage.fasta \
  --output results/test_run \
  --prefix escherichia_phage \
  --genomad-db /path/to/genomad_db
```

If your model files are stored elsewhere:

```bash
vhamster \
  --fasta test_data/escherichia_phage.fasta \
  --output results/test_run \
  --prefix escherichia_phage \
  --ensemble-dir /path/to/vhamster_models_v1.2.0 \
  --genomad-db /path/to/genomad_db
```

This writes two files:

| File | Contents |
|------|----------|
| `results/test_run/escherichia_phage.chunks.tsv` | Per-chunk predictions |
| `results/test_run/escherichia_phage.genomes.tsv` | Genome-level consensus |

The `predicted_host` column in the genome file should read **Prokaryote** for
this *Escherichia* phage.

---

## Run

Minimal example (models at their default installed location):

```bash
vhamster \
  --fasta /path/to/input.fasta \
  --output /path/to/results_dir \
  --genomad-db /path/to/genomad_db
```

Add a custom output prefix:

```bash
vhamster \
  --fasta /path/to/input.fasta \
  --output /path/to/results_dir \
  --genomad-db /path/to/genomad_db \
  --prefix sampleA
```

Use a custom ensemble directory:

```bash
vhamster \
  --fasta /path/to/input.fasta \
  --output /path/to/results_dir \
  --genomad-db /path/to/genomad_db \
  --ensemble-dir /path/to/vhamster_models_v1.2.0
```

Benchmark using a single fold (for example, only `fold_3`):

```bash
vhamster \
  --fasta /path/to/input.fasta \
  --output /path/to/results_dir \
  --genomad-db /path/to/genomad_db \
  --ensemble-dir /path/to/vhamster_models_v1.2.0 \
  --fold-index 3
```

Use a non-default calibration parameters file:

```bash
vhamster \
  --fasta /path/to/input.fasta \
  --output /path/to/results_dir \
  --genomad-db /path/to/genomad_db \
  --calibration-params /path/to/length_aware_vector_scaling_anchors_5.json
```

## Outputs

For prefix `sampleA`, output files are:
- `/path/to/results_dir/sampleA.chunks.tsv`
- `/path/to/results_dir/sampleA.genomes.tsv`

Chunk file columns include:
- accession, predicted_host, confidence
- class probability columns
- prokaryote_score, eukaryote_score

Genome file columns include:
- genome, predicted_host, confidence
- mean-pooled class probability columns
- prokaryote_score, eukaryote_score

