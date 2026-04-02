# V-HAMSTeR

**V**irus **H**ost **A**ssignment **M**odel using **S**equence **T**ransformers and **R**eading-frames

V-HAMSTeR uses a genomic language model to predict the host of a virus as one of: animal, plant, fungi, protist, or prokaryote.

It is designed for viral sequences up to 10 kbp. Longer sequences are split
into 10 kbp chunks, each chunk is scored independently, and the chunk predictions are
mean-pooled to produce a genome-level consensus prediction.

It runs a 5-fold ensemble and writes:
- chunk-level predictions (10kbp)
- genome-level consensus predictions (mean-pooled over chunks)

## Dependencies

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

## Installation

### pip (editable install from repo)

This is the easiest way to install V-HAMSTeR right now.

Clone the repository and install from the repository root:

```bash
git clone https://code.jgi.doe.gov/SusieGrigson/vhamster.git
cd vhamster
pip install -e .
```

This installs all dependencies listed in `pyproject.toml` and registers the
`vhamster` and `vhamster-install-models` shell commands.

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

An `environment.yml` is provided. If you want GPU support, edit the
`pytorch-cuda` version to match your cluster before running:

```bash
conda env create -f environment.yml
conda activate vhamster
```

## Model installation

Model installation is a separate step after installing `vhamster` itself.

Install the pretrained model bundle (v1.0.0) from NERSC:

```bash
vhamster-install-models
```

By default, this installs into an environment-scoped location in the active Python
environment: `site-packages/vhamster_models`.

If that default location is not writable, install to your own directory instead:

```bash
vhamster-install-models -o /path/to/my_vhamster_models
```

Reinstall if needed:

```bash
vhamster-install-models --force
```

The installer places files at:
- `<install_root>/best_params_20260331`
- `<install_root>/joint_temperature.pt`

If your models are stored elsewhere, pass explicit paths when running inference:

```bash
vhamster \
  --fasta input.fasta \
  --output results/ \
  --ensemble-dir /path/to/best_params_20260331 \
  --temperature-file /path/to/joint_temperature.pt
```

Runtime logs are written to `<output>/<prefix>.log` and also shown in the
terminal.

## Quick-start example

A test genome (accession NC_110914.1) is
included in `test_data/`. After installing `vhamster` and the model bundle, run from the repository root
(the model and temperature file at their default locations will be picked up
automatically):

```bash
vhamster \
  --fasta test_data/escherichia_phage.fasta \
  --output results/test_run \
  --prefix escherichia_phage
```

If your model files are stored elsewhere:

```bash
vhamster \
  --fasta test_data/escherichia_phage.fasta \
  --output results/test_run \
  --prefix escherichia_phage \
  --ensemble-dir /path/to/best_params_20260331 \
  --temperature-file /path/to/joint_temperature.pt
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

Minimal example (uses the default model directory in the active Python environment):

```bash
vhamster \
  --fasta /path/to/input.fasta \
  --output /path/to/results_dir
```

Add a custom output prefix:

```bash
vhamster \
  --fasta /path/to/input.fasta \
  --output /path/to/results_dir \
  --prefix sampleA
```

Use a custom ensemble directory:

```bash
vhamster \
  --fasta /path/to/input.fasta \
  --output /path/to/results_dir \
  --ensemble-dir /path/to/model_root_with_fold_dirs
```

Benchmark using a single fold model (for example, only `fold_3`):

```bash
vhamster \
  --fasta /path/to/input.fasta \
  --output /path/to/results_dir \
  --ensemble-dir /path/to/model_root_with_fold_dirs \
  --fold-index 3
```

Use a non-default temperature file:

```bash
vhamster \
  --fasta /path/to/input.fasta \
  --output /path/to/results_dir \
  --temperature-file /path/to/joint_temperature.pt
```

You can also pass temperature directly as a scalar (useful for benchmarking):

```bash
vhamster \
  --fasta /path/to/input.fasta \
  --output /path/to/results_dir \
  --temperature-file 1.0
```

## Outputs

For prefix `sampleA`, output files are:
- `/path/to/results_dir/sampleA.chunks.tsv`
- `/path/to/results_dir/sampleA.genomes.tsv`

Chunk file columns include:
- accession, predicted_host, confidence
- class probability columns
- prokaryote_score, eukaryote_score
- feature columns

Genome file columns include:
- genome, predicted_host, confidence
- mean-pooled class probability columns
- prokaryote_score, eukaryote_score

