# V-HAMSTeR

**V**irus **H**ost **A**ssignment **M**odel using **S**equence **T**ransformers and **R**eading-frame

V-HAMSTeR uses a genomic language model to predict the host of a virus as one of: animal, plant, fungi, protist, or prokaryote.

It runs a 5-fold ensemble and writes:
- chunk-level predictions (10kbp)
- genome-level consensus predictions (mean-pooled over chunks)

**Default paths** (used automatically when no flags are given):
- Model ensemble: `model/best_params_20260331`
- Temperature file: `model/joint_temperature.pt`

> **Note:** Until the model is available for automatic download (e.g. from Zenodo), you need to
> supply the model directory manually if it is not at the default location:
>
> ```bash
> vhamster \
>   --fasta input.fasta \
>   --output results/ \
>   --ensemble-dir /path/to/best_params_20260331 \
>   --temperature-file /path/to/joint_temperature.pt
> ```

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
- `pyrodigal-rv`
- `tqdm`
- `scikit-learn`

### PyTorch

PyTorch must be installed with the right CUDA version for your system **before**
running `pip install -e .`.  Visit https://pytorch.org/get-started/locally/ to
get the correct command, e.g.:

```bash
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Conda environment (recommended for HPC)

An `environment.yml` is provided.  Edit the `pytorch-cuda` version to match
your cluster before running:

```bash
conda env create -f environment.yml
conda activate vhamster
```

### pip (editable install from repo)

From the repository root:

```bash
pip install -e .
```

This installs all dependencies listed in `pyproject.toml` and registers the
`vhamster` shell command.  Install PyTorch separately first (see above) if you
need GPU/CUDA support, as the default `torch` wheel may be CPU-only.

Runtime logs are written to `<output>/<prefix>.log` and also shown in the
terminal.

## Quick-start example

A test genome (accession NC_110914.1) is
included in `test_data/`.  After installing, run from the repository root
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

Minimal example (uses the default model directory in this repo):

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

