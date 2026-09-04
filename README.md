# VHAMSTeR
![](vhamsterlogo.png)


**V**irus **H**ost **A**ssignment **M**odel using **S**equence **T**ransformers and **R**eading-frames

VHAMSTeR uses a genomic language model to predict the host of a virus as one of: animal, plant, fungi, protist, or prokaryote.

It is designed for viral sequences up to 10 kbp. Longer sequences are split
into 10 kbp chunks (with a `_chunk<start>_<end>` suffix appended to the accession). 
Each chunk is scored independently, and the chunk predictions are mean-pooled to 
produce a genome-level consensus prediction.

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

### External tools

- `mmseqs2` — used for geNomad marker search. If you use the conda environment (see below), it is included automatically. Otherwise install it via conda/mamba:

```bash
conda install -c bioconda mmseqs2
```

Alternatively, precompiled static binaries are available from the [MMseqs2 GitHub releases](https://github.com/soedinglab/MMseqs2/releases).

### geNomad database

A geNomad database is required at runtime. `vhamster-install-models` downloads it automatically from [Zenodo](https://zenodo.org/records/14886553). If you have an existing geNomad database you can point to it with `--genomad-db` instead.

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

> **Note:** `mmseqs2` is not a Python package and must be installed separately (see above).

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
pip install -e .
```

## Model installation

Model installation is a separate step after installing `vhamster` itself.

This downloads the pretrained model weights from [HuggingFace](https://huggingface.co/DOEJGI/vhamster-models) and the geNomad marker database from [Zenodo](https://zenodo.org/records/14886553):

```bash
vhamster-install-models
```

By default, everything is installed into an environment-scoped location in the active Python
environment: `site-packages/vhamster_models_v1.3.0`.

If that default location is not writable, install to your own directory instead:

```bash
vhamster-install-models -o /path/to/my_vhamster_models
```

Reinstall if needed:

```bash
vhamster-install-models -f
```

The installer places files at:
- `<install_root>/fold_0/` … `<install_root>/fold_4/` — ensemble model weights
- `<install_root>/proportional_vector_scaling_scalar_nll_notclassbalanced_posthoc_fungi_nolength.json` — calibration parameters
- `<install_root>/genomad_db/` — geNomad marker database

Once installed, vhamster will find the geNomad database automatically. If you have an existing geNomad database elsewhere, you can point to it with `--genomad-db`:

```bash
vhamster \
  --fasta input.fasta \
  --output results/ \
  --genomad-db /path/to/genomad_db
```

If your models are stored in a non-default location, pass the path with `--ensemble-dir`:

```bash
vhamster \
  --fasta input.fasta \
  --output results/ \
  --ensemble-dir /path/to/vhamster_models_v1.3.0
```

Runtime logs are written to `<output>/<prefix>.log` and also shown in the
terminal.

## Quick-start example

A test genome (accession NC_110914.1) is included in `test_data/`. After
installing `vhamster` and running `vhamster-install-models`, run from the
repository root:

```bash
vhamster \
  --fasta test_data/escherichia_phage.fasta \
  --output results/test_run \
  --prefix escherichia_phage
```

If your model files are stored in a non-default location:

```bash
vhamster \
  --fasta test_data/escherichia_phage.fasta \
  --output results/test_run \
  --prefix escherichia_phage \
  --ensemble-dir /path/to/vhamster_models_v1.3.0
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
  --ensemble-dir /path/to/vhamster_models_v1.3.0
```

Run using a single fold (for example, only `fold_3`):

```bash
vhamster \
  --fasta /path/to/input.fasta \
  --output /path/to/results_dir \
  --fold-index 3
```

Use a non-default calibration parameters file:

```bash
vhamster \
  --fasta /path/to/input.fasta \
  --output /path/to/results_dir \
  --calibration-params /path/to/proportional_vector_scaling_scalar_nll_notclassbalanced_posthoc_fungi_nolength.json
```

## Two-stage pipeline (HPC)

For large datasets, the CPU-intensive feature extraction (PyRodigal + MMseqs2) and the GPU-intensive GLM inference can be run as two separate jobs. This lets you pipeline batches: one batch's features are computed on a CPU node while the previous batch's GLM inference runs on a GPU node.

**Stage 1 — feature extraction (CPU node, no GPU needed):**

```bash
vhamster-features \
  --fasta batch_001.fasta \
  --output features/batch_001 \
  --prefix batch_001
```

This writes:
- `features/batch_001/batch_001.arch_features.tsv` — per-chunk architectural features
- `features/batch_001/batch_001.genomad_hits.json` — geNomad marker hits
- `features/batch_001/batch_001.gene_predictions.tsv` — per-gene annotations

**Stage 2 — GLM inference (GPU node):**

```bash
vhamster \
  --fasta batch_001.fasta \
  --output results/batch_001 \
  --prefix batch_001 \
  --precomputed-features features/batch_001
```

The `--chunk-size` and `--overlap` values must match between the two stages (defaults are the same, so no flags needed if you use defaults for both).


## Outputs

For prefix `sampleA`, the unconditional output files are:
- `/path/to/results_dir/sampleA.chunks.tsv` — per-chunk predictions
- `/path/to/results_dir/sampleA.genomes.tsv` — genome-level consensus (mean-pooled over chunks)
- `/path/to/results_dir/sampleA.folds.tsv` — per-fold predictions and GLM gate weights for all 5 ensemble members

If the `--verbose` flag is passed, an additional file is generated:
- `/path/to/results_dir/sampleA.verbose.tsv` — detailed per-fold uncalibrated stream probabilities, uncalibrated ensemble probabilities, and raw feature arrays.

### Chunk Naming Convention
Sequences longer than the specified chunk size (default 10 kbp) are split into smaller fragments. The `accession` column for these fragments will include a `_chunk<start>_<end>` suffix (e.g., `NC_007026.1_chunk0_10000`). You can use this suffix or the `sampleA.genomes.tsv` file to join chunk-level predictions back to your original input sequences.

### File Schemas

**Chunk file columns include:**
- `accession`, `predicted_host`, `confidence`
- calibrated class probability columns
- `prokaryote_score`, `eukaryote_score`

**Genome file columns include:**
- `genome`, `predicted_host`, `confidence`
- calibrated class probability columns
- `prokaryote_score`, `eukaryote_score`

**Folds file columns include:**
- `accession`, `fold`, `predicted_host`, `confidence`, `glm_gate_weight`
- calibrated class probability columns

**Verbose file columns include:**
- `accession`, `fold`, `n_genes`, `predicted_host`, `confidence`, `glm_gate_weight`
- calibrated class probability columns
- `xgb_<class>` (pure, uncalibrated XGBoost probabilities)
- `glm_<class>` (pure, uncalibrated GLM probabilities)
- `uncalibrated_ensemble_<class>` (the exact mathematical output of the dynamic gate before temperature scaling)
- All raw architectural and marker features

## Performance & Batching Tips

**1. Batch your inputs into a single FASTA file**
The geNomad marker search relies on MMseqs2, which has a high fixed startup cost (often 1–2 minutes) to load the marker database into memory. This cost is incurred every time `vhamster` runs. 

To avoid paying this startup penalty multiple times, **do not run vhamster in a bash loop over individual files**. Instead, concatenate your sequences into a single multi-FASTA file and run `vhamster` once:

```
# Inefficient (Loads DB 3 times)
vhamster --fasta seq1.fasta --output out/ --prefix seq1
vhamster --fasta seq2.fasta --output out/ --prefix seq2
vhamster --fasta seq3.fasta --output out/ --prefix seq3

# Highly Efficient (Loads DB once)
cat seq1.fasta seq2.fasta seq3.fasta > all_seqs.fasta
vhamster --fasta all_seqs.fasta --output out/ --prefix all_seqs
```
## Citation 
Preprint coming soon! 

## License Agreement
Lawrence Berkeley National Laboratory 
NON-COMMERCIAL USE ONLY LICENSE
 
V-HAMSTeR Copyright (c) 2026, The Regents of the University of California, through Lawrence Berkeley National Laboratory (“Berkeley Lab”) subject to receipt of any required approvals from the U.S. Dept. of Energy.  All rights reserved.
 
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
(1) Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
(2) Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
(3) Neither the name of the University of California, Berkeley Lab, U.S. Dept. of Energy nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission of Berkeley Lab.
 
(4) Use of the software, in source or binary form is for NON-COMMERCIAL USE purposes ONLY. The software is not available for commercial use. If you have any questions regarding this software, please contact Berkeley Lab at IPO@lbl.gov.

(5) User agrees to indemnify, defend, and hold harmless Berkeley Lab, the U.S. Government, the software developers, the software sponsors, and their agents, officers, and employees, against any and all claims, suits, losses, damage, costs, fees, and expenses arising out of or in connection with this Agreement.  User agrees to pay all costs incurred by Berkeley Lab in enforcing this provision, including reasonable attorney fees.

(6) In the event User creates any bug fixes, patches, upgrades, updates, modifications, derivative works or enhancements to the source code or binary code of the software ("Enhancements") User hereby grants Berkeley Lab and the U.S. Government a paid-up, non-exclusive, irrevocable, worldwide license in the Enhancements to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Copyright Notice 
V-HAMSTeR Copyright (c) 2026, The Regents of the University of California, through Lawrence Berkeley National Laboratory (“Berkeley Lab”) subject to receipt of any required approvals from the U.S. Dept. of Energy.  All rights reserved.
If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.
NOTICE: This Software was developed under Contract No. DE-AC02-05CH11231 with the Department of Energy (“DOE”). During the period of commercialization or such other time period specified by DOE, the U.S. Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable, worldwide license in the Software to reproduce, prepare derivative works, and perform publicly and display publicly, by or on behalf of the U.S. Government. Subsequent to that period, the U.S. Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable, worldwide license in the Software to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so. The specific term of the license can be identified by inquiry made to Lawrence Berkeley National Laboratory or DOE. NEITHER THE UNITED STATES NOR THE UNITED STATES DEPARTMENT OF ENERGY, NOR ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY DATA, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.


