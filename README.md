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


## License + Copyright

*** License Agreement ***
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

*** Copyright Notice ***
V-HAMSTeR Copyright (c) 2026, The Regents of the University of California, through Lawrence Berkeley National Laboratory (“Berkeley Lab”) subject to receipt of any required approvals from the U.S. Dept. of Energy.  All rights reserved.
If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.
NOTICE: This Software was developed under Contract No. DE-AC02-05CH11231 with the Department of Energy (“DOE”). During the period of commercialization or such other time period specified by DOE, the U.S. Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable, worldwide license in the Software to reproduce, prepare derivative works, and perform publicly and display publicly, by or on behalf of the U.S. Government. Subsequent to that period, the U.S. Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable, worldwide license in the Software to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so. The specific term of the license can be identified by inquiry made to Lawrence Berkeley National Laboratory or DOE. NEITHER THE UNITED STATES NOR THE UNITED STATES DEPARTMENT OF ENERGY, NOR ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY DATA, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.


