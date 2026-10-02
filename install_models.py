#!/usr/bin/env python3
"""
VHAMSTeR Model Installation Script

Downloads VHAMSTeR models from HuggingFace and the geNomad marker database
from Zenodo.
"""

import gzip
import hashlib
import json
import os
import shutil
import subprocess as sp
import sys
import sysconfig
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download, get_token, login, model_info
from loguru import logger
import click


HF_REPO_ID = "DOEJGI/vhamster-models"

# Only the MMseqs2 database and metadata are needed — vhamster does not use
# the HMM or MSA files from the full geNomad bundle.
GENOMAD_ZENODO_BASE = "https://zenodo.org/records/14886553/files"
GENOMAD_FILES = [
    {
        "filename": "genomad_db_v1.9.tar.gz",
        "md5": "67244b528bb8bed464d1ca147136d33e",
        "size_mb": 842,
    },
    {
        "filename": "genomad_metadata_v1.9.tsv.gz",
        "md5": "d4fa26b7a77017543bd80fb6bb4ee9d0",
        "size_mb": 7,
    },
]

REQUIRED_MODEL_FILES = ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]
REQUIRED_ROOT_FILES = ["proportional_vector_scaling_scalar_nll_notclassbalanced_posthoc_fungi_nolength.json"]
DEFAULT_MODEL_DIRNAME = "vhamster_models_v1.4.0"


def configure_logging(debug: bool = False):
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if debug else "INFO")


def get_default_model_dir() -> str:
    purelib = sysconfig.get_path("purelib")
    if purelib is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logger.warning("Could not resolve environment site-packages; falling back to script directory")
        return os.path.join(script_dir, DEFAULT_MODEL_DIRNAME)
    env_model_dir = os.path.join(purelib, DEFAULT_MODEL_DIRNAME)
    logger.info(f"Using environment-scoped default model location: {env_model_dir}")
    return env_model_dir


def check_model_installation(model_dir: str) -> bool:
    for file_name in REQUIRED_ROOT_FILES:
        file_path = os.path.join(model_dir, file_name)
        if not os.path.isfile(file_path):
            logger.warning(f"Required file missing: {file_path}")
            return False
    for fold_name in REQUIRED_MODEL_FILES:
        fold_path = os.path.join(model_dir, fold_name)
        if not os.path.isdir(fold_path):
            logger.warning(f"Fold directory missing: {fold_path}")
            return False
    logger.info("All required model files are present")
    return True


def check_genomad_installation(genomad_db_dir: str) -> bool:
    return os.path.isfile(os.path.join(genomad_db_dir, "genomad_marker_metadata.tsv")) and \
           os.path.isfile(os.path.join(genomad_db_dir, "genomad_db"))


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_file(url: str, dest: str, size_mb: int):
    """Download a file with a simple progress indicator."""
    def reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = min(count * block_size * 100 // total_size, 100)
            sys.stdout.write(f"\r  {pct}%")
            sys.stdout.flush()

    try:
        logger.info(f"Downloading {os.path.basename(dest)} (~{size_mb} MB)")
        urllib.request.urlretrieve(url, dest, reporthook)
        sys.stdout.write("\n")
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(f"Could not download {url}\n{e}")


def get_models_huggingface(model_dir: str):
    abs_path = os.path.abspath(model_dir)
    logger.info(f"Downloading VHAMSTeR models from HuggingFace ({HF_REPO_ID})")
    try:
        snapshot_download(repo_id=HF_REPO_ID, repo_type="model", local_dir=abs_path, revision='v1.4.0')
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(f"Coul d not download models from HuggingFace.\n{e}")
    logger.info("Model download complete.")


_NTV3_LICENSE_SUMMARY = """
╔══════════════════════════════════════════════════════════════════════════════╗
║          InstaDeep Open Model Licence — key terms (NTv3_650M_pre)          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  • NON-COMMERCIAL USE ONLY.                                                 ║
║  • You may use, reproduce, and share the model and its outputs solely for   ║
║    non-commercial purposes.                                                 ║
║  • You may NOT sublicense, resell, or distribute copies of the model.       ║
║  • You may NOT use it to train or improve commercial derivative models.     ║
║  • Full licence text: NTV3_MODEL-LICENSE.md in this repository.             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def ensure_base_model_auth(model_id: str) -> bool:
    """Ensure the user is authenticated and has access to a gated HuggingFace model.

    Displays the NTv3 licence summary, asks for agreement, and guides the user
    through HuggingFace login if they are not already authenticated.  Returns
    True when the model is accessible, False otherwise.
    """
    # Check whether the model is actually gated before asking anything.
    try:
        info = model_info(model_id)
        is_gated = bool(info.gated)
    except Exception:
        is_gated = True  # assume gated if we cannot check

    if not is_gated:
        return True

    # Show licence summary and request agreement.
    logger.info(_NTV3_LICENSE_SUMMARY)
    logger.info(
        f"Downloading '{model_id}' requires agreeing to InstaDeepAI's terms and\n"
        "a free HuggingFace account.  If you do not already have one, you can\n"
        "create one for free at https://huggingface.co/join\n"
        f"and then request access at https://huggingface.co/{model_id}\n"
    )

    try:
        answer = input("Do you agree to the NTv3 licence terms above? [yes/no]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""

    if answer not in ("yes", "y"):
        logger.warning("Licence not accepted — skipping base model download.")
        return False

    # Check whether a HuggingFace token is already cached.
    token = get_token()
    if token:
        logger.info("HuggingFace credentials already present.")
        return True

    # No token — offer browser-based login (opens huggingface.co in the browser).
    logger.info(
        "No HuggingFace login detected.  Opening your browser to log in.\n"
        "If you prefer the command line, cancel and run:  huggingface-cli login"
    )
    try:
        login()  # opens a browser tab; falls back to token prompt if no browser
    except Exception as e:
        logger.warning(f"Browser login failed ({e}). Run 'huggingface-cli login' manually, then re-run vhamster-install-models")
        return False

    token = get_token()
    if not token:
        logger.warning("Login did not complete. Re-run vhamster-install-models after authenticating.")
        return False

    logger.info("HuggingFace login successful.")
    return True


def cache_base_models(model_dir: str, force: bool = False) -> bool:
    """Download and cache the base transformer model(s) required by each fold.

    Reads each fold's config.json to discover the HuggingFace model ID, then
    stores the full snapshot under <model_dir>/base_models/<org>--<model>/.
    Returns True if all base models are cached successfully.
    """
    base_models_dir = os.path.join(model_dir, "base_models")

    base_model_ids: set = set()
    for fold_name in REQUIRED_MODEL_FILES:
        config_path = os.path.join(model_dir, fold_name, "config.json")
        if not os.path.isfile(config_path):
            logger.warning(f"config.json not found for {fold_name} — skipping")
            continue
        with open(config_path) as f:
            fold_config = json.load(f)
        model_id = fold_config.get("model")
        if model_id:
            base_model_ids.add(model_id)

    if not base_model_ids:
        logger.warning("No base model IDs found in fold configs. Skipping base model cache.")
        return False

    os.makedirs(base_models_dir, exist_ok=True)
    all_ok = True
    for model_id in sorted(base_model_ids):
        local_name = model_id.replace("/", "--")
        local_dir = os.path.join(base_models_dir, local_name)
        already_cached = os.path.isdir(local_dir) and os.listdir(local_dir)
        if already_cached and not force:
            logger.info(f"Base model already cached: {model_id}")
            continue
        if not ensure_base_model_auth(model_id):
            all_ok = False
            continue
        logger.info(f"Downloading base model: {model_id}")
        try:
            snapshot_download(repo_id=model_id, repo_type="model", local_dir=local_dir)
            logger.info(f"Base model cached: {local_dir}")
        except Exception as e:
            logger.warning(
                f"Could not download base model '{model_id}': {e}\n"
                "  If this model is gated on HuggingFace you must:\n"
                f"    1. Request access at https://huggingface.co/{model_id}\n"
                "    2. Run: huggingface-cli login\n"
                "    3. Re-run: vhamster-install-models\n"
                "  Users will still be prompted for HuggingFace credentials at runtime until "
                "the base model is cached here."
            )
            all_ok = False
    return all_ok


def download_genomad_from_zenodo(genomad_db_dir: str):
    """Download and extract the geNomad marker database from Zenodo."""
    os.makedirs(genomad_db_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        for entry in GENOMAD_FILES:
            filename = entry["filename"]
            url = f"{GENOMAD_ZENODO_BASE}/{filename}?download=1"
            tmp_path = os.path.join(tmp, filename)

            _download_file(url, tmp_path, entry["size_mb"])

            logger.info(f"Verifying checksum for {filename}")
            actual = _md5(tmp_path)
            if actual != entry["md5"]:
                sys.exit(
                    f"MD5 mismatch for {filename}.\n"
                    f"  expected: {entry['md5']}\n"
                    f"  got:      {actual}"
                )

            if filename.endswith(".tar.gz"):
                logger.info(f"Extracting {filename}")
                with tarfile.open(tmp_path, "r:gz") as tar:
                    # Extract to a temp subdir so we can inspect the structure
                    extract_dir = os.path.join(tmp, "extracted")
                    os.makedirs(extract_dir, exist_ok=True)
                    for member in tar.getmembers():
                        if member.name.startswith("/") or ".." in member.name:
                            continue
                        tar.extract(member, path=extract_dir)

                # If everything landed in a single top-level directory, use its contents
                extracted_items = os.listdir(extract_dir)
                if len(extracted_items) == 1 and os.path.isdir(os.path.join(extract_dir, extracted_items[0])):
                    extract_dir = os.path.join(extract_dir, extracted_items[0])

                for item in os.listdir(extract_dir):
                    src = os.path.join(extract_dir, item)
                    dst = os.path.join(genomad_db_dir, item)
                    if os.path.exists(dst):
                        if os.path.isdir(dst):
                            shutil.rmtree(dst)
                        else:
                            os.remove(dst)
                    shutil.move(src, dst)

            elif filename.endswith(".tsv.gz"):
                out_name = filename.replace("_v1.9", "").replace(".gz", "")
                # metadata file should be named genomad_marker_metadata.tsv
                if "metadata" in filename:
                    out_name = "genomad_marker_metadata.tsv"
                out_path = os.path.join(genomad_db_dir, out_name)
                logger.info(f"Decompressing {filename} -> {os.path.basename(out_path)}")
                with gzip.open(tmp_path, "rb") as f_in, open(out_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

    logger.info(f"geNomad database installed at: {genomad_db_dir}")


def instantiate_install(model_dir: str, force: bool = False):
    abs_path = os.path.abspath(model_dir)
    os.makedirs(abs_path, exist_ok=True)
    logger.info(f"Model installation directory: {abs_path}")

    if check_model_installation(model_dir) and not force:
        logger.info(f"All VHAMSTeR models already present in: {abs_path}")
    else:
        if force:
            logger.info("Force reinstall requested.")
        get_models_huggingface(model_dir)


def install_genomad(model_dir: str, force: bool = False):
    genomad_db_dir = os.path.join(model_dir, "genomad_db")

    if check_genomad_installation(genomad_db_dir) and not force:
        logger.info(f"geNomad database already present at: {genomad_db_dir}")
        return

    download_genomad_from_zenodo(genomad_db_dir)


@click.command()
@click.option("-o", "--outdir", type=click.Path(path_type=str), default=None,
              help="Directory to install models into (default: environment site-packages).")
@click.option("-f", "--force", is_flag=True, default=False,
              help="Force reinstallation even if models already exist.")
@click.option("--skip-base-model", is_flag=True, default=False,
              help="Skip downloading the NTv3 base transformer model. "
                   "Use only if you have already cached it or intend to authenticate later.")
@click.option("--debug", is_flag=True, default=False,
              help="Enable verbose debug logging.")
def main(outdir, force, debug, skip_base_model):
    """Download and install VHAMSTeR models from HuggingFace and the geNomad
    marker database from Zenodo."""
    configure_logging(debug)

    model_dir = os.path.abspath(outdir) if outdir else get_default_model_dir()
    logger.info(f"Model installation directory: {model_dir}")

    instantiate_install(model_dir, force)
    install_genomad(model_dir, force)

    if skip_base_model:
        logger.info("Skipping base transformer model download (--skip-base-model).")
    else:
        logger.info("Downloading NTv3 base transformer model for offline use...")
        cache_base_models(model_dir, force)

    logger.info("\n" + "=" * 60)
    logger.info("INSTALLATION SUMMARY")
    logger.info("=" * 60)

    # mmseqs2 availability check
    import shutil as _shutil
    if _shutil.which("mmseqs") is not None:
        logger.info("✓ mmseqs2 found in PATH")
    else:
        logger.warning(
            "✗ mmseqs2 not found in PATH — required at runtime for geNomad marker search.\n"
            "  Install via conda:  conda install -c bioconda mmseqs2\n"
            "  Or via mamba:       mamba install -c bioconda mmseqs2"
        )

    fold_dirs = [d for d in os.listdir(model_dir) if d.startswith("fold_")]
    if len(fold_dirs) == 5:
        logger.info(f"✓ Fold models installed: {len(fold_dirs)}/5")
    else:
        logger.warning(f"✗ Fold models incomplete. Found: {len(fold_dirs)}/5")

    calibration_file = os.path.join(model_dir, REQUIRED_ROOT_FILES[0])
    if os.path.exists(calibration_file):
        logger.info(f"✓ Calibration file present")
    else:
        logger.warning(f"✗ Calibration file not found: {calibration_file}")

    genomad_db_dir = os.path.join(model_dir, "genomad_db")
    if check_genomad_installation(genomad_db_dir):
        logger.info(f"✓ geNomad database present")
    else:
        logger.warning(f"✗ geNomad database incomplete at: {genomad_db_dir}")

    base_models_dir = os.path.join(model_dir, "base_models")
    if os.path.isdir(base_models_dir) and os.listdir(base_models_dir):
        cached = [d for d in os.listdir(base_models_dir) if os.path.isdir(os.path.join(base_models_dir, d))]
        logger.info(f"✓ Base model(s) cached locally: {', '.join(cached)}")
    else:
        logger.info(
            "  Base transformer model not cached locally.\n"
            "  Re-run vhamster-install-models to download it, or authenticate first with:\n"
            "    huggingface-cli login"
        )

    logger.info(f"\nTo run vhamster:")
    logger.info(f"  vhamster --fasta <input.fasta> --output <output_dir> --ensemble-dir {model_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
