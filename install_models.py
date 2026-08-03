#!/usr/bin/env python3
"""
VHAMSTeR Model Installation Script

Downloads VHAMSTeR models from HuggingFace and the geNomad marker database
from Zenodo.
"""

import gzip
import hashlib
import os
import shutil
import subprocess as sp
import sys
import sysconfig
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download
from loguru import logger
import click


HF_REPO_ID = "DOEJGI/vhamster-models-v1.2.0"

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
REQUIRED_ROOT_FILES = ["length_aware_vector_scaling_anchors_toplabel_5.json"]
DEFAULT_MODEL_DIRNAME = "vhamster_models_v1.2.0"


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
        if not os.path.isfile(os.path.join(model_dir, file_name)):
            logger.warning(f"Required file missing: {file_name}")
            return False
    for fold_name in REQUIRED_MODEL_FILES:
        if not os.path.isdir(os.path.join(model_dir, fold_name)):
            logger.warning(f"Fold directory missing: {fold_name}")
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
        snapshot_download(repo_id=HF_REPO_ID, repo_type="model", local_dir=abs_path)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(f"Could not download models from HuggingFace.\n{e}")
    logger.info("Model download complete.")


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
@click.option("--debug", is_flag=True, default=False,
              help="Enable verbose debug logging.")
def main(outdir, force, debug):
    """Download and install VHAMSTeR models from HuggingFace and the geNomad
    marker database from Zenodo."""
    configure_logging(debug)

    model_dir = os.path.abspath(outdir) if outdir else get_default_model_dir()
    logger.info(f"Model installation directory: {model_dir}")

    instantiate_install(model_dir, force)
    install_genomad(model_dir, force)

    logger.info("\n" + "=" * 60)
    logger.info("INSTALLATION SUMMARY")
    logger.info("=" * 60)

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

    logger.info(f"\nTo run vhamster:")
    logger.info(f"  vhamster --fasta <input.fasta> --output <output_dir> --ensemble-dir {model_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
