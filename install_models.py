#!/usr/bin/env python3
"""
vHAMSTeR Model Installation Script

Downloads and installs the pre-trained vHAMSTeR models from NERSC.
Based on code from https://github.com/gbouras13/pharokka/blob/master/bin/databases.py
and https://github.com/oschwengers/bakta
"""

import hashlib
import os
import re
import shutil
import subprocess as sp
import sys
import sysconfig
import tarfile
from pathlib import Path
import requests
from loguru import logger
import click


# Model configuration for v1.0.1
VERSION_DICTIONARY = {
    "1.0.1": {
        "md5": None,  # Will be calculated or user can provide
        "db_url": "https://portal.nersc.gov/cfs/m342/V-HAMSTeR/vhamster_models_v1.0.1.tar.gz",
        "dir_name": "vhamster_models_v1.0.1",
        "expected_structure": ["best_params_20260331", "length_class_temperatures_continuous_brier.json"],
    }
}

REQUIRED_MODEL_FILES = [
    "fold_0",
    "fold_1", 
    "fold_2",
    "fold_3",
    "fold_4",
]

REQUIRED_ROOT_FILES = [
    "length_class_temperatures_continuous_brier.json",
]

DEFAULT_MODEL_DIRNAME = "vhamster_models"


def configure_logging(debug: bool = False):
    """Configure installer logging with debug output disabled by default."""
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if debug else "INFO")


def get_default_model_dir() -> str:
    """Return the default model directory in the active Python environment."""
    purelib = sysconfig.get_path("purelib")
    if purelib is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logger.warning("Could not resolve environment site-packages; falling back to script directory")
        return os.path.join(script_dir, DEFAULT_MODEL_DIRNAME)

    env_model_dir = os.path.join(purelib, DEFAULT_MODEL_DIRNAME)
    logger.info(f"Using environment-scoped default model location: {env_model_dir}")
    return env_model_dir


def instantiate_install(model_dir: str, force: bool = False, version: str = "1.0.1"):
    """
    Begin model install

    :param model_dir: path to install the models (should be 'model/' directory, not 'model/best_params_*')
    :param force: if True, reinstall models even if they already exist
    :param version: version of models to install
    """
    # Show absolute path for clarity
    abs_path = os.path.abspath(model_dir)
    logger.info(f"Model installation directory (absolute path): {abs_path}")

    instantiate_dir(model_dir)
    downloaded_flag = check_model_installation(model_dir, version)
    
    if downloaded_flag and not force:
        logger.info(f"All vHAMSTeR models have already been downloaded and verified in: {abs_path}")
    else:
        if force and downloaded_flag:
            logger.info("Force reinstall requested. Reinstalling models...")
        else:
            logger.info("Some models are missing.")
        get_models_nersc(model_dir, version)
        logger.info(f"Models successfully downloaded to: {abs_path}")


def instantiate_dir(model_dir: str):
    """
    Create directory to download models

    :param model_dir: path to the model directory
    """
    if os.path.isdir(model_dir) == False:
        logger.info(f"Creating models directory: {model_dir}")
        try:
            os.makedirs(model_dir, exist_ok=True)
        except PermissionError as e:
            logger.error(f"Failed to create directory {model_dir}: {e}")
            sys.exit(
                "Cannot write to the default model directory. "
                "Re-run with --outdir /path/to/writable/location to install models elsewhere."
            )
        except Exception as e:
            logger.error(f"Failed to create directory {model_dir}: {e}")
            sys.exit(1)


def check_model_installation(model_dir: str, version: str = "1.0.1") -> bool:
    """
    Check that all required models have been installed

    :param model_dir: path to the models directory
    :param version: version of models to check
    :return: True if all models are present, False otherwise
    """
    downloaded_flag = True
    
    # Check for root-level files
    for file_name in REQUIRED_ROOT_FILES:
        path = os.path.join(model_dir, file_name)
        if not os.path.isfile(path):
            logger.warning(f"Required file missing: {file_name}")
            downloaded_flag = False
            break
    
    # Check for ensemble directory
    ensemble_dir = os.path.join(model_dir, "best_params_20260331")
    if not os.path.isdir(ensemble_dir):
        logger.warning("Ensemble directory 'best_params_20260331' not found")
        downloaded_flag = False
        return downloaded_flag
    
    # Check for fold directories
    for fold_name in REQUIRED_MODEL_FILES:
        fold_path = os.path.join(ensemble_dir, fold_name)
        if not os.path.isdir(fold_path):
            logger.warning(f"Fold directory missing: {fold_name}")
            downloaded_flag = False
            break
    
    if downloaded_flag:
        logger.info("All required model files are present")
    
    return downloaded_flag


def download(db_url: str, tarball_path: Path):
    """
    Download tarball from URL with progress tracking
    Adapted from bakta db.py
    """
    try:
        with requests.get(db_url, stream=True, timeout=60) as resp:
            if resp.status_code >= 400:
                msg = (
                    f"HTTP {resp.status_code} when downloading model bundle from {db_url}. "
                    "This usually means the file is private or access is restricted."
                )
                logger.error(msg)
                sys.exit(msg)

            content_type = (resp.headers.get("content-type") or "").lower()
            if "text/html" in content_type or "text/plain" in content_type:
                msg = (
                    f"URL returned content-type '{content_type}' instead of a tar.gz archive: {db_url}. "
                    "Check the URL/version or NERSC access permissions."
                )
                logger.error(msg)
                sys.exit(msg)

            with tarball_path.open("wb") as fh_out:
                total_length = resp.headers.get("content-length")
                if total_length is not None:  # content length header is set
                    total_length = int(total_length)
                    total_length_mb = total_length / (1024 * 1024)
                else:
                    total_length_mb = 0

                downloaded = 0
                logger.info(f"Downloading file, total size: {total_length_mb:.2f} MB")

                for data in resp.iter_content(chunk_size=1024 * 1024):
                    fh_out.write(data)
                    downloaded += len(data)

                    if total_length is not None:
                        percent = (downloaded / total_length) * 100
                        downloaded_mb = downloaded / (1024 * 1024)
                        sys.stdout.write(f"\rProgress: {percent:.1f}% ({downloaded_mb:.1f}/{total_length_mb:.1f} MB)")
                        sys.stdout.flush()

                if total_length is not None:
                    sys.stdout.write("\n")  # New line after progress tracking is complete
                    sys.stdout.flush()
                
    except IOError as e:
        logger.error(f"Could not download file from NERSC! url={db_url}, path={tarball_path}")
        logger.error(f"Error: {e}")
        sys.exit(
            f"Please try again or manually download from {db_url}"
        )
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(f"Download error: {e}")


def calc_md5_sum(tarball_path: Path, buffer_size: int = 1024 * 1024) -> str:
    """
    Calculate the MD5 checksum for a tarball
    """
    md5 = hashlib.md5()
    with tarball_path.open("rb") as fh:
        data = fh.read(buffer_size)
        while data:
            md5.update(data)
            data = fh.read(buffer_size)
    return md5.hexdigest()


def remove_directory(dir_path: str):
    """
    Remove directory if it exists
    """
    if os.path.exists(dir_path):
        logger.info(f"Removing directory: {dir_path}")
        shutil.rmtree(dir_path)


def check_paths_permissions(tarball_path: Path, output_path: str):
    """
    Check if paths exist and have proper permissions.
    """
    results = {}
    # Check tarball
    results["tarball_exists"] = os.path.exists(tarball_path)
    results["tarball_readable"] = os.access(tarball_path, os.R_OK) if results["tarball_exists"] else False
    results["tarball_size"] = os.path.getsize(tarball_path) if results["tarball_exists"] else 0
    
    # Check output directory
    results["output_exists"] = os.path.exists(output_path)
    results["output_writable"] = os.access(output_path, os.W_OK) if results["output_exists"] else False
    
    return results


def inspect_tarball(tarball_path: Path):
    """
    Inspect the structure of the tarball and return details.
    """
    try:
        with tarfile.open(tarball_path, 'r:gz') as tar:
            members = tar.getmembers()
            file_count = len(members)
            top_level_dirs = {member.name.split('/')[0] for member in members if '/' in member.name}
            sample_files = [member.name for member in members[:10]]
            return {
                "status": "success",
                "file_count": file_count,
                "top_level_dirs": list(top_level_dirs),
                "sample_files": sample_files
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def untar(tarball_path: Path, output_path: str, version: str = "1.0.1"):
    """
    Extract tarball and organize files into the expected directory structure
    """
    try:
        # Check permissions and existence first
        path_checks = check_paths_permissions(tarball_path, output_path)
        logger.info(f"Path checks: {path_checks}")
        
        if not path_checks["tarball_exists"]:
            logger.error(f"Tarball does not exist: {tarball_path}")
            sys.exit(f"Tarball not found at {tarball_path}")
            
        if not path_checks["tarball_readable"]:
            logger.error(f"Tarball is not readable: {tarball_path}")
            sys.exit(f"Cannot read tarball at {tarball_path}. Check permissions.")
            
        if not path_checks["output_exists"] or not path_checks["output_writable"]:
            logger.error(f"Output directory does not exist or is not writable: {output_path}")
            sys.exit(
                f"Cannot write to output directory: {output_path}. "
                "Check permissions or re-run with --outdir /path/to/writable/location."
            )
        
        # Inspect tarball structure
        tarball_info = inspect_tarball(tarball_path)
        logger.info(f"Tarball inspection: {tarball_info}")
        
        # Create temporary extraction directory
        temp_extract_dir = os.path.join(output_path, ".vhamster_extract_temp")
        if os.path.exists(temp_extract_dir):
            remove_directory(temp_extract_dir)
        os.makedirs(temp_extract_dir, exist_ok=True)
        
        # Extract the tarball to temp directory first
        logger.info("Extraction may take several minutes; please be patient!")
        logger.info(f"Extracting tarball to temporary directory: {temp_extract_dir}")
        with tarfile.open(tarball_path, 'r:gz') as tar:
            all_members = tar.getmembers()
            logger.info(f"Found {len(all_members)} items in tarball")
            
            for member in all_members:
                # Security check
                if member.name.startswith('/') or '..' in member.name:
                    logger.warning(f"Skipping potentially insecure path: {member.name}")
                    continue
                
                # Extract the file
                logger.debug(f"Extracting: {member.name}")
                try:
                    tar.extract(member, path=temp_extract_dir)
                except Exception as e:
                    logger.error(f"Failed to extract {member.name}: {str(e)}")
        
        logger.info("Initial extraction completed")
        
        # Organize extracted files
        extracted_items = os.listdir(temp_extract_dir)
        logger.info(f"Extracted items: {extracted_items}")
        
        # Look for the expected extracted model directory for this version
        source_prefix = VERSION_DICTIONARY[version]["dir_name"]
        source_path = os.path.join(temp_extract_dir, source_prefix)
        
        if not os.path.exists(source_path):
            logger.error(f"Expected directory not found after extraction: {source_path}")
            logger.error(f"Available items: {extracted_items}")
            remove_directory(temp_extract_dir)
            sys.exit(f"Tarball structure unexpected. Please verify the download.")
        
        logger.info(f"Found source directory: {source_path}")
        
        # Move best_params_20260331 and the calibration JSON into output_path
        source_ensemble = os.path.join(source_path, "best_params_20260331")
        source_temperature = os.path.join(source_path, "length_class_temperatures_continuous_brier.json")
        
        dest_ensemble = os.path.join(output_path, "best_params_20260331")
        dest_temperature = os.path.join(output_path, "length_class_temperatures_continuous_brier.json")
        
        # Handle best_params_20260331
        if os.path.exists(source_ensemble):
            # Remove existing if force was used
            if os.path.exists(dest_ensemble):
                logger.info(f"Removing existing ensemble directory: {dest_ensemble}")
                remove_directory(dest_ensemble)
            
            logger.info(f"Moving ensemble directory from {source_ensemble} to {dest_ensemble}")
            shutil.move(source_ensemble, dest_ensemble)
        else:
            logger.error(f"Ensemble directory not found: {source_ensemble}")
            remove_directory(temp_extract_dir)
            sys.exit("Ensemble directory 'best_params_20260331' not found in tarball")
        
        # Handle calibration JSON
        if os.path.exists(source_temperature):
            if os.path.exists(dest_temperature):
                logger.info(f"Overwriting existing temperature file: {dest_temperature}")
                os.remove(dest_temperature)
            
            logger.info(f"Moving temperature file from {source_temperature} to {dest_temperature}")
            shutil.move(source_temperature, dest_temperature)
        else:
            logger.warning(f"Temperature file not found: {source_temperature}")
        
        # Clean up temporary directory
        logger.info(f"Cleaning up temporary directory: {temp_extract_dir}")
        remove_directory(temp_extract_dir)
        
        logger.info("File organization completed")
        
    except tarfile.ReadError as e:
        logger.error(f"Failed to read tarfile: {str(e)}")
        logger.error(f"Tarball may be corrupted or not a valid tar.gz file: {tarball_path}")
        if os.path.exists(temp_extract_dir):
            remove_directory(temp_extract_dir)
        sys.exit(f"Invalid tarfile. Please try downloading again.")
        
    except Exception as e:
        import traceback
        logger.error(f"Error extracting tarball: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        if os.path.exists(temp_extract_dir):
            remove_directory(temp_extract_dir)
        sys.exit(f"Extraction error: {e}")


def get_models_nersc(model_dir: str, version: str = "1.0.1"):
    """
    Download vHAMSTeR models from NERSC portal
    
    :param model_dir: directory to install the models
    :param version: version of models to download
    """
    abs_path = os.path.abspath(model_dir)
    logger.info(f"Models will be installed to: {abs_path}")
    
    download_path = os.path.abspath(model_dir)
    logger.info(f"Downloading tarball to: {download_path}")
    
    db_url = VERSION_DICTIONARY[version]["db_url"]
    
    # Extract filename from URL
    tarball = re.split("/", db_url)[-1]
    tarball_path = Path(f"{download_path}/{tarball}")
    
    # Download the tarball
    logger.info(f"Downloading vHAMSTeR Models v{version} from {db_url}")
    download(db_url, tarball_path)
    
    # Calculate MD5
    md5_sum = calc_md5_sum(tarball_path)
    logger.info(f"Downloaded file MD5: {md5_sum}")
    
    # Extract to the model directory
    logger.info(f"Extracting to: {abs_path}")
    untar(tarball_path, model_dir, version)
    
    # Keep tarball for user reference
    logger.info(f"Keeping tarball at {tarball_path}")
    
    # List models after installation
    ensemble_dir = os.path.join(model_dir, "best_params_20260331")
    if os.path.isdir(ensemble_dir):
        fold_dirs = [d for d in os.listdir(ensemble_dir) if d.startswith("fold_")]
        logger.info(f"Installed {len(fold_dirs)} fold models: {sorted(fold_dirs)}")
    
    temperature_file = os.path.join(model_dir, "length_class_temperatures_continuous_brier.json")
    if os.path.exists(temperature_file):
        file_size = os.path.getsize(temperature_file) / (1024 * 1024)
        logger.info(f"Length-class calibration file installed ({file_size:.2f} MB)")
    
    logger.info("Installation completed successfully!")


@click.command()
@click.option(
    "-o",
    "--outdir",
    type=click.Path(path_type=str),
    help="Path to directory where models will be installed",
    default=None,
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Force reinstallation of models even if they already exist",
    default=False,
)
@click.option(
    "-v",
    "--version",
    type=str,
    default="1.0.1",
    show_default=True,
    help="Version of vHAMSTeR models to install",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable verbose debug logging during installation",
)
def main(outdir, force, version, debug):
    """
    Download and install vHAMSTeR models from NERSC.
    
    By default, models are installed in an environment-scoped directory under the
    active Python environment. Use -o to specify a custom installation directory
    if that location is not writable.
    """
    configure_logging(debug)

    
    if outdir is None:
        model_dir = get_default_model_dir()
    else:
        model_dir = os.path.abspath(outdir)
        logger.info(f"Using specified directory: {model_dir}")
    
    abs_path = os.path.abspath(model_dir)
    logger.info(f"Model installation directory: {abs_path}")
    
    if force:
        logger.info("Force reinstall requested. Will reinstall models even if they exist.")
    
    instantiate_install(model_dir, force, version)
    
    # Final verification and summary
    logger.info("\n" + "="*60)
    logger.info("INSTALLATION SUMMARY")
    logger.info("="*60)
    
    ensemble_dir = os.path.join(model_dir, "best_params_20260331")
    if os.path.exists(ensemble_dir):
        fold_dirs = [d for d in os.listdir(ensemble_dir) if d.startswith("fold_")]
        logger.info(f"✓ Ensemble directory: {ensemble_dir}")
        logger.info(f"✓ Fold models installed: {len(fold_dirs)}/5")
    else:
        logger.warning(f"✗ Ensemble directory not found: {ensemble_dir}")
    
    temperature_file = os.path.join(model_dir, "length_class_temperatures_continuous_brier.json")
    if os.path.exists(temperature_file):
        logger.info(f"✓ Temperature file: {temperature_file}")
    else:
        logger.warning(f"✗ Temperature file not found: {temperature_file}")
    
    logger.info("\nTo use the models with vhamster:")
    logger.info(f"  vhamster --fasta <input.fasta> --output <output_dir> --ensemble-dir {ensemble_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
