"""
Bootstrap script to download Open-Sora v2 model weights from GCS.
Runs before the FastAPI server starts to ensure weights are available.
"""

import os
import sys
from pathlib import Path

# Add /app to Python path BEFORE imports so absolute imports work when run as script
sys.path.insert(0, "/app")

from loguru import logger

from app.gcs_io import download_directory


class WeightBootstrapper:
    """
    Downloads and verifies Open-Sora v2 model weights from GCS.

    Required checkpoint files:
    - Open_Sora_v2.safetensors (main model, ~42GB)
    - hunyuan_vae.safetensors (VAE model)
    - google/t5-v1_1-xxl/ (T5 encoder)
    - openai/clip-vit-large-patch14/ (CLIP encoder)
    """

    # Open-Sora v2 required checkpoint files
    CRITICAL_FILES = [
        "Open_Sora_v2.safetensors",
        "hunyuan_vae.safetensors",
    ]

    def __init__(
        self,
        bucket_name: str,
        source_prefix: str,
        destination_dir: str,
    ):
        """
        Initialize bootstrapper.

        Args:
            bucket_name: GCS bucket containing model weights
            source_prefix: Prefix/path in bucket (e.g., "ckpts/")
            destination_dir: Local directory where weights will be downloaded
        """
        self.bucket_name = bucket_name
        self.source_prefix = source_prefix
        self.destination_dir = Path(destination_dir)

        logger.info("🔧 WeightBootstrapper initialized")
        logger.info(f"   Source: gs://{bucket_name}/{source_prefix}")
        logger.info(f"   Destination: {destination_dir}")

    def check_weights_exist(self) -> bool:
        """
        Check if Open-Sora v2 critical weights exist locally.

        Returns:
            bool: True if critical files exist
        """
        logger.info("🔍 Checking for Open-Sora v2 weights locally...")

        for filename in self.CRITICAL_FILES:
            filepath = self.destination_dir / filename
            if not filepath.exists():
                logger.info(f"   ❌ Missing: {filename}")
                return False
            size_gb = filepath.stat().st_size / 1e9
            logger.info(f"   ✅ Found: {filename} ({size_gb:.2f} GB)")

        return True

    def download_weights(self, skip_existing: bool = True) -> None:
        """
        Download weights from GCS.

        Args:
            skip_existing: If True, skip files that already exist locally

        Raises:
            Exception: If download fails
        """
        logger.info("📦 Downloading Open-Sora v2 weights from GCS...")

        try:
            downloaded = download_directory(
                bucket_name=self.bucket_name,
                source_prefix=self.source_prefix,
                destination_dir=str(self.destination_dir),
                skip_existing=skip_existing,
            )

            logger.info(f"✅ Downloaded {downloaded} files")

        except Exception as e:
            logger.error(f"❌ Weight download failed: {e}")
            raise

    def verify_weights(self) -> None:
        """
        Verify that Open-Sora v2 critical files exist.

        Raises:
            FileNotFoundError: If critical files are missing
        """
        logger.info("🔍 Verifying Open-Sora v2 weights...")

        missing = []
        for filename in self.CRITICAL_FILES:
            filepath = self.destination_dir / filename
            if not filepath.exists():
                missing.append(filename)
            else:
                size_gb = filepath.stat().st_size / 1e9
                logger.info(f"   ✅ {filename}: {size_gb:.2f} GB")

        if missing:
            raise FileNotFoundError(
                f"Missing Open-Sora v2 weights: {', '.join(missing)}"
            )

        logger.info("✅ Open-Sora v2 weight verification complete!")

    def bootstrap(self, force_download: bool = False) -> None:
        """
        Main bootstrap process: check, download, and verify weights.

        Args:
            force_download: If True, re-download even if files exist

        Raises:
            Exception: If bootstrap fails
        """
        logger.info("🚀 Starting Open-Sora v2 weight bootstrap...")

        self.destination_dir.mkdir(parents=True, exist_ok=True)

        if not force_download and self.check_weights_exist():
            logger.info("✅ Weights already exist locally, skipping download")
        else:
            if force_download:
                logger.info("⚠️  Force download enabled, re-downloading weights")
            self.download_weights(skip_existing=not force_download)

        self.verify_weights()

        logger.info("✅ Open-Sora v2 weight bootstrap complete!")


def main():
    """
    Main entry point for weight bootstrap script.
    Called from start.sh before starting the API server.
    """
    bucket_name = os.getenv("WEIGHT_BUCKET")
    if not bucket_name:
        logger.error("❌ WEIGHT_BUCKET environment variable is required")
        sys.exit(1)

    source_prefix = os.getenv("WEIGHT_PREFIX", "ckpts/")
    destination_dir = os.getenv("MODEL_PATH", "/app/ckpts")
    force_download = os.getenv("FORCE_DOWNLOAD", "false").lower() == "true"

    logger.info("📋 Bootstrap Configuration:")
    logger.info(f"   WEIGHT_BUCKET: {bucket_name}")
    logger.info(f"   WEIGHT_PREFIX: {source_prefix}")
    logger.info(f"   MODEL_PATH: {destination_dir}")
    logger.info(f"   FORCE_DOWNLOAD: {force_download}")

    try:
        bootstrapper = WeightBootstrapper(
            bucket_name=bucket_name,
            source_prefix=source_prefix,
            destination_dir=destination_dir,
        )

        bootstrapper.bootstrap(force_download=force_download)

        logger.info("✅ Bootstrap successful - ready to start API server")
        sys.exit(0)

    except Exception as e:
        logger.error(f"❌ Bootstrap failed: {e}")
        logger.exception(e)
        logger.error("Cannot start API server without model weights")
        sys.exit(1)


if __name__ == "__main__":
    main()
