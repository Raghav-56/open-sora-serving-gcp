"""
GCS I/O utilities for Open-Sora v2 video uploads and weight downloads.
"""

import shutil
from pathlib import Path

from google.cloud import storage
from loguru import logger


def upload_video_to_gcs(
    local_path: str,
    bucket_name: str,
    prefix: str = "",
) -> str:
    """
    Upload video file to GCS.

    Args:
        local_path: Path to local video file
        bucket_name: GCS bucket name
        prefix: GCS path prefix (e.g., "videos/job_id/")

    Returns:
        str: GCS URI (gs://bucket/path/to/video.mp4)

    Raises:
        FileNotFoundError: If local file doesn't exist
        PermissionError: If upload permission denied
        ValueError: If bucket not found
    """
    local_file = Path(local_path)

    if not local_file.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    filename = local_file.name
    gcs_path = f"{prefix}{filename}" if prefix else filename
    gcs_path = gcs_path.replace("//", "/")

    logger.info("📤 Uploading to GCS:")
    logger.info(f"   Bucket: {bucket_name}")
    logger.info(f"   Path: {gcs_path}")
    logger.info(f"   Size: {local_file.stat().st_size / 1e6:.1f} MB")

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)

        blob.upload_from_filename(str(local_path))

        gcs_uri = f"gs://{bucket_name}/{gcs_path}"
        logger.info(f"✅ Upload successful: {gcs_uri}")

        return gcs_uri

    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")

        if "403" in str(e) or "Forbidden" in str(e):
            raise PermissionError(
                f"Permission denied: gs://{bucket_name}/{gcs_path}. "
                "Ensure service account has 'Storage Object Creator' role."
            ) from e
        elif "404" in str(e) or "Not Found" in str(e):
            raise ValueError(f"Bucket not found: {bucket_name}") from e
        else:
            raise


def save_video_locally(video_path: str, output_path: str) -> str:
    """
    Copy video to output path.

    Args:
        video_path: Path to source video file
        output_path: Path where video will be saved

    Returns:
        str: Path to saved video file
    """
    logger.info(f"💾 Saving video to: {output_path}")

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(video_path, output_path)

    if not Path(output_path).exists():
        raise RuntimeError(f"Video file was not created: {output_path}")

    file_size = Path(output_path).stat().st_size
    logger.info(f"✅ Video saved ({file_size / 1e6:.1f} MB)")

    return output_path


def download_blob(
    bucket_name: str,
    source_blob_name: str,
    destination_file_name: str,
) -> None:
    """Download a single blob from GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    Path(destination_file_name).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(destination_file_name)

    logger.debug(f"✅ Downloaded: {source_blob_name}")


def download_directory(
    bucket_name: str,
    source_prefix: str,
    destination_dir: str,
    skip_existing: bool = True,
) -> int:
    """
    Download all blobs from a GCS directory.

    Args:
        bucket_name: GCS bucket name
        source_prefix: Prefix path in bucket
        destination_dir: Local destination directory
        skip_existing: Skip files that already exist locally

    Returns:
        int: Number of files downloaded
    """
    logger.info(f"📦 Downloading from gs://{bucket_name}/{source_prefix}")
    logger.info(f"   Destination: {destination_dir}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix=source_prefix))

    if not blobs:
        raise ValueError(f"No files found: gs://{bucket_name}/{source_prefix}")

    logger.info(f"   Found {len(blobs)} files")

    downloaded = 0
    skipped = 0

    for i, blob in enumerate(blobs, 1):
        if blob.name.endswith("/"):
            continue

        relative_path = blob.name[len(source_prefix):]
        local_path = Path(destination_dir) / relative_path

        if skip_existing and local_path.exists():
            logger.debug(f"[{i}/{len(blobs)}] ⏭️  Skipping: {relative_path}")
            skipped += 1
            continue

        size_gb = blob.size / 1e9
        logger.info(f"[{i}/{len(blobs)}] ⬇️  {blob.name} ({size_gb:.2f} GB)")

        download_blob(bucket_name, blob.name, str(local_path))
        downloaded += 1

    logger.info("✅ Download complete!")
    logger.info(f"   Downloaded: {downloaded} files")
    logger.info(f"   Skipped: {skipped} files")

    return downloaded


def verify_file_exists_in_gcs(bucket_name: str, blob_name: str) -> bool:
    """Check if a file exists in GCS."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.exists()
    except Exception as e:
        logger.error(f"❌ Error checking GCS file: {e}")
        return False
