"""
Open-Sora v2 inference worker with async job queue processing.
Runs background thread to process jobs from queue.
"""

import os
import shutil
import time
from pathlib import Path
from threading import Thread
from typing import Optional

from loguru import logger

from app.gcs_io import upload_video_to_gcs
from app.job_manager import JobManager, JobStatus
from app.opensora_runner import OpenSoraRunner


class InferenceWorker:
    """
    Worker that processes Open-Sora v2 video generation jobs.

    Features:
    - Background thread processes jobs sequentially
    - One job at a time (GPU constraint)
    - Configurable timeout per job
    - Proper error handling and status updates
    
    Configuration via environment variables:
    - GENERATION_TIMEOUT: Max time per job in seconds (default: 1800)
    """

    def __init__(self, runner: OpenSoraRunner, job_manager: JobManager):
        """
        Initialize inference worker.

        Args:
            runner: OpenSoraRunner instance for video generation
            job_manager: JobManager instance for queue management
        """
        self.runner = runner
        self.job_manager = job_manager
        self.local_output_dir = Path("/tmp/opensora_outputs")
        self.local_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurable timeout
        self.generation_timeout = int(
            os.getenv("GENERATION_TIMEOUT", "1800")
        )

        self._running = False
        self._worker_thread: Optional[Thread] = None

        logger.info("🔧 InferenceWorker initialized")
        logger.info(f"   Timeout: {self.generation_timeout}s")
        logger.info(f"   Local output: {self.local_output_dir}")

        self._start_worker()

    def _start_worker(self):
        """Start background worker thread."""
        self._running = True
        self._worker_thread = Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info("✅ Background worker thread started")

    def _worker_loop(self):
        """Background worker loop - processes jobs from queue."""
        logger.info("🔄 Worker loop started")

        while self._running:
            try:
                job_id = self.job_manager.get_next_job()

                if job_id is None:
                    time.sleep(1)
                    continue

                self._process_job(job_id)

            except Exception as e:
                logger.error(f"❌ Worker loop error: {e}")
                logger.exception(e)
                time.sleep(5)

    def _process_job(self, job_id: str):
        """Process a single Open-Sora v2 video generation job."""
        job = self.job_manager.get_job(job_id)

        if job is None:
            logger.error(f"❌ Job not found: {job_id}")
            return

        if job.status == JobStatus.CANCELLED:
            logger.info(f"🚫 Job cancelled before processing: {job_id}")
            return

        logger.info(f"🎬 Processing job: {job_id}")
        logger.info(f"  Prompt: {job.prompt[:100]}...")
        logger.info(f"  Resolution: {job.resolution}")
        logger.info(f"  Frames: {job.num_frames}")
        logger.info(f"  Aspect Ratio: {job.aspect_ratio}")
        logger.info(f"  Motion Score: {job.motion_score}")

        start_time = time.time()

        try:
            logger.info("🎨 Generating video with Open-Sora v2...")
            video_path, actual_seed = self.runner.generate(
                prompt=job.prompt,
                resolution=job.resolution,
                num_frames=job.num_frames,
                aspect_ratio=job.aspect_ratio,
                motion_score=job.motion_score,
                seed=job.seed,
            )

            if job.status == JobStatus.CANCELLED:
                logger.info(f"🚫 Job cancelled during generation: {job_id}")
                return

            generation_time = time.time() - start_time
            logger.info(f"✅ Video generated in {generation_time:.1f}s")

            # Save locally
            filename = f"{job_id}.mp4"
            local_path = self.local_output_dir / filename
            shutil.copy2(video_path, str(local_path))

            file_size_mb = local_path.stat().st_size / (1024 * 1024)
            logger.info(f"💾 Saved: {local_path} ({file_size_mb:.1f} MB)")

            # Upload to GCS
            logger.info("☁️  Uploading to GCS...")
            prefix = job.output_prefix.rstrip("/") + "/" if job.output_prefix else ""
            gcs_prefix = f"{prefix}{job_id}/"

            video_uri = upload_video_to_gcs(
                str(local_path),
                job.output_bucket,
                gcs_prefix,
            )

            logger.info(f"✅ Uploaded: {video_uri}")

            # Cleanup
            try:
                local_path.unlink()
                logger.info("🗑️  Cleaned up local file")
            except Exception as e:
                logger.warning(f"⚠️  Failed to cleanup: {e}")

            # Mark completed
            elapsed_time = time.time() - start_time
            self.job_manager.complete_job(job_id, video_uri, actual_seed)

            logger.info(f"✅ Job completed: {job_id}")
            logger.info(f"   Total time: {elapsed_time:.1f}s")
            logger.info(f"   Video URI: {video_uri}")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Job failed: {job_id}")
            logger.error(f"   Error: {error_msg}")
            logger.exception(e)
            self.job_manager.fail_job(job_id, error_msg)

    def shutdown(self):
        """Graceful shutdown - wait for current job to complete."""
        logger.info("🛑 Shutting down worker...")
        self._running = False

        if self._worker_thread and self._worker_thread.is_alive():
            logger.info("⏳ Waiting for current job to complete...")
            self._worker_thread.join(timeout=60)

        logger.info("✅ Worker shutdown complete")
