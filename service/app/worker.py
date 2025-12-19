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

from app.utils.gcs_io import upload_video_to_gcs
from app.jobs import JobManager, JobStatus
from app.opensora import OpenSoraRunner
from app.core.config import GENERATION_TIMEOUT, BASE_OUTPUT_DIR


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
        self.base_output_dir = Path(BASE_OUTPUT_DIR)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurable timeout
        self.generation_timeout = GENERATION_TIMEOUT

        self._running = False
        self._worker_thread: Optional[Thread] = None

        logger.info("🔧 InferenceWorker initialized")
        logger.info(f"   Timeout: {self.generation_timeout}s")
        logger.info(f"   Local output base: {self.base_output_dir}")

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
        logger.info(f"  Num Steps: {job.num_steps}")
        logger.info(f"  Mode: {job.mode}")
        logger.info(f"  FPS: {job.fps}")
        logger.info(f"  Num Samples: {job.num_samples}")

        start_time = time.time()
        job_output_dir = self.base_output_dir / job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info("🎨 Generating video with Open-Sora v2...")
            video_paths, actual_seed, log_tail = self.runner.generate(
                prompt=job.prompt,
                resolution=job.resolution,
                num_frames=job.num_frames,
                aspect_ratio=job.aspect_ratio,
                motion_score=job.motion_score,
                seed=job.seed,
                num_steps=job.num_steps,
                mode=job.mode,
                fps=job.fps,
                num_samples=job.num_samples,
                guidance=job.guidance,
                prompt_refine=job.prompt_refine,
                save_dir=str(job_output_dir),
                timeout_seconds=self.generation_timeout,
            )

            if job.status == JobStatus.CANCELLED:
                logger.info(f"🚫 Job cancelled during generation: {job_id}")
                self._cleanup_dir(job_output_dir)
                return

            generation_time = time.time() - start_time
            logger.info(f"✅ Video generated in {generation_time:.1f}s")

            # Normalize filenames and log sizes
            normalized_paths = []
            for idx, path in enumerate(video_paths):
                src = Path(path)
                suffix = f"_{idx+1}" if len(video_paths) > 1 else ""
                normalized = job_output_dir / f"{job_id}{suffix}.mp4"
                shutil.copy2(src, normalized)
                file_size_mb = normalized.stat().st_size / (1024 * 1024)
                logger.info(
                    f"💾 Saved: {normalized} ({file_size_mb:.1f} MB)"
                )
                normalized_paths.append(normalized)

            # Upload to GCS
            logger.info("☁️  Uploading to GCS...")
            prefix = (
                job.output_prefix.rstrip("/") + "/"
                if job.output_prefix
                else ""
            )
            gcs_prefix = f"{prefix}{job_id}/"

            if job.status == JobStatus.CANCELLED:
                logger.info(f"🚫 Job cancelled before upload: {job_id}")
                self._cleanup_dir(job_output_dir)
                return

            uploaded_uris = []
            try:
                for path in normalized_paths:
                    uri = upload_video_to_gcs(
                        str(path),
                        job.output_bucket,
                        gcs_prefix,
                    )
                    uploaded_uris.append(uri)
                logger.info(f"✅ Uploaded: {uploaded_uris}")
            except Exception as upload_exc:
                logger.error(f"❌ Upload failed: {upload_exc}")
                logger.exception(upload_exc)
                self._cleanup_dir(job_output_dir)
                raise

            # Cleanup
            try:
                self._cleanup_dir(job_output_dir)
                logger.info("🗑️  Cleaned up job directory")
            except Exception as e:
                logger.warning(f"⚠️  Failed to cleanup: {e}")

            # Mark completed
            elapsed_time = time.time() - start_time
            self.job_manager.complete_job(
                job_id,
                uploaded_uris,
                actual_seed,
                log_tail,
            )

            logger.info(f"✅ Job completed: {job_id}")
            logger.info(f"   Total time: {elapsed_time:.1f}s")
            logger.info(f"   Video URIs: {uploaded_uris}")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Job failed: {job_id}")
            logger.error(f"   Error: {error_msg}")
            logger.exception(e)
            self.job_manager.fail_job(job_id, error_msg)

            # Ensure cleanup on failure
            self._cleanup_dir(job_output_dir)

    def shutdown(self):
        """Graceful shutdown - wait for current job to complete."""
        logger.info("🛑 Shutting down worker...")
        self._running = False

        if self._worker_thread and self._worker_thread.is_alive():
            logger.info("⏳ Waiting for current job to complete...")
            self._worker_thread.join(timeout=60)

        logger.info("✅ Worker shutdown complete")

    def _cleanup_dir(self, path: Path):
        """Remove a job-specific directory if it exists."""
        try:
            if path.exists():
                shutil.rmtree(path)
        except Exception as exc:
            logger.warning(f"⚠️ Cleanup failed for {path}: {exc}")
