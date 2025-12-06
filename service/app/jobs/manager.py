"""
Job manager for Open-Sora v2 async video generation.
Handles job submission, status tracking, and cancellation.
"""

from collections import defaultdict
from datetime import datetime
from queue import Queue, Empty
from threading import Lock
from typing import Dict, Optional, Union, List

from loguru import logger

from app.jobs.models import Job, JobStatus
from app.core.config import JOB_RETENTION_SECONDS, MAX_COMPLETED_JOBS


class JobManager:
    """
    Manages job queue and status tracking for async video generation.
    
    Features:
    - Submit jobs with unique IDs
    - Track job status
    - Cancel queued/processing jobs
    - Query queue state
    - Auto-cleanup old completed/failed jobs
    
    Configuration via environment variables:
    - JOB_RETENTION_SECONDS: How long to keep completed jobs (default: 3600)
    - MAX_COMPLETED_JOBS: Maximum completed jobs to retain (default: 100)
    """
    
    def __init__(self):
        """Initialize job manager with configurable retention settings."""
        # Configurable via environment variables
        self.job_retention_seconds = JOB_RETENTION_SECONDS
        self.max_completed_jobs = MAX_COMPLETED_JOBS
        
        self.jobs: Dict[str, Job] = {}
        self.job_queue: Queue = Queue()
        self.current_job_id: Optional[str] = None
        # Lock to protect shared state across threads
        self.lock: Lock = Lock()
        
        # Counter for sequential job IDs
        self.job_counters: Dict[str, int] = defaultdict(int)
        
        logger.info("📋 JobManager initialized")
        logger.info(f"   Retention: {self.job_retention_seconds}s")
        logger.info(f"   Max completed jobs: {self.max_completed_jobs}")
    
    def generate_job_id(self) -> str:
        """
        Generate unique job ID with format: YYYYMMDD_HHMMSS_mmm_XXX
        
        Uses sequential counter per millisecond to prevent collisions.
        
        Returns:
            str: Unique job ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        # YYYYMMDD_HHMMSS_mmm
        
        # Increment counter for this millisecond
        self.job_counters[timestamp] += 1
        counter = self.job_counters[timestamp]
        
        job_id = f"{timestamp}_{counter:03d}"
        
        logger.debug(f"Generated job_id: {job_id}")
        return job_id
    
    def submit_job(
        self,
        prompt: str,
        resolution: str,
        num_frames: int,
        aspect_ratio: str,
        motion_score: Optional[Union[int, str]],
        seed: Optional[int],
        mode: str,
        fps: int,
        num_samples: int,
        guidance: Optional[float],
        prompt_refine: bool,
        output_bucket: str,
        output_prefix: str,
    ) -> str:
        """
        Submit a new Open-Sora v2 video generation job.

        Args:
            prompt: Text prompt for video generation
            resolution: "256px" or "768px"
            num_frames: Frame count (4k+1 format)
            aspect_ratio: "16:9", "9:16", "1:1", "2.39:1"
            motion_score: Motion intensity (1-10, None for auto)
            seed: Random seed (optional)
            mode: Generation mode ("t2i2v" or "t2v")
            fps: Frames per second
            num_samples: Number of samples to generate
            guidance: Guidance scale override
            prompt_refine: Enable prompt refinement
            output_bucket: GCS bucket name
            output_prefix: GCS path prefix

        Returns:
            str: Job ID
        """
        job_id = self.generate_job_id()

        job = Job(
            job_id=job_id,
            status=JobStatus.QUEUED,
            prompt=prompt,
            resolution=resolution,
            num_frames=num_frames,
            aspect_ratio=aspect_ratio,
            motion_score=motion_score,
            seed=seed,
            mode=mode,
            fps=fps,
            num_samples=num_samples,
            guidance=guidance,
            prompt_refine=prompt_refine,
            output_bucket=output_bucket,
            output_prefix=output_prefix,
        )

        with self.lock:
            self.jobs[job_id] = job
            self.job_queue.put(job_id)

        logger.info(f"✅ Job submitted: {job_id}")
        logger.info(f"   Queue size: {self.queue_size()}")

        return job_id
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get job by ID.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job object or None if not found
        """
        with self.lock:
            return self.jobs.get(job_id)
    
    def get_next_job(self) -> Optional[str]:
        """
        Get next job ID from queue (blocking).
        
        Returns:
            str: Job ID or None if queue is empty
        """
        # Pop items until we find a queued job or the queue is empty
        while True:
            try:
                job_id = self.job_queue.get_nowait()
            except Empty:
                return None

            job = self.jobs.get(job_id)
            if not job:
                # job removed from store; skip it
                continue

            if job.status != JobStatus.QUEUED:
                # This job was likely cancelled or already being processed
                logger.debug(
                    "Skipping job in queue (status=%s): %s",
                    job.status.value,
                    job_id,
                )
                continue

            # Mark as processing and return
            with self.lock:
                job.status = JobStatus.PROCESSING
                job.started_at = datetime.now().timestamp()
                self.current_job_id = job_id

            logger.info(f"🎬 Starting job: {job_id}")
            return job_id
    
    def complete_job(
        self,
        job_id: str,
        video_uris: List[str],
        actual_seed: int,
        log_tail: Optional[List[str]] = None,
    ):
        """
        Mark job as completed.
        
        Args:
            job_id: Job ID
            video_uris: List of GCS URIs of generated videos
            actual_seed: Actual seed used for generation
            log_tail: Last N lines of generation logs
        """
        if job_id not in self.jobs:
            logger.error(f"❌ Job not found: {job_id}")
            return

        with self.lock:
            # If the job was cancelled while processing, do not mark as
            # completed and do not save results.
            job = self.jobs[job_id]
            if job.status == JobStatus.CANCELLED:
                logger.warning(
                    "Completion called for cancelled job %s - discarding"
                    " results",
                    job_id,
                )
                # ensure completed_at exists so retention/cleanup works
                if not job.completed_at:
                    job.completed_at = datetime.now().timestamp()
                if self.current_job_id == job_id:
                    self.current_job_id = None
                return

            job.status = JobStatus.COMPLETED
            job.video_uri = video_uris[0] if video_uris else None
            job.video_uris = video_uris
            job.actual_seed = actual_seed
            job.log_tail = log_tail or []
            job.completed_at = datetime.now().timestamp()

            if self.current_job_id == job_id:
                self.current_job_id = None
        
        logger.info(f"✅ Job completed: {job_id}")
        logger.info(f"   Videos: {video_uris}")
    
    def fail_job(self, job_id: str, error: str):
        """
        Mark job as failed.
        
        Args:
            job_id: Job ID
            error: Error message
        """
        if job_id not in self.jobs:
            logger.error(f"❌ Job not found: {job_id}")
            return

        with self.lock:
            # If the job was cancelled while processing, keep CANCELLED as
            # final status.
            if self.jobs[job_id].status == JobStatus.CANCELLED:
                logger.warning(
                    "Failure reported for cancelled job %s - keeping"
                    " CANCELLED status",
                    job_id,
                )
                if not self.jobs[job_id].completed_at:
                    self.jobs[job_id].completed_at = datetime.now().timestamp()
                if self.current_job_id == job_id:
                    self.current_job_id = None
                return

            self.jobs[job_id].status = JobStatus.FAILED
            self.jobs[job_id].error = error
            self.jobs[job_id].completed_at = datetime.now().timestamp()

            if self.current_job_id == job_id:
                self.current_job_id = None
        
        logger.error(f"❌ Job failed: {job_id}")
        logger.error(f"   Error: {error}")
    
    def cancel_job(self, job_id: str) -> dict:
        """
        Cancel a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            dict: Status message
        """
        if job_id not in self.jobs:
            return {"error": "Job not found"}

        now = datetime.now().timestamp()
        with self.lock:
            job = self.jobs[job_id]

            if job.status == JobStatus.COMPLETED:
                return {"message": "Job already completed, cannot cancel"}

            if job.status == JobStatus.FAILED:
                return {"message": "Job already failed"}

            if job.status == JobStatus.CANCELLED:
                return {"message": "Job already cancelled"}

            if job.status == JobStatus.PROCESSING:
                # Mark for cancellation (can't stop mid-generation).
                # We won't remove current_job_id here because processing is
                # still ongoing; the worker should respect the CANCELLED state.
                job.status = JobStatus.CANCELLED
                logger.warning(
                    (
                        "Job marked for cancellation (currently processing):"
                        f" {job_id}"
                    )
                )
                return {
                    "message": (
                        "Job marked for cancellation. Video generation cannot"
                        " be stopped mid-process, but result will not be"
                        " saved."
                    )
                }

            # Job is queued - can remove from queue. Set `completed_at` so the
            # cleanup routine can remove it.
            job.status = JobStatus.CANCELLED
            job.completed_at = now

        logger.info(f"🚫 Job cancelled: {job_id}")
        return {"message": "Job cancelled successfully"}
    
    def get_queue_status(self) -> dict:
        """
        Get current queue status.
        
        Returns:
            dict: Queue information
        """
        with self.lock:
            queued_jobs = [
                job_id for job_id, job in self.jobs.items()
                if job.status == JobStatus.QUEUED
            ]
        
        return {
            "queue_size": len(queued_jobs),
            "currently_processing": self.current_job_id,
            "queued_job_ids": queued_jobs,
            "total_jobs": len(self.jobs),
            "completed_jobs": sum(
                1 for job in self.jobs.values()
                if job.status == JobStatus.COMPLETED
            ),
            "failed_jobs": sum(
                1 for job in self.jobs.values()
                if job.status == JobStatus.FAILED
            ),
            "cancelled_jobs": sum(
                1 for job in self.jobs.values()
                if job.status == JobStatus.CANCELLED
            ),
        }
    
    def queue_size(self) -> int:
        """Get number of jobs in queue."""
        # Return count of jobs currently QUEUED (ignore cancelled ones still in
        # the queue)
        with self.lock:
            return sum(
                1 for job in self.jobs.values()
                if job.status == JobStatus.QUEUED
            )
    
    def is_busy(self) -> bool:
        """Check if currently processing a job."""
        with self.lock:
            return self.current_job_id is not None

    def cleanup_old_jobs(self) -> int:
        """
        Remove old completed/failed/cancelled jobs to prevent memory leaks.
        
        Returns:
            int: Number of jobs removed
        """
        now = datetime.now().timestamp()
        to_remove = []

        # Snapshot state under lock then operate on snapshot to avoid long
        # lock holds
        with self.lock:
            jobs_snapshot = dict(self.jobs)

        # Find jobs older than retention period
        for job_id, job in jobs_snapshot.items():
            if job.status in (
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            ):
                if job.completed_at and (
                    now - job.completed_at
                ) > self.job_retention_seconds:
                    to_remove.append(job_id)
        
        # Also remove if we have too many completed jobs
        completed_jobs = [
            (job_id, job) for job_id, job in jobs_snapshot.items()
            if job.status == JobStatus.COMPLETED and job.completed_at
        ]
        completed_jobs.sort(key=lambda x: x[1].completed_at or 0)
        
        if len(completed_jobs) > self.max_completed_jobs:
            excess = len(completed_jobs) - self.max_completed_jobs
            for job_id, _ in completed_jobs[:excess]:
                if job_id not in to_remove:
                    to_remove.append(job_id)
        
        # Remove jobs
        for job_id in to_remove:
            # acquire lock while deleting
            with self.lock:
                if job_id in self.jobs:
                    del self.jobs[job_id]
        
        if to_remove:
            logger.info(f"🧹 Cleaned up {len(to_remove)} old jobs")
        
        return len(to_remove)
