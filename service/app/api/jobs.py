"""
Job management endpoints for Open-Sora API.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, status

from app.models.responses import JobStatusResponse, QueueStatusResponse
from app.jobs import JobManager


router = APIRouter(prefix="/v1", tags=["jobs"])

# Global job_manager reference (set by lifespan in main.py)
job_manager: Optional[JobManager] = None


def set_job_manager(job_mgr: JobManager):
    """Set the global job manager instance."""
    global job_manager
    job_manager = job_mgr


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get job status"
)
async def get_job_status(job_id: str):
    """
    Get status of a video generation job.
    
    Args:
        job_id: Job identifier returned from /predict or /v1/generate
        
    Returns:
        Job status including video_uri when completed
        
    Status values:
        - queued: Job is waiting in queue
        - processing: Video is being generated
        - completed: Video is ready (video_uri available)
        - failed: Generation failed (error message available)
        - cancelled: Job was cancelled
    """
    if job_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    
    job = job_manager.get_job(job_id)
    
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "job_not_found",
                "message": f"No job found with ID: {job_id}",
            }
        )
    
    return job.to_dict()


@router.delete(
    "/jobs/{job_id}",
    status_code=status.HTTP_200_OK,
    summary="Cancel job"
)
async def cancel_job(job_id: str):
    """
    Cancel a video generation job.
    
    - Queued jobs: Removed from queue
    - Processing jobs: Marked for cancellation (cannot stop mid-generation,
      but result will not be saved)
    - Completed/failed jobs: Cannot be cancelled
    
    Args:
        job_id: Job identifier
        
    Returns:
        Cancellation status message
    """
    if job_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    
    result = job_manager.cancel_job(job_id)
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result
        )
    
    return result


@router.get(
    "/queue",
    response_model=QueueStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get queue status"
)
async def get_queue_status():
    """
    Get current queue status and statistics.
    
    Returns:
        Queue information including size, current job, and statistics
    """
    if job_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    
    return job_manager.get_queue_status()
