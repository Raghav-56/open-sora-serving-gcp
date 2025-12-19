"""
Video generation endpoints for Open-Sora API.
"""

from typing import Optional, Union, cast

from fastapi import APIRouter, HTTPException, status, Request
from loguru import logger

from app.models.requests import VideoGenerationRequest
from app.models.responses import JobSubmissionResponse
from app.jobs import JobManager
from app.worker import InferenceWorker


router = APIRouter(tags=["generation"])

# Global references (set by lifespan in main.py)
job_manager: Optional[JobManager] = None
worker: Optional[InferenceWorker] = None


def set_dependencies(job_mgr: JobManager, worker_instance: InferenceWorker):
    """Set the global job manager and worker instances."""
    global job_manager, worker
    job_manager = job_mgr
    worker = worker_instance


async def submit_video_generation(request: VideoGenerationRequest) -> dict:
    """
    Submit video generation job to the queue.
    Used by both /predict (Vertex AI) and /v1/generate (direct API).
    
    Args:
        request: VideoGenerationRequest with prompt and parameters
        
    Returns:
        dict with job_id, status, and expected_video_uris
    """
    if job_manager is None or worker is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized",
        )

    logger.info("📹 New video generation request:")
    logger.info(f"  Prompt: {request.prompt[:100]}...")
    logger.info(f"  Resolution: {request.resolution}")
    logger.info(f"  Frames: {request.num_frames}")
    logger.info(f"  Aspect Ratio: {request.aspect_ratio}")
    logger.info(f"  Motion Score: {request.motion_score}")
    logger.info(f"  Num Steps: {request.num_steps}")
    logger.info(f"  Mode: {request.mode}")
    logger.info(f"  FPS: {request.fps}")
    logger.info(f"  Num Samples: {request.num_samples}")
    logger.info(f"  Guidance: {request.guidance}")
    logger.info(f"  Prompt Refine: {request.prompt_refine}")
    logger.info(f"  Seed: {request.seed}")
    bucket_path = f"gs://{request.output_bucket}/{request.output_prefix}"
    logger.info(f"  Output: {bucket_path}")

    try:
        job_id = job_manager.submit_job(
            prompt=request.prompt,
            resolution=request.resolution,
            num_frames=request.num_frames,
            aspect_ratio=request.aspect_ratio,
            motion_score=cast(
                Optional[Union[int, str]], request.motion_score
            ),
            seed=request.seed,
            num_steps=request.num_steps,
            mode=request.mode,
            fps=request.fps,
            num_samples=request.num_samples,
            guidance=request.guidance,
            prompt_refine=request.prompt_refine,
            output_bucket=request.output_bucket,
            output_prefix=request.output_prefix,
        )

        prefix = (
            request.output_prefix.rstrip("/") + "/"
            if request.output_prefix
            else ""
        )
        gcs_prefix = f"gs://{request.output_bucket}/{prefix}{job_id}/"
        expected_video_uris = [
            f"{gcs_prefix}{job_id}{f'_{i+1}' if i else ''}.mp4"
            for i in range(request.num_samples)
        ]

        logger.info(f"✅ Job queued: {job_id}")
        logger.info(f"   Expected URIs: {expected_video_uris}")

        return {
            "job_id": job_id,
            "status": "queued",
            "expected_video_uris": expected_video_uris,
            "expected_gcs_prefix": gcs_prefix,
        }

    except Exception as e:
        logger.error(f"❌ Failed to submit job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "submission_failed", "message": str(e)},
        )


@router.post(
    "/predict",
    response_model=JobSubmissionResponse,
    status_code=status.HTTP_200_OK,
    summary="Vertex AI prediction endpoint"
)
async def predict(request: Request):
    """
    Vertex AI prediction endpoint (required by Vertex AI deployment).
    Handles both Vertex AI's instances format and direct requests.
    
    Returns job_id immediately. Video generates in background.
    Poll /v1/jobs/{job_id} to check status and get video URI when complete.
    
    Args:
        request: Raw request (can be instances format or direct)
        
    Returns:
        JobSubmissionResponse with job_id and expected_video_uris
    """
    try:
        body = await request.json()
        
        # Check if this is Vertex AI's instances format
        if isinstance(body, dict) and "instances" in body:
            logger.info("📦 Received Vertex AI instances format")
            instances = body["instances"]
            if not instances or len(instances) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="instances array is empty"
                )
            # Use the first instance
            video_request = VideoGenerationRequest(**instances[0])
        else:
            # Direct format (for rawPredict or direct API calls)
            logger.info("📦 Received direct request format")
            video_request = VideoGenerationRequest(**body)
        
        return await submit_video_generation(video_request)
        
    except ValueError as e:
        logger.error(f"❌ Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "validation_error", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"❌ Request parsing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_request", "message": str(e)}
        )


@router.post(
    "/v1/generate",
    response_model=JobSubmissionResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit video generation job"
)
async def generate_v1(request: VideoGenerationRequest):
    """
    Submit video generation job (async).
    
    Same as /predict, but exposed as versioned API endpoint.
    Returns job_id immediately. Video generates in background.
    
    Args:
        request: VideoGenerationRequest with prompt and parameters
        
    Returns:
        JobSubmissionResponse with job_id and expected_video_uris
    """
    return await submit_video_generation(request)
