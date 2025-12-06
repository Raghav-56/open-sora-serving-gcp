"""
Open-Sora v2 API Server for Vertex AI
Async video generation with job queueing and status tracking.
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Union, cast, List

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from app.job_manager import JobManager
from app.opensora_runner import OpenSoraRunner
from app.worker import InferenceWorker

# Open-Sora v2 valid parameters
VALID_RESOLUTIONS = ["256px", "768px"]
VALID_NUM_FRAMES = [17, 33, 49, 65, 81, 97, 113, 129]  # 4k+1 format
VALID_ASPECT_RATIOS = ["16:9", "9:16", "1:1", "2.39:1"]
VALID_MODES = ["t2i2v", "t2v"]


class VideoGenerationRequest(BaseModel):
    """Request model for Open-Sora v2 video generation."""

    prompt: str = Field(
        ...,
        description="Text prompt describing the video to generate",
        min_length=1,
        max_length=2000,
    )

    resolution: str = Field(
        "256px",
        description=(
            "Video resolution: '256px' (fast) or '768px' (high quality)"
        ),
    )

    num_frames: int = Field(
        49,
        description=(
            "Number of frames (4k+1 format): "
            "17, 33, 49, 65, 81, 97, 113, 129"
        ),
    )

    aspect_ratio: str = Field(
        "16:9",
        description="Video aspect ratio: '16:9', '9:16', '1:1', '2.39:1'",
    )

    motion_score: Optional[Union[int, str]] = Field(
        None,
        description="Motion intensity 1-5 or 'dynamic'. None for default (4).",
    )

    seed: Optional[int] = Field(
        None,
        description="Random seed for reproducibility (0-4294967295)",
    )

    mode: str = Field(
        "t2i2v",
        description="Generation mode: 't2i2v' (default) or 't2v'",
    )

    fps: int = Field(
        24,
        description="Output frames per second",
        ge=1,
        le=60,
    )

    num_samples: int = Field(
        1,
        description="Number of samples to generate",
        ge=1,
        le=4,
    )

    guidance: Optional[float] = Field(
        None,
        description="Guidance scale override; None uses config default",
    )

    prompt_refine: bool = Field(
        False,
        description="Enable built-in prompt refinement",
    )

    output_bucket: str = Field(
        ...,
        description="GCS bucket name for video output",
    )

    output_prefix: str = Field(
        "",
        description="GCS path prefix (e.g., 'videos/')",
    )

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v: str) -> str:
        if v not in VALID_RESOLUTIONS:
            raise ValueError(f"resolution must be one of {VALID_RESOLUTIONS}")
        return v

    @field_validator("num_frames")
    @classmethod
    def validate_num_frames(cls, v: int) -> int:
        if v not in VALID_NUM_FRAMES:
            raise ValueError(
                f"num_frames must be one of {VALID_NUM_FRAMES} (4k+1)"
            )
        return v

    @field_validator("aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls, v: str) -> str:
        if v not in VALID_ASPECT_RATIOS:
            raise ValueError(
                f"aspect_ratio must be one of {VALID_ASPECT_RATIOS}"
            )
        return v

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}")
        return v

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < 0 or v > 2**32 - 1):
            raise ValueError("seed must be between 0 and 4294967295")
        return v

    @field_validator("output_bucket")
    @classmethod
    def validate_bucket(cls, v: str) -> str:
        if not v or len(v) < 3:
            raise ValueError("output_bucket must be a valid GCS bucket name")
        return v

    @field_validator("motion_score")
    @classmethod
    def validate_motion_score(
        cls, v: Optional[Union[int, str]]
    ) -> Optional[Union[int, str]]:
        if v is None:
            return v
        if isinstance(v, str):
            if v != "dynamic":
                raise ValueError("motion_score string must be 'dynamic'")
            return v
        if not 1 <= v <= 5:
            raise ValueError("motion_score must be between 1 and 5")
        return v


class JobSubmissionResponse(BaseModel):
    """Response model for job submission."""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status (queued)")
    expected_video_uris: List[str] = Field(
        ..., description="Expected GCS URIs when complete"
    )
    expected_gcs_prefix: str = Field(
        ..., description="GCS prefix where outputs will be written"
    )


class JobStatusResponse(BaseModel):
    """Response model for job status queries."""

    job_id: str
    status: str
    created_at: str
    prompt: str
    resolution: str
    frames: int
    aspect_ratio: str
    motion_score: Optional[Union[int, str]]
    seed: Optional[int]
    mode: str
    fps: int
    num_samples: int
    guidance: Optional[float]
    prompt_refine: bool
    output_bucket: str
    output_prefix: str
    video_uri: Optional[str] = None
    video_uris: Optional[List[str]] = None
    generation_time_seconds: Optional[float] = None
    completed_at: Optional[str] = None
    started_at: Optional[str] = None
    error: Optional[str] = None
    log_tail: Optional[List[str]] = None


class QueueStatusResponse(BaseModel):
    """Response model for queue status."""

    queue_size: int
    currently_processing: Optional[str]
    queued_job_ids: List[str]
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    cancelled_jobs: int


# Global instances (initialized at startup via lifespan)
runner: Optional[OpenSoraRunner] = None
worker: Optional[InferenceWorker] = None
job_manager: Optional[JobManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    
    Handles startup (model loading, worker initialization) and
    shutdown (graceful worker termination) lifecycle events.
    
    See: https://fastapi.tiangolo.com/advanced/events/
    """
    global runner, worker, job_manager
    
    # === STARTUP ===
    logger.info("🚀 Starting Open-Sora API server v1.0.0...")
    
    try:
        # Load model
        model_path = os.getenv("MODEL_PATH", "/app/ckpts")
        logger.info(f"📂 Loading model from: {model_path}")
        runner = OpenSoraRunner(model_path)
        
        # Initialize job manager
        job_manager = JobManager()
        
        # Initialize worker
        worker = InferenceWorker(runner, job_manager)
        
        logger.info("✅ Server ready!")
        logger.info(f"   Device: {runner.device}")
        logger.info("   API version: 1.0.0")
        logger.info(
            "   Endpoints: /predict, /v1/generate, /v1/jobs, /v1/queue"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        raise
    
    yield  # Server is running, handle requests
    
    # === SHUTDOWN ===
    logger.info("🛑 Shutting down Open-Sora API server...")
    if worker:
        worker.shutdown()
    logger.info("✅ Shutdown complete")


# FastAPI Application with lifespan

app = FastAPI(
    title="Open-Sora API",
    description="Text-to-video generation API using Open-Sora model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# Helper Function

async def submit_video_generation(request: VideoGenerationRequest) -> dict:
    """
    Submit video generation job to the queue.
    Used by both /predict (Vertex AI) and /v1/generate (direct API).
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


# API Endpoints

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint for Vertex AI."""
    if runner is None or not runner.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready"
        )
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Open-Sora API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (Vertex AI endpoint)",
            "generate": "/v1/generate (async job submission)",
            "job_status": "/v1/jobs/{job_id}",
            "cancel": "DELETE /v1/jobs/{job_id}",
            "queue": "/v1/queue",
            "docs": "/docs"
        },
        "model_info": runner.get_info() if runner else None
    }


# Vertex AI Endpoint (Required)

@app.post(
    "/predict",
    response_model=JobSubmissionResponse,
    status_code=status.HTTP_200_OK,
    summary="Vertex AI prediction endpoint"
)
async def predict(request: VideoGenerationRequest):
    """
    Vertex AI prediction endpoint (required by Vertex AI deployment).
    
    Returns job_id immediately. Video generates in background.
    Poll /v1/jobs/{job_id} to check status and get video URI when complete.
    
    Args:
        request: VideoGenerationRequest with prompt and parameters
        
    Returns:
        JobSubmissionResponse with job_id and expected_video_uri
    """
    return await submit_video_generation(request)


# API v1 Endpoints

@app.post(
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
        JobSubmissionResponse with job_id and expected_video_uri
    """
    return await submit_video_generation(request)


@app.get(
    "/v1/jobs/{job_id}",
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


@app.delete(
    "/v1/jobs/{job_id}",
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


@app.get(
    "/v1/queue",
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


# Exception Handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom handler for HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail if isinstance(exc.detail, dict) else {
            "error": "http_error",
            "message": str(exc.detail)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all handler for unexpected exceptions."""
    logger.error(f"❌ Unhandled exception: {exc}")
    logger.exception(exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_error",
            "message": "An unexpected error occurred",
            "details": {"error_type": type(exc).__name__}
        }
    )


# Main Entry Point (for local testing)

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8080"))
    
    logger.info(f"🚀 Starting server on port {port}...")
    logger.info(f"   Health check: http://localhost:{port}/health")
    logger.info(f"   API docs: http://localhost:{port}/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
