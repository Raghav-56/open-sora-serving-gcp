"""
Response models for Open-Sora v2 video generation API.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


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
    motion_score: Optional[int | str]
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
