"""
Job models for Open-Sora v2 async video generation.
Defines job status states and job data structure.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Union, List


class JobStatus(Enum):
    """Job status states."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Open-Sora v2 video generation job."""

    job_id: str
    status: JobStatus
    prompt: str
    resolution: str
    num_frames: int
    aspect_ratio: str
    motion_score: Optional[Union[int, str]]
    seed: Optional[int]
    num_steps: Optional[int]
    mode: str
    fps: int
    num_samples: int
    guidance: Optional[float]
    prompt_refine: bool
    output_bucket: str
    output_prefix: str

    # Results (populated after completion)
    video_uri: Optional[str] = None
    video_uris: Optional[List[str]] = None
    log_tail: Optional[List[str]] = None
    actual_seed: Optional[int] = None
    error: Optional[str] = None

    # Timestamps
    created_at: float = field(
        default_factory=lambda: datetime.now().timestamp()
    )
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert job to dictionary for API response."""
        base = {
            "job_id": self.job_id,
            "status": self.status.value,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "prompt": self.prompt,
            "resolution": self.resolution,
            "frames": self.num_frames,
            "aspect_ratio": self.aspect_ratio,
            "motion_score": self.motion_score,
            "seed": self.seed,
            "num_steps": self.num_steps,
            "mode": self.mode,
            "fps": self.fps,
            "num_samples": self.num_samples,
            "guidance": self.guidance,
            "prompt_refine": self.prompt_refine,
            "output_bucket": self.output_bucket,
            "output_prefix": self.output_prefix,
        }

        if self.status == JobStatus.COMPLETED:
            gen_time = None
            if self.completed_at and self.started_at:
                gen_time = round(self.completed_at - self.started_at, 2)
            base.update({
                "video_uri": self.video_uri,
                "video_uris": self.video_uris,
                "seed": self.actual_seed,
                "generation_time_seconds": gen_time,
                "completed_at": (
                    datetime.fromtimestamp(self.completed_at).isoformat()
                    if self.completed_at
                    else None
                ),
                "log_tail": self.log_tail,
            })
        elif self.status == JobStatus.FAILED:
            base["error"] = self.error
        elif self.status == JobStatus.PROCESSING:
            base["started_at"] = (
                datetime.fromtimestamp(self.started_at).isoformat()
                if self.started_at
                else None
            )
        elif self.status == JobStatus.CANCELLED:
            if self.completed_at:
                base["completed_at"] = (
                    datetime.fromtimestamp(self.completed_at).isoformat()
                )

        return base
