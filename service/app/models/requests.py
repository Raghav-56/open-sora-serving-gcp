"""
Request models for Open-Sora v2 video generation API.
"""

import os
from typing import Optional, Union

from pydantic import BaseModel, Field, field_validator

from app.core.config import (
    VALID_RESOLUTIONS,
    VALID_NUM_FRAMES,
    VALID_ASPECT_RATIOS,
    VALID_MODES,
    DEFAULT_RESOLUTION,
    DEFAULT_NUM_FRAMES,
    DEFAULT_ASPECT_RATIO,
    DEFAULT_MODE,
    DEFAULT_FPS,
    DEFAULT_NUM_SAMPLES,
)


class VideoGenerationRequest(BaseModel):
    """Request model for Open-Sora v2 video generation."""

    prompt: str = Field(
        ...,
        description="Text prompt describing the video to generate",
        min_length=1,
        max_length=2000,
    )

    resolution: str = Field(
        default_factory=lambda: DEFAULT_RESOLUTION,
        description=(
            "Video resolution: '256px' (fast) or '768px' (high quality)"
        ),
    )

    num_frames: int = Field(
        default_factory=lambda: DEFAULT_NUM_FRAMES,
        description=(
            "Number of frames (4k+1 format): "
            "17, 33, 49, 65, 81, 97, 113, 129"
        ),
    )

    aspect_ratio: str = Field(
        default_factory=lambda: DEFAULT_ASPECT_RATIO,
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
        default_factory=lambda: DEFAULT_MODE,
        description="Generation mode: 't2i2v' (default) or 't2v'",
    )

    fps: int = Field(
        default_factory=lambda: DEFAULT_FPS,
        description="Output frames per second",
        ge=1,
        le=60,
    )

    num_samples: int = Field(
        default_factory=lambda: DEFAULT_NUM_SAMPLES,
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
