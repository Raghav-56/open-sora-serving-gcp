"""
Core configuration and constants for Open-Sora v2 API.
Centralizes all environment variables and validation constants.
"""

import os
from typing import List

# Video Generation Parameters - Valid Options

VALID_RESOLUTIONS: List[str] = ["256px", "768px"]
VALID_NUM_FRAMES: List[int] = [17, 33, 49, 65, 81, 97, 113, 129]  # 4k+1 format
VALID_ASPECT_RATIOS: List[str] = ["16:9", "9:16", "1:1", "2.39:1"]
VALID_MODES: List[str] = ["t2i2v", "t2v", "t2v_single_gpu"]

# API Request Defaults (from environment)

DEFAULT_RESOLUTION: str = os.getenv("DEFAULT_RESOLUTION", "256px")
DEFAULT_NUM_FRAMES: int = int(os.getenv("DEFAULT_NUM_FRAMES", "49"))
DEFAULT_ASPECT_RATIO: str = os.getenv("DEFAULT_ASPECT_RATIO", "16:9")
DEFAULT_MODE: str = os.getenv("DEFAULT_MODE", "t2i2v")
DEFAULT_FPS: int = int(os.getenv("DEFAULT_FPS", "24"))
DEFAULT_NUM_SAMPLES: int = int(os.getenv("DEFAULT_NUM_SAMPLES", "1"))

# Job Manager Configuration

JOB_RETENTION_SECONDS: int = int(os.getenv("JOB_RETENTION_SECONDS", "3600"))
MAX_COMPLETED_JOBS: int = int(os.getenv("MAX_COMPLETED_JOBS", "100"))

# Open-Sora Runner Configuration

DEFAULT_MOTION_SCORE_ENV: str = os.getenv("DEFAULT_MOTION_SCORE", "4")
DEFAULT_NUM_STEPS_ENV: str | None = os.getenv("DEFAULT_NUM_STEPS")
DEFAULT_GUIDANCE_ENV: str | None = os.getenv("DEFAULT_GUIDANCE")
DEFAULT_FPS_ENV: str | None = os.getenv("DEFAULT_FPS")
DEFAULT_TIMEOUT_SECONDS_ENV: str | None = os.getenv("DEFAULT_TIMEOUT_SECONDS")

# Worker Configuration

GENERATION_TIMEOUT: int = int(os.getenv("GENERATION_TIMEOUT", "1800"))
BASE_OUTPUT_DIR: str = os.getenv("BASE_OUTPUT_DIR", "/tmp/opensora_outputs")

# Model Path Configuration

MODEL_PATH: str = os.getenv("MODEL_PATH", "/app/ckpts")

# Bootstrap Configuration

WEIGHT_BUCKET: str = os.getenv("WEIGHT_BUCKET", "")
WEIGHT_PREFIX: str = os.getenv("WEIGHT_PREFIX", "ckpts/")
FORCE_DOWNLOAD: bool = os.getenv("FORCE_DOWNLOAD", "false").lower() == "true"
