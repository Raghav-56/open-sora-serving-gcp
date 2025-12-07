"""
Open-Sora v2 configuration and constants.
"""

import os
from dataclasses import dataclass


# Environment-based defaults for runtime tuning
DEFAULT_MOTION_SCORE_ENV = os.getenv("DEFAULT_MOTION_SCORE", "4")
DEFAULT_NUM_STEPS_ENV = os.getenv("DEFAULT_NUM_STEPS")
DEFAULT_GUIDANCE_ENV = os.getenv("DEFAULT_GUIDANCE")
DEFAULT_FPS_ENV = os.getenv("DEFAULT_FPS")
DEFAULT_TIMEOUT_SECONDS_ENV = os.getenv("DEFAULT_TIMEOUT_SECONDS")


@dataclass
class OpenSoraConfig:
    """Open-Sora v2 model configuration."""
    model_type: str = "flux"
    hidden_size: int = 3072
    num_heads: int = 24
    depth: int = 19
    depth_single_blocks: int = 38
    vae_type: str = "hunyuan_vae"
    latent_channels: int = 16
    default_num_steps: int = 50
    default_guidance: float = 7.5
    default_fps: int = 24


# Open-Sora v2 mode-specific config paths
MODE_CONFIGS = {
    "t2i2v": {
        "256px": "configs/diffusion/inference/t2i2v_256px.py",
        "768px": "configs/diffusion/inference/t2i2v_768px.py",
    },
    "t2v": {
        "256px": "configs/diffusion/inference/256px.py",
        "768px": "configs/diffusion/inference/768px.py",
    },
}

# Valid generation parameters
VALID_ASPECT_RATIOS = ["16:9", "9:16", "1:1", "2.39:1"]
VALID_NUM_FRAMES = [17, 33, 49, 65, 81, 97, 113, 129]

# Default values
DEFAULT_NUM_FRAMES = 49
DEFAULT_NUM_STEPS = 50
DEFAULT_ASPECT_RATIO = "16:9"


def resolve_runtime_defaults(config: OpenSoraConfig) -> dict:
    """
    Resolve environment and config-derived default runtime values.

    Returns a dict with keys:
      - motion_default: str or int
      - num_steps: int
      - fps: int
      - timeout_seconds: Optional[int]
      - guidance: Optional[float]
    """
    resolved_motion_default = DEFAULT_MOTION_SCORE_ENV
    if DEFAULT_NUM_STEPS_ENV and DEFAULT_NUM_STEPS_ENV.isdigit():
        resolved_num_steps = int(DEFAULT_NUM_STEPS_ENV)
    else:
        resolved_num_steps = config.default_num_steps

    if DEFAULT_FPS_ENV and DEFAULT_FPS_ENV.isdigit():
        resolved_fps = int(DEFAULT_FPS_ENV)
    else:
        resolved_fps = config.default_fps

    resolved_timeout = None
    if DEFAULT_TIMEOUT_SECONDS_ENV and DEFAULT_TIMEOUT_SECONDS_ENV.isdigit():
        resolved_timeout = int(DEFAULT_TIMEOUT_SECONDS_ENV)

    guidance = None
    if DEFAULT_GUIDANCE_ENV is not None:
        try:
            guidance = float(DEFAULT_GUIDANCE_ENV)
        except ValueError:
            guidance = None

    return {
        "motion_default": resolved_motion_default,
        "num_steps": resolved_num_steps,
        "fps": resolved_fps,
        "timeout_seconds": resolved_timeout,
        "guidance": guidance,
    }
