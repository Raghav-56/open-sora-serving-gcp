"""
Open-Sora v2 runner for text-to-video generation.

Open-Sora v2 Reference:
- Model: 11B parameter text-to-video model
- Resolutions: 256px (fast) and 768px (high quality)
- Frame counts: 4k+1 format (17, 33, 49, 65, 81, 97, 113, 129)
- Aspect ratios: 16:9, 9:16, 1:1, 2.39:1
"""

import os
import random
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Union

from loguru import logger
import torch

from app.opensora.config import (
    OpenSoraConfig,
    MODE_CONFIGS,
    VALID_ASPECT_RATIOS,
    VALID_NUM_FRAMES,
    resolve_runtime_defaults,
)
from app.opensora.command_builder import CommandBuilder
from app.opensora.gpu import detect_gpus
from app.opensora.verifier import verify_installation, verify_weights
from app.opensora.process_runner import run_process_and_tail_logs
from app.opensora.outputs import find_videos


class OpenSoraRunner:
    """
    Wrapper around Open-Sora v2 for text-to-video generation.
    
    Open-Sora v2 supports:
    - Text-to-video (t2v): Direct text to video generation
    - Text-to-image-to-video (t2i2v): Higher quality via Flux image generation
    - Image-to-video (i2v): Animate a reference image
    
    This wrapper uses t2i2v mode for best quality results.
    """
    
    def __init__(self, model_path: str, opensora_dir: Optional[str] = None):
        """Initialize Open-Sora runner with model checkpoint path."""
        self.model_path = Path(model_path)
        # default to container mount but allow override for local development
        self.opensora_dir = (
            Path(opensora_dir) if opensora_dir else Path("/app")
        )
        self.config = OpenSoraConfig()
        self.device = None
        self.num_gpus = 0
        self._ready = False
        self.default_output_dir = Path("/tmp/opensora_outputs")
        
        # GPU detection
        self.device, self.num_gpus = detect_gpus()
        # Verify installation and model weights
        verify_installation(self.opensora_dir)
        verify_weights(self.model_path)
        self._ready = True
    
    
    def generate(
        self,
        prompt: str,
        resolution: str = "256px",
        num_frames: int = 49,
        aspect_ratio: str = "16:9",
        seed: Optional[int] = None,
        num_steps: Optional[int] = None,
        motion_score: Union[int, str] = 4,
        mode: str = "t2i2v",
        fps: Optional[int] = None,
        num_samples: int = 1,
        guidance: Optional[float] = None,
        prompt_refine: bool = False,
        save_dir: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ) -> Tuple[List[str], int, List[str]]:
        """
        Generate video from text prompt using Open-Sora v2.
        
        Args:
            prompt: Text description of the video
            resolution: "256px" or "768px"
            num_frames: Number of frames (4k+1 format, max 129)
            aspect_ratio: "16:9", "9:16", "1:1", or "2.39:1"
            seed: Random seed for reproducibility
            num_steps: Diffusion steps (default 50)
            motion_score: Motion intensity 1-5 or "dynamic" (default 4)
            mode: "t2i2v" (default) or "t2v" (direct text-to-video)
            fps: Frames per second when writing the video
            num_samples: How many samples to produce per prompt (sequentially)
            guidance: Guidance scale override
            prompt_refine: Whether to enable built-in prompt refinement
            save_dir: Output directory override
            timeout_seconds: Override per-job timeout (seconds)
            
        Returns:
            Tuple of (video_paths, seed_used, log_tail)
        """
        mode = mode.lower()
        if mode not in MODE_CONFIGS:
            raise ValueError(
                f"Invalid mode: {mode}. Choose from {list(MODE_CONFIGS)}"
            )

        if resolution not in MODE_CONFIGS[mode]:
            raise ValueError(
                f"Invalid resolution {resolution} for mode {mode}"
            )
        
        if aspect_ratio not in VALID_ASPECT_RATIOS:
            raise ValueError(f"Invalid aspect_ratio: {aspect_ratio}")
        
        if num_frames not in VALID_NUM_FRAMES:
            closest = min(
                VALID_NUM_FRAMES,
                key=lambda x: abs(x - num_frames),
            )
            logger.warning(f"Adjusting frames {num_frames} -> {closest}")
            num_frames = closest

        # Resolve defaults from environment and config
        defaults = resolve_runtime_defaults(self.config)
        resolved_motion_default = defaults["motion_default"]
        resolved_num_steps = defaults["num_steps"]
        resolved_fps = defaults["fps"]
        resolved_timeout = defaults["timeout_seconds"]
        if guidance is None:
            guidance = defaults.get("guidance")

        if motion_score is None:
            if resolved_motion_default == "dynamic":
                motion_score = "dynamic"
            else:
                try:
                    motion_score = int(resolved_motion_default)
                except ValueError:
                    logger.warning(
                        "Invalid DEFAULT_MOTION_SCORE; falling back to 4"
                    )
                    motion_score = 4
        elif isinstance(motion_score, str):
            if motion_score != "dynamic":
                raise ValueError("motion_score must be 1-5 or 'dynamic'")
        else:
            if not 1 <= int(motion_score) <= 5:
                raise ValueError("motion_score must be between 1 and 5")

        # Resolve numeric defaults now that validation is done
        num_steps = num_steps or resolved_num_steps
        fps = fps or resolved_fps
        effective_timeout = timeout_seconds or resolved_timeout or 1800

        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        
        config_path = MODE_CONFIGS[mode][resolution]
        output_dir = Path(save_dir) if save_dir else self.default_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir = output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "torchrun.log"

        
        logger.info("Starting video generation:")
        logger.info(f"  Prompt: {prompt[:80]}...")
        logger.info(
            "  Mode: %s, Resolution: %s, Aspect: %s, FPS: %s, Samples: %s",
            mode,
            resolution,
            aspect_ratio,
            fps,
            num_samples,
        )
        logger.info(
            "  Frames: %s, Steps: %s, Seed: %s, Motion: %s",
            num_frames,
            num_steps,
            seed,
            motion_score,
        )

        nproc = self.num_gpus if self.num_gpus > 0 else 1
        if nproc != 1 and self.num_gpus == 1:
            logger.warning(
                "Detected 1 GPU but computed nproc=%s; forcing nproc=1",
                nproc,
            )
            nproc = 1

        # If prompt contains CLI-like tokens, write it to a temporary CSV
        # inside the output dir and return an explicit dataset argument.
        prompt_arg: List[str] = []
        temp_prompt_file = None
        prompt_arg, temp_prompt_file = CommandBuilder.prepare_prompt_arg(
            prompt, output_dir
        )
        
        # Build command using CommandBuilder
        cmd = CommandBuilder.build_inference_command(
            config_path=config_path,
            output_dir=output_dir,
            prompt_arg=prompt_arg,
            seed=seed,
            num_frames=num_frames,
            num_steps=num_steps,
            aspect_ratio=aspect_ratio,
            motion_score=motion_score,
            fps=fps,
            num_samples=num_samples,
            guidance=guidance,
            prompt_refine=prompt_refine,
            model_path=self.model_path,
            mode=mode,
            nproc=nproc,
        )

        # Normalize the command for single-process runs.
        cmd = CommandBuilder.normalize_command_for_single_gpu(cmd, nproc)
        
        runner_mode = "torchrun"
        if cmd[0] == "python":
            runner_mode = "python"
        logger.info("nproc computed: %s, using %s", nproc, runner_mode)
        short_cmd = " ".join(cmd[:10]) + (" ..." if len(cmd) > 10 else "")
        logger.debug("short execution command: %s", short_cmd)
        # Build the runtime environment from command builder helper
        env = CommandBuilder.build_runtime_env(self.opensora_dir, nproc, os.environ.copy())
        
        try:
            returncode, output_lines, tail_lines = run_process_and_tail_logs(
                cmd=cmd,
                cwd=self.opensora_dir,
                env=env,
                log_path=log_path,
                timeout_seconds=effective_timeout,
            )

            if returncode != 0:
                error = "".join(output_lines[-20:])
                logger.error(
                    "Generation failed (return code %s)", returncode
                )
                logger.error(f"Stderr: {error}")
                raise RuntimeError(f"Video generation failed: {error[-500:]}")

            video_paths = find_videos(output_dir)
            if not video_paths:
                raise RuntimeError("No video file generated")

            logger.info(f"Video generated: {video_paths}")
            return [str(p) for p in video_paths], seed, tail_lines

        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Generation timed out ({effective_timeout} sec)"
            ) from exc
        finally:
            if temp_prompt_file is not None:
                try:
                    Path(temp_prompt_file).unlink(missing_ok=True)
                except Exception:
                    pass
    
    
    def is_ready(self) -> bool:
        """Check if runner is ready for inference."""
        return self._ready

    def get_info(self) -> dict:
        """Get model and GPU information for API responses."""
        info = {
            "device": self.device,
            "model_path": str(self.model_path),
            "ready": self._ready,
            "num_gpus": self.num_gpus,
            "resolutions": list(next(iter(MODE_CONFIGS.values())).keys()),
            "modes": list(MODE_CONFIGS.keys()),
            "aspect_ratios": VALID_ASPECT_RATIOS,
            "valid_frame_counts": VALID_NUM_FRAMES,
        }
        
        if self.device == "cuda" and self.num_gpus > 0:
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / 1e9, 1
                )
            except Exception:
                pass
        
        return info
