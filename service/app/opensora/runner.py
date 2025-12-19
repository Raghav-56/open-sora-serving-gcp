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
from collections import deque

import torch
from loguru import logger

from app.opensora.config import (
    OpenSoraConfig,
    MODE_CONFIGS,
    VALID_ASPECT_RATIOS,
    VALID_NUM_FRAMES,
    resolve_runtime_defaults,
)
from app.opensora.command_builder import CommandBuilder


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
        
        self._setup_gpu()
        self._verify_installation()
    
    def _setup_gpu(self):
        """Detect and configure GPU with A100-specific optimizations."""
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            self.device = "cuda"
            
            logger.info("GPU Configuration:")
            for i in range(self.num_gpus):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"  GPU {i}: {name} ({memory:.1f} GB)")
            # Production optimizations for inference
            # Enable TF32 for A100 GPUs (faster matrix multiplications)
            try:
                if any("A100" in torch.cuda.get_device_name(i) for i in range(self.num_gpus)):
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    logger.info("  Enabled TF32 for A100 GPUs")
            except Exception:
                logger.debug("Could not determine GPU model for TF32 optimization")

            # Set optimal CUDNN settings for inference
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            # Clear cached CUDA memory to reduce fragmentation
            torch.cuda.empty_cache()
            logger.info("  Configured CUDA for production inference")
        else:
            self.device = "cpu"
            logger.warning(
                "No GPU detected - inference will be extremely slow"
            )
            
    def _verify_installation(self):
        """Verify Open-Sora installation and model weights."""
        logger.info("Verifying Open-Sora installation...")
        
        inference_script = self.opensora_dir / "scripts/diffusion/inference.py"
        if not inference_script.exists():
            raise FileNotFoundError(
                f"Open-Sora inference script not found at {inference_script}"
            )
        
        for mode, configs in MODE_CONFIGS.items():
            for resolution, config_path in configs.items():
                full_path = self.opensora_dir / config_path
                if not full_path.exists():
                    raise FileNotFoundError(
                        (
                            f"Config for {mode} {resolution} "
                            f"not found at {full_path}"
                        )
                    )
        
        logger.info("Open-Sora code verified")
        self._verify_weights()
        self._ready = True
    
    def _verify_weights(self):
        """Verify model checkpoint files exist."""
        logger.info(f"Verifying weights at: {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model path not found: {self.model_path}. "
                "Run bootstrap_weights.py first."
            )

        required_paths = {
            "Main checkpoint": self.model_path / "Open_Sora_v2.safetensors",
            "VAE checkpoint": self.model_path / "hunyuan_vae.safetensors",
            "Flux t2i2v checkpoint": self.model_path / "flux1-dev.safetensors",
            "Flux AE": self.model_path / "flux1-dev-ae.safetensors",
            "T5 encoder": self.model_path / "google" / "t5-v1_1-xxl",
            "CLIP encoder": (
                self.model_path / "openai" / "clip-vit-large-patch14"
            ),
        }

        missing = []
        found = []

        for label, path in required_paths.items():
            if path.is_file():
                size_gb = path.stat().st_size / 1e9
                found.append(f"{label} ({size_gb:.1f} GB)")
            elif path.is_dir():
                found.append(f"{label} directory")
            else:
                missing.append(f"{label}: {path}")

        if found:
            logger.info(f"Found: {', '.join(found)}")

        if missing:
            raise FileNotFoundError(
                "Missing Open-Sora weights: "
                + "; ".join(missing)
                + ". Run bootstrap_weights.py to download."
            )
    
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

        # Bounded tail buffer for returning last N lines to callers
        log_tail: deque[str] = deque(maxlen=200)
        
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
        if nproc != 1 and torch.cuda.device_count() == 1:
            logger.warning(
                "Detected 1 GPU but computed nproc=%s; forcing nproc=1",
                nproc,
            )
            nproc = 1

        # Prepare prompt argument using helper
        prompt_arg, temp_prompt_file = CommandBuilder.prepare_prompt_arg(
            prompt, output_dir
        )
        
        # Build command using CommandBuilder
        cmd = CommandBuilder.build_inference_command(
            config_path=config_path,
            output_dir=output_dir,
            prompt_arg=prompt_arg,
            seed=seed,
            resolution=resolution,
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

        # Normalize command for single-GPU runs
        cmd = CommandBuilder.normalize_command_for_single_gpu(cmd, nproc)
        
        runner_mode = "torchrun"
        if cmd[0] == "python":
            runner_mode = "python"
        logger.info("nproc computed: %s, using %s", nproc, runner_mode)
        short_cmd = " ".join(cmd[:10]) + (" ..." if len(cmd) > 10 else "")
        logger.debug("short execution command: %s", short_cmd)
        # Build runtime environment
        env = CommandBuilder.build_runtime_env(
            self.opensora_dir, nproc, os.environ.copy()
        )
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.opensora_dir),
                env=env,
                start_new_session=True,
            )

            if process.stdout is None:
                raise RuntimeError("Failed to capture inference output")

            output_lines = []
            stderr_lines = []
            with open(log_path, "a", encoding="utf-8") as log_file:
                # Read stdout
                for line in process.stdout:
                    output_lines.append(line)
                    log_tail.append(line.rstrip())
                    log_file.write(line)
                    log_file.flush()
                    logger.debug(line.rstrip())

            process.wait(timeout=effective_timeout)
            
            # Capture any stderr after process completes
            if process.stderr:
                stderr_output = process.stderr.read()
                if stderr_output:
                    stderr_lines = stderr_output.splitlines(keepends=True)
            
            # Check for video FIRST, even if process returned non-zero
            # (Open-Sora may crash at dist.barrier() after saving video)
            video_paths = self._find_videos(output_dir)
            
            if video_paths:
                logger.info(f"Video generated: {video_paths}")
                if process.returncode != 0:
                    logger.warning(
                        f"Process exited with code {process.returncode} "
                        "but video was saved"
                    )
                return [str(p) for p in video_paths], seed, list(log_tail)
            
            # No video found and non-zero exit code
            if process.returncode != 0:
                # Capture ALL output - no truncation
                full_output = "".join(output_lines)
                if stderr_lines:
                    full_stderr = "".join(stderr_lines)
                else:
                    full_stderr = "(no stderr)"
                
                # Log EVERYTHING
                logger.error("="*60)
                logger.error(
                    "Generation failed with return code %s",
                    process.returncode
                )
                logger.error("COMPLETE STDOUT:")
                logger.error(full_output)
                logger.error("COMPLETE STDERR:")
                logger.error(full_stderr)
                logger.error("="*60)
                
                # Include full output in exception message too
                full_error = (
                    f"Exit code {process.returncode}\n\n"
                    f"STDOUT:\n{full_output}\n\n"
                    f"STDERR:\n{full_stderr}"
                )
                raise RuntimeError(
                    f"Video generation failed:\n{full_error}"
                )
            
            # Video not found but exit code was 0
            raise RuntimeError("No video file generated")
            
        except subprocess.TimeoutExpired as exc:
            process.kill()
            raise RuntimeError(
                f"Generation timed out ({effective_timeout} sec)"
            ) from exc
        finally:
            if temp_prompt_file is not None:
                try:
                    Path(temp_prompt_file).unlink(missing_ok=True)
                except Exception:
                    pass
    
    def _find_videos(self, output_dir: Path) -> List[Path]:
        """Collect generated videos sorted by mtime descending."""
        video_files: List[Path] = []
        for ext in ["mp4", "avi", "mov", "webm"]:
            video_files.extend(output_dir.glob(f"**/*.{ext}"))
        
        # Sort by modification time (newest first)
        video_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return video_files
    
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
