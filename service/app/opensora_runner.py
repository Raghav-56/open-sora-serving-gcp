"""
Model wrapper for video generation.

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
from dataclasses import dataclass
from collections import deque
import csv
import tempfile

import torch
from loguru import logger


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


class OpenSoraRunner:
    """
    Wrapper around Open-Sora v2 for text-to-video generation.
    
    Open-Sora v2 supports:
    - Text-to-video (t2v): Direct text to video generation
    - Text-to-image-to-video (t2i2v): Higher quality via Flux image generation
    - Image-to-video (i2v): Animate a reference image
    
    This wrapper uses t2i2v mode for best quality results.
    """
    
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
    
    VALID_ASPECT_RATIOS = ["16:9", "9:16", "1:1", "2.39:1"]
    VALID_NUM_FRAMES = [17, 33, 49, 65, 81, 97, 113, 129]
    DEFAULT_NUM_FRAMES = 49
    DEFAULT_NUM_STEPS = 50
    DEFAULT_ASPECT_RATIO = "16:9"
    
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
        """Detect and configure GPU."""
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            self.device = "cuda"
            
            logger.info("GPU Configuration:")
            for i in range(self.num_gpus):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"  GPU {i}: {name} ({memory:.1f} GB)")
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
        
        for mode, configs in self.MODE_CONFIGS.items():
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
        num_steps: int = 50,
        motion_score: Union[
            int, str
        ] = 4,  # Supports integers 1-5 or "dynamic"
        mode: str = "t2i2v",
        fps: int = 24,
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
            Tuple of (video_path, seed_used)
        """
        mode = mode.lower()
        if mode not in self.MODE_CONFIGS:
            raise ValueError(
                f"Invalid mode: {mode}. Choose from {list(self.MODE_CONFIGS)}"
            )

        if resolution not in self.MODE_CONFIGS[mode]:
            raise ValueError(
                f"Invalid resolution {resolution} for mode {mode}"
            )
        
        if aspect_ratio not in self.VALID_ASPECT_RATIOS:
            raise ValueError(f"Invalid aspect_ratio: {aspect_ratio}")
        
        if num_frames not in self.VALID_NUM_FRAMES:
            closest = min(
                self.VALID_NUM_FRAMES,
                key=lambda x: abs(x - num_frames),
            )
            logger.warning(f"Adjusting frames {num_frames} -> {closest}")
            num_frames = closest

        if isinstance(motion_score, str):
            if motion_score != "dynamic":
                raise ValueError("motion_score must be 1-5 or 'dynamic'")
        else:
            if not 1 <= int(motion_score) <= 5:
                raise ValueError("motion_score must be between 1 and 5")

        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        
        config_path = self.MODE_CONFIGS[mode][resolution]
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
        
        nproc = max(1, self.num_gpus)

        # If prompt contains cli-like tokens, route via temp CSV to avoid
        # accidental parsing surprises. Otherwise pass directly.
        prompt_arg: List[str] = []
        temp_prompt_file = None
        try:
            if "--" in prompt:
                temp_prompt_file = tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".csv",
                    delete=False,
                    dir=str(output_dir),
                    encoding="utf-8",
                )
                writer = csv.writer(temp_prompt_file)
                writer.writerow(["text"])
                writer.writerow([prompt])
                temp_prompt_file.flush()
                prompt_arg = ["--dataset.data-path", temp_prompt_file.name]
            else:
                prompt_arg = ["--prompt", prompt]
        except Exception:
            if temp_prompt_file is not None:
                temp_prompt_file.close()
            raise
        
        cmd = [
            "torchrun",
            "--nproc_per_node", str(nproc),
            "--standalone",
            "scripts/diffusion/inference.py",
            config_path,
            "--save-dir", str(output_dir),
            *prompt_arg,
            "--seed", str(seed),
            "--sampling_option.seed", str(seed),
            "--sampling_option.num_frames", str(num_frames),
            "--sampling_option.num_steps", str(num_steps),
            "--aspect_ratio", aspect_ratio,
            "--motion-score", str(motion_score),
            "--fps_save", str(fps),
            "--num_sample", str(num_samples),
        ]

        if guidance is not None:
            cmd.extend(["--guidance", str(guidance)])

        if prompt_refine:
            cmd.extend(["--prompt_refine", "True"])

        # Explicitly point to weights to avoid relying on cwd
        cmd.extend([
            "--model.from_pretrained",
            str(self.model_path / "Open_Sora_v2.safetensors"),
            "--ae.from_pretrained",
            str(self.model_path / "hunyuan_vae.safetensors"),
            "--t5.from_pretrained",
            str(self.model_path / "google" / "t5-v1_1-xxl"),
            "--clip.from_pretrained",
            str(self.model_path / "openai" / "clip-vit-large-patch14"),
        ])

        if mode == "t2i2v":
            cmd.extend([
                "--img_flux.from_pretrained",
                str(self.model_path / "flux1-dev.safetensors"),
                "--img_flux_ae.from_pretrained",
                str(self.model_path / "flux1-dev-ae.safetensors"),
            ])
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.opensora_dir)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        effective_timeout = timeout_seconds or 1800
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(self.opensora_dir),
                env=env,
            )

            if process.stdout is None:
                raise RuntimeError("Failed to capture inference output")

            output_lines = []
            with open(log_path, "a", encoding="utf-8") as log_file:
                for line in process.stdout:
                    output_lines.append(line)
                    log_tail.append(line.rstrip())
                    log_file.write(line)
                    log_file.flush()
                    logger.debug(line.rstrip())

            process.wait(timeout=effective_timeout)
            
            if process.returncode != 0:
                error = "".join(output_lines[-20:])
                logger.error(
                    "Generation failed (return code %s)", process.returncode
                )
                logger.error(f"Stderr: {error}")
                raise RuntimeError(f"Video generation failed: {error[-500:]}")
            
            video_paths = self._find_videos(output_dir)
            if not video_paths:
                raise RuntimeError("No video file generated")
            
            logger.info(f"Video generated: {video_paths}")
            return [str(p) for p in video_paths], seed, list(log_tail)
            
        except subprocess.TimeoutExpired as exc:
            process.kill()
            raise RuntimeError(
                f"Generation timed out ({effective_timeout} sec)"
            ) from exc
        finally:
            if temp_prompt_file is not None:
                try:
                    temp_prompt_file.close()
                    Path(temp_prompt_file.name).unlink(missing_ok=True)
                except Exception:
                    pass
    
    def _find_videos(self, output_dir: Path) -> List[Path]:
        """Collect generated videos sorted by mtime descending."""
        video_files: List[Path] = []
        for ext in ["mp4", "avi", "mov", "webm"]:
            video_files.extend(output_dir.glob(f"**/*.{ext}"))
        return sorted(
            video_files, key=lambda p: p.stat().st_mtime, reverse=True
        )

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
            "resolutions": list(next(iter(self.MODE_CONFIGS.values())).keys()),
            "modes": list(self.MODE_CONFIGS.keys()),
            "aspect_ratios": self.VALID_ASPECT_RATIOS,
            "valid_frame_counts": self.VALID_NUM_FRAMES,
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
