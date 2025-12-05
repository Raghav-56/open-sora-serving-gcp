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
from typing import Optional, Tuple
from dataclasses import dataclass

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
    
    RESOLUTION_CONFIGS = {
        "256px": "configs/diffusion/inference/t2i2v_256px.py",
        "768px": "configs/diffusion/inference/t2i2v_768px.py",
    }
    
    VALID_ASPECT_RATIOS = ["16:9", "9:16", "1:1", "2.39:1"]
    VALID_NUM_FRAMES = [17, 33, 49, 65, 81, 97, 113, 129]
    DEFAULT_NUM_FRAMES = 49
    DEFAULT_NUM_STEPS = 50
    DEFAULT_ASPECT_RATIO = "16:9"
    
    def __init__(self, model_path: str):
        """Initialize Open-Sora runner with model checkpoint path."""
        self.model_path = Path(model_path)
        self.opensora_dir = Path("/app")
        self.config = OpenSoraConfig()
        self.device = None
        self.num_gpus = 0
        self._ready = False
        
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
            logger.warning("No GPU detected - inference will be extremely slow")
    
    def _verify_installation(self):
        """Verify Open-Sora installation and model weights."""
        logger.info("Verifying Open-Sora installation...")
        
        inference_script = self.opensora_dir / "scripts/diffusion/inference.py"
        if not inference_script.exists():
            raise FileNotFoundError(
                f"Open-Sora inference script not found at {inference_script}"
            )
        
        for resolution, config_path in self.RESOLUTION_CONFIGS.items():
            full_path = self.opensora_dir / config_path
            if not full_path.exists():
                raise FileNotFoundError(
                    f"Config for {resolution} not found at {full_path}"
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
        
        main_ckpt = self.model_path / "Open_Sora_v2.safetensors"
        vae_ckpt = self.model_path / "hunyuan_vae.safetensors"
        
        found = []
        for ckpt in [main_ckpt, vae_ckpt]:
            if ckpt.exists():
                size_gb = ckpt.stat().st_size / 1e9
                found.append(f"{ckpt.name} ({size_gb:.1f} GB)")
        
        t5_dir = self.model_path / "google" / "t5-v1_1-xxl"
        clip_dir = self.model_path / "openai" / "clip-vit-large-patch14"
        
        if t5_dir.exists():
            found.append("T5 encoder")
        if clip_dir.exists():
            found.append("CLIP encoder")
        
        if found:
            logger.info(f"Found: {', '.join(found)}")
        
        if not main_ckpt.exists():
            raise FileNotFoundError("Main checkpoint not found")
    
    def generate(
        self,
        prompt: str,
        resolution: str = "256px",
        num_frames: int = 49,
        aspect_ratio: str = "16:9",
        seed: Optional[int] = None,
        num_steps: int = 50,
        motion_score: int = 4,  # Default from Open-Sora config
    ) -> Tuple[str, int]:
        """
        Generate video from text prompt using Open-Sora v2.
        
        Args:
            prompt: Text description of the video
            resolution: "256px" or "768px"
            num_frames: Number of frames (4k+1 format, max 129)
            aspect_ratio: "16:9", "9:16", "1:1", or "2.39:1"
            seed: Random seed for reproducibility
            num_steps: Diffusion steps (default 50)
            motion_score: Motion intensity 1-5 (default 4)
            
        Returns:
            Tuple of (video_path, seed_used)
        """
        if resolution not in self.RESOLUTION_CONFIGS:
            raise ValueError(f"Invalid resolution: {resolution}")
        
        if aspect_ratio not in self.VALID_ASPECT_RATIOS:
            raise ValueError(f"Invalid aspect_ratio: {aspect_ratio}")
        
        if num_frames not in self.VALID_NUM_FRAMES:
            closest = min(self.VALID_NUM_FRAMES, key=lambda x: abs(x - num_frames))
            logger.warning(f"Adjusting frames {num_frames} -> {closest}")
            num_frames = closest
        
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        
        config_path = self.RESOLUTION_CONFIGS[resolution]
        output_dir = Path("/tmp/opensora_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting video generation:")
        logger.info(f"  Prompt: {prompt[:80]}...")
        logger.info(f"  Resolution: {resolution}, Aspect: {aspect_ratio}")
        logger.info(f"  Frames: {num_frames}, Steps: {num_steps}, Seed: {seed}")
        
        nproc = max(1, self.num_gpus)
        
        cmd = [
            "torchrun",
            "--nproc_per_node", str(nproc),
            "--standalone",
            "scripts/diffusion/inference.py",
            config_path,
            "--save-dir", str(output_dir),
            "--prompt", prompt,
            "--seed", str(seed),
            "--sampling_option.seed", str(seed),
            "--sampling_option.num_frames", str(num_frames),
            "--sampling_option.num_steps", str(num_steps),
            "--aspect_ratio", aspect_ratio,
            "--motion-score", str(motion_score),
        ]
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.opensora_dir)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(self.opensora_dir),
                env=env,
            )
            
            output_lines = []
            for line in process.stdout:
                output_lines.append(line)
            
            process.wait(timeout=1800)
            
            if process.returncode != 0:
                error = "".join(output_lines[-20:])
                logger.error(f"Generation failed: {error[:500]}")
                raise RuntimeError(f"Generation failed: {error[:500]}")
            
            video_path = self._find_latest_video(output_dir)
            if not video_path:
                raise RuntimeError("No video file generated")
            
            logger.info(f"Video generated: {video_path}")
            return str(video_path), seed
            
        except subprocess.TimeoutExpired:
            process.kill()
            raise RuntimeError("Generation timed out (30 min)")
    
    def _find_latest_video(self, output_dir: Path) -> Optional[Path]:
        """Find the most recently created video file."""
        video_files = []
        for ext in ["mp4", "avi", "mov", "webm"]:
            video_files.extend(output_dir.glob(f"**/*.{ext}"))
        
        return max(video_files, key=lambda p: p.stat().st_mtime) if video_files else None

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
            "resolutions": list(self.RESOLUTION_CONFIGS.keys()),
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
