"""
Command builder for Open-Sora torchrun inference.
"""

from pathlib import Path
from typing import List, Optional, Union


class CommandBuilder:
    """Builds torchrun commands for Open-Sora inference."""
    
    @staticmethod
    def build_inference_command(
        config_path: str,
        output_dir: Path,
        prompt_arg: List[str],
        seed: int,
        num_frames: int,
        num_steps: int,
        aspect_ratio: str,
        motion_score: Union[int, str],
        fps: int,
        num_samples: int,
        guidance: Optional[float],
        prompt_refine: bool,
        model_path: Path,
        mode: str,
        nproc: int,
    ) -> List[str]:
        """
        Build complete torchrun command for Open-Sora inference.
        
        Args:
            config_path: Path to Open-Sora config file (relative to opensora_dir)
            output_dir: Directory to save generated videos
            prompt_arg: Either ["--prompt", "text"] or ["--dataset.data-path", "file.csv"]
            seed: Random seed for reproducibility
            num_frames: Number of frames to generate (4k+1 format)
            num_steps: Number of diffusion steps
            aspect_ratio: Video aspect ratio (16:9, 9:16, 1:1, 2.39:1)
            motion_score: Motion intensity (1-5 or "dynamic")
            fps: Frames per second for output video
            num_samples: Number of samples to generate
            guidance: Guidance scale (optional)
            prompt_refine: Whether to enable prompt refinement
            model_path: Path to model weights directory
            mode: Generation mode ("t2i2v" or "t2v")
            nproc: Number of processes (GPUs) to use
            
        Returns:
            Complete command as list of strings
        """
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

        # Optional guidance scale
        if guidance is not None:
            cmd.extend(["--guidance", str(guidance)])

        # Optional prompt refinement
        if prompt_refine:
            cmd.extend(["--prompt_refine", "True"])

        # Explicitly point to weights (avoid relying on cwd)
        cmd.extend([
            "--model.from_pretrained",
            str(model_path / "Open_Sora_v2.safetensors"),
            "--ae.from_pretrained",
            str(model_path / "hunyuan_vae.safetensors"),
            "--t5.from_pretrained",
            str(model_path / "google" / "t5-v1_1-xxl"),
            "--clip.from_pretrained",
            str(model_path / "openai" / "clip-vit-large-patch14"),
        ])

        # t2i2v mode requires Flux models
        if mode == "t2i2v":
            cmd.extend([
                "--img_flux.from_pretrained",
                str(model_path / "flux1-dev.safetensors"),
                "--img_flux_ae.from_pretrained",
                str(model_path / "flux1-dev-ae.safetensors"),
            ])
        
        return cmd
