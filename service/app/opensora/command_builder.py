"""
Command builder for Open-Sora torchrun inference.
"""

from pathlib import Path
from typing import List, Optional, Union, Tuple
import tempfile
import csv
import os


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

    @staticmethod
    def prepare_prompt_arg(
        prompt: str, output_dir: Path
    ) -> Tuple[List[str], Optional[str]]:
        """
        If prompt contains CLI-style tokens that could confuse parsing
        we write it to a temporary CSV file and return an arg to point the
        CLI at that file. Returns (prompt_arg_list, temp_file_path or None).
        The caller is responsible for removing the temp file if provided.
        """
        if "--" in prompt:
            # Use a temporary file inside output_dir so it's easy for callers to locate
            fd, path = tempfile.mkstemp(suffix=".csv", dir=str(output_dir), text=True)
            os.close(fd)
            with open(path, "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["text"])
                writer.writerow([prompt])
            return ["--dataset.data-path", path], path
        return ["--prompt", prompt], None

    @staticmethod
    def normalize_command_for_single_gpu(
        cmd: List[str], nproc: int
    ) -> List[str]:
        """
        If the command uses torchrun and there's only a single GPU process,
        convert it into a direct python invocation for a simpler single-process run.
        """
        if nproc != 1 or not cmd or cmd[0] != "torchrun":
            return cmd

        filtered = []
        i = 0
        while i < len(cmd):
            if cmd[i] == "torchrun":
                i += 1
                continue
            if cmd[i] in ("--nproc_per_node",):
                i += 2
                continue
            if cmd[i] == "--standalone":
                i += 1
                continue
            filtered.append(cmd[i])
            i += 1
        if filtered and filtered[0].endswith(".py"):
            filtered = ["python"] + filtered
        return filtered

    @staticmethod
    def build_runtime_env(
        opensora_dir: Path, nproc: int, base_env: Optional[dict] = None
    ) -> dict:
        """
        Build a suitable environment dict for running the inference command.
        """
        env = dict(base_env) if base_env else os.environ.copy()
        env["PYTHONPATH"] = str(opensora_dir)
        env["PYTORCH_CUDA_ALLOC_CONF"] = (
            "expandable_segments:True,max_split_size_mb:512"
        )

        if nproc == 1:
            env.setdefault("CUDA_VISIBLE_DEVICES", "0")
            env.setdefault("MASTER_ADDR", "localhost")
            env.setdefault("MASTER_PORT", "29500")
            env.setdefault("RANK", "0")
            env.setdefault("LOCAL_RANK", "0")
            env.setdefault("NCCL_DEBUG", "WARN")
            env.setdefault("NCCL_TIMEOUT", "1800")
            env.setdefault("TORCH_NCCL_BLOCKING_WAIT", "0")
            env.setdefault("COLOSSALAI_LAZY_INIT", "1")
        else:
            env.setdefault("WORLD_SIZE", str(nproc))
            env.setdefault("MASTER_ADDR", env.get("MASTER_ADDR", "localhost"))
            env.setdefault("MASTER_PORT", env.get("MASTER_PORT", "29500"))
        return env
