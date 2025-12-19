# Open-Sora Serving API for Vertex AI
# Optimized for NVIDIA H100/A100 GPU deployment
# 
# Build stages:
# 1. base: CUDA + system packages (rarely changes, excellent caching)
# 2. python-deps: Open-Sora + Python dependencies (changes on code updates)
# 3. runtime: Application code + startup logic (changes frequently)

# Stage 1: Base System Dependencies
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# System packages and build tools
# These rarely change, so this layer caches well
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    ffmpeg \
    build-essential \
    cmake \
    ninja-build \
    libaio1 \
    libaio-dev \
    liburing-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -s /root/.local/bin/uv /usr/local/bin/uv && \
    ln -s /root/.local/bin/uvx /usr/local/bin/uvx

# Stage 2: Python Dependencies & Open-Sora
FROM base AS python-deps

WORKDIR /app

# Clone Open-Sora repository
# This layer only rebuilds if Open-Sora repo changes
RUN git clone --depth 1 https://github.com/Raghav-56/Open-Sora.git /app

# Install Open-Sora and core dependencies
# This is the most expensive operation, so we do it before copying app code
RUN uv sync --no-dev --no-cache

# Activate the virtual environment created by uv sync
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install API service dependencies
RUN uv add \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    google-cloud-storage \
    loguru \
    "huggingface_hub[cli]" \
    decord \
    imageio \
    imageio-ffmpeg

# Stage 3: Runtime Application
FROM python-deps AS runtime

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    # PyTorch CUDA allocator config optimized for A100
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.8 \
    # A100-specific CUDA optimization flags
    CUDA_LAUNCH_BLOCKING=0 \
    CUDA_CACHE_DISABLE=0 \
    CUDA_CACHE_MAXSIZE=2147483648 \
    # TF32 settings
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
    HF_HOME=/root/.cache/huggingface \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    # API Server
    PORT=8080 \
    ENABLE_GPU_LOGGING=0 \
    # Model weights configuration
    WEIGHT_BUCKET="nannie-opensora-weights-so" \
    MODEL_PATH=/app/ckpts \
    WEIGHT_PREFIX=ckpts/ \
    FORCE_DOWNLOAD=false \
    # Job manager settings
    JOB_RETENTION_SECONDS=3600 \
    MAX_COMPLETED_JOBS=100 \
    # Video generation defaults
    DEFAULT_RESOLUTION=256px \
    DEFAULT_NUM_FRAMES=49 \
    DEFAULT_ASPECT_RATIO=16:9 \
    DEFAULT_MODE=t2v \
    DEFAULT_FPS=24 \
    DEFAULT_NUM_SAMPLES=1 \
    # Generation parameters
    DEFAULT_MOTION_SCORE=4 \
    DEFAULT_NUM_STEPS=50 \
    DEFAULT_GUIDANCE="" \
    DEFAULT_TIMEOUT_SECONDS="" \
    GENERATION_TIMEOUT=1800 \
    BASE_OUTPUT_DIR=/tmp/opensora_outputs \
    # Python path
    PYTHONPATH=/app

# Copy application code
# This changes frequently, so we do it LAST to maximize cache hits
COPY app/ /app/app/

# Copy custom single-GPU configs (shardformer disabled for stability)
COPY app/configs/ /app/configs/diffusion/inference/

# Copy startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Verify app structure at build time
RUN test -f /app/app/main.py && \
    test -f /app/app/opensora/runner.py && \
    test -f /app/configs/diffusion/inference/256px_single_gpu.py && \
    test -f /app/configs/diffusion/inference/768px_single_gpu.py && \
    echo "✅ Critical files verified" || \
    (echo "ERROR: Missing critical application files" && exit 1)

RUN python -c "\
from app.main import app; \
from app.scripts.bootstrap_weights import main; \
from app.api import create_router; \
from app.models import VideoGenerationRequest; \
from app.jobs import JobManager; \
from app.opensora import OpenSoraRunner; \
from app.worker import InferenceWorker; \
from app.utils import upload_video_to_gcs; \
print('✅ All imports successful')\
" || (echo "ERROR: Application code structure validation failed" && exit 1)

# Create directories with appropriate permissions
RUN mkdir -p /app/ckpts /tmp/opensora_outputs /app/samples && \
    chmod 777 /tmp/opensora_outputs /app/samples

# Expose API port
EXPOSE 8080

# Health check with extended start period for model loading
HEALTHCHECK --interval=60s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run startup script (handles weight bootstrap + server start)
ENTRYPOINT ["/app/start.sh"]

# Build Information
LABEL maintainer="d2460361@gmail.com" \
      description="Open-Sora text-to-video API for Vertex AI (Multi-Stage Optimized)" \
      version="1.1.0" \
      gpu="NVIDIA H100/A100" \
      model="Open-Sora v2 (11B parameters)" \
      optimization="multi-stage-build"
