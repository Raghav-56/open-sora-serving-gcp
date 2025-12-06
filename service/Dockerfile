# Open-Sora Serving API for Vertex AI
# Optimized for NVIDIA H100/A100 GPU deployment

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Environment configuration
# System settings
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/root/.cache/huggingface \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    # API Server (override via docker run -e or deployment config)
    PORT=8080 \
    # Model weights configuration (REQUIRED: set WEIGHT_BUCKET at runtime)
    WEIGHT_BUCKET="" \
    MODEL_PATH=/app/ckpts \
    WEIGHT_PREFIX=ckpts/ \
    FORCE_DOWNLOAD=false \
    # Job manager settings
    JOB_RETENTION_SECONDS=3600 \
    MAX_COMPLETED_JOBS=100 \
    # Video generation defaults (API request defaults)
    DEFAULT_RESOLUTION=256px \
    DEFAULT_NUM_FRAMES=49 \
    DEFAULT_ASPECT_RATIO=16:9 \
    DEFAULT_MODE=t2v \
    DEFAULT_FPS=24 \
    DEFAULT_NUM_SAMPLES=1 \
    # Generation parameters (runner-level defaults)
    DEFAULT_MOTION_SCORE=4 \
    DEFAULT_NUM_STEPS=50 \
    DEFAULT_GUIDANCE="" \
    DEFAULT_TIMEOUT_SECONDS="" \
    GENERATION_TIMEOUT=1800 \
    BASE_OUTPUT_DIR=/tmp/opensora_outputs

# Install system dependencies
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

WORKDIR /app

# Clone Open-Sora repository
RUN git clone --depth 1 https://github.com/Raghav-56/Open-Sora.git /app

# Install Open-Sora and dependencies
RUN uv sync --no-dev --no-cache

# Activate the virtual environment created by uv sync
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install flash-attention (optional, requires build)
# Using --no-build-isolation to use existing CUDA environment
RUN uv add --no-build-isolation flash-attn || echo "Flash attention build skipped (optional)"

# Install tensornvme for high-performance IO (optional)
RUN uv add git+https://github.com/hpcaitech/tensornvme.git || echo "TensorNVMe build skipped (optional)"

# Install API service dependencies using native uv add
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

# Copy application code
COPY app/ /app/app/

# Verify app structure at build time using Python imports
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

# Copy startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Create directories with appropriate permissions
RUN mkdir -p /app/ckpts /tmp/opensora_outputs /app/samples && \
    chmod 777 /tmp/opensora_outputs /app/samples

# Set Python path for module imports
ENV PYTHONPATH=/app

# Expose API port
EXPOSE 8080

# Health check with extended start period for model loading
# - start-period: 10 min grace for ~42GB weight download
# - interval: Check every minute after healthy
# - timeout: 10s for health response
# - retries: 3 failures before marking unhealthy
HEALTHCHECK --interval=60s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run startup script (handles weight bootstrap + server start)
ENTRYPOINT ["/app/start.sh"]

# Build Information
LABEL maintainer="raghavrgupta56@gmail.com" \
      description="Open-Sora text-to-video API for Vertex AI" \
      version="1.0.5" \
      gpu="NVIDIA H100/A100" \
      model="Open-Sora v2 (11B parameters)"
