# Open-Sora Serving API for Vertex AI
# Optimized for NVIDIA H100/A100 GPU deployment

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Environment configuration
# System settings
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/root/.cache/huggingface \
    # uv package manager settings
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    # API Server (override via docker run -e)
    PORT=8080 \
    # Model weights configuration
    WEIGHT_BUCKET=nannie-opensora-weights-so \
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
    GENERATION_TIMEOUT=1800

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

# Clone Open-Sora repository (Raghav's fork for Nannie AI)
RUN git clone --depth 1 https://github.com/Raghav-56/Open-Sora.git /app

# Install Open-Sora and dependencies using native uv sync
# Using --no-cache to reduce memory usage during build
RUN uv sync --no-dev --no-cache

# Install flash-attention (optional, requires build)
# Using --no-build-isolation to use existing CUDA environment
RUN uv add --no-build-isolation flash-attn || echo "Flash attention build skipped (optional)"

# Install tensornvme for high-performance IO (optional)
RUN uv add git+https://github.com/hpcaitech/tensornvme.git || echo "TensorNVMe build skipped (optional)"

# Install API service dependencies using native uv add
RUN uv add \
    fastapi==0.115.0 \
    "uvicorn[standard]==0.30.6" \
    pydantic==2.9.2 \
    google-cloud-storage==2.18.2 \
    loguru==0.7.2 \
    "huggingface_hub[cli]" \
    decord \
    imageio \
    imageio-ffmpeg

# Copy application code
COPY app/ /app/app/

# Verify app structure at build time
RUN test -f /app/app/main.py && \
    test -f /app/app/opensora_runner.py && \
    test -f /app/app/worker.py && \
    test -f /app/app/bootstrap_weights.py && \
    test -f /app/app/gcs_io.py || \
    (echo "ERROR: Application code incomplete" && exit 1)

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
LABEL maintainer="raghav56.dev@gmail.com" \
      description="Open-Sora text-to-video API for Vertex AI" \
      version="1.0.0" \
      gpu="NVIDIA H100/A100" \
      model="Open-Sora v2 (11B parameters)"
