#!/bin/bash
# Open-Sora v2 API - Container Startup
# Nannie AI - Proprietary System
set -euo pipefail
IFS=$'\n\t'

# uv sync creates .venv during docker build, we must use it
export VIRTUAL_ENV=/app/.venv
export PATH="$VIRTUAL_ENV/bin:$PATH"

verify_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "ERROR: Required command '$1' not found on PATH";
        exit 1;
    }
}

# Verify we're using the correct Python
if [ ! -f "$VIRTUAL_ENV/bin/python" ]; then
    echo "ERROR: Virtual environment not found at $VIRTUAL_ENV"
    echo "This indicates a Docker build issue. Rebuild the image."
    exit 1
fi
verify_cmd uv
verify_cmd python

# Signal Handling for Graceful Shutdown
cleanup() {
    echo ""
    echo "Received shutdown signal, cleaning up..."
    # Kill child processes gracefully
    if [ -n "$UVICORN_PID" ]; then
        kill -TERM "$UVICORN_PID" 2>/dev/null || true
        wait "$UVICORN_PID" 2>/dev/null || true
    fi
    # Kill the GPU monitor if it was started
    if [ -n "${NVIDIA_SMI_LOG_PID:-}" ]; then
        kill -TERM "${NVIDIA_SMI_LOG_PID}" 2>/dev/null || true
    fi
    exit 0
}
trap cleanup SIGTERM SIGINT SIGQUIT

# Startup Banner
echo "  Open-Sora v2 API - Container Startup"
echo ""

export PYTHONUNBUFFERED=1
cd /app

# Provide runtime defaults for distributed environment variables only if not provided
# These are intentionally set here rather than in the image so orchestration can override them.
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}
# WORLD_SIZE: intentionally not defaulted here for single-GPU runs.
if [ -n "${WORLD_SIZE:-}" ]; then
    export WORLD_SIZE=${WORLD_SIZE}
fi
export RANK=${RANK:-0}
export LOCAL_RANK=${LOCAL_RANK:-0}


# Step 1: Environment Validation
echo "[1/4] Environment Configuration"
echo "-------------------------------------------------------------"
echo "  Core Settings:"
echo "    WEIGHT_BUCKET        : ${WEIGHT_BUCKET:-<NOT SET - REQUIRED>}"
echo "    WEIGHT_PREFIX        : ${WEIGHT_PREFIX:-ckpts/}"
echo "    MODEL_PATH           : ${MODEL_PATH:-/app/ckpts}"
echo "    FORCE_DOWNLOAD       : ${FORCE_DOWNLOAD:-false}"
echo ""
echo "  Server Settings:"
echo "    PORT                 : ${PORT:-8080}"
echo "    GENERATION_TIMEOUT   : ${GENERATION_TIMEOUT:-1800}s"
echo "  Distributed:"
echo "    MASTER_ADDR          : ${MASTER_ADDR}"
echo "    MASTER_PORT          : ${MASTER_PORT}"
echo "    WORLD_SIZE           : ${WORLD_SIZE:-<NOT SET>}"
echo "    RANK                 : ${RANK}"
echo "    LOCAL_RANK           : ${LOCAL_RANK}"
echo ""
echo "  Job Manager:"
echo "    JOB_RETENTION_SECONDS: ${JOB_RETENTION_SECONDS:-3600}s"
echo "    MAX_COMPLETED_JOBS   : ${MAX_COMPLETED_JOBS:-100}"
echo ""
echo "  Video Defaults:"
echo "    DEFAULT_RESOLUTION   : ${DEFAULT_RESOLUTION:-256px}"
echo "    DEFAULT_NUM_FRAMES   : ${DEFAULT_NUM_FRAMES:-49}"
echo "    DEFAULT_ASPECT_RATIO : ${DEFAULT_ASPECT_RATIO:-16:9}"
echo ""
echo "  Runtime:"
echo "    Python               : $(python --version 2>&1)"
echo "    Which Python         : $(which python 2>&1)"
echo ""

# Check required environment variable
if [ -z "${WEIGHT_BUCKET}" ]; then
    echo "ERROR: WEIGHT_BUCKET environment variable is not set!"
    echo ""
    echo "  This variable must point to a GCS bucket containing:"
    echo "    - ckpts/Open_Sora_v2.safetensors"
    echo "    - ckpts/hunyuan_vae.safetensors"
    echo ""
    echo "  Example: docker run -e WEIGHT_BUCKET=my-weights-bucket ..."
    exit 1
fi

# Step 2: GPU Detection
echo "[2/4] GPU Detection"
echo "-------------------------------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "Unable to query")
    echo "  GPU: $GPU_INFO"
    
    # Check CUDA availability in Python
    CUDA_CHECK=$(uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')" 2>/dev/null || echo "CUDA check failed")
    echo "  $CUDA_CHECK"
    
    # Warn if no GPU detected
    if ! uv run python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo ""
        echo "WARNING: CUDA not available! Video generation will be extremely slow."
        echo "  Ensure container is started with --gpus all"
    fi
else
    echo "  WARNING: nvidia-smi not found. GPU may not be available."
fi
echo ""

# Print more GPU & CUDA details for debug/compatibility checks
if command -v nvidia-smi &> /dev/null; then
    echo "  Full GPU query:"
    nvidia-smi -L || true
    echo "  GPU driver and processes:"
    nvidia-smi --query-gpu=driver_version,cuda_version --format=csv,noheader || true
fi
if command -v nvcc &> /dev/null; then
    echo "  nvcc: $(nvcc --version | tail -n+1 | sed -n '1p')"
else
    echo "  nvcc: not present"
fi
echo "  torch & cuda: $(uv run python -c 'import torch; print(torch.__version__, getattr(torch.version, "cuda", None), torch.cuda.is_available(), torch.cuda.device_count())' 2>/dev/null || echo 'torch-info-failed')"

# Optional: continuous GPU logging using nvidia-smi loop (ENABLE_GPU_LOGGING=1 to enable)
if [ "${ENABLE_GPU_LOGGING:-0}" = "1" ] && command -v nvidia-smi &> /dev/null; then
    echo "  Starting GPU monitor logging to /tmp/nvidia-smi.log (CTRL+C to stop container)"
    (while true; do date --iso-8601=seconds; nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv -l 1 --filename=- -i 0 2>/dev/null | sed -n '1,5p'; sleep 4; done) > /tmp/nvidia-smi.log 2>&1 &
    export NVIDIA_SMI_LOG_PID=$!
fi

# Step 3: Weight Bootstrap
echo "[3/4] Downloading Open-Sora v2 weights from GCS..."
echo "-------------------------------------------------------------"

uv run python -m app.scripts.bootstrap_weights
BOOTSTRAP_EXIT=$?

if [ $BOOTSTRAP_EXIT -ne 0 ]; then
    echo ""
    echo "  ERROR: Weight bootstrap failed! (exit code: $BOOTSTRAP_EXIT)"
    echo ""
    echo "  Cannot start API server without model weights."
    echo ""
    echo "  Troubleshooting checklist:"
    echo "  1. WEIGHT_BUCKET is set correctly: ${WEIGHT_BUCKET}"
    echo "  2. Service account has 'Storage Object Viewer' role"
    echo "  3. Bucket contains Open-Sora v2 weights at:"
    echo "       gs://${WEIGHT_BUCKET}/ckpts/Open_Sora_v2.safetensors"
    echo "       gs://${WEIGHT_BUCKET}/ckpts/hunyuan_vae.safetensors"
    echo "  4. Network connectivity to GCS is working"
    exit 1
fi

echo ""
echo "  Weight bootstrap complete"
echo ""

# Step 4: Start FastAPI Server
echo "[4/4] Starting FastAPI server..."
echo "-------------------------------------------------------------"

PORT=${PORT:-8080}

echo "  Host     : 0.0.0.0"
echo "  Port     : $PORT"
echo "  Health   : http://localhost:$PORT/health"
echo "  API Docs : http://localhost:$PORT/docs"
echo "  Workers  : 1 (GPU memory constraint)"
echo ""
echo "  Server starting... (Ctrl+C to stop)"
echo ""

# Use exec to replace shell process for proper signal handling
exec uv run python -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1 \
    --log-level info \
    --timeout-keep-alive 1200
