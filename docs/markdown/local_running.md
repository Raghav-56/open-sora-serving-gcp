# Running Locally

Complete guide for running the Open-Sora API locally for development and testing.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Deployment Scenarios](#deployment-scenarios)
   - [Scenario 1: Docker + GCS Weights (Recommended)](#scenario-1-docker--gcs-weights-recommended)
   - [Scenario 2: Docker + Local Weights](#scenario-2-docker--local-weights)
   - [Scenario 3: Native Python + GCS Weights](#scenario-3-native-python--gcs-weights)
   - [Scenario 4: Native Python + Local Weights + Local Open-Sora](#scenario-4-native-python--local-weights--local-open-sora)
3. [Using the API](#using-the-api)
4. [Development Tips](#development-tips)

---

## Prerequisites

### Hardware Requirements

- NVIDIA GPU (A100, H100, L4, or RTX 4090)
- 80GB+ GPU VRAM (for 768px resolution)
- 40GB+ GPU VRAM (for 256px resolution)
- 100GB+ disk space for model weights

### Software Requirements

**For Docker deployment:**
- Docker 20.10+ with NVIDIA Container Toolkit
- NVIDIA GPU drivers 525.60.13+

**For native Python deployment:**
- Python 3.11
- CUDA 12.1+
- PyTorch 2.5.1+
- UV package manager (recommended) or pip

### Model Weight Options

You can use weights from either:

1. **GCS Bucket** (recommended for production)
   - Requires Google Cloud authentication
   - Automatic download on first startup
   - ~42GB download size

2. **Local Filesystem** (faster for development)
   - Pre-downloaded weights
   - No GCS dependency
   - Mounted into container or referenced directly

---

## Deployment Scenarios

---

## Deployment Scenarios

### Scenario 1: Docker + GCS Weights (Recommended)

**Best for:** Production-like testing, clean environment, automatic weight management

**Setup:**

```bash
# Clone repository
git clone https://github.com/Raghav-56/open-sora-serving-gcp.git
cd open-sora-serving-gcp/service

# Build Docker image
docker build -t opensora-api:dev .

# Run with GCS weights (auto-download on startup)
docker run -d \
  --name opensora-api \
  --gpus all \
  -p 8081:8080 \
  -e WEIGHT_BUCKET=your-weights-bucket \
  -e WEIGHT_PREFIX=ckpts/ \
  -e MODEL_PATH=/app/ckpts \
  -e OUTPUT_BUCKET=your-output-bucket \
  opensora-api:dev

# Monitor startup (weight download takes ~10-20 min first time)
docker logs -f opensora-api
```

**Verify:**

```bash
curl http://localhost:8081/health
```

**Pros:**
- ✅ Clean, isolated environment
- ✅ Production parity
- ✅ Automatic dependency management
- ✅ Easy cleanup

**Cons:**
- ❌ Slower iteration (rebuild for code changes)
- ❌ Initial weight download time

---

### Scenario 2: Docker + Local Weights

**Best for:** Faster iteration, no GCS dependency, offline development

**Prerequisites:**

Download Open-Sora weights locally:

```bash
# Install UV if not already installed
pip install uv

# Clone Open-Sora and download weights
git clone https://github.com/hpcaitech/Open-Sora.git
cd Open-Sora
uv sync
uv run huggingface-cli download hpcai-tech/Open-Sora-v2 --local-dir ./ckpts
```

**Setup:**

```bash
# Build image
cd /path/to/open-sora-serving-gcp/service
docker build -t opensora-api:dev .

# Run with volume-mounted local weights
docker run -d \
  --name opensora-api \
  --gpus all \
  -p 8081:8080 \
  -v /path/to/Open-Sora/ckpts:/app/ckpts:ro \
  -e MODEL_PATH=/app/ckpts \
  -e OUTPUT_BUCKET=your-output-bucket \
  opensora-api:dev

# No WEIGHT_BUCKET needed - skips download
docker logs -f opensora-api
```

**Pros:**
- ✅ No GCS dependency
- ✅ Instant startup (no download)
- ✅ Offline development
- ✅ Weights shared across containers

**Cons:**
- ❌ Manual weight management
- ❌ Still requires rebuild for code changes

---

### Scenario 3: Native Python + GCS Weights

**Best for:** Rapid development, debugging, code changes without rebuild

**Prerequisites:**

```bash
# Install Python 3.11 and UV
# Windows: winget install Python.Python.3.11
# Linux: sudo apt install python3.11 python3.11-venv

pip install uv
```

**Setup:**

```bash
# Clone repository
git clone https://github.com/Raghav-56/open-sora-serving-gcp.git
cd open-sora-serving-gcp/service

# Install dependencies
uv venv
uv pip install -r requirements.txt

# Set environment variables
export WEIGHT_BUCKET=your-weights-bucket
export MODEL_PATH=/tmp/ckpts  # or any local path
export OUTPUT_BUCKET=your-output-bucket
export PORT=8080

# Download weights (one-time)
uv run python -m app.scripts.bootstrap_weights

# Start server
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

**Windows PowerShell:**

```powershell
# Set environment variables
$env:WEIGHT_BUCKET="your-weights-bucket"
$env:MODEL_PATH="D:\temp\ckpts"
$env:OUTPUT_BUCKET="your-output-bucket"
$env:PORT=8080

# Download weights
uv run python -m app.scripts.bootstrap_weights

# Start server
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080
```

**Pros:**
- ✅ Instant code changes (no rebuild)
- ✅ Easy debugging with breakpoints
- ✅ Direct Python environment access
- ✅ Faster iteration cycle

**Cons:**
- ❌ Requires proper Python environment setup
- ❌ Manual dependency management
- ❌ Less isolated (potential conflicts)

---

### Scenario 4: Native Python + Local Weights + Local Open-Sora

**Best for:** Deep development, modifying Open-Sora code, full control

**This scenario is for when you want to:**
- Modify the Open-Sora model code itself
- Test custom Open-Sora features
- Develop without any external dependencies

**Prerequisites:**

```bash
# Clone both repositories side-by-side
mkdir ~/dev
cd ~/dev

# Clone Open-Sora (your custom fork or main repo)
git clone https://github.com/Raghav-56/Open-Sora.git
cd Open-Sora

# Install Open-Sora dependencies
uv sync

# Download weights
uv run huggingface-cli download hpcai-tech/Open-Sora-v2 --local-dir ./ckpts

# Clone API service
cd ~/dev
git clone https://github.com/Raghav-56/open-sora-serving-gcp.git
cd open-sora-serving-gcp/service

# Install service dependencies
uv venv
uv pip install -r requirements.txt
```

**Setup Environment:**

```bash
# Linux/Mac
export WEIGHT_BUCKET=""  # Empty = skip download
export MODEL_PATH=/home/user/dev/Open-Sora/ckpts
export OUTPUT_BUCKET=your-output-bucket
export PORT=8080
export OPENSORA_PATH=/home/user/dev/Open-Sora  # Custom Open-Sora location
```

```powershell
# Windows
$env:WEIGHT_BUCKET=""
$env:MODEL_PATH="D:\dev\Open-Sora\ckpts"
$env:OUTPUT_BUCKET="your-output-bucket"
$env:PORT=8080
$env:OPENSORA_PATH="D:\dev\Open-Sora"
```

**Modify Runner (Optional):**

If using custom Open-Sora code, update `app/opensora/runner.py`:

```python
# Add custom Open-Sora path to Python path
import sys
import os

custom_opensora = os.getenv("OPENSORA_PATH")
if custom_opensora:
    sys.path.insert(0, custom_opensora)
```

**Start Server:**

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

**Pros:**
- ✅ Full control over both API and model
- ✅ Modify Open-Sora code on the fly
- ✅ No external dependencies (GCS, Docker)
- ✅ Hot reload with `--reload` flag
- ✅ Best debugging experience

**Cons:**
- ❌ Most complex setup
- ❌ Requires managing two repositories
- ❌ Manual environment configuration
- ❌ Less production parity

---

## Scenario Comparison

| Feature | Docker + GCS | Docker + Local | Native + GCS | Native + Local + Custom |
|---------|--------------|----------------|--------------|-------------------------|
| **Setup Time** | 🟡 Medium | 🟢 Fast | 🟡 Medium | 🔴 Slow |
| **Iteration Speed** | 🔴 Slow | 🔴 Slow | 🟢 Fast | 🟢 Fast |
| **Production Parity** | 🟢 High | 🟢 High | 🟡 Medium | 🔴 Low |
| **Debugging** | 🔴 Hard | 🔴 Hard | 🟢 Easy | 🟢 Easy |
| **Isolation** | 🟢 High | 🟢 High | 🟡 Medium | 🔴 Low |
| **GCS Dependency** | ✅ Yes | ❌ No | ✅ Yes | ❌ No |
| **Custom Model Code** | ❌ No | ❌ No | ❌ No | ✅ Yes |
| **Best For** | Testing | Dev | Rapid Dev | Research |

---

## Using the API

All scenarios expose the same API endpoints on `http://localhost:8081` (or your configured port).

### Generate Video

```bash
curl -X POST http://localhost:8081/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene waterfall in a lush forest, sunlight filtering through trees",
    "resolution": "256px",
    "num_frames": 49,
    "output_bucket": "your-output-bucket"
  }'
```

Response:

```json
{
  "job_id": "20251202_143022_847_001",
  "status": "queued",
  "expected_video_uris": [
    "gs://your-output-bucket/20251202_143022_847_001/20251202_143022_847_001.mp4"
  ],
  "expected_gcs_prefix": "gs://your-output-bucket/20251202_143022_847_001/"
}
```

### Check Job Status

```bash
curl http://localhost:8081/v1/jobs/20251202_143022_847_001
```

Response (while processing):

```json
{
  "job_id": "20251202_143022_847_001",
  "status": "processing",
  "created_at": "2025-12-02T14:30:22Z",
  "started_at": "2025-12-02T14:30:25Z"
}
```

Response (when completed):

```json
{
  "job_id": "20251202_143022_847_001",
  "status": "completed",
  "video_uri": "gs://your-output-bucket/20251202_143022_847_001/20251202_143022_847_001.mp4",
  "video_uris": [
    "gs://your-output-bucket/20251202_143022_847_001/20251202_143022_847_001.mp4"
  ],
  "seed": 42,
  "prompt": "A serene waterfall...",
  "resolution": "256px",
  "frames": 49,
  "generation_time_seconds": 125.4,
  "log_tail": ["...last log lines..."]
}
```

### Check Queue Status

```bash
curl http://localhost:8081/v1/queue
```

Response:

```json
{
  "queue_size": 2,
  "currently_processing": "20251202_143022_847_001",
  "queued_job_ids": ["20251202_143025_123_001", "20251202_143028_456_001"],
  "total_jobs": 10,
  "completed_jobs": 5,
  "failed_jobs": 0
}
```

### Cancel a Job

```bash
curl -X DELETE http://localhost:8081/v1/jobs/20251202_143025_123_001
```

---

## Local Endpoints vs Vertex AI

**Local API has full endpoint access:**

| Endpoint | Local | Vertex AI |
|----------|-------|-----------|
| `/health` | ✅ | ✅ |
| `/predict` | ✅ | ✅ |
| `/v1/generate` | ✅ | ❌ |
| `/v1/jobs/{id}` | ✅ | ❌ |
| `/v1/queue` | ✅ | ❌ |
| `/docs` | ✅ | ❌ |

When deployed on Vertex AI, only `/predict` and `/health` are accessible externally.

---

## Development Tips

### Hot Reload for Native Python

When running natively, use `--reload` for automatic code reloading:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

Changes to Python files will automatically restart the server.

### Live Code Changes in Docker

For Docker with live code updates, mount the service directory:

```bash
docker run -d \
  --name opensora-api \
  --gpus all \
  -p 8081:8080 \
  -v $(pwd)/app:/app/app:ro \
  -e WEIGHT_BUCKET=your-bucket \
  opensora-api:dev
```

Restart container after code changes: `docker restart opensora-api`

### Mounting Local Weights

Reuse downloaded weights across runs:

```bash
docker run -d \
  --name opensora-api \
  --gpus all \
  -p 8081:8080 \
  -v /path/to/local/ckpts:/app/ckpts:ro \
  -e MODEL_PATH=/app/ckpts \
  opensora-api:dev
```

### Skip Weight Download

Set empty `WEIGHT_BUCKET` to skip GCS download:

```bash
# Docker
-e WEIGHT_BUCKET="" -e MODEL_PATH=/app/ckpts

# Native
export WEIGHT_BUCKET=""
export MODEL_PATH=/path/to/ckpts
```

### Test Different Resolutions

```bash
# Fast 256px testing
curl -X POST http://localhost:8081/v1/generate \
  -d '{"prompt": "test", "resolution": "256px", "num_frames": 17, "output_bucket": "test"}'

# High quality 768px
curl -X POST http://localhost:8081/v1/generate \
  -d '{"prompt": "test", "resolution": "768px", "num_frames": 49, "output_bucket": "test"}'
```

### Monitor GPU Usage

```bash
# Watch GPU memory and utilization
watch -n 1 nvidia-smi

# Detailed GPU stats
nvidia-smi dmon -s pucvmet
```

### Debugging Inside Container

```bash
docker exec -it opensora-api bash

# Test imports
cd /app
python -c "from app.opensora.runner import OpenSoraRunner; print('Runner OK')"
python -c "from app.core.config import get_config; print('Config OK')"

# Check GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

# Test model loading
python -c "from app.opensora.runner import OpenSoraRunner; runner = OpenSoraRunner(); print(f'Ready: {runner.is_ready()}')"
```

### Python Interactive Debugging

```bash
# Native Python: Add breakpoint in code
import pdb; pdb.set_trace()

# Or use IPython for better REPL
pip install ipython
from IPython import embed; embed()
```

### Access FastAPI Interactive Docs

Navigate to `http://localhost:8081/docs` for Swagger UI with interactive API testing.

### Common Issues

**Issue: CUDA out of memory**

```bash
# Reduce batch size or use 256px resolution
export DEFAULT_RESOLUTION=256px
export DEFAULT_NUM_FRAMES=17
```

**Issue: Weights not found**

```bash
# Verify weight path
docker exec -it opensora-api ls -lh /app/ckpts/

# Force re-download
-e FORCE_DOWNLOAD=true
```

**Issue: GCS permission denied**

```bash
# Authenticate gcloud
gcloud auth application-default login

# Docker: Mount credentials
-v ~/.config/gcloud:/root/.config/gcloud:ro
```

**Issue: Port already in use**

```bash
# Use different port
docker run -p 9000:8080 ...  # Access via localhost:9000
```

### Viewing Container Logs

```bash
# Follow logs
docker logs -f opensora-api

# Last 100 lines
docker logs --tail 100 opensora-api
```

### Stopping and Removing Container

```bash
docker stop opensora-api
docker rm opensora-api
```
