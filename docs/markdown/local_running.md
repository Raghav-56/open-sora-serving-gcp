# Running Locally on GPU VM

The API can be run locally on a GCP VM with A100 or H100 for development and testing.

## Prerequisites

- GCP VM with NVIDIA GPU (A100, H100, or L4)
- Docker with NVIDIA runtime configured
- Model weights in GCS bucket

## Quick Start

### Clone Repository

```bash
git clone https://github.com/Raghav-56/open-sora-serving-gcp.git
cd open-sora-serving-gcp/service
```

### Build Docker Image

```bash
docker build -t opensora-local:dev .
```

### Run Container

```bash
docker run -d \
  --name opensora-api \
  --gpus all \
  -p 8081:8080 \
  -e WEIGHT_BUCKET=YOUR_WEIGHTS_BUCKET \
  -e MODEL_PATH=/app/ckpts \
  -e PORT=8080 \
  opensora-local:dev

# Watch logs (first startup downloads weights, takes ~10-20 min)
docker logs -f opensora-api
```

### Verify Running

```bash
curl http://localhost:8081/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

## Using the Local API

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
  "expected_video_uri": "gs://your-output-bucket/20251202_143022_847_001/20251202_143022_847_001.mp4"
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
  "seed": 42,
  "prompt": "A serene waterfall...",
  "resolution": "256px",
  "frames": 49,
  "generation_time_seconds": 125.4
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

### Mounting Local Weights

If you have weights downloaded locally:

```bash
docker run -d \
  --name opensora-api \
  --gpus all \
  -p 8081:8080 \
  -v /path/to/local/ckpts:/app/ckpts \
  -e MODEL_PATH=/app/ckpts \
  opensora-local:dev
```

### Debugging Inside Container

```bash
docker exec -it opensora-api bash

# Test inference manually
cd /app
python -c "import opensora; print('OK')"

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"
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
