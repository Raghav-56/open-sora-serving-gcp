# Code Reference Documentation

This guide provides detailed documentation for the Open-Sora v2 API.

Nannie AI - Proprietary System

## Table of Contents

1. [API Endpoints](#api-endpoints)
2. [Request/Response Examples](#requestresponse-examples)
3. [Source Files](#source-files)

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check for Vertex AI |
| GET | `/` | API information |
| POST | `/predict` | Vertex AI prediction endpoint |
| POST | `/v1/generate` | Submit video generation job |
| GET | `/v1/jobs/{job_id}` | Get job status |
| DELETE | `/v1/jobs/{job_id}` | Cancel job |
| GET | `/v1/queue` | Queue status and statistics |

---

## Request/Response Examples

### POST `/predict` or `/v1/generate`

Submit a video generation job.

**Request:**

```json
{
  "prompt": "A cat playing piano in a jazz club, cinematic lighting",
  "resolution": "256px",
  "num_frames": 49,
  "aspect_ratio": "16:9",
  "motion_score": 6,
  "seed": 42,
  "output_bucket": "my-output-bucket",
  "output_prefix": "videos/"
}
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | Text description (1-2000 chars) |
| `resolution` | string | No | `"256px"` | `"256px"` or `"768px"` |
| `num_frames` | int | No | `49` | 17, 33, 49, 65, 81, 97, 113, 129 (4k+1 format) |
| `aspect_ratio` | string | No | `"16:9"` | `"16:9"`, `"9:16"`, `"1:1"`, `"2.39:1"` |
| `motion_score` | int | No | `4` | 1-10 (4-6 recommended for natural motion) |
| `seed` | int | No | auto | 0 to 4294967295 |
| `output_bucket` | string | Yes | - | GCS bucket name |
| `output_prefix` | string | No | `""` | Path prefix in bucket |

**Response:**

```json
{
  "job_id": "20251205_143052_847_001",
  "status": "queued",
  "expected_video_uri": "gs://my-output-bucket/videos/20251205_143052_847_001.mp4"
}
```

---

### GET `/v1/jobs/{job_id}`

Get status of a video generation job.

**Response (queued):**

```json
{
  "job_id": "20251205_143052_847_001",
  "status": "queued",
  "created_at": "2025-12-05T14:30:52.847000"
}
```

**Response (processing):**

```json
{
  "job_id": "20251205_143052_847_001",
  "status": "processing",
  "created_at": "2025-12-05T14:30:52.847000",
  "started_at": "2025-12-05T14:30:55.123000"
}
```

**Response (completed):**

```json
{
  "job_id": "20251205_143052_847_001",
  "status": "completed",
  "created_at": "2025-12-05T14:30:52.847000",
  "video_uri": "gs://my-output-bucket/videos/20251205_143052_847_001/20251205_143052_847_001.mp4",
  "seed": 42,
  "prompt": "A cat playing piano in a jazz club, cinematic lighting",
  "resolution": "256px",
  "frames": 49,
  "aspect_ratio": "16:9",
  "generation_time_seconds": 145.32,
  "completed_at": "2025-12-05T14:33:18.167000"
}
```

**Response (failed):**

```json
{
  "job_id": "20251205_143052_847_001",
  "status": "failed",
  "created_at": "2025-12-05T14:30:52.847000",
  "error": "CUDA out of memory"
}
```

**Status Values:**

| Status | Description |
|--------|-------------|
| `queued` | Job is waiting in queue |
| `processing` | Video is being generated |
| `completed` | Video ready (`video_uri` available) |
| `failed` | Generation failed (`error` available) |
| `cancelled` | Job was cancelled |

---

### DELETE `/v1/jobs/{job_id}`

Cancel a video generation job.

**Response:**

```json
{
  "message": "Job cancelled successfully"
}
```

**Notes:**

- Queued jobs: Removed from queue immediately
- Processing jobs: Marked for cancellation (cannot stop mid-generation)
- Completed/failed jobs: Cannot be cancelled

---

### GET `/v1/queue`

Get current queue status and statistics.

**Response:**

```json
{
  "queue_size": 3,
  "currently_processing": "20251205_143052_847_001",
  "queued_job_ids": [
    "20251205_143055_123_001",
    "20251205_143058_456_001",
    "20251205_143101_789_001"
  ],
  "total_jobs": 15,
  "completed_jobs": 10,
  "failed_jobs": 1,
  "cancelled_jobs": 0
}
```

---

### GET `/health`

Health check endpoint for Vertex AI.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "timestamp": "2025-12-05T14:30:52.847000"
}
```

---

### GET `/`

Root endpoint with API information.

**Response:**

```json
{
  "service": "Open-Sora API",
  "version": "1.0.0",
  "status": "running",
  "endpoints": {
    "health": "/health",
    "predict": "/predict (Vertex AI endpoint)",
    "generate": "/v1/generate (async job submission)",
    "job_status": "/v1/jobs/{job_id}",
    "cancel": "DELETE /v1/jobs/{job_id}",
    "queue": "/v1/queue",
    "docs": "/docs"
  },
  "model_info": {
    "device": "cuda",
    "model_path": "/app/ckpts",
    "ready": true,
    "num_gpus": 1,
    "gpu_name": "NVIDIA H100 80GB HBM3",
    "gpu_memory_gb": 79.6,
    "resolutions": ["256px", "768px"],
    "aspect_ratios": ["16:9", "9:16", "1:1", "2.39:1"],
    "valid_frame_counts": [17, 33, 49, 65, 81, 97, 113, 129]
  }
}
```

---

## Source Files

### start.sh

Container startup script that:

1. Downloads model weights from GCS via `bootstrap_weights.py`
2. Starts FastAPI server with uvicorn

### app/bootstrap_weights.py

Downloads Open-Sora v2 weights from GCS before API starts.


**Required Files:**

- `Open_Sora_v2.safetensors` (~42GB) - Main diffusion model
- `hunyuan_vae.safetensors` - VAE encoder/decoder
- `google/t5-v1_1-xxl/` - T5 text encoder
- `openai/clip-vit-large-patch14/` - CLIP text encoder

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `WEIGHT_BUCKET` | - | GCS bucket with weights (required) |
| `WEIGHT_PREFIX` | `ckpts/` | Path prefix in bucket |
| `MODEL_PATH` | `/app/ckpts` | Local destination |
| `FORCE_DOWNLOAD` | `false` | Re-download if true |

### app/gcs_io.py

GCS utilities for uploading videos and downloading weights.

### app/opensora_runner.py

Wrapper around Open-Sora v2 inference using `torchrun`.

**Resolution Configs:**

| Resolution | Config | Description |
|------------|--------|-------------|
| `256px` | `t2i2v_256px.py` | Fast generation |
| `768px` | `t2i2v_768px.py` | High quality |

### app/job_manager.py

Manages async job queue with status tracking.

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `JOB_RETENTION_SECONDS` | `3600` | How long to keep completed jobs (seconds) |
| `MAX_COMPLETED_JOBS` | `100` | Maximum completed jobs to retain |

**Job Status Flow:**

```text
queued → processing → completed
                   ↘ failed
       ↘ cancelled
```

### app/worker.py

Background worker thread that processes jobs sequentially.

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `GENERATION_TIMEOUT` | `1800` | Max time per video generation (seconds) |

### app/main.py

FastAPI application with all endpoint definitions.
Uses lifespan context manager for startup/shutdown events.

---

## Environment Variables Summary

| Variable | Default | Description |
|----------|---------|-------------|
| `WEIGHT_BUCKET` | - | GCS bucket with model weights (required) |
| `WEIGHT_PREFIX` | `ckpts/` | Path in bucket |
| `MODEL_PATH` | `/app/ckpts` | Local checkpoint path |
| `PORT` | `8080` | API server port |
| `FORCE_DOWNLOAD` | `false` | Force re-download weights |
| `JOB_RETENTION_SECONDS` | `3600` | Keep completed jobs for this duration |
| `MAX_COMPLETED_JOBS` | `100` | Max completed jobs in memory |
| `GENERATION_TIMEOUT` | `1800` | Video generation timeout (30 min) |

