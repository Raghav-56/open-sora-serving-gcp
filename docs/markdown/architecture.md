# Open-Sora API Architecture Documentation

This documentation is for individuals curious about how this API is designed. Below sections provide the overall **rationale** of the architecture design.

## Table of Contents

1. [Why This Architecture?](#why-this-architecture)
2. [Component Breakdown](#component-breakdown)
3. [Request Flow](#request-flow)
4. [File Organization](#file-organization)
5. [Summary](#summary)

---

## Why This Architecture?

### Problem: Video Generation is Slow and Resource-Intensive

#### Challenge 1: Long Processing Time

- Video generation takes 2-15 minutes per request depending on resolution
- HTTP requests would timeout if we waited for completion
- **Solution**: Async job queue - return job_id immediately, process in background

#### Challenge 2: Limited GPU Memory

- GPU can only process one video at a time
- Multiple simultaneous requests would cause OOM errors
- **Solution**: Sequential job queue with worker thread and per-job
  output/log directories

#### Challenge 3: Model Weights are Large

- Open-Sora weights are several GB
- Downloading weights during each container start would be slow
- **Solution**: Bootstrap script downloads weights once at startup, before API starts

#### Challenge 4: Vertex AI Requirements

- Must respond to `/health` endpoint
- Must handle `/predict` POST requests
- Must return quickly (can't block on long operations)
- **Solution**: Async architecture with job tracking

### Architecture Pattern: Producer-Consumer Queue

```text
Client Request → FastAPI (Producer) → Job Queue → Worker Thread (Consumer) → Video Generation
                      ↓                                                              ↓
                   Job ID returned                                           Upload to GCS
                   immediately
```

This is a classic **producer-consumer pattern**:

- **Producer** (FastAPI): Receives requests, creates jobs, returns job_id
- **Queue**: Stores pending jobs in order
- **Consumer** (Worker): Processes jobs one at a time

---

## Component Breakdown

### 1. Container Startup (`start.sh`)

**Purpose**: Orchestrates container initialization in the correct order

**What it does**:

```bash
1. Set environment variables
2. Run bootstrap_weights.py (download model weights)
3. If bootstrap fails → exit with error
4. If bootstrap succeeds → start FastAPI server
```

### 2. Weight Bootstrapping (`app/scripts/bootstrap_weights.py`)

**Purpose**: Download model weights from GCS to local disk before API starts

**What it does**:

```python
1. Check if weights already exist locally (from previous container run)
2. If missing → download from GCS bucket
3. Verify all required files are present
4. Exit with error code if verification fails
```

### 3. GCS I/O (`app/utils/gcs_io.py`)

**Purpose**: Centralized module for all Google Cloud Storage operations

**Functions**:

- `upload_video_to_gcs()`: Uploads generated videos (supports multiple
  samples) to GCS
- `save_video_locally()`: Saves video to local disk
- `download_directory()`: Downloads model weights from GCS
- `download_blob()`: Downloads a single file

### 4. Open-Sora Runner (`app/opensora/`)

**Purpose**: Wrapper around the Open-Sora model

**Module Structure**:

- `config.py`: Model configuration constants and MODE_CONFIGS
- `command_builder.py`: Builds torchrun CLI commands
- `runner.py`: OpenSoraRunner class (main interface)

**What it does**:

- Verifies model weights exist (main checkpoint, VAE, T5, CLIP)
- Detects GPU configuration
- Runs `torchrun` subprocess for video generation
- Returns generated video paths (for multi-sample), seed used, and a
  tail of recent logs

**Resolution Configurations**:

- `256px`: Lower resolution, faster generation (~2-3 min for 49 frames)
- `768px`: Higher resolution, slower generation (~10-15 min for 49 frames)

**Motion Score**: Controls motion intensity (1-5 or "dynamic", default 4). Lower values produce calmer videos, higher values produce more dynamic motion.

### 5. Job Manager (`app/jobs/`)

**Purpose**: Manages the job queue and tracks job status

**Module Structure**:

- `models.py`: Job dataclass and JobStatus enum
- `manager.py`: JobManager class (thread-safe queue operations)

**Key Classes**:

- `JobStatus`: Enum for job states (queued, processing, completed, failed, cancelled)
- `Job`: Dataclass storing job information
- `JobManager`: Main class for queue management

### 6. Worker (`app/worker.py`)

**Purpose**: Background worker that processes jobs from the queue

**Architecture**:

```text
Main Thread (FastAPI)          |  Background Thread (Worker)
-------------------------------|--------------------------------
Receive HTTP request           |
Create job                     |
Add to queue                   |
Return job_id immediately      |
                               |  ← Get job from queue
                               |  ← Generate video
                               |  ← Upload to GCS (all samples)
                               |  ← Mark job complete with URIs/log tail
```

### 7. Main API (`app/main.py` and `app/api/`)

**Purpose**: FastAPI application that handles HTTP requests

**Module Structure**:

- `main.py`: Minimal FastAPI app assembly (73 lines, down from 605)
- `core/config.py`: Environment variables and validation constants
- `core/lifespan.py`: Startup/shutdown lifecycle management
- `models/requests.py`: VideoGenerationRequest with 7 field validators
- `models/responses.py`: API response schemas
- `api/health.py`: Health check endpoints
- `api/generation.py`: Video generation endpoints
- `api/jobs.py`: Job status endpoints

**Key Endpoints**:

- `GET /health`: Health check for Vertex AI
- `POST /predict`: Vertex AI prediction endpoint
- `POST /v1/generate`: Async job submission
- `GET /v1/jobs/{job_id}`: Get job status
- `DELETE /v1/jobs/{job_id}`: Cancel job
- `GET /v1/queue`: Get queue status

**Modular Design Benefits**:

- **Separation of Concerns**: Each module has a single, clear responsibility
- **Testability**: Smaller modules are easier to unit test
- **Maintainability**: Changes isolated to relevant modules
- **Readability**: 73-line main.py vs. 605-line monolithic file

---

## Request Flow

### Complete Request Lifecycle

#### Step 1: Client Submits Request

```json
POST /predict
{
  "prompt": "a cat playing piano",
  "resolution": "256px",
  "num_frames": 51,
  "output_bucket": "my-videos"
}
```

#### Step 2: FastAPI Receives Request

- Pydantic validates request
- If invalid → return 400 error
- If valid → continue to handler

#### Step 3: Job Submission

- Generate unique job_id
- Create Job object with status=QUEUED
- Add to queue
- Return response immediately

#### Step 4: Worker Picks Up Job

- Worker loop calls `get_next_job()`
- Job status → PROCESSING
- Video generation starts

#### Step 5: Video Generation

- Open-Sora model runs inference
- Video saved locally

#### Step 6: Save and Upload

- Normalize per-job outputs under `/tmp/opensora_outputs/{job_id}`
- Upload one or more videos to GCS
- Delete local job directory
- Mark job as COMPLETED with URIs and log tail

#### Step 7: Client Polls Status

```json
GET /v1/jobs/{job_id}

Response:
{
  "job_id": "20251202_143022_847_001",
  "status": "completed",
  "video_uri": "gs://my-videos/20251202_143022_847_001/20251202_143022_847_001.mp4",
  "video_uris": [
    "gs://my-videos/20251202_143022_847_001/20251202_143022_847_001.mp4"
  ],
  "seed": 42,
  "log_tail": ["...last log lines..."]
}
```

---

## File Organization

```text
open-sora-serving-gcp/service/
│
├── Dockerfile                 # Container image definition
├── start.sh                   # Container startup script
│
└── app/                       # Python application package
    ├── __init__.py            # Makes 'app' a Python package
    ├── main.py                # FastAPI application (73 lines)
    ├── worker.py              # Background worker thread
    │
    ├── core/                  # Configuration & lifecycle
    │   ├── __init__.py
    │   ├── config.py          # Environment variables & constants
    │   └── lifespan.py        # Startup/shutdown management
    │
    ├── models/                # Pydantic schemas
    │   ├── __init__.py
    │   ├── requests.py        # VideoGenerationRequest
    │   └── responses.py       # API response models
    │
    ├── jobs/                  # Job queue management
    │   ├── __init__.py
    │   ├── models.py          # Job dataclass & JobStatus enum
    │   └── manager.py         # JobManager (thread-safe queue)
    │
    ├── opensora/              # Open-Sora model wrapper
    │   ├── __init__.py
    │   ├── config.py          # Model configuration
    │   ├── command_builder.py # CLI command construction
    │   └── runner.py          # OpenSoraRunner (model interface)
    │
    ├── api/                   # FastAPI routers
    │   ├── __init__.py
    │   ├── health.py          # Health check endpoints
    │   ├── generation.py      # Video generation endpoints
    │   └── jobs.py            # Job status endpoints
    │
    ├── utils/                 # Utilities
    │   ├── __init__.py
    │   ├── exceptions.py      # Custom exceptions
    │   └── gcs_io.py          # GCS upload/download
    │
    └── scripts/               # Bootstrap scripts
        ├── __init__.py
        └── bootstrap_weights.py  # Model weight download
```

**Modular Architecture**:

follows **domain-driven design** pattern:

- **`core/`**: Application-wide configuration and lifecycle (config, lifespan)
- **`models/`**: Data validation layer (Pydantic schemas)
- **`jobs/`**: Job queue domain (queue management, state tracking)
- **`opensora/`**: Model domain (runner, config, command builder)
- **`api/`**: API layer (organized by endpoint purpose)
- **`utils/`**: Cross-cutting concerns (GCS I/O, exceptions)
- **`scripts/`**: Operational scripts (weight bootstrap)


---

## Summary

This architecture balances:

- ✅ **Reliability**: Jobs tracked, errors handled
- ✅ **Performance**: Async queue, sequential GPU processing
- ✅ **Simplicity**: Clear separation of concerns
- ✅ **Maintainability**: Well-organized code, comprehensive logging
- ✅ **Vertex AI Compatibility**: Meets all platform requirements
