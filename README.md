# Open-Sora v2 API for GCP Vertex AI

**Nannie AI - Proprietary System**

A production-ready text-to-video generation API using Open-Sora v2 (11B model) deployed on Google Cloud Vertex AI.

## Model

- **Open-Sora v2** - 11B parameter text-to-video model by HPC-AI Tech
-- Repository: [Raghav-56/Open-Sora](https://github.com/Raghav-56/Open-Sora) — see the [Architecture docs](docs/markdown/architecture.md) for details on how the model is integrated into the service.
- Checkpoints: `Open_Sora_v2.safetensors` (~42GB), `hunyuan_vae.safetensors`

## Features

- Text-to-video generation (256px fast, 768px high quality)
- Optimized single-GPU mode with torchrun for stable distributed initialization
- Custom configurations with shardformer disabled for A100 stability
- Multiple aspect ratios: 16:9, 9:16, 1:1, 2.39:1
- Frame counts: 17, 33, 49, 65, 81, 97, 113, 129 (4k+1 format)
- Async job queue for non-blocking generation
- Automatic GCS upload of generated videos
- PyTorch SDPA attention backend for hardware compatibility
- Vertex AI compatible with health checks

## Project Structure

```text
open-sora-serving-gcp/
├── README.md
├── requirements.txt
├── .env.example              # Configuration template
├── ENV_VARS.md               # Environment variable guide
├── scripts/                  # Build/deploy PowerShell scripts and helpers
│   ├── myconfig.ps1         # Loads .env (create from .env.example), exports env vars
│   ├── build.ps1            # Build & push image with Cloud Build
│   ├── upload.ps1           # Upload image as Vertex AI model
│   ├── model_id.ps1         # Lookup MODEL_ID and ENDPOINT_ID
│   ├── endpoint.ps1         # Create Vertex AI Endpoint
│   ├── deploy.ps1           # Deploy model to endpoint
│   ├── auth.ps1             # Retrieve access token
│   ├── reqt.ps1             # Example rawPredict request (PowerShell)
│   ├── api.ps1              # Print the API :rawPredict URL
│   ├── gcl_re.ps1           # Sequence wrapper (build/upload/deploy, etc.)
│   └── client.py            # Minimal client for testing/polling
├── docs/
│   ├── mkdocs.yml
│   └── markdown/
│       ├── index.md
│       ├── architecture.md
│       ├── deployment_guide.md
│       ├── local_running.md
│       └── code_reference.md
└── service/
    ├── Dockerfile
    ├── start.sh
    └── app/
        ├── __init__.py
        ├── main.py           # FastAPI app assembly
        ├── worker.py         # Background job processor
        ├── core/             # Configuration & lifecycle
        │   ├── __init__.py
        │   ├── config.py     # Environment variables & constants
        │   └── lifespan.py   # Startup/shutdown management
        ├── models/           # Pydantic schemas
        │   ├── __init__.py
        │   ├── requests.py   # VideoGenerationRequest with validators
        │   └── responses.py  # API response models
        ├── jobs/             # Job queue management
        │   ├── __init__.py
        │   ├── models.py     # Job dataclass & JobStatus enum
        │   └── manager.py    # JobManager (thread-safe queue)
        ├── opensora/         # Open-Sora model wrapper
        │   ├── __init__.py
        │   ├── config.py     # Model configuration & MODE_CONFIGS
        │   ├── command_builder.py  # Torchrun command construction
        │   └── runner.py     # OpenSoraRunner (model interface)
        ├── configs/          # Custom single-GPU configurations
        │   ├── __init__.py
        │   ├── 256px_single_gpu.py  # Single GPU config (shardformer disabled)
        │   └── 768px_single_gpu.py  # High-res single GPU config
        ├── api/              # FastAPI routers
        │   ├── __init__.py
        │   ├── health.py     # Health check endpoints
        │   ├── generation.py # Video generation endpoints
        │   └── jobs.py       # Job status endpoints
        ├── utils/            # Utilities
        │   ├── __init__.py
        │   ├── exceptions.py # Custom exceptions
        │   └── gcs_io.py     # GCS upload/download
        └── scripts/          # Bootstrap scripts
            ├── __init__.py
            └── bootstrap_weights.py  # Model weight download
    ├── frontend/                # Minimal web UI (in development)
    │   ├── index.html
    │   ├── app.js
    │   ├── server.py            # Minimal Flask dev server for the frontend
    │   ├── README.md            # Frontend README & run instructions
    │   └── style.css
```

### Architecture

The service uses a **modular architecture** with clear separation of concerns; see the [Architecture docs](docs/markdown/architecture.md) for the full design and diagrams:

-- **`main.py`**: Minimal FastAPI app assembly (see [Code Reference](docs/markdown/code_reference.md))
- **`core/`**: Configuration management and application lifecycle
- **`models/`**: Pydantic request/response schemas with validation
- **`jobs/`**: Thread-safe job queue and state management
- **`opensora/`**: Open-Sora model wrapper with configuration
  - **`command_builder.py`**: Builds torchrun commands with proper distributed environment setup
  - **`runner.py`**: Handles video generation with robust error handling (checks for video existence even on non-zero exit codes)
- **`configs/`**: Custom single-GPU configurations (shardformer disabled for stability)
- **`api/`**: FastAPI routers organized by domain (health, generation, jobs)
- **`utils/`**: Shared utilities (GCS I/O, custom exceptions)
-- **`scripts/`**: Bootstrap scripts for weight management — see the [Scripts docs](docs/markdown/scripts.md)
- **`worker.py`**: Background thread for async job processing

### Key Implementation Details

- **Torchrun Integration**: Uses `torchrun --standalone` even for single GPU to properly initialize distributed environment required by Open-Sora's `dist.barrier()` calls
- **Custom Configs**: Single-GPU configurations with `shardformer=False` for T5 encoder stability
- **Attention Backend**: Uses PyTorch's native SDPA instead of Flash Attention via `TORCH_SDPA_FLASH_ATTENTION=0` and `ATTN_BACKEND=sdpa`
- **Robust Error Handling**: Checks for generated video files before failing on non-zero exit codes, as Open-Sora may crash at cleanup after successful generation

This structure emphasizes maintainability, testability, and code clarity.

## Quick Start

### Prerequisites

- Docker with NVIDIA runtime (H100 or A100 GPU)
- Google Cloud CLI authenticated
- Access to model weights GCS bucket

### Configuration Setup

```powershell
# 1. Create local environment configuration for scripts
Copy-Item .env.example .env
# Edit .env with your GCP project settings

# 2. For container runtime, set variables when deploying
# See [Configuration](docs/markdown/configuration.md) (or `ENV_VARS.md`) for a comprehensive list of environment variables used by the scripts and service
```

### Local Development

```powershell
cd service
docker build -t opensora-api:dev .

# Run with required environment variables (see [Local Running](docs/markdown/local_running.md))
docker run --gpus all -p 8081:8080 `
  -e WEIGHT_BUCKET=your-weights-bucket `
  -e WEIGHT_PREFIX=opensora_v2 `
  -e OUTPUT_BUCKET=your-output-bucket `
  opensora-api:dev
```

> **Note**: See [`ENV_VARS.md`](ENV_VARS.md) or the [Configuration docs](docs/markdown/configuration.md) for complete environment variable reference.

### Test the API

```bash
# Health check (see [Local Running](docs/markdown/local_running.md) for running locally and [Code Reference](docs/markdown/code_reference.md) for detailed endpoints)
curl http://localhost:8081/health

# Generate video
curl -X POST http://localhost:8081/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat playing piano in a jazz club",
    "resolution": "256px",
    "num_frames": 49,
    "aspect_ratio": "16:9",
    "output_bucket": "your-output-bucket"
  }'
```

## API Endpoints (see [Code Reference](docs/markdown/code_reference.md) for request/response models and example usages)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Vertex AI prediction endpoint |
| `/v1/generate` | POST | Submit video generation job |
| `/v1/jobs/{id}` | GET | Get job status |
| `/v1/jobs/{id}` | DELETE | Cancel job |
| `/v1/queue` | GET | Queue status |

## Generation Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `prompt` | string | required | Text description (up to 2000 chars) |
| `resolution` | `256px`, `768px` | `256px` | Video resolution |
| `num_frames` | 17, 33, 49, 65, 81, 97, 113, 129 | 49 | Frame count (4k+1) |
| `aspect_ratio` | `16:9`, `9:16`, `1:1`, `2.39:1` | `16:9` | Video aspect ratio |
| `mode` | `t2v_single_gpu`, `t2v`, `t2i2v` | `t2v_single_gpu` | Generation mode |
| `motion_score` | 1-5 or "dynamic" | 4 | Motion intensity |
| `seed` | integer | auto | For reproducibility |
| `output_bucket` | string | required | GCS bucket for output |

## Documentation

```bash
pip install mkdocs-material
cd docs
mkdocs serve
```

Then open `http://localhost:8000`

Take a look through the documentation pages for step-by-step instructions and reference material:

- [Overview](docs/markdown/index.md)
- [Architecture](docs/markdown/architecture.md)
- [Configuration](docs/markdown/configuration.md) (also see `ENV_VARS.md`)
- [Deployment Guide](docs/markdown/deployment_guide.md)
- [Local Running](docs/markdown/local_running.md)
- [Scripts & Helpers](docs/markdown/scripts.md)
- [Code Reference (API)](docs/markdown/code_reference.md)

## Deployment

Using PowerShell scripts (requires `.env` configuration):

```powershell
# Build and push container
.\scripts\build.ps1

# Deploy to Vertex AI
.\scripts\deploy.ps1
```

See [Deployment Guide](docs/markdown/deployment_guide.md) for full Vertex AI deployment instructions and [`ENV_VARS.md`](ENV_VARS.md) for environment variable reference.

## Scripts

This repository ships several PowerShell helper scripts under `scripts/` to make building, deploying and testing on Vertex AI straightforward. For more detailed usage examples, see the `Scripts` page in the docs: `docs/markdown/scripts.md`.

Short descriptions:

- `myconfig.ps1` — Load `.env` into the current process and compute derived variables. Dot-source it to export environment variables into your shell: `. .\scripts\myconfig.ps1`.
- `build.ps1` — Submits a Cloud Build that builds & pushes the Docker image defined in `service/Dockerfile`.
- `upload.ps1` — Uploads a container image as a Vertex AI model artifact.
- `model_id.ps1` — Retrieves the deployed `MODEL_ID` and `ENDPOINT_ID` and stores them as environment variables.
- `endpoint.ps1` — Creates a Vertex AI endpoint and prints the new Endpoint ID.
- `deploy.ps1` — Deploys a model to the selected Endpoint using `gcloud ai endpoints deploy-model`.
- `auth.ps1` — Sets `$env:ACCESS_TOKEN` using `gcloud auth print-access-token` and prints it.
- `reqt.ps1` — Example rawPredict request for quick testing; update `ENDPOINT` or call `api.ps1` to get the URL.
- `api.ps1` — Prints the final `:rawPredict` API URL for your environment.
- `client.py` — A small test client you can use to submit jobs and poll the job status.
- `gcl_re.ps1` — A convenience wrapper (runs the other scripts in order). Supports `-DryRun`, `-StartStep`, `-EndStep`, and `-ContinueOnError`.

Example: build and deploy with the wrapper (safe dry-run):

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\gcl_re.ps1 -DryRun
```

See `docs/markdown/scripts.md` for a more detailed description and examples.

### Frontend (experimental)

We include a small web frontend under `frontend/` aimed at demos and testing. This frontend is in active development and should NOT be considered production-ready. It provides a lightweight UI with a small Flask server for local testing and is useful for quick manual checks and demos.

Quick run instructions (development):

```powershell
cd frontend
pip install -r requirements.txt
python server.py
# Browse to http://localhost:5000
```

Notes:

- The UI is a demo; features are limited (preview, submit form, basic status).  
- For production usage you should instead call the API directly, or build a robust web app with production-grade auth and rate-limiting.
- See `frontend/README.md` for additional details.

## Maintainer

Raghav ([@Raghav-56](https://github.com/Raghav-56))

## License

Proprietary - Nannie AI Internal Use Only
