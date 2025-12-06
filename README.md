# Open-Sora v2 API for GCP Vertex AI

**Nannie AI - Proprietary System**

A production-ready text-to-video generation API using Open-Sora v2 (11B model) deployed on Google Cloud Vertex AI.

## Model

- **Open-Sora v2** - 11B parameter text-to-video model by HPC-AI Tech
- Repository: [Raghav-56/Open-Sora](https://github.com/Raghav-56/Open-Sora)
- Checkpoints: `Open_Sora_v2.safetensors` (~42GB), `hunyuan_vae.safetensors`

## Features

- Text-to-video generation (256px fast, 768px high quality)
- Multiple aspect ratios: 16:9, 9:16, 1:1, 2.39:1
- Frame counts: 17, 33, 49, 65, 81, 97, 113, 129 (4k+1 format)
- Async job queue for non-blocking generation
- Automatic GCS upload of generated videos
- Vertex AI compatible with health checks

## Project Structure

```text
open-sora-serving-gcp/
├── README.md
├── requirements.txt
├── .env.example              # Configuration template
├── ENV_VARS.md               # Environment variable guide
├── scripts/                  # Build/deploy PowerShell scripts
│   ├── myconfig.ps1         # Loads .env (create from .env.example)
│   ├── build.ps1
│   ├── deploy.ps1
│   ├── api.ps1
│   └── client.py
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
        │   ├── config.py     # Model configuration
        │   ├── command_builder.py  # CLI command construction
        │   └── runner.py     # OpenSoraRunner (model interface)
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
```

### Architecture

The service uses a **modular architecture** with clear separation of concerns:

- **`main.py`**: Minimal FastAPI app assembly
- **`core/`**: Configuration management and application lifecycle
- **`models/`**: Pydantic request/response schemas with validation
- **`jobs/`**: Thread-safe job queue and state management
- **`opensora/`**: Open-Sora model wrapper with configuration
- **`api/`**: FastAPI routers organized by domain (health, generation, jobs)
- **`utils/`**: Shared utilities (GCS I/O, custom exceptions)
- **`scripts/`**: Bootstrap scripts for weight management
- **`worker.py`**: Background thread for async job processing

This structure has maintainability, testability, and code clarity

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
# See ENV_VARS.md for comprehensive guide on environment variables
```

### Local Development

```powershell
cd service
docker build -t opensora-api:dev .

# Run with required environment variables
docker run --gpus all -p 8081:8080 `
  -e WEIGHT_BUCKET=your-weights-bucket `
  -e WEIGHT_PREFIX=opensora_v2 `
  -e OUTPUT_BUCKET=your-output-bucket `
  opensora-api:dev
```

> **Note**: See [`ENV_VARS.md`](ENV_VARS.md) for complete environment variable reference.

### Test the API

```bash
# Health check
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

## API Endpoints

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
| `motion_score` | 1-10 | auto | Motion intensity (5-7 recommended) |
| `seed` | integer | auto | For reproducibility |
| `output_bucket` | string | required | GCS bucket for output |

## Documentation

```bash
pip install mkdocs-material
cd docs
mkdocs serve
```

Then open `http://localhost:8000`

## Deployment

Using PowerShell scripts (requires `.env` configuration):

```powershell
# Build and push container
.\scripts\build.ps1

# Deploy to Vertex AI
.\scripts\deploy.ps1
```

See [Deployment Guide](docs/markdown/deployment_guide.md) for full Vertex AI deployment instructions and [`ENV_VARS.md`](ENV_VARS.md) for environment variable reference.

## Maintainer

Raghav ([@Raghav-56](https://github.com/Raghav-56))

## License

Proprietary - Nannie AI Internal Use Only
