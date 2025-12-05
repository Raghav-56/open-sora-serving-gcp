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
        ├── main.py
        ├── opensora_runner.py
        ├── job_manager.py
        ├── worker.py
        ├── gcs_io.py
        └── bootstrap_weights.py
```

## Quick Start

### Prerequisites

- Docker with NVIDIA runtime (H100 or A100 GPU)
- Google Cloud CLI authenticated
- Access to model weights GCS bucket

### Local Development

```bash
cd service
docker build -t opensora-api:dev .
docker run --gpus all -p 8081:8080 \
  -e WEIGHT_BUCKET=your-weights-bucket \
  opensora-api:dev
```

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

See [Deployment Guide](docs/markdown/deployment_guide.md) for full Vertex AI deployment instructions.

## Maintainer

Raghav ([@Raghav-56](https://github.com/Raghav-56))

## License

Proprietary - Nannie AI Internal Use Only
