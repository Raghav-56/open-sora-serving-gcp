# Scripts & Helpers

This page documents the convenience scripts shipped under the `scripts/` directory in this repository. These are PowerShell helpers designed to simplify builds, deployments and testing against Vertex AI. They are intended to be used from a PowerShell environment on Windows or from `pwsh` on Linux/macOS.

> Tip: Most scripts rely on configuration from a `.env` file. Create `.env` from `.env.example` and fill in your settings (see `ENV_VARS.md`). Dot-source `myconfig.ps1` to load the variables into your current session.

## Quick Environment Setup

```powershell
# Create the local .env and edit it
Copy-Item .env.example .env
# Inspect and update .env
notepad .env
# Load .env into current PowerShell session (dot-source)
. .\scripts\myconfig.ps1
```

Dot-sourcing `myconfig.ps1` is important because it sets several environment variables used by the other scripts.

> Note: There is a small, experimental web frontend in the `frontend/` folder that you can use for local manual testing and demos. It is not production-ready; see `frontend/README.md` for instructions.

## Script List & Usage

- `myconfig.ps1` (required):
  - Loads the `.env` file, validates required variables, and computes derived environment variables (IMAGE_URI, MODEL_NAME, ENDPOINT_NAME).
  - Usage: `. .\scripts\myconfig.ps1` (dot-source to keep env variables in the current session)

- `build.ps1`:
  - Runs Google Cloud Build to build and push the project container to Artifact Registry.
  - Usage: `.\uild.ps1` (ensure `myconfig.ps1` has been sourced so `$env:IMAGE_URI` and `$env:REGION` are defined)

- `upload.ps1`:
  - Uploads the container image to Vertex AI as a model artifact (using `gcloud ai models upload`).
  - Usage: `.\upload.ps1` (after `build.ps1`)

- `endpoint.ps1`:
  - Creates a Vertex AI endpoint and prints the new endpoint ID.
  - Usage: `.\endpoint.ps1` (after `myconfig.ps1`)

- `model_id.ps1`:
  - Fetches the current `MODEL_ID` and `ENDPOINT_ID` using `gcloud` list commands and exports them as environment variables.
  - Usage: `.\model_id.ps1` (useful before `deploy.ps1`)

- `deploy.ps1`:
  - Deploys the previously uploaded model to the endpoint.
  - Usage: `.\deploy.ps1` (requires `$env:ENDPOINT_ID` and `$env:MODEL_ID` from `model_id.ps1`)

- `auth.ps1`:
  - Retrieves a bearer token for `gcloud` and places it in `$env:ACCESS_TOKEN`.
  - Usage: `.\auth.ps1` (helpful before making authenticated requests or using `reqt.ps1`)

- `reqt.ps1`:
  - Example request to the Vertex AI `:rawPredict` endpoint. Edit `ENDPOINT` and `PROJECT` strings in the file or load them via `myconfig.ps1` and `model_id.ps1` first.
  - Usage: `.\reqt.ps1`

- `api.ps1`:
  - Quick helper that shows the `:rawPredict` endpoint URL for the current configuration.
  - Usage: `.\api.ps1` (saves the API URL to `$env:API_URL`)

- `gcl_re.ps1` (wrapper):
  - Convenience wrapper that runs the other scripts in order. Add options for partial runs, dry-run, and continue-on-error.
  - Usage: `.\gcl_re.ps1 -DryRun` or `pwsh -File .\scripts\gcl_re.ps1 -StartStep 2 -EndStep 5`
  - Flags:
    - `-DryRun`: show what would run instead of executing
    - `-StartStep`: start at a specific step index
    - `-EndStep`: finish at a specific step index
    - `-ContinueOnError`: keep going if a step fails

- `client.py`:
  - A PyPI-style Python script to submit jobs and poll status (requires `requests`).
  - Usage: `python ./scripts/client.py --endpoint <rawPredict url> --prompt "my prompt" --output-bucket mybucket --token "$(gcloud auth print-access-token)"`

## Typical Workflows

1. Build and push using `gcloud` and helper scripts

```powershell
. .\scripts\myconfig.ps1
.\scripts\build.ps1
.\scripts\upload.ps1
.\scripts\model_id.ps1
.\scripts\endpoint.ps1   # only if you need to create a new endpoint
.\scripts\deploy.ps1
```

2. Full sequence using the wrapper (recommended for quick runs)

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\gcl_re.ps1
```

3. Test the deployed endpoint

```powershell
. .\scripts\myconfig.ps1
.\scripts\auth.ps1
.\scripts\api.ps1
# Use the printed API URL from api.ps1 and the access token from auth.ps1
python .\scripts\client.py --endpoint $env:API_URL --prompt "A cat playing piano" --output-bucket $env:WEIGHT_BUCKET --token $env:ACCESS_TOKEN
```

## Safety Tips

- Always test with `-DryRun` before running the full sequence.
- Use `-StartStep` and `-EndStep` to rerun a specific subset of steps (e.g., only deploy or only build).
- Review the `.env` file carefully and avoid committing secrets to the repo.

## Bash / Linux usage

Most commands can be reimplemented in a Bash script if you prefer. Use `gcloud` and `docker` equivalents — `myconfig.ps1` needs to be replaced by a Bash script that exports the required variables.

---

If you want I can also add a Bash-friendly set of scripts or a small Makefile to provide an OS-agnostic workflow.
