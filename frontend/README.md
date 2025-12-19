# Open Sora Frontend (Experimental)

Status: Experimental — this frontend is a demo and in active development. It is useful for quick manual testing and demos but is not production ready and should not be used in a public-facing deployment without proper hardening.

A simple, clean web interface to test the Open Sora video generation model.

## Quick Start

1. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

2. **Make sure you're authenticated with Google Cloud:**
   ```powershell
   gcloud auth login
   gcloud config set project nannieai-website-stealth
   ```

3. **Start the server:**
   ```powershell
   python server.py
   ```

4. **Open your browser:**
   Navigate to `http://localhost:5000`

## Features

- Clean, modern UI with gradient background
- Easy form inputs for all video generation parameters:
  - Text prompt
  - Resolution (256px or 768px)
  - Number of frames (17-129)
  - Aspect ratio (16:9, 9:16, 1:1, etc.)
  - Mode (text-to-video or image-to-video)
- Real-time status updates
- Video preview when generation completes
- Responsive design

## Limitations

- No production authentication or rate-limiting is configured. The server is a local dev proxy intended to simplify manual testing.
- Not all generator features are supported in the UI — for full control use the API directly.

## How It Works

The frontend (`index.html`, `app.js`, `style.css`) provides a user-friendly interface that sends requests to a Flask backend (`server.py`). The backend handles authentication with Google Cloud and proxies requests to your Vertex AI endpoint.

## Configuration

The API endpoint is configured in `server.py`:
```python
API_ENDPOINT = "https://europe-west4-aiplatform.googleapis.com/v1/projects/nannieai-website-stealth/locations/europe-west4/endpoints/179257778722832384:rawPredict"
```

The default output bucket is set in `app.js`:
```javascript
output_bucket: 'nannie-opensora-weights-so',
output_prefix: 'test-videos/'
```
