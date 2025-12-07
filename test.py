"""
Test script for Open-Sora Vertex AI Endpoint
Submits a video generation request and polls for completion.
"""

import json
import time
import subprocess
import requests
from datetime import datetime

# =============================================================================
# Configuration - Update these values
# =============================================================================

PROJECT_ID = "nannieai-website-stealth"
REGION = "europe-west4"
ENDPOINT_ID = "YOUR_ENDPOINT_ID"  # Update this after deployment

# Test parameters
TEST_PROMPT = "A golden retriever puppy playing with a red ball in a sunny garden, cinematic lighting, slow motion"
OUTPUT_BUCKET = "nannie-opensora-weights-so"  # Your GCS bucket for output
OUTPUT_PREFIX = "test-videos/"

# Video generation settings
RESOLUTION = "256px"  # "256px" (fast) or "768px" (high quality)
NUM_FRAMES = 17       # 17, 33, 49, 65, 81, 97, 113, 129
ASPECT_RATIO = "16:9" # "16:9", "9:16", "1:1", "2.39:1"

# =============================================================================
# Helper Functions
# =============================================================================

def get_access_token():
    """Get GCP access token using gcloud CLI."""
    try:
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to get access token: {e}")
        print("Run: gcloud auth login")
        raise


def get_endpoint_id():
    """Get the endpoint ID from the deployment."""
    global ENDPOINT_ID
    
    if ENDPOINT_ID != "YOUR_ENDPOINT_ID":
        return ENDPOINT_ID
    
    print("🔍 Looking up endpoint ID...")
    try:
        result = subprocess.run(
            [
                "gcloud", "ai", "endpoints", "list",
                "--region", REGION,
                "--project", PROJECT_ID,
                "--filter", "displayName~opensora",
                "--format", "value(name)"
            ],
            capture_output=True,
            text=True,
            check=True
        )
        
        endpoints = result.stdout.strip().split('\n')
        if endpoints and endpoints[0]:
            # Extract endpoint ID from full path
            ENDPOINT_ID = endpoints[0].split('/')[-1]
            print(f"✅ Found endpoint: {ENDPOINT_ID}")
            return ENDPOINT_ID
        else:
            print("❌ No OpenSora endpoint found")
            print("Deploy first using: ./full_deploy.sh")
            raise ValueError("No endpoint found")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to list endpoints: {e}")
        raise


def submit_generation_request(token: str) -> dict:
    """Submit video generation request to Vertex AI endpoint."""
    
    endpoint_url = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}:rawPredict"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "prompt": TEST_PROMPT,
        "resolution": RESOLUTION,
        "num_frames": NUM_FRAMES,
        "aspect_ratio": ASPECT_RATIO,
        "output_bucket": OUTPUT_BUCKET,
        "output_prefix": OUTPUT_PREFIX
    }
    
    print(f"\n📤 Submitting request to: {endpoint_url}")
    print(f"   Prompt: {TEST_PROMPT[:60]}...")
    print(f"   Resolution: {RESOLUTION}")
    print(f"   Frames: {NUM_FRAMES}")
    print(f"   Aspect Ratio: {ASPECT_RATIO}")
    print(f"   Output: gs://{OUTPUT_BUCKET}/{OUTPUT_PREFIX}")
    
    response = requests.post(endpoint_url, headers=headers, json=payload, timeout=60)
    
    if response.status_code != 200:
        print(f"❌ Request failed: {response.status_code}")
        print(f"   Response: {response.text}")
        raise Exception(f"API request failed: {response.status_code}")
    
    return response.json()


def check_job_status(token: str, job_id: str) -> dict:
    """Check job status via the API."""
    
    # Use rawPredict with a custom status check endpoint
    endpoint_url = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}:rawPredict"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Send a GET-style request to check status
    # The API expects POST, so we'll construct a status check payload
    # Actually, for job status we need to call a different endpoint path
    
    # For Vertex AI custom containers, we need to use the predict endpoint
    # with a special request format, or access the container directly
    
    # Alternative: Check if the video file exists in GCS
    video_path = f"gs://{OUTPUT_BUCKET}/{OUTPUT_PREFIX}{job_id}/{job_id}.mp4"
    
    try:
        result = subprocess.run(
            ["gsutil", "stat", video_path],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return {"status": "completed", "video_uri": video_path}
    except:
        pass
    
    return {"status": "processing"}


def poll_for_completion(token: str, job_id: str, max_wait: int = 1800) -> dict:
    """Poll for job completion."""
    
    print(f"\n⏳ Waiting for video generation (max {max_wait}s)...")
    print(f"   Job ID: {job_id}")
    
    start_time = time.time()
    poll_interval = 30  # Check every 30 seconds
    
    while time.time() - start_time < max_wait:
        elapsed = int(time.time() - start_time)
        print(f"   [{elapsed}s] Checking status...")
        
        status = check_job_status(token, job_id)
        
        if status.get("status") == "completed":
            print(f"\n✅ Video generated!")
            return status
        elif status.get("status") == "failed":
            print(f"\n❌ Generation failed: {status.get('error', 'Unknown error')}")
            return status
        
        time.sleep(poll_interval)
    
    print(f"\n⏰ Timeout after {max_wait}s")
    return {"status": "timeout"}


def main():
    """Main test function."""
    
    print("=" * 60)
    print("  Open-Sora Vertex AI Endpoint Test")
    print("=" * 60)
    print(f"\n📋 Configuration:")
    print(f"   Project: {PROJECT_ID}")
    print(f"   Region: {REGION}")
    
    # Get endpoint ID if not set
    endpoint_id = get_endpoint_id()
    print(f"   Endpoint: {endpoint_id}")
    
    # Get access token
    print("\n🔑 Getting access token...")
    token = get_access_token()
    print("✅ Token acquired")
    
    # Submit generation request
    try:
        result = submit_generation_request(token)
        print(f"\n✅ Job submitted successfully!")
        print(f"   Job ID: {result.get('job_id', 'N/A')}")
        print(f"   Status: {result.get('status', 'N/A')}")
        print(f"   Expected URI: {result.get('expected_video_uri', 'N/A')}")
        
        job_id = result.get('job_id')
        
        if job_id:
            # Poll for completion
            final_status = poll_for_completion(token, job_id)
            
            if final_status.get("status") == "completed":
                video_uri = final_status.get("video_uri")
                print(f"\n🎬 Video ready!")
                print(f"   URI: {video_uri}")
                print(f"\n   Download with:")
                print(f"   gsutil cp {video_uri} ./output_video.mp4")
            else:
                print(f"\n⚠️  Final status: {final_status.get('status')}")
                print("\n   Check logs with:")
                print(f"   gcloud logging read 'resource.type=aiplatform.googleapis.com/Endpoint AND resource.labels.endpoint_id={endpoint_id}' --limit=50 --project={PROJECT_ID}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n   Check endpoint health:")
        print(f"   curl -H 'Authorization: Bearer $(gcloud auth print-access-token)' \\")
        print(f"     'https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}:rawPredict' \\")
        print(f"     -H 'Content-Type: application/json' \\")
        print(f"     -d '{{\"prompt\": \"test\", \"output_bucket\": \"{OUTPUT_BUCKET}\"}}'")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

