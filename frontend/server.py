"""
Simple frontend server for Open Sora video generation.
Serves the HTML frontend and proxies requests to Vertex AI.
"""

import subprocess
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__, 
            static_folder='.',
            template_folder='.')
CORS(app)

# Configuration
API_ENDPOINT = "https://europe-west4-aiplatform.googleapis.com/v1/projects/nannieai-website-stealth/locations/europe-west4/endpoints/179257778722832384:rawPredict"

def get_access_token():
    """Get Google Cloud access token using gcloud."""
    try:
        result = subprocess.run(
            ['gcloud', 'auth', 'print-access-token'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error getting access token: {e}")
        return None

@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files (CSS, JS)."""
    return send_from_directory('.', path)

@app.route('/api/generate', methods=['POST'])
def generate_video():
    """Proxy request to Vertex AI endpoint."""
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Prompt is required'}), 400
        
        # Get access token
        access_token = get_access_token()
        if not access_token:
            return jsonify({'error': 'Failed to get authentication token'}), 500
        
        # Make request to Vertex AI
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            API_ENDPOINT,
            headers=headers,
            json=data,
            timeout=300  # 5 minute timeout
        )
        
        # Return response
        if response.ok:
            return jsonify(response.json()), 200
        else:
            return jsonify({
                'error': f'API request failed: {response.status_code}',
                'details': response.text
            }), response.status_code
            
    except requests.Timeout:
        return jsonify({'error': 'Request timed out'}), 504
    except Exception as e:
        print(f"Error in generate_video: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎬 Open Sora Frontend Server")
    print("="*60)
    print(f"\n✅ Server starting at: http://localhost:5000")
    print(f"✅ API Endpoint: {API_ENDPOINT}")
    print("\nOpen http://localhost:5000 in your browser to test the model")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
