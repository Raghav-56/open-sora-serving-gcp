const API_ENDPOINT = 'https://europe-west4-aiplatform.googleapis.com/v1/projects/nannieai-website-stealth/locations/europe-west4/endpoints/179257778722832384:rawPredict';

const form = document.getElementById('videoForm');
const generateBtn = document.getElementById('generateBtn');
const statusDiv = document.getElementById('status');
const resultDiv = document.getElementById('result');
const videoContainer = document.getElementById('videoContainer');

function getTimestamp() {
    return new Date().toISOString().split('T')[1].slice(0, -1);
}

function log(message, type = 'INFO') {
    const line = document.createElement('div');
    line.style.marginBottom = '4px';
    line.innerHTML = `<span style="opacity: 0.5">[${getTimestamp()}]</span> <span style="color: ${type === 'ERROR' ? 'var(--error-text)' : type === 'SUCCESS' ? 'var(--success-text)' : 'var(--text-primary)'}">[${type}]</span> ${message}`;
    statusDiv.appendChild(line);
    statusDiv.scrollTop = statusDiv.scrollHeight;
}

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData(form);
    const payload = {
        prompt: formData.get('prompt'),
        resolution: formData.get('resolution'),
        num_frames: parseInt(formData.get('num_frames')),
        aspect_ratio: formData.get('aspect_ratio'),
        mode: formData.get('mode'),
        output_bucket: 'nannie-opensora-weights-so',
        output_prefix: 'test-videos/'
    };

    // Update UI
    generateBtn.disabled = true;
    generateBtn.textContent = 'Processing...';
    statusDiv.className = 'status';
    statusDiv.innerHTML = ''; // Clear previous logs
    resultDiv.className = 'result hidden';
    
    log('Initializing generation request...');
    log(`Configuration: ${payload.resolution}, ${payload.num_frames} frames, ${payload.aspect_ratio}`);

    try {
        log('Sending payload to Vertex AI endpoint...');
        
        // Make request through backend proxy
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to generate video');
        }

        // Show success
        log('Response received from endpoint.', 'SUCCESS');
        log('Video generation completed successfully.', 'SUCCESS');

        // Display result
        resultDiv.className = 'result';
        
        if (data.video_url) {
            videoContainer.innerHTML = `
                <video controls width="100%" autoplay loop>
                    <source src="${data.video_url}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <div class="info">
                    <div>ID: ${data.job_id || 'N/A'}</div>
                    ${data.gcs_path ? `<div>GCS: ${data.gcs_path}</div>` : ''}
                </div>
            `;
        } else if (data.job_id) {
            videoContainer.innerHTML = `
                <div class="info">
                    <div>Job ID: ${data.job_id}</div>
                    <div>Status: ${data.status || 'Processing'}</div>
                    ${data.gcs_path ? `<div>GCS: ${data.gcs_path}</div>` : ''}
                </div>
            `;
        }

    } catch (error) {
        log(`Error: ${error.message}`, 'ERROR');
        console.error('Error:', error);
    } finally {
        generateBtn.disabled = false;
        generateBtn.textContent = 'Initialize Generation';
    }
});

// Add example prompt functionality
const examplePrompts = [
    "A dog and cat playing together in a park",
    "A beautiful sunset over ocean waves",
    "A robot dancing in a futuristic city",
    "Cyberpunk street scene with neon lights",
    "Astronaut floating in deep space"
];

// Optional: Add a random example button
document.addEventListener('DOMContentLoaded', () => {
    const promptField = document.getElementById('prompt');
    promptField.placeholder = examplePrompts[Math.floor(Math.random() * examplePrompts.length)];
});
