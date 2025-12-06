"""
Health check endpoints for Open-Sora API.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from loguru import logger

from app.opensora import OpenSoraRunner


router = APIRouter(tags=["health"])

# Global runner reference (set by lifespan in main.py)
runner: Optional[OpenSoraRunner] = None


def set_runner(runner_instance: OpenSoraRunner):
    """Set the global runner instance."""
    global runner
    runner = runner_instance


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint for Vertex AI.
    
    Returns:
        Health status with model loaded state
    """
    if runner is None or not runner.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready"
        )
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/", status_code=status.HTTP_200_OK)
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        Service information and available endpoints
    """
    return {
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
        "model_info": runner.get_info() if runner else None
    }
