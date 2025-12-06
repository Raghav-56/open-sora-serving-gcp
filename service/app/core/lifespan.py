"""
FastAPI lifespan context manager for startup and shutdown.
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from loguru import logger

from app.opensora import OpenSoraRunner
from app.jobs import JobManager
from app.worker import InferenceWorker
from app.api import health, generation, jobs


# Shared state
runner: Optional[OpenSoraRunner] = None
worker: Optional[InferenceWorker] = None
job_manager: Optional[JobManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles:
    - Startup: Model loading, job manager initialization, worker start
    - Shutdown: Graceful worker termination
    
    See: https://fastapi.tiangolo.com/advanced/events/
    """
    global runner, worker, job_manager
    
    # STARTUP
    logger.info("🚀 Starting Open-Sora API server v1.0.0...")
    
    try:
        # Load model
        model_path = os.getenv("MODEL_PATH", "/app/ckpts")
        logger.info(f"📂 Loading model from: {model_path}")
        runner = OpenSoraRunner(model_path)
        
        # Initialize job manager
        job_manager = JobManager()
        
        # Initialize worker
        worker = InferenceWorker(runner, job_manager)
        
        # Set dependencies on routers
        health.set_runner(runner)
        generation.set_dependencies(job_manager, worker)
        jobs.set_job_manager(job_manager)
        
        logger.info("✅ Server ready!")
        logger.info(f"   Device: {runner.device}")
        logger.info("   API version: 1.0.0")
        logger.info(
            "   Endpoints: /predict, /v1/generate, /v1/jobs, /v1/queue"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        raise
    
    yield  # Server is running, handle requests
    
    # SHUTDOWN
    logger.info("🛑 Shutting down Open-Sora API server...")
    if worker:
        worker.shutdown()
    logger.info("✅ Shutdown complete")


__all__ = ["lifespan"]
