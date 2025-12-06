"""
Open-Sora v2 API Server for Vertex AI
Async video generation with job queueing and status tracking.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.core.lifespan import lifespan
from app.api import create_router


# Create FastAPI application
app = FastAPI(
    title="Open-Sora API",
    description="Text-to-video generation API using Open-Sora model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Include all API routes
app.include_router(create_router())


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom handler for HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail if isinstance(exc.detail, dict) else {
            "error": "http_error",
            "message": str(exc.detail)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all handler for unexpected exceptions."""
    from loguru import logger
    logger.error(f"❌ Unhandled exception: {exc}")
    logger.exception(exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An unexpected error occurred",
            "details": {"error_type": type(exc).__name__}
        }
    )


# Main entry point for local testing
if __name__ == "__main__":
    import uvicorn
    from loguru import logger
    
    port = int(os.getenv("PORT", "8080"))
    
    logger.info(f"🚀 Starting server on port {port}...")
    logger.info(f"   Health check: http://localhost:{port}/health")
    logger.info(f"   API docs: http://localhost:{port}/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
