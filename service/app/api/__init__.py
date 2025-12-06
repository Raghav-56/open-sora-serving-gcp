"""
API routers for Open-Sora service.
"""

from fastapi import APIRouter
from app.api import health, generation, jobs


def create_router() -> APIRouter:
    """
    Combine all routers into a single router.
    
    Returns:
        Combined APIRouter with all endpoints
    """
    router = APIRouter()
    router.include_router(health.router)
    router.include_router(generation.router)
    router.include_router(jobs.router)
    return router


__all__ = ["create_router", "health", "generation", "jobs"]
