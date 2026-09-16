# src/api/routes/health.py
"""Liveness endpoint."""

from fastapi import APIRouter

from src.api.sessions import active_session_count

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "healthy", "active_sessions": active_session_count()}
