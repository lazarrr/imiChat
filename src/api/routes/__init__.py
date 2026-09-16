# src/api/routes/__init__.py
"""Aggregates all API routers into a single router for the app."""

from fastapi import APIRouter

from src.api.routes.chat import router as chat_router
from src.api.routes.health import router as health_router

api_router = APIRouter()
api_router.include_router(chat_router)
api_router.include_router(health_router)

__all__ = ["api_router"]
