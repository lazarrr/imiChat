# src/api/main.py
"""
FastAPI application for the University Chatbot.

v2 — Multi-turn conversation support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Each request carries an optional ``session_id``.
- The server maintains an in-memory conversation history per session.
- Every turn: decide semantically whether the request is a follow-up and
  rewrite it into a standalone query if needed → embed → search → build
  a two-message prompt (system + resolved question + context) → call
  Ollama /api/chat → stream tokens + sources → save turn to history.
  History is never sent to the answer model.

Module layout
~~~~~~~~~~~~~
- ``src.api.schemas``      — request/response models
- ``src.api.sessions``     — in-memory session store
- ``src.api.dependencies`` — shared QA system instance
- ``src.api.middleware``   — CORS setup
- ``src.api.routes``       — chat streaming and health endpoints
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from config.settings import settings
from src.api.middleware import configure_middleware
from src.api.routes import api_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="University Chatbot API")

configure_middleware(app)
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )
