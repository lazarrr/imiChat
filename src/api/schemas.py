# src/api/schemas.py
"""Request and response models for the University Chatbot API."""

from __future__ import annotations

from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None   # pass to enable multi-turn
    script: str = "latinic"         # cyrillic or latinic
    language: str = "sr"            # sr, mk, bg


class ChatResponse(BaseModel):
    question: str
    answer: str
    # sources: list
    # confidence: float
