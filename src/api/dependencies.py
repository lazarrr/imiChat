# src/api/dependencies.py
"""
Shared, process-wide application dependencies.

The QA system (translator + LLM + vector store) is expensive to build and
is stateless across requests, so a single instance is created at import
time and reused by every route.
"""

from __future__ import annotations

import logging

from config.settings import settings
from src.chains.qa_chain import UniversityQASystem
from src.core.llm_manager import LLMManager
from src.core.vector_store import VectorStoreManager
from src.processing.translator import ScriptTranslator

logger = logging.getLogger(__name__)


def initialize_system() -> UniversityQASystem:
    """Initialize all components."""
    logger.info("Initializing system components...")

    translator = ScriptTranslator()
    llm_manager = LLMManager(settings)
    vector_store = VectorStoreManager(settings)

    qa_system = UniversityQASystem(llm_manager, vector_store, translator)
    logger.info("System initialized successfully!")
    return qa_system


qa_system = initialize_system()
