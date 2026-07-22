"""
run.py
~~~~~~
Entry point — initialises the QA system and runs an interactive loop.

The ChromaDB vector store must already be populated by the
university-scraper project.  This chatbot opens it read-only.
"""

import logging

from src.core.vector_store import VectorStoreManager
from src.core.llm_manager import LLMManager
from src.chains.qa_chain import UniversityQASystem
from src.processing.translator import ScriptTranslator
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_system():
    logger.info("Initializing system components...")

    translator   = ScriptTranslator()
    llm_manager  = LLMManager(settings)
    vector_store = VectorStoreManager(settings)

    qa_system = UniversityQASystem(llm_manager, vector_store, translator)
    logger.info("System ready.")
    return qa_system


def print_response(response: dict):
    print(f"\nOdgovor:\n{response['answer']}")
    if response["sources"]:
        print("\nIzvori:")
        for src in response["sources"]:
            print(f"  {src}")


if __name__ == "__main__":
    qa_system = initialize_system()

    print("\n" + "=" * 60)
    print("  Univerzitetski Chatbot")
    print("  (ChromaDB se otvara samo za citanje — scraper ga puni)")
    print("  Ukucaj 'kraj' za izlaz")
    print("=" * 60 + "\n")

    while True:
        question = input("\nPitanje (ili 'kraj'): ").strip()
        if question.lower() in ("kraj", "quit", "exit"):
            break
        if not question:
            continue
        response = qa_system.query(question)
        print_response(response)