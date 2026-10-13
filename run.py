"""
run.py
~~~~~~
Entry point — runs the QA loop with formatted source citations.

The vector store is populated by the scraper project
(C:\\Projects\\scraper — `python run.py ingest`), which labels every
chunk with its location and study-program version.  Nothing is indexed
from here: unlabeled chunks would bring back the old/new program mix-ups.
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

    if not vector_store.is_populated():
        logger.warning(
            "Collection '%s' at %s is empty — run the scraper first "
            "(C:\\Projects\\scraper: python run.py ingest).",
            vector_store.collection_name, settings.VECTOR_DB_PATH,
        )

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

    conversation_history: list[dict[str, str]] = []

    while True:
        question = input("\nPitanje (ili 'kraj'): ").strip()
        if question.lower() in ("kraj", "quit", "exit"):
            break
        if question.lower() == "novo":
            conversation_history = []
            print("Zapocet novi razgovor.")
            continue
        if not question:
            continue
        response = qa_system.query(question, conversation_history)
        print_response(response)
        conversation_history.append({"role": "user", "content": question})
        conversation_history.append({"role": "assistant", "content": response["answer"]})