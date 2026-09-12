"""
run.py
~~~~~~
Entry point — shows how to index both local files and web pages,
then run the QA loop with formatted source citations.
"""

import logging
from src.core.vector_store import VectorStoreManager
from src.core.llm_manager import LLMManager
from src.chains.qa_chain import UniversityQASystem
from src.processing.translator import ScriptTranslator
from src.processing.document_loader import DocumentProcessor, WebLoader
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_system():
    logger.info("Initializing system components...")

    translator   = ScriptTranslator()
    llm_manager  = LLMManager(settings)
    vector_store = VectorStoreManager(settings)

    if not vector_store.is_populated():
        logger.info("Collection is empty — indexing documents...")

        # ── Local files (PDF + DOCX, headers/footers stripped) ───────── #
        doc_processor = DocumentProcessor(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        file_docs = doc_processor.process_directory("./data/raw")
        logger.info(f"Local files: {len(file_docs)} chunks")

        # ── Web pages ─────────────────────────────────────────────────── #
        web_loader = WebLoader(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        web_urls = [
            # Add your university web page URLs here
            "https://imi.pmf.kg.ac.rs/vesti/11836-novi-smerovi-na-oas-matematike-i-oas-informatike",
        ]
        web_docs = web_loader.load_urls(web_urls) if web_urls else []
        logger.info(f"Web pages: {len(web_docs)} chunks")

        all_docs = file_docs + web_docs
        if all_docs:
            vector_store.add_documents(all_docs)
    else:
        logger.info("Collection already populated — skipping indexing.")

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
    used_chunk_ids: list[str] = []

    while True:
        question = input("\nPitanje (ili 'kraj'): ").strip()
        if question.lower() in ("kraj", "quit", "exit"):
            break
        if question.lower() == "novo":
            conversation_history = []
            used_chunk_ids = []
            print("Zapocet novi razgovor.")
            continue
        if not question:
            continue
        response = qa_system.query(
            question, conversation_history, exclude_chunk_ids=used_chunk_ids
        )
        print_response(response)
        used_chunk_ids = list(response.get("chunk_ids") or [])
        conversation_history.append({"role": "user", "content": question})
        conversation_history.append({"role": "assistant", "content": response["answer"]})