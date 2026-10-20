"""
run.py
~~~~~~
Entry point — runs the QA loop with formatted source citations.

    python run.py                   # interactive Q&A
    python run.py eval              # score retrieval on the golden questions
    python run.py eval --compare    # ... with vs. without each optional
                                    #     retrieval step (synonyms, hybrid
                                    #     search, reranker)

The vector store is populated by the scraper project
(C:\\Projects\\scraper — `python run.py ingest`), which labels every
chunk with its location and study-program version.  Nothing is indexed
from here: unlabeled chunks would bring back the old/new program mix-ups.
"""

import argparse
import logging
from pathlib import Path

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


def chat_loop():
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


def eval_retrieval(args):
    from src.utils.retrieval_eval import load_cases, print_comparison, print_eval

    if args.questions:
        path = Path(args.questions)
    else:
        path = Path(settings.EVAL_QUESTIONS_FILE)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
    cases = load_cases(path)
    print(f"{len(cases)} golden question(s) from {path}")

    vector_store = VectorStoreManager(settings)
    # The store clamps every search to MAX_CONTEXT_CHUNKS; report the k that
    # is actually used so hit@k is not mislabelled.
    k = min(
        args.k or settings.TOP_K_RESULTS,
        getattr(settings, "MAX_CONTEXT_CHUNKS", 4),
    )
    if args.compare:
        print_comparison(vector_store, cases, k)
    else:
        print_eval(vector_store, cases, k)


def parse_args():
    parser = argparse.ArgumentParser(
        description="IMI chatbot — interactive Q&A, or `eval` to score retrieval."
    )
    commands = parser.add_subparsers(dest="command")
    ev = commands.add_parser("eval", help="score retrieval on the golden questions (hit@k, MRR)")
    ev.add_argument(
        "--compare", action="store_true",
        help="score baseline, each optional retrieval step and all steps together",
    )
    ev.add_argument("--questions", help="golden questions YAML (default: EVAL_QUESTIONS_FILE)")
    ev.add_argument("-k", type=int, help="chunks per question (default: TOP_K_RESULTS)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "eval":
        eval_retrieval(args)
    else:
        chat_loop()
