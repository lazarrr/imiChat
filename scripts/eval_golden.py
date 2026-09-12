#!/usr/bin/env python3
"""
eval_golden.py
~~~~~~~~~~~~~~
Small golden-set evaluation for the University RAG pipeline.

Metrics
-------
retrieval hit@k      — at least one retrieved chunk comes from an expected source
citation correctness — cited [Izvor N] numbers exist and point at expected sources
refusal rate         — unanswerable questions receive the fixed refusal phrase
false refusal rate   — answerable questions are not refused

Usage
-----
    conda activate uni
    cd university-chatbot

    python scripts/eval_golden.py                 # retrieval metrics only (dense)
    python scripts/eval_golden.py --with-llm      # + citation/refusal metrics
    python scripts/eval_golden.py --hybrid        # dense + BM25 fusion
    python scripts/eval_golden.py --no-hybrid     # force dense-only
    python scripts/eval_golden.py --json report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402
from src.chains.retriever import HybridRetriever  # noqa: E402
from src.core.vector_store import VectorStoreManager  # noqa: E402
from src.processing.translator import ScriptTranslator  # noqa: E402

logger = logging.getLogger("eval_golden")

CITATION_RE = re.compile(r"\[\s*izvor\s*(\d+)\s*\]", re.IGNORECASE)
DEFAULT_GOLDEN = ROOT / "data" / "eval" / "golden_set.json"


def _matches_expected(metadata: dict, expected: list[str]) -> bool:
    haystack = " ".join(
        str(metadata.get(key) or "") for key in ("filename", "source", "url")
    ).lower()
    return any(token.lower() in haystack for token in expected)


def _answer_is_refusal(answer: str) -> bool:
    from src.chains.qa_chain import _is_refusal

    return _is_refusal(answer)


def evaluate_item_retrieval(retriever: HybridRetriever, item: dict, k: int) -> dict:
    query = item["question"]
    chunks = retriever.retrieve(query, k=k)
    expected = item.get("expected_sources") or []

    top_sources: list[str] = []
    hit = False
    for chunk in chunks:
        meta = chunk.metadata or {}
        top_sources.append(str(meta.get("filename") or meta.get("source") or "?"))
        if expected and _matches_expected(meta, expected):
            hit = True

    return {"hit": hit if expected else None, "retrieved": top_sources}


def evaluate_item_llm(qa_system, item: dict) -> dict:
    response = qa_system.query(item["question"])
    answer = response["answer"]
    structured = response.get("sources_structured") or []
    number_to_meta = {row["number"]: row for row in structured}

    cited_numbers = [int(n) for n in CITATION_RE.findall(answer)]
    invalid = [n for n in cited_numbers if n not in number_to_meta]
    expected = item.get("expected_sources") or []
    cited_expected = [
        n
        for n in cited_numbers
        if n in number_to_meta and _matches_expected(number_to_meta[n], expected)
    ]

    return {
        "answer": answer,
        "cited": cited_numbers,
        "invalid_citations": invalid,
        "cited_expected": bool(cited_expected),
        "refusal": _answer_is_refusal(answer),
        "sources": [row["label"] for row in structured],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    parser.add_argument("--k", type=int, default=settings.TOP_K_RESULTS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--with-llm", action="store_true",
                        help="also generate answers and score citations/refusals")
    parser.add_argument("--hybrid", action="store_true",
                        help="force-enable dense + BM25 fusion for this run")
    parser.add_argument("--no-hybrid", action="store_true",
                        help="force-disable BM25 fusion (dense-only)")
    parser.add_argument("--json", default="", help="write the full report to this path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    golden_path = Path(args.golden)
    items = json.loads(golden_path.read_text(encoding="utf-8"))
    if args.limit:
        items = items[: args.limit]

    if args.hybrid:
        settings.HYBRID_RETRIEVAL_ENABLED = True
    if args.no_hybrid:
        settings.HYBRID_RETRIEVAL_ENABLED = False

    translator = ScriptTranslator()
    vector_store = VectorStoreManager(settings)
    retriever = HybridRetriever(vector_store, settings)

    qa_system = None
    if args.with_llm:
        from src.chains.qa_chain import UniversityQASystem
        from src.core.llm_manager import LLMManager

        qa_system = UniversityQASystem(LLMManager(settings), vector_store, translator)

    rows: list[dict] = []
    for item in items:
        normalized = translator.normalize_query(item["question"])
        retrieval = evaluate_item_retrieval(retriever, {**item, "question": normalized}, args.k)
        row = {
            "question": item["question"],
            "expect_refusal": bool(item.get("expect_refusal")),
            "expected_sources": item.get("expected_sources") or [],
            **retrieval,
        }
        if qa_system is not None:
            row.update(evaluate_item_llm(qa_system, item))
        rows.append(row)
        print(
            f"{'HIT ' if row['hit'] else ('MISS' if row['hit'] is False else '----')} "
            f"| {item['question']}"
        )

    answerable = [r for r in rows if r["expected_sources"]]
    refusal_items = [r for r in rows if r["expect_refusal"]]

    hit_rate = (
        sum(1 for r in answerable if r["hit"]) / len(answerable) if answerable else 0.0
    )

    summary: dict = {
        "items": len(rows),
        f"retrieval_hit@{args.k}": round(hit_rate, 4),
        "hybrid": settings.HYBRID_RETRIEVAL_ENABLED,
        "with_llm": args.with_llm,
    }

    if args.with_llm:
        total_citations = sum(len(r["cited"]) for r in rows)
        valid_citations = sum(
            len(r["cited"]) - len(r["invalid_citations"]) for r in rows
        )
        summary["citation_validity"] = round(
            valid_citations / total_citations if total_citations else 1.0, 4
        )
        summary["answers_with_expected_citation"] = round(
            sum(1 for r in answerable if r.get("cited_expected")) / len(answerable)
            if answerable
            else 0.0,
            4,
        )
        summary["refusal_rate"] = round(
            sum(1 for r in refusal_items if r["refusal"]) / len(refusal_items)
            if refusal_items
            else 0.0,
            4,
        )
        summary["false_refusal_rate"] = round(
            sum(1 for r in answerable if r["refusal"]) / len(answerable)
            if answerable
            else 0.0,
            4,
        )

    print("\n=== Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nReport written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
