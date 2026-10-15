"""
retrieval_eval.py
~~~~~~~~~~~~~~~~~
Scores retrieval on golden questions, so the optional retrieval steps
(QUERY_SYNONYMS, HYBRID_SEARCH, RERANKER_ENABLED) can be compared with
and without:

    python run.py eval              # with the steps switched on in config
    python run.py eval --compare    # baseline vs. each step vs. all steps

The questions use the scraper's eval_questions.yaml format.  A question is
a hit when an expected url / breadcrumb / text substring appears in one of
the top-k chunks (the same rule as the scraper's `run.py eval`; matching
ignores case, diacritics and script).  MRR also rewards finding it near
the top: 1.0 when it is the first chunk, 0.5 the second, and so on.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from pathlib import Path

from src.core.keyword_search import fold

# variant → (use_synonyms, use_hybrid, use_reranker)
VARIANTS = {
    "baseline": (False, False, False),
    "synonyms": (True, False, False),
    "hybrid":   (False, True, False),
    "reranker": (False, False, True),
    "all":      (True, True, True),
}


def _match_fold(text: str) -> str:
    text = re.sub(r"[‐-―−]", "-", fold(text))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s-]", " ", text)).strip()


def load_cases(path: str | Path) -> list[dict]:
    import yaml  # only needed for evaluation

    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []


def find_rank(expect: dict, documents: list[str], metadatas: list[dict]) -> int | None:
    """1-based rank of the first chunk matching an expected substring, else None."""
    needles = {
        field: [_match_fold(n) for n in expect.get(field, []) if str(n).strip()]
        for field in ("url", "breadcrumb", "text")
    }
    for rank, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        meta = meta or {}
        haystacks = {
            "url": f"{meta.get('url', '')} {meta.get('page_url', '')}",
            "breadcrumb": f"{meta.get('breadcrumb', '')} {meta.get('version_label', '')}",
            "text": doc or "",
        }
        for field, field_needles in needles.items():
            haystack = _match_fold(haystacks[field])
            if any(needle in haystack for needle in field_needles):
                return rank
    return None


def evaluate(store, cases: list[dict], k: int) -> dict:
    """Ranks per question, hits, MRR and average seconds per question."""
    ranks: list[int | None] = []
    started = time.perf_counter()
    for case in cases:
        results = store.similarity_search(case["question"], k=k)
        ranks.append(find_rank(case.get("expect", {}), results["documents"][0], results["metadatas"][0]))
    n = len(cases) or 1
    return {
        "ranks": ranks,
        "hits": sum(rank is not None for rank in ranks),
        "mrr": sum(1.0 / rank for rank in ranks if rank) / n,
        "seconds": (time.perf_counter() - started) / n,
    }


def print_eval(store, cases: list[dict], k: int, out: Callable[[str], None] = print) -> None:
    """Score the store as configured and print one line per question."""
    result = evaluate(store, cases, k)
    out(f"\n{'#':>2}  {'hit':4} {'rank':>4}  question")
    for i, (case, rank) in enumerate(zip(cases, result["ranks"]), start=1):
        out(f"{i:>2}  {'yes' if rank else 'NO':4} {rank or '-':>4}  {case['question']}")
    out(
        f"\nhit@{k}: {result['hits']}/{len(cases)}   MRR: {result['mrr']:.3f}   "
        f"{result['seconds']:.2f} s/question"
    )


def print_comparison(store, cases: list[dict], k: int, out: Callable[[str], None] = print) -> None:
    """Score every variant in VARIANTS and print them side by side."""
    saved = (store.use_synonyms, store.use_hybrid, store.use_reranker)
    store_logger = logging.getLogger(store.__class__.__module__)
    saved_level = store_logger.level
    store_logger.setLevel(logging.WARNING)  # keep per-question diagnostics out of the table
    results: dict[str, dict | str] = {}
    try:
        for name, flags in VARIANTS.items():
            out(f"Scoring {name}...")
            store.use_synonyms, store.use_hybrid, store.use_reranker = flags
            try:
                if cases:  # untimed first search loads the step's components
                    store.similarity_search(cases[0]["question"], k=k)
                results[name] = evaluate(store, cases, k)
            except Exception as e:  # e.g. the reranker model cannot be downloaded
                results[name] = f"failed: {e}"
    finally:
        store.use_synonyms, store.use_hybrid, store.use_reranker = saved
        store_logger.setLevel(saved_level)

    scored = [name for name, r in results.items() if isinstance(r, dict)]
    out("\nRank of the first matching chunk per question (- = not in the top k):")
    out(f"{'#':>2}  " + " ".join(f"{name:>8}" for name in scored) + "  question")
    for i, case in enumerate(cases):
        cells = " ".join(f"{results[name]['ranks'][i] or '-':>8}" for name in scored)
        out(f"{i + 1:>2}  {cells}  {case['question']}")

    out(f"\n{'variant':10} {'hit@' + str(k):>7} {'MRR':>6} {'s/question':>11}")
    for name, r in results.items():
        if isinstance(r, dict):
            out(f"{name:10} {r['hits']:>3}/{len(cases):<3} {r['mrr']:>6.3f} {r['seconds']:>11.2f}")
        else:
            out(f"{name:10} {r}")
