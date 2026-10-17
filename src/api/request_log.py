"""
request_log.py
~~~~~~~~~~~~~~
Append-only JSONL trace log: one line per answered chat turn.

Why a separate file and not the normal log
------------------------------------------
``logging.basicConfig(level=INFO)`` writes human-readable lines to stderr
for operators.  This module writes *structured* records to a file for a
separate visualization project: every line is a self-contained JSON object,
so that project needs no access to this codebase, the vector store or
Ollama — only the file.

What a record contains
----------------------
The whole pipeline for one turn, so the interesting question ("what did the
system actually search for, and why?") is answerable from the file alone:

    request    id, timestamp, session, endpoint, client, rate-limit state
    question   as typed, normalized (Cyrillic → Latin), language, unclear?
    history    the last turns the rewrite was allowed to see
    rewrite    the exact prompt sent to the LLM, its raw output, why the
               result was accepted or rejected, the final search question
    retrieval  every query tried (rewrite, original, +previous question,
               +previous answer) with hit counts and which one won
    chunks     the chunks that went into the LLM prompt, with metadata and
               distance scores — what the answer is grounded in
    prompt     the exact message list sent to the answering model
    answer     full text, sources block, per-stage timings, errors

Layout on disk
--------------
``logs/requests.jsonl`` grows one line per turn and rotates at
``REQUEST_LOG_MAX_BYTES`` into ``requests.jsonl.1`` … ``.N``.  Rotation is
size-based only, so a long-running server cannot fill the disk.

The record is written once, after the answer has streamed, because only
then are the answer and the token timings known.

Schema
------
The top-level key set is pinned by ``SCHEMA_VERSION``; a reader should
tolerate unknown keys and missing optional ones.  ``failed`` is set when the
turn ended with an error, in which case ``error`` explains it and later
sections may be absent.

Writing from the event loop
---------------------------
The handler writes synchronously and briefly holds a lock, which is fine for
a chat endpoint (one small append per turn).  Records are written with a
single ``write()`` call so concurrent turns in different worker threads
cannot interleave within one line.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

SCHEMA_VERSION = 1

LOGGER_NAME = "university_chatbot.request_log"


class _JsonLineFormatter(logging.Formatter):
    """Emit the record's ``payload`` attribute as one compact JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(record.payload, ensure_ascii=False, default=str)


class RequestLogger:
    """
    Writes one JSON object per chat turn to ``settings.REQUEST_LOG_PATH``.

    A thin object rather than module-level functions so tests can point it at
    a temporary file: ``RequestLogger(path)``.  ``write`` is the only method
    the API needs.
    """

    def __init__(self, path, enabled: bool = True, max_bytes: int = 20_000_000,
                 backups: int = 5):
        self.path = Path(path)
        self.enabled = enabled
        self.max_bytes = max_bytes
        self.backups = backups
        self._logger: logging.Logger | None = None
        if enabled:
            self._logger = self._build_logger()

    def _build_logger(self) -> logging.Logger:
        """
        A dedicated logger with its own rotating file handler.

        ``propagate = False`` keeps these JSON lines out of the root
        logger's stderr stream, where they would drown the readable logs.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(f"{LOGGER_NAME}.{uuid.uuid4().hex[:8]}")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        handler = RotatingFileHandler(
            self.path, maxBytes=self.max_bytes, backupCount=self.backups,
            encoding="utf-8", delay=True,  # don't create the file until turn 1
        )
        handler.setFormatter(_JsonLineFormatter())
        logger.addHandler(handler)
        return logger

    def write(self, record: dict) -> None:
        """
        Append one turn's record.  Never raises: a full disk or a permission
        error must not turn a good answer into a 500.
        """
        if self._logger is None:
            return
        try:
            self._logger.info("", extra={"payload": record})
        except Exception:  # pragma: no cover - defensive, see docstring
            logging.getLogger(__name__).exception("Could not write request log record")


def new_request_id() -> str:
    """Short unique id — the visualizer groups a turn's records by this."""
    return uuid.uuid4().hex[:12]


def utc_now() -> str:
    """ISO-8601 UTC timestamp with millisecond precision and a Z suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def truncate(text, limit: int) -> tuple[str, bool]:
    """
    ``(text, truncated)`` — cut to ``limit`` chars for the log file.

    Prompts and chunk bodies are what make a record worth reading, but an
    answer can be arbitrarily long, so each text field is capped separately
    and the reader is told when something was cut rather than silently
    getting a partial value.
    """
    if text is None:
        return "", False
    text = str(text)
    if limit <= 0 or len(text) <= limit:
        return text, False
    return text[:limit], True


def strip_sources_block(answer: str) -> str:
    """
    The answer without its trailing ``--- Izvori:`` block.

    That block is generated by the chain, not by the model, and it is what
    the API keeps out of the session history, so the log stores the model's
    own text here and the rendered citations under ``answer.sources``.
    """
    idx = answer.rfind("\n\n---\n")
    return answer[:idx].strip() if idx != -1 else answer


def build_record(
    *,
    request_id: str,
    session_id: str | None,
    endpoint: str,
    client: str | None,
    user_agent: str | None,
    question: str,
    trace: dict,
    answer: str,
    history_turns: int,
    duration_ms: float,
    max_text: int,
    max_excluded: int = 40,
    error: str | None = None,
) -> dict:
    """
    Assemble one JSONL record: the chain's trace plus the request-level facts
    the API knows and the pieces only the API can compute (duration, error).

    The visualizer reads these records, so the shape is the contract.  Text
    fields are capped at ``max_text`` chars each and each capped field comes
    with its real length, so a truncated prompt is visibly truncated:

        ``<key>``             the capped text
        ``<key>_chars``       characters present
        ``<key>_full_chars``  characters before capping
        ``<key>_truncated``   whether anything was cut
    """
    rewrite = trace.get("rewrite") or {}
    retrieval = trace.get("retrieval") or {}
    prompt = trace.get("prompt") or {}

    def capped(value, key: str) -> dict:
        text, cut = truncate(value, max_text)
        return {
            # None stays None: "the rewrite was never attempted" must be
            # distinguishable from "the rewrite ran and produced nothing".
            key: text if value is not None else None,
            f"{key}_chars": len(text),
            f"{key}_full_chars": len(str(value)) if value is not None else 0,
            f"{key}_truncated": cut,
        }

    # Chunks are the evidence the answer is grounded in, so their bodies are
    # kept (up to the same cap) rather than reduced to file names.  The whole
    # vector-store metadata dict travel with them: the visualizer shows every
    # field, including ones this project does not interpret.
    chunks = []
    for chunk in trace.get("chunks") or []:
        text, cut = truncate(chunk.get("text", ""), max_text)
        chunks.append({
            **{k: v for k, v in chunk.items() if k != "text"},
            "metadata": dict(chunk.get("metadata") or {}),
            "text": text,
            "text_truncated": cut,
        })

    # Chunks retrieval threw away before the prompt: they explain why the
    # answer had to work with what it got.  Bodies are capped like the kept
    # ones; a turn can drop a few dozen candidates, so the list is bounded.
    excluded = []
    for chunk in (trace.get("chunks_excluded") or [])[:max_excluded]:
        text, cut = truncate(chunk.get("text", ""), max_text)
        excluded.append({
            **{k: v for k, v in chunk.items() if k != "text"},
            "metadata": dict(chunk.get("metadata") or {}),
            "text": text,
            "text_truncated": cut,
        })

    messages = prompt.get("messages") or [{}]
    # The text block the rewrite saw, as a list of "Korisnik: …"/"Asistent: …"
    # lines.  Counted here so the numbers describe what is actually in the
    # field, not the whole session history.
    history_lines = [
        line for line in (rewrite.get("history_text") or "").splitlines() if line.strip()
    ]
    record = {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "timestamp": utc_now(),
        "endpoint": endpoint,
        "session_id": session_id,
        "client": client,
        "user_agent": user_agent,
        "duration_ms": round(duration_ms, 1),
        "question": question,
        "search_question": trace.get("search_question", question),
        "unclear": trace.get("unclear", False),
        # Flat summary so a list view needs no nested lookups.
        "summary": {
            "rewritten": bool(rewrite.get("changed")),
            "rewrite_result": rewrite.get("result"),
            "retrieval_strategy": retrieval.get("winning_strategy", "none"),
            "retrieval_attempts": len(retrieval.get("attempts") or []),
            "chunks": len(trace.get("chunks") or []),
            "chunks_excluded": len(trace.get("chunks_excluded") or []),
            "chunks_excluded_total": trace.get("chunks_excluded_total", 0),
            "sources": len(trace.get("sources") or []),
            "answer_chars": len(answer),
            "failed": error is not None,
        },
        # What the rewrite was allowed to see: the last 1–2 turns, not the
        # whole session.
        "history": {
            "session_turns": history_turns,
            "messages_included": len(history_lines),
            "truncated_for_rewrite": len(history_lines) < history_turns * 2,
            **capped(rewrite.get("history_text"), "text"),
        },
        # What the rewrite was given ("what was used to rewrite the initial
        # question") and what it produced, guards included.
        "rewrite": {
            "result": rewrite.get("result"),
            "matched_pattern": rewrite.get("matched_pattern"),
            "normalized": rewrite.get("normalized", question),
            "question_in": rewrite.get("question_in", question),
            "question_out": rewrite.get("question_out", question),
            "changed": rewrite.get("changed", False),
            "raw_response": rewrite.get("raw_response"),
            "rejected": rewrite.get("rejected"),
            **capped(rewrite.get("prompt"), "prompt"),
        },
        # Every query tried, in order, with its hit count — including the ones
        # that found nothing — and which strategy finally won.
        "retrieval": {
            "winning_strategy": retrieval.get("winning_strategy", "none"),
            "attempts": retrieval.get("attempts") or [],
            "excluded_total": trace.get("chunks_excluded_total", 0),
            "excluded_kept": len(excluded),
        },
        # The chunks that went into the prompt.
        "chunks": chunks,
        # Chunks that were retrieved but rejected, with the reason and the
        # search attempt they came from.  Capped at ``max_excluded``; the real
        # number is ``summary.chunks_excluded_total``.
        "chunks_excluded": excluded,
        # How many were dropped in total, even when the list above is capped.
        "chunks_excluded_total": trace.get("chunks_excluded_total", 0),
        "chunks_excluded_truncated": (
            len(trace.get("chunks_excluded") or []) > len(excluded)
        ),
        "prompt": {
            "char_count": prompt.get("char_count", 0),
            "system_chars": len(prompt.get("system_prompt") or ""),
            "system_prompt": prompt.get("system_prompt") or "",
            **capped(messages[-1].get("content"), "user_content"),
        },
        "answer": {
            "text": strip_sources_block(answer),
            "full_chars": len(answer),
            "sources": trace.get("sources") or [],
        },
        "timings_ms": trace.get("timings_ms") or {},
        "failed": error is not None,
    }
    if error is not None:
        record["error"] = error
    return record
