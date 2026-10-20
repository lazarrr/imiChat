"""
Unit tests for the request trace log (src/api/request_log.py).

The log file is the only interface between this project and the separate
visualization project, so these tests pin two things: that each line is
valid standalone JSON, and that the fields the visualizer needs are present
and correctly derived from the pipeline trace.

No FastAPI, Ollama or ChromaDB involved — the record builder is pure.

    python -m unittest tests.test_request_log -v
"""

import json
import logging
import tempfile
import unittest
from pathlib import Path

from src.api.request_log import (
    RequestLogger,
    SCHEMA_VERSION,
    build_record,
    strip_sources_block,
    truncate,
)

TRACE = {
    "unclear": True,
    "search_question": "Koji su preduslovi za Uvod u programiranje?",
    "rewrite": {
        "question_in": "Koji su preduslovi za njega?",
        "question_out": "Koji su preduslovi za Uvod u programiranje?",
        "normalized": "Koji su preduslovi za njega?",
        "changed": True,
        "result": "accepted",
        "matched_pattern": "njega",
        "raw_response": "Koji su preduslovi za Uvod u programiranje?",
        "rejected": None,
        "prompt": "Ti si modul za prepisivanje korisničkih pitanja...\n"
                  "Istorija razgovora:\nKorisnik: Koliko ESPB ima Uvod?",
        "history_text": "Korisnik: Koliko ESPB ima Uvod u programiranje?\n"
                        "Asistent: Uvod u programiranje nosi 6 ESPB bodova.",
    },
    "retrieval": {
        "winning_strategy": "search_question",
        "attempts": [{
            "rank": 1, "strategy": "search_question",
            "query": "Koji su preduslovi za Uvod u programiranje?",
            "query_chars": 45, "hits": 2, "top_distance": 0.21,
        }],
    },
    "chunks": [{
        "rank": 1, "source": "informatika.pdf#page=3", "filename": "informatika.pdf",
        "type": "pdf", "page": 3, "url": "", "distance": 0.21,
        "chars": 40, "text": "Predmet nema preduslova.",
        # The whole vector-store metadata, extra fields included.
        "metadata": {
            "source": "informatika.pdf#page=3", "filename": "informatika.pdf",
            "type": "pdf", "page": 3, "llm_processed": False,
            "llm_original_chars": 830, "version_label": "akreditacija 2026 | aktuelni",
        },
    }],
    "chunks_excluded": [{
        "chunk_id": "old-1", "reason": "excluded_legacy_version", "distance": 0.12,
        "metadata": {"filename": "plan2019.pdf", "status": "legacy",
                     "version_label": "akreditacija 2019 | prethodni"},
        "text": "Po starom planu bilo je tri smera.",
        "attempt_rank": 1, "attempt_strategy": "search_question", "attempt_kept": True,
    }],
    "chunks_excluded_total": 7,
    "prompt": {
        "messages": [
            {"role": "system", "content": "Vi ste korisni univerzitetski asistent."},
            {"role": "user", "content": "Kontekst:\n[Izvor 1]\n...\n\nPitanje: Koji su preduslovi?"},
        ],
        "system_prompt": "Vi ste korisni univerzitetski asistent.",
        "char_count": 120,
    },
    "sources": ["[1] informatika.pdf — strana 3"],
    "timings_ms": {"rewrite": 812.4, "retrieval": 96.1, "answer": 3204.7, "total": 4113.2},
}


def make_record(**overrides):
    kwargs = dict(
        request_id="abc123",
        session_id="sess-1",
        endpoint="/chat/stream",
        client="127.0.0.1",
        user_agent="curl/8.4.0",
        question="Koji su preduslovi za njega?",
        trace=TRACE,
        answer="Uvod u programiranje nema preduslova. [Izvor 1]\n\n---\nIzvori:\n[1] informatika.pdf — strana 3",
        history_turns=1,
        duration_ms=4123.456,
        max_text=8000,
        error=None,
    )
    kwargs.update(overrides)
    return build_record(**kwargs)


class RecordBuilderTests(unittest.TestCase):
    def test_record_is_json_serializable(self):
        record = make_record()

        # ensure_ascii=False: the file holds Serbian text, not \u escapes.
        line = json.dumps(record, ensure_ascii=False)
        self.assertEqual(json.loads(line)["request_id"], "abc123")

    def test_rewrite_section_explains_what_was_used(self):
        rewrite = make_record()["rewrite"]

        self.assertEqual(rewrite["result"], "accepted")
        self.assertEqual(rewrite["matched_pattern"], "njega")
        self.assertEqual(rewrite["question_in"], "Koji su preduslovi za njega?")
        self.assertEqual(rewrite["question_out"], "Koji su preduslovi za Uvod u programiranje?")
        self.assertTrue(rewrite["changed"])
        # The actual prompt and history the rewrite was given.
        self.assertIn("Ti si modul za prepisivanje", rewrite["prompt"])
        self.assertEqual(rewrite["prompt_truncated"], False)

    def test_history_block_reports_its_size(self):
        history = make_record()["history"]

        self.assertEqual(history["session_turns"], 1)
        # The block handed to the rewrite, counted rather than guessed at.
        self.assertEqual(history["messages_included"], 2)
        self.assertFalse(history["truncated_for_rewrite"])
        self.assertIn("Asistent:", history["text"])
        self.assertFalse(history["text_truncated"])

    def test_history_block_distinguishes_session_from_what_the_rewrite_saw(self):
        # A long session: the rewrite only gets the last 1–2 turns.
        record = make_record(history_turns=9)

        self.assertEqual(record["history"]["session_turns"], 9)
        self.assertEqual(record["history"]["messages_included"], 2)
        self.assertTrue(record["history"]["truncated_for_rewrite"])

    def test_summary_block_flattens_the_interesting_facts(self):
        summary = make_record()["summary"]

        self.assertEqual(summary, {
            "rewritten": True,
            "rewrite_result": "accepted",
            "retrieval_strategy": "search_question",
            "retrieval_attempts": 1,
            "chunks": 1,
            "chunks_excluded": 1,
            "chunks_excluded_total": 7,
            "sources": 1,
            "answer_chars": len(
                "Uvod u programiranje nema preduslova. [Izvor 1]"
                "\n\n---\nIzvori:\n[1] informatika.pdf — strana 3"
            ),
            "failed": False,
        })

    def test_retrieval_attempts_and_winner_are_kept(self):
        retrieval = make_record()["retrieval"]

        self.assertEqual(retrieval["winning_strategy"], "search_question")
        self.assertEqual(retrieval["attempts"][0]["hits"], 2)
        self.assertEqual(retrieval["attempts"][0]["top_distance"], 0.21)

    def test_chunks_keep_their_text_and_metadata(self):
        chunk = make_record()["chunks"][0]

        self.assertEqual(chunk["filename"], "informatika.pdf")
        self.assertEqual(chunk["page"], 3)
        self.assertEqual(chunk["text"], "Predmet nema preduslova.")
        self.assertFalse(chunk["text_truncated"])

    def test_answer_text_excludes_the_generated_sources_block(self):
        answer = make_record()["answer"]

        self.assertEqual(answer["text"], "Uvod u programiranje nema preduslova. [Izvor 1]")
        self.assertEqual(answer["sources"], ["[1] informatika.pdf — strana 3"])
        self.assertGreater(answer["full_chars"], len(answer["text"]))

    def test_prompt_section_records_the_user_message(self):
        prompt = make_record()["prompt"]

        self.assertIn("Pitanje: Koji su preduslovi?", prompt["user_content"])
        self.assertEqual(prompt["system_prompt"],
                         "Vi ste korisni univerzitetski asistent.")
        self.assertEqual(prompt["char_count"], 120)

    def test_chunks_carry_the_full_vector_store_metadata(self):
        chunk = make_record()["chunks"][0]

        self.assertEqual(chunk["metadata"]["llm_original_chars"], 830)
        self.assertEqual(chunk["metadata"]["version_label"], "akreditacija 2026 | aktuelni")
        # and the flattened keys the list view uses still work
        self.assertEqual(chunk["filename"], "informatika.pdf")

    def test_dropped_chunks_are_logged_with_their_reason(self):
        record = make_record()
        excluded = record["chunks_excluded"]

        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["reason"], "excluded_legacy_version")
        self.assertEqual(excluded[0]["metadata"]["status"], "legacy")
        self.assertIn("tri smera", excluded[0]["text"])
        self.assertTrue(excluded[0]["attempt_kept"])
        self.assertEqual(record["retrieval"]["excluded_total"], 7)
        self.assertEqual(record["retrieval"]["excluded_kept"], 1)
        self.assertEqual(record["summary"]["chunks_excluded"], 1)
        self.assertEqual(record["summary"]["chunks_excluded_total"], 7)

    def test_the_dropped_chunk_list_is_capped_but_counted(self):
        trace = {**TRACE, "chunks_excluded": [
            {"chunk_id": f"c{i}", "reason": "excluded_threshold", "distance": 0.9,
             "metadata": {}, "text": "x"}
            for i in range(100)
        ], "chunks_excluded_total": 250}

        record = make_record(trace=trace, max_excluded=10)

        self.assertEqual(len(record["chunks_excluded"]), 10)
        self.assertTrue(record["chunks_excluded_truncated"])
        # The record-level total is what the visualizer shows ("10 of 250").
        self.assertEqual(record["chunks_excluded_total"], 250)
        self.assertEqual(record["retrieval"]["excluded_total"], 250)
        self.assertEqual(record["summary"]["chunks_excluded_total"], 250)

    def test_no_dropped_chunks_is_an_empty_list_not_missing(self):
        trace = {**TRACE, "chunks_excluded": [], "chunks_excluded_total": 0}

        record = make_record(trace=trace)

        self.assertEqual(record["chunks_excluded"], [])
        self.assertFalse(record["chunks_excluded_truncated"])

    def test_timings_are_carried_through(self):
        self.assertEqual(make_record()["timings_ms"]["rewrite"], 812.4)

    def test_long_text_is_capped_and_flagged(self):
        record = make_record(
            trace={**TRACE, "rewrite": {**TRACE["rewrite"], "prompt": "x" * 500}},
            max_text=100,
        )

        self.assertEqual(len(record["rewrite"]["prompt"]), 100)
        self.assertEqual(record["rewrite"]["prompt_full_chars"], 500)
        self.assertTrue(record["rewrite"]["prompt_truncated"])

    def test_a_clear_question_is_logged_as_skipped(self):
        trace = {
            "unclear": False,
            "search_question": "Koliko ESPB ima Uvod u programiranje?",
            "rewrite": {
                "question_in": "Koliko ESPB ima Uvod u programiranje?",
                "question_out": "Koliko ESPB ima Uvod u programiranje?",
                "normalized": "Koliko ESPB ima Uvod u programiranje?",
                "changed": False,
                "result": "skipped_clear_question",
                "matched_pattern": None,
                "prompt": None,
                "raw_response": None,
                "rejected": None,
                "history_text": "",
            },
        }

        record = make_record(trace=trace)

        self.assertEqual(record["rewrite"]["result"], "skipped_clear_question")
        self.assertIsNone(record["rewrite"]["prompt"])
        self.assertFalse(record["unclear"])
        self.assertEqual(record["retrieval"]["attempts"], [])
        self.assertEqual(record["retrieval"]["winning_strategy"], "none")

    def test_failed_turn_is_marked_and_explains_itself(self):
        record = make_record(error="Ollama is not reachable")

        self.assertTrue(record["failed"])
        self.assertEqual(record["error"], "Ollama is not reachable")

    def test_successful_turn_is_not_marked_failed(self):
        self.assertFalse(make_record()["failed"])
        self.assertNotIn("error", make_record())

    def test_schema_version_is_pinned(self):
        self.assertEqual(make_record()["schema_version"], SCHEMA_VERSION)

    def test_empty_trace_still_produces_a_valid_record(self):
        # e.g. the model was never reached, or the client disconnected first
        record = make_record(trace={}, answer="")
        line = json.dumps(record, ensure_ascii=False)

        self.assertEqual(json.loads(line)["question"], "Koji su preduslovi za njega?")
        self.assertEqual(record["chunks"], [])
        self.assertFalse(record["failed"])


class HelpersTests(unittest.TestCase):
    def test_strip_sources_block(self):
        self.assertEqual(strip_sources_block("Odgovor.\n\n---\nIzvori:\n[1] a.pdf"), "Odgovor.")
        self.assertEqual(strip_sources_block("Odgovor bez izvora."), "Odgovor bez izvora.")

    def test_truncate_reports_whether_it_cut(self):
        self.assertEqual(truncate("abc", 5), ("abc", False))
        self.assertEqual(truncate("abcdef", 3), ("abc", True))
        self.assertEqual(truncate(None, 5), ("", False))
        self.assertEqual(truncate("abc", 0), ("abc", False))  # 0 = no cap


class RequestLoggerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "nested" / "requests.jsonl"

    def tearDown(self):
        self._tmp.cleanup()
        # The handler holds the file open; drop the loggers these tests made.
        for name in list(logging.Logger.manager.loggerDict):
            if name.startswith("university_chatbot.request_log."):
                logger = logging.getLogger(name)
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)

    def read_lines(self):
        return self.path.read_text(encoding="utf-8").splitlines()

    def test_writes_one_valid_json_line_per_turn(self):
        log = RequestLogger(self.path)

        log.write(make_record(request_id="one"))
        log.write(make_record(request_id="two"))

        lines = self.read_lines()
        self.assertEqual(len(lines), 2)
        self.assertEqual([json.loads(line)["request_id"] for line in lines], ["one", "two"])
        # One line = one turn, whatever the payload size.
        self.assertEqual(len(lines[0].splitlines()), 1)

    def test_serbian_text_is_written_readable_not_escaped(self):
        log = RequestLogger(self.path)

        log.write(make_record())

        self.assertIn("Koji su preduslovi", self.path.read_text(encoding="utf-8"))

    def test_creates_the_directory_and_defers_the_file(self):
        log = RequestLogger(self.path)
        self.assertFalse(self.path.exists())  # nothing written yet

        log.write(make_record())

        self.assertTrue(self.path.exists())

    def test_disabled_logger_writes_nothing(self):
        log = RequestLogger(self.path, enabled=False)

        log.write(make_record())

        self.assertFalse(self.path.exists())

    def test_records_survive_a_reopen(self):
        RequestLogger(self.path).write(make_record(request_id="first"))
        RequestLogger(self.path).write(make_record(request_id="second"))

        self.assertEqual(len(self.read_lines()), 2)

    def test_rotation_keeps_the_file_bounded(self):
        # A record is ~2 KB, so 20 turns overflow a 6 KB file repeatedly.
        log = RequestLogger(self.path, max_bytes=6000, backups=2)
        records = [make_record(request_id=f"turn-{i}") for i in range(20)]

        for record in records:
            log.write(record)

        files = sorted(p.name for p in self.path.parent.iterdir())
        self.assertEqual(files, ["requests.jsonl", "requests.jsonl.1", "requests.jsonl.2"])
        # Rotation is size-based, so the live file holds the last few records,
        # not all 20 — the point is that the log cannot grow without bound.
        self.assertLess(self.path.stat().st_size, sum(len(json.dumps(r)) for r in records))

        recent = self.path.read_text(encoding="utf-8")
        self.assertIn("turn-19", recent)
        self.assertNotIn("turn-0", recent)
        # The backup that rotated out most recently carries the turn before
        # the ones still in the live file.
        self.assertIn("turn-16", (self.path.parent / "requests.jsonl.1").read_text("utf-8"))

    def test_a_write_failure_does_not_raise(self):
        log = RequestLogger(self.path)
        log.write(make_record())
        # Simulate a full disk / removed directory.
        log._logger.handlers[0].stream = None

        log.write(make_record())  # must not raise


if __name__ == "__main__":
    unittest.main()
