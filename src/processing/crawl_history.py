"""
crawl_history.py
~~~~~~~~~~~~~~~~
Persistent crawl history backed by a local JSON file.

Every URL that has been successfully crawled + ingested is recorded
with a timestamp, content type, and chunk count.  On subsequent runs
the history is loaded and already-ingested URLs are skipped, so you
don't waste time (or server politeness) re-fetching the same pages.

Usage
-----
    history = CrawlHistoryManager("./data/crawl_history.json")
    history.load()

    # Before crawling:
    if history.is_ingested(url):
        print("Already done — skip")

    # After successful ingestion:
    history.mark_ingested(url, url_type, num_chunks)

    history.save()
"""

from __future__ import annotations

import json
import os
import time
from typing import Any


class CrawlHistoryManager:
    """
    Reads/writes a simple JSON crawl history.

    File format:
      {
        "version": 1,
        "base_url": "https://imi.pmf.kg.ac.rs/",
        "urls": {
          "https://imi.pmf.kg.ac.rs/matematika-studije": {
            "type": "html",
            "chunks": 12,
            "ingested_at": 1712345678.123
          },
          ...
        }
      }
    """

    def __init__(self, file_path: str, base_url: str | None = None):
        self.file_path = file_path
        self.base_url = base_url
        self._data: dict[str, Any] = {"version": 1, "base_url": base_url, "urls": {}}

    # ------------------------------------------------------------------ #
    #  I/O                                                                  #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Load history from disk.  No-op if the file doesn't exist."""
        if not os.path.exists(self.file_path):
            return
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        except (json.JSONDecodeError, KeyError):
            # Corrupted file — start fresh
            self._data = {"version": 1, "base_url": self.base_url, "urls": {}}

    def save(self) -> None:
        """Persist the current history to disk."""
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ #
    #  Query                                                                #
    # ------------------------------------------------------------------ #

    def is_ingested(self, url: str) -> bool:
        """Has this URL already been crawled and ingested?"""
        return url in self._data.get("urls", {})

    def ingested_count(self) -> int:
        """Total number of URLs in the history."""
        return len(self._data.get("urls", {}))

    # ------------------------------------------------------------------ #
    #  Mutate                                                               #
    # ------------------------------------------------------------------ #

    def mark_ingested(self, url: str, url_type: str, num_chunks: int = 0) -> None:
        """Record a URL as successfully ingested."""
        self._data.setdefault("urls", {})[url] = {
            "type": url_type,
            "chunks": num_chunks,
            "ingested_at": time.time(),
        }

    def clear(self) -> None:
        """Remove all history entries (forces a full re-crawl next run)."""
        self._data["urls"] = {}

    def set_base_url(self, base_url: str) -> None:
        """Update the tracked base URL."""
        self._data["base_url"] = base_url
