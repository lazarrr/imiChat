"""
crawler.py
~~~~~~~~~~
Domain-scoped web crawler that discovers all reachable URLs from a base URL.

Features
--------
• Stays within the same domain (no external links)
• Respects robots.txt via urllib.robotparser
• Classifies each discovered URL: html, pdf, docx, or skip
• Maintains a visited set to avoid cycles
• Configurable depth limit and politeness delay between requests

Output: list[CrawledURL] — each with .url, .type, .depth
"""

from __future__ import annotations

import re
import time
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from collections import deque
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup


@dataclass
class CrawledURL:
    url:   str
    type:  str   # "html" | "pdf" | "docx" | "skip"
    depth: int


class DomainCrawler:
    """Recursive, breadth-first crawler for a single domain."""

    EXTRACTABLE_EXTENSIONS: set[str] = {".pdf", ".docx", ".doc"}
    SKIP_EXTENSIONS: set[str] = {
        ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
        ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
        ".mp4", ".mp3", ".avi", ".mov", ".wmv",
        ".css", ".js", ".json", ".xml", ".ico",
        ".xls", ".xlsx", ".ppt", ".pptx",
        ".exe", ".msi", ".dmg", ".pkg",
    }

    def __init__(
        self,
        base_url: str,
        max_depth: int = 3,
        delay: float = 1.0,
        timeout: int = 15,
        user_agent: str = "Mozilla/5.0 (compatible; UniversityRAGBot/1.0)",
    ):
        self.base_url  = base_url.rstrip("/")
        self.max_depth = max_depth
        self.delay     = delay
        self.timeout   = timeout

        parsed = urlparse(self.base_url)
        self.domain = parsed.netloc.lower()
        self.scheme = parsed.scheme

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})

        self._visited: set[str] = set()
        self._robots: RobotFileParser | None = None

    # ------------------------------------------------------------------ #
    #  Robots.txt                                                          #
    # ------------------------------------------------------------------ #

    def _load_robots(self) -> None:
        """Fetch and parse robots.txt for the domain (cached)."""
        if self._robots is not None:
            return
        robots_url = f"{self.scheme}://{self.domain}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            rp.read()
        except Exception:
            rp.allow_all = True
        self._robots = rp

    def _is_allowed(self, url: str) -> bool:
        self._load_robots()
        return self._robots.can_fetch(
            self._session.headers.get("User-Agent", "*"), url
        )

    # ------------------------------------------------------------------ #
    #  URL classification                                                   #
    # ------------------------------------------------------------------ #

    @classmethod
    def _classify_url(cls, url: str) -> str:
        """Returns one of: "html", "pdf", "docx", "skip"."""
        path = urlparse(url).path.lower()

        for ext in cls.EXTRACTABLE_EXTENSIONS:
            if path.endswith(ext):
                return "pdf" if ext == ".pdf" else "docx"

        for ext in cls.SKIP_EXTENSIONS:
            if path.endswith(ext):
                return "skip"

        # No extension or .html/.php/.asp → treat as HTML
        if not re.search(r"\.\w{2,5}$", path):
            return "html"
        if re.search(r"\.(html?|php|asp|jsp|cfm)$", path):
            return "html"

        return "html"

    # ------------------------------------------------------------------ #
    #  Link extraction                                                      #
    # ------------------------------------------------------------------ #

    def _extract_links(self, html: str, current_url: str) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")
        links: list[str] = []
        for tag in soup.find_all("a", href=True):
            href = tag["href"].strip()
            absolute = urljoin(current_url, href)
            clean = urlparse(absolute)._replace(fragment="").geturl()
            links.append(clean)
        return links

    def _is_same_domain(self, url: str) -> bool:
        return urlparse(url).netloc.lower() == self.domain

    # ------------------------------------------------------------------ #
    #  Crawl                                                               #
    # ------------------------------------------------------------------ #

    def crawl(self) -> list[CrawledURL]:
        """Run the BFS crawl and return all discovered URLs."""
        results: list[CrawledURL] = []
        queue: deque[tuple[str, int]] = deque()
        queue.append((self.base_url, 0))

        while queue:
            current_url, depth = queue.popleft()

            if current_url in self._visited:
                continue
            if not self._is_same_domain(current_url):
                continue
            if not self._is_allowed(current_url):
                print(f"  [robots.txt] Skipping: {current_url}")
                continue

            self._visited.add(current_url)

            url_type = self._classify_url(current_url)
            crawled = CrawledURL(url=current_url, type=url_type, depth=depth)
            results.append(crawled)

            if url_type == "skip":
                print(f"  [skip] {current_url}")
                continue

            print(f"  [depth {depth}] {url_type.upper():>4s}  {current_url}")

            if url_type != "html" or depth >= self.max_depth:
                continue

            # Fetch HTML to discover more links
            try:
                response = self._session.get(current_url, timeout=self.timeout)
                response.raise_for_status()
            except Exception as exc:
                print(f"  [error] {current_url}: {exc}")
                continue

            links = self._extract_links(response.text, current_url)
            for link in links:
                if link not in self._visited:
                    queue.append((link, depth + 1))

            if self.delay > 0:
                time.sleep(self.delay)

        print(f"\nCrawl complete: {len(results)} URLs "
              f"({len([r for r in results if r.type != 'skip'])} extractable)")
        return results

    @staticmethod
    def filter_by_type(results: list[CrawledURL], url_type: str) -> list[str]:
        """Return just the URLs of a given type."""
        return [r.url for r in results if r.type == url_type]