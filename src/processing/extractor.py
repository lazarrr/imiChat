"""
extractor.py
~~~~~~~~~~~~
Extracts clean text from crawled resources (HTML, PDF, DOCX).

HTML   → BeautifulSoup (strips nav/header/footer/scripts)
PDF    → pdfplumber (inline stream or download fallback)
DOCX   → python-docx (always downloads to temp dir, then parses)

Output: (text: str, metadata: dict) for each resource.
"""

from __future__ import annotations

import os
import tempfile
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


class ResourceExtractor:
    """Extract clean text from web resources by type."""

    def __init__(
        self,
        timeout: int = 15,
        user_agent: str = "Mozilla/5.0 (compatible; UniversityRAGBot/1.0)",
    ):
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})

    # ------------------------------------------------------------------ #
    #  Fetch helper                                                         #
    # ------------------------------------------------------------------ #

    def _fetch(self, url: str, stream: bool = False):
        """GET a URL, returning a requests.Response or raising on failure."""
        response = self._session.get(url, timeout=self.timeout, stream=stream)
        response.raise_for_status()
        return response

    # ------------------------------------------------------------------ #
    #  HTML extraction                                                      #
    # ------------------------------------------------------------------ #

    def _extract_html(self, url: str) -> tuple[str, dict]:
        """Extract clean text from an HTML page."""
        response = self._fetch(url)
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "form", "noscript", "iframe"]):
            tag.decompose()

        title = (
            soup.title.string.strip()
            if soup.title and soup.title.string
            else url
        )

        # Prefer <main> or <article>, fall back to <body>
        body = soup.find("main") or soup.find("article") or soup.body
        text = ""
        if body:
            lines = [
                line.strip()
                for line in body.get_text(separator="\n").splitlines()
                if line.strip()
            ]
            text = "\n".join(lines)

        meta = {
            "source":   url,
            "filename": title,
            "type":     "web",
            "page":     0,
            "url":      url,
        }
        return text, meta

    # ------------------------------------------------------------------ #
    #  PDF extraction                                                       #
    # ------------------------------------------------------------------ #

    def _extract_pdf(self, url: str) -> tuple[str, dict]:
        """
        Extract text from a PDF at the given URL.
        Tries inline streaming first; falls back to downloading to a
        temp file if pdfplumber can't parse the stream directly.
        """
        filename = urlparse(url).path.rsplit("/", 1)[-1] or url

        # Attempt 1: stream into pdfplumber
        try:
            response = self._fetch(url, stream=True)
            raw = response.raw

            import pdfplumber
            with pdfplumber.open(raw) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            text = "\n\n".join(p for p in pages if p.strip())

            if text.strip():
                meta = {
                    "source":   url,
                    "filename": filename,
                    "type":     "pdf",
                    "page":     0,
                    "url":      url,
                }
                return text, meta
        except Exception:
            pass  # Fall through to download approach

        # Attempt 2: download to temp file
        try:
            response = self._fetch(url)
            suffix = ".pdf"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

            import pdfplumber
            with pdfplumber.open(tmp_path) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            text = "\n\n".join(p for p in pages if p.strip())

            meta = {
                "source":   url,
                "filename": filename,
                "type":     "pdf",
                "page":     0,
                "url":      url,
            }
            return text, meta
        finally:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ------------------------------------------------------------------ #
    #  DOCX extraction                                                      #
    # ------------------------------------------------------------------ #

    def _extract_docx(self, url: str) -> tuple[str, dict]:
        """
        Extract text from a DOCX file. Always downloads to temp — there
        is no way to parse .docx from a stream.
        """
        filename = urlparse(url).path.rsplit("/", 1)[-1] or url
        tmp_path = None

        try:
            response = self._fetch(url)
            suffix = os.path.splitext(filename)[1] or ".docx"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

            from docx import Document
            doc = Document(tmp_path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            text = "\n\n".join(paragraphs)

            meta = {
                "source":   url,
                "filename": filename,
                "type":     "docx",
                "page":     0,
                "url":      url,
            }
            return text, meta
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ------------------------------------------------------------------ #
    #  Public API                                                           #
    # ------------------------------------------------------------------ #

    def extract(self, url: str, url_type: str) -> tuple[str, dict] | None:
        """
        Extract clean text from a URL.

        Parameters
        ----------
        url : str
        url_type : str
            One of "html", "pdf", "docx".

        Returns
        -------
        (text, metadata) or None if extraction fails.
        """
        try:
            if url_type == "html":
                return self._extract_html(url)
            elif url_type == "pdf":
                return self._extract_pdf(url)
            elif url_type == "docx":
                return self._extract_docx(url)
            else:
                print(f"  Unknown type '{url_type}' for {url}")
                return None
        except Exception as exc:
            print(f"  [error] Failed to extract {url}: {exc}")
            return None