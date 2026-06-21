"""
document_loader.py
~~~~~~~~~~~~~~~~~~
Loads .docx files, .pdf files, and web pages — strips headers/footers —
and returns LangChain Document chunks with a unified metadata schema:
 
    {
        "source":   str   # full file path or URL
        "filename": str   # display name (basename or page title)
        "type":     str   # "pdf" | "docx" | "web"
        "page":     int   # 1-based page number for PDFs, 0 otherwise
        "url":      str   # populated for web sources, empty string for files
    }
 
This makes it trivial to render consistent citations in the QA layer
regardless of where the content came from.
"""

import os
import re
import time
from collections import Counter

import pdfplumber
import requests
from bs4 import BeautifulSoup
from docx import Document
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter



# --------------------------------------------------------------------------- #
#  Shared text splitter factory                                                #
# --------------------------------------------------------------------------- #
 
def _make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
 
 

class DocumentProcessor:
    # ------------------------------------------------------------------ #
    #  Tuneable constants                                                  #
    # ------------------------------------------------------------------ #

    # Fraction of page height reserved for header / footer zones
    DEFAULT_HEADER_FRACTION: float = 0.08   # top 8 %
    DEFAULT_FOOTER_FRACTION: float = 0.08   # bottom 8 %

    # A line that appears on this share of pages or more is treated as a
    # running header / footer even if it falls inside the body crop zone.
    DEFAULT_REPETITION_THRESHOLD: float = 0.40

    # ------------------------------------------------------------------ #

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        header_fraction: float = DEFAULT_HEADER_FRACTION,
        footer_fraction: float = DEFAULT_FOOTER_FRACTION,
        repetition_threshold: float = DEFAULT_REPETITION_THRESHOLD,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.header_fraction = header_fraction
        self.footer_fraction = footer_fraction
        self.repetition_threshold = repetition_threshold

        self.text_splitter = _make_splitter(chunk_size, chunk_overlap)

    # ================================================================== #
    #  DOCX                                                               #
    # ================================================================== #

    def _collect_header_footer_texts(self, doc: Document) -> set[str]:
        """
        Walk every section's header and footer collections and return
        the set of non-empty stripped strings found there.
        We also include even-page and first-page variants, which Word
        uses for different header/footer content on alternating pages.
        """
        hf_texts: set[str] = set()
        for section in doc.sections:
            for hf_object in (
                section.header,
                section.footer,
                section.even_page_header,
                section.even_page_footer,
                section.first_page_header,
                section.first_page_footer,
            ):
                # python-docx returns None when the slot is not linked/used
                if hf_object is None:
                    continue
                for para in hf_object.paragraphs:
                    text = para.text.strip()
                    if text:
                        hf_texts.add(text)
        return hf_texts

    def _extract_docx_body(self, file_path: str) -> str:
        """
        Return clean body text from a .docx file with headers and
        footers removed.
        """
        doc = Document(file_path)
        hf_texts = self._collect_header_footer_texts(doc)

        body_lines: list[str] = []
        for para in doc.paragraphs:
            text = para.text.strip()
            # Skip blank paragraphs and anything that matches a known
            # header / footer string (catches duplicates that some
            # converters inject into the body stream).
            if text and text not in hf_texts:
                body_lines.append(text)

        return "\n\n".join(body_lines)

    def load_docx(self, file_path: str) -> list[LCDocument]:
        text = self._extract_docx_body(file_path)
        return self.text_splitter.create_documents(
            [text],
            metadatas=[{"source": file_path, "type": "docx"}],
        )

    # ================================================================== #
    #  PDF                                                                 #
    # ================================================================== #

    def _detect_repeated_lines(self, pdf) -> set[str]:
        """
        Collect lines that appear on a large fraction of pages.
        These are almost always running headers, footers, or page numbers
        that the margin-crop alone may not fully remove (e.g. when a PDF
        has unusually small margins).

        A line must appear on at least 2 pages AND on at least
        `repetition_threshold` of all pages to be flagged.
        """
        page_line_sets: list[set[str]] = []
        for page in pdf.pages:
            raw = page.extract_text() or ""
            lines = {
                line.strip()
                for line in raw.splitlines()
                if line.strip()
            }
            page_line_sets.append(lines)

        if not page_line_sets:
            return set()

        counter: Counter = Counter()
        for line_set in page_line_sets:
            for line in line_set:
                counter[line] += 1

        n_pages = len(page_line_sets)
        min_count = max(2, int(n_pages * self.repetition_threshold))

        return {line for line, count in counter.items() if count >= min_count}

    def _extract_page_text(
        self, page, repeated_lines: set[str]
    ) -> str:
        """
        Crop the header and footer zones from a single pdfplumber page,
        then drop any remaining repeated lines.

        pdfplumber crop() takes a bounding box: (x0, top, x1, bottom)
        where (0, 0) is the top-left corner.
        """
        h = float(page.height)
        w = float(page.width)
        top_cutoff    = h * self.header_fraction   # pixels from top to skip
        bottom_cutoff = h * self.footer_fraction   # pixels from bottom to skip

        body_page = page.crop(
            (0, top_cutoff, w, h - bottom_cutoff),
            relative=False,
        )
        raw_text = body_page.extract_text() or ""

        clean_lines: list[str] = []
        for line in raw_text.splitlines():
            stripped = line.strip()
            # Drop blank lines and known repeated header/footer lines
            if stripped and stripped not in repeated_lines:
                # Also drop pure page-number patterns: "1", "- 1 -", "Page 1 of 5"
                if not re.fullmatch(
                    r"[-–—]?\s*\d+\s*[-–—]?|[Pp]age\s+\d+(\s+of\s+\d+)?",
                    stripped,
                ):
                    clean_lines.append(stripped)

        return "\n".join(clean_lines)

    def _extract_pdf_body(self, file_path: str) -> str:
        """
        Return clean body text from a PDF with headers, footers, and
        page numbers removed.
        """
        with pdfplumber.open(file_path) as pdf:
            repeated = self._detect_repeated_lines(pdf)
            pages_text: list[str] = []
            for page in pdf.pages:
                text = self._extract_page_text(page, repeated)
                if text.strip():
                    pages_text.append(text)

        return "\n\n".join(pages_text)

    def load_pdf(self, file_path: str) -> list[LCDocument]:
        text = self._extract_pdf_body(file_path)
        return self.text_splitter.create_documents(
            [text],
            metadatas=[{"source": file_path, "type": "pdf"}],
        )

    # ================================================================== #
    #  Public API                                                          #
    # ================================================================== #

    def process_file(self, file_path: str) -> list[LCDocument]:
        """Load, clean, and chunk a single .docx or .pdf file."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".docx":
            docs = self.load_docx(file_path)
        elif ext == ".pdf":
            docs = self.load_pdf(file_path)
        else:
            print(f"Warning: unsupported file type '{ext}' — skipping {file_path}")
            return []

        print(
            f"Loaded {os.path.basename(file_path)} "
            f"({ext[1:].upper()}) → {len(docs)} chunks"
        )
        return docs

    def process_directory(self, directory: str) -> list[LCDocument]:
        """
        Recursively find and process every .docx and .pdf file under
        `directory`.  Files are sorted so indexing order is deterministic.
        """
        all_docs: list[LCDocument] = []
        for root, _, files in os.walk(directory):
            for fname in sorted(files):
                if os.path.splitext(fname)[1].lower() in (".docx", ".pdf"):
                    full_path = os.path.join(root, fname)
                    all_docs.extend(self.process_file(full_path))
        return all_docs
    
# --------------------------------------------------------------------------- #
#  WebLoader                                                                   #
# --------------------------------------------------------------------------- #
 
class WebLoader:
    """
    Fetch one or more web pages and turn them into clean, chunked
    LangChain Documents with the page URL stored in metadata.
 
    Content extraction strategy
    ---------------------------
    1. Try trafilatura (best-in-class article extractor -- strips nav,
       ads, sidebars).  Install with:  pip install trafilatura
    2. Fall back to BeautifulSoup if trafilatura is not installed or
       fails to extract any content.
    """
 
    DEFAULT_TIMEOUT:    int   = 15
    DEFAULT_RATE_LIMIT: float = 1.0   # seconds between consecutive fetches
 
    def __init__(
        self,
        chunk_size:    int   = 1000,
        chunk_overlap: int   = 200,
        timeout:       int   = DEFAULT_TIMEOUT,
        rate_limit:    float = DEFAULT_RATE_LIMIT,
    ):
        self.timeout       = timeout
        self.rate_limit    = rate_limit
        self.text_splitter = _make_splitter(chunk_size, chunk_overlap)
        self._session      = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (compatible; UniversityRAGBot/1.0)"
            )
        })
 
    # ------------------------------------------------------------------ #
    #  Extraction helpers                                                  #
    # ------------------------------------------------------------------ #
 
    @staticmethod
    def _extract_with_trafilatura(html: str, url: str) -> tuple[str, str]:
        """Returns (title, body_text). Raises ImportError if not installed."""
        import trafilatura
 
        content = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )
        meta  = trafilatura.extract_metadata(html)
        title = (meta.title if meta and meta.title else "") or url
        return title, content or ""
 
    @staticmethod
    def _extract_with_bs4(html: str, url: str) -> tuple[str, str]:
        """BeautifulSoup fallback extractor."""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "form", "noscript", "iframe"]):
            tag.decompose()
 
        title    = soup.title.string.strip() if soup.title and soup.title.string else url
        body_tag = soup.find("main") or soup.find("article") or soup.body
        if body_tag is None:
            return title, ""
 
        lines = [
            line.strip()
            for line in body_tag.get_text(separator="\n").splitlines()
            if line.strip()
        ]
        return title, "\n".join(lines)
 
    def _fetch_and_extract(self, url: str) -> tuple[str, str]:
        """Fetch a URL and return (title, clean_text)."""
        response = self._session.get(url, timeout=self.timeout)
        response.raise_for_status()
        html = response.text
 
        try:
            title, text = self._extract_with_trafilatura(html, url)
            if text.strip():
                return title, text
        except ImportError:
            pass
 
        return self._extract_with_bs4(html, url)
 
    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #
 
    def load_url(self, url: str) -> list[LCDocument]:
        """Fetch a single URL, clean the content, and return chunks."""
        try:
            title, text = self._fetch_and_extract(url)
        except Exception as exc:
            print(f"  Failed to fetch {url}: {exc}")
            return []
 
        if not text.strip():
            print(f"  No content extracted from {url}")
            return []
 
        meta = {
            "source":   url,
            "filename": title,   # page <title> or the URL itself
            "type":     "web",
            "page":     0,
            "url":      url,
        }
        chunks = self.text_splitter.create_documents([text], metadatas=[meta])
        print(f"  {url} -> {len(chunks)} chunks  (title: {title!r})")
        return chunks
 
    def load_urls(self, urls: list[str]) -> list[LCDocument]:
        """Fetch multiple URLs with a polite delay between requests."""
        all_docs: list[LCDocument] = []
        for i, url in enumerate(urls):
            docs = self.load_url(url)
            all_docs.extend(docs)
            if i < len(urls) - 1:
                time.sleep(self.rate_limit)
        return all_docs
 