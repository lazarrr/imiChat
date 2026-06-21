"""
qa_chain.py
~~~~~~~~~~~
University QA system — retrieves relevant chunks, formats citations,
and generates a grounded answer via the local LLM.

Source references
-----------------
Every chunk in ChromaDB now carries a unified metadata schema:

    source   — full path or URL
    filename — human-readable name (basename or page title)
    type     — "pdf" | "docx" | "web"
    page     — 1-based page number for PDFs, 0 otherwise
    url      — canonical URL for web sources, "" for local files

_format_sources() turns that into human-readable citation lines:

    [1] lecture_notes.pdf — strana 4
    [2] Uvod u baze podataka.docx
    [3] Osnovi programiranja — https://example.com/article
"""

from collections.abc import AsyncGenerator
from langchain_core.prompts import PromptTemplate


class UniversityQASystem:
    def __init__(self, llm_manager, vector_store, translator):
        self.llm_manager  = llm_manager
        self.vector_store = vector_store
        self.translator   = translator

        self.prompt_template = PromptTemplate(
            template="""Vi ste korisni univerzitetski asistent. Odgovarajte na pitanja na osnovu priloženog konteksta.

Kontekst:
{context}

Pitanje: {question}

Uputstva:
1. Odgovarajte iskljucivo na srpskom latinskom pismu
2. Pruzite tacne informacije iz konteksta
3. Ako informacija nije u kontekstu, jasno to navedite
4. Na kraju odgovora navedite izvore u obliku [1], [2] itd.

Odgovor:""",
            input_variables=["context", "question"],
        )

    # ------------------------------------------------------------------ #
    #  Source formatting                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_source(meta: dict, index: int) -> str:
        """
        Turn a single chunk's metadata dict into a numbered citation line.

        Examples
        --------
        [1] lecture_notes.pdf — strana 4
        [2] Uvod u baze podataka.docx
        [3] Osnovi programiranja — https://example.com/article
        """
        filename = meta.get("filename") or meta.get("source", "Nepoznat izvor")
        doc_type = meta.get("type", "")
        page     = meta.get("page", 0)
        url      = meta.get("url", "")

        parts = [f"[{index}] {filename}"]

        if doc_type == "pdf" and page:
            parts.append(f"strana {page}")

        if url:
            parts.append(url)

        return " — ".join(parts)

    def _format_sources(self, metadatas: list[dict]) -> list[str]:
        """
        Deduplicate and format all sources for a query result.
        Deduplication key: (source_path_or_url, page) so we don't list
        the same file page twice when two chunks came from the same page.
        """
        seen:    set[tuple] = set()
        sources: list[str]  = []

        for meta in metadatas:
            key = (meta.get("source", ""), meta.get("page", 0))
            if key in seen:
                continue
            seen.add(key)
            sources.append(self._format_source(meta, len(sources) + 1))

        return sources

    # ------------------------------------------------------------------ #
    #  Context building                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_context(documents: list[str], metadatas: list[dict]) -> str:
        """
        Interleave each chunk with a small inline source tag so the LLM
        can reference specific sources in its answer.

        Example:
            [Izvor 1 — lecture_notes.pdf, strana 4]
            <chunk text here>
        """
        parts: list[str] = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
            filename = meta.get("filename") or meta.get("source", "")
            page     = meta.get("page", 0)
            url      = meta.get("url", "")

            if page:
                label = f"{filename}, strana {page}"
            elif url:
                label = f"{filename} ({url})"
            else:
                label = filename

            parts.append(f"[Izvor {i} — {label}]\n{doc}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    #  Query                                                               #
    # ------------------------------------------------------------------ #

    def query(self, question: str, k: int = 5) -> dict:
        """
        Retrieve relevant chunks, build a grounded prompt, call the LLM,
        and return the answer alongside formatted source references.

        Returns
        -------
        {
            "question": str,
            "answer":   str,
            "sources":  list[str],   # human-readable citation lines
        }
        """
        normalized = self.translator.normalize_query(question)
        results    = self.vector_store.similarity_search(normalized, k=k)
        documents  = results["documents"][0]
        metadatas  = results["metadatas"][0]

        if not documents:
            return {
                "question": question,
                "answer":   "Nisam pronasao relevantne informacije u bazi znanja za ovo pitanje.",
                "sources":  [],
            }

        context  = self._build_context(documents, metadatas)
        prompt   = self.prompt_template.format(context=context, question=question)
        response = self.llm_manager.generate(prompt)
        sources  = self._format_sources(metadatas)

        return {
            "question": question,
            "answer":   response,
            "sources":  sources,
        }

    async def query_stream(self, question: str) -> AsyncGenerator[str, None]:
        """Stream response tokens — sources are yielded as a trailing block."""
        normalized = self.translator.normalize_query(question)
        results    = self.vector_store.similarity_search(normalized)
        documents  = results["documents"][0]
        metadatas  = results["metadatas"][0]

        context = self._build_context(documents, metadatas)
        prompt  = self.prompt_template.format(context=context, question=question)

        async for token in self.llm_manager.generate_stream(prompt):
            yield token

        sources = self._format_sources(metadatas)
        if sources:
            yield "\n\n---\nIzvori:\n" + "\n".join(sources)