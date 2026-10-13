# University RAG Chatbot — Instructions

## Overview

A Retrieval-Augmented Generation (RAG) chatbot for the IMI PMF Kragujevac
university, split into two projects that share a single ChromaDB vector store:

| Project | What it does |
|---|---|
| `scraper` (C:\Projects\scraper) | Crawls the links listed in `links.txt`, parses HTML (incl. every tab panel) / PDF / DOCX, labels each chunk with its location and study-program version, validates + enriches it with DeepSeek, embeds, and writes to ChromaDB. |
| `university-chatbot` | Reads the shared ChromaDB store, retrieves relevant chunks, and answers questions via Llama 3.1 8B (Ollama). |

Answers are in **Serbian (Latin script)** with inline `[Izvor N]` citations.

---

## Prerequisites

| Component | Required |
|---|---|
| **Conda** (Miniconda or Anaconda) | Environment management |
| **Ollama** | Local LLM inference (`brew install ollama` on macOS) |
| **Python** ≥ 3.10 | Language runtime |
| **Git** | Clone / version control (optional) |

---

## 1. Environment setup (Conda)

```bash
# Create the conda environment
conda create -n uni python=3.11 -y (this already exist on development machine)

# Activate it
conda activate uni

# Install both projects' dependencies
cd /path/to/master/university-scraper
pip install -r requirements.txt

cd /path/to/master/university-chatbot
pip install -r requirements.txt
```

> **All subsequent commands assume `conda activate uni` is already run.**

---

## 2. Ollama setup

```bash
# Start the Ollama server (keep this running in a separate terminal)
ollama serve

# Pull the LLM (in another terminal, or before starting the server)
ollama pull llama3.1:latest
```

Ollama runs at `http://localhost:11434` by default.  The chatbot
connects to it automatically — no extra configuration needed.

The chatbot also downloads `intfloat/multilingual-e5-large` for
embeddings on first run (≈ 2.2 GB, one-time download).

---

## 3. Configuration

Both projects use `pydantic-settings` — every value can be overridden
via environment variables or a `.env` file.

### Shared settings (must match across both projects)

| Variable | Default | Notes |
|---|---|---|
| `VECTOR_DB_PATH` | `./data/vector_store` | Absolute path recommended if projects live in different folders |
| `CHROMA_COLLECTION` | `university_knowledge` | Point both at e.g. `university_knowledge_test` to try a test ingest |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | Must be identical |
| `chromadb` version | `1.5.9` | Pinned in both `requirements.txt` files |

### Chatbot-specific (`.env` in `university-chatbot/`)

| Variable | Default | Notes |
|---|---|---|
| `LLM_MODEL` | `llama3.1:latest` | Must match `ollama list` |
| `LLM_TEMPERATURE` | `0.0` | 0 = deterministic |
| `LLM_MAX_TOKENS` | `2048` | Max response length |
| `LLM_NUM_CTX` | `8192` | Context window (tokens) |
| `SIMILARITY_THRESHOLD` | `0.7` | Cosine similarity cutoff |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |
| `RETRIEVAL_FETCH_K` | `20` | Candidates fetched before the version policy drops legacy / duplicate chunks |
| `API_HOST` | `0.0.0.0` | |
| `API_PORT` | `8000` | |
| `RATE_LIMIT` | `100` | `/chat` requests per minute per client IP (`0` = no limit); HTTP 429 above it |
| `REDIS_URL` | *(unset)* | Optional, e.g. `redis://localhost:6379/0` — sessions and rate limits shared by all workers and kept across restarts (otherwise in memory) |

### Scraper-specific (`.env` in `C:\Projects\scraper`)

| Variable | Default | Notes |
|---|---|---|
| `DEEPSEEK_API_KEY` | – | Required unless `--no-llm` |
| `DEEPSEEK_MODEL` | `deepseek-flash` | DeepSeek-V4.1-Flash |
| `CRAWL_DELAY` | `1.0` | Seconds between requests |
| `CRAWL_DOC_PATTERNS` | `["/pub/"]` | Which linked PDFs `docs=true` follows |
| `CHUNK_MAX_TOKENS` | `300` | E5 tokens per chunk (E5 reads at most 512) |

---

## 4. Running the scraper (populate the vector store)

Run this **first** — it populates the shared ChromaDB store that the
chatbot reads from.  **Stop the chatbot while ingesting and restart it
afterwards** — Chroma's local mode is not multi-process safe and every
process keeps the index in memory.

```bash
cd C:\Projects\scraper
python run.py ingest --links links.test.txt --dry-run      # try it: report only, nothing written
python run.py ingest --links links.test.txt --collection university_knowledge_test
python run.py search "Koliko modula ima informatika?" --collection university_knowledge_test
python run.py ingest                                         # everything in links.txt
```

What happens:
1. Fetches **only the links listed in `links.txt`** (`| depth=N` / `| docs=true` per line to follow links; robots.txt + 1 s delay).
2. Parses every tab panel of a page as its own section with a breadcrumb
   (e.g. `Informatika › Osnovne (Upis 2026/27) › Računarske nauke`);
   panels no visitor can open are skipped; course tables keep their columns.
3. Labels each section with its study-program version from `programs.yaml`
   (accreditation year + status `enrolling | phase_out | upcoming | legacy`).
   DeepSeek drafts entries for unknown panels into `programs.draft.yaml` for review.
4. DeepSeek (`deepseek-flash`) adds page type / summary / date and, per chunk,
   context, likely questions, keywords and a quality flag; boilerplate is dropped.
5. Adds a generated, verified "Pregled verzija" chunk per program
   (e.g. 2026: 2 modula … 2019: 3 modula …).
6. Embeds with `intfloat/multilingual-e5-large` (`passage: ` prefix) and writes
   to the collection: upsert, then delete that page's stale chunks.
7. Writes `data/reports/<time>/report.md` — read it after every run.

Unchanged pages are skipped on the next run (`--force` re-processes them).
`python run.py remove <url>` deletes a page's chunks.

> **The persist directory** (`data/vector_store/`) is shared between
> projects.  The scraper writes a `shared_config.json` there recording
> the embedding model and prefixes.

---

## 5. Running the chatbot

```bash
conda activate uni
cd university-chatbot
python run.py          # CLI interactive mode (follow-ups work; 'novo' resets)
# OR
python -m src.api.main  # FastAPI server (multi-turn with sessions)
```

### CLI mode (`run.py`)

Simple REPL loop.  Type questions, get answers + sources.  The conversation
history is kept so that follow-ups like "a o njima?" can be rewritten (see
§8); type `novo` to start a new conversation, `kraj` to exit.

```
Pitanje (ili 'kraj'): Koliko ESPB ima Uvod u programiranje?
Odgovor:
...
Izvori:
  [1] informatika_oas_2021.pdf — strana 12
```

### Server mode (`src.api.main`)

FastAPI server at `http://localhost:8000`.  Supports **multi-turn
conversations** via session management.

Start:
```bash
python -m src.api.main
```

---

## 6. API usage

### Health check

```bash
curl http://localhost:8000/health
# → {"status":"healthy","active_sessions":0}
```

### Single-turn question (no session)

```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"Koliko ESPB ima Uvod u programiranje?"}'
```

Response is **Server-Sent Events** (SSE) — tokens arrive one by one:
```
data: {"token":"Predmet"}
data: {"token":" Uvod"}
data: {"token":" u"}
...
data: {"token":"\n\n---\nIzvori:\n[1] informatika_oas_2021.pdf — strana 12"}
data: {"done":true}
```

### Multi-turn conversation (with session)

```bash
# Turn 1 — start a session
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"Koliko ESPB ima Uvod u programiranje?","session_id":"abc123"}'

# Turn 2 — follow-up with implicit reference
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"Koji su preduslovi za njega?","session_id":"abc123"}'
```

**What the chatbot does internally on turn 2:**
1. Detects an unclear follow-up: the question contains a reference word
   (`njega`) and session `abc123` has history.
2. Calls Llama 3.1 to rewrite `"Koji su preduslovi za njega?"` →
   `"Koji su preduslovi za Uvod u programiranje?"`.
3. Embeds the **rewritten** question (with `query: ` prefix) and
   retrieves relevant chunks from ChromaDB.
4. Sends **only** the system prompt + retrieved chunks + rewritten question
   to Ollama `/api/chat` — the conversation history is not sent.
5. Streams the answer token-by-token + appends `--- Izvori:` block.
6. Saves both the original question and answer to the session history.

Sessions expire after **1 hour** of inactivity.  They are kept in the
server's memory (lost on restart) unless `REDIS_URL` is set — then they live
in Redis and are shared by all API workers.  `/chat/stream` accepts at most
`RATE_LIMIT` requests per minute per client IP and answers HTTP 429 above
that.

---

## 7. Project structure

```
C:\Projects\
├── scraper/                     # Scraper & embedding project (see its README.md)
│   ├── links.txt                # ← what to crawl (one URL / local PDF per line)
│   ├── programs.yaml            # ← reviewed study-program versions (authoritative)
│   ├── run.py                   # CLI: ingest | search | eval | stats | remove
│   ├── config/settings.py       # Scraper config (.env overridable)
│   ├── src/core/                # DeepSeek client, E5 embeddings, Chroma store, retrieval policy
│   ├── src/processing/          # fetch, HTML/table/PDF parsing, registry, chunking, enrichment, ingest
│   └── data/                    # caches, state.json, reports/
│
├── university-chatbot/          # Retrieval & chat project
│   ├── config/settings.py       # Chatbot config
│   ├── run.py                   # CLI entry point
│   ├── src/
│   │   ├── api/main.py          # FastAPI server (routes, wiring)
│   │   ├── api/sessions.py      # Session history (memory or Redis)
│   │   ├── api/middleware.py    # Rate limiting (RATE_LIMIT per client IP)
│   │   ├── core/
│   │   │   ├── llm_manager.py   # Ollama wrappers (/api/chat & /api/generate)
│   │   │   ├── embeddings.py    # Embedding utilities
│   │   │   └── vector_store.py  # ChromaDB reader (E5 prefixes)
│   │   ├── chains/
│   │   │   ├── qa_chain.py      # Core RAG logic (retrieve → prompt → answer)
│   │   │   └── retriever.py     # Retrieval utilities
│   │   ├── processing/
│   │   │   ├── document_loader.py # PDF/DOCX/Web loaders
│   │   │   ├── text_splitter.py
│   │   │   └── translator.py    # Cyrillic ↔ Latin conversion
│   │   └── utils/
│   └── data/vector_store/       # ← SHARED ChromaDB persist directory
```

---

## 8. Key design details

### E5 model prefixes (CRITICAL)

The `intfloat/multilingual-e5-large` model requires explicit task prefixes:

| Operation | Prefix | Applied in |
|---|---|---|
| Indexing (adding docs) | `passage: ` | `vector_store.create_embeddings(is_query=False)` |
| Searching (querying) | `query: ` | `vector_store.create_embeddings(is_query=True)` |

Omitting these prefixes **degrades retrieval quality significantly**.

### Multi-turn conversation (v2)

1. **History is never sent to the answering LLM** — every answer call is `[system, user("Kontekst: <chunks> … Pitanje: <question>")]` via `/api/chat`, so old answers (e.g. one still citing an outdated program) cannot leak into new ones.
2. **Unclear follow-ups are rewritten** — only questions with a reference word (`njih`, `o njima`, `njega`, `to`/`taj`/`tome`, `više o …`, or starting with `A …`/`I …`; see `_UNCLEAR_PATTERNS` in `qa_chain.py`) are rewritten from the last 1–2 turns into a standalone question (e.g. `"Koji su preduslovi za njega?"` → `"Koji su preduslovi za Uvod u programiranje?"`). The rewritten question is used for retrieval **and** is the question the LLM answers. Every other question is used exactly as typed — a bare `"Koliko ESPB?"` is treated as a new question.
3. **Explicit context window** — `num_ctx=8192` is set on every Ollama call.

### Study-program versions (old vs new program)

Program pages hold several versions of the same program (e.g. Informatika
OAS 2026 with 2 modules and 2019 with 3 modules).  The scraper labels every
chunk (`version_label`, `accreditation_year`, `status`) and puts the label in
the chunk header the LLM sees.  The scraper's DeepSeek check also flags
unversioned text that contradicts the current program facts (e.g. an FAQ
still saying "3 smera") with `outdated: true`.
`VectorStoreManager._apply_version_policy` drops `status == "legacy"` and
`outdated` chunks unless the question names an old year (2000–2018) or
"stari program", and collapses duplicates (`dedup_key`); the system prompt
tells the LLM to say which version each fact belongs to.

### Citations

Every retrieved chunk carries metadata (`source`, `filename`, `type`, `page`, `url`).
From the scraper, `filename` is the tab path (e.g. `Informatika › Osnovne (Upis 2026/27) › Računarske nauke`)
and `url` opens that tab where the site supports it (`?osnovne_u_akreditaciji`); PDF links end in `#page=N`.
The LLM sees chunks tagged as `[Izvor 1 — filename, strana N]` and can
reference them inline.  The API appends a `--- Izvori:` block with
deduplicated, human-readable citations.

---

## 9. Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'langchain_community'` | Run `pip install -r requirements.txt` in the activated conda env |
| Ollama connection refused | Ensure `ollama serve` is running in another terminal |
| Empty answers / "Nisam pronašao..." | Run the scraper first to populate the vector store; check `VECTOR_DB_PATH` and `CHROMA_COLLECTION` match |
| New scraper data not visible | Restart the chatbot after every ingest (each process caches the Chroma index) |
| Streaming doesn't work (full response at once) | Ensure `langchain-community` is installed — `ChatOllama` handles token-by-token streaming |
| Cyrillic answers when you want Latin | Set `DEFAULT_SCRIPT=latinic` or pass `"script":"latinic"` in the API request |
| ChromaDB version mismatch | Both projects must use the **same** `chromadb` version — check with `pip show chromadb` |
