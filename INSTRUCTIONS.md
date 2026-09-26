# University RAG Chatbot — Instructions

## Overview

A Retrieval-Augmented Generation (RAG) chatbot for the IMI PMF Kragujevac
university, split into two projects that share a single ChromaDB vector store:

| Project | What it does |
|---|---|
| `university-scraper` | Crawls the university website, extracts text from HTML/PDF/DOCX, chunks, embeds, and upserts into ChromaDB. |
| `university-chatbot` | Reads the shared ChromaDB store, retrieves relevant chunks, and answers questions via Llama 3.1 8B (Ollama). |

Answers are in **Serbian (Latin script)** with no citation markers in the
text; the sources the model used travel in a structured `sources` payload.

---

## Prerequisites

| Component | Required |
|---|---|
| **Conda** (Miniconda or Anaconda) | Environment management |
| **Ollama** | Local LLM inference (`brew install ollama` on macOS) — only if `LLM_USE_LOCAL=True` |
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

## 2. Ollama setup (only when `LLM_USE_LOCAL=True`)

The default backend is the remote OpenAI-compatible server, so this section
is only needed if you switch to local inference.

```bash
# Start the Ollama server (keep this running in a separate terminal)
ollama serve

# Pull the LLM (in another terminal, or before starting the server)
ollama pull llama3.1:latest
```

Ollama runs at `http://localhost:11434` by default.  The chatbot
connects to it automatically when `LLM_USE_LOCAL=True` — no extra
configuration needed.  With the default `LLM_USE_LOCAL=False` the
chatbot instead calls the remote OpenAI-compatible server configured
via `LLM_REMOTE_*`.

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
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |

### Chatbot-specific (`.env` in `university-chatbot/`)

| Variable | Default | Notes |
|---|---|---|
| `LLM_USE_LOCAL` | `False` | `True` = local Ollama, `False` = remote API |
| `LLM_MODEL` | `llama3.1:latest` | Local model; must match `ollama list` |
| `LLM_REMOTE_BASE_URL` | `http://dyres.pmf.kg.ac.rs:11500/v1` | OpenAI-compatible root |
| `LLM_REMOTE_MODEL` | `llama3.1:8b` | Model name on the remote server |
| `LLM_REMOTE_API_KEY` | — | `Authorization: Bearer <key>` (see `.env.example`) |
| `LLM_REMOTE_TIMEOUT` | `300` | Read timeout in seconds |
| `LLM_REPEAT_PENALTY` | `1.2` | Repetition penalty; raise if answers loop (`1.0` = off) |
| `LLM_REPEAT_LAST_N` | `256` | Repetition window in tokens (local Ollama only) |
| `LLM_TEMPERATURE` | `0.0` | 0 = deterministic |
| `LLM_MAX_TOKENS` | `2048` | Max response length |
| `LLM_NUM_CTX` | `8192` | Context window (tokens) |
| `SIMILARITY_THRESHOLD` | `0.7` | Cosine similarity cutoff |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |
| `API_HOST` | `0.0.0.0` | |
| `API_PORT` | `8000` | |

### Scraper-specific (`.env` in `university-scraper/`)

| Variable | Default | Notes |
|---|---|---|
| `CRAWLER_MAX_DEPTH` | `1` | How deep to follow links |
| `CRAWLER_DELAY` | `1.0` | Seconds between requests |
| `CRAWLER_INCLUDE_PATTERNS` | `[...]` | URL paths to include |
| `CRAWL_HISTORY_ENABLED` | `True` | Skip already-scraped URLs |

---

## 4. Running the scraper (populate the vector store)

Run this **first** — it populates the shared ChromaDB store that the
chatbot reads from.

```bash
conda activate uni
cd university-scraper
python run.py
```

What happens:
1. Crawls `https://imi.pmf.kg.ac.rs/` (breadth-first, respects `robots.txt`).
2. Extracts clean text from HTML, PDF, and DOCX resources.
3. Chunks text (1000 chars, 200 overlap).
4. Embeds with `intfloat/multilingual-e5-large` (using `passage: ` prefix).
5. Upserts into ChromaDB collection `university_knowledge`.
6. Records ingested URLs in `data/crawl_history.json` → subsequent runs skip them.

Set `CRAWL_HISTORY_ENABLED=false` to force a full re-crawl.

> **The persist directory** (`data/vector_store/`) is shared between
> projects.  A `shared_config.json` is written there on first run to
> keep both projects in sync.

---

## 5. Running the chatbot

```bash
conda activate uni
cd university-chatbot
python run.py          # CLI interactive mode (single-turn, no session)
# OR
python -m src.api.main  # FastAPI server (multi-turn with sessions)
```

### CLI mode (`run.py`)

Simple REPL loop.  Type questions, get answers + sources.  **No multi-turn
memory** — each question is independent.

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

Response is **Server-Sent Events** (SSE) — tokens arrive one by one, followed
by a structured `sources` event and a final `done` marker:
```
data: {"token":"Predmet"}
data: {"token":" Uvod"}
data: {"token":" u"}
...
data: {"sources":[{"number":1,"filename":"informatika_oas_2021.pdf","page":12,"url":"","type":"pdf","label":"informatika_oas_2021.pdf — strana 12"}]}
data: {"done":true}
```

Source numbers come from a single per-query citation registry: the LLM tags
facts with `[Izvor N]`, those markers are stripped from the streamed answer,
and only the numbers that are valid and actually cited are sent in the
`sources` event.  When the model answers from the context without any
markers, the answer is attributed back to the chunks it visibly draws from
(rare-word overlap), so references still appear without listing unrelated
documents.  On a refusal the `sources` list is empty; the server never
appends a textual `--- Izvori:` block to the stream.

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
1. Detects this is a follow-up (session `abc123` has history).
2. Skips the LLM rewrite if a heuristic already considers the question
   standalone. A short predicate follow-up that names no topic of its own
   ("Od kad radi sa studentima?" after "Od kada radi studentska služba?")
   is not considered standalone: its elided subject is filled in by the
   rewrite from the previous answer. Referential follow-ups
   (`"Šta možeš o njima?"`) and other context-dependent questions are
   rewritten by Llama 3.1 (bounded by a timeout, temperature 0, fixed
   seed) using the model's previous *answer* and the latest question;
   successful rewrites are cached by `(history, question)` so they are
   reproducible. A rewrite that drifts from the conversation topic, or
   that swaps the topic explicitly named in the latest question for the
   one from the previous answer, is rejected. On failure, timeout or
   rejection a deterministic fallback retrieves with the previous clear
   question (plus the new one when it has its own topic).
3. Embeds the **rewritten** question (with `query: ` prefix) and retrieves
   relevant chunks with hybrid dense (E5) + BM25 search, fused by
   Reciprocal Rank Fusion. If that returns nothing, the original question
   is retried verbatim (no other fallbacks).
4. Sends the full message list (system prompt + history + new
   context-augmented question) to the configured LLM backend (local
   Ollama `/api/chat` or remote `/v1/chat/completions`).
5. Streams the answer token-by-token with `[Izvor N]` markers removed
   (records them internally) and emits a structured `sources` event with
   only the sources actually cited; refusals carry no sources.
6. Saves both the original question and answer to the session history.

Sessions expire after **1 hour** of inactivity.

---

## 7. Project structure

```
master/
├── university-scraper/          # Scraper & embedding project
│   ├── config/settings.py       # Scraper config
│   ├── run.py                   # Entry point
│   ├── src/
│   │   ├── core/vector_store.py # ChromaDB wrapper (E5 prefixes)
│   │   └── processing/
│   │       ├── crawler.py       # Domain crawler
│   │       ├── extractor.py     # HTML/PDF/DOCX extraction
│   │       ├── ingest.py        # Crawl → Extract → Chunk → Embed pipeline
│   │       └── crawl_history.py # Already-ingested URL tracker
│   └── data/
│       ├── raw/                 # (optional) local files to ingest
│       ├── processed/
│       └── vector_store/        # ← SHARED ChromaDB persist directory
│
├── university-chatbot/          # Retrieval & chat project
│   ├── config/settings.py       # Chatbot config
│   ├── run.py                   # CLI entry point
│   ├── src/
│   │   ├── api/                 # FastAPI app: main.py assembles the app;
│   │   │                        # schemas.py, sessions.py, dependencies.py,
│   │   │                        # middleware.py, routes/{chat,health}.py
│   │   ├── core/
│   │   │   ├── llm_manager.py   # Local Ollama / remote OpenAI-compatible LLM
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

Three fixes were applied to enable proper multi-turn conversations:

1. **Full history sent to LLM** — every turn sends `[system, user1, assistant1, user2, ...]` via `/api/chat`.
2. **Query contextualization** — follow-up questions are rewritten into standalone form before embedding (e.g. `"Koliko ESPB ima?"` → `"Koliko ESPB ima Uvod u programiranje?"`).
3. **Context window guard** — `num_ctx=8192` is set explicitly; oldest history turns are trimmed if the budget is exceeded.

### Citations

Every retrieved chunk carries metadata (`source`, `filename`, `type`, `page`, `url`).
A single `CitationRegistry` deduplicates `(source, page)` pairs and assigns
each one a stable `[Izvor N]` number. The LLM sees chunks tagged as
`[Izvor N — filename, strana M]`; the markers are stripped from the answer
text while the cited numbers are recorded, and the API emits the very same
numbering in the structured `sources` SSE event. Only sources the model
actually cited are emitted; uncited grounded answers are attributed back to
the chunks they share rare words with. Invalid numbers are dropped and
refusals suppress sources entirely.

### Retrieval

Dense retrieval (`intfloat/multilingual-e5-large`) ranks chunks by cosine
similarity in ChromaDB. An optional lexical BM25 stage can be fused in with
weighted Reciprocal Rank Fusion by setting `HYBRID_RETRIEVAL_ENABLED=true`
(`BM25_FUSION_WEIGHT`, default `0.1`) — it is disabled for now, so only the
cosine similarity search runs, and the BM25 cache is not loaded. An optional
cross-encoder rerank (`mmarco-mMiniLMv2`) can be enabled with
`RERANK_ENABLED=true`.

### Evaluation

A small golden set lives in `data/eval/golden_set.json`. It scores
retrieval hit@k, citation validity/grounding and refusal rates:

```bash
python scripts/eval_golden.py --with-llm          # full evaluation
python scripts/eval_golden.py                     # retrieval metrics only
python scripts/eval_golden.py --no-hybrid         # dense-only ablation
```

---

## 9. Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'langchain_community'` | Run `pip install -r requirements.txt` in the activated conda env |
| Ollama connection refused | Ensure `ollama serve` is running in another terminal |
| Empty answers / "Nisam pronašao..." | Run the scraper first to populate the vector store; check `VECTOR_DB_PATH` matches |
| Streaming doesn't work (full response at once) | Ensure `langchain-community` is installed — `ChatOllama` handles token-by-token streaming |
| Cyrillic answers when you want Latin | Set `DEFAULT_SCRIPT=latinic` or pass `"script":"latinic"` in the API request |
| ChromaDB version mismatch | Both projects must use the **same** `chromadb` version — check with `pip show chromadb` |
