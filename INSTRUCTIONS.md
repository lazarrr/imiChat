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
| `QUERY_SYNONYMS` | `false` | Better retrieval (§8): add official terms for student wording (`SYNONYMS_FILE`, default `config/synonyms.txt`) |
| `HYBRID_SEARCH` | `false` | Better retrieval (§8): fuse BM25 keyword search with vector search |
| `RERANKER_ENABLED` | `false` | Better retrieval (§8): re-score candidates with `RERANKER_MODEL` (default `BAAI/bge-reranker-v2-m3`) on `RERANKER_DEVICE` (`cpu` / `cuda`) |
| `RERANKER_MIN_SCORE` | `0.0` | Drop chunks the reranker scores below this (0–1); `0` = keep all |
| `EVAL_QUESTIONS_FILE` | `../scraper/eval_questions.yaml` | Golden questions for `python run.py eval` |
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

python run.py eval            # score retrieval on the golden questions
python run.py eval --compare  # baseline vs. synonyms / hybrid / reranker (§8)
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
   `"Koji su preduslovi za Uvod u programiranje?"`. The rewrite prompt gets
   the last 1–2 turns and is told to use **both** the previous user question
   (topic and entity names) and the previous assistant answer (what the
   pronoun refers to).
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
│   ├── config/synonyms.txt      # Synonym groups for QUERY_SYNONYMS
│   ├── run.py                   # CLI entry point (chat, `eval`)
│   ├── src/
│   │   ├── api/main.py          # FastAPI server (routes, wiring)
│   │   ├── api/sessions.py      # Session history (memory or Redis)
│   │   ├── api/middleware.py    # Rate limiting (RATE_LIMIT per client IP)
│   │   ├── core/
│   │   │   ├── llm_manager.py   # Ollama wrappers (/api/chat & /api/generate)
│   │   │   ├── embeddings.py    # Embedding utilities
│   │   │   ├── vector_store.py  # ChromaDB reader (E5 prefixes) + retrieval steps
│   │   │   ├── keyword_search.py # BM25, Serbian tokenizer, synonyms, rank fusion
│   │   │   └── reranker.py      # Cross-encoder reranking
│   │   ├── chains/
│   │   │   ├── qa_chain.py      # Core RAG logic (retrieve → prompt → answer)
│   │   │   └── retriever.py     # Retrieval utilities
│   │   ├── processing/
│   │   │   ├── document_loader.py # PDF/DOCX/Web loaders
│   │   │   ├── text_splitter.py
│   │   │   └── translator.py    # Cyrillic ↔ Latin conversion
│   │   └── utils/
│   │       └── retrieval_eval.py # `run.py eval` — hit@k / MRR on golden questions
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
2. **Unclear follow-ups are rewritten** — only questions with a reference word (`njih`, `o njima`, `njega`, `to`/`taj`/`tome`, `više o …`, or starting with `A …`/`I …`; see `_UNCLEAR_PATTERNS` in `qa_chain.py`) are rewritten from the last 1–2 turns into a standalone question (e.g. `"Koji su preduslovi za njega?"` → `"Koji su preduslovi za Uvod u programiranje?"`). The rewrite prompt (`LLMManager.rewrite_query`) is fed the whole previous turn and told to use both parts: the previous **question** supplies the topic and entity names, the previous **answer** supplies what the pronoun points at; an elliptical follow-up (`"A koliko traje?"`, `"I šta još?"`) inherits the previous question's subject. The rewritten question is used for retrieval **and** is the question the LLM answers. Every other question is used exactly as typed — a bare `"Koliko ESPB?"` is treated as a new question.
3. **Retrieval fallbacks use the history (unclear follow-ups only)** — if the rewritten query returns nothing, the search is retried in order with the original question, then `previous question + current question`, then `last answer + current question`. A clear question never falls back to history.
4. **Explicit context window** — `num_ctx=8192` is set on every Ollama call.
5. **Every turn is traced to `logs/requests.jsonl`** — one JSON object per answered turn, written by the API after the answer streams (`src/api/request_log.py`). It holds the question as typed, the history text the rewrite saw, the full rewrite prompt and raw model output with the guard that accepted or rejected it, every retrieval query tried with hit counts, the chunks sent to the LLM **with their whole vector-store metadata**, the chunks retrieval dropped with the reason and search attempt that dropped them, the exact prompt, the answer with sources, and per-stage timings. Dropped chunks are collected by `similarity_search(..., diagnostics=True)` — an extra Chroma query without the current-version filter, run only when the request is being logged, so it never changes the visible result or costs anything in normal operation. A failure is logged too (`failed: true`), including a client that hung up mid-answer. The chain fills in a `trace` dict passed by the API and never writes files itself; a separate visualization project reads the file.

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

### Better retrieval (optional)

`VectorStoreManager.similarity_search` always runs the vector search
(`RETRIEVAL_FETCH_K` candidates, `SIMILARITY_THRESHOLD`, version policy).
Three extra steps can each be switched on in `.env`; with all of them off
(the default) retrieval is exactly the plain vector search:

1. **`QUERY_SYNONYMS`** — before searching, adds the other terms of any
   synonym group the question mentions (`config/synonyms.txt`: `smer,
   modul`, `OAS, osnovne akademske studije`, …).  Matching ignores case,
   diacritics, script and word endings.
2. **`HYBRID_SEARCH`** — BM25 keyword search over all chunks (a Serbian
   tokenizer folds diacritics and cuts case endings, so `predmeta` matches
   `predmet`), merged with the vector ranking by reciprocal rank fusion.
   Legacy / outdated chunks are filtered the same way as in the vector
   search.  The keyword index is built in memory on the first question and
   rebuilt when the collection's size changes.
3. **`RERANKER_ENABLED`** — the version policy keeps `RETRIEVAL_FETCH_K`
   candidates; a multilingual cross-encoder (`BAAI/bge-reranker-v2-m3`)
   scores each against the original question (0–1) and the best
   `TOP_K_RESULTS` are kept.  `RERANKER_MIN_SCORE` drops weak chunks, so
   the bot answers "nemam informaciju" instead of using loosely related
   text — tune it with the eval below and the scores in the log.

Measure before switching anything on:

```bash
python run.py eval --compare
```

It runs the golden questions (`EVAL_QUESTIONS_FILE`, the scraper's
`eval_questions.yaml`) through baseline, each step alone and all three, and
prints the rank of the right chunk per question plus hit@k, MRR and
seconds per question for each variant.  Add real student questions to that
file to make the comparison meaningful.

### Citations

Every retrieved chunk carries metadata (`source`, `filename`, `type`, `page`, `url`).
From the scraper, `filename` is the tab path (e.g. `Informatika › Osnovne (Upis 2026/27) › Računarske nauke`)
and `url` opens that tab where the site supports it (`?osnovne_u_akreditaciji`); PDF links end in `#page=N`.
The LLM sees chunks tagged as `[Izvor 1 — filename, strana N]` and can
reference them inline.  The API appends a `--- Izvori:` block with
deduplicated, human-readable citations.  Chunks from the same page or
section share one number, so `[Izvor N]` is always line `[N]` of that
block; an answer saying there is no information gets no block.

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
