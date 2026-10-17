# University RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for the IMI PMF Kragujevac
university. It reads a shared ChromaDB vector store, retrieves the most
relevant chunks for a question, and answers using **Llama 3.1 8B** (served
locally by [Ollama](https://ollama.com)).

Answers are in **Serbian (Latin script)** with inline `[Izvor N]` citations.

## Features

- RAG pipeline over university documents (HTML, PDF, DOCX)
- Multi-turn conversations via FastAPI sessions (1-hour TTL)
- Streaming responses (Server-Sent Events)
- Query rewriting for follow-up questions
- Serbian Cyrillic ↔ Latin conversion
- CLI and API interfaces

## Prerequisites

| Component | Required |
|---|---|
| [Conda](https://docs.conda.io) | Environment management |
| [Ollama](https://ollama.com) | Local LLM inference |
| Python ≥ 3.10 | Language runtime |

> This guide assumes a conda environment named **`uni`** (already created on the development machine).

## 1. Setup

```bash
# Activate the environment
conda activate uni

# Install dependencies
pip install -r requirements.txt
```

## 2. Start Ollama

Run in a separate terminal:

```bash
ollama serve
```

Then pull the model (once):

```bash
ollama pull llama3.1:latest
```

Ollama runs at `http://localhost:11434` by default — the chatbot connects
to it automatically. On first run, the `intfloat/multilingual-e5-large`
embedding model is also downloaded (~2.2 GB, one-time).

## 3. Run the chatbot

### CLI mode

```bash
conda activate uni
python run.py
```

Type questions and get answers with sources:

```
Pitanje (ili 'kraj'): Koliko ESPB ima Uvod u programiranje?
Odgovor:
...
Izvori:
  [1] informatika_oas_2021.pdf — strana 12
```

Commands: `kraj` / `quit` / `exit` to stop, `novo` to start a new conversation.

### Server mode (FastAPI)

```bash
conda activate uni
python -m src.api.main
```

The API is served at `http://localhost:8000` (see `config/settings.py` for
`API_HOST` / `API_PORT`).

## 4. API usage

### Health check

```bash
curl http://localhost:8000/health
# → {"status":"healthy","active_sessions":0}
```

### Single-turn question (streaming)

```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"Koliko ESPB ima Uvod u programiranje?"}'
```

The response is Server-Sent Events (SSE), token by token:

```
data: {"token":"Predmet"}
data: {"token":" Uvod"}
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

On turn 2 the chatbot spots the reference word (`njega`) and rewrites the
follow-up into a standalone question (e.g. `"Koji su preduslovi za Uvod u
programiranje?"`). That rewritten question is used for retrieval and is the
only question the LLM sees, together with the retrieved chunks. The rewrite
uses the whole previous turn — the previous **question** names the topic
(elliptical follow-ups like `"A koliko traje?"` inherit it) and the previous
**answer** names what a pronoun such as `njega`/`njima` refers to. The
conversation history itself is never sent to the LLM; it is kept only for
these rewrites. Questions without such a reference are searched and
answered exactly as typed.

## Configuration

All settings live in `config/settings.py` and can be overridden via
environment variables or a `.env` file (see `.env.example`).

| Variable | Default | Notes |
|---|---|---|
| `LLM_MODEL` | `llama3.1:latest` | Must match `ollama list` |
| `LLM_TEMPERATURE` | `0.0` | 0 = deterministic |
| `LLM_MAX_TOKENS` | `2048` | Max response length |
| `LLM_NUM_CTX` | `8192` | Context window (tokens) |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | Embedding model |
| `VECTOR_DB_PATH` | `./data/vector_store` | Shared ChromaDB path (filled by the scraper project) |
| `CHROMA_COLLECTION` | `university_knowledge` | Collection to read (e.g. `university_knowledge_test`) |
| `RETRIEVAL_FETCH_K` | `20` | Candidates fetched before legacy versions / duplicates are dropped |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | `0.7` | Cosine similarity cutoff |
| `QUERY_SYNONYMS` | `false` | Better retrieval: add official terms for student wording (`SYNONYMS_FILE`, default `config/synonyms.txt`) |
| `HYBRID_SEARCH` | `false` | Better retrieval: fuse BM25 keyword search with vector search |
| `RERANKER_ENABLED` | `false` | Better retrieval: re-score candidates with `RERANKER_MODEL` (default `BAAI/bge-reranker-v2-m3`) on `RERANKER_DEVICE` (`cpu` / `cuda`) |
| `RERANKER_MIN_SCORE` | `0.0` | Drop chunks the reranker scores below this (0–1); `0` = keep all |
| `EVAL_QUESTIONS_FILE` | `../scraper/eval_questions.yaml` | Golden questions for `python run.py eval` |
| `DEFAULT_SCRIPT` | `latinc` | `latinc` or `cyrillic` |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | API server |
| `RATE_LIMIT` | `100` | `/chat` requests per minute per client IP (`0` = no limit); HTTP 429 above it |
| `REDIS_URL` | *(unset)* | Optional, e.g. `redis://localhost:6379/0` — sessions and rate limits shared by all workers and kept across restarts (otherwise in memory) |
| `REQUEST_LOG_ENABLED` | `true` | Write one JSON trace line per answered turn |
| `REQUEST_LOG_PATH` | `./logs/requests.jsonl` | Where that file lives |
| `REQUEST_LOG_MAX_BYTES` / `REQUEST_LOG_BACKUPS` | `20000000` / `5` | Rotate the file at ~20 MB, keeping `requests.jsonl.1` … `.5` |
| `REQUEST_LOG_MAX_TEXT` | `8000` | Per-field cap (chars) for prompts, chunk bodies and history text |
| `REQUEST_LOG_MAX_EXCLUDED` | `40` | Dropped chunks logged per turn (the total is always logged) |

## Request log (for the visualization project)

Every answered turn appends **one JSON object per line** to
`logs/requests.jsonl` (see `src/api/request_log.py`). The file is the entire
interface between this project and a separate project that visualizes what
the chatbot did — the visualizer needs no access to this code, ChromaDB or
Ollama, only the file. Nothing here ever reads it back.

Each line answers "what did the system actually search for, and why?":

| Field | Contents |
|---|---|
| `request_id`, `timestamp`, `session_id`, `client`, `duration_ms` | Who asked, which session, when, how long |
| `summary` | Flat one-liner for a list view: rewritten?, strategy, chunk/source counts, failed? |
| `question`, `search_question`, `unclear` | As typed, as searched, and the unclear verdict |
| `history.session_turns` / `messages_included` / `text` | Exactly the last 1–2 turns the rewrite was allowed to see |
| `rewrite.prompt`, `.raw_response`, `.result`, `.matched_pattern` | The full rewrite prompt, the model's raw output, and the guard that accepted or rejected it (`accepted`, `empty`, `too_short`, `too_long`, `unchanged`, `skipped_clear_question`, `skipped_no_history`) — with `rejected` holding the discarded candidate |
| `retrieval.attempts`, `.winning_strategy` | Every query tried in order with its hit count and top distance, including the ones that found nothing |
| `chunks` | The chunks that went into the prompt, with source, page, distance, body and **the whole vector-store metadata dict** (`metadata`) — every field, including ones the chatbot does not use |
| `chunks_excluded` | Chunks that were retrieved but rejected before the prompt, each with its `reason` (below the similarity threshold, previous program version, outdated, duplicate, reranker, did not fit in top-k), its metadata, its text and the search attempt that produced it. Capped at `REQUEST_LOG_MAX_EXCLUDED`; `chunks_excluded_total` always holds the real count |
| `prompt.user_content` | The exact message sent to the answering model (no history) |
| `answer.text`, `.sources` | The model's answer and the rendered `Izvori` block |
| `timings_ms` | Per stage: `rewrite`, `retrieval`, `answer`, `total` |
| `failed`, `error` | Set when the turn ended in an error; a client that hung up mid-answer is still logged |

Long fields are capped at `REQUEST_LOG_MAX_TEXT` chars and carry
`<field>_chars`, `<field>_full_chars` and `<field>_truncated`, so a truncated
prompt is recognizable as truncated. `schema_version` (currently `1`) is on
every record; readers should ignore unknown keys.

```bash
# peek at the last turn
tail -n 1 logs/requests.jsonl | python -m json.tool
```

## Better retrieval (optional)

Three retrieval steps can each be switched on or off in `.env`. All three are
off by default, so retrieval is the plain vector search unless you enable them:

| Setting | What it does | Cost |
|---|---|---|
| `QUERY_SYNONYMS=true` | Adds official terms for student wording before searching (`smerovi` → also `modul`, `OAS` → `osnovne akademske studije`). Edit `config/synonyms.txt` to add your own. | none |
| `HYBRID_SEARCH=true` | Adds BM25 keyword search to the vector search and merges both rankings. Finds exact names, codes and numbers that embeddings miss. | builds a keyword index in memory on the first question |
| `RERANKER_ENABLED=true` | A cross-encoder reads the question together with each of the `RETRIEVAL_FETCH_K` candidates and keeps the best `TOP_K_RESULTS`. `RERANKER_MIN_SCORE` can also drop weak chunks, which lets the bot say "nemam informaciju" instead of answering from loosely related text. | ~2.3 GB model download; a few seconds per question on CPU (`RERANKER_DEVICE=cuda` if you have a GPU) |

Compare them on the golden questions (the scraper's `eval_questions.yaml`)
before switching them on:

```bash
python run.py eval              # with the steps switched on in .env
python run.py eval --compare    # baseline vs. each step vs. all three
```

`--compare` prints the rank of the right chunk for every question under each
variant, then hit@k, MRR and seconds per question for each variant.

## Project structure

```
.
├── config/
│   ├── settings.py            # Configuration (pydantic-settings)
│   └── synonyms.txt           # Synonym groups for QUERY_SYNONYMS
├── run.py                     # CLI entry point (chat, `eval`)
├── src/
│   ├── api/
│   │   ├── main.py            # FastAPI server (routes, wiring)
│   │   ├── sessions.py        # Session history (memory or Redis)
│   │   ├── request_log.py     # Per-turn JSONL trace read by the visualizer
│   │   └── middleware.py      # Rate limiting (RATE_LIMIT per client IP)
│   ├── core/
│   │   ├── llm_manager.py     # Ollama wrappers (/api/chat & /api/generate)
│   │   ├── embeddings.py      # Embedding utilities
│   │   ├── vector_store.py    # ChromaDB reader (E5 prefixes) + retrieval steps
│   │   ├── keyword_search.py  # BM25, Serbian tokenizer, synonyms, rank fusion
│   │   └── reranker.py        # Cross-encoder reranking
│   ├── chains/
│   │   ├── qa_chain.py        # Core RAG logic (retrieve → prompt → answer)
│   │   └── retriever.py       # Retrieval utilities
│   ├── processing/
│   │   ├── document_loader.py # PDF/DOCX/Web loaders
│   │   ├── text_splitter.py
│   │   └── translator.py      # Cyrillic ↔ Latin conversion
│   └── utils/
│       └── retrieval_eval.py  # `run.py eval` — hit@k / MRR on golden questions
├── data/
│   ├── raw/                   # Local files to index
│   └── vector_store/          # Shared ChromaDB persist directory
├── logs/
│   └── requests.jsonl         # Per-turn trace (gitignored) — visualizer input
└── tests/
```

## How it works

1. **Embed** the question with the `query: ` prefix (E5 model).
2. **Retrieve** the top-K most similar chunks from ChromaDB
   (collection `university_knowledge`), optionally with synonyms, hybrid
   keyword search and reranking (see [Better retrieval](#better-retrieval-optional)).
3. **Build** a prompt with `[Izvor N — filename, strana N]` context.
4. **Generate** the answer via Ollama `/api/chat`, streaming tokens.
5. **Append** a deduplicated `--- Izvori:` citation block.

> The E5 model requires explicit task prefixes — `passage: ` when indexing
> and `query: ` when searching. Omitting them degrades retrieval quality.

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'langchain_community'` | Run `pip install -r requirements.txt` in the `uni` env |
| Ollama connection refused | Ensure `ollama serve` is running in another terminal |
| Empty answers / "Nisam pronašao..." | Populate the vector store first; check `VECTOR_DB_PATH` |
| Streaming returns a single chunk | Ensure `langchain-community` is installed |
| Cyrillic answers when you want Latin | Set `DEFAULT_SCRIPT=latinc` or pass `"script":"latinic"` in the API request |

## Web UI (Flutter)

A modern web chat interface lives in [`web_ui/`](web_ui/). It talks to the
FastAPI server above (streaming via SSE, multi-turn sessions, source
citations, dark mode, persisted conversations).

Keyboard: `Enter` sends the question, `Shift + Enter` inserts a line break.
The chat history is searchable and grouped by recency in the sidebar.

### Run (development)

```bash
# 1. Backend
conda activate uni
python -m src.api.main

# 2. Frontend (another terminal)
cd web_ui
flutter run -d chrome --web-port 5000
```

The frontend targets `http://localhost:8000` by default. Point it at a
different backend with a query parameter:

```
http://localhost:5000/?api=http://other-host:8000
```

### Build for production

```bash
cd web_ui
flutter build web --release
# static output in web_ui/build/web — serve with any static file server
```

## Tests

```bash
conda activate uni
pytest
```
