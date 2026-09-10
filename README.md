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

On turn 2 the chatbot rewrites the follow-up into a standalone question
(e.g. `"Koji su preduslovi za Uvod u programiranje?"`) before retrieving.

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
| `VECTOR_DB_PATH` | `./data/vector_store` | Shared ChromaDB path |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | `0.7` | Cosine similarity cutoff |
| `DEFAULT_SCRIPT` | `latinc` | `latinc` or `cyrillic` |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | API server |

## Project structure

```
.
├── config/
│   └── settings.py            # Configuration (pydantic-settings)
├── run.py                     # CLI entry point
├── src/
│   ├── api/
│   │   └── main.py            # FastAPI server + session management
│   ├── core/
│   │   ├── llm_manager.py     # Ollama wrappers (/api/chat & /api/generate)
│   │   ├── embeddings.py      # Embedding utilities
│   │   └── vector_store.py    # ChromaDB reader (E5 prefixes)
│   ├── chains/
│   │   ├── qa_chain.py        # Core RAG logic (retrieve → prompt → answer)
│   │   └── retriever.py       # Retrieval utilities
│   ├── processing/
│   │   ├── document_loader.py # PDF/DOCX/Web loaders
│   │   ├── text_splitter.py
│   │   └── translator.py      # Cyrillic ↔ Latin conversion
│   └── utils/
├── data/
│   ├── raw/                   # Local files to index
│   └── vector_store/          # Shared ChromaDB persist directory
└── tests/
```

## How it works

1. **Embed** the question with the `query: ` prefix (E5 model).
2. **Retrieve** the top-K most similar chunks from ChromaDB
   (collection `university_knowledge`).
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
