# university-chatbot

A Serbian-language university assistant chatbot (RAG pipeline) for the
[IMI PMF Kragujevac](https://imi.pmf.kg.ac.rs/) website.

The chatbot retrieves relevant chunks from a pre-populated ChromaDB
vector store, constructs a grounded prompt, and generates an answer
via a local LLM (Ollama + Llama 3.1).

## What it does

1. **Accepts** a question in Serbian (Latin or Cyrillic — auto-converts).
2. **Embeds** the query with `multilingual-e5-large` using the
   `query: ` prefix (as required by the E5 model family).
3. **Retrieves** the top-k most similar chunks from ChromaDB, filtered
   by a configurable similarity threshold.
4. **Formats** a prompt with the retrieved context and source citations.
5. **Generates** an answer via Ollama / Llama 3.1.

## What it does NOT do

- It does **not** crawl websites or write to ChromaDB.
  That is the job of the separate [university-scraper](../university-scraper) project.
- It opens the ChromaDB persist directory **read-only** and only
  performs similarity search.

## How it connects to university-scraper

The scraper writes to a ChromaDB **persist directory** (default:
`./data/vector_store`).  This chatbot opens the **same directory**
read-only.

A file named `shared_config.json` in that directory records the
embedding model, collection name, E5 prefixes, and HNSW space.
Both projects read it at startup to stay in sync.

> **Important:** Both projects must point to the **same absolute
> `VECTOR_DB_PATH`**.  The default is `./data/vector_store` relative
> to each project root.  If the two projects are in different
> directories, set `VECTOR_DB_PATH` to an absolute shared path via
> a `.env` file or environment variable.

## Quick start

```bash
cd university-chatbot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Make sure Ollama is running with llama3.1:
ollama pull llama3.1:latest

# Run the interactive QA loop:
python run.py
```

To start the FastAPI server instead:

```bash
cd src/api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Configuration

All settings live in `config/settings.py` and can be overridden via
environment variables or a `.env` file.  Key settings:

| Variable | Default | Description |
|---|---|---|
| `VECTOR_DB_PATH` | `./data/vector_store` | ChromaDB persist directory (shared with scraper) |
| `LLM_MODEL` | `llama3.1:latest` | Ollama model name |
| `TOP_K_RESULTS` | `5` | Number of chunks to retrieve |
| `SIMILARITY_THRESHOLD` | `0.7` | Minimum cosine similarity for retrieval |
| `API_PORT` | `8000` | FastAPI server port |

## Project structure (after split)

```
university-chatbot/
├── config/settings.py       # Chatbot-only config
├── src/
│   ├── core/
│   │   ├── llm_manager.py   # Ollama LLM wrapper
│   │   └── vector_store.py  # ChromaDB READ-ONLY (similarity search only)
│   ├── chains/
│   │   └── qa_chain.py      # Prompt construction + source formatting
│   ├── processing/
│   │   └── translator.py    # Cyrillic/Latin conversion
│   └── api/
│       └── main.py          # FastAPI server
├── data/vector_store/       # ChromaDB persist dir (shared with scraper)
├── run.py                   # Interactive QA demo
└── requirements.txt
```
