# IVIA RAG Chatbot Backend

> **A production-grade, 5-Layer Intelligent Search Orchestrator** — combining local PDF knowledge, pre-trained LLM reasoning, Google Search, web scraping, and Tavily live web search into a single unified RAG API.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [5-Layer Search Pipeline](#5-layer-search-pipeline)
5. [Features](#features)
6. [Requirements](#requirements)
7. [Installation](#installation)
8. [Configuration](#configuration)
9. [Running the Server](#running-the-server)
10. [API Reference](#api-reference)
11. [Knowledge Base](#knowledge-base)
12. [Malayalam Language Support](#malayalam-language-support)
13. [PDF Ingestion](#pdf-ingestion)
14. [Vector Store](#vector-store)
15. [LLM Backends](#llm-backends)
16. [Development & Testing](#development--testing)

---

## Overview

IVIA RAG Chatbot Backend is a **FastAPI-based REST API** that powers an intelligent question-answering system. It orchestrates a sequential 5-layer fallback pipeline to find the most accurate and reliable answer to any user query:

- **Offline-first**: Prioritizes local PDF knowledge and a curated Q&A knowledge base.
- **LLM-augmented**: Uses Ollama (llama3.2) or HuggingFace models for reasoning.
- **Online fallback**: Falls through to Google Search → Web Scraping → Tavily when local knowledge is insufficient.
- **Multilingual**: Native Malayalam ↔ English translation support.
- **OpenAI-compatible**: Exposes a `/v1/chat/completions` endpoint for drop-in compatibility with OpenAI SDK clients.

---

## Architecture

```
Client Request (POST /chat)
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│                   FastAPI Application (main.py)              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            RAG System Orchestrator (rag.py)           │   │
│  │                                                       │   │
│  │  Pre-processing:                                      │   │
│  │    ├─ Translation detection (Malayalam → English)     │   │
│  │    ├─ Translation request routing                     │   │
│  │    └─ Entity-role validation (question_processor.py)  │   │
│  │                                                       │   │
│  │  5-Layer Sequential Fallback:                         │   │
│  │    Layer 1 ──► Q&A KB + PDF Vector Store (offline)    │   │
│  │    Layer 2 ──► Ollama / HuggingFace LLM (offline)     │   │
│  │    Layer 3 ──► Google Search (online)                 │   │
│  │    Layer 4 ──► Web Scraping (online)                  │   │
│  │    Layer 5 ──► Tavily Live Search (online)            │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
  JSON Response {answer, source_type, confidence, sources}
```

---

## Project Structure

```
bot-roshni/
├── main.py                  # FastAPI app — endpoints & startup
├── rag.py                   # RAGSystem orchestrator (5-layer pipeline)
├── chatbot.py               # LLM loading, QA chains, prompt templates, layer evaluators
├── qa_knowledge.py          # QAKnowledgeBase — curated Q&A with semantic search
├── question_processor.py    # Query analysis, entity-role validation, normalization
├── translator.py            # Malayalam ↔ English translation & detection
├── embeddings.py            # Embedding model loader (sentence-transformers)
├── vector_store.py          # ChromaDB / FAISS vector store management
├── pdf_loader.py            # PDF extraction (pypdf / pdfplumber / PyMuPDF)
├── google_search.py         # Google Search integration (Layer 3)
├── web_scraper.py           # Web scraping & content extraction (Layer 4)
├── tavily_search.py         # Tavily live web search (Layer 5)
├── create_db.py             # Standalone script to (re)build the vector database
├── test_upload.py           # Quick upload/test script
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── data/                    # Knowledge base data directory
│   ├── qa_knowledge.json            # Primary curated Q&A knowledge base
│   ├── ivia_greetings_rag.json      # Greeting & identity Q&A dataset
│   ├── nextgenpro_rag.json          # NextGenPro product Q&A dataset
│   ├── *.pdf                        # Auto-ingested PDF documents
│   └── ...
└── db/                      # Persisted ChromaDB vector store (auto-created)
```

---

## 5-Layer Search Pipeline

The core of the system is `RAGSystem.orchestrate_search()` in `rag.py`. Each layer has a confidence threshold; if a layer's answer meets its threshold, the pipeline **stops immediately** and returns that answer.

### Layer 1 — Custom PDF Knowledge Base (Offline)
- **1a. Direct Q&A Match**: Checks `qa_knowledge.json` + supplemental RAG JSON files (`nextgenpro_rag.json`, `ivia_greetings_rag.json`) for an exact or high-similarity semantic match.
- **1b. Vector DB Retrieval**: Performs semantic search over ChromaDB/FAISS using the uploaded PDF chunks + keyword boosting.
- Threshold: `THRESHOLD_PDF` (default **0.80**)

### Layer 2 — LLM Reasoning (Offline)
- Queries the active LLM (Ollama `llama3.2` or HuggingFace) with any available Q&A context.
- If the LLM returns `NEEDS_WEB_SEARCH` or low-confidence output → proceeds to online layers.
- Threshold: `THRESHOLD_OLLAMA` (default **0.85**)
- In `offline` mode, the pipeline stops here regardless.

### Layer 3 — Google Search (Online)
- Searches Google for the query and collects up to 5 relevant URLs.
- If URLs are found → hands off to Layer 4 (web scraping).
- If no URLs found → skips to Layer 5.

### Layer 4 — Web Scraping (Online)
- Scrapes the top 3 Google URLs, strips ads/navigation/scripts.
- Synthesizes a concise answer from the scraped content via the LLM.
- Threshold: `THRESHOLD_SCRAPING` (default **0.75**)

### Layer 5 — Tavily Live Web Search (Online, Final Fallback)
- Uses the Tavily API for a structured, real-time web search.
- Returns the Tavily-synthesized answer if confidence is sufficient.
- Threshold: `THRESHOLD_TAVILY` (default **0.70**)
- If Tavily is not configured (`TAVILY_API_KEY` missing), this layer is skipped.

If **all layers fail**, the system returns a graceful "could not find reliable information" message.

---

## Features

| Feature | Details |
|---|---|
| **5-layer fallback RAG** | PDF → LLM → Google → Scraping → Tavily |
| **Dual vector stores** | ChromaDB (default) or FAISS |
| **Multi-LLM backends** | Ollama, HuggingFace Transformers, OpenAI |
| **PDF auto-ingestion** | Scans `data/` on startup; skips already-indexed files |
| **Q&A Knowledge Base** | JSON-driven curated Q&A with semantic similarity search |
| **Malayalam support** | Auto-detect, translate input, translate answer back |
| **Translation routing** | Direct translation requests handled without web search |
| **Entity-role validation** | Catches and corrects invalid question combinations |
| **OOM retry logic** | Automatically retries LLM calls on out-of-memory errors |
| **OpenAI-compatible API** | `/v1/chat/completions` endpoint |
| **CORS enabled** | Ready for web/mobile frontend integration |
| **Confidence scoring** | Every response includes a `confidence` score (0.0–1.0) |
| **Source attribution** | Every response includes `sources` with titles, URLs, and page numbers |

---

## Requirements

- **Python** 3.11+
- **Ollama** (recommended): Install from [ollama.ai](https://ollama.ai) and pull your model:
  ```bash
  ollama pull llama3.2
  ```
- **Tavily API Key** (optional, for Layer 5): Get one at [tavily.com](https://tavily.com)
- **CUDA** (optional): For GPU acceleration with HuggingFace models.

### Python Dependencies

```bash
pip install -r requirements.txt
```

Key packages:

| Package | Purpose |
|---|---|
| `fastapi`, `uvicorn` | REST API framework & ASGI server |
| `langchain`, `langchain-community` | RAG pipeline orchestration |
| `langchain-ollama`, `langchain-huggingface` | LLM backend integrations |
| `sentence-transformers` | Embedding model (`all-MiniLM-L6-v2`) |
| `chromadb` | Default vector database |
| `faiss-cpu` | Alternative vector database |
| `pypdf`, `pdfplumber`, `pymupdf` | PDF text extraction (multi-library fallback) |
| `transformers`, `torch` | HuggingFace model inference |
| `deep-translator` | Malayalam ↔ English translation |
| `beautifulsoup4`, `requests` | Web scraping |
| `python-dotenv` | Environment variable loading |
| `pydantic` | Request/response validation |

---

## Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd bot-roshni

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your actual settings

# 5. (Recommended) Pull the default Ollama model
ollama pull llama3.2
```

---

## Configuration

Copy `.env.example` to `.env` and configure:

```env
# ── LLM Configuration ────────────────────────────────────────
# Choices: "ollama" (default), "huggingface", "openai"
DEFAULT_LLM_BACKEND=ollama
DEFAULT_OLLAMA_MODEL=llama3.2
DEFAULT_HF_MODEL=google/flan-t5-base
DEFAULT_OPENAI_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_openai_api_key_here   # Required only if using openai backend

# ── Embedding Model & Vector Store ───────────────────────────
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_STORE_TYPE=chroma          # "chroma" or "faiss"
CHROMA_DB_DIR=db
FAISS_INDEX_DIR=faiss_db

# ── RAG Chunking & Retrieval Parameters ──────────────────────
RAG_TOP_K=6                       # Number of chunks retrieved per query
CHUNK_SIZE=1000                   # Characters per PDF chunk
CHUNK_OVERLAP=200                 # Overlap between consecutive chunks
RAG_MAX_CONTEXT_CHARS=5000        # Max context characters fed to LLM

# ── Q&A Knowledge Base Settings ──────────────────────────────
QA_KNOWLEDGE_FILE=data/qa_knowledge.json
QA_SIMILARITY_THRESHOLD=0.80
QA_TOP_K=3

# ── Search Mode & Confidence Thresholds ──────────────────────
DEFAULT_SEARCH_MODE=auto          # "auto", "offline", or "online"
THRESHOLD_PDF=0.80
THRESHOLD_OLLAMA=0.85
THRESHOLD_SCRAPING=0.75
THRESHOLD_TAVILY=0.70

# ── Tavily Live Web Search API (Layer 5) ─────────────────────
TAVILY_API_KEY=your_tavily_api_key_here
```

### Search Mode Options

| Mode | Behavior |
|---|---|
| `auto` (default) | Offline layers first; falls through to online layers if needed and internet is available |
| `offline` | Only Layers 1 & 2; never makes online requests |
| `online` | Skips Layers 1 & 2; goes directly to Layers 3–5 |

---

## Running the Server

```bash
# Development (auto-reload on file changes)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
```

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **OpenAPI schema**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

---

## API Reference

### `GET /health`

Returns current system status, active search mode, vector store info, and indexed PDF sources.

**Response:**
```json
{
  "status": "healthy",
  "search_mode": "auto",
  "tavily_status": "configured",
  "llm_backend": "ollama",
  "ollama_model": "llama3.2",
  "vector_store": "chroma",
  "indexed_sources": ["Q&A.pdf", "NextGenPro_RAG_Knowledge_Base.pdf"]
}
```

---

### `POST /chat`

Also aliased at: `/query`, `/rag/query`, `/rag/chat`

Process a user query through the 5-layer search orchestrator.

**Request body:**
```json
{
  "query": "Who is the Chief Minister of Kerala?",
  "session_id": "session_123",
  "mode": "auto"
}
```

| Field | Type | Description |
|---|---|---|
| `query` | `string` | User question (also accepted as `question`, `message`, `prompt`, `text`) |
| `messages` | `list` | OpenAI-style messages list (last user message is used) |
| `mode` | `string` | `"auto"`, `"offline"`, or `"online"` |
| `session_id` | `string` | Optional session identifier |

**Response:**
```json
{
  "answer": "Pinarayi Vijayan is the Chief Minister of Kerala.",
  "source_type": "pdf",
  "mode": "offline",
  "confidence": 0.80,
  "sources": [
    {
      "title": "Kerala_Current_CM_and_Ministers_2026_QA.pdf",
      "source": "Kerala_Current_CM_and_Ministers_2026_QA.pdf",
      "url": null,
      "page": 1
    }
  ]
}
```

| Field | Description |
|---|---|
| `answer` | The generated answer |
| `source_type` | `"pdf"`, `"ollama"`, `"web_scraping"`, `"tavily"`, or `"none"` |
| `mode` | Actual mode used: `"offline"` or `"online"` |
| `confidence` | Confidence score (0.0–1.0) |
| `sources` | List of source documents or web pages |

---

### `POST /upload-pdf`

Upload a PDF file to chunk, embed, and index into the vector store.

**Request:** `multipart/form-data` with a `file` field (`.pdf` only).

```bash
curl -X POST "http://localhost:8000/upload-pdf" \
  -F "file=@/path/to/document.pdf"
```

**Response:**
```json
{
  "message": "PDF uploaded and indexed successfully into vector store.",
  "filename": "document.pdf",
  "num_chunks": 47
}
```

---

### `POST /v1/chat/completions`

OpenAI-compatible endpoint for drop-in integration with OpenAI SDK clients.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "What is NextGenPro?"}
  ]
}
```

**Response:** Standard OpenAI format plus additional RAG fields (`answer`, `source_type`, `mode`, `confidence`, `sources`).

---

## Knowledge Base

The Q&A Knowledge Base provides fast, deterministic answers without invoking the LLM.

### Knowledge Files (in `data/`)

| File | Contents |
|---|---|
| `qa_knowledge.json` | Primary curated Q&A knowledge base |
| `ivia_greetings_rag.json` | Greeting, identity, and small-talk responses |
| `nextgenpro_rag.json` | NextGenPro product and service Q&A (~50 entries) |

All `*_rag.json` files in `data/` are automatically loaded into the `QAKnowledgeBase` on startup.

### JSON Format

```json
{
  "documents": [
    {
      "question": "What is IVIA?",
      "answer": "IVIA is an intelligent virtual assistant powered by RAG technology.",
      "tags": ["identity", "ivia"],
      "aliases": ["who are you", "tell me about yourself"]
    }
  ]
}
```

### How It Works

1. **Direct match**: On every query, the KB checks for an exact or near-exact match (cosine similarity ≥ `QA_SIMILARITY_THRESHOLD`).
2. **Context augmentation**: Top-K matching Q&A pairs are passed as context to the LLM in Layers 1 and 2.
3. **Runtime management**: The RAGSystem exposes `add_qa_entry()`, `delete_qa_entry()`, `get_qa_entries()`, and `reload_qa_kb()` for programmatic KB management.

---

## Malayalam Language Support

The system provides automatic bidirectional Malayalam ↔ English translation using `deep-translator`.

**How it works:**
1. Input is analyzed — if detected as Malayalam, it is translated to English for processing.
2. All knowledge lookups, LLM calls, and search operations run in English.
3. The final answer is translated back to Malayalam before returning.

**Translation requests** (e.g. _"translate 'hello' to Malayalam"_) are short-circuited — they bypass all 5 layers and return immediately.

---

## PDF Ingestion

### Auto-Ingestion on Startup

The `data/` folder is scanned automatically on each server start. Any new PDFs not yet indexed are ingested:

1. Text extraction (pypdf → pdfplumber → PyMuPDF, with automatic fallback)
2. Document chunking (configurable `CHUNK_SIZE` / `CHUNK_OVERLAP`)
3. Embedding generation (`all-MiniLM-L6-v2`)
4. Storage in ChromaDB or FAISS

Already-indexed PDFs are **skipped** to avoid duplication.

### Manual Upload

Use the `/upload-pdf` endpoint to upload PDFs at runtime without restarting the server.

### Rebuilding the Database

To clear and rebuild the vector database from scratch:

```bash
python create_db.py
```

---

## Vector Store

| Backend | Configuration | Persistence Directory |
|---|---|---|
| **ChromaDB** (default) | `VECTOR_STORE_TYPE=chroma` | `db/` |
| **FAISS** | `VECTOR_STORE_TYPE=faiss` | `faiss_db/` |

Both stores persist to disk and reload automatically on server restart.

| Variable | Default | Description |
|---|---|---|
| `RAG_TOP_K` | `6` | Number of chunks retrieved per query |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RAG_MAX_CONTEXT_CHARS` | `5000` | Max context fed to the LLM |

---

## LLM Backends

### Ollama (Default)

```env
DEFAULT_LLM_BACKEND=ollama
DEFAULT_OLLAMA_MODEL=llama3.2
```

Requires Ollama running locally at `http://localhost:11434`. Pull the model first:

```bash
ollama pull llama3.2
```

### HuggingFace Transformers

```env
DEFAULT_LLM_BACKEND=huggingface
DEFAULT_HF_MODEL=google/flan-t5-base
```

Model is downloaded from HuggingFace Hub on first run. CUDA is used automatically if available.

### OpenAI

```env
DEFAULT_LLM_BACKEND=openai
DEFAULT_OPENAI_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

### Automatic Fallback

If Ollama fails to load, the server automatically falls back to HuggingFace. If HuggingFace also fails, the server starts without an LLM and returns `503` on `/chat` until a backend is available.

---

## Development & Testing

### Quick Test

```bash
# Health check
curl http://localhost:8000/health

# Chat query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is NextGenPro?"}'

# Offline mode only
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Who is the CM of Kerala?", "mode": "offline"}'
```

### Test Upload Script

```bash
python test_upload.py
```

### Rebuild Vector DB

```bash
python create_db.py
```

### Debugging Tips

- Set `DEFAULT_SEARCH_MODE=offline` during development to avoid hitting external APIs.
- Lower `THRESHOLD_PDF=0.5` temporarily to debug Layer 1 retrieval quality.
- Watch `[rag]` prefixed console logs to trace which layer answered each query.

---

## License

This project is proprietary. All rights reserved.