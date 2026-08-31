# IVIA RAG Chatbot Backend 🤖📚

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C.svg?style=flat&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-FF6600.svg?style=flat)](https://www.trychroma.com/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU%201.8+-00599C.svg?style=flat)](https://github.com/facebookresearch/faiss)
[![Ollama](https://img.shields.io/badge/Ollama-llama3.2-black.svg?style=flat&logo=ollama&logoColor=white)](https://ollama.ai/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E.svg?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat)](LICENSE)

**IVIA** is an enterprise-grade, production-ready **Retrieval-Augmented Generation (RAG)** chatbot backend powered by **FastAPI**. Engineered for edge, local workstation, and cloud deployments (serving Flutter/mobile, React/Vue web applications, and microservices), IVIA seamlessly ingests multi-format PDF documents and curated JSON datasets into persistent vector stores.

It resolves user queries using a **5-Layer Sequential Fallback Architecture with Strict Early Stopping**, a **7-Step Question Understanding & Entity-Role Validation Pipeline**, a **Bilingual Malayalam ↔ English Translation Layer with Term Preservation Masking**, and **Multi-Engine Live Web Fallbacks** (Google Search URL Discovery + Clean HTML Scraping/Sanitization + Tavily Live Web Search API).

---

## 📑 Table of Contents

- [🌟 Key Features](#-key-features)
- [🏗️ System Architecture](#️-system-architecture)
- [🧩 5-Layer Search Fallback Pipeline](#-5-layer-search-fallback-pipeline)
  - [Layer 1: Custom PDF & Q&A Knowledge Base](#layer-1-custom-pdf-knowledge-base--qa-kb)
  - [Layer 2: Ollama Reasoning & Static Fact Check](#layer-2-ollama-reasoning--static-fact-check)
  - [Layer 3: Google Search URL Discovery](#layer-3-google-search-url-discovery)
  - [Layer 4: Web Scraping & Synthesis Engine](#layer-4-web-scraping--synthesis-engine)
  - [Layer 5: Tavily Live Web Search (Final Fallback)](#layer-5-tavily-live-web-search-final-fallback)
  - [🎯 Configurable Confidence Thresholds](#-configurable-confidence-thresholds)
- [🔍 7-Step Question Understanding Engine](#-7-step-question-understanding-engine)
- [🗣️ Malayalam ↔ English Translation Engine](#️-malayalam--english-translation-engine)
- [🗄️ Knowledge Base, Datasets & Auto-Ingestion](#️-knowledge-base-datasets--auto-ingestion)
- [📁 Project Directory Structure](#-project-directory-structure)
- [🛠️ Prerequisites & Installation](#️-prerequisites--installation)
- [⚙️ Environment Configuration (.env)](#️-environment-configuration-env)
- [🚀 Running the Application](#-running-the-application)
- [📡 API Reference & Endpoints](#-api-reference--endpoints)
  - [POST /chat](#post-chat)
  - [POST /upload-pdf](#post-upload-pdf)
  - [GET /health](#get-health)
  - [GET /](#get-)
- [📱 Client Integration Examples](#-client-integration-examples)
  - [Flutter / Dart](#flutter--dart)
  - [JavaScript / TypeScript (Fetch / React)](#javascript--typescript-fetch--react)
  - [Python Client (Requests & HTTPX Async)](#python-client-requests--httpx-async)
  - [cURL](#curl)
- [🔧 CLI Utilities & Management Scripts](#-cli-utilities--management-scripts)
  - [1. Pre-indexing PDFs (create_db.py)](#1-pre-indexing-pdfs-create_dbpy)
  - [2. Automated Upload Testing (test_upload.py)](#2-automated-upload-testing-test_uploadpy)
  - [3. Test PDF Generator (make_pdf.py)](#3-test-pdf-generator-make_pdfpy)
- [🌐 LLM & Vector Store Backends](#-llm--vector-store-backends)
- [❓ Troubleshooting & FAQ](#-troubleshooting--faq)
- [📜 License](#-license)

---

## 🌟 Key Features

- 🧠 **5-Layer Sequential Search Pipeline** — Prioritizes offline local knowledge before triggering online search, applying strict early stopping as soon as a verified, high-confidence answer is obtained to eliminate latency and API costs.
- 📄 **Multi-Engine PDF Ingestion** — Layout-accurate text extraction via `pdfplumber` with fallback to `PyMuPDF` (`fitz`) and `pypdf`, automated recursive chunking, and metadata tagging.
- 🔄 **Zero-Downtime Hot Drop Auto-Ingestion** — Automatically detects and indexes new `.pdf` files dropped into the `data/` folder on startup or at request time without requiring manual database rebuilds.
- 🗂️ **Dual Vector Store Support** — Seamless switching between disk-persistent **ChromaDB** and in-memory/persisted **FAISS** vector indexes.
- 🗃️ **Curated JSON Knowledge Bases** — Instant semantic matching and alias lookup for high-frequency Q&A (`qa_knowledge.json`, `ivia_greetings_rag.json`, `nextgenpro_rag.json`).
- 🛡️ **Role & Intent Protection Matrix** — Pre-retrieval validation that prevents hallucinations for administrative, political, and corporate roles (e.g., distinguishing Prime Minister vs. Chief Minister, Governor vs. President).
- 🦙 **Flexible Multi-Backend LLM Engine** — Primary local inference with **Ollama** (`llama3.2`), automatic fallback to local **HuggingFace** pipelines (`google/flan-t5-base`), and cloud **OpenAI** GPT API support (`gpt-4o-mini`).
- 🌐 **Robust Online Fallback Mechanism** — Real-time Google Search scraping with DuckDuckGo fallback, clean HTML content sanitization via `BeautifulSoup4`, and final fallback to the **Tavily Web Search API**.
- 🗣️ **Bilingual Support (English ↔ Malayalam)** — Automatic Malayalam Unicode script detection, query translation with placeholder-based term protection, direct translation intent bypass, and response localization.
- ⚡ **Production-Ready FastAPI Server** — OpenAPI/Swagger docs (`/docs`), ReDoc (`/redoc`), CORS middleware enabled for all origins, health monitoring (`/health`), and structured Pydantic response models.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    User([User Query / Client App]) --> Trans{Language Check\n(Unicode Script)}
    Trans -- "Malayalam [\u0D00-\u0D7F]" --> ML_EN[Translate Query to English\n(Term Preservation Masking)]
    Trans -- "English" --> QP[7-Step Question Processor]
    ML_EN --> QP

    QP --> TransIntent{Translation\nIntent?}
    TransIntent -- "Yes (e.g. Translate 'Hello' to ML)" --> DirectTrans[Execute Direct Translation] --> OutTrans
    TransIntent -- "No" --> RoleCheck{Role & Entity\nValid Matrix?}

    RoleCheck -- "Invalid (e.g. Governor of India)" --> Correction[Generate Concept Clarification] --> OutTrans
    RoleCheck -- "Valid / Normalized" --> L1[Layer 1: PDF + Q&A Knowledge Base]

    subgraph "5-Layer Search Orchestrator"
        L1 --> L1_Check{Direct Match >= 0.95 or\nPDF Eval >= 0.80?}
        L1_Check -- Yes --> ResL1[Return PDF / Q&A Answer\nsource_type='pdf']

        L1_Check -- No / Insufficient --> L2[Layer 2: Ollama / LLM Reasoning]
        L2 --> L2_Check{Confident Answer?\n!= NEEDS_WEB_SEARCH}
        L2_Check -- Yes --> ResL2[Return LLM Answer\nsource_type='ollama']

        L2_Check -- No / Uncertain --> ModeCheck{Search Mode\nOffline?}
        ModeCheck -- Yes --> OfflineNotice[Return Offline KB Notice]
        ModeCheck -- No --> NetCheck{Internet Available?\n(Ping 8.8.8.8)}
        
        NetCheck -- No --> OfflineFallback[Return Offline Connectivity Notice]
        NetCheck -- Yes --> L3[Layer 3: Google Search URL Discovery]

        L3 --> L3_Check{URLs Found?}
        L3_Check -- Yes --> L4[Layer 4: Web Content Scraping & Synthesis]
        L4 --> L4_Check{Scraped Confidence\n>= 0.75?}
        L4_Check -- Yes --> ResL4[Return Scraped Web Answer\nsource_type='web_scraping']

        L4_Check -- No --> L5[Layer 5: Tavily Live Web Search]
        L3_Check -- No --> L5
        L5 --> L5_Check{Tavily Verified?\nConfidence >= 0.70}
        L5_Check -- Yes --> ResL5[Return Tavily Search Answer\nsource_type='tavily']
        L5_Check -- No --> Fallback[Return 'No reliable information found']
    end

    ResL1 --> OutTrans{Original Query\nMalayalam?}
    ResL2 --> OutTrans
    ResL4 --> OutTrans
    ResL5 --> OutTrans
    Correction --> OutTrans
    DirectTrans --> OutTrans
    OfflineNotice --> OutTrans
    OfflineFallback --> OutTrans
    Fallback --> OutTrans

    OutTrans -- Yes --> EN_ML[Translate Answer to Malayalam\n(Restore Placeholders)] --> Response([FastAPI JSON Response])
    OutTrans -- No --> Response
```

---

## 🧩 5-Layer Search Fallback Pipeline

The search orchestrator (`rag.py`) executes query resolution in a strict sequential order. **The first layer to produce a confident, verified answer terminates the pipeline immediately.**

```text
[Layer 1: PDF Knowledge Base + JSON Q&A]
      │
      ├── Exact / High-Confidence Match? (confidence >= 0.80 | direct match >= 0.95)
      │     └── YES ──> Return answer (source_type="pdf", mode="offline") ──> [STOP!]
      └── NO
            │
            ▼
[Layer 2: Ollama Reasoning & Knowledge Check]
      │
      ├── Confident Answer? (general knowledge / facts, != NEEDS_WEB_SEARCH)
      │     └── YES ──> Return answer (source_type="ollama", mode="offline") ──> [STOP!]
      └── NO / NEEDS_WEB_SEARCH
            │
            ▼
[Layer 3: Google Search (URL Discovery)]
      │
      ├── Relevant Authority URLs Found?
      │     ├── YES ──> [Layer 4: Web Scraping & Sanitization]
      │     │                 │
      │     │                 ├── Reliable Answer? (confidence >= 0.75)
      │     │                 │     └── YES ──> Return (source_type="web_scraping", mode="online") ──> [STOP!]
      │     │                 └── NO ─────────────────────────────────┐
      │     └── NO ──────────────────────────────────────────────┐ │
      │                                                           ▼ ▼
      └─────────────────────────────────────> [Layer 5: Tavily Live Web Search]
                                                                  │
                                                                  ├── Reliable Answer? (confidence >= 0.70)
                                                                  │     └── YES ──> Return (source_type="tavily", mode="online") ──> [STOP!]
                                                                  └── NO ──> Return Graceful Fallback Notice ──> [STOP!]
```

### Layer Details

#### Layer 1: Custom PDF Knowledge Base & Q&A KB
- **Mechanism:** Searches pre-indexed ChromaDB / FAISS vector stores and JSON Q&A datasets (`qa_knowledge.json`, `ivia_greetings_rag.json`, `nextgenpro_rag.json`).
- **Direct Match Check:** If cosine similarity on question or aliases is $\ge 0.95$, immediately returns the answer without LLM invocation.
- **Context Synthesis:** Ingests top-$k$ retrieved chunks with keyword boosting through the IVIA persona prompt. If the answer is present, returns `source_type="pdf"`. If absent (`NO_PDF_ANSWER`), escalates to Layer 2.

#### Layer 2: Ollama Reasoning & Static Fact Check
- **Mechanism:** Evaluates whether static general knowledge (history, geography, mathematics, established definitions) can answer the question with high certainty.
- **Escalation:** If the query requires real-time information or the model is uncertain, it outputs `NEEDS_WEB_SEARCH` and escalates to online search (Layer 3).

#### Layer 3: Google Search URL Discovery
- **Mechanism:** Performs live Google Search using realistic browser headers and URL sanitization.
- **Authority Filter:** Prioritizes `.gov`, `.nic.in`, `.org`, `.edu`, `wikipedia.org`, and accredited news sources while discarding social media links and video feeds.
- **DuckDuckGo Fallback:** If Google blocks or rate-limits requests, automatically falls back to DuckDuckGo HTML search.

#### Layer 4: Web Scraping & Synthesis Engine
- **Mechanism:** Scrapes and sanitizes readable text from collected URLs via `BeautifulSoup4` (stripping scripts, styling, navigation, headers, footers, forms, and SVGs).
- **JSON Synthesis Prompt:** Uses a structured LLM prompt with a strong bias toward answering across multi-page content:
  - `0.90 - 1.00`: Direct and explicit match.
  - `0.80 - 0.89`: Reasonable multi-sentence inference across pages.
  - `0.75 - 0.79`: Partial or indirect topical answer.
  - `< 0.75`: Only returned when pages are blocked, empty, or completely off-topic.
- **Output Validation:** Automatically parses structured JSON output (`{"answer": "...", "confidence": 0.85, "reasoning": "..."}`).

#### Layer 5: Tavily Live Web Search (Final Fallback)
- **Mechanism:** Leverages Tavily's real-time search API optimized for LLMs.
- **Termination:** Synthesizes the final verified answer (`source_type="tavily"`) or returns a graceful offline/no-information notice if all layers fail.

### 🎯 Configurable Confidence Thresholds

| Layer | Component | Default Threshold | Description |
|---|---|---|---|
| **Layer 1** | PDF Vector Store / Q&A KB | `>= 0.80` (Direct: `0.95`) | Vector cosine similarity & LLM context verification |
| **Layer 2** | Ollama General Knowledge | `>= 0.85` | Evaluates static facts without real-time drift |
| **Layer 4** | Web Scraping & Synthesis | `>= 0.75` | Structured JSON confidence evaluation from scraped HTML |
| **Layer 5** | Tavily Live Web Search | `>= 0.70` | Final online fallback using Tavily's search index |

---

## 🔍 7-Step Question Understanding Engine

Implemented in `question_processor.py`, every question passes through an analytical pipeline before vector retrieval:

1. **Question Normalization:** Fixes common spelling mistakes (e.g., `govner` → `Governor`, `prime minster` → `Prime Minister`, `presidant` → `President`), normalizes abbreviations (`wb` → `West Bengal`, `jk` → `Jammu and Kashmir`, `up` → `Uttar Pradesh`), and cleans grammatical artifacts.
2. **Intent Classification:** Detects whether the query is seeking a person (`who`), a numerical count (`how many districts`), geographical capital, role clarification, definition, or translation.
3. **Entity Detection:** Identifies Countries (India, USA, UK, etc.), Indian States & Union Territories (Kerala, Tamil Nadu, Delhi, Ladakh, etc.), and corporate organizations (NextGenPro).
4. **Role Extraction:** Identifies administrative positions (`Governor`, `Chief Minister`, `Prime Minister`, `President`, `Vice President`, `Mayor`, `MP`, `MLA`, `CEO`, `CTO`, `COO`, `CGO`).
5. **Entity + Role Validation Matrix:** Cross-checks if a requested role is legally valid for the specified entity type:
   - A country cannot have a *Governor* or *Chief Minister*.
   - A state cannot have a *Prime Minister* or *President*.
6. **Automatic Concept Correction:** If an invalid combination is detected (e.g., *"Who is the Governor of India?"*), IVIA immediately returns a polite, authoritative clarification without hallucinating or making unnecessary LLM calls.
7. **Temporal / Date Disambiguation:** Isolates specific historical years (e.g., `2023`, `2026`) from general current-term questions.

---

## 🗣️ Malayalam ↔ English Translation Engine

Implemented in `translator.py`, IVIA delivers seamless native Malayalam interaction:

- **Script Range Detection:** Scans text against Unicode block `[\u0D00-\u0D7F]`.
- **Term Preservation Masking:** Prior to translation, critical tokens are extracted and replaced with unique non-translatable placeholders (`XYZPH0XYZ`):
  - Source citations (`Source: ...`)
  - Technical terms & acronyms (`RAG`, `PDF`, `LLM`, `API`, `ChromaDB`, `FAISS`, `GPT`, etc.)
  - URLs (`https://...`) and email addresses
  - Markdown code blocks and inline code
  - Numbers, dates, percentages, and currencies
- **Translation Pipeline:**
  1. Input query (Malayalam) → Masked → Translated to English → Placeholders Restored.
  2. English query processed through the 5-layer search pipeline.
  3. Generated English response → Masked → Translated to Malayalam → Placeholders Restored.
- **Direct Translation Intent:** Automatically recognizes translation commands (e.g., *"Translate 'Good Morning' to Malayalam"*) and fulfills them directly.

---

## 🗄️ Knowledge Base, Datasets & Auto-Ingestion

IVIA comes bundled with pre-indexed domain datasets located in the `data/` folder:

| File | Type | Description |
|---|---|---|
| `data/Kerala_Current_CM_and_Ministers_2026_QA.pdf` | PDF Document | Official cabinet ministers and portfolio assignments for Kerala (2026). |
| `data/NextGenPro_RAG_Knowledge_Base.pdf` | PDF Document | Corporate profile, departments, education park, and tech services. |
| `data/NextGenPro_RAG_Question_Answer_Knowledge_Base.pdf` | PDF Document | Detailed QA document for NextGenPro operations and staff roles. |
| `data/Q&A.pdf` | PDF Document | Supplementary question-and-answer reference document. |
| `data/Q&A2.pdf` | PDF Document | Additional multi-topic QA reference document. |
| `data/qa_knowledge.json` | JSON Dataset | Structured general knowledge, Indian political entities, state districts, and roles with aliases. |
| `data/ivia_greetings_rag.json` | JSON Dataset | Conversational dataset for greetings, bot identity (IVIA), introductions, small talk, and farewells. |
| `data/nextgenpro_rag.json` | JSON Dataset | Comprehensive structured question-variation-answer pairs for NextGenPro leadership (CEO, CTO, COO, CGOs) and staff. |

### 🔄 Zero-Downtime Hot Drop Auto-Ingestion
Whenever a new `.pdf` is placed into the `data/` directory:
- `rag_system.auto_ingest_data_folder()` automatically discovers unindexed PDFs on startup or upon receiving incoming API calls.
- The new document is chunked, embedded, and appended to the active vector database in real-time.

---

## 📁 Project Directory Structure

```text
bot-roshni/
├── data/                                                   # Knowledge source documents & JSON datasets
│   ├── Kerala_Current_CM_and_Ministers_2026_QA.pdf         # Kerala Ministers PDF
│   ├── NextGenPro_RAG_Knowledge_Base.pdf                   # NextGenPro Overview PDF
│   ├── NextGenPro_RAG_Question_Answer_Knowledge_Base.pdf   # NextGenPro QA PDF
│   ├── Q&A.pdf                                             # Supplemental QA PDF
│   ├── Q&A2.pdf                                            # Extended QA PDF
│   ├── ivia_greetings_rag.json                             # IVIA bot greetings & small talk intents
│   ├── nextgenpro_rag.json                                 # NextGenPro structured QA knowledge base
│   └── qa_knowledge.json                                   # Core entity & administrative QA dataset
├── db/                                                     # Persisted ChromaDB vector database files
├── faiss_db/                                               # Persisted FAISS vector index files (optional)
├── main.py                                                 # FastAPI REST API server & endpoint definitions
├── rag.py                                                  # 5-Layer Search Orchestrator (RAGSystem)
├── chatbot.py                                              # LLM manager (Ollama/HF/OpenAI), prompts & layer evaluators
├── google_search.py                                        # Layer 3: Google Search URL collector + DDG fallback
├── web_scraper.py                                          # Layer 4: Clean HTML web scraping & text sanitizer
├── tavily_search.py                                        # Layer 5: Tavily Live Web Search API client & ping check
├── vector_store.py                                         # ChromaDB & FAISS vector store abstraction layer
├── embeddings.py                                           # HuggingFace sentence-transformers embedding loader
├── pdf_loader.py                                           # Multi-engine PDF text extractor (pdfplumber/PyMuPDF/PyPDF)
├── question_processor.py                                   # 7-Step question normalizer & role validation engine
├── qa_knowledge.py                                         # Q&A Knowledge Base semantic search & alias engine
├── translator.py                                           # Malayalam <-> English translation & term masking layer
├── create_db.py                                            # Standalone CLI tool to pre-index PDFs into vector store
├── test_upload.py                                          # Automated test script for PDF upload endpoint
├── make_pdf.py                                             # Utility script for sample PDF generation
├── .env                                                    # Environment variables & runtime configuration
└── requirements.txt                                        # Python dependencies
```

---

## 🛠️ Prerequisites & Installation

### 1. System Requirements
- **Operating System:** Windows 10/11, macOS, or Linux
- **Python:** Version `3.11+` recommended
- **RAM:** Minimum 8 GB (16 GB recommended for local LLM inference)
- **Ollama:** (Recommended for local LLM) [Download Ollama](https://ollama.ai/)

### 2. Setup Guide

#### Step 1: Clone or Open Project Directory
```bash
cd bot-roshni
```

#### Step 2: Create and Activate Virtual Environment
**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 4: Pull Ollama Local Model
If using the default local LLM backend:
```bash
ollama pull llama3.2
```

#### Step 5: Initialize Vector Database
Pre-index all PDFs located in `data/`:
```bash
python create_db.py
```

---

## ⚙️ Environment Configuration (.env)

Create a `.env` file in the root directory (or edit the existing one):

```env
# ── LLM Configuration ────────────────────────────────────────
# Backend choices: "ollama", "huggingface", or "openai"
DEFAULT_LLM_BACKEND=ollama

# Model identifiers
DEFAULT_OLLAMA_MODEL=llama3.2
DEFAULT_HF_MODEL=google/flan-t5-base
DEFAULT_OPENAI_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_openai_api_key_here

# ── Embedding Model & Vector Store ───────────────────────────
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_STORE_TYPE=chroma          # "chroma" or "faiss"
CHROMA_DB_DIR=db
FAISS_INDEX_DIR=faiss_db

# ── RAG Chunking & Retrieval Parameters ──────────────────────
RAG_TOP_K=6
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RAG_MAX_CONTEXT_CHARS=5000

# ── Q&A Knowledge Base Settings ──────────────────────────────
QA_KNOWLEDGE_FILE=data/qa_knowledge.json
QA_SIMILARITY_THRESHOLD=0.80
QA_TOP_K=3

# ── Search Mode & Confidence Thresholds ──────────────────────
# Search mode: "auto" (default, uses online if needed), "offline", or "online"
DEFAULT_SEARCH_MODE=auto

THRESHOLD_PDF=0.80
THRESHOLD_OLLAMA=0.85
THRESHOLD_SCRAPING=0.75
THRESHOLD_TAVILY=0.70

# ── Tavily Live Web Search API (Layer 5) ─────────────────────
TAVILY_API_KEY=your_tavily_api_key_here
```

### 📋 Configuration Parameters Reference

| Variable | Type | Default | Description |
|---|---|---|---|
| `DEFAULT_LLM_BACKEND` | `str` | `ollama` | Active LLM engine (`ollama`, `huggingface`, `openai`). |
| `DEFAULT_OLLAMA_MODEL`| `str` | `llama3.2`| Ollama model tag. |
| `DEFAULT_HF_MODEL`    | `str` | `google/flan-t5-base` | HuggingFace pipeline model name. |
| `DEFAULT_OPENAI_MODEL`| `str` | `gpt-4o-mini` | OpenAI model identifier. |
| `OPENAI_API_KEY`      | `str` | `""` | API key required when using OpenAI backend. |
| `EMBEDDING_MODEL`     | `str` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for semantic vectorization. |
| `VECTOR_STORE_TYPE`   | `str` | `chroma` | Vector database engine (`chroma` or `faiss`). |
| `CHROMA_DB_DIR`       | `str` | `db` | Storage folder for ChromaDB vectors. |
| `FAISS_INDEX_DIR`     | `str` | `faiss_db` | Storage folder for FAISS index files. |
| `RAG_TOP_K`           | `int` | `6` | Number of relevant chunks retrieved per query. |
| `CHUNK_SIZE`          | `int` | `1000` | Target character size per chunk. |
| `CHUNK_OVERLAP`       | `int` | `200` | Overlap characters between adjacent chunks. |
| `RAG_MAX_CONTEXT_CHARS` | `int` | `5000` | Max characters fed to LLM context window. |
| `QA_KNOWLEDGE_FILE`   | `str` | `data/qa_knowledge.json` | Path to JSON Q&A knowledge base. |
| `QA_SIMILARITY_THRESHOLD` | `float` | `0.80` | Minimum cosine similarity score for JSON Q&A match. |
| `QA_TOP_K`            | `int` | `3` | Maximum entries returned from Q&A search. |
| `DEFAULT_SEARCH_MODE` | `str` | `auto` | Global mode: `auto` (adaptive), `offline`, or `online`. |
| `THRESHOLD_PDF`       | `float` | `0.80` | Layer 1 acceptance threshold. |
| `THRESHOLD_OLLAMA`    | `float` | `0.85` | Layer 2 acceptance threshold. |
| `THRESHOLD_SCRAPING`  | `float` | `0.75` | Layer 4 acceptance threshold. |
| `THRESHOLD_TAVILY`    | `float` | `0.70` | Layer 5 acceptance threshold. |
| `TAVILY_API_KEY`      | `str` | `""` | API key for Tavily live web search. |

---

## 🚀 Running the Application

Start the FastAPI backend with Uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 🌐 Server URLs
- **API Base:** `http://localhost:8000`
- **Interactive Swagger UI:** `http://localhost:8000/docs`
- **ReDoc Documentation:** `http://localhost:8000/redoc`
- **Health Check:** `http://localhost:8000/health`

---

## 📡 API Reference & Endpoints

### POST `/chat`
Submits a user query through the 5-layer search orchestrator and returns a structured response.

#### Request Body (`application/json`)
```json
{
  "query": "Who is the Chief Minister of Kerala?",
  "mode": "auto",
  "session_id": "session_123"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | `string` | **Yes** | User question (English or Malayalam). |
| `mode` | `string` | No | `"auto"` (default), `"offline"`, or `"online"`. |
| `session_id` | `string` | No | Optional tracking identifier for client sessions. |

#### Response Body (`200 OK`)
```json
{
  "answer": "The current Chief Minister of Kerala is V. D. Satheesan.",
  "source_type": "pdf",
  "mode": "offline",
  "confidence": 0.95,
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

#### `source_type` Values
- `"pdf"`: Resolved from indexed PDF vectors or JSON Q&A Knowledge Base (Layer 1).
- `"ollama"`: Resolved from LLM general knowledge / reasoning (Layer 2).
- `"web_scraping"`: Resolved from live scraped web pages via Google Search (Layer 4).
- `"tavily"`: Resolved from Tavily Live Web Search API (Layer 5).
- `"none"`: No verified answer found across any layer.

#### cURL Example
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"query": "Who is the CTO of NextGenPro?", "mode": "auto"}'
```

---

### POST `/upload-pdf`
Uploads and indexes a new PDF document into the active vector database in real-time.

#### Request (`multipart/form-data`)
- `file`: The `.pdf` file binary.

#### Response Body (`200 OK`)
```json
{
  "message": "PDF uploaded and indexed successfully into vector store.",
  "filename": "annual_report.pdf",
  "num_chunks": 24
}
```

#### cURL Example
```bash
curl -X POST "http://localhost:8000/upload-pdf" \
     -F "file=@/path/to/document.pdf"
```

---

### GET `/health`
Returns operational health, active LLM backend, search mode, Tavily configuration status, and all currently indexed document sources.

#### Response Body (`200 OK`)
```json
{
  "status": "healthy",
  "search_mode": "auto",
  "tavily_status": "configured",
  "llm_backend": "ollama",
  "ollama_model": "llama3.2",
  "vector_store": "chroma",
  "indexed_sources": [
    "Kerala_Current_CM_and_Ministers_2026_QA.pdf",
    "NextGenPro_RAG_Knowledge_Base.pdf",
    "NextGenPro_RAG_Question_Answer_Knowledge_Base.pdf",
    "Q&A.pdf",
    "Q&A2.pdf"
  ]
}
```

---

### GET `/`
Automatically redirects client browsers to the interactive Swagger UI at `/docs`.

---

## 📱 Client Integration Examples

### Flutter / Dart

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

Future<Map<String, dynamic>> sendChatMessage(String query, {String? sessionId}) async {
  final url = Uri.parse('http://YOUR_SERVER_IP:8000/chat');
  final response = await http.post(
    url,
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({
      'query': query,
      'mode': 'auto',
      'session_id': sessionId,
    }),
  );

  if (response.statusCode == 200) {
    return jsonDecode(utf8.decode(response.bodyBytes));
  } else {
    throw Exception('Failed to get answer (${response.statusCode}): ${response.body}');
  }
}
```

### JavaScript / TypeScript (Fetch / React)

```typescript
interface ChatResponse {
  answer: string;
  source_type: 'pdf' | 'ollama' | 'web_scraping' | 'tavily' | 'none';
  mode: 'offline' | 'online';
  confidence: number;
  sources: Array<{ title: string; source: string; url?: string | null; page?: number | null }>;
}

async function askBot(query: string, mode: string = 'auto'): Promise<ChatResponse> {
  const res = await fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, mode }),
  });
  
  if (!res.ok) {
    throw new Error(`API error: ${res.status} - ${await res.text()}`);
  }
  return await res.json();
}

// Example usage:
askBot('Who is the Founder and CEO of NextGenPro?').then((data) => {
  console.log('Answer:', data.answer);
  console.log('Source:', data.source_type);
});
```

### Python Client (Requests & HTTPX Async)

**Synchronous (`requests`):**
```python
import requests

def chat_with_ivia(query: str, mode: str = "auto"):
    response = requests.post(
        "http://localhost:8000/chat",
        json={"query": query, "mode": mode}
    )
    response.raise_for_status()
    return response.json()

# Test Malayalam query
res = chat_with_ivia("കേരളത്തിലെ മുഖ്യമന്ത്രി ആരാണ്?")
print(res["answer"])
```

**Asynchronous (`httpx`):**
```python
import httpx
import asyncio

async def main():
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        resp = await client.post("/chat", json={"query": "Who is the CEO of NextGenPro?"})
        print(resp.json())

asyncio.run(main())
```

### cURL

```bash
# General query
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"query": "Who is the Chief Minister of Kerala?", "mode": "auto"}'

# Malayalam query
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"query": "കേരളത്തിലെ മുഖ്യമന്ത്രി ആരാണ്?"}'
```

---

## 🔧 CLI Utilities & Management Scripts

### 1. Pre-indexing PDFs (`create_db.py`)
Builds or updates vector databases directly from the command line:

```bash
# Default: Ingests all PDFs in data/ into ChromaDB
python create_db.py

# Custom directory and FAISS vector store
python create_db.py --input ./my_documents --store faiss

# Custom chunking configuration
python create_db.py --chunk-size 500 --chunk-overlap 100 --store chroma
```

| Flag | Short | Default | Description |
|---|---|---|---|
| `--input` | `-i` | `data` | Directory containing source PDF documents. |
| `--store` | `-s` | `chroma` | Vector store backend (`chroma` or `faiss`). |
| `--output` | `-o` | Auto | Custom output folder for vector persistence. |
| `--chunk-size` | | `1000` | Character count per document chunk. |
| `--chunk-overlap` | | `200` | Overlap character count between consecutive chunks. |

### 2. Automated Upload Testing (`test_upload.py`)
Runs an end-to-end test against the running FastAPI server to verify file validation and indexing:
```bash
python test_upload.py
```

### 3. Test PDF Generator (`make_pdf.py`)
Utility to quickly generate blank/sample PDF files for upload pipeline validation:
```bash
python make_pdf.py
```

---

## 🌐 LLM & Vector Store Backends

### LLM Backends

| Backend | Mode | Pros | Setup |
|---|---|---|---|
| **Ollama** *(Default)* | Local | Private, fast, zero API cost, high reasoning performance | Install Ollama and run `ollama pull llama3.2` |
| **HuggingFace** | Local | Runs standalone on CPU/GPU without external services; automatic fallback | Pre-installed via `transformers` (`google/flan-t5-base`) |
| **OpenAI** | Cloud | Highest accuracy and multilingual nuances | Set `DEFAULT_LLM_BACKEND=openai` and `OPENAI_API_KEY` |

### Vector Database Comparison

| Feature | ChromaDB (`chroma`) | FAISS (`faiss`) |
|---|---|---|
| **Persistence** | Full disk persistence with SQLite/Parquet metadata | In-memory with binary index save/load |
| **Metadata Filtering** | Built-in native metadata filtering | In-memory metadata map |
| **Best For** | Production deployments, dynamic uploads | High-speed prototyping, read-only indexes |

---

## ❓ Troubleshooting & FAQ

### 1. Ollama connection error (`503 Service Unavailable`)
- **Cause:** The Ollama service is not running locally or the model is missing.
- **Fix:** Start Ollama (`ollama serve`) and ensure the model is pulled (`ollama pull llama3.2`). The backend will automatically attempt a fallback to HuggingFace (`google/flan-t5-base`) if Ollama is unreachable.

### 2. PDF upload fails with `Could not extract readable text`
- **Cause:** The PDF contains scanned image pages without OCR digital text layers.
- **Fix:** Run the PDF through an OCR pre-processor or ensure the PDF contains digital selectable text.

### 3. Google Search throttling / rate limit
- **Behavior:** `google_search.py` automatically detects Google blocks and falls back to DuckDuckGo HTML search.
- **Recommendation:** Ensure `TAVILY_API_KEY` is configured in `.env` for seamless Layer 5 fallback.

### 4. Memory issues on low-RAM machines
- Set `DEFAULT_LLM_BACKEND=huggingface` and `DEFAULT_HF_MODEL=google/flan-t5-base` or use a smaller quantized model in Ollama (`llama3.2:1b`).

### 5. Windows Terminal Character Display
- When running in Windows PowerShell, set the output encoding to UTF-8 for proper Malayalam script and Unicode character rendering:
```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

---

## 📜 License

This project is licensed under the **MIT License**.
