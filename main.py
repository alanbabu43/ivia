"""
main.py — FastAPI Backend for RAG PDF Chatbot
=============================================
Exposes the RAG + Ollama PDF chatbot pipeline via a REST API for mobile (Flutter)
and external web clients.

Endpoints:
  - GET  /health     : Health check & service status.
  - POST /upload-pdf : Ingests PDF file, chunks & embeds into vector store.
  - POST /chat       : Receives query & optional session_id, returns RAG / Ollama answer.
"""

import io
import os
import tempfile
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from rag import RAGSystem

# Initialize FastAPI application
app = FastAPI(
    title="BOR RAG Chatbot API",
    description="FastAPI backend wrapping RAG + Ollama PDF ingestion and chat pipeline",
    version="1.0.0"
)

# Enable CORS for Flutter mobile/web clients on local network
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system singleton
print("[main] Initializing RAG System...")
rag_system = RAGSystem()

# Configure LLM backend from environment with fallback support
DEFAULT_LLM_BACKEND = os.getenv("DEFAULT_LLM_BACKEND", "ollama")
OLLAMA_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "llama3.2")
HF_MODEL = os.getenv("DEFAULT_HF_MODEL", "google/flan-t5-base")

def init_llm():
    try:
        print(f"[main] Setting LLM backend to '{DEFAULT_LLM_BACKEND}'...")
        if DEFAULT_LLM_BACKEND == "huggingface":
            rag_system.set_llm(backend="huggingface", model_name=HF_MODEL)
        else:
            rag_system.set_llm(backend="ollama", model_name=OLLAMA_MODEL)
    except Exception as e:
        print(f"[main] Warning: Failed to load '{DEFAULT_LLM_BACKEND}' backend ({e}).")
        if DEFAULT_LLM_BACKEND == "ollama":
            print(f"[main] Attempting fallback to HuggingFace model ('{HF_MODEL}')...")
            try:
                rag_system.set_llm(backend="huggingface", model_name=HF_MODEL)
            except Exception as hf_e:
                print(f"[main] Warning: HuggingFace fallback failed: {hf_e}")

init_llm()


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Request & Response Models
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Who is the Chief Minister of Kerala?",
                "session_id": "session_123"
                # mode is intentionally omitted — system auto-detects based on internet connectivity
            }
        }
    }

    query: str = Field(..., description="User query / question")
    mode: Optional[str] = Field(
        None,
        description=(
            "Search mode: 'offline' (Local PDF RAG), 'online' (Tavily Live Web Search), "
            "or omit / 'auto' to auto-detect based on internet connectivity."
        ),
    )
    session_id: Optional[str] = Field(None, description="Optional session identifier for chat context tracking")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Generated answer from search orchestrator")
    source_type: str = Field("none", description="Source layer: 'pdf', 'ollama', 'web_scraping', 'tavily', or 'none'")
    mode: str = Field("offline", description="Operational search mode actually used ('offline' or 'online')")
    confidence: float = Field(0.0, description="Confidence score of accepted answer (0.0 to 1.0)")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="List of source document chunks or web search results")


class UploadResponse(BaseModel):
    message: str = Field(..., description="Success message")
    filename: str = Field(..., description="Uploaded PDF filename")
    num_chunks: int = Field(..., description="Number of text chunks embedded into vector store")


class HealthResponse(BaseModel):
    status: str
    search_mode: str
    tavily_status: str
    llm_backend: str
    ollama_model: str
    vector_store: str
    indexed_sources: List[str]


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root path to interactive API documentation (/docs)."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint returning system status, search mode, and indexed sources.
    """
    sources = []
    if hasattr(rag_system, "get_indexed_sources"):
        sources = rag_system.get_indexed_sources()
    elif rag_system.db:
        # Fallback source list check
        try:
            get_data = rag_system.db.get()
            if get_data and "metadatas" in get_data and get_data["metadatas"]:
                sources = list({m.get("source") for m in get_data["metadatas"] if m and "source" in m})
        except Exception:
            sources = []

    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    tavily_status = "configured" if tavily_key and tavily_key != "your_tavily_api_key_here" else "missing"

    return HealthResponse(
        status="healthy",
        search_mode=os.getenv("DEFAULT_SEARCH_MODE", "auto"),
        tavily_status=tavily_status,
        llm_backend=os.getenv("DEFAULT_LLM_BACKEND", "ollama"),
        ollama_model=OLLAMA_MODEL,
        vector_store=rag_system.vector_store_type,
        indexed_sources=sources
    )


@app.post("/upload-pdf", response_model=UploadResponse, tags=["PDF Ingestion"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF document to run through chunking, embedding, and vector store ingestion.
    """
    # 1. Validate file extension and content type
    filename = file.filename or "uploaded.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type for '{filename}'. Only PDF files (.pdf) are accepted."
        )

    try:
        content = await file.read()
        if not content or len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )

        # 2. Pass bytes stream to RAG ingestion pipeline
        pdf_stream = io.BytesIO(content)
        num_chunks = rag_system.ingest_pdf_rag(pdf_stream, filename=filename)

        if num_chunks == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not extract readable text from the provided PDF file."
            )

        return UploadResponse(
            message="PDF uploaded and indexed successfully into vector store.",
            filename=filename,
            num_chunks=num_chunks
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[main] Error during PDF ingestion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process PDF document: {str(e)}"
        )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Process a user query using the 5-Layer Intelligent RAG Search Orchestrator.
    """
    # 1. Validate empty or whitespace query
    query = request.query.strip() if request.query else ""
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty."
        )

    # 2. Ensure LLM is loaded
    if rag_system.llm is None:
        init_llm()
        if rag_system.llm is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM service is not available. Please ensure Ollama model 'llama3.2' is pulled (`ollama pull llama3.2`) or use HuggingFace backend."
            )

    # 3. Auto-ingest any new PDFs in data/ folder if present
    rag_system.auto_ingest_data_folder()

    # 4. Perform query execution via 5-Layer Search Orchestrator
    try:
        result = rag_system.orchestrate_search(query, mode=request.mode)
        if isinstance(result, dict):
            answer_text = result.get("answer", "")
            source_type = result.get("source_type", "none")
            used_mode = result.get("mode", "offline")
            confidence = float(result.get("confidence", 0.0))
            raw_sources = result.get("sources", [])
            formatted_sources = []
            for s in raw_sources:
                if isinstance(s, dict):
                    formatted_sources.append(s)
                elif hasattr(s, "metadata"):
                    formatted_sources.append(s.metadata)
                else:
                    formatted_sources.append({"source": str(s)})
        else:
            answer_text = str(result)
            source_type = "none"
            used_mode = request.mode or "offline"
            confidence = 0.0
            formatted_sources = []

        return ChatResponse(
            answer=answer_text,
            source_type=source_type,
            mode=used_mode,
            confidence=confidence,
            sources=formatted_sources
        )

    except Exception as e:
        err_msg = str(e).lower()
        if "connection" in err_msg or "unreachable" in err_msg or "refused" in err_msg:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ollama server is unreachable at http://localhost:11434. Please start the Ollama service."
            )
        print(f"[main] Error during chat processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating answer: {str(e)}"
        )
