"""
rag.py — RAG System Orchestrator
==================================
The central coordinator for the entire PDF chatbot system.

Handles:
  - Method 1: Direct PDF text extraction → LLM QA (no vector DB)
  - Method 2: Full RAG pipeline (PDF → chunks → embeddings → vector DB → retrieval → LLM)

Both ChromaDB and FAISS vector stores are supported (configured via .env).

Usage:
    rag = RAGSystem()
    rag.set_llm(backend="huggingface", model_name="google/flan-t5-base")

    # Method 1 - Direct
    text = rag.ingest_pdf_direct(pdf_file, filename)
    answer = rag.query_direct(text, "What is the main topic?")

    # Method 2 - RAG
    chunks = rag.ingest_pdf_rag(pdf_file, filename)
    result = rag.query_rag("What is the main topic?")
"""

import glob
import io
import os
import re
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv

from pdf_loader import extract_text_direct, extract_documents_from_pdf, split_documents
from embeddings import get_embedding_model
from vector_store import (
    create_vector_store, load_vector_store, add_documents_to_store, clear_vector_store,
    CHROMA_DB_DIR, FAISS_INDEX_DIR
)
from chatbot import (
    get_llm, get_qa_chain, answer_from_text, answer_pretrained,
    _clean_llm_response, RAG_PROMPT_TEMPLATE, SYSTEM_PROMPT, WEB_SEARCH_PROMPT,
    format_error_message, invoke_llm_with_oom_retry,
    evaluate_pdf_layer, evaluate_ollama_layer, evaluate_scraping_layer, evaluate_tavily_layer
)
from translator import (
    is_malayalam, translate_text, detect_language,
    is_translation_request, translate_direct_request
)
from qa_knowledge import QAKnowledgeBase
from question_processor import analyze_question
from tavily_search import TavilySearcher, is_online
from google_search import GoogleSearcher
from web_scraper import WebScraper

load_dotenv()

# Confidence thresholds for each layer
THRESHOLD_PDF = float(os.getenv("THRESHOLD_PDF", "0.80"))
THRESHOLD_OLLAMA = float(os.getenv("THRESHOLD_OLLAMA", "0.85"))
THRESHOLD_SCRAPING = float(os.getenv("THRESHOLD_SCRAPING", "0.75"))
THRESHOLD_TAVILY = float(os.getenv("THRESHOLD_TAVILY", "0.70"))


class RAGSystem:
    """
    Orchestrates the complete PDF chatbot and 5-layer search fallback workflow.

    Supports two QA modes:
      - Method 1: extract_text_direct() → answer_from_text() (no vector DB)
      - Method 2: ingest_pdf_rag() → query_rag() (5-layer sequential orchestrator)

    The LLM and vector store backends are fully configurable at runtime.
    """

    def __init__(
        self,
        db_dir: str = None,
        embedding_model_name: str = None,
        vector_store_type: str = None
    ):
        """
        Initializes the RAG System.

        Loads the embedding model and attempts to reload any existing vector
        database from disk for persistence across sessions.

        Args:
            db_dir: Override the persistence directory for the vector store.
                    Defaults to CHROMA_DB_DIR or FAISS_INDEX_DIR from env.
            embedding_model_name: Override embedding model. Defaults to env/MiniLM.
            vector_store_type: "chroma" or "faiss". Defaults to env or "chroma".
        """
        self.vector_store_type = vector_store_type or os.getenv("VECTOR_STORE_TYPE", "chroma")
        self.db_dir = db_dir or (
            FAISS_INDEX_DIR if self.vector_store_type == "faiss" else CHROMA_DB_DIR
        )
        self.embedding_model_name = embedding_model_name
        self.backend_name = os.getenv("DEFAULT_LLM_BACKEND", "ollama")
        self.model_name = os.getenv("DEFAULT_OLLAMA_MODEL", "llama3.2")

        # Initialize embedding model (shared between both methods)
        print("[rag] Initializing embedding model...")
        self.embeddings = get_embedding_model(self.embedding_model_name)

        # Internal state
        self.db = None           # Vector store instance (Chroma or FAISS)
        self.retriever = None    # LangChain retriever
        self.qa_chain = None     # RetrievalQA chain for Method 2
        self.llm = None          # Active LLM instance (shared across both methods)

        # Initialize Q&A Knowledge Base (shares the same embedding model)
        print("[rag] Initializing Q&A Knowledge Base...")
        self.qa_kb = QAKnowledgeBase(embeddings=self.embeddings)

        # Initialize search & scraper clients
        self.google_searcher = GoogleSearcher()
        self.web_scraper = WebScraper()
        self.tavily_searcher = TavilySearcher()

        # Try to restore existing vector database from disk
        self._load_existing_db()

        # Automatically ingest any new PDFs in data/ directory
        self.auto_ingest_data_folder()

    # ─────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _load_existing_db(self) -> bool:
        """
        Attempts to load an existing vector database from disk.

        Returns:
            True if a database was found and loaded, False otherwise.
        """
        self.db = load_vector_store(
            self.embeddings,
            store_type=self.vector_store_type,
            persist_directory=self.db_dir
        )
        if self.db:
            top_k = int(os.getenv("RAG_TOP_K", "8"))
            self.retriever = self.db.as_retriever(search_kwargs={"k": top_k})
            if self.llm:
                self.qa_chain = get_qa_chain(self.llm, self.retriever)
            print(f"[rag] Existing {self.vector_store_type.upper()} database restored.")
            return True
        return False

    def _refresh_retriever_and_chain(self) -> None:
        """Rebuilds retriever and QA chain after DB changes."""
        if self.db:
            # Always read RAG_TOP_K fresh so .env changes take effect without restart
            top_k = int(os.getenv("RAG_TOP_K", "8"))
            self.retriever = self.db.as_retriever(search_kwargs={"k": top_k})
            if self.llm:
                self.qa_chain = get_qa_chain(self.llm, self.retriever)

    def auto_ingest_data_folder(
        self,
        data_dir: str = "data",
        force_reingest: bool = False
    ) -> int:
        """
        Scans the specified data folder for PDF files and ingests any PDFs
        that are not yet indexed in the vector store.

        Args:
            data_dir: Folder to scan for PDFs (default: "data").
            force_reingest: If True, re-index ALL PDFs even if already stored.
                            Useful after a DB reset or if content changed.

        Returns:
            Total number of new chunks indexed across all new PDFs.
        """
        if not os.path.exists(data_dir):
            print(f"[rag] Data folder '{data_dir}' does not exist. Skipping auto-ingest.")
            return 0

        pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
        if not pdf_files:
            print(f"[rag] No PDF files found in '{data_dir}'.")
            return 0

        # Build set of already-indexed filenames from vector store
        indexed = set() if force_reingest else set(self.get_indexed_sources())
        total_new_chunks = 0
        skipped = 0
        failed = []

        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)

            if filename in indexed:
                skipped += 1
                print(f"[rag] Skipping already-indexed PDF: '{filename}'")
                continue

            print(f"[rag] Auto-ingesting new PDF from '{data_dir}': '{filename}'")
            try:
                num_chunks = self.ingest_pdf_rag(pdf_path, filename=filename)
                if num_chunks > 0:
                    total_new_chunks += num_chunks
                    indexed.add(filename)  # Only mark indexed if chunks were actually stored
                    print(f"[rag] Successfully indexed '{filename}' → {num_chunks} chunks.")
                else:
                    # PDF produced 0 chunks — likely scanned/image-only or empty.
                    # Do NOT add to indexed set so we retry on next request / server restart.
                    print(
                        f"[rag] WARNING: '{filename}' produced 0 chunks. "
                        "It may be a scanned/image-only PDF with no text layer. "
                        "Use an OCR tool to add a text layer, then retry."
                    )
                    failed.append(filename)
            except Exception as e:
                print(f"[rag] ERROR auto-ingesting '{filename}': {e}")
                failed.append(filename)

        print(
            f"[rag] Auto-ingest complete: {total_new_chunks} new chunks indexed, "
            f"{skipped} PDFs already indexed, {len(failed)} failed."
        )
        if failed:
            print(f"[rag] Failed PDFs (check logs above): {failed}")

        return total_new_chunks

    # ─────────────────────────────────────────────────────────────────────
    # LLM Configuration (shared by both methods)
    # ─────────────────────────────────────────────────────────────────────

    def set_llm(
        self,
        backend: str = None,
        model_name: str = None,
        temperature: float = 0.0
    ) -> None:
        """
        Loads the specified LLM backend and rebuilds the RAG chain if ready.

        Args:
            backend: "huggingface", "ollama", or "openai".
            model_name: Model identifier (depends on backend).
            temperature: Generation temperature (0.0 = deterministic).
        """
        self.backend_name = backend or os.getenv("DEFAULT_LLM_BACKEND", "ollama")
        self.model_name = model_name or (
            os.getenv("DEFAULT_OLLAMA_MODEL", "llama3.2")
            if self.backend_name == "ollama"
            else os.getenv("DEFAULT_HF_MODEL", "google/flan-t5-base")
        )
        self.llm = get_llm(backend=backend, model_name=model_name, temperature=temperature)

        # Rebuild QA chain if a vector database is already loaded
        if self.retriever:
            self.qa_chain = get_qa_chain(self.llm, self.retriever)
            print("[rag] QA chain rebuilt with new LLM.")

    # ─────────────────────────────────────────────────────────────────────
    # Method 1 — Direct PDF Extraction
    # ─────────────────────────────────────────────────────────────────────

    def ingest_pdf_direct(
        self,
        pdf_file: Union[str, io.BytesIO]
    ) -> Dict[str, Any]:
        """
        Extracts text from a PDF directly without building a vector index.

        Designed for Method 1 (direct QA). The extracted text is returned
        and should be passed to query_direct() for question answering.

        Args:
            pdf_file: File path or BytesIO stream of the PDF.

        Returns:
            Dictionary with:
              - "text" (str): Full cleaned extracted text.
              - "page_count" (int): Number of pages in the PDF.
              - "char_count" (int): Total character count of extracted text.

        Raises:
            RuntimeError: If text extraction fails for all available libraries.
        """
        print("[rag] Extracting PDF text directly (Method 1)...")
        text, page_count = extract_text_direct(pdf_file)
        return {
            "text": text,
            "page_count": page_count,
            "char_count": len(text)
        }

    def query_direct(self, document_text: str, question: str) -> str:
        """
        Answers a question using directly extracted PDF text (Method 1).

        Supports automatic Malayalam ↔ English translation.
        Checks Q&A Knowledge Base first before invoking LLM.
        """
        if not self.llm:
            return (
                "No LLM loaded yet. Please configure a model in the settings panel "
                "or wait for the default model to finish loading."
            )
        if not document_text or not document_text.strip():
            return "No document text available. Please upload and extract a PDF first."

        # Detect Malayalam and translate question to English if needed
        is_ml = is_malayalam(question)
        search_question = (
            translate_text(question, target_lang="en", source_lang="ml")
            if is_ml else question
        )

        # 0. Question analysis: normalize, detect entity+role, validate combination
        analysis = analyze_question(search_question)
        if not analysis["is_valid_combination"] and analysis["correction_response"]:
            correction = analysis["correction_response"]
            print(f"[rag] Invalid entity-role combination detected. Returning correction.")
            if is_ml:
                return translate_text(correction, target_lang="ml", source_lang="en", llm=self.llm)
            return correction
        # Use normalized question for all subsequent lookups
        search_question = analysis["normalized_question"] if analysis["normalized_question"] else search_question

        # 1. Check Q&A Knowledge Base for direct match first (bypasses LLM call)
        direct_qa = self.qa_kb.get_direct_match(search_question)
        if direct_qa:
            direct_ans = direct_qa["answer"]
            if is_ml:
                return translate_text(direct_ans, target_lang="ml", source_lang="en", llm=self.llm)
            return direct_ans

        # 2. Search Q&A Knowledge Base for context string
        qa_context = self.qa_kb.format_context(search_question)

        # 3. Process direct text QA in English with Q&A KB context & OOM retry
        raw_answer = answer_from_text(
            self.llm, document_text, search_question, qa_context=qa_context
        )

        # Translate answer back to Malayalam if input was Malayalam
        if is_ml:
            return translate_text(raw_answer, target_lang="ml", source_lang="en", llm=self.llm)
        return raw_answer

    # ─────────────────────────────────────────────────────────────────────
    # Pre-trained — General Knowledge QA
    # ─────────────────────────────────────────────────────────────────────

    def query_pretrained(self, question: str) -> str:
        """
        Answers a question using Q&A KB + LLM's pre-trained knowledge.

        Checks Q&A KB first for direct answers.
        """
        if not self.llm:
            return (
                "No LLM loaded yet. Please configure a model in the settings panel "
                "or wait for the default model to finish loading."
            )

        is_ml = is_malayalam(question)
        search_question = (
            translate_text(question, target_lang="en", source_lang="ml")
            if is_ml else question
        )

        # 0. Question analysis: normalize, detect entity+role, validate combination
        analysis = analyze_question(search_question)
        if not analysis["is_valid_combination"] and analysis["correction_response"]:
            correction = analysis["correction_response"]
            print(f"[rag] Invalid entity-role combination detected. Returning correction.")
            if is_ml:
                return translate_text(correction, target_lang="ml", source_lang="en", llm=self.llm)
            return correction
        # Use normalized question for all subsequent lookups
        search_question = analysis["normalized_question"] if analysis["normalized_question"] else search_question

        # 1. Check Q&A Knowledge Base for direct match first (bypasses LLM call)
        direct_qa = self.qa_kb.get_direct_match(search_question)
        if direct_qa:
            direct_ans = direct_qa["answer"]
            if is_ml:
                return translate_text(direct_ans, target_lang="ml", source_lang="en", llm=self.llm)
            return direct_ans

        # 2. Search Q&A Knowledge Base for context string
        qa_context = self.qa_kb.format_context(search_question)

        raw_answer = answer_pretrained(self.llm, search_question, qa_context=qa_context)

        if is_ml:
            return translate_text(raw_answer, target_lang="ml", source_lang="en", llm=self.llm)
        return raw_answer

    # ─────────────────────────────────────────────────────────────────────
    # Method 2 — RAG Pipeline
    # ─────────────────────────────────────────────────────────────────────

    def ingest_pdf_rag(
        self,
        pdf_file: Union[str, io.BytesIO],
        filename: str,
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> int:
        """
        Full RAG ingestion pipeline for a PDF file (Method 2).

        Steps:
          1. Extract text from PDF (page by page)
          2. Split into overlapping chunks
          3. Generate embeddings for each chunk
          4. Store embeddings in vector database (Chroma or FAISS)

        Args:
            pdf_file: File path or BytesIO stream of the PDF.
            filename: Original filename for metadata labeling.
            chunk_size: Characters per chunk. Reads CHUNK_SIZE env var if None.
            chunk_overlap: Overlap characters. Reads CHUNK_OVERLAP env var if None.

        Returns:
            Number of chunks stored in the vector database.
        """
        chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "1000"))
        chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "200"))

        print(f"[rag] Ingesting '{filename}' into RAG pipeline (Method 2)...")

        # 1. Extract text as LangChain Documents (one per page)
        raw_docs = extract_documents_from_pdf(pdf_file, filename)
        if not raw_docs:
            print(f"[rag] No text extracted from '{filename}'.")
            return 0

        # 2. Split into overlapping chunks for better retrieval granularity
        split_docs = split_documents(raw_docs, chunk_size, chunk_overlap)

        # 3. Embed and store in vector database
        if self.db is None:
            # Create a new database if none exists
            self.db = create_vector_store(
                split_docs,
                self.embeddings,
                store_type=self.vector_store_type,
                persist_directory=self.db_dir
            )
        else:
            # Append to existing database (multi-document support)
            add_documents_to_store(
                self.db,
                split_docs,
                store_type=self.vector_store_type,
                index_dir=self.db_dir
            )

        # 4. Rebuild retriever and QA chain with updated database
        self._refresh_retriever_and_chain()

        print(f"[rag] Indexed {len(split_docs)} chunks from '{filename}' into {self.vector_store_type.upper()}.")
        return len(split_docs)

    # Keep backward-compatible alias for app.py calls
    def ingest_pdf(
        self,
        pdf_file: Union[str, io.BytesIO],
        filename: str,
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> int:
        """Alias for ingest_pdf_rag() for backward compatibility."""
        return self.ingest_pdf_rag(pdf_file, filename, chunk_size, chunk_overlap)

    def orchestrate_search(self, question: str, mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Intelligent RAG Search Orchestrator with 5-Layer Sequential Fallback Architecture
        and Strict Early Stop Rules.

        Pipeline Priority:
          Layer 1: Custom PDF Knowledge Base (PDF RAG + Vector DB + QA KB)
                   -> If reliable answer exists -> RETURN & STOP.
          Layer 2: Ollama Reasoning & Knowledge Check
                   -> If confident answer without web search -> RETURN & STOP.
                   -> If NEEDS_WEB_SEARCH / uncertain -> Proceed to Layer 3.
          Layer 3: Google Search
                   -> Search Google and extract relevant URLs.
                   -> If URLs found -> Proceed to Layer 4 (Web Scraping).
                   -> If no URLs found -> Skip Layer 4, Proceed to Layer 5 (Tavily).
          Layer 4: Web Scraping
                   -> Scrape readable content, strip ads/nav/scripts.
                   -> If reliable answer extracted -> RETURN & STOP.
                   -> If failed/insufficient -> Proceed to Layer 5 (Tavily).
          Layer 5: Tavily Search (Final Fallback)
                   -> If reliable answer found -> RETURN & STOP.
                   -> If all fail -> Return 'No reliable information found' message.
        """
        raw_mode = (mode or os.getenv("DEFAULT_SEARCH_MODE", "auto")).lower().strip()
        is_ml = is_malayalam(question)
        english_question = (
            translate_text(question, target_lang="en", source_lang="ml")
            if is_ml else question
        )

        # 0a. Translation request routing
        if is_translation_request(question):
            print(f"[rag] Translation request detected: '{question}'")
            translated_answer = translate_direct_request(question, self.llm)
            return {
                "answer": translated_answer,
                "source_type": "ollama",
                "mode": "offline",
                "confidence": 0.95,
                "sources": [],
                "is_translated": True,
                "original_question": question
            }

        # 0b. Entity-role validation
        analysis = analyze_question(english_question)
        if not analysis["is_valid_combination"] and analysis["correction_response"]:
            correction = analysis["correction_response"]
            print(f"[rag] Invalid entity-role combination detected. Returning correction.")
            if is_ml:
                correction = translate_text(correction, target_lang="ml", source_lang="en", llm=self.llm)
            return {
                "answer": correction,
                "source_type": "ollama",
                "mode": "offline",
                "confidence": 0.95,
                "sources": [],
                "is_translated": is_ml,
                "original_question": question
            }
        english_question = analysis["normalized_question"] if analysis["normalized_question"] else english_question

        # Ensure LLM is loaded
        if not self.llm:
            return {
                "answer": "LLM service is not available. Please ensure Ollama or HuggingFace backend is loaded.",
                "source_type": "none",
                "mode": "offline",
                "confidence": 0.0,
                "sources": []
            }

        # =========================================================================
        # LAYER 1: Custom PDF Knowledge Base (PDF RAG + Vector DB + QA KB)
        # =========================================================================
        if raw_mode != "online":
            print(f"[rag] [Layer 1] Checking Custom PDF Knowledge Base...")

            # 1a. Direct Q&A match
            direct_qa = self.qa_kb.get_direct_match(english_question)
            if direct_qa:
                print(f"[rag] [Layer 1] Exact/High-confidence Q&A KB match found! Stopping pipeline.")
                direct_ans = direct_qa["answer"]
                if is_ml:
                    direct_ans = translate_text(direct_ans, target_lang="ml", source_lang="en", llm=self.llm)
                return {
                    "answer": direct_ans,
                    "source_type": "pdf",
                    "mode": "offline",
                    "confidence": 0.95,
                    "sources": [{
                        "title": "Q&A Knowledge Base",
                        "source": os.path.basename(os.getenv("QA_KNOWLEDGE_FILE", "data/qa_knowledge.json")),
                        "url": None,
                        "page": None
                    }],
                    "is_translated": is_ml,
                    "original_question": question
                }

            # 1b. Vector DB retrieval
            qa_context = self.qa_kb.format_context(english_question)
            pdf_context = ""
            source_docs = []

            if self.retriever and self.db:
                try:
                    sem_docs = self.retriever.invoke(english_question)
                    query_words = [w.lower() for w in re.findall(r'\b[a-zA-Z0-9_-]{3,}\b', english_question)]
                    boosted_docs = []
                    other_docs = []
                    seen_contents = set()

                    for doc in sem_docs:
                        c = doc.page_content
                        if c in seen_contents:
                            continue
                        seen_contents.add(c)
                        c_lower = c.lower()
                        match_count = sum(1 for w in query_words if w in c_lower)
                        if match_count >= 1:
                            boosted_docs.append((match_count, doc))
                        else:
                            other_docs.append(doc)

                    boosted_docs.sort(key=lambda x: x[0], reverse=True)
                    ordered_docs = [d for _, d in boosted_docs] + other_docs
                    source_docs = ordered_docs

                    if source_docs:
                        max_context_chars = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "5000"))
                        context_parts = []
                        total_chars = 0
                        for doc in source_docs:
                            if total_chars + len(doc.page_content) > max_context_chars:
                                remaining = max_context_chars - total_chars
                                if remaining > 100:
                                    context_parts.append(doc.page_content[:remaining])
                                break
                            context_parts.append(doc.page_content)
                            total_chars += len(doc.page_content)
                        pdf_context = "\n\n".join(context_parts)
                except Exception as e:
                    print(f"[rag] [Layer 1] Vector retrieval warning: {e}")

            if pdf_context or qa_context:
                pdf_ans, is_accepted = evaluate_pdf_layer(
                    self.llm,
                    question=english_question,
                    pdf_context=pdf_context,
                    qa_context=qa_context,
                    backend=self.backend_name,
                    model_name=self.model_name
                )
                if is_accepted and pdf_ans:
                    print(f"[rag] [Layer 1] PDF answer verified and accepted! Stopping pipeline.")
                    if is_ml:
                        pdf_ans = translate_text(pdf_ans, target_lang="ml", source_lang="en", llm=self.llm)

                    formatted_sources = [
                        {
                            "title": doc.metadata.get("source", "PDF Document"),
                            "source": doc.metadata.get("source", "PDF Document"),
                            "url": None,
                            "page": doc.metadata.get("page", None)
                        }
                        for doc in source_docs
                    ]
                    if qa_context and not formatted_sources:
                        formatted_sources.append({
                            "title": "Q&A Knowledge Base",
                            "source": "qa_knowledge.json",
                            "url": None,
                            "page": None
                        })
                    return {
                        "answer": pdf_ans,
                        "source_type": "pdf",
                        "mode": "offline",
                        "confidence": THRESHOLD_PDF,
                        "sources": formatted_sources,
                        "is_translated": is_ml,
                        "original_question": question
                    }
                else:
                    print(f"[rag] [Layer 1] PDF context insufficient or NO_PDF_ANSWER. Proceeding to Layer 2.")
            else:
                print(f"[rag] [Layer 1] No PDF chunks retrieved. Proceeding to Layer 2.")

        # =========================================================================
        # LAYER 2: Ollama Reasoning and Knowledge Check
        # =========================================================================
        if raw_mode != "online":
            print(f"[rag] [Layer 2] Checking Ollama reasoning and knowledge...")
            qa_context = self.qa_kb.format_context(english_question) if hasattr(self, "qa_kb") else ""
            ollama_ans, is_confident = evaluate_ollama_layer(
                self.llm,
                question=english_question,
                partial_context=qa_context,
                backend=self.backend_name,
                model_name=self.model_name
            )

            if is_confident and ollama_ans:
                print(f"[rag] [Layer 2] Ollama provided confident answer! Stopping pipeline.")
                if is_ml:
                    ollama_ans = translate_text(ollama_ans, target_lang="ml", source_lang="en", llm=self.llm)
                return {
                    "answer": ollama_ans,
                    "source_type": "ollama",
                    "mode": "offline",
                    "confidence": THRESHOLD_OLLAMA,
                    "sources": [],
                    "is_translated": is_ml,
                    "original_question": question
                }
            else:
                print(f"[rag] [Layer 2] Ollama returned NEEDS_WEB_SEARCH or low confidence. Proceeding to Layer 3.")

            # If user explicitly specified offline mode, we must not go online
            if raw_mode == "offline":
                fallback = "I could not find enough reliable information in the offline knowledge base to answer this question accurately."
                if is_ml:
                    fallback = translate_text(fallback, target_lang="ml", source_lang="en", llm=self.llm)
                return {
                    "answer": fallback,
                    "source_type": "none",
                    "mode": "offline",
                    "confidence": 0.0,
                    "sources": [],
                    "is_translated": is_ml,
                    "original_question": question
                }

        # Check internet connectivity before online layers
        if not is_online():
            print(f"[rag] Internet connectivity unavailable. Cannot execute online layers.")
            fallback = "I could not find enough reliable information in the local knowledge base, and internet connection is currently unavailable."
            if is_ml:
                fallback = translate_text(fallback, target_lang="ml", source_lang="en", llm=self.llm)
            return {
                "answer": fallback,
                "source_type": "none",
                "mode": "offline",
                "confidence": 0.0,
                "sources": [],
                "is_translated": is_ml,
                "original_question": question
            }

        # =========================================================================
        # LAYER 3 & LAYER 4: Google Search -> Web Scraping
        # =========================================================================
        print(f"[rag] [Layer 3] Executing Google Search for '{english_question}'...")
        google_results = self.google_searcher.search(english_question, max_results=5)

        if google_results:
            print(f"[rag] [Layer 3] Google Search returned {len(google_results)} URLs. Proceeding to Layer 4 (Web Scraping)...")
            scraped_data = self.web_scraper.scrape_multiple(google_results, max_pages=3)
            scraped_context = scraped_data.get("formatted_context", "")
            scraped_docs = scraped_data.get("scraped_docs", [])

            if scraped_context:
                print(f"[rag] [Layer 4] Synthesizing answer from {len(scraped_docs)} scraped web pages...")
                scraped_res = evaluate_scraping_layer(
                    self.llm,
                    question=english_question,
                    scraped_context=scraped_context,
                    backend=self.backend_name,
                    model_name=self.model_name,
                    min_confidence=THRESHOLD_SCRAPING
                )
                scraped_ans = scraped_res[0]
                is_accepted = scraped_res[1]
                scraped_conf = scraped_res[2] if len(scraped_res) > 2 and scraped_res[2] else THRESHOLD_SCRAPING

                if is_accepted and scraped_ans:
                    print(f"[rag] [Layer 4] Web-scraped answer verified and accepted (confidence: {scraped_conf:.2f})! Stopping pipeline.")
                    if is_ml:
                        scraped_ans = translate_text(scraped_ans, target_lang="ml", source_lang="en", llm=self.llm)

                    formatted_sources = [
                        {
                            "title": doc.get("title", doc.get("domain", "Web Page")),
                            "source": doc.get("domain", "Web Source"),
                            "url": doc.get("url"),
                            "page": None
                        }
                        for doc in scraped_docs
                    ]
                    return {
                        "answer": scraped_ans,
                        "source_type": "web_scraping",
                        "mode": "online",
                        "confidence": scraped_conf,
                        "sources": formatted_sources,
                        "is_translated": is_ml,
                        "original_question": question
                    }
                else:
                    print(f"[rag] [Layer 4] Web-scraped content insufficient. Proceeding to Layer 5 (Tavily).")
            else:
                print(f"[rag] [Layer 4] Could not scrape readable content from URLs. Proceeding to Layer 5 (Tavily).")
        else:
            print(f"[rag] [Layer 3] Google search returned no useful URLs. Skipping Layer 4 and proceeding to Layer 5 (Tavily).")

        # =========================================================================
        # LAYER 5: Tavily Search (Final Fallback)
        # =========================================================================
        if self.tavily_searcher.is_configured():
            print(f"[rag] [Layer 5] Executing Tavily Live Web Search (Final Fallback)...")
            try:
                search_data = self.tavily_searcher.search(english_question)
                tavily_context = search_data.get("formatted_context", "")
                results_list = search_data.get("results", [])

                if tavily_context:
                    tavily_ans, is_accepted = evaluate_tavily_layer(
                        self.llm,
                        question=english_question,
                        tavily_context=tavily_context,
                        backend=self.backend_name,
                        model_name=self.model_name
                    )
                    if is_accepted and tavily_ans:
                        print(f"[rag] [Layer 5] Tavily answer verified and accepted! Stopping pipeline.")
                        if is_ml:
                            tavily_ans = translate_text(tavily_ans, target_lang="ml", source_lang="en", llm=self.llm)

                        formatted_sources = [
                            {
                                "title": r.get("title", "Web Result"),
                                "source": r.get("url", "Tavily Search"),
                                "url": r.get("url"),
                                "page": None
                            }
                            for r in results_list
                        ]
                        return {
                            "answer": tavily_ans,
                            "source_type": "tavily",
                            "mode": "online",
                            "confidence": THRESHOLD_TAVILY,
                            "sources": formatted_sources,
                            "is_translated": is_ml,
                            "original_question": question
                        }
            except Exception as e:
                print(f"[rag] [Layer 5] Tavily search error: {e}")

        # =========================================================================
        # ALL LAYERS FAILED
        # =========================================================================
        print(f"[rag] All 5 search layers failed to find reliable information.")
        fallback = "I could not find enough reliable information to answer this question accurately."
        if is_ml:
            fallback = translate_text(fallback, target_lang="ml", source_lang="en", llm=self.llm)
        return {
            "answer": fallback,
            "source_type": "none",
            "mode": "online" if is_online() else "offline",
            "confidence": 0.0,
            "sources": [],
            "is_translated": is_ml,
            "original_question": question
        }

    def query_rag(self, question: str, mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Main query entry point delegating to the 5-layer search orchestrator.
        """
        return self.orchestrate_search(question, mode=mode)

    def query_online(self, question: str) -> Dict[str, Any]:
        """Direct online query delegating to orchestrator with mode='online'."""
        return self.orchestrate_search(question, mode="online")

    def query(self, question: str, mode: Optional[str] = None) -> Dict[str, Any]:
        """Backward-compatible alias for orchestrate_search."""
        return self.orchestrate_search(question, mode=mode)

    # Keep backward-compatible alias
    def query(self, question: str, mode: Optional[str] = None) -> Dict[str, Any]:
        """Alias for query_rag() for backward compatibility."""
        return self.query_rag(question, mode=mode)

    # ─────────────────────────────────────────────────────────────────────
    # Database Management
    # ─────────────────────────────────────────────────────────────────────

    def get_indexed_sources(self) -> List[str]:
        """
        Returns a list of unique source document filenames indexed in the vector DB.

        Returns:
            List of unique filenames currently stored in the vector database.
        """
        if self.db is None:
            return []
        try:
            data = self.db.get()
            if data and "metadatas" in data and data["metadatas"]:
                sources = list(set(
                    m.get("source", "Unknown")
                    for m in data["metadatas"]
                    if m
                ))
                return sorted(sources)
        except Exception:
            pass
        return []

    def clear(self) -> bool:
        """
        Clears the vector database from memory and disk, and resets all state.

        Returns:
            True if cleared successfully, False otherwise.
        """
        success = clear_vector_store(
            db=self.db,
            store_type=self.vector_store_type,
            persist_directory=self.db_dir
        )
        if success:
            self.db = None
            self.retriever = None
            self.qa_chain = None
            print("[rag] Vector database cleared and state reset.")
        return success

    # ─────────────────────────────────────────────────────────────────────
    # Q&A Knowledge Base Access (for UI)
    # ─────────────────────────────────────────────────────────────────────

    def get_qa_entries(self):
        """Returns all Q&A knowledge base entries for UI display."""
        return self.qa_kb.get_all_entries()

    def add_qa_entry(self, question, answer, tags=None, aliases=None):
        """Adds a new Q&A entry and returns its index."""
        return self.qa_kb.add_entry(question, answer, tags, aliases)

    def delete_qa_entry(self, index):
        """Deletes a Q&A entry by index."""
        return self.qa_kb.delete_entry(index)

    def qa_entry_count(self):
        """Returns the number of Q&A knowledge base entries."""
        return self.qa_kb.entry_count()

    def reload_qa_kb(self):
        """Reloads the Q&A knowledge base from disk."""
        self.qa_kb.reload()
