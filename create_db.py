"""
create_db.py — CLI Database Builder
====================================
Creates a vector database from PDF files in a specified directory.

This is a standalone script that can be run from the command line to
pre-index PDF documents before launching the Streamlit chatbot.

Usage:
    # Build ChromaDB from all PDFs in data/ folder (default)
    python create_db.py

    # Build FAISS index from a specific folder
    python create_db.py --input ./my_pdfs --store faiss

    # Custom chunk settings
    python create_db.py --chunk-size 500 --chunk-overlap 100

Environment variables (from .env):
    EMBEDDING_MODEL      — HuggingFace embedding model
    VECTOR_STORE_TYPE    — "chroma" or "faiss"
    CHROMA_DB_DIR        — ChromaDB persistence directory
    FAISS_INDEX_DIR      — FAISS index persistence directory
    CHUNK_SIZE           — Characters per text chunk
    CHUNK_OVERLAP        — Overlap between chunks
"""

import os
import sys
import glob
import argparse

from dotenv import load_dotenv

# Load environment variables before other imports
load_dotenv()

from pdf_loader import extract_documents_from_pdf, split_documents
from embeddings import get_embedding_model
from vector_store import (
    create_vector_store,
    CHROMA_DB_DIR, FAISS_INDEX_DIR
)


def build_database(
    input_dir: str,
    store_type: str,
    output_dir: str,
    chunk_size: int,
    chunk_overlap: int
) -> int:
    """
    Scans a directory for PDF files, extracts text, chunks it, and builds
    a vector database for RAG-based question answering.

    Args:
        input_dir: Path to directory containing PDF files.
        store_type: "chroma" or "faiss".
        output_dir: Directory to persist the vector database.
        chunk_size: Maximum characters per text chunk.
        chunk_overlap: Overlap characters between consecutive chunks.

    Returns:
        Total number of chunks indexed across all PDFs.
    """
    # ── Discover PDF files ────────────────────────────────────────────────
    pdf_pattern = os.path.join(input_dir, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)

    if not pdf_files:
        print(f"[create_db] No PDF files found in '{input_dir}'.")
        print(f"[create_db] Searched pattern: {pdf_pattern}")
        return 0

    print(f"[create_db] Found {len(pdf_files)} PDF file(s) in '{input_dir}':")
    for f in pdf_files:
        print(f"  - {os.path.basename(f)}")

    # ── Extract and chunk all documents ───────────────────────────────────
    all_chunks = []
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"\n[create_db] Processing '{filename}'...")

        try:
            # Extract text as LangChain Documents (one per page)
            raw_docs = extract_documents_from_pdf(pdf_path, filename)
            if not raw_docs:
                print(f"[create_db] WARNING: No text extracted from '{filename}'. Skipping.")
                continue

            # Split into overlapping chunks
            chunks = split_documents(raw_docs, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
            print(f"[create_db] '{filename}' -> {len(raw_docs)} pages -> {len(chunks)} chunks")

        except Exception as e:
            print(f"[create_db] ERROR processing '{filename}': {e}")
            continue

    if not all_chunks:
        print("\n[create_db] No chunks generated from any PDF. Database not created.")
        return 0

    # ── Load embedding model ──────────────────────────────────────────────
    print(f"\n[create_db] Loading embedding model...")
    embeddings = get_embedding_model()

    # ── Build vector database ─────────────────────────────────────────────
    print(f"[create_db] Building {store_type.upper()} database at '{output_dir}'...")
    create_vector_store(
        documents=all_chunks,
        embeddings=embeddings,
        store_type=store_type,
        persist_directory=output_dir
    )

    print(f"\n{'='*60}")
    print(f"  Database Created Successfully!")
    print(f"  Store type : {store_type.upper()}")
    print(f"  Location   : {output_dir}")
    print(f"  Total PDFs : {len(pdf_files)}")
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"{'='*60}")

    return len(all_chunks)


def main():
    """Entry point for CLI usage with argparse."""
    parser = argparse.ArgumentParser(
        description="Build a vector database from PDF documents for the RAG chatbot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_db.py
  python create_db.py --input ./my_pdfs --store faiss
  python create_db.py --chunk-size 500 --chunk-overlap 100
        """
    )

    parser.add_argument(
        "--input", "-i",
        default="data",
        help="Directory containing PDF files (default: 'data')"
    )
    parser.add_argument(
        "--store", "-s",
        choices=["chroma", "faiss"],
        default=os.getenv("VECTOR_STORE_TYPE", "chroma"),
        help="Vector store type (default: from .env or 'chroma')"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory for the vector database (default: auto from store type)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("CHUNK_SIZE", "1000")),
        help="Maximum characters per chunk (default: from .env or 1000)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=int(os.getenv("CHUNK_OVERLAP", "200")),
        help="Overlap characters between chunks (default: from .env or 200)"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output:
        output_dir = args.output
    elif args.store == "faiss":
        output_dir = FAISS_INDEX_DIR
    else:
        output_dir = CHROMA_DB_DIR

    # Validate input directory
    if not os.path.isdir(args.input):
        print(f"[create_db] ERROR: Input directory '{args.input}' does not exist.")
        sys.exit(1)

    # Build the database
    total = build_database(
        input_dir=args.input,
        store_type=args.store,
        output_dir=output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    if total == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()