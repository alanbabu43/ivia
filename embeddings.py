"""
embeddings.py — Embedding Model Loader
=======================================
Provides a HuggingFace sentence-transformer embedding model for converting
text chunks into dense vector representations used in the RAG pipeline.

Default model: sentence-transformers/all-MiniLM-L6-v2
  - 384-dimensional vectors
  - ~22M parameters — fast and lightweight
  - Excellent for semantic similarity and QA retrieval tasks

Environment variable:
  EMBEDDING_MODEL — override the default HuggingFace model name
"""

import os
import torch
from dotenv import load_dotenv

# Load .env configuration if available
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Use langchain_huggingface (modern, non-deprecated package)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback for environments that only have langchain-community installed
    from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedding_model(
    model_name: str = None
) -> HuggingFaceEmbeddings:
    """
    Initializes and returns a HuggingFaceEmbeddings model instance.

    Automatically detects GPU (CUDA) availability for faster inference.
    Falls back to CPU if CUDA is not available.

    The model is loaded from HuggingFace Hub on first use and cached locally.
    Subsequent calls use the cached model.

    Args:
        model_name: HuggingFace model identifier. If None, reads from the
                    EMBEDDING_MODEL environment variable. Defaults to
                    'sentence-transformers/all-MiniLM-L6-v2'.

    Returns:
        HuggingFaceEmbeddings instance ready for use with LangChain vector stores.

    Raises:
        Exception: If the model cannot be loaded (e.g., network issues, bad model name).
    """
    # Resolve model name: argument → env var → hardcoded default
    if model_name is None:
        model_name = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    # Detect computation device: GPU preferred for speed, CPU as universal fallback
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[embeddings] Loading '{model_name}' on device '{device}'...")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            # Normalize embeddings to unit vectors → cosine similarity via dot product
            encode_kwargs={"normalize_embeddings": True}
        )
        print(f"[embeddings] Model loaded successfully.")
        return embeddings

    except Exception as e:
        print(f"[embeddings] Failed to load embedding model '{model_name}': {e}")
        raise
