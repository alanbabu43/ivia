"""
vector_store.py — Vector Database Management
============================================
Supports two vector store backends:

  1. ChromaDB  — Persistent, disk-backed vector store. Best for production use
                 with large document collections. Stores data in a local folder.

  2. FAISS     — Fast in-memory vector store with optional disk persistence.
                 Best for quick prototyping and smaller document sets.

Environment variables (from .env):
  VECTOR_STORE_TYPE  — "chroma" or "faiss" (default: "chroma")
  CHROMA_DB_DIR      — directory for ChromaDB persistence (default: "db")
  FAISS_INDEX_DIR    — directory for FAISS index files (default: "faiss_db")
  RAG_TOP_K          — number of top results to retrieve (default: 3)
"""

import os
import shutil
from enum import Enum
from typing import List, Optional, Union

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Constants (loaded from .env with sensible defaults)
# ─────────────────────────────────────────────────────────────────────────────
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "db")
FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "faiss_db")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))


class VectorStoreType(str, Enum):
    """Supported vector store backend types."""
    CHROMA = "chroma"
    FAISS = "faiss"


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB Functions
# ─────────────────────────────────────────────────────────────────────────────

def create_chroma_store(
    documents: List[Document],
    embeddings: Embeddings,
    persist_directory: str = CHROMA_DB_DIR
) -> Chroma:
    """
    Creates a new ChromaDB vector store from a list of documents and persists
    it to the specified directory.

    Args:
        documents: Split Document chunks to embed and store.
        embeddings: Embedding model to use for vectorization.
        persist_directory: Path where ChromaDB will save its data files.

    Returns:
        Chroma database instance.
    """
    print(f"[vector_store] Creating ChromaDB at '{persist_directory}' with {len(documents)} chunks...")
    os.makedirs(persist_directory, exist_ok=True)

    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    # persist() is deprecated/auto in Chroma >= 0.4; call only if the method exists
    try:
        db.persist()
    except AttributeError:
        pass  # Auto-persisted in newer versions

    print(f"[vector_store] ChromaDB created with {db._collection.count()} total vectors.")
    return db


def load_chroma_store(
    embeddings: Embeddings,
    persist_directory: str = CHROMA_DB_DIR
) -> Optional[Chroma]:
    """
    Loads an existing ChromaDB vector store from disk.

    Args:
        embeddings: Embedding model (must match the one used at creation time).
        persist_directory: Path where ChromaDB files reside.

    Returns:
        Chroma instance if the database exists, otherwise None.
    """
    if (
        os.path.exists(persist_directory)
        and os.path.isdir(persist_directory)
        and len(os.listdir(persist_directory)) > 0
    ):
        print(f"[vector_store] Loading existing ChromaDB from '{persist_directory}'...")
        try:
            db = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            count = db._collection.count()
            print(f"[vector_store] ChromaDB loaded with {count} vectors.")
            return db
        except Exception as e:
            print(f"[vector_store] Failed to load ChromaDB: {e}")
            return None
    return None


def add_to_chroma_store(db: Chroma, documents: List[Document]) -> None:
    """
    Appends new document chunks to an existing ChromaDB store.

    Args:
        db: Active Chroma database instance.
        documents: New Document chunks to add.
    """
    print(f"[vector_store] Adding {len(documents)} chunks to ChromaDB...")
    db.add_documents(documents)

    try:
        db.persist()
    except AttributeError:
        pass  # Auto-persisted in newer Chroma versions


def clear_chroma_store(
    db: Optional[Chroma] = None,
    persist_directory: str = CHROMA_DB_DIR
) -> bool:
    """
    Clears the ChromaDB collection (releases file locks) and deletes data from disk.

    Args:
        db: Active Chroma instance to delete collection from (releases Windows locks).
        persist_directory: Path to the ChromaDB directory.

    Returns:
        True if cleared successfully, False otherwise.
    """
    print(f"[vector_store] Clearing ChromaDB at '{persist_directory}'...")
    collection_deleted = False

    # Step 1: Delete the in-memory collection to release Windows file locks
    if db is not None:
        try:
            db.delete_collection()
            collection_deleted = True
            print("[vector_store] ChromaDB collection deleted (file locks released).")
        except Exception as e:
            print(f"[vector_store] Could not delete collection: {e}")

    # Step 2: Remove the persisted directory from disk
    if os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory)
            print("[vector_store] ChromaDB directory removed.")
            return True
        except Exception as e:
            print(f"[vector_store] Could not remove directory '{persist_directory}': {e}")
            # Try removing files individually (Windows lock workaround)
            try:
                for item in os.listdir(persist_directory):
                    item_path = os.path.join(persist_directory, item)
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                return True
            except Exception as inner_e:
                print(f"[vector_store] Failed to clear contents: {inner_e}")
                return collection_deleted  # Return True if at least collection was reset

    return True


# ─────────────────────────────────────────────────────────────────────────────
# FAISS Functions
# ─────────────────────────────────────────────────────────────────────────────

def create_faiss_store(
    documents: List[Document],
    embeddings: Embeddings,
    index_dir: str = FAISS_INDEX_DIR
) -> "FAISS":
    """
    Creates a FAISS vector store from document chunks and saves it to disk.

    FAISS is a highly optimized in-memory similarity search library. The index
    is saved as two files: index.faiss and index.pkl (document metadata).

    Args:
        documents: Split Document chunks to embed and index.
        embeddings: Embedding model for vectorization.
        index_dir: Directory to persist the FAISS index files.

    Returns:
        FAISS vector store instance.
    """
    from langchain_community.vectorstores import FAISS

    print(f"[vector_store] Creating FAISS index at '{index_dir}' with {len(documents)} chunks...")
    os.makedirs(index_dir, exist_ok=True)

    db = FAISS.from_documents(documents=documents, embedding=embeddings)

    # Persist FAISS index to disk
    db.save_local(index_dir)
    print(f"[vector_store] FAISS index saved to '{index_dir}'.")
    return db


def load_faiss_store(
    embeddings: Embeddings,
    index_dir: str = FAISS_INDEX_DIR
) -> Optional["FAISS"]:
    """
    Loads a previously saved FAISS index from disk.

    Args:
        embeddings: Embedding model (must match the one used at creation time).
        index_dir: Directory containing the FAISS index files.

    Returns:
        FAISS instance if index files exist, otherwise None.
    """
    from langchain_community.vectorstores import FAISS

    index_path = os.path.join(index_dir, "index.faiss")
    if os.path.exists(index_path):
        print(f"[vector_store] Loading FAISS index from '{index_dir}'...")
        try:
            # allow_dangerous_deserialization=True is required for LangChain FAISS loads
            db = FAISS.load_local(
                index_dir,
                embeddings,
                allow_dangerous_deserialization=True
            )
            print("[vector_store] FAISS index loaded.")
            return db
        except Exception as e:
            print(f"[vector_store] Failed to load FAISS index: {e}")
            return None
    return None


def add_to_faiss_store(
    db: "FAISS",
    documents: List[Document],
    index_dir: str = FAISS_INDEX_DIR
) -> None:
    """
    Merges new document chunks into an existing FAISS store and re-saves it.

    Note: FAISS does not support true incremental updates; we add documents
    to the in-memory index and then overwrite the saved index on disk.

    Args:
        db: Active FAISS vector store instance.
        documents: New Document chunks to add.
        index_dir: Directory to save the updated index.
    """
    from langchain_community.vectorstores import FAISS

    print(f"[vector_store] Adding {len(documents)} chunks to FAISS store...")
    db.add_documents(documents)
    db.save_local(index_dir)
    print(f"[vector_store] FAISS index updated and saved.")


def clear_faiss_store(index_dir: str = FAISS_INDEX_DIR) -> bool:
    """
    Deletes the FAISS index directory from disk.

    Args:
        index_dir: Path to the FAISS index directory to delete.

    Returns:
        True if deleted successfully or directory did not exist, False otherwise.
    """
    if os.path.exists(index_dir):
        try:
            shutil.rmtree(index_dir)
            print(f"[vector_store] FAISS index directory '{index_dir}' deleted.")
            return True
        except Exception as e:
            print(f"[vector_store] Failed to delete FAISS directory: {e}")
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Unified API (used by rag.py — backend-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def create_vector_store(
    documents: List[Document],
    embeddings: Embeddings,
    store_type: str = None,
    persist_directory: str = None
) -> Union[Chroma, "FAISS"]:
    """
    Creates a vector store of the specified type (ChromaDB or FAISS).

    Args:
        documents: Document chunks to embed and store.
        embeddings: Embedding model instance.
        store_type: "chroma" or "faiss". Reads VECTOR_STORE_TYPE env var if None.
        persist_directory: Override default persistence directory.

    Returns:
        Vector store instance (Chroma or FAISS).
    """
    if store_type is None:
        store_type = os.getenv("VECTOR_STORE_TYPE", "chroma")

    if store_type == VectorStoreType.FAISS:
        return create_faiss_store(documents, embeddings, persist_directory or FAISS_INDEX_DIR)
    else:
        return create_chroma_store(documents, embeddings, persist_directory or CHROMA_DB_DIR)


def load_vector_store(
    embeddings: Embeddings,
    store_type: str = None,
    persist_directory: str = None
) -> Optional[Union[Chroma, "FAISS"]]:
    """
    Loads an existing vector store from disk (ChromaDB or FAISS).

    Args:
        embeddings: Embedding model instance.
        store_type: "chroma" or "faiss". Reads VECTOR_STORE_TYPE env var if None.
        persist_directory: Override default persistence directory.

    Returns:
        Vector store instance if found, otherwise None.
    """
    if store_type is None:
        store_type = os.getenv("VECTOR_STORE_TYPE", "chroma")

    if store_type == VectorStoreType.FAISS:
        return load_faiss_store(embeddings, persist_directory or FAISS_INDEX_DIR)
    else:
        return load_chroma_store(embeddings, persist_directory or CHROMA_DB_DIR)


def add_documents_to_store(
    db: Union[Chroma, "FAISS"],
    documents: List[Document],
    store_type: str = None,
    index_dir: str = None
) -> None:
    """
    Adds documents to an existing vector store (ChromaDB or FAISS).

    Args:
        db: Active vector store instance.
        documents: Document chunks to add.
        store_type: "chroma" or "faiss". Reads VECTOR_STORE_TYPE env var if None.
        index_dir: FAISS index directory (only used when store_type is "faiss").
    """
    if store_type is None:
        store_type = os.getenv("VECTOR_STORE_TYPE", "chroma")

    if store_type == VectorStoreType.FAISS:
        add_to_faiss_store(db, documents, index_dir or FAISS_INDEX_DIR)
    else:
        add_to_chroma_store(db, documents)


def clear_vector_store(
    db=None,
    store_type: str = None,
    persist_directory: str = None
) -> bool:
    """
    Clears the vector store (both in-memory and on disk).

    Args:
        db: Active vector store instance (used to release locks in ChromaDB).
        store_type: "chroma" or "faiss". Reads VECTOR_STORE_TYPE env var if None.
        persist_directory: Override default persistence directory.

    Returns:
        True if cleared successfully, False otherwise.
    """
    if store_type is None:
        store_type = os.getenv("VECTOR_STORE_TYPE", "chroma")

    if store_type == VectorStoreType.FAISS:
        return clear_faiss_store(persist_directory or FAISS_INDEX_DIR)
    else:
        return clear_chroma_store(db, persist_directory or CHROMA_DB_DIR)
