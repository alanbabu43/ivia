"""
qa_knowledge.py — Q&A Knowledge Base with Semantic Search
==========================================================
Manages a curated Question-Answer knowledge base stored as a JSON file.
Provides semantic similarity search using the same HuggingFace embedding
model used by the RAG pipeline — no extra model loading required.

Knowledge Base Priority Order (used by the system prompt):
  1. Q&A Knowledge Base (this module)  — Highest priority
  2. Uploaded PDF documents             — Second priority
  3. LLM pre-trained knowledge          — Fallback

Storage format:
  JSON array in `data/qa_knowledge.json`:
  [
    {
      "question": "Who is the Prime Minister of India?",
      "answer": "The Prime Minister of India is Narendra Modi (as of 2024).",
      "tags": ["india", "politics"],
      "aliases": ["Who is India's PM?", "Who leads India?"]
    },
    ...
  ]

Usage:
    from embeddings import get_embedding_model
    embeddings = get_embedding_model()
    qa_kb = QAKnowledgeBase(embeddings=embeddings)
    results = qa_kb.search("Who is Kerala CM?")
"""

import json
import os
import sys
import numpy as np
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Constants (from .env)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_QA_FILE = os.getenv("QA_KNOWLEDGE_FILE", "data/qa_knowledge.json")
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("QA_SIMILARITY_THRESHOLD", "0.80"))
DEFAULT_QA_TOP_K = int(os.getenv("QA_TOP_K", "3"))


import re

# ─────────────────────────────────────────────────────────────────────────────
# Role & Intent Protection Mapping
# ─────────────────────────────────────────────────────────────────────────────
ROLE_INTENT_MAP = {
    "president": "president",
    "governor": "governor",
    "chief minister": "cm",
    "cm": "cm",
    "prime minister": "pm",
    "pm": "pm",
    "district": "district_count",
    "districts": "district_count",
    "capital": "capital"
}


def extract_role_intents(text: str) -> set:
    """Extracts role and intent keywords from query/text for role protection."""
    if not text:
        return set()
    text_lower = text.lower()
    found = set()
    for kw, intent in ROLE_INTENT_MAP.items():
        if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
            found.add(intent)
    return found


class QAKnowledgeBase:
    """
    Manages a curated Q&A knowledge base with semantic similarity search.

    Uses the shared HuggingFace embedding model to compute cosine similarity
    between user questions and stored Q&A entries (including aliases).

    The knowledge base is loaded from a JSON file on disk and can be modified
    at runtime (add/delete entries). Changes are persisted immediately.

    Attributes:
        json_path: Path to the JSON knowledge base file.
        threshold: Minimum cosine similarity score to consider a match.
        top_k: Maximum number of results to return from search.
        entries: In-memory list of Q&A entries.
    """

    def __init__(
        self,
        embeddings: Any,
        json_path: str = None,
        threshold: float = None,
        top_k: int = None
    ):
        """
        Initializes the Q&A Knowledge Base.

        Loads entries from the JSON file and pre-computes embeddings for
        all questions and aliases for fast semantic search.

        Args:
            embeddings: HuggingFace embedding model instance (shared with RAG pipeline).
            json_path: Path to the JSON file. Defaults to QA_KNOWLEDGE_FILE env var.
            threshold: Minimum cosine similarity for a match. Defaults to QA_SIMILARITY_THRESHOLD.
            top_k: Max results per search. Defaults to QA_TOP_K.
        """
        self.embeddings = embeddings
        self.json_path = json_path or DEFAULT_QA_FILE
        self.threshold = threshold if threshold is not None else DEFAULT_SIMILARITY_THRESHOLD
        self.top_k = top_k if top_k is not None else DEFAULT_QA_TOP_K

        # Internal state
        self.entries: List[Dict] = []
        self._embedded_texts: List[str] = []       # Flattened list of all questions + aliases
        self._text_to_entry_idx: List[int] = []    # Maps each embedded text → entry index
        self._embeddings_matrix: Optional[np.ndarray] = None  # Pre-computed embedding vectors

        # Load and index
        self._load_entries()
        self._build_index()

    # ─────────────────────────────────────────────────────────────────────
    # Internal: Loading & Indexing
    # ─────────────────────────────────────────────────────────────────────

    def _load_entries(self) -> None:
        """Loads Q&A entries from the primary JSON file and any additional dataset files (like ivia_greetings_rag.json)."""
        self.entries = []
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        self.entries.extend(loaded)
                    elif isinstance(loaded, dict) and "documents" in loaded:
                        self.entries.extend(loaded["documents"])
                print(f"[qa_knowledge] Loaded {len(self.entries)} Q&A entries from '{self.json_path}'.")
            except (json.JSONDecodeError, IOError) as e:
                print(f"[qa_knowledge] Error loading '{self.json_path}': {e}", file=sys.stderr)

        # Check for ivia_greetings_rag.json in data/
        greetings_file = os.path.join(os.path.dirname(self.json_path) or "data", "ivia_greetings_rag.json")
        if os.path.exists(greetings_file) and os.path.abspath(greetings_file) != os.path.abspath(self.json_path):
            try:
                with open(greetings_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    docs = data.get("documents", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                    added = 0
                    for d in docs:
                        aliases = d.get("variations", []) or d.get("aliases", [])
                        entry = {
                            "question": d.get("question", ""),
                            "answer": d.get("answer", ""),
                            "aliases": aliases,
                            "intent": d.get("intent", ""),
                            "category": d.get("category", "greetings"),
                            "tags": d.get("keywords", ["greeting", "ivia"])
                        }
                        self.entries.append(entry)
                        added += 1
                    print(f"[qa_knowledge] Loaded {added} greeting entries from '{greetings_file}'.")
            except Exception as e:
                print(f"[qa_knowledge] Error loading '{greetings_file}': {e}", file=sys.stderr)

    def _save_entries(self) -> None:
        """Persists the current entries list to the JSON file on disk."""
        try:
            os.makedirs(os.path.dirname(self.json_path) or ".", exist_ok=True)
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, indent=2, ensure_ascii=False)
            print(f"[qa_knowledge] Saved {len(self.entries)} entries to '{self.json_path}'.")
        except IOError as e:
            print(f"[qa_knowledge] Error saving to '{self.json_path}': {e}", file=sys.stderr)

    def _build_index(self) -> None:
        """
        Pre-computes embeddings for all questions and their aliases.

        Creates a flat list of all searchable text strings, maps each one
        back to its parent entry index, and generates an embedding matrix
        for fast batch cosine similarity computation.
        """
        self._embedded_texts = []
        self._text_to_entry_idx = []

        for idx, entry in enumerate(self.entries):
            # Add the primary question
            question = entry.get("question", "").strip()
            if question:
                self._embedded_texts.append(question)
                self._text_to_entry_idx.append(idx)

            # Add all aliases (alternative phrasings)
            for alias in entry.get("aliases", []):
                alias = alias.strip()
                if alias:
                    self._embedded_texts.append(alias)
                    self._text_to_entry_idx.append(idx)

        if not self._embedded_texts:
            self._embeddings_matrix = None
            print("[qa_knowledge] No entries to index.")
            return

        try:
            # Batch embed all questions + aliases
            vectors = self.embeddings.embed_documents(self._embedded_texts)
            self._embeddings_matrix = np.array(vectors, dtype=np.float32)

            # Normalize to unit vectors for cosine similarity via dot product
            norms = np.linalg.norm(self._embeddings_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Prevent division by zero
            self._embeddings_matrix = self._embeddings_matrix / norms

            print(f"[qa_knowledge] Indexed {len(self._embedded_texts)} searchable texts "
                  f"({len(self.entries)} entries).")
        except Exception as e:
            print(f"[qa_knowledge] Error building embedding index: {e}", file=sys.stderr)
            self._embeddings_matrix = None

    # ─────────────────────────────────────────────────────────────────────
    # Public: Semantic Search
    # ─────────────────────────────────────────────────────────────────────

    def search(self, question: str, top_k: int = None) -> List[Dict]:
        """
        Performs semantic similarity search against the Q&A knowledge base.

        Embeds the user's question and computes cosine similarity against
        all pre-computed Q&A embeddings (questions + aliases). Returns
        matching entries above the similarity threshold.

        Args:
            question: User's question string.
            top_k: Override the default max results. Defaults to self.top_k.

        Returns:
            List of matched entries as dicts:
            [{"question": "...", "answer": "...", "score": 0.92, "matched_text": "..."}, ...]

            Empty list if no matches meet the threshold.
        """
        if not question or not question.strip():
            return []

        if self._embeddings_matrix is None or len(self._embedded_texts) == 0:
            return []

        top_k = top_k or self.top_k

        try:
            # Embed the user's query
            query_vector = np.array(
                self.embeddings.embed_query(question.strip()),
                dtype=np.float32
            )

            # Normalize query vector
            query_norm = np.linalg.norm(query_vector)
            if query_norm == 0:
                return []
            query_vector = query_vector / query_norm

            # Compute cosine similarity via dot product (both vectors are unit-normalized)
            similarities = self._embeddings_matrix @ query_vector

            # Find indices above threshold, sorted by score descending
            candidate_indices = np.where(similarities >= self.threshold)[0]
            if len(candidate_indices) == 0:
                return []

            # Sort by similarity score (descending)
            sorted_indices = candidate_indices[np.argsort(-similarities[candidate_indices])]

            # Deduplicate by entry index (keep highest-scoring match per entry)
            query_intents = extract_role_intents(question)
            seen_entries = set()
            results = []

            for idx in sorted_indices:
                entry_idx = self._text_to_entry_idx[idx]
                if entry_idx in seen_entries:
                    continue

                matched_str = self._embedded_texts[idx]
                matched_intents = extract_role_intents(matched_str)

                # Role & Intent protection: if query asks for a specific role (e.g. President)
                # and candidate is for a conflicting role (e.g. CM / Governor), skip it!
                if query_intents and matched_intents and query_intents.isdisjoint(matched_intents):
                    print(f"[qa_knowledge] Rejecting role mismatch: query intents {query_intents} vs matched intents {matched_intents}")
                    continue

                # Entity-level protection: if the query is for entity X and the matched entry
                # is for a different entity Y, and both are role questions, reject the match
                # to prevent e.g. "Governor of India" from returning "Governor of Kerala"
                entry = self.entries[entry_idx]
                entry_entity = entry.get("entity", "").strip().lower()
                entry_role = entry.get("role", None)

                if query_intents and entry_entity and entry_role:
                    # Determine query entity from the question text directly
                    from question_processor import detect_entity as _detect_entity
                    q_entity, q_entity_type = _detect_entity(question)
                    if q_entity and q_entity.lower() != entry_entity:
                        print(f"[qa_knowledge] Rejecting entity mismatch: query entity '{q_entity}' vs entry entity '{entry_entity}'")
                        continue

                seen_entries.add(entry_idx)

                entry = self.entries[entry_idx]
                results.append({
                    "question": entry.get("question", ""),
                    "answer": entry.get("answer", ""),
                    "score": float(similarities[idx]),
                    "matched_text": matched_str,
                    "tags": entry.get("tags", [])
                })

                if len(results) >= top_k:
                    break

            return results

        except Exception as e:
            print(f"[qa_knowledge] Search error: {e}", file=sys.stderr)
            return []

    def format_context(self, question: str, top_k: int = None) -> str:
        """
        Searches the Q&A KB and formats the results as a context string
        suitable for injection into the system prompt's {qa_context} variable.

        Args:
            question: User's question string.
            top_k: Override the default max results.

        Returns:
            Formatted string with matched Q&A pairs, or
            "No relevant Q&A entries found." if nothing matches.
        """
        results = self.search(question, top_k=top_k)

        if not results:
            return "No relevant Q&A entries found."

        context_parts = []
        for i, r in enumerate(results, 1):
            score_pct = f"{r['score'] * 100:.0f}%"
            context_parts.append(
                f"Q{i}: {r['question']}\n"
                f"A{i}: {r['answer']}\n"
                f"(Match confidence: {score_pct})"
            )

        return "\n\n".join(context_parts)

    def get_direct_match(self, question: str, direct_threshold: float = 0.85) -> Optional[Dict]:
        """
        Checks if the question has a high-confidence match in the Q&A KB.

        If a match is found with similarity >= direct_threshold (default 0.85),
        returns the matching dictionary directly:
          {"answer": "...", "score": 0.95, "question": "..."}

        This enables direct answering without sending requests to Llama, saving VRAM.
        """
        results = self.search(question, top_k=1)
        if results and results[0]["score"] >= direct_threshold:
            print(f"[qa_knowledge] Direct Q&A match found for '{question[:50]}' (score: {results[0]['score']:.3f})")
            return results[0]
        return None

    # ─────────────────────────────────────────────────────────────────────
    # Public: CRUD Operations
    # ─────────────────────────────────────────────────────────────────────

    def add_entry(
        self,
        question: str,
        answer: str,
        tags: List[str] = None,
        aliases: List[str] = None
    ) -> int:
        """
        Adds a new Q&A entry to the knowledge base.

        Persists to disk and rebuilds the embedding index immediately.

        Args:
            question: The primary question text.
            answer: The answer text.
            tags: Optional list of category tags.
            aliases: Optional list of alternative question phrasings.

        Returns:
            Index of the newly added entry.
        """
        entry = {
            "question": question.strip(),
            "answer": answer.strip(),
            "tags": [t.strip() for t in (tags or []) if t.strip()],
            "aliases": [a.strip() for a in (aliases or []) if a.strip()]
        }

        self.entries.append(entry)
        self._save_entries()
        self._build_index()

        print(f"[qa_knowledge] Added entry #{len(self.entries) - 1}: '{question[:60]}...'")
        return len(self.entries) - 1

    def delete_entry(self, index: int) -> bool:
        """
        Deletes a Q&A entry by its index.

        Persists to disk and rebuilds the embedding index immediately.

        Args:
            index: Zero-based index of the entry to delete.

        Returns:
            True if deleted successfully, False if index is out of range.
        """
        if 0 <= index < len(self.entries):
            removed = self.entries.pop(index)
            self._save_entries()
            self._build_index()
            print(f"[qa_knowledge] Deleted entry #{index}: '{removed.get('question', '')[:60]}'")
            return True
        print(f"[qa_knowledge] Invalid index {index} (have {len(self.entries)} entries).")
        return False

    def get_all_entries(self) -> List[Dict]:
        """
        Returns all Q&A entries in the knowledge base.

        Returns:
            List of entry dictionaries (question, answer, tags, aliases).
        """
        return list(self.entries)

    def reload(self) -> None:
        """
        Reloads entries from disk and rebuilds the embedding index.

        Useful after external edits to the JSON file.
        """
        print("[qa_knowledge] Reloading knowledge base...")
        self._load_entries()
        self._build_index()

    def entry_count(self) -> int:
        """Returns the number of entries in the knowledge base."""
        return len(self.entries)
