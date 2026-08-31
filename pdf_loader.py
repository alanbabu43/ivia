"""
pdf_loader.py — PDF Text Extraction Module
==========================================
Supports two extraction methods:

  Method 1 — Direct Extraction:
      extract_text_direct()  →  Returns a clean string of all PDF text.
      Clean text is passed directly to the LLM for QA (no vector DB needed).

  Method 2 — RAG Extraction:
      extract_documents_from_pdf()  →  Returns LangChain Document objects per page.
      split_documents()             →  Chunks the documents for vector embedding.
"""

import io
import re
from typing import List, Tuple, Union

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ─────────────────────────────────────────────────────────────────────────────
# Utility: Text Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Cleans raw extracted PDF text by removing common artifacts.

    Operations performed:
      - Collapse repeated whitespace and blank lines
      - Remove control characters (non-printable)
      - Strip leading/trailing whitespace

    Args:
        text: Raw string extracted from a PDF page.

    Returns:
        Cleaned string with normalized whitespace.
    """
    if not text:
        return ""

    # Remove non-printable control characters (keep newlines and tabs)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", text)

    # Collapse 3+ consecutive newlines into 2 (paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces/tabs into a single space
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Method 1 — Direct PDF Text Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_direct(pdf_file: Union[str, io.BytesIO]) -> Tuple[str, int]:
    """
    Extracts and cleans ALL text from a PDF as a single string.
    Uses pdfplumber (most accurate) with PyMuPDF as fallback.

    This is designed for Method 1 (direct QA without a vector database).
    The full text is passed directly into the LLM context window.

    Args:
        pdf_file: File path (str) or BytesIO stream of the PDF.

    Returns:
        Tuple of:
          - full_text (str): Cleaned, concatenated text from all pages.
          - page_count (int): Total number of pages in the PDF.

    Raises:
        RuntimeError: If no text could be extracted by any available library.
    """
    # ── Attempt 1: pdfplumber (most reliable for complex layouts) ─────────────
    try:
        import pdfplumber

        # pdfplumber needs a seekable stream or a file path
        if isinstance(pdf_file, io.BytesIO):
            pdf_file.seek(0)
            context = pdfplumber.open(pdf_file)
        else:
            context = pdfplumber.open(pdf_file)

        pages_text = []
        page_count = 0
        with context as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                raw = page.extract_text() or ""
                pages_text.append(clean_text(raw))

        full_text = "\n\n".join([t for t in pages_text if t])
        if full_text.strip():
            print(f"[pdf_loader] pdfplumber extracted {page_count} pages successfully.")
            return full_text, page_count

    except ImportError:
        print("[pdf_loader] pdfplumber not available, trying PyMuPDF...")
    except Exception as e:
        print(f"[pdf_loader] pdfplumber failed: {e}. Trying PyMuPDF...")

    # ── Attempt 2: PyMuPDF (fitz) — excellent for complex PDFs ──────────────
    try:
        import fitz  # PyMuPDF

        if isinstance(pdf_file, io.BytesIO):
            pdf_file.seek(0)
            doc = fitz.open(stream=pdf_file, filetype="pdf")
        else:
            doc = fitz.open(pdf_file)

        pages_text = []
        page_count = len(doc)
        for page in doc:
            raw = page.get_text("text") or ""
            pages_text.append(clean_text(raw))
        doc.close()

        full_text = "\n\n".join([t for t in pages_text if t])
        if full_text.strip():
            print(f"[pdf_loader] PyMuPDF extracted {page_count} pages successfully.")
            return full_text, page_count

    except ImportError:
        print("[pdf_loader] PyMuPDF not available, trying pypdf...")
    except Exception as e:
        print(f"[pdf_loader] PyMuPDF failed: {e}. Trying pypdf...")

    # ── Attempt 3: pypdf — pure-Python fallback ───────────────────────────────
    try:
        from pypdf import PdfReader

        if isinstance(pdf_file, io.BytesIO):
            pdf_file.seek(0)

        reader = PdfReader(pdf_file)
        pages_text = []
        page_count = len(reader.pages)
        for page in reader.pages:
            raw = page.extract_text() or ""
            pages_text.append(clean_text(raw))

        full_text = "\n\n".join([t for t in pages_text if t])
        if full_text.strip():
            print(f"[pdf_loader] pypdf extracted {page_count} pages successfully.")
            return full_text, page_count

    except Exception as e:
        print(f"[pdf_loader] pypdf also failed: {e}")

    raise RuntimeError(
        "Could not extract text from the PDF. "
        "Ensure pdfplumber, PyMuPDF, or pypdf is installed. "
        "The PDF might be image-based (scanned) and require OCR."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Method 2 — RAG Document Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_documents_from_pdf(pdf_file: Union[str, io.BytesIO], filename: str) -> List[Document]:
    """
    Extracts text from a PDF and returns a list of LangChain Document objects,
    one per page, with metadata (source filename and page number).

    This is designed for Method 2 (RAG pipeline) where text is chunked and
    stored in a vector database for similarity search.

    Args:
        pdf_file: File path (str) or BytesIO stream of the PDF.
        filename: Name of the PDF file — stored in document metadata.

    Returns:
        List of Document objects containing page text and metadata.

    Raises:
        Exception: If the PDF cannot be read at all.
    """
    documents = []

    # ── Attempt 1: pypdf (LangChain-native, no binary deps) ──────────────────
    try:
        from pypdf import PdfReader

        if isinstance(pdf_file, io.BytesIO):
            pdf_file.seek(0)

        reader = PdfReader(pdf_file)
        for page_num, page in enumerate(reader.pages):
            raw = page.extract_text() or ""
            text = clean_text(raw)
            if text:
                documents.append(Document(
                    page_content=text,
                    metadata={"source": filename, "page": page_num + 1}
                ))

        if documents:
            print(f"[pdf_loader] Extracted {len(documents)} pages from '{filename}' via pypdf.")
            return documents

    except Exception as e:
        print(f"[pdf_loader] pypdf extraction failed for '{filename}': {e}")

    # ── Attempt 2: pdfplumber fallback ────────────────────────────────────────
    try:
        import pdfplumber

        if isinstance(pdf_file, io.BytesIO):
            pdf_file.seek(0)

        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                raw = page.extract_text() or ""
                text = clean_text(raw)
                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": filename, "page": page_num + 1}
                    ))

        if documents:
            print(f"[pdf_loader] Extracted {len(documents)} pages from '{filename}' via pdfplumber.")
            return documents

    except Exception as e:
        print(f"[pdf_loader] pdfplumber fallback failed for '{filename}': {e}")

    if not documents:
        raise RuntimeError(
            f"No text could be extracted from '{filename}'. "
            "The PDF may be image-based (scanned). Try an OCR-enabled workflow."
        )

    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Splits a list of LangChain Documents into smaller, overlapping text chunks.

    Uses RecursiveCharacterTextSplitter which splits by paragraphs → sentences
    → words to maintain semantic coherence as much as possible.

    Args:
        documents: List of LangChain Document objects (one per PDF page).
        chunk_size: Maximum characters per chunk (default 1000).
        chunk_overlap: Overlap characters between consecutive chunks (default 200).

    Returns:
        List of split Document objects ready for embedding and indexing.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Smart split priority
    )
    chunks = splitter.split_documents(documents)
    print(f"[pdf_loader] Split {len(documents)} pages into {len(chunks)} chunks.")
    return chunks
