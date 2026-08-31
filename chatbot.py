"""
chatbot.py — LLM Backend Manager & QA Chains
=============================================
Manages three LLM provider backends:
  1. HuggingFace Pipeline   — Local inference, no API key required (default)
  2. Ollama                 — Local models via Ollama server
  3. OpenAI                 — Cloud API (GPT-4o-mini, GPT-4o, etc.)

Two QA modes:
  - get_qa_chain()       →  Method 2 RAG: retriever-based RetrievalQA chain
  - answer_from_text()   →  Method 1 Direct: raw text + question → LLM answer

Environment variables (from .env):
  OPENAI_API_KEY        — Required for OpenAI backend
  DEFAULT_LLM_BACKEND   — "huggingface", "ollama", or "openai"
  DEFAULT_HF_MODEL      — HuggingFace model ID
  DEFAULT_OLLAMA_MODEL  — Ollama model name
  DEFAULT_OPENAI_MODEL  — OpenAI model name
"""

import json
import os
import re
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

# Load .env config before any other imports that may need env vars
load_dotenv()

# ── LangChain imports with graceful fallbacks ──────────────────────────────
# PromptTemplate lives in langchain_core (always available)
from langchain_core.prompts import PromptTemplate

# RetrievalQA: try langchain_classic (installed in this env) → langchain → error
try:
    from langchain_classic.chains import RetrievalQA
except ImportError:
    try:
        from langchain.chains import RetrievalQA
    except ImportError:
        raise ImportError(
            "Could not import RetrievalQA. Install 'langchain' or 'langchain-classic': "
            "pip install langchain"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Layer Pipeline Prompts (5-Layer Sequential Fallback Architecture)
# ─────────────────────────────────────────────────────────────────────────────

# Layer 1: PDF Knowledge Base Evaluation Prompt (IVIA)
PDF_QA_PROMPT = """SYSTEM PROMPT — LAYER 1: PDF / KNOWLEDGE BASE ANSWERING (IVIA)

You are IVIA, a helpful assistant chatbot. In this step, you answer the user's question using the content retrieved from the PDF knowledge base and qa_knowledge.json.

You are given:
1. USER_QUESTION: {question}
2. RETRIEVED_CHUNKS (from all indexed PDFs and Q&A knowledge base):
{context}

RULES:
1. Check every retrieved chunk from every source PDF — don't favor one file over another.
2. If the answer is present or can be reasonably inferred from the chunks, write the answer the way IVIA would say it in chat: natural, conversational, directly addressing the question.
   - Underlying facts (names, numbers, dates, roles) must be copied exactly as they appear in the source — only the phrasing/wrapper is conversational.
   Examples:
     User: "who is the CM of kerala"
     Good: "The current Chief Minister of Kerala is V. D. Satheesan."
     Bad:  "V. D. Satheesan" or "Answer: V. D. Satheesan (source: pdf)"
3. If multiple chunks are relevant, synthesize them into one coherent, natural reply. Don't add anything not present in the chunks.
4. Give the user the benefit of the doubt — even if the match is partial, provide what you found and note any limitation briefly (e.g. "Based on the documents I have, ...").
5. ONLY if the provided context contains absolutely NO useful information related to the question at all, respond with exactly:
NO_PDF_ANSWER
Do NOT apologize, do NOT say "I'm sorry", do NOT say "I was unable to find". Reply ONLY with: NO_PDF_ANSWER

ANSWER:"""

# Layer 2: Ollama Reasoning & Knowledge Check Prompt
OLLAMA_KNOWLEDGE_PROMPT = """You are a knowledgeable reasoning assistant evaluating whether you can answer a user's question accurately using your general knowledge without needing live internet search.

INSTRUCTIONS:
1. If the question is about well-established facts, geography, world capitals, science, history, definitions, mathematics, or established knowledge that you know with high certainty, provide a direct, factual, and concise answer.
2. ONLY if the question requires live real-time information, breaking news, today's weather/stock/scores, rapidly changing current events, or if you are genuinely uncertain and would have to guess, respond EXACTLY with:
NEEDS_WEB_SEARCH

USER QUESTION:
{question}

ANSWER:"""

# Layer 4: Scraped Web Content Synthesis Prompt
WEB_SCRAPING_PROMPT = """SYSTEM / TASK PROMPT — Layer 4: Web Scraping Answer Synthesis
================================================================

You are the answer synthesizer for a web-scraping fallback layer in a RAG
pipeline. You will be given:
1. The user's question.
2. Raw scraped text content from one or more web pages (already fetched
   via Google/DuckDuckGo search results).

Your job is to extract the best possible answer from this scraped content
and assign a confidence score.

STRICT BEHAVIOR RULES:

1. BIAS TOWARD ANSWERING. Your default assumption is that the scraped
   content DOES contain a usable answer, even if it is phrased
   differently, split across multiple paragraphs, or only partially
   matches the question. Synthesize and answer whenever there is any
   reasonable, defensible basis in the text.

2. CONFIDENCE SCORING — use this scale strictly:
   - 0.90–1.00 → The scraped text directly and explicitly answers the
     question.
   - 0.80–0.89 → The answer can be reasonably inferred/combined from
     the scraped text, even if not stated in one exact sentence.
   - 0.75–0.79 → The scraped text is topically relevant and gives a
     partial or indirect answer that a careful reader would accept.
   - Below 0.75 (RARE — use only when ALL of the following are true):
       a) The scraped content is empty, blocked, or a login/error page.
       b) The scraped content is completely unrelated to the topic of
          the question (not just missing one detail).
       c) There is no reasonable inference path from the text to any
          answer at all.

3. DO NOT default to low confidence just because the answer isn't
   phrased as a clean one-liner. Extract, summarize, and commit to an
   answer using the best available evidence in the scraped pages.

4. If multiple scraped pages are provided, combine information across
   them before concluding the content is insufficient — do not judge
   each page in isolation and give up early.

5. Only return confidence < 0.75 as an absolute last resort. This
   pipeline treats a fallback to the next layer as expensive and
   undesirable, so err strongly on the side of synthesizing an answer
   from what was scraped.

OUTPUT FORMAT (JSON only, no extra text):
{{
  "answer": "<synthesized answer in plain language>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<one short sentence on why this confidence was chosen>"
}}

Question: {user_question}

Scraped Content:
{scraped_text}
"""

# Layer 5: Tavily Live Web Search Prompt
WEB_SEARCH_PROMPT = """You are an intelligent real-time Question Answering chatbot operating in ONLINE MODE.

Your goal is to provide accurate, up-to-date answers based on live web search results from Tavily.

LIVE WEB SEARCH CONTEXT:
{web_context}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Answer the question using the live web search results provided above.
2. Be concise, factual, and clear.
3. Do not invent or assume facts not present in the live search results.

ANSWER:"""

# Single prompt template used by offline modes (Pre-trained, Direct PDF, RAG).
SYSTEM_PROMPT = """You are an intelligent real-time Question Answering chatbot.

Your goal is to understand the MEANING of the user's question,
not just match exact words.

You have three knowledge sources:
1. Question-Answer Knowledge Base
2. Uploaded documents / PDF
3. Llama 3.2 general knowledge

IMPORTANT RULES:

1. UNDERSTAND THE QUESTION
Always understand what the user is actually asking.
Do not blindly accept an incorrect title, assumption, or terminology.

2. CORRECT INCORRECT QUESTIONS
If the user's question contains an incorrect assumption,
politely correct it and then provide the correct information.

Example:
User: "Who is the CM of India?"
Correct response: "India does not have a single Chief Minister (CM). The office of Chief Minister exists for individual states and some Union Territories. At the national level, India has a Prime Minister."
Do NOT answer with the name of a state Chief Minister.

3. STATE VS COUNTRY
India is a country. A country has a Prime Minister at the national level.
Indian states have Chief Ministers.
Therefore:
"CM of India" → Explain that India has no single CM.
"CM of Kerala" → Answer the Chief Minister of Kerala.
"CM of Tamil Nadu" → Answer the Chief Minister of Tamil Nadu.

4. UPLOADED PDF DOCUMENT KNOWLEDGE
In RAG mode, the uploaded PDF / document context contains the most up-to-date factual information. Always give highest priority to the uploaded PDF document context when generating the answer.

5. QUESTION-ANSWER KNOWLEDGE BASE
Use the Q&A knowledge base as supplementary context. If there is any discrepancy or conflict between the Q&A knowledge base and the uploaded PDF context, ALWAYS rely on the uploaded PDF document context.

6. GENERAL KNOWLEDGE
If the answer is not available in the uploaded PDF context or Q&A knowledge base, use Llama 3.2 general knowledge.
Do not say: "The answer is not in the PDF." The PDF is only one source of knowledge.

7. DO NOT HALLUCINATE
Never invent names, dates, positions, statistics, or facts.
If you are not confident, clearly say that you do not have enough reliable information.

8. YEAR-AWARE QUESTIONS
If the user asks: "Who was the CM of Kerala in 2023?", answer for 2023. Do not automatically give the current CM.

9. CURRENT / REAL-TIME QUESTIONS
If the user asks: "Who is the current CM?", use the latest information available in the chatbot's knowledge base.

10. RESPONSE STYLE
Give a direct answer first. Then give a short explanation when necessary. Do not unnecessarily repeat the question.

QUESTION:
{question}

Q&A KNOWLEDGE:
{qa_context}

DOCUMENT/PDF CONTEXT:
{context}

Now understand the question and provide the most accurate answer."""

# Legacy aliases
RAG_PROMPT_TEMPLATE = SYSTEM_PROMPT

RAG_PROMPT = PromptTemplate(
    template=SYSTEM_PROMPT,
    input_variables=["context", "question", "qa_context"]
)


# ─────────────────────────────────────────────────────────────────────────────
# LLM Loaders
# ─────────────────────────────────────────────────────────────────────────────

# Cache dict to prevent creating duplicate LLM instances across Streamlit reruns
_LLM_CACHE = {}


def get_llm(
    backend: str = None,
    model_name: str = None,
    temperature: float = 0.0,
    num_ctx: int = 2048
) -> Any:
    """
    Initializes and returns the appropriate LangChain LLM instance.

    Supported backends:
      - "huggingface"  → Local HuggingFace pipeline (no API key needed)
      - "ollama"       → Local Ollama server (must be running at localhost:11434)
      - "openai"       → OpenAI ChatCompletion API (requires OPENAI_API_KEY)

    Args:
        backend: LLM provider. Reads DEFAULT_LLM_BACKEND env var if None.
                 Defaults to "huggingface".
        model_name: Model identifier. Reads the corresponding env var if None.
        temperature: Generation temperature. 0.0 = deterministic answers.

    Returns:
        LangChain-compatible LLM or ChatModel instance.

    Raises:
        ValueError: If backend is unsupported or OpenAI API key is missing.
        ImportError: If required packages for the backend are not installed.
    """
    # Resolve backend from env var if not specified
    if backend is None:
        backend = os.getenv("DEFAULT_LLM_BACKEND", "huggingface")

    backend = backend.lower().strip()

    # ── HuggingFace Local Pipeline ─────────────────────────────────────────
    if backend == "huggingface":
        if model_name is None:
            model_name = os.getenv("DEFAULT_HF_MODEL", "google/flan-t5-base")

        print(f"[chatbot] Loading HuggingFace pipeline: '{model_name}'...")

        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
            import torch
        except ImportError:
            raise ImportError(
                "The 'transformers' and 'torch' packages are required for HuggingFace backend. "
                "Install with: pip install transformers torch"
            )

        try:
            from langchain_huggingface import HuggingFacePipeline
        except ImportError:
            from langchain_community.llms import HuggingFacePipeline

        device = 0 if torch.cuda.is_available() else -1

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=True)
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    device=device,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0
                )
            except Exception as e:
                print(f"[chatbot] Seq2Seq load failed ({e}), attempting CausalLM...")
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    device=device,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    return_full_text=False
                )
        except Exception as ex:
            raise RuntimeError(f"Could not load HuggingFace model '{model_name}': {ex}")

        print(f"[chatbot] HuggingFace pipeline ready on device {'GPU' if device == 0 else 'CPU'}.")
        return HuggingFacePipeline(pipeline=pipe)

    # ── Ollama (local server) ──────────────────────────────────────────────
    elif backend == "ollama":
        if model_name is None:
            model_name = os.getenv("DEFAULT_OLLAMA_MODEL", "llama3.2:latest")

        cache_key = (backend, model_name, temperature, num_ctx)
        if cache_key in _LLM_CACHE:
            print(f"[chatbot] Returning cached ChatOllama instance ({model_name}, num_ctx={num_ctx})...")
            return _LLM_CACHE[cache_key]

        print(f"[chatbot] Connecting to Ollama model: '{model_name}' (num_ctx={num_ctx})...")

        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "The 'langchain-ollama' package is required. "
                "Install with: pip install langchain-ollama"
            )

        llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            num_ctx=num_ctx
        )
        _LLM_CACHE[cache_key] = llm
        return llm

    # ── OpenAI API ────────────────────────────────────────────────────────
    elif backend == "openai":
        if model_name is None:
            model_name = os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o-mini")

        print(f"[chatbot] Initializing OpenAI model: '{model_name}'...")

        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "The 'langchain-openai' package is required. "
                "Install with: pip install langchain-openai openai"
            )

        # Read API key from environment (set via .env or UI input)
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError(
                "OPENAI_API_KEY is not set. Please add it to your .env file "
                "or enter it in the sidebar settings."
            )

        return ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=api_key
        )

    else:
        raise ValueError(
            f"Unsupported backend: '{backend}'. "
            "Choose from: 'huggingface', 'ollama', 'openai'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Response Cleaning Helper
# ─────────────────────────────────────────────────────────────────────────────

def _clean_llm_response(response, prompt: str) -> str:
    """
    Extracts and cleans the LLM response text.

    Handles:
      - ChatModel responses (AIMessage with .content attribute)
      - Plain string responses (HuggingFacePipeline)
      - Prompt echo stripping (when pipeline returns input + output)
    """
    if hasattr(response, "content"):
        text = response.content.strip()
    else:
        text = str(response).strip()

    # Strip echoed prompt if the model returned input + output
    if text.startswith(prompt.strip()):
        text = text[len(prompt.strip()):].strip()

    # If 'Source:' is present in text, extract starting from 'Source:'
    if "Source:" in text:
        idx = text.find("Source:")
        text = text[idx:].strip()
    # Otherwise handle partial echo — strip everything before "Answer:" if present
    elif "Answer:" in text:
        parts = text.rsplit("Answer:", 1)
        if len(parts) == 2 and parts[1].strip():
            text = parts[1].strip()

    return text if text else "I'm sorry, I couldn't generate an answer. Please try rephrasing your question."


# ─────────────────────────────────────────────────────────────────────────────
# Method 2 — RAG QA Chain
# ─────────────────────────────────────────────────────────────────────────────

def get_qa_chain(llm: Any, retriever: Any) -> RetrievalQA:
    """
    Creates a LangChain RetrievalQA chain for Method 2 (RAG pipeline).

    The chain:
      1. Receives a user question
      2. Retrieves relevant chunks from the vector store via the retriever
      3. Passes the chunks + question to the LLM using the RAG prompt
      4. Returns the generated answer and source documents

    Args:
        llm: LangChain LLM or ChatModel instance.
        retriever: LangChain retriever from a vector store (Chroma or FAISS).

    Returns:
        RetrievalQA chain configured with 'stuff' chain type and custom prompt.
    """
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",          # 'stuff' = concatenate all retrieved chunks into context
        retriever=retriever,
        return_source_documents=True, # Include matched chunks in response for citation display
        chain_type_kwargs={"prompt": RAG_PROMPT}
    )


# ─────────────────────────────────────────────────────────────────────────────
# Method 1 — Direct Text QA
# ─────────────────────────────────────────────────────────────────────────────

def format_error_message(e: Exception) -> str:
    """Formats raw LLM exceptions into clear, actionable user messages."""
    err_str = str(e)
    err_lower = err_str.lower()

    if "out-of-memory" in err_lower or "cuda_host" in err_lower or "failed to allocate" in err_lower:
        return (
            "⚠️ **Out of Memory Error (Ollama/GPU)**\n\n"
            "The LLM server ran out of memory while allocating buffers for model inference.\n\n"
            "**How to Fix:**\n"
            "1. In the **⚙️ Settings** sidebar on the left, switch the **Provider** to **`huggingface`** (runs lightweight models on CPU).\n"
            "2. If using Ollama, close other GPU-intensive applications or switch to a smaller model.\n"
            "3. Restart your Ollama server."
        )
    elif "api_key" in err_lower or "openai_api_key" in err_lower:
        return (
            "⚠️ **OpenAI API Key Missing**\n\n"
            "Please provide a valid `OPENAI_API_KEY` in the **⚙️ Settings** sidebar."
        )
    elif "connection" in err_lower or "refused" in err_lower or "11434" in err_lower:
        return (
            "⚠️ **Ollama Server Connection Failed**\n\n"
            "Could not connect to local Ollama server at `localhost:11434`.\n\n"
            "**How to Fix:**\n"
            "1. Ensure Ollama is running (`ollama serve`).\n"
            "2. Or switch provider to **`huggingface`** in the **⚙️ Settings** sidebar."
        )

    return f"An error occurred while generating the answer: {err_str}"


def invoke_llm_with_oom_retry(
    prompt_or_llm: Any = None,
    prompt: str = None,
    question: str = "",
    qa_context: str = "",
    context: str = "",
    backend: str = "ollama",
    model_name: str = "llama3.2:latest",
    llm: Any = None
) -> str:
    """
    Invokes LLM with fallback retry on CUDA / Out-Of-Memory (OOM) errors using num_ctx=1024.

    Flexibly supports all call styles:
      - invoke_llm_with_oom_retry(prompt)
      - invoke_llm_with_oom_retry(llm, prompt, ...)
      - invoke_llm_with_oom_retry(llm=llm, prompt=prompt, ...)
    """
    if llm is not None:
        actual_llm = llm
        actual_prompt = prompt or (prompt_or_llm if isinstance(prompt_or_llm, str) else "")
    elif isinstance(prompt_or_llm, str) and prompt is None:
        actual_prompt = prompt_or_llm
        actual_llm = get_llm(backend=backend, model_name=model_name, num_ctx=2048)
    elif hasattr(prompt_or_llm, "invoke") and isinstance(prompt, str):
        actual_llm = prompt_or_llm
        actual_prompt = prompt
    else:
        actual_prompt = str(prompt or prompt_or_llm or "")
        actual_llm = get_llm(backend=backend, model_name=model_name, num_ctx=2048)

    try:
        response = actual_llm.invoke(actual_prompt)
        return _clean_llm_response(response, actual_prompt)
    except Exception as e:
        error_text = str(e).lower()
        is_oom = (
            "out of memory" in error_text
            or "out-of-memory" in error_text
            or "cuda" in error_text
            or "cudahost" in error_text
            or "failed to allocate" in error_text
            or "status code: 500" in error_text
            or "buffer" in error_text
        )

        if is_oom and backend == "ollama":
            msg = "⚠️ The local Llama 3.2 model does not have enough available memory. The system is reducing the context size and retrying."
            print(f"[chatbot] {msg}")

            try:
                from langchain_ollama import ChatOllama
                fallback_llm = ChatOllama(
                    model=model_name,
                    temperature=0,
                    num_ctx=1024
                )
                truncated_context = context[:800] if context else "No document context available."
                fallback_prompt = SYSTEM_PROMPT.format(
                    question=question or "Question",
                    qa_context=qa_context[:500] if qa_context else "No relevant Q&A entries found.",
                    context=truncated_context
                )
                response = fallback_llm.invoke(fallback_prompt)
                retry_answer = _clean_llm_response(response, fallback_prompt)
                return f"{msg}\n\n{retry_answer}"
            except Exception as retry_error:
                retry_text = str(retry_error).lower()
                if any(k in retry_text for k in ["out of memory", "cuda", "failed to allocate", "buffer"]):
                    return (
                        "⚠️ The local Llama 3.2 model does not have enough available memory. "
                        "Please close other GPU/AI applications or reduce the model context."
                    )
                raise RuntimeError(
                    "Llama 3.2 could not run because there is not enough available system/GPU memory."
                ) from retry_error

        # Non-OOM Python errors are re-raised for debugging (Requirement 14)
        raise e


def answer_from_text(
    llm: Any,
    document_text: str,
    question: str,
    qa_context: str = "No relevant Q&A entries found.",
    max_context_chars: int = 1500,
    backend: str = "ollama",
    model_name: str = "llama3.2:latest"
) -> str:
    """
    Answers a user question using directly extracted PDF text (Method 1).

    Truncates text to `max_context_chars` (default 1500) to keep context small for Llama 3.2.
    """
    if len(document_text) > max_context_chars:
        truncated_text = document_text[:max_context_chars]
        truncated_text += f"\n\n[... Document truncated at {max_context_chars} characters ...]"
    else:
        truncated_text = document_text

    prompt = SYSTEM_PROMPT.format(
        question=question,
        qa_context=qa_context,
        context=truncated_text
    )

    return invoke_llm_with_oom_retry(
        llm=llm,
        prompt=prompt,
        question=question,
        qa_context=qa_context,
        context=truncated_text,
        backend=backend,
        model_name=model_name
    )


def answer_pretrained(
    llm: Any,
    question: str,
    qa_context: str = "No relevant Q&A entries found.",
    backend: str = "ollama",
    model_name: str = "llama3.2:latest"
) -> str:
    """
    Answers a user question using the LLM's pre-trained knowledge.
    """
    prompt = SYSTEM_PROMPT.format(
        question=question,
        qa_context=qa_context,
        context="No document context available."
    )

    return invoke_llm_with_oom_retry(
        llm=llm,
        prompt=prompt,
        question=question,
        qa_context=qa_context,
        context="",
        backend=backend,
        model_name=model_name
    )


# ─────────────────────────────────────────────────────────────────────────────
# Layer-Specific Evaluators for Sequential Fallback Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pdf_layer(
    llm: Any,
    question: str,
    pdf_context: str,
    qa_context: str = "",
    backend: str = "ollama",
    model_name: str = "llama3.2:latest"
) -> tuple[Optional[str], bool]:
    """
    Evaluates whether the retrieved PDF context reliably answers the question (Layer 1).

    Returns:
        (answer, True) if accepted, or (None, False) if insufficient / NO_PDF_ANSWER.
    """
    combined_context_parts = []
    if qa_context and qa_context.strip():
        combined_context_parts.append(f"--- Q&A KNOWLEDGE BASE ---\n{qa_context.strip()}")
    if pdf_context and pdf_context.strip():
        combined_context_parts.append(f"--- RETRIEVED PDF CHUNKS ---\n{pdf_context.strip()}")

    full_context = "\n\n".join(combined_context_parts)
    if not full_context:
        return None, False

    prompt = PDF_QA_PROMPT.format(
        question=question,
        context=full_context
    )

    raw_response = invoke_llm_with_oom_retry(
        llm=llm,
        prompt=prompt,
        question=question,
        context=full_context,
        backend=backend,
        model_name=model_name
    )

    cleaned = raw_response.strip()
    if not cleaned or "NO_PDF_ANSWER" in cleaned:
        return None, False

    lower_ans = cleaned.lower()

    # Comprehensive refusal / lack-of-information patterns
    refusal_patterns = [
        r"\bno_pdf_answer\b",
        r"\bno\s+pdf\s+answer\b",
        r"\b(?:i['’]?m\s+)?sorry\b.*?\b(?:don['’]?t\s+have|do\s+not\s+have|unable|cannot|no\s+information|insufficient|not\s+enough)\b",
        r"\b(?:i\s+)?apologize\b.*?\b(?:don['’]?t\s+have|do\s+not\s+have|unable|cannot|no\s+information|insufficient|not\s+enough)\b",
        r"\b(?:don['’]?t|do\s+not|doesn['’]?t|does\s+not)\s+have\s+(?:enough|sufficient|any|relevant)?\s*information\b",
        r"\b(?:not\s+enough|insufficient|lack\s+of|no\s+relevant|no)\s+information\b",
        r"\b(?:unable\s+to|cannot|can\s+not|could\s+not|can['’]?t)\s+(?:answer|find|provide|locate|determine)\b",
        r"\b(?:not\s+found|not\s+mentioned|not\s+provided|not\s+available|not\s+stated|not\s+present|no\s+mention)\s+(?:in|from)\s+(?:the\s+)?(?:document|pdf|context|chunks|text|provided)\b",
        r"\b(?:context|document|pdf|text)\s+does\s+not\s+(?:contain|mention|provide|state|have)\b",
        r"\bdoes\s+not\s+(?:contain|mention|provide|state)\s+(?:any|the|enough|information|details)\b",
    ]

    for pat in refusal_patterns:
        if re.search(pat, lower_ans):
            return None, False

    # Hard-reject if response contains clear lack-of-context phrases
    short_evasions = [
        "unable to find", "could not find", "cannot find", "not available in",
        "not provided in", "no information", "not mentioned", "not enough information",
        "i don't have enough information", "i do not have enough information",
        "don't have enough information", "do not have enough information",
        "don't have information", "do not have information", "no mention of"
    ]
    if any(p in lower_ans for p in short_evasions) and len(cleaned) < 250:
        return None, False

    return cleaned, True


def evaluate_ollama_layer(
    llm: Any,
    question: str,
    partial_context: str = "",
    backend: str = "ollama",
    model_name: str = "llama3.2:latest"
) -> tuple[Optional[str], bool]:
    """
    Evaluates whether Ollama can answer with high confidence without live web search (Layer 2).

    Returns:
        (answer, True) if confident, or (None, False) if NEEDS_WEB_SEARCH / uncertain.
    """
    prompt = OLLAMA_KNOWLEDGE_PROMPT.format(
        question=question
    )

    raw_response = invoke_llm_with_oom_retry(
        llm=llm,
        prompt=prompt,
        question=question,
        context="",
        backend=backend,
        model_name=model_name
    )

    cleaned = raw_response.strip()
    if not cleaned or "NEEDS_WEB_SEARCH" in cleaned:
        return None, False

    lower_ans = cleaned.lower()

    # Check for uncertainty, future events, real-time data needs
    uncertainty_patterns = [
        r"\bneeds_web_search\b",
        r"\bneeds\s+web\s+search\b",
        r"\bneed(?:s)?\s+(?:live\s+)?web\s+search\b",
        r"\bi\s+don['’]?t\s+know\b",
        r"\bi\s+do\s+not\s+know\b",
        r"\b(?:don['’]?t|do\s+not|doesn['’]?t|does\s+not)\s+have\s+(?:enough|sufficient|any|real-time|current|up-to-date)?\s*information\b",
        r"\brequires?\s+(?:live|real-time|current|recent|updated)\s+information\b",
        r"\bknowledge\s+cutoff\b",
        r"\bas\s+an\s+ai\b",
        r"\bhas\s+not\s+(?:taken\s+place|occurred|happened|been\s+held|been\s+played|finished)\s+yet\b",
        r"\byet\s+to\s+(?:take\s+place|occur|happen|be\s+held|be\s+played)\b",
        r"\bscheduled\s+to\s+(?:take\s+place|be\s+held|happen)\b",
        r"\b(?:cannot|can\s+not|unable\s+to)\s+predict\s+the\s+future\b",
        r"\bfuture\s+event\b",
    ]

    for pat in uncertainty_patterns:
        if re.search(pat, lower_ans):
            return None, False

    if any(p in lower_ans for p in ["i don't know", "i do not know", "need live web search", "requires current information", "as an ai", "my knowledge cutoff", "cannot predict", "don't have information", "do not have information"]) and len(cleaned) < 200:
        return None, False

    return cleaned, True


def evaluate_scraping_layer(
    llm: Any,
    question: str,
    scraped_context: str,
    backend: str = "ollama",
    model_name: str = "llama3.2:latest",
    min_confidence: float = 0.75
) -> tuple[Optional[str], bool, float]:
    """
    Evaluates and synthesizes an answer from scraped web pages (Layer 4) using JSON output format.

    Returns:
        (answer, True, confidence) if accepted (confidence >= min_confidence),
        or (None, False, confidence) if insufficient/below threshold.
    """
    if not scraped_context or not scraped_context.strip():
        return None, False, 0.0

    prompt = WEB_SCRAPING_PROMPT.format(
        user_question=question,
        scraped_text=scraped_context
    )

    raw_response = invoke_llm_with_oom_retry(
        llm=llm,
        prompt=prompt,
        question=question,
        context=scraped_context,
        backend=backend,
        model_name=model_name
    )

    cleaned = raw_response.strip()
    if not cleaned:
        return None, False, 0.0

    parsed_json = None

    # Attempt 1: Direct JSON parsing
    try:
        parsed_json = json.loads(cleaned)
    except Exception:
        pass

    # Attempt 2: Extract JSON from markdown code block or curly braces
    if not parsed_json or not isinstance(parsed_json, dict):
        match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", cleaned, re.IGNORECASE)
        if match:
            try:
                parsed_json = json.loads(match.group(1))
            except Exception:
                pass

    if not parsed_json or not isinstance(parsed_json, dict):
        match = re.search(r"(\{[\s\S]*\})", cleaned)
        if match:
            try:
                parsed_json = json.loads(match.group(1))
            except Exception:
                pass

    # Process parsed JSON
    if isinstance(parsed_json, dict):
        answer = str(parsed_json.get("answer", "")).strip()
        try:
            confidence = float(parsed_json.get("confidence", 0.0))
        except (ValueError, TypeError):
            confidence = 0.0

        if answer and confidence >= min_confidence:
            return answer, True, confidence
        elif answer and confidence < min_confidence:
            print(f"[chatbot] Layer 4 synthesized answer rejected: confidence {confidence:.2f} < {min_confidence:.2f}")
            return None, False, confidence
        else:
            return None, False, confidence

    # Fallback if model returned plain text despite JSON instruction
    lower_ans = cleaned.lower()
    if "insufficient" in lower_ans or "cannot answer" in lower_ans or len(cleaned) < 30:
        return None, False, 0.0

    # If it returned a direct, substantial text answer, accept with baseline confidence
    return cleaned, True, min_confidence


def evaluate_tavily_layer(
    llm: Any,
    question: str,
    tavily_context: str,
    backend: str = "ollama",
    model_name: str = "llama3.2:latest"
) -> tuple[Optional[str], bool]:
    """
    Synthesizes answer from Tavily search results (Layer 5).

    Returns:
        (answer, True) if generated, or (None, False) if empty.
    """
    if not tavily_context or not tavily_context.strip() or tavily_context == "No live web results found.":
        return None, False

    prompt = WEB_SEARCH_PROMPT.format(
        question=question,
        web_context=tavily_context
    )

    raw_response = invoke_llm_with_oom_retry(
        llm=llm,
        prompt=prompt,
        question=question,
        context=tavily_context,
        backend=backend,
        model_name=model_name
    )

    cleaned = raw_response.strip()
    if not cleaned:
        return None, False

    return cleaned, True



