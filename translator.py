"""
translator.py — English ↔ Malayalam Translation Layer
======================================================
Provides language detection, term preservation masking, and robust translation
between English and Malayalam using `deep-translator`.

Features:
  - Malayalam script detection (Unicode range U+0D00 to U+0D7F)
  - Term preservation: protects technical terms, numbers, URLs, code blocks,
    and document citations from being corrupted during translation
  - Error handling: fallback to original text if network or translation fails
"""

import re
import sys
from typing import Dict, Tuple, Any, Optional

try:
    from deep_translator import GoogleTranslator
    DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    DEEP_TRANSLATOR_AVAILABLE = False
    print("[translator] Warning: 'deep-translator' package not found. Translation will fallback to original text.", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Language Detection
# ─────────────────────────────────────────────────────────────────────────────

# Malayalam Unicode Script Range: U+0D00 to U+0D7F
MALAYALAM_PATTERN = re.compile(r'[\u0D00-\u0D7F]')


def is_malayalam(text: str) -> bool:
    """
    Detects if the given text contains Malayalam script characters.

    Args:
        text: Input string to test.

    Returns:
        True if Malayalam characters are found, False otherwise.
    """
    if not text:
        return False
    return bool(MALAYALAM_PATTERN.search(text))


def detect_language(text: str) -> str:
    """
    Returns 'ml' if Malayalam script is detected in text, else 'en'.

    Args:
        text: Input string to test.

    Returns:
        Language code ('ml' or 'en').
    """
    return "ml" if is_malayalam(text) else "en"


# ─────────────────────────────────────────────────────────────────────────────
# Term Preservation (Masking & Unmasking)
# ─────────────────────────────────────────────────────────────────────────────

# Regular expressions matching terms that MUST be preserved without translation
PROTECTED_PATTERNS = [
    r'Source:\s*[^\n]+',                             # Source citations (e.g., Source: Uploaded PDF)
    r'https?://[^\s]+',                              # URLs
    r'```[\s\S]*?```',                               # Code blocks
    r'`[^`]+`',                                      # Inline code snippets
    r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', # Email addresses
    r'\b[A-Z0-9_]{2,}\b',                             # Technical acronyms (RAG, PDF, LLM, API, GPT, FAISS, etc.)
    r'\b\d+(?:\.\d+)?%?\b',                           # Standalone numbers & percentages (123, 45.6, 100%)
]

COMBINED_PROTECTED_REGEX = re.compile('|'.join(f'({p})' for p in PROTECTED_PATTERNS))


def mask_protected_terms(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Replaces protected elements (URLs, code, technical terms, numbers, citations)
    with placeholder tokens like XYZPH0XYZ before translation.

    Args:
        text: Original text string.

    Returns:
        Tuple of (masked_text, placeholders_dict).
    """
    if not text:
        return text, {}

    placeholders = {}
    counter = 0

    def replacer(match):
        nonlocal counter
        matched_val = match.group(0)
        key = f"XYZPH{counter}XYZ"
        placeholders[key] = matched_val
        counter += 1
        return key

    masked_text = COMBINED_PROTECTED_REGEX.sub(replacer, text)
    return masked_text, placeholders


def unmask_protected_terms(text: str, placeholders: Dict[str, str]) -> str:
    """
    Restores masked placeholder tokens back to their original preserved values.

    Handles potential space insertion by translation engines (e.g. XYZ PH 0 XYZ).

    Args:
        text: Translated text containing placeholder tokens.
        placeholders: Dictionary mapping placeholder tokens to original strings.

    Returns:
        Unmasked string with original technical terms restored.
    """
    if not text or not placeholders:
        return text

    unmasked_text = text

    for key, original_val in placeholders.items():
        # Extract the token index number
        token_num = key.replace("XYZPH", "").replace("XYZ", "")
        
        # Regex to catch loose spacing added by translation services e.g., XYZ PH 0 XYZ
        fuzzy_token_pattern = re.compile(
            r'XYZ\s*PH\s*' + re.escape(token_num) + r'\s*XYZ',
            re.IGNORECASE
        )
        unmasked_text = fuzzy_token_pattern.sub(original_val, unmasked_text)
        
        # Direct string replacement fallback
        unmasked_text = unmasked_text.replace(key, original_val)

    return unmasked_text


# ─────────────────────────────────────────────────────────────────────────────
# LLM Translation System Prompt & Functions
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Translation Request Intent Detection & Direct Request Translation
# ─────────────────────────────────────────────────────────────────────────────

RAG_TRANSLATION_SYSTEM_PROMPT = """You are an English-to-Malayalam translation assistant.

Your ONLY task in translation requests is to translate English text into Malayalam.

Rules:
1. Translate English into natural Malayalam.
2. Output ONLY the Malayalam translation.
3. Do not provide explanations.
4. Do not provide French, Spanish, German, Chinese, or any other language.
5. Do not use PDF information for translation.
6. Do not perform RAG retrieval for simple translation requests.
7. Words such as "translate", "Malayalam", "in Malayalam", and "to Malayalam" are instructions and should not be included in the translated text.
8. Preserve names, numbers, and technical terms when appropriate.

Examples:

Input:
Good morning Malayalam

Output:
സുപ്രഭാതം

Input:
Translate good morning to Malayalam

Output:
സുപ്രഭാതം

Input:
How are you in Malayalam?

Output:
നിങ്ങൾക്ക് സുഖമാണോ?

Input:
I am going to school - translate to Malayalam

Output:
ഞാൻ സ്കൂളിലേക്ക് പോകുന്നു."""


def is_translation_request(query: str) -> bool:
    """
    Detects if the user query is an explicit English-to-Malayalam translation request.

    Examples:
      - "Good morning Malayalam"
      - "Translate good morning to Malayalam"
      - "How are you in Malayalam?"
      - "I am going to school - translate to Malayalam"
    """
    if not query or not query.strip():
        return False
    if is_malayalam(query):
        return False

    q = query.strip().lower()

    if "in malayalam" in q or "to malayalam" in q:
        return True
    if "translate" in q and "malayalam" in q:
        return True
    if q.endswith("malayalam") or q.startswith("malayalam"):
        return True
    if re.search(r'-\s*translate', q):
        return True

    return False


def extract_target_text(query: str) -> str:
    """
    Extracts the core English text to translate by stripping instruction words.
    """
    text = query.strip()
    text = re.sub(r'^(translate|translation of)\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*-\s*translate(\s*to)?(\s*malayalam)?\??$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*(in|to)\s+malayalam\??$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+malayalam\??$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^malayalam\s+', '', text, flags=re.IGNORECASE)
    return text.strip()


def translate_direct_request(query: str, llm: Any = None) -> str:
    """
    Handles direct English-to-Malayalam translation requests, bypassing PDF retrieval.

    Tries LLM-based translation with system prompt first, falling back to GoogleTranslator.
    """
    target_text = extract_target_text(query)
    if not target_text:
        target_text = query

    masked_text, placeholders = mask_protected_terms(target_text)

    # 1. Try LLM translation if LLM is loaded
    if llm is not None:
        try:
            prompt = f"{RAG_TRANSLATION_SYSTEM_PROMPT}\n\nInput:\n{query}\n\nOutput:"
            response = llm.invoke(prompt)

            if hasattr(response, "content"):
                raw_output = str(response.content).strip()
            else:
                raw_output = str(response).strip()

            if raw_output.startswith("Output:"):
                raw_output = raw_output.replace("Output:", "").strip()

            final_output = unmask_protected_terms(raw_output, placeholders)
            if is_malayalam(final_output):
                return final_output
        except Exception as e:
            print(f"[translator] Direct LLM translation warning ({e}), falling back to GoogleTranslator.", file=sys.stderr)

    # 2. Fallback to GoogleTranslator
    if DEEP_TRANSLATOR_AVAILABLE:
        try:
            translator = GoogleTranslator(source="en", target="ml")
            translated = translator.translate(masked_text)
            if translated:
                return unmask_protected_terms(translated, placeholders)
        except Exception as e:
            print(f"[translator] Fallback GoogleTranslator error: {e}", file=sys.stderr)

    return target_text


# ─────────────────────────────────────────────────────────────────────────────
# Main Translation Function
# ─────────────────────────────────────────────────────────────────────────────

def translate_text(
    text: str,
    target_lang: str = "en",
    source_lang: str = "auto",
    llm: Any = None
) -> str:
    """
    Translates text between English and Malayalam with automatic term preservation.

    Supports LLM-powered translation for English -> Malayalam when an LLM is provided,
    with automatic fallback to GoogleTranslator.

    Args:
        text: String to translate.
        target_lang: Target language code ('en' or 'ml').
        source_lang: Source language code ('auto', 'en', or 'ml').
        llm: Optional LLM instance for LLM-powered translation.

    Returns:
        Translated string, or original text if translation fails or is unnecessary.
    """
    if not text or not text.strip():
        return text

    # If target language matches detected input language, return as-is
    if target_lang == "ml" and is_malayalam(text) and source_lang != "en":
        return text
    if target_lang == "en" and not is_malayalam(text) and source_lang != "ml":
        return text

    # Attempt LLM translation first for English -> Malayalam if LLM is supplied
    if target_lang == "ml" and llm is not None and not is_malayalam(text):
        try:
            return translate_with_llm(text, llm)
        except Exception:
            pass

    if not DEEP_TRANSLATOR_AVAILABLE:
        print("[translator] Skipping translation (deep-translator package unavailable).", file=sys.stderr)
        return text

    try:
        # Step 1: Mask protected technical terms and numbers
        masked_text, placeholders = mask_protected_terms(text)

        # Step 2: Perform translation via GoogleTranslator
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated_masked = translator.translate(masked_text)

        if not translated_masked:
            return text

        # Step 3: Unmask preserved terms back to original
        final_text = unmask_protected_terms(translated_masked, placeholders)
        return final_text

    except Exception as e:
        print(f"[translator] Translation error ({source_lang} -> {target_lang}): {e}", file=sys.stderr)
        # Fallback gracefully to original text if error occurs
        return text
