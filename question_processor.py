"""
question_processor.py — Intelligent Question Understanding, Normalization & Role Validation Engine
=================================================================================================
Implements the 7-step Question Processing Pipeline:
  1. Question Normalization (typo, grammar, abbreviation, entity/role cleanup)
  2. Intent Detection (who, count, capital, role_holder, definition, year-aware)
  3. Entity Detection (Country vs State vs UT vs Organization)
  4. Role Detection (Governor, Chief Minister, Prime Minister, President)
  5. Entity + Role Validation (Detect invalid administrative level combinations)
  6. Automatic Concept Correction (Polite explanation of incorrect assumptions)
  7. Year/Date Extraction (e.g. 2023 vs current)
"""

import re
from typing import Dict, Optional, Tuple, Any


# ─────────────────────────────────────────────────────────────────────────────
# Recognized Entities & Administrative Levels
# ─────────────────────────────────────────────────────────────────────────────
COUNTRIES = {
    "india", "indian", "usa", "united states", "uk", "united kingdom", "canada",
    "australia", "china", "japan", "germany", "france", "russia", "pak", "pakistan",
    "bangladesh", "sri lanka", "nepal", "bhutan", "myanmar"
}

INDIAN_STATES_UTS = {
    "kerala", "tamil nadu", "karnataka", "andhra pradesh", "telangana",
    "maharashtra", "goa", "gujarat", "rajasthan", "punjab", "haryana",
    "uttar pradesh", "up", "bihar", "west bengal", "wb", "odisha", "assam",
    "delhi", "puducherry", "pondicherry", "jammu and kashmir", "jk", "ladakh",
    "himachal pradesh", "uttarakhand", "jharkhand", "chhattisgarh",
    "madhya pradesh", "mp", "sikkim", "arunachal pradesh", "nagaland",
    "manipur", "mizoram", "tripura", "meghalaya", "chandigarh",
    "lakshadweep", "andaman and nicobar"
}

# Normalized names for abbreviations/alternate spellings
ENTITY_NORMALIZE = {
    "up": "Uttar Pradesh",
    "wb": "West Bengal",
    "jk": "Jammu and Kashmir",
    "mp": "Madhya Pradesh",
    "pondy": "Puducherry",
    "pondicherry": "Puducherry",
    "bombay": "Maharashtra",
    "madras": "Tamil Nadu",
    "calcutta": "West Bengal",
    "trivandrum": "Kerala",
    "cochin": "Kerala",
    "calicut": "Kerala",
    "indian": "India",
}


# ─────────────────────────────────────────────────────────────────────────────
# Recognized Roles with Typo/Abbreviation Patterns
# ─────────────────────────────────────────────────────────────────────────────
ROLE_PATTERNS = {
    "governor":       [r"\bgovernor\b", r"\bgovner\b", r"\bgoverner\b", r"\bgov\b(?!\s*general)"],
    "chief_minister": [r"\bchief\s*minister\b", r"\bcm\b", r"\bchief\s*minster\b", r"\bchief\s*ministter\b"],
    "prime_minister": [r"\bprime\s*minister\b", r"\bpm\b", r"\bprime\s*minster\b", r"\bprime\s*ministter\b"],
    "president":      [r"\bpresident\b", r"\bprez\b", r"\bpresidant\b"],
    "vice_president": [r"\bvice\s*president\b", r"\bvp\b"],
    "mayor":          [r"\bmayor\b"],
    "mp_member":      [r"\bmember of parliament\b", r"\bmp\b(?!\s*state)"],
    "mla":            [r"\bstate legislat\b", r"\bmla\b", r"\blegislature\b"],
}

# Role → entity_type validity map
# True = valid for that entity_type
ROLE_ENTITY_TYPE_RULES = {
    # role            country  state    ut      other
    "governor":       (False,  True,   True,   False),
    "chief_minister": (False,  True,   True,   False),
    "prime_minister": (True,   False,  False,  False),
    "president":      (True,   False,  False,  False),
    "vice_president": (True,   False,  False,  False),
    "mayor":          (False,  False,  False,  True),   # cities
    "mp_member":      (True,   True,   True,   True),
    "mla":            (False,  True,   True,   False),
}


# ─────────────────────────────────────────────────────────────────────────────
# Typo / Abbreviation Normalizations (applied before entity/role detection)
# ─────────────────────────────────────────────────────────────────────────────
_TYPO_REPLACEMENTS = [
    # Spelling / grammar typos for roles
    (r"\bprime\s*minster\b",   "Prime Minister"),
    (r"\bprime\s*ministter\b", "Prime Minister"),
    (r"\bchief\s*minster\b",   "Chief Minister"),
    (r"\bchief\s*ministter\b", "Chief Minister"),
    (r"\bgovner\b",            "Governor"),
    (r"\bgoverner\b",          "Governor"),
    (r"\bpresidant\b",         "President"),
    # Geographic typos / alternate forms
    (r"\bindian\b",            "India"),         # "governor of Indian" → "India"
    # Grammar fixes for count questions
    (r"\bhow many district\b(?!\s*s)", "how many districts"),
    (r"\bhow many district in\b",      "how many districts are there in"),
    (r"\bhow many districts in\b",     "how many districts are there in"),
    # Abbreviation expansion for common short queries
    (r"\bwho\s+is\s+cm\b(?!\s+of)",   "who is the Chief Minister of"),
    (r"\bwho\s+is\s+cm\s+of\b",       "who is the Chief Minister of"),
    (r"\bwho\s+is\s+pm\b(?!\s+of)",   "who is the Prime Minister of"),
    (r"\bwho\s+is\s+pm\s+of\b",       "who is the Prime Minister of"),
    (r"\bwho\s+governor\b",           "who is the Governor of"),
    (r"\bwho\s+prime\s+minister\b",   "who is the Prime Minister of"),
    (r"\bwho\s+prime\s+minster\b",    "who is the Prime Minister of"),
    (r"\bwho\s+cm\b",                 "who is the Chief Minister of"),
    (r"\bwas\s+cm\b",                 "was the Chief Minister of"),
    (r"\bwas\s+pm\b",                 "was the Prime Minister of"),
]


def normalize_text(text: str) -> str:
    """
    Normalizes spelling, grammar, and common typos in user queries.
    Expands abbreviations, fixes role titles, and corrects common entity misspellings.
    """
    if not text:
        return ""

    t = text.strip()

    # Apply all typo replacements
    for pattern, replacement in _TYPO_REPLACEMENTS:
        t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)

    return t


def extract_year(text: str) -> Optional[int]:
    """Extracts a 4-digit year if specified in the query (e.g. 2023)."""
    match = re.search(r"\b(19\d\d|20\d\d)\b", text)
    if match:
        return int(match.group(1))
    return None


def detect_entity(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Detects the main geographic entity and entity_type (country vs state/UT).

    Detection order:
      1. States/UTs (checked first to correctly classify "Kerala" before "India")
      2. Countries

    Returns:
      (entity_name, entity_type) e.g. ("India", "country") or ("Kerala", "state")
    """
    t_lower = text.lower()

    # Check states first (longer names first to avoid partial matches)
    sorted_states = sorted(INDIAN_STATES_UTS, key=len, reverse=True)
    for state in sorted_states:
        if re.search(r"\b" + re.escape(state) + r"\b", t_lower):
            # Normalize abbreviations
            normalized = ENTITY_NORMALIZE.get(state, None)
            if normalized:
                return normalized, "state"
            # Capitalize nicely
            words = state.split()
            formatted = " ".join(w.capitalize() for w in words)
            return formatted, "state"

    # Check countries
    for country in COUNTRIES:
        if re.search(r"\b" + re.escape(country) + r"\b", t_lower):
            # Map alternate spellings → canonical name
            normalized = ENTITY_NORMALIZE.get(country.lower(), None)
            if normalized:
                return normalized, "country"
            return "India", "country"

    return None, None


def detect_role(text: str) -> Optional[str]:
    """Detects requested political/governmental role in text."""
    t_lower = text.lower()
    for role, patterns in ROLE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, t_lower):
                return role
    return None


def detect_intent(text: str) -> str:
    """
    Detects the high-level intent of the question.

    Intents:
      - role_holder: "who is the CM / PM / Governor..."
      - count:       "how many districts / states..."
      - capital:     "capital of..."
      - definition:  "what is..."
      - historical:  "who was... in [year]"
      - current:     "current / latest / now"
      - general:     fallback
    """
    t_lower = text.lower()

    if re.search(r"\bhow many\b", t_lower):
        return "count"
    if re.search(r"\bcapital\b", t_lower):
        return "capital"
    if re.search(r"\bwhat is\b|\bwhat are\b|\bdefinition\b|\bexplain\b|\bmeaning\b", t_lower):
        return "definition"
    if re.search(r"\bwho was\b|\bwho were\b", t_lower) and extract_year(text):
        return "historical"
    if re.search(r"\bcurrent\b|\blatest\b|\bnow\b|\bpresent\b", t_lower):
        return "current"
    if re.search(r"\bwho is\b|\bwho are\b|\bwho was\b", t_lower):
        return "role_holder"
    return "general"


def validate_entity_role(
    entity: Optional[str],
    entity_type: Optional[str],
    role: Optional[str]
) -> Tuple[bool, Optional[str]]:
    """
    Validates if the requested role is valid for the given entity/administrative level.

    Returns:
      (True,  None)               — valid combination, proceed normally
      (False, correction_text)    — invalid combination, return correction to user
    """
    if not entity or not role:
        return True, None

    # ── Rule 1: Governor + Country (e.g. "Governor of India") ──────────────
    if role == "governor" and entity_type == "country":
        return False, (
            f"{entity} does not have a Governor. Governors are appointed for individual "
            "states and certain Union Territories. At the national level, India has a President "
            "and a Prime Minister."
        )

    # ── Rule 2: Chief Minister + Country (e.g. "CM of India") ──────────────
    if role == "chief_minister" and entity_type == "country":
        return False, (
            f"{entity} does not have a single Chief Minister. Chief Ministers head individual "
            "states and some Union Territories. At the national level, India has a Prime Minister."
        )

    # ── Rule 3: Prime Minister + State (e.g. "Prime Minister of Kerala") ────
    if role == "prime_minister" and entity_type == "state":
        return False, (
            f"{entity} does not have a Prime Minister. {entity} is a state of India and has a "
            "Chief Minister. The Prime Minister is the head of India's central government."
        )

    # ── Rule 4: President + State (e.g. "President of Kerala") ──────────────
    if role == "president" and entity_type == "state":
        return False, (
            f"{entity} does not have a President. {entity} has a Governor as its constitutional "
            "head and a Chief Minister who heads the elected state government."
        )

    # ── Rule 5: Vice President + State ────────────────────────────────────
    if role == "vice_president" and entity_type == "state":
        return False, (
            f"{entity} does not have a Vice President. The Vice President is a national-level "
            "position in India. States have a Governor as the constitutional head."
        )

    return True, None


def analyze_question(raw_question: str) -> Dict[str, Any]:
    """
    Full Question Analysis Pipeline.

    Runs:
      1. normalize_text()          — fix typos, expand abbreviations
      2. detect_entity()           — find state/country entity
      3. detect_role()             — find requested role
      4. extract_year()            — find historical year if any
      5. detect_intent()           — classify question type
      6. validate_entity_role()    — check role vs entity_type validity

    Returns:
      dict with keys:
        - raw_question         (str)
        - normalized_question  (str)
        - entity               (str | None)
        - entity_type          (str | None)  → "country" | "state" | None
        - role                 (str | None)
        - year                 (int | None)
        - intent               (str)
        - is_valid_combination (bool)
        - correction_response  (str | None)  → non-None means return this immediately
    """
    normalized = normalize_text(raw_question)
    entity, entity_type = detect_entity(normalized)
    role = detect_role(normalized)
    year = extract_year(normalized)
    intent = detect_intent(normalized)

    is_valid, correction = validate_entity_role(entity, entity_type, role)

    return {
        "raw_question": raw_question,
        "normalized_question": normalized,
        "entity": entity,
        "entity_type": entity_type,
        "role": role,
        "year": year,
        "intent": intent,
        "is_valid_combination": is_valid,
        "correction_response": correction
    }
