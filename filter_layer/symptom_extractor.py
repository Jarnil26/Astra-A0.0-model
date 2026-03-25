"""
filter_layer/symptom_extractor.py
Extracts clinical symptom terms from normalized English text using
keyword matching and fuzzy matching.
"""

import re
import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in symptom list (augmented from symptoms_list.txt at runtime)
# ---------------------------------------------------------------------------
BASE_SYMPTOMS = [
    "fever", "high fever", "low grade fever", "chills",
    "headache", "migraine", "severe headache",
    "cough", "dry cough", "productive cough", "whooping cough",
    "cold", "runny nose", "nasal congestion", "sneezing",
    "sore throat", "throat pain", "hoarseness",
    "body ache", "muscle pain", "joint pain", "back pain",
    "chest pain", "chest tightness", "chest pressure",
    "breathing difficulty", "shortness of breath", "wheezing",
    "nausea", "vomiting", "diarrhea", "constipation",
    "stomach pain", "abdominal pain", "bloating", "acidity",
    "fatigue", "weakness", "dizziness", "fainting", "unconsciousness",
    "palpitations", "irregular heartbeat",
    "skin rash", "itching", "hives", "redness", "swelling",
    "eye pain", "eye irritation", "blurred vision", "red eyes",
    "ear pain", "hearing loss",
    "loss of appetite", "weight loss", "excessive thirst",
    "frequent urination", "painful urination",
    "anxiety", "insomnia", "depression",
    "numbness", "tingling", "tremors", "seizures", "confusion",
    "jaundice", "pale skin", "yellow skin", "dark urine",
    "hair loss", "dry skin", "excessive hunger",
]

_symptoms_loaded = False
_symptom_list: List[str] = []


def _load_symptoms(symptoms_file: Optional[str] = None):
    """Load and merge symptom vocabulary."""
    global _symptom_list, _symptoms_loaded
    if _symptoms_loaded:
        return

    combined = list(BASE_SYMPTOMS)

    if symptoms_file and os.path.exists(symptoms_file):
        try:
            with open(symptoms_file, "r", encoding="utf-8") as f:
                extra = [ln.strip().lower() for ln in f if ln.strip()]
                combined = list(set(combined + extra))
            logger.debug("Loaded %d symptoms from file.", len(combined))
        except Exception as e:
            logger.warning("Could not load symptoms file: %s", e)

    # Sort longest first so multi-word symptoms match before single words
    _symptom_list = sorted(set(combined), key=len, reverse=True)
    _symptoms_loaded = True


def extract_symptoms(text: str, symptoms_file: Optional[str] = None) -> List[str]:
    """
    Extract symptoms from normalized English text.

    Strategy:
    1. Direct substring match (longest-first to avoid partial overlaps)
    2. Fuzzy token matching for single-word symptoms not caught in step 1

    Args:
        text: Normalized English text
        symptoms_file: Optional path to symptoms_list.txt

    Returns:
        Deduplicated list of detected symptoms in order of appearance
    """
    _load_symptoms(symptoms_file)

    if not text:
        return []

    text_lower = text.lower()
    found: List[str] = []
    consumed = set()  # Track character positions already claimed by a match

    # --- Pass 1: Direct substring matching (multi-word aware) ---
    for symptom in _symptom_list:
        pattern = r'\b' + re.escape(symptom) + r'\b'
        for match in re.finditer(pattern, text_lower):
            start, end = match.start(), match.end()
            # Check that this span doesn't overlap with an already-matched span
            positions = set(range(start, end))
            if positions.isdisjoint(consumed):
                found.append(symptom)
                consumed.update(positions)
                logger.debug("Extracted symptom (direct): '%s'", symptom)
                break  # Don't double-count the same symptom

    # --- Pass 2: Fuzzy matching on remaining uncovered tokens ---
    try:
        from rapidfuzz import process, fuzz
        # Identify uncovered tokens
        remaining_text = list(text_lower)
        for pos in consumed:
            if pos < len(remaining_text):
                remaining_text[pos] = ' '
        remaining = ''.join(remaining_text)

        tokens = [t for t in remaining.split() if len(t) >= 4]
        single_symptoms = [s for s in _symptom_list if ' ' not in s]

        for token in tokens:
            match = process.extractOne(
                token,
                single_symptoms,
                scorer=fuzz.WRatio,
                score_cutoff=82
            )
            if match:
                candidate = match[0]
                if candidate not in found:
                    found.append(candidate)
                    logger.debug("Extracted symptom (fuzzy): '%s' from token '%s'", candidate, token)
    except ImportError:
        logger.debug("rapidfuzz not available for fuzzy extraction pass.")

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for s in found:
        if s not in seen:
            seen.add(s)
            deduped.append(s)

    return deduped


def reset_symptom_cache():
    """Force reload of symptom list on next call (useful for testing)."""
    global _symptoms_loaded
    _symptoms_loaded = False
