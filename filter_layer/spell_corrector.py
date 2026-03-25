"""
filter_layer/spell_corrector.py
Corrects spelling mistakes in symptom tokens using rapidfuzz.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

# Core symptom vocabulary for fuzzy matching
SYMPTOM_VOCABULARY = [
    "fever", "high fever", "low grade fever", "chills",
    "headache", "cough", "cold", "runny nose", "nasal congestion",
    "sore throat", "body ache", "fatigue", "weakness", "dizziness", "nausea",
    "vomiting", "diarrhea", "constipation", "bloating", "acidity", "acid reflux",
    "stomach pain", "abdominal pain", "chest pain", "chest tightness",
    "breathing difficulty", "shortness of breath", "palpitations",
    "back pain", "joint pain", "muscle pain", "leg pain", "arm pain",
    "skin rash", "itching", "redness", "swelling", "inflammation",
    "eye pain", "eye irritation", "blurred vision",
    "ear pain", "hearing loss", "ringing in ears",
    "anxiety", "depression", "insomnia", "stress",
    "high blood pressure", "low blood pressure",
    "diabetes", "thyroid", "anemia",
    "loss of appetite", "weight loss", "weight gain",
    "urinary infection", "frequent urination", "painful urination",
    "dry skin", "hair loss", "excessive thirst", "excessive hunger",
    "pale skin", "yellow skin", "jaundice", "dark urine",
    "numbness", "tingling", "tremors", "seizures", "unconsciousness",
    "confusion", "memory loss",
]

# Similarity threshold (0–100). Scores below this will NOT be corrected.
CORRECTION_THRESHOLD = 78

_vocab_loaded = False


def _load_custom_vocab(symptoms_file: str = None):
    """Optionally extend vocabulary from the project's symptoms_list.txt."""
    global SYMPTOM_VOCABULARY, _vocab_loaded
    if _vocab_loaded:
        return
    if symptoms_file:
        try:
            with open(symptoms_file, "r", encoding="utf-8") as f:
                extra = [line.strip().lower() for line in f if line.strip()]
                SYMPTOM_VOCABULARY = list(set(SYMPTOM_VOCABULARY + extra))
                logger.debug("Loaded %d extra symptoms from %s", len(extra), symptoms_file)
        except FileNotFoundError:
            logger.warning("Symptoms file not found: %s", symptoms_file)
    _vocab_loaded = True


def correct_token(token: str) -> str:
    """
    Correct a single token against the symptom vocabulary.
    Returns the best match if score >= threshold, else returns the original token.
    """
    if not token or len(token) < 3:
        return token

    # Use a lower threshold for short tokens (≤5 chars) to catch typos like coff→cough
    threshold = CORRECTION_THRESHOLD if len(token) > 5 else 72

    try:
        from rapidfuzz import process, fuzz
        match = process.extractOne(
            token,
            SYMPTOM_VOCABULARY,
            scorer=fuzz.WRatio,
            score_cutoff=threshold
        )
        if match:
            corrected = match[0]
            if corrected != token:
                logger.debug("Spell corrected: '%s' → '%s' (score: %d)", token, corrected, match[1])
            return corrected
    except ImportError:
        logger.warning("rapidfuzz not installed. Skipping spell correction.")

    return token


def correct_text(text: str, symptoms_file: str = None) -> str:
    """
    Correct spelling in a normalized English text string.
    Works on a token-by-token basis to fix individual misspellings.

    Args:
        text: Normalized English text
        symptoms_file: Optional path to extend vocabulary

    Returns:
        Spell-corrected text
    """
    _load_custom_vocab(symptoms_file)

    if not text:
        return ""

    # Split into tokens preserving punctuation context
    tokens = text.split()
    corrected_tokens = []

    i = 0
    while i < len(tokens):
        # Try two-word phrases first
        if i + 1 < len(tokens):
            bigram = tokens[i] + " " + tokens[i + 1]
            try:
                from rapidfuzz import process, fuzz
                match = process.extractOne(
                    bigram,
                    SYMPTOM_VOCABULARY,
                    scorer=fuzz.WRatio,
                    score_cutoff=85
                )
                if match:
                    corrected_tokens.append(match[0])
                    i += 2
                    continue
            except ImportError:
                pass

        corrected_tokens.append(correct_token(tokens[i]))
        i += 1

    return " ".join(corrected_tokens)


def get_vocabulary() -> List[str]:
    """Return the current symptom vocabulary."""
    return list(SYMPTOM_VOCABULARY)
