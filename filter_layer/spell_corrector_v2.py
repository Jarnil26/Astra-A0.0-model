"""
filter_layer/spell_corrector_v2.py
Production-safe spell corrector for Filter Layer v2.

Critical safety rules:
  1. Only correct tokens NOT already matched by phrase_matcher (skip_tokens set)
  2. Threshold ≥ 90 (strict — avoids corruption)
  3. Only single tokens, NEVER multi-word phrases
  4. Only correct if token is NOT already a valid English symptom
  5. Skip tokens shorter than 4 characters (too ambiguous)
  6. Returns (corrected_text, Dict[original→corrected]) for full auditability
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Minimum score to accept a correction (0-100)
SAFE_THRESHOLD = 90

# Tokens that are definitely valid and must never be altered
ENGLISH_SYMPTOM_TOKENS: Set[str] = {
    "fever", "headache", "cough", "cold", "nausea", "vomiting", "diarrhea",
    "constipation", "pain", "ache", "fatigue", "weakness", "dizziness",
    "rash", "itching", "swelling", "anxiety", "insomnia", "stress",
    "jaundice", "anemia", "thyroid", "diabetes",
    "numbness", "tingling", "tremors", "seizures", "unconsciousness",
    "bleeding", "infection", "inflammation",
    # Multi-word (check prefix too)
    "stomach", "chest", "breathing", "shortness", "palpitations",
    "appetite", "urination", "confusion", "memory",
}

# Symptom vocabulary for fuzzy matching (clean clinical terms only)
CLINICAL_VOCAB: List[str] = [
    "fever", "high fever", "chills",
    "headache", "migraine",
    "cough", "dry cough", "productive cough",
    "cold", "runny nose", "nasal congestion", "sneezing",
    "sore throat", "hoarseness",
    "body ache", "muscle pain", "joint pain", "back pain",
    "chest pain", "chest tightness",
    "breathing difficulty", "shortness of breath", "wheezing",
    "nausea", "vomiting", "diarrhea", "constipation",
    "stomach pain", "abdominal pain", "bloating", "acidity",
    "fatigue", "weakness", "dizziness", "fainting", "unconsciousness",
    "palpitations",
    "skin rash", "itching", "redness", "swelling",
    "eye pain", "eye irritation", "blurred vision",
    "ear pain", "hearing loss",
    "loss of appetite", "weight loss", "excessive thirst",
    "frequent urination", "painful urination",
    "anxiety", "insomnia", "depression",
    "numbness", "tingling", "tremors", "seizures", "confusion",
    "jaundice", "pale skin", "dark urine",
    "hair loss", "excessive hunger",
]


def correct_tokens_safe(
    text: str,
    skip_tokens: Optional[Set[str]] = None,
) -> Tuple[str, Dict[str, str]]:
    """
    Apply safe spell correction to individual tokens in text.

    Args:
        text: Text that has already been normalized/reconstructed
        skip_tokens: Set of tokens to skip (already-matched English phrases)

    Returns:
        (corrected_text, {original_token: corrected_token})
    """
    skip_tokens = skip_tokens or set()
    corrections: Dict[str, str] = {}

    try:
        from rapidfuzz import process, fuzz
    except ImportError:
        logger.warning("[v2 spell] rapidfuzz not installed, skipping correction")
        return text, {}

    tokens = text.split()
    corrected_tokens = []

    for token in tokens:
        token_lower = token.lower()

        # Rule 1: Skip already-correct English terms
        if token_lower in ENGLISH_SYMPTOM_TOKENS:
            corrected_tokens.append(token)
            continue

        # Rule 2: Skip tokens that are part of matched phrase output
        if token_lower in skip_tokens:
            corrected_tokens.append(token)
            continue

        # Rule 3: Skip very short tokens (too ambiguous)
        if len(token_lower) < 4:
            corrected_tokens.append(token)
            continue

        # Rule 4: Skip tokens that look like transliterated Indian words
        # (they should have been caught by phrase_matcher; don't corrupt them)
        if _looks_indian(token_lower):
            logger.debug("[v2 spell] skipping Indian token: '%s'", token_lower)
            corrected_tokens.append(token)
            continue

        # Rule 5: Try fuzzy correction with strict threshold
        # Use single-word vocab entries only
        single_word_vocab = [t for t in CLINICAL_VOCAB if ' ' not in t]
        match = process.extractOne(
            token_lower,
            single_word_vocab,
            scorer=fuzz.WRatio,
            score_cutoff=SAFE_THRESHOLD,
        )

        if match:
            corrected = match[0]
            if corrected != token_lower:
                corrections[token_lower] = corrected
                corrected_tokens.append(corrected)
                logger.debug(
                    "[v2 spell] '%s' → '%s' (score: %d)",
                    token_lower, corrected, match[1]
                )
            else:
                corrected_tokens.append(token)
        else:
            corrected_tokens.append(token)

    corrected_text = " ".join(corrected_tokens)
    return corrected_text, corrections


def _looks_indian(token: str) -> bool:
    """
    Heuristic: does this token look like a transliterated Indian word?
    Indian transliterations often contain these patterns.
    """
    # Common transliteration patterns
    indian_patterns = [
        "kh", "gh", "ch", "dh", "bh", "ph", "th",  # aspirated consonants
        "aa", "oo", "ee", "ai", "au",                # long vowels
        "ng", "nk",
    ]
    # If a 4-6 char token contains Indian patterns AND is not English
    if len(token) <= 6 and any(p in token for p in indian_patterns):
        return True

    # Known Indian word markers
    indian_suffixes = ("ke", "ka", "ki", "ne", "me", "pe", "se", "ko",
                       "hai", "he", "che", "chhe", "aur", "ya")
    if any(token.endswith(sfx) for sfx in indian_suffixes):
        return True

    return False
