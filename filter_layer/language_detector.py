"""
filter_layer/language_detector.py
Detects the language of user input.
Supports: English, Hindi, Gujarati, Hinglish (fallback)
"""

import logging

logger = logging.getLogger(__name__)

# Words strongly indicative of Hinglish / transliterated Indian languages
HINGLISH_MARKERS = {
    "mujhe", "mera", "meri", "hai", "hain", "aur", "nahi", "kuch",
    "thoda", "bohot", "bahut", "kal", "aaj", "abhi", "kaafi", "achha",
    "theek", "bhi", "toh", "wala", "wali", "kya", "kaise", "kyun",
    "main", "tum", "aap", "hum", "unhe", "usse", "isko", "usko"
}

HINDI_SYMPTOM_MARKERS = {
    "bukhar", "sar", "dard", "khansi", "jwar", "tav", "pet", "dast",
    "ulti", "ukhad", "jodon", "kamzori", "chakkar", "thakan", "khujli"
}

GUJARATI_SYMPTOM_MARKERS = {
    "mane", "aayo", "che", "tave", "dukhe", "aapo", "nathi", "chhe",
    "thaak", "ubki", "dard", "khansee", "tav"
}


def detect_language(text: str) -> str:
    """
    Detect the language of the given text.

    Returns one of:
        'en'         - English
        'hi'         - Hindi (Devanagari or transliterated)
        'gu'         - Gujarati (transliterated)
        'hinglish'   - Mixed Hindi-English
        'unknown'    - Could not determine (treated as English)
    """
    if not text or not text.strip():
        return "unknown"

    text_lower = text.lower().strip()
    tokens = set(text_lower.split())

    # Check for Devanagari script (native Hindi/Marathi)
    if any('\u0900' <= ch <= '\u097F' for ch in text):
        return "hi"

    # Check for Gujarati script
    if any('\u0A80' <= ch <= '\u0AFF' for ch in text):
        return "gu"

    # Transliterated detection via marker words
    gujarati_hits = len(tokens & GUJARATI_SYMPTOM_MARKERS)
    hindi_hits = len(tokens & HINDI_SYMPTOM_MARKERS)
    hinglish_hits = len(tokens & HINGLISH_MARKERS)

    if gujarati_hits >= 1:
        return "gu"
    if hindi_hits >= 1:
        return "hi"
    if hinglish_hits >= 1:
        return "hinglish"

    # Try langdetect as fallback
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 42
        lang = detect(text)
        if lang in ("hi", "gu", "mr", "pa", "bn", "te", "ta", "kn", "ml"):
            return lang
        return "en"
    except Exception:
        # Short strings often fail langdetect — default to English
        logger.debug("langdetect failed for input: %s", text)
        return "en"


def is_indian_language(lang_code: str) -> bool:
    """Return True if the detected language is a non-English Indian language."""
    return lang_code not in ("en", "unknown")
