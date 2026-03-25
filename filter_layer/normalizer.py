"""
filter_layer/normalizer.py
Translates multilingual symptom keywords (Hindi, Gujarati, Hinglish)
into standardized English clinical terms.
"""

import re
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Master translation dictionary
# Keys: transliterated words/phrases (lowercase)
# Values: English clinical term
# ---------------------------------------------------------------------------
MULTILINGUAL_MAP = {
    # --- Fever ---
    "bukhar": "fever",
    "bukhaar": "fever",
    "tav": "fever",
    "tave": "fever",
    "jwar": "fever",
    "jwara": "fever",
    "bimari": "illness",
    "garmi": "fever",

    # --- Headache ---
    "sar dard": "headache",
    "sir dard": "headache",
    "matha dukhay": "headache",
    "matha dukhe": "headache",
    "matha dard": "headache",
    "sar me dard": "headache",
    "sir me dard": "headache",
    "matha": "headache",

    # --- Cough ---
    "khansi": "cough",
    "khansee": "cough",
    "khaansi": "cough",
    "sukhi khansi": "dry cough",
    "balgam wali khansi": "productive cough",

    # --- Cold / Runny nose ---
    "nazla": "cold",
    "naak behna": "runny nose",
    "naak band": "nasal congestion",
    "sardi": "cold",
    "zukam": "cold",
    "jukam": "cold",

    # --- Stomach & Digestion ---
    "pet dard": "stomach pain",
    "pet me dard": "stomach pain",
    "pet dukhe": "stomach pain",
    "peth dard": "stomach pain",
    "peth me dard": "stomach pain",
    "dast": "diarrhea",
    "loose motion": "diarrhea",
    "qabz": "constipation",
    "kabz": "constipation",
    "ulti": "vomiting",
    "ultee": "vomiting",
    "ubkaayi": "nausea",
    "ubki": "nausea",
    "jalan": "acidity",
    "khatta dakar": "acid reflux",
    "gas": "bloating",

    # --- Body pain ---
    "badan dard": "body ache",
    "badan me dard": "body ache",
    "jodon ka dard": "joint pain",
    "jodon me dard": "joint pain",
    "kamar dard": "back pain",
    "kamar me dard": "back pain",
    "pair dard": "leg pain",
    "haath dard": "arm pain",
    "muscle dard": "muscle pain",

    # --- Weakness / Fatigue ---
    "kamzori": "weakness",
    "thakan": "fatigue",
    "thaak": "fatigue",
    "thaakaan": "fatigue",
    "halka mehsoos": "dizziness",

    # --- Dizziness ---
    "chakkar": "dizziness",
    "chakkar aana": "dizziness",
    "sir ghoomna": "dizziness",

    # --- Throat ---
    "gale me dard": "sore throat",
    "gala kharab": "sore throat",
    "gala dukhna": "sore throat",

    # --- Eyes ---
    "aankhon me dard": "eye pain",
    "aankhon me jalan": "eye irritation",
    "aankhon me lali": "red eyes",

    # --- Skin ---
    "khujli": "itching",
    "kharish": "itching",
    "daane": "rash",
    "lal daane": "rash",

    # --- Chest ---
    "seene me dard": "chest pain",
    "seene me jalan": "chest burning",
    "sans lene me takleef": "breathing difficulty",
    "sans lena mushkil": "breathing difficulty",
    "sans phoolna": "shortness of breath",

    # --- Gujarati specific ---
    "mane tav aayo che": "fever",
    "mane tav aayo": "fever",
    "sar dukhe che": "headache",
    "sar dukhe": "headache",
    "pet dukhe che": "stomach pain",
    "udhras": "vomiting",
    "sardi khansee": "cold cough",
    "thaak lagey che": "fatigue",
    "chakkar aave che": "dizziness",
    "ubki aave che": "nausea",
    "khansee aave che": "cough",

    # --- Common Hinglish phrases ---
    "bahut dard": "severe pain",
    "thoda dard": "mild pain",
    "bohot dard": "severe pain",
    "mujhe bukhar hai": "fever",
    "mujhe bukhar he": "fever",
    "mujhe sar dard hai": "headache",
    "mujhe khansi hai": "cough",
    "mujhe ulti ho rahi hai": "vomiting",
    "mujhe chakkar aa rahe hain": "dizziness",
    "mujhe sans lene me takleef hai": "breathing difficulty",
    "mujhe seene me dard hai": "chest pain",
    "mujhe pet me dard hai": "stomach pain",
    # --- English short-form typos (supplement spell corrector) ---
    "coff": "cough",
    "koff": "cough",
    "kogh": "cough",
    "stomch": "stomach pain",
    "stomache": "stomach pain",
    "stomachache": "stomach pain",
    "headach": "headache",
    "hadache": "headache",
    "hedache": "headache",
    "hedche": "headache",
    "fevr": "fever",
    "feve": "fever",
    "diarrhea": "diarrhea",  # keep correct spelling idempotent
    "diarrea": "diarrhea",
    "nausea": "nausea",
    "vomitting": "vomiting",
    "breathin": "breathing difficulty",
    "breating": "breathing difficulty",
}

# Sorted by length (longest first) to ensure multi-word phrases match before single words
_SORTED_KEYS = sorted(MULTILINGUAL_MAP.keys(), key=len, reverse=True)


def normalize(text: str) -> str:
    """
    Normalize multilingual input to English clinical terms.

    Steps:
    1. Lowercase + strip
    2. Replace multi-word and single-word phrases using MULTILINGUAL_MAP
    3. Return cleaned English string
    """
    if not text:
        return ""

    result = text.lower().strip()

    for phrase in _SORTED_KEYS:
        if phrase in result:
            result = result.replace(phrase, MULTILINGUAL_MAP[phrase])
            logger.debug("Normalized '%s' → '%s'", phrase, MULTILINGUAL_MAP[phrase])

    # Clean up extra whitespace
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def get_supported_languages() -> list:
    """Return list of supported language codes."""
    return ["en", "hi", "gu", "hinglish"]
