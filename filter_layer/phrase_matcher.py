"""
filter_layer/phrase_matcher.py
Production-grade phrase-first multilingual symptom matcher.

Strategy:
  - Canonical dictionary: English symptom → [list of multilingual aliases]
  - Compiled regex patterns with word-boundary safety
  - Longest-match-first (4-gram > 3-gram > 2-gram > 1-gram)
  - Returns non-overlapping matches with character spans
"""

import re
import logging
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CANONICAL SYMPTOM DICTIONARY
# Format: english_term → [multilingual aliases, longest first recommended]
#
# Rules for adding entries:
#  - Include all spelling variants, conjugations, trailing particles (che, hai, etc.)
#  - Numbers like "6" appearing in OCR/transliteration should be stripped by tokenizer
#  - All keys lowercase
# ---------------------------------------------------------------------------
SYMPTOM_ALIASES: Dict[str, List[str]] = {

    # ── Fever ────────────────────────────────────────────────────────────────
    "fever": [
        "mujhe bukhar hai", "mujhe bukhar he", "mane tav aayo che",
        "mane tav aayo", "mujhe tav hai",
        "bukhar aayo", "tav aayo che", "tav aayu che",
        "bukhar hai", "bukhar he",
        "bukhaar", "bukhar", "jwar", "jwara", "garmi",
        "tave", "tav",
    ],

    # ── Headache ─────────────────────────────────────────────────────────────
    "headache": [
        # Gujarati variants — CRITICAL missing ones
        "mathu dukhe che", "mathu dukhe", "mathuu dukhe",
        "mathu ma dard", "mathu nu dard",
        "matha dukhe che", "matha dukhe", "matha dukhay", "matha dard",
        # Hindi variants
        "mujhe sar dard hai", "sar me dard", "sir me dard",
        "sar dard hai", "sir dard hai",
        "sar dukhe che", "sar dukhe",
        "sar dard", "sir dard",
        # Single words
        "matha",
    ],

    # ── Cough ────────────────────────────────────────────────────────────────
    "cough": [
        "khansee aave che", "balgam wali khansi",
        "sukhi khansi", "khansi aave", "khansi hai",
        "khaansi", "khansee", "khansi",
    ],

    # ── Dry cough ────────────────────────────────────────────────────────────
    "dry cough": ["sukhi khansi", "dry khansi"],

    # ── Cold ─────────────────────────────────────────────────────────────────
    "cold": ["sardi khansee", "nazla zukam", "nazla jukam", "nazla", "zukam", "jukam", "sardi"],

    # ── Runny nose ───────────────────────────────────────────────────────────
    "runny nose": ["naak behna", "naak se paani"],

    # ── Nasal congestion ─────────────────────────────────────────────────────
    "nasal congestion": ["naak band"],

    # ── Stomach pain ─────────────────────────────────────────────────────────
    "stomach pain": [
        "mujhe pet me dard hai", "pet me dard", "peth me dard",
        "pet dukhe che", "pet dukhe",
        "pet dard", "peth dard",
    ],

    # ── Nausea ───────────────────────────────────────────────────────────────
    "nausea": ["ubki aave che", "ubkaayi", "ubki"],

    # ── Vomiting ─────────────────────────────────────────────────────────────
    "vomiting": [
        "mujhe ulti ho rahi hai", "ulti aave che",
        "udhras", "ultee", "ulti",
    ],

    # ── Diarrhea ─────────────────────────────────────────────────────────────
    "diarrhea": ["loose motion", "dast"],

    # ── Constipation ─────────────────────────────────────────────────────────
    "constipation": ["qabz", "kabz"],

    # ── Acidity ──────────────────────────────────────────────────────────────
    "acidity": ["jalan", "khatta dakar"],

    # ── Bloating ─────────────────────────────────────────────────────────────
    "bloating": ["gas", "afara"],

    # ── Body ache ────────────────────────────────────────────────────────────
    "body ache": ["badan me dard", "badan dard", "badan dukhay"],

    # ── Joint pain ───────────────────────────────────────────────────────────
    "joint pain": ["jodon me dard", "jodon ka dard", "jodon dukhe"],

    # ── Back pain ────────────────────────────────────────────────────────────
    "back pain": ["kamar me dard", "kamar dard"],

    # ── Weakness ─────────────────────────────────────────────────────────────
    "weakness": ["kamzori"],

    # ── Fatigue ──────────────────────────────────────────────────────────────
    "fatigue": [
        "thaak lagey che", "thaakaan", "thaak", "thakan",
    ],

    # ── Dizziness ────────────────────────────────────────────────────────────
    "dizziness": [
        "mujhe chakkar aa rahe hain", "chakkar aave che",
        "chakkar aana", "sir ghoomna", "halka mehsoos",
        "chakkar",
    ],

    # ── Sore throat (English self-alias) ─────────────────────────────────────
    "sore throat": [
        "sore throat", "throat pain",                    # English self-aliases
        "gale me dard", "gala kharab", "gala dukhna",
    ],

    # ── Chest pain ───────────────────────────────────────────────────────────
    "chest pain": [
        "chest pain", "chest tightness", "chest pressure",  # English self-aliases
        "mujhe seene me dard hai", "seene me dard",
    ],

    # ── Breathing difficulty ─────────────────────────────────────────────────
    "breathing difficulty": [
        "breathing difficulty", "difficulty breathing",    # English self-aliases
        "shortness of breath", "breathing problem",
        "mujhe sans lene me takleef hai",
        "sans lene me takleef", "sans lena mushkil", "sans phoolna",
    ],

    # ── Itching ──────────────────────────────────────────────────────────────
    "itching": ["khujli", "kharish"],

    # ── Rash ─────────────────────────────────────────────────────────────────
    "rash": ["lal daane", "daane"],

    # ── Illness (generic) ────────────────────────────────────────────────────
    "illness": ["bimari"],
}

# ---------------------------------------------------------------------------
# English-language typo → correct (kept separate, lower priority)
# These are applied ONLY to lone tokens that didn't match any phrase
# ---------------------------------------------------------------------------
ENGLISH_TYPOS: Dict[str, str] = {
    "fevr": "fever", "feve": "fever",
    "hedache": "headache", "hedche": "headache",
    "headach": "headache", "hadache": "headache",
    "coff": "cough", "koff": "cough", "kogh": "cough",
    "stomch": "stomach pain", "stomache": "stomach pain", "stomachache": "stomach pain",
    "diarrea": "diarrhea", "vomitting": "vomiting",
    "breathin": "breathing difficulty", "breating": "breathing difficulty",
}

# ---------------------------------------------------------------------------
# Build compiled patterns at import time (fast at runtime)
# ---------------------------------------------------------------------------

PhrasePattern = Tuple[re.Pattern, str, int]  # (pattern, english_term, n_words)


def _build_patterns() -> List[PhrasePattern]:
    """
    Compile all alias→term mappings into sorted regex patterns.
    Longest aliases first to ensure greedy longest-match.
    """
    patterns: List[PhrasePattern] = []

    for english_term, aliases in SYMPTOM_ALIASES.items():
        for alias in aliases:
            alias_clean = alias.strip().lower()
            n_words = len(alias_clean.split())
            # Word-boundary-safe pattern; \s+ between words allows variable whitespace
            tokens = re.escape(alias_clean).replace(r'\ ', r'\s+')
            pattern = re.compile(
                r'(?<!\w)' + tokens + r'(?!\w)',
                re.IGNORECASE
            )
            patterns.append((pattern, english_term, n_words))

    # Sort: more words first (longest match wins), then alphabetical for stability
    patterns.sort(key=lambda x: (-x[2], x[1]))
    return patterns


_COMPILED_PATTERNS: List[PhrasePattern] = _build_patterns()

# Compile English typo patterns (single word only)
_TYPO_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'(?<!\w)' + re.escape(typo) + r'(?!\w)', re.IGNORECASE), correct)
    for typo, correct in ENGLISH_TYPOS.items()
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

MatchResult = Tuple[int, int, str]  # (start, end, english_term)


def match_phrases(text: str) -> List[MatchResult]:
    """
    Find all non-overlapping phrase matches in the text.

    Returns a list of (start, end, english_term) sorted by position.
    Longest match wins when spans overlap.

    Args:
        text: Normalized (lowercased) input text

    Returns:
        List of (start_char, end_char, english_symptom) tuples
    """
    text_lower = text.lower()
    # (start, end, term, n_words)
    all_matches: List[Tuple[int, int, str, int]] = []

    for pattern, term, n_words in _COMPILED_PATTERNS:
        for m in pattern.finditer(text_lower):
            all_matches.append((m.start(), m.end(), term, n_words))

    # Greedy non-overlapping: longer matches win; ties → leftmost
    all_matches.sort(key=lambda x: (x[0], -(x[3])))
    selected: List[MatchResult] = []
    consumed_end = 0

    for start, end, term, _ in sorted(all_matches, key=lambda x: (x[0], -(x[1]-x[0]))):
        if start >= consumed_end:
            selected.append((start, end, term))
            consumed_end = end
            logger.debug("Phrase match: '%s' → '%s' [%d:%d]", text[start:end], term, start, end)

    return selected


def apply_typo_corrections(text: str, skip_positions: List[Tuple[int, int]]) -> str:
    """
    Apply English typo corrections ONLY to tokens not already covered by phrase matches.

    Args:
        text: Text after phrase matches have been replaced
        skip_positions: List of (start, end) spans to skip

    Returns:
        Text with typos corrected
    """
    result = text
    offset = 0

    for typo_pattern, correct in _TYPO_PATTERNS:
        for m in list(typo_pattern.finditer(result)):
            pos_start = m.start()
            pos_end = m.end()
            # Skip if this range overlaps a protected span (offset-adjusted)
            overlaps = any(
                not (pos_end <= ps or pos_start >= pe)
                for ps, pe in skip_positions
            )
            if not overlaps:
                result = result[:pos_start] + correct + result[pos_end:]
                # Adjust offsets for future iterations
                offset += len(correct) - (pos_end - pos_start)
                logger.debug("Typo corrected: '%s' → '%s'", m.group(), correct)
                break  # recompute on next outer loop pass

    return result


def get_all_aliases() -> Dict[str, List[str]]:
    """Return the full alias dictionary (for debugging / extension)."""
    return {k: list(v) for k, v in SYMPTOM_ALIASES.items()}
