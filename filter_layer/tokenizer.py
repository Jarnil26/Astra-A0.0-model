"""
filter_layer/tokenizer.py
Safe tokenizer and n-gram generator for multilingual clinical input.

Handles:
- Stripping noise characters (standalone digits from OCR, "6" artifacts)
- Preserving multi-word structures
- Generating n-grams for phrase matching
"""

import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Characters/patterns to strip before tokenization
# Note: standalone digits (OCR artifacts like "6") are stripped
_NOISE_PATTERN = re.compile(
    r'\b\d+\b'           # standalone numbers (OCR artifacts)
    r'|[^\w\s\'-]',      # non-word except apostrophe/hyphen
    re.UNICODE
)

# Normalize whitespace
_WHITESPACE_PATTERN = re.compile(r'\s+')

# Words that carry no symptom meaning (skip for token matching, keep for phrase matching)
STOP_WORDS = frozenset({
    "and", "or", "the", "a", "an", "i", "have", "has", "had",
    "am", "is", "are", "was", "were", "be", "been",
    "me", "my", "mane", "mujhe", "mera", "meri",
    "hai", "hain", "he", "ho", "raha", "rahi",
    "aur", "ya", "bhi", "toh", "nahi",
    "che", "chhe", "aayu", "aayo",  # Gujarati particles
    "kuch",
})


def clean_text(text: str) -> str:
    """
    Light cleaning: lowercase, strip OCR digit artifacts, normalize whitespace.
    Does NOT remove words — preserves structure for phrase matching.

    Args:
        text: Raw user input

    Returns:
        Cleaned, lowercased text safe for phrase matching
    """
    if not text:
        return ""
    text = text.lower().strip()
    text = _NOISE_PATTERN.sub(" ", text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """
    Tokenize cleaned text into individual tokens.
    Includes all words (stop words kept for n-gram context).

    Args:
        text: Already cleaned text (output of clean_text)

    Returns:
        List of lowercase tokens
    """
    return text.split()


def meaningful_tokens(tokens: List[str]) -> List[str]:
    """
    Filter stop words for single-token symptom extraction pass.
    Keeps tokens that might be symptom-relevant.

    Args:
        tokens: Full token list

    Returns:
        Tokens with stop words removed
    """
    return [t for t in tokens if t not in STOP_WORDS and len(t) >= 3]


def ngrams(tokens: List[str], n: int) -> List[Tuple[str, int, int]]:
    """
    Generate n-grams as (phrase_string, start_index, end_index).

    Args:
        tokens: Token list
        n: Window size

    Returns:
        List of (phrase, start_token_idx, end_token_idx)
    """
    result = []
    for i in range(len(tokens) - n + 1):
        phrase = " ".join(tokens[i:i + n])
        result.append((phrase, i, i + n - 1))
    return result


def all_ngrams(tokens: List[str], max_n: int = 5) -> List[Tuple[str, int, int]]:
    """
    Generate all n-grams from 2 up to max_n, plus unigrams.
    Sorted longest-first.

    Args:
        tokens: Token list
        max_n: Maximum n-gram size

    Returns:
        All (phrase, start, end) tuples, longest first
    """
    result = []
    for n in range(max_n, 0, -1):
        result.extend(ngrams(tokens, n))
    return result


def reconstruct_text(tokens: List[str]) -> str:
    """Rejoin tokens into a string."""
    return " ".join(tokens)
