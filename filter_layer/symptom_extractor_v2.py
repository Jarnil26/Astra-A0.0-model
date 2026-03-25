"""
filter_layer/symptom_extractor_v2.py
Hybrid phrase + token symptom extractor for Filter Layer v2.

Two-pass strategy:
  Pass 1 — Phrase pass: collect English terms from phrase_matcher results
  Pass 2 — Token pass: apply rapidfuzz to remaining uncovered tokens
  Merge + deduplicate

This extractor never re-processes tokens already claimed by phrase matches.
"""

import re
import logging
from typing import List, Set, Tuple, Optional

from filter_layer.phrase_matcher import match_phrases
from filter_layer.tokenizer import clean_text, tokenize, meaningful_tokens
from filter_layer.spell_corrector_v2 import CLINICAL_VOCAB, ENGLISH_SYMPTOM_TOKENS

logger = logging.getLogger(__name__)

# Direct single-word symptom look-up set (fast O(1) check)
_SINGLE_WORD_SYMPTOMS: Set[str] = {
    term for term in CLINICAL_VOCAB if ' ' not in term
}

# Threshold for token-level fuzzy matching (lower than spell correction — more permissive)
TOKEN_MATCH_THRESHOLD = 84


def extract_symptoms_v2(
    text: str,
    extra_vocab: Optional[List[str]] = None,
) -> List[str]:
    """
    Full hybrid extraction pipeline.

    Args:
        text: Raw user input (any language)
        extra_vocab: Optional additional symptom terms to include in vocab

    Returns:
        Deduplicated list of English clinical symptoms in order of detection
    """
    if not text or not text.strip():
        return []

    vocab = list(CLINICAL_VOCAB)
    if extra_vocab:
        vocab = list(set(vocab + extra_vocab))

    # ── Step 0: Clean input ───────────────────────────────────────────────
    cleaned = clean_text(text)
    logger.debug("[v2 extract] cleaned: '%s'", cleaned)

    # ── Pass 1: Phrase matching ───────────────────────────────────────────
    phrase_matches: List[Tuple[int, int, str]] = match_phrases(cleaned)
    phrase_symptoms: List[str] = [term for _, _, term in phrase_matches]

    # Track which character positions are consumed by phrase matches
    consumed_positions: Set[int] = set()
    for start, end, _ in phrase_matches:
        consumed_positions.update(range(start, end))

    logger.debug("[v2 extract] pass1 phrase symptoms: %s", phrase_symptoms)

    # ── Pass 2: Token-level matching on remaining text ────────────────────
    # Reconstruct remaining text by masking consumed positions with spaces
    remaining_chars = list(cleaned)
    for pos in consumed_positions:
        if pos < len(remaining_chars):
            remaining_chars[pos] = ' '
    remaining_text = ''.join(remaining_chars)

    token_symptoms: List[str] = _extract_from_tokens(remaining_text, vocab)
    logger.debug("[v2 extract] pass2 token symptoms: %s", token_symptoms)

    # ── Merge + deduplicate ───────────────────────────────────────────────
    all_symptoms = _deduplicate(phrase_symptoms + token_symptoms)
    logger.info("[v2 extract] final symptoms: %s (from: '%s')", all_symptoms, text[:60])

    return all_symptoms


def _extract_from_tokens(text: str, vocab: List[str]) -> List[str]:
    """
    Extract symptoms from remaining (non-phrase-matched) text.
    Uses direct lookup and fuzzy matching on individual tokens.
    """
    found: List[str] = []
    tokens = meaningful_tokens(tokenize(text))

    single_vocab = [t for t in vocab if ' ' not in t]

    try:
        from rapidfuzz import process, fuzz
        fuzzy_available = True
    except ImportError:
        fuzzy_available = False
        logger.warning("[v2 extract] rapidfuzz not available")

    for token in tokens:
        if len(token) < 3:
            continue

        # Direct exact match (fast path)
        if token in _SINGLE_WORD_SYMPTOMS or token in ENGLISH_SYMPTOM_TOKENS:
            found.append(token)
            logger.debug("[v2 extract] direct match: '%s'", token)
            continue

        # Fuzzy match
        if fuzzy_available:
            match = process.extractOne(
                token,
                single_vocab,
                scorer=fuzz.WRatio,
                score_cutoff=TOKEN_MATCH_THRESHOLD,
            )
            if match:
                found.append(match[0])
                logger.debug(
                    "[v2 extract] fuzzy token match: '%s' → '%s' (%d)",
                    token, match[0], match[1],
                )

    return found


def _deduplicate(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
