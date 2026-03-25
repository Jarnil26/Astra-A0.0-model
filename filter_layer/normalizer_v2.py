"""
filter_layer/normalizer_v2.py
Production-grade phrase-first, slot-aware multilingual normalizer.

Key improvements over v1:
  - Uses phrase_matcher for boundary-safe regex matching (no str.replace corruption)
  - Slot-marking: once a span is claimed, it is never re-processed
  - Applies English typo corrections ONLY to unclaimed tokens
  - Returns structured debug info alongside the normalized text
"""

import re
import logging
from typing import Dict, List, Tuple, Optional

from filter_layer.phrase_matcher import match_phrases, apply_typo_corrections
from filter_layer.tokenizer import clean_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class NormalizationResult:
    """Structured result from v2 normalization."""

    def __init__(
        self,
        original: str,
        cleaned: str,
        normalized: str,
        phrase_matches: List[Tuple[int, int, str]],
        english_terms: List[str],
    ):
        self.original = original
        self.cleaned = cleaned
        self.normalized = normalized          # English text with symptoms inlined
        self.phrase_matches = phrase_matches  # [(start, end, term), ...]
        self.english_terms = english_terms    # deduplicated extracted terms

    def __repr__(self):
        return (
            f"NormalizationResult("
            f"terms={self.english_terms}, "
            f"matches={len(self.phrase_matches)}, "
            f"normalized='{self.normalized}')"
        )


def normalize_v2(text: str) -> NormalizationResult:
    """
    Phrase-first, slot-aware normalization pipeline.

    Steps:
    1. Clean text (lowercase, strip OCR digits, normalize whitespace)
    2. Run phrase_matcher → get non-overlapping (start, end, term) spans
    3. Reconstruct text by replacing matched spans with English terms
    4. Apply English typo corrections to remaining (unclaimed) tokens
    5. Return NormalizationResult with full debug info

    Args:
        text: Raw user input (any language)

    Returns:
        NormalizationResult
    """
    if not text or not text.strip():
        return NormalizationResult(
            original=text, cleaned="", normalized="",
            phrase_matches=[], english_terms=[]
        )

    original = text
    cleaned = clean_text(text)

    logger.debug("[v2] cleaned: '%s'", cleaned)

    # ── Step 1: Phrase matching ───────────────────────────────────────────
    phrase_matches = match_phrases(cleaned)
    logger.debug("[v2] phrase_matches: %s", phrase_matches)

    # ── Step 2: Slot-safe text reconstruction ────────────────────────────
    # Build output by walking character positions, replacing matched spans
    normalized_parts: List[str] = []
    prev_end = 0

    for start, end, term in sorted(phrase_matches, key=lambda x: x[0]):
        # Keep any text between previous match and this one
        gap = cleaned[prev_end:start].strip()
        if gap:
            normalized_parts.append(gap)
        normalized_parts.append(term)
        prev_end = end

    # Append remaining text after the last match
    tail = cleaned[prev_end:].strip()
    if tail:
        normalized_parts.append(tail)

    reconstructed = " ".join(normalized_parts)

    # ── Step 3: Apply typo corrections to unclaimed tokens ───────────────
    # Compute "protected" positions in `reconstructed`:
    # The English terms we just inserted are already correct — protect them.
    protected_spans: List[Tuple[int, int]] = []
    offset = 0
    for part in normalized_parts:
        protected_spans.append((offset, offset + len(part)))
        offset += len(part) + 1  # +1 for the space separator

    final_text = apply_typo_corrections(reconstructed, protected_spans)

    # Clean up extra whitespace
    final_text = re.sub(r'\s+', ' ', final_text).strip()

    # ── Step 4: Extract English terms list ───────────────────────────────
    english_terms: List[str] = _deduplicate(
        [term for _, _, term in phrase_matches]
    )

    logger.debug("[v2] final_text: '%s'", final_text)
    logger.debug("[v2] english_terms: %s", english_terms)

    return NormalizationResult(
        original=original,
        cleaned=cleaned,
        normalized=final_text,
        phrase_matches=phrase_matches,
        english_terms=english_terms,
    )


def _deduplicate(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
