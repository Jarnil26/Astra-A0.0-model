"""
filter_layer/fallback_semantic.py
Semantic fallback for Filter Layer v2.

Used ONLY when phrase + token extraction returns zero symptoms.
Uses sentence-transformers (paraphrase-MiniLM-L3-v2) with lazy loading
and embedding cache for speed.

Performance:
  - First call: ~1-2s (model load + encode symptom list)
  - Subsequent calls: < 5ms (cached embeddings, cosine similarity)
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Similarity threshold (0-1) for semantic match acceptance
SEMANTIC_THRESHOLD = 0.45

# Model name — small, fast, 12MB
_MODEL_NAME = "paraphrase-MiniLM-L3-v2"

# Symptom phrases for semantic matching
SEMANTIC_SYMPTOM_PHRASES: List[str] = [
    "fever", "high fever", "chills",
    "headache", "migraine", "head pain",
    "cough", "dry cough", "wet cough",
    "cold", "runny nose", "nasal block",
    "sore throat", "throat pain",
    "body ache", "muscle pain", "joint pain",
    "back pain", "chest pain",
    "breathing difficulty", "shortness of breath",
    "nausea", "vomiting", "diarrhea", "constipation",
    "stomach pain", "abdominal pain", "bloating",
    "fatigue", "weakness", "dizziness", "fainting",
    "skin rash", "itching", "redness", "swelling",
    "eye pain", "ear pain",
    "loss of appetite", "weight loss",
    "anxiety", "insomnia", "depression",
    "numbness", "tingling", "seizures",
    "jaundice", "dark urine", "pale skin",
]

# Module-level cache
_model = None
_symptom_embeddings = None
_embeddings_built = False


def _get_model():
    """Lazy-load the sentence transformer model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("[fallback] Loading semantic model '%s'...", _MODEL_NAME)
            _model = SentenceTransformer(_MODEL_NAME)
            logger.info("[fallback] Model loaded.")
        except ImportError:
            logger.warning(
                "[fallback] sentence-transformers not installed. "
                "Semantic fallback disabled. Install with: pip install sentence-transformers"
            )
    return _model


def _get_symptom_embeddings():
    """Build and cache symptom embeddings (once per process)."""
    global _symptom_embeddings, _embeddings_built
    if _embeddings_built:
        return _symptom_embeddings

    model = _get_model()
    if model is None:
        _embeddings_built = True
        return None

    import numpy as np
    logger.info("[fallback] Building symptom embeddings cache...")
    _symptom_embeddings = model.encode(SEMANTIC_SYMPTOM_PHRASES, convert_to_numpy=True)
    _embeddings_built = True
    logger.info("[fallback] Embeddings cached for %d symptoms.", len(SEMANTIC_SYMPTOM_PHRASES))
    return _symptom_embeddings


def semantic_fallback(text: str, top_k: int = 3) -> List[str]:
    """
    Find the closest matching symptoms to the input text using semantic similarity.

    Args:
        text: Raw or partially normalized user input
        top_k: Maximum number of symptoms to return

    Returns:
        List of matched English symptom terms (may be empty if model unavailable)
    """
    if not text or not text.strip():
        return []

    model = _get_model()
    if model is None:
        return []

    symptom_embeddings = _get_symptom_embeddings()
    if symptom_embeddings is None:
        return []

    try:
        import numpy as np

        query_embedding = model.encode([text.lower().strip()], convert_to_numpy=True)

        # Cosine similarity
        from numpy.linalg import norm
        query_norm = query_embedding / (norm(query_embedding, axis=1, keepdims=True) + 1e-10)
        symptom_norm = symptom_embeddings / (norm(symptom_embeddings, axis=1, keepdims=True) + 1e-10)
        scores = (query_norm @ symptom_norm.T).flatten()

        # Collect matches above threshold
        matches: List[Tuple[float, str]] = []
        for i, score in enumerate(scores):
            if score >= SEMANTIC_THRESHOLD:
                matches.append((float(score), SEMANTIC_SYMPTOM_PHRASES[i]))

        matches.sort(key=lambda x: -x[0])
        results = [sym for _, sym in matches[:top_k]]

        logger.info("[fallback] semantic matches for '%s': %s", text[:40], results)
        return results

    except Exception as e:
        logger.error("[fallback] Semantic matching failed: %s", e)
        return []


def is_available() -> bool:
    """Return True if the semantic model can be loaded."""
    return _get_model() is not None
