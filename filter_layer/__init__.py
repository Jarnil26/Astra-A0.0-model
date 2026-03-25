"""
filter_layer/__init__.py
FilterLayer orchestrator — supports both v1 (legacy) and v2 (production) pipelines.

v2 pipeline (default):
  tokenizer → phrase_matcher → normalizer_v2 → spell_corrector_v2
  → symptom_extractor_v2 → [fallback_semantic] → response_formatter

v1 pipeline (legacy, use_v2=False):
  language_detector → normalizer → spell_corrector → symptom_extractor
"""

import logging
import os
import time
from typing import Dict, Any, List, Optional

# ── v1 imports ────────────────────────────────────────────────────────────
from filter_layer.language_detector import detect_language
from filter_layer.normalizer import normalize
from filter_layer.spell_corrector import correct_text
from filter_layer.symptom_extractor import extract_symptoms
from filter_layer.intent_classifier import classify_intent
from filter_layer.session_manager import SessionManager
from filter_layer.response_formatter import (
    check_critical_symptoms,
    format_safety_warning,
    format_general_response,
    format_fallback,
    build_json_response,
)

# ── v2 imports ────────────────────────────────────────────────────────────
from filter_layer.normalizer_v2 import normalize_v2
from filter_layer.spell_corrector_v2 import correct_tokens_safe
from filter_layer.symptom_extractor_v2 import extract_symptoms_v2
from filter_layer.fallback_semantic import semantic_fallback

logging.basicConfig(
    level=logging.INFO,
    format="[FilterLayer] %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

_SYMPTOMS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "symptoms_list.txt"
)


class FilterLayer:
    """
    Multilingual pre-processing filter for the Astra A0 Clinical Engine.

    Args:
        use_v2: If True (default) uses the v2 phrase-first pipeline.
                Set to False for legacy v1 behavior.
        symptoms_file: Optional path to extra symptom vocabulary.
        enable_semantic_fallback: If True, uses sentence-transformers when
                                   phrase+token extraction yields nothing.
    """

    def __init__(
        self,
        use_v2: bool = True,
        symptoms_file: Optional[str] = None,
        enable_semantic_fallback: bool = False,
        persist: bool = True,
    ):
        self.use_v2 = use_v2
        self.enable_semantic_fallback = enable_semantic_fallback
        self.symptoms_file = symptoms_file or (
            _SYMPTOMS_FILE if os.path.exists(_SYMPTOMS_FILE) else None
        )
        self.session_manager = SessionManager(persist=persist)

        pipeline = "v2 (phrase-first)" if use_v2 else "v1 (legacy)"
        storage = "MongoDB" if persist else "in-memory"
        logger.info(
            "FilterLayer initialized. Pipeline: %s | Storage: %s",
            pipeline, storage
        )

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def process(self, user_input: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Process raw user input through the full filter pipeline.

        Args:
            user_input: Raw user text (any language)
            session_id: Unique session identifier

        Returns:
            {
                "intent": str,
                "language": str,
                "symptoms": List[str],          # this turn only
                "session_symptoms": List[str],  # accumulated
                "response": str,
                "warnings": List[str],
                "debug": Dict,                  # v2 only
            }
        """
        t0 = time.perf_counter()

        if not user_input or not user_input.strip():
            return build_json_response(
                intent="general",
                symptoms=[],
                session_symptoms=self.session_manager.get_symptoms(session_id),
                formatted_response="Please describe your symptoms.",
                language="en",
            )

        if self.use_v2:
            result = self._process_v2(user_input, session_id)
        else:
            result = self._process_v1(user_input, session_id)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Processed in %.2fms | pipeline=%s intent=%s symptoms=%s",
            elapsed_ms,
            "v2" if self.use_v2 else "v1",
            result["intent"],
            result["symptoms"],
        )
        return result

    def reset_session(self, session_id: str = "default"):
        self.session_manager.clear_session(session_id)

    def get_session_symptoms(self, session_id: str = "default") -> List[str]:
        return self.session_manager.get_symptoms(session_id)

    # ──────────────────────────────────────────────────────────────────────
    # v2 Pipeline
    # ──────────────────────────────────────────────────────────────────────

    def _process_v2(self, raw: str, session_id: str) -> Dict[str, Any]:
        debug: Dict[str, Any] = {"pipeline": "v2"}

        # Step 1: Language detection (used for response localization)
        language = detect_language(raw)
        debug["language"] = language

        # Step 2: Normalize via phrase_matcher + safe reconstruction
        norm_result = normalize_v2(raw)
        debug["cleaned"] = norm_result.cleaned
        debug["normalized"] = norm_result.normalized
        debug["phrase_matches"] = [
            {"span": f"{s}:{e}", "term": t}
            for s, e, t in norm_result.phrase_matches
        ]

        # Step 3: Safe spell correction on remaining tokens
        matched_terms_set = set(norm_result.english_terms)
        corrected_text, corrections = correct_tokens_safe(
            norm_result.normalized,
            skip_tokens=matched_terms_set,
        )
        debug["spell_corrections"] = corrections

        # Step 4: Hybrid symptom extraction (phrases already extracted in step 2)
        # Run full v2 extraction on original input to catch anything missed
        symptoms = extract_symptoms_v2(raw)

        # If extraction is still empty, try on corrected text as fallback
        if not symptoms and corrected_text.strip():
            symptoms = extract_symptoms_v2(corrected_text)

        debug["extracted_symptoms"] = symptoms

        # Step 5: Semantic fallback if no symptoms found
        if not symptoms and self.enable_semantic_fallback:
            symptoms = semantic_fallback(raw)
            if symptoms:
                debug["semantic_fallback"] = symptoms
                logger.info("Semantic fallback triggered for: '%s'", raw[:40])

        # Step 6: Intent classification
        intent, general_reply = classify_intent(raw, symptoms)
        debug["intent"] = intent

        # Step 7: Session accumulation
        if intent == "clinical" and symptoms:
            session_symptoms = self.session_manager.add_symptoms(session_id, symptoms)
        else:
            session_symptoms = self.session_manager.get_symptoms(session_id)

        # Step 8: Safety warnings
        warnings: List[str] = []
        all_syms = list(set(symptoms + session_symptoms))
        if check_critical_symptoms(all_syms):
            warnings.append(format_safety_warning(all_syms))

        # Step 9: Response
        if intent == "general":
            response = format_general_response(general_reply, language)
        elif not symptoms and not session_symptoms:
            response = format_fallback()
        else:
            response = ""  # Filled by main.py after clinical engine runs

        result = build_json_response(
            intent=intent,
            symptoms=symptoms,
            session_symptoms=session_symptoms,
            formatted_response=response,
            warnings=warnings,
            language=language,
        )
        result["debug"] = debug
        return result

    # ──────────────────────────────────────────────────────────────────────
    # v1 Pipeline (legacy — unchanged from original)
    # ──────────────────────────────────────────────────────────────────────

    def _process_v1(self, raw: str, session_id: str) -> Dict[str, Any]:
        language = detect_language(raw)
        normalized = normalize(raw)
        corrected = correct_text(normalized, self.symptoms_file)
        symptoms = extract_symptoms(corrected, self.symptoms_file)
        intent, general_reply = classify_intent(corrected, symptoms)

        if intent == "clinical" and symptoms:
            session_symptoms = self.session_manager.add_symptoms(session_id, symptoms)
        else:
            session_symptoms = self.session_manager.get_symptoms(session_id)

        warnings: List[str] = []
        all_syms = list(set(symptoms + session_symptoms))
        if check_critical_symptoms(all_syms):
            warnings.append(format_safety_warning(all_syms))

        if intent == "general":
            response = format_general_response(general_reply, language)
        elif not symptoms and not session_symptoms:
            response = format_fallback()
        else:
            response = ""

        return build_json_response(
            intent=intent,
            symptoms=symptoms,
            session_symptoms=session_symptoms,
            formatted_response=response,
            warnings=warnings,
            language=language,
        )
