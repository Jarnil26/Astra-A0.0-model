"""
clinical/engine.py — wraps FilterLayer + Retriever + Predictor
Singleton that initialises once at startup and is reused for all requests.
"""
import logging
import os
import time
from typing import List, Optional, Dict, Any

logger = logging.getLogger("astra.clinical")


class ClinicalEngine:
    """
    Thread-safe singleton wrapper around:
      - FilterLayer (v2 phrase-first NLP pipeline)
      - Retriever (FAISS semantic search)
      - AdvancedPredictor (disease scoring + Ayurvedic remedies)

    Usage:
        engine = ClinicalEngine.instance()
        result = engine.process_message("mane tav 6", session_id="abc123")
    """

    _instance: Optional["ClinicalEngine"] = None

    def __init__(self):
        from config import FAISS_INDEX_PATH, DB_PATH, PREVALENCE_PATH
        t0 = time.time()

        # Check files exist
        for path in [FAISS_INDEX_PATH, DB_PATH, PREVALENCE_PATH]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing model file: {path}. Run 'python build_astra.py' first."
                )

        from retriever import Retriever
        from predictor import AdvancedPredictor
        from filter_layer import FilterLayer
        from filter_layer.response_formatter import format_clinical_response

        self._retriever = Retriever(index_path=FAISS_INDEX_PATH, db_path=DB_PATH)
        self._predictor = AdvancedPredictor(db_path=DB_PATH, prevalence_path=PREVALENCE_PATH)
        # persist=False — we handle session storage ourselves in chat_db, not filter_layer's ChatStore
        self._filter = FilterLayer(use_v2=True, persist=False)
        self._formatter = format_clinical_response

        logger.info("ClinicalEngine ready in %.2fs", time.time() - t0)

    @classmethod
    def instance(cls) -> "ClinicalEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ─────────────────────────────────────────────────────────────────────────
    # Core API used by chat service
    # ─────────────────────────────────────────────────────────────────────────

    def process_message(
        self,
        message: str,
        accumulated_symptoms: List[str],
        session_id: str = "tmp",
    ) -> Dict[str, Any]:
        """
        Process one user message through the full pipeline.

        Args:
            message: Raw user text (any language)
            accumulated_symptoms: All symptoms seen so far in this session
            session_id: Used as internal filter-layer session key (in-memory only)

        Returns dict with keys:
            intent, language, symptoms_this_turn, all_symptoms,
            warnings, reply, prediction (or None)
        """
        # Reset in-memory session to inject accumulated_symptoms from our DB
        self._filter.reset_session(session_id)
        if accumulated_symptoms:
            self._filter.session_manager.add_symptoms(session_id, accumulated_symptoms)

        filter_result = self._filter.process(message, session_id=session_id)

        intent = filter_result["intent"]
        new_symptoms = filter_result["symptoms"]
        all_symptoms = filter_result["session_symptoms"]  # accumulated + new (deduplicated)
        warnings = filter_result["warnings"]
        language = filter_result["language"]

        # ── General chat ──────────────────────────────────────────────────────
        if intent == "general":
            reply = filter_result.get("response", "Hello! Please describe your symptoms.")
            return {
                "intent": "general",
                "language": language,
                "symptoms_this_turn": new_symptoms,
                "all_symptoms": all_symptoms,
                "warnings": warnings,
                "reply": reply,
                "prediction": None,
            }

        # ── No symptoms extracted ─────────────────────────────────────────────
        if not all_symptoms:
            reply = filter_result.get("response") or "Could not extract symptoms. Please describe more clearly."
            return {
                "intent": "clinical",
                "language": language,
                "symptoms_this_turn": new_symptoms,
                "all_symptoms": all_symptoms,
                "warnings": warnings,
                "reply": reply,
                "prediction": None,
            }

        # ── Full clinical prediction ──────────────────────────────────────────
        results = self._retriever.retrieve(all_symptoms)
        prediction = self._predictor.aggregate(results, all_symptoms)
        formatted = self._formatter(prediction, all_symptoms, language)

        return {
            "intent": "clinical",
            "language": language,
            "symptoms_this_turn": new_symptoms,
            "all_symptoms": all_symptoms,
            "warnings": warnings,
            "reply": formatted,
            "prediction": {
                "diseases": [
                    {"name": p["disease"], "confidence": round(p.get("confidence", 0), 3)}
                    for p in prediction.get("predictions", [])[:5]
                ],
                "remedies": prediction.get("remedies", {}),
                "clinical_note": prediction.get("notes", ""),
                "dosha": prediction.get("dosha", []),
            },
        }
