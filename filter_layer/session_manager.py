"""
filter_layer/session_manager.py
Multi-turn session management for Astra A0 Clinical Engine.

Two modes:
  persist=False  (default for tests) → in-memory only, same behaviour as v1
  persist=True   (production)        → delegates to MongoDB ChatStore
"""

import logging
import time
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

SESSION_TIMEOUT_SECONDS = 2 * 60 * 60  # 2 hours


# ---------------------------------------------------------------------------
# In-Memory SessionData (used when persist=False)
# ---------------------------------------------------------------------------

class SessionData:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.symptoms: List[str] = []
        self.messages: List[Dict[str, Any]] = []
        self.created_at: float = time.time()
        self.last_active: float = time.time()
        self.turn_count: int = 0

    def touch(self):
        self.last_active = time.time()
        self.turn_count += 1

    def is_expired(self) -> bool:
        return (time.time() - self.last_active) > SESSION_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class SessionManager:
    """
    Manages per-session symptom accumulation and message history.

    Args:
        persist: If True, uses MongoDB ChatStore for durability.
                 If False, uses in-memory dict (fast, for tests).
    """

    def __init__(self, persist: bool = False):
        self.persist = persist
        self._sessions: Dict[str, SessionData] = {}  # in-memory fallback
        self._store = None

        if persist:
            try:
                from filter_layer.chat_store import get_store
                self._store = get_store()
                if self._store.is_connected:
                    logger.info("[SessionManager] Using MongoDB ChatStore.")
                else:
                    logger.warning("[SessionManager] ChatStore offline — using in-memory mode.")
                    self.persist = False
            except Exception as e:
                logger.error("[SessionManager] ChatStore init failed: %s", e)
                self.persist = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def _get_in_memory(self, session_id: str) -> SessionData:
        if session_id in self._sessions:
            s = self._sessions[session_id]
            if s.is_expired():
                s = SessionData(session_id)
                self._sessions[session_id] = s
                logger.info("[SessionManager] Session '%s' expired — reset.", session_id)
            return s
        s = SessionData(session_id)
        self._sessions[session_id] = s
        logger.debug("[SessionManager] New in-memory session: '%s'", session_id)
        return s

    def clear_session(self, session_id: str):
        """Reset symptoms for a session (keeps message history in MongoDB)."""
        if self.persist and self._store:
            self._store.clear_symptoms(session_id)
        if session_id in self._sessions:
            self._sessions[session_id].symptoms = []
        logger.debug("[SessionManager] Cleared symptoms: '%s'", session_id)

    def delete_session(self, session_id: str):
        """Permanently remove a session."""
        if self.persist and self._store:
            self._store.delete_session(session_id)
        self._sessions.pop(session_id, None)

    # ── Symptoms ──────────────────────────────────────────────────────────

    def add_symptoms(self, session_id: str, new_symptoms: List[str]) -> List[str]:
        """
        Add symptoms to the session (deduplicated).
        Returns the full updated symptom list.
        """
        if self.persist and self._store:
            return self._store.add_symptoms(session_id, new_symptoms)

        # In-memory path
        s = self._get_in_memory(session_id)
        s.touch()
        existing = set(s.symptoms)
        for sym in new_symptoms:
            clean = sym.strip().lower()
            if clean and clean not in existing:
                s.symptoms.append(clean)
                existing.add(clean)
                logger.debug("[SessionManager] [%s] + symptom: '%s'", session_id, clean)
        return s.symptoms.copy()

    def get_symptoms(self, session_id: str) -> List[str]:
        """Return all accumulated symptoms."""
        if self.persist and self._store:
            return self._store.get_symptoms(session_id)
        if session_id in self._sessions:
            return self._sessions[session_id].symptoms.copy()
        return []

    # ── Message history ───────────────────────────────────────────────────

    def add_message(
        self,
        session_id: str,
        role: str,
        text: str,
        turn: int,
        symptoms_this_turn: Optional[List[str]] = None,
    ):
        """Store a user or system message."""
        msg = {
            "turn": turn,
            "role": role,
            "text": text,
            "ts": time.time(),
        }
        if symptoms_this_turn:
            msg["symptoms_this_turn"] = symptoms_this_turn

        if self.persist and self._store:
            self._store.add_message(session_id, role, text, turn, symptoms_this_turn)
        else:
            s = self._get_in_memory(session_id)
            s.messages.append(msg)

    def get_history(self, session_id: str, last_n: int = 20) -> List[Dict[str, Any]]:
        """Return the last N messages in the session."""
        if self.persist and self._store:
            return self._store.get_history(session_id, last_n)
        if session_id in self._sessions:
            msgs = self._sessions[session_id].messages
            return msgs[-last_n:] if last_n else msgs
        return []

    def get_turn_count(self, session_id: str) -> int:
        if self.persist and self._store:
            return self._store.get_turn_count(session_id)
        if session_id in self._sessions:
            return self._sessions[session_id].turn_count
        return 0

    # ── Utilities ─────────────────────────────────────────────────────────

    def list_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent sessions (MongoDB only; returns [] in memory mode)."""
        if self.persist and self._store:
            return self._store.list_sessions(limit)
        return [
            {"_id": sid, "symptoms": s.symptoms, "turn_count": s.turn_count}
            for sid, s in self._sessions.items()
            if not s.is_expired()
        ]

    def purge_expired(self):
        expired = [sid for sid, s in self._sessions.items() if s.is_expired()]
        for sid in expired:
            self._sessions.pop(sid, None)
        if expired:
            logger.info("[SessionManager] Purged %d expired in-memory sessions.", len(expired))

    def active_session_count(self) -> int:
        return sum(1 for s in self._sessions.values() if not s.is_expired())


# ---------------------------------------------------------------------------
# Default instance for legacy imports
# ---------------------------------------------------------------------------
default_session = SessionManager(persist=False)
