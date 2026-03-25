"""
filter_layer/chat_store.py
MongoDB-backed persistent chat history for Astra A0 Clinical Engine.

Collection: astra_sessions  (one document per session)
Database  : astra_clinical

Document schema:
{
  "_id": "<session_id>",
  "created_at": <timestamp>,
  "last_active": <timestamp>,
  "turn_count": <int>,
  "symptoms": ["fever", "headache", ...],
  "messages": [
    {
      "turn": 1,
      "role": "user" | "system",
      "text": "...",
      "symptoms_this_turn": [...],
      "ts": <timestamp>
    },
    ...
  ]
}
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)

# MongoDB connection — reads from environment (Render) or falls back to dev URI
_DEFAULT_URI = (
    "mongodb+srv://risusolutions_db_user:94fT2hhqgWhCUODr"
    "@cluster0.h2wquwv.mongodb.net/?appName=Cluster0"
)
MONGO_URI = os.environ.get("MONGO_URI", _DEFAULT_URI)

DB_NAME         = "astra_clinical"
COLLECTION_NAME = "astra_sessions"

# Session inactivity timeout (seconds) — 2 hours for API use
SESSION_TIMEOUT = 2 * 60 * 60


class ChatStore:
    """
    Persistent, MongoDB-backed chat session store.

    Each session document stores:
      - full message history (user + system turns)
      - accumulated symptom list (deduplicated)
      - timestamps and turn counter

    Designed to be API-ready: the same session_id can be passed from
    any client and will resume exactly where it left off.
    """

    def __init__(self):
        self._client = None
        self._col = None
        self._connected = False
        self._connect()

    # ──────────────────────────────────────────────────────────────────────
    # Connection
    # ──────────────────────────────────────────────────────────────────────

    def _connect(self):
        try:
            from pymongo import MongoClient
            from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

            self._client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            # Force a connection check
            self._client.admin.command("ping")
            db = self._client[DB_NAME]
            self._col = db[COLLECTION_NAME]

            # Indexes for fast lookups
            self._col.create_index("last_active", background=True)
            self._connected = True
            logger.info("[ChatStore] Connected to MongoDB Atlas (%s.%s)", DB_NAME, COLLECTION_NAME)

        except Exception as e:
            self._connected = False
            logger.error("[ChatStore] MongoDB connection failed: %s", e)
            logger.warning("[ChatStore] Falling back to in-memory mode (history will NOT persist)")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ──────────────────────────────────────────────────────────────────────
    # Session lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return the full session document or None if not found."""
        if not self._connected:
            return None
        try:
            return self._col.find_one({"_id": session_id})
        except Exception as e:
            logger.error("[ChatStore] get_session error: %s", e)
            return None

    def get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        """Return existing session or create a new one."""
        doc = self.get_session(session_id)
        if doc:
            # Check timeout
            if time.time() - doc.get("last_active", 0) > SESSION_TIMEOUT:
                logger.info("[ChatStore] Session '%s' timed out — resetting symptoms.", session_id)
                self._col.update_one(
                    {"_id": session_id},
                    {"$set": {"symptoms": [], "turn_count": 0, "last_active": time.time()}}
                )
                doc["symptoms"] = []
                doc["turn_count"] = 0
            return doc

        # Create new document
        now = time.time()
        new_doc = {
            "_id": session_id,
            "created_at": now,
            "last_active": now,
            "turn_count": 0,
            "symptoms": [],
            "messages": [],
        }
        if self._connected:
            try:
                self._col.insert_one(new_doc)
                logger.info("[ChatStore] New session created: '%s'", session_id)
            except Exception as e:
                logger.error("[ChatStore] insert error: %s", e)
        return new_doc

    def delete_session(self, session_id: str):
        """Permanently delete a session from MongoDB."""
        if self._connected:
            self._col.delete_one({"_id": session_id})
            logger.info("[ChatStore] Session deleted: '%s'", session_id)

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return most recently active sessions (metadata only)."""
        if not self._connected:
            return []
        try:
            cursor = self._col.find(
                {},
                {"_id": 1, "created_at": 1, "last_active": 1, "turn_count": 1, "symptoms": 1}
            ).sort("last_active", -1).limit(limit)
            return list(cursor)
        except Exception as e:
            logger.error("[ChatStore] list_sessions error: %s", e)
            return []

    # ──────────────────────────────────────────────────────────────────────
    # Symptoms
    # ──────────────────────────────────────────────────────────────────────

    def add_symptoms(self, session_id: str, new_symptoms: List[str]) -> List[str]:
        """
        Add new symptoms to the session (deduplicated).

        Returns the full updated symptom list.
        """
        doc = self.get_or_create_session(session_id)
        existing = set(doc.get("symptoms", []))
        to_add = [s.lower().strip() for s in new_symptoms
                  if s.lower().strip() and s.lower().strip() not in existing]

        if to_add:
            updated = doc.get("symptoms", []) + to_add
            if self._connected:
                try:
                    self._col.update_one(
                        {"_id": session_id},
                        {
                            "$push": {"symptoms": {"$each": to_add}},
                            "$set": {"last_active": time.time()},
                        }
                    )
                except Exception as e:
                    logger.error("[ChatStore] add_symptoms error: %s", e)
            logger.debug("[ChatStore] [%s] Added symptoms: %s", session_id, to_add)
            return updated

        return doc.get("symptoms", [])

    def get_symptoms(self, session_id: str) -> List[str]:
        """Return all accumulated symptoms for a session."""
        doc = self.get_session(session_id)
        return doc.get("symptoms", []) if doc else []

    def clear_symptoms(self, session_id: str):
        """Reset accumulated symptoms (but keep message history)."""
        if self._connected:
            try:
                self._col.update_one(
                    {"_id": session_id},
                    {"$set": {"symptoms": [], "last_active": time.time()}}
                )
                logger.info("[ChatStore] Symptoms cleared for session '%s'", session_id)
            except Exception as e:
                logger.error("[ChatStore] clear_symptoms error: %s", e)

    # ──────────────────────────────────────────────────────────────────────
    # Message history
    # ──────────────────────────────────────────────────────────────────────

    def add_message(
        self,
        session_id: str,
        role: str,           # "user" or "system"
        text: str,
        turn: int,
        symptoms_this_turn: Optional[List[str]] = None,
    ):
        """
        Append a message to the session's history.

        Args:
            session_id: Session identifier
            role: "user" or "system"
            text: Message content
            turn: Turn number (same for user+system pair in one round)
            symptoms_this_turn: Symptoms extracted this turn (for user messages)
        """
        self.get_or_create_session(session_id)
        message = {
            "turn": turn,
            "role": role,
            "text": text,
            "ts": time.time(),
        }
        if symptoms_this_turn:
            message["symptoms_this_turn"] = symptoms_this_turn

        if self._connected:
            try:
                self._col.update_one(
                    {"_id": session_id},
                    {
                        "$push": {"messages": message},
                        "$inc": {"turn_count": 1} if role == "user" else {},
                        "$set": {"last_active": time.time()},
                    }
                )
            except Exception as e:
                logger.error("[ChatStore] add_message error: %s", e)

    def get_history(self, session_id: str, last_n: int = 20) -> List[Dict[str, Any]]:
        """
        Return the last N messages for a session.

        Args:
            session_id: Session identifier
            last_n: Maximum number of messages to return

        Returns:
            List of message dicts [{"role", "text", "turn", "ts", ...}]
        """
        doc = self.get_session(session_id)
        if not doc:
            return []
        messages = doc.get("messages", [])
        return messages[-last_n:] if last_n else messages

    def get_turn_count(self, session_id: str) -> int:
        """Return number of user turns in this session."""
        doc = self.get_session(session_id)
        return doc.get("turn_count", 0) if doc else 0


# ---------------------------------------------------------------------------
# Module-level singleton — shared across the whole process
# ---------------------------------------------------------------------------
_store: Optional[ChatStore] = None


def get_store() -> ChatStore:
    """Return the global ChatStore singleton (lazy init)."""
    global _store
    if _store is None:
        _store = ChatStore()
    return _store
