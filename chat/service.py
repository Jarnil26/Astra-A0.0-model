"""
chat/service.py — ChatSession CRUD + Astra A0 engine integration
Stores sessions + messages in chat MongoDB.
"""
import uuid
import time
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("astra.chat")


def _col():
    from db.chat_db import get_sessions_collection
    return get_sessions_collection()


def _engine():
    from clinical.engine import ClinicalEngine
    return ClinicalEngine.instance()


# ─────────────────────────────────────────────────────────────────────────────
# Session lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def create_session(user_id: str, title: str = "New Chat") -> Dict[str, Any]:
    """Create a blank chat session for the user."""
    now = time.time()
    doc = {
        "_id": str(uuid.uuid4()),
        "user_id": user_id,
        "title": title,
        "created_at": now,
        "last_active": now,
        "symptoms": [],       # accumulated symptom list across this session
        "messages": [],
    }
    _col().insert_one(doc)
    logger.info("New session %s created for user %s", doc["_id"], user_id)
    return doc


def get_session(session_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Return session if it belongs to the user."""
    return _col().find_one({"_id": session_id, "user_id": user_id})


def list_sessions(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Return user's sessions sorted by most recent activity.
    Each item has: _id, title, created_at, last_active, symptom_count, message_count
    """
    cursor = _col().find(
        {"user_id": user_id},
        {"_id": 1, "title": 1, "created_at": 1, "last_active": 1,
         "symptoms": 1, "messages": 1}
    ).sort("last_active", -1).limit(limit)

    sessions = []
    for s in cursor:
        msgs = s.get("messages", [])
        # Build preview from last user message
        preview = ""
        for m in reversed(msgs):
            if m["role"] == "user":
                preview = m["text"][:80]
                break
        sessions.append({
            "session_id": s["_id"],
            "title": s.get("title", "Untitled"),
            "last_active": s.get("last_active"),
            "created_at": s.get("created_at"),
            "message_count": len(msgs),
            "symptom_count": len(s.get("symptoms", [])),
            "preview": preview,
        })
    return sessions


def delete_session(session_id: str, user_id: str) -> bool:
    res = _col().delete_one({"_id": session_id, "user_id": user_id})
    return res.deleted_count > 0


def rename_session(session_id: str, user_id: str, new_title: str) -> bool:
    res = _col().update_one(
        {"_id": session_id, "user_id": user_id},
        {"$set": {"title": new_title.strip(), "last_active": time.time()}}
    )
    return res.modified_count > 0


# ─────────────────────────────────────────────────────────────────────────────
# Message handling (core chat turn)
# ─────────────────────────────────────────────────────────────────────────────

def send_message(session_id: str, user_id: str, message: str) -> Dict[str, Any]:
    """
    Process one user message turn:
    1. Load accumulated symptoms from MongoDB
    2. Run clinical engine (with symptom accumulation)
    3. Save user + system messages to MongoDB
    4. Update symptoms list + last_active
    5. Auto-title session from first meaningful message

    Returns the full turn result (intent, reply, prediction, etc.)
    """
    session = get_session(session_id, user_id)
    if not session:
        raise ValueError("Session not found or access denied.")

    accumulated = session.get("symptoms", [])
    messages = session.get("messages", [])
    turn = len([m for m in messages if m["role"] == "user"]) + 1

    # ── Run clinical engine ────────────────────────────────────────────────
    result = _engine().process_message(
        message=message,
        accumulated_symptoms=accumulated,
        session_id=f"{user_id}_{session_id}",   # unique per user+session
    )

    new_symptoms = result["all_symptoms"]  # already deduplicated

    # ── Build messages to append ───────────────────────────────────────────
    user_msg = {
        "role": "user",
        "text": message,
        "turn": turn,
        "ts": time.time(),
        "symptoms_this_turn": result["symptoms_this_turn"],
        "language": result["language"],
    }
    system_msg = {
        "role": "system",
        "text": result["reply"],
        "turn": turn,
        "ts": time.time() + 0.001,
        "intent": result["intent"],
    }
    if result.get("prediction"):
        system_msg["prediction"] = result["prediction"]

    # ── Auto-title: use first user message as session title ────────────────
    update_fields: Dict[str, Any] = {
        "last_active": time.time(),
        "symptoms": new_symptoms,
    }
    if not messages and turn == 1:
        auto_title = message[:50].strip()
        if result["symptoms_this_turn"]:
            auto_title = " + ".join(s.title() for s in result["symptoms_this_turn"][:3])
        update_fields["title"] = auto_title

    _col().update_one(
        {"_id": session_id},
        {
            "$push": {"messages": {"$each": [user_msg, system_msg]}},
            "$set": update_fields,
        }
    )

    return {
        "session_id": session_id,
        "turn": turn,
        **result,
    }


def get_history(session_id: str, user_id: str) -> Dict[str, Any]:
    """Return the full session document (title + all messages + symptoms)."""
    session = get_session(session_id, user_id)
    if not session:
        raise ValueError("Session not found.")
    return {
        "session_id": session["_id"],
        "title": session.get("title", "Untitled"),
        "created_at": session.get("created_at"),
        "last_active": session.get("last_active"),
        "symptoms": session.get("symptoms", []),
        "messages": session.get("messages", []),
    }
