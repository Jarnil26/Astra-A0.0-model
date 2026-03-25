"""
chat/routes.py — /chat/* endpoints
All require Bearer JWT authentication.
"""
import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional

from auth.routes import get_current_user
import chat.service as svc

logger = logging.getLogger("astra.chat.routes")
router = APIRouter(prefix="/chat", tags=["Chat"])


# ── Request models ────────────────────────────────────────────────────────────

class NewSessionRequest(BaseModel):
    title: Optional[str] = Field(None, example="My health check", description="Optional title; auto-generated if not provided")


class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, example="mane tav 6 ane mathu dukhe 6")


class RenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/new", summary="Create a new chat session")
def new_session(req: NewSessionRequest, user=Depends(get_current_user)):
    """
    Creates a blank session. Returns `session_id` to use in subsequent messages.
    """
    session = svc.create_session(
        user_id=str(user["_id"]),
        title=req.title or "New Chat",
    )
    return {
        "session_id": session["_id"],
        "title": session["title"],
        "created_at": session["created_at"],
    }


@router.post("/{session_id}/message", summary="Send a message in a session")
def send_message(session_id: str, req: MessageRequest, user=Depends(get_current_user)):
    """
    Processes one turn in the conversation.

    - Supports any language (English, Hindi, Gujarati, Hinglish)
    - Accumulates symptoms across turns automatically
    - Turn 1: 'I have fever' → diagnoses fever
    - Turn 2: 'I also have headache' → diagnoses fever + headache
    - Re-opening an old session resumes exactly where it left off
    """
    try:
        result = svc.send_message(
            session_id=session_id,
            user_id=str(user["_id"]),
            message=req.message,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Error in session %s: %s", session_id, e)
        raise HTTPException(status_code=500, detail=f"Engine error: {e}")
    return result


@router.get("/sessions", summary="List all your chat sessions (side panel)")
def list_sessions(user=Depends(get_current_user)):
    """
    Returns all sessions for the authenticated user, sorted most-recent first.
    Each item includes: session_id, title, preview, message_count, symptom_count.
    """
    sessions = svc.list_sessions(user_id=str(user["_id"]))
    return {"sessions": sessions, "count": len(sessions)}


@router.get("/{session_id}", summary="Get full session history")
def get_session(session_id: str, user=Depends(get_current_user)):
    """
    Returns the full session: title, all messages, and accumulated symptoms.
    Use to re-open a previous conversation in the UI.
    """
    try:
        return svc.get_history(session_id, user_id=str(user["_id"]))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{session_id}", summary="Delete a chat session")
def delete_session(session_id: str, user=Depends(get_current_user)):
    deleted = svc.delete_session(session_id, user_id=str(user["_id"]))
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"message": "Session deleted.", "session_id": session_id}


@router.patch("/{session_id}/title", summary="Rename a chat session")
def rename_session(session_id: str, req: RenameRequest, user=Depends(get_current_user)):
    updated = svc.rename_session(session_id, user_id=str(user["_id"]), new_title=req.title)
    if not updated:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"message": "Session renamed.", "session_id": session_id, "title": req.title}
