"""
routers/chat.py — Conversational symptom collection endpoints.

Endpoints:
  POST /chat/message          — Send a message, get a reply
  GET  /chat/sessions         — List user's chat sessions
  GET  /chat/sessions/{id}    — Get full session with message history
  POST /chat/sessions/{id}/end — Manually end/abandon a session
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from app.dependencies import CurrentUser, SupabaseClient
from app.services import chat_service
import structlog

log = structlog.get_logger()
router = APIRouter(prefix="/chat", tags=["Chatbot"])


# ── Request / Response models ──────────────────────────────────────────────────

class ChatMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User's message")
    session_id: Optional[str] = Field(
        None,
        description="Existing session ID to continue. Omit to start a new session."
    )


class ChatMessageResponse(BaseModel):
    session_id: str
    reply: str
    session_status: str          # collecting | analyzing | complete
    extracted_symptoms: list
    prediction: Optional[dict]   # full prediction payload when analysis completes
    emergency: bool


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/message", response_model=ChatMessageResponse)
async def send_message(
    request: ChatMessageRequest,
    current_user: CurrentUser,
    supabase: SupabaseClient,
):
    """
    Send a message to the health chatbot.

    The bot will:
    1. Respond conversationally to collect symptom information
    2. Ask follow-up questions about severity and duration
    3. Automatically trigger RAG+ML analysis when enough info is gathered
    4. Return the prediction inline in the chat

    Start a new session by omitting `session_id`.
    Continue a session by passing the `session_id` from a previous response.

    Example first message: "I've been feeling really tired and I keep needing to pee"
    """
    try:
        result = await chat_service.process_message(
            user_message=request.message,
            session_id=request.session_id,
            user_id=current_user.id,
            supabase=supabase,
        )
        return ChatMessageResponse(**result)

    except Exception as e:
        log.error("chat.message_error", user_id=current_user.id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Chat service encountered an error. Please try again.",
        )


@router.get("/sessions")
async def list_sessions(
    current_user: CurrentUser,
    supabase: SupabaseClient,
    limit: int = 20,
):
    """
    List the current user's recent chat sessions.
    Returns session metadata (status, extracted symptoms, dates) without full message history.
    """
    sessions = await chat_service.get_user_sessions(
        user_id=current_user.id,
        supabase=supabase,
        limit=min(limit, 50),
    )
    return {"sessions": sessions, "total": len(sessions)}


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    current_user: CurrentUser,
    supabase: SupabaseClient,
):
    """
    Get the full message history of a specific chat session.
    Includes all conversation turns and extracted symptoms.
    """
    session = await chat_service.get_session(
        session_id=session_id,
        user_id=current_user.id,
        supabase=supabase,
    )
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    return session


@router.post("/sessions/{session_id}/end", status_code=200)
async def end_session(
    session_id: str,
    current_user: CurrentUser,
    supabase: SupabaseClient,
):
    """
    Manually abandon a chat session.
    Use when the user wants to start fresh without completing analysis.
    """
    session = await chat_service.get_session(
        session_id=session_id,
        user_id=current_user.id,
        supabase=supabase,
    )
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    if session["session_status"] == "complete":
        return {"message": "Session already complete.", "session_id": session_id}

    try:
        supabase.table("chat_sessions").update(
            {"session_status": "abandoned"}
        ).eq("id", session_id).execute()
        return {"message": "Session ended.", "session_id": session_id}
    except Exception as e:
        log.error("chat.end_session_failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to end session.")
