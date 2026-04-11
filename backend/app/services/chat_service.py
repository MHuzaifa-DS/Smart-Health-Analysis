"""
services/chat_service.py — Conversational symptom collection via Claude.

Flow per message:
  1. Load session history from DB
  2. Append user message
  3. Call Claude with special system prompt
  4. Parse response for <EXTRACT> or <EMERGENCY> tags
  5. If symptoms ready → trigger prediction pipeline
  6. Save updated session to DB
  7. Return assistant reply + any prediction result
"""
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

import anthropic
import structlog

from app.config import settings

log = structlog.get_logger()

# ── System prompt ──────────────────────────────────────────────────────────────

CHAT_SYSTEM_PROMPT = """You are a warm, empathetic medical intake assistant for a Smart Health Assistant app.
Your job is to have a natural, friendly conversation to understand what symptoms the patient is experiencing,
then trigger an AI-powered health analysis.

CONVERSATION RULES:
1. Ask ONE focused question per response — never overwhelm the user
2. Use plain everyday language — never medical jargon
3. Be warm and empathetic, like a caring friend who happens to know medicine
4. Keep responses SHORT — 2-4 sentences maximum
5. Start with an open question, then drill into: severity (1-10), duration, and any related symptoms
6. After 3-5 exchanges when you have enough information, output the extraction block

WHEN YOU HAVE ENOUGH SYMPTOMS (after 3-5 exchanges):
At the END of your message append this block (user won't see it — it's parsed by the system):

<EXTRACT>
{
  "symptoms": ["symptom1", "symptom2"],
  "severity": {"symptom1": 7},
  "duration_days": 14,
  "free_text": "brief summary of what patient described in their own words"
}
</EXTRACT>

SYMPTOM VOCABULARY — map what users say to these standard terms:
- "tired / exhausted / no energy" → "fatigue"
- "peeing a lot / bathroom frequently" → "frequent urination"
- "always thirsty / dry mouth" → "excessive thirst"
- "blurry / fuzzy vision" → "blurred vision"
- "head hurts / headache" → "headache"
- "dizzy / lightheaded" → "dizziness"
- "chest hurts / chest tight" → "chest pain"
- "can't breathe / short of breath" → "shortness of breath"
- "pale / look white" → "pale skin"
- "cold hands/feet" → "cold hands and feet"
- "heart racing / pounding" → "palpitations"
- "lost weight without trying" → "weight loss"
- "wounds slow to heal" → "slow wound healing"
- "numb / tingling" → "numbness in hands or feet"
- "nausea / feel sick" → "nausea"
- "throwing up" → "vomiting"
- "can't sleep" → "insomnia"
- "anxious / worried" → "anxiety"

EMERGENCY — If user mentions ANY of these, immediately output <EMERGENCY>true</EMERGENCY>
and advise them to call emergency services or go to ER RIGHT NOW:
- severe chest pain, heart attack symptoms
- difficulty breathing / can't breathe
- stroke symptoms (face drooping, arm weakness, speech difficulty)
- loss of consciousness
- severe allergic reaction
- uncontrolled bleeding

IMPORTANT: Only output <EXTRACT> when you genuinely have 2+ symptoms with some context.
Do not rush — a good conversation gets better predictions."""


# ── Pydantic-style dataclasses ─────────────────────────────────────────────────

class ChatMessage:
    def __init__(self, role: str, content: str, timestamp: str = None):
        self.role = role          # "user" or "assistant"
        self.content = content
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}

    def to_api_format(self) -> dict:
        """Format for Anthropic messages API (no timestamp)."""
        return {"role": self.role, "content": self.content}


# ── Core service ───────────────────────────────────────────────────────────────

async def process_message(
    user_message: str,
    session_id: Optional[str],
    user_id: str,
    supabase,
) -> dict:
    """
    Main entry point. Handles one user message turn.

    Returns:
    {
        "session_id": str,
        "reply": str,              # assistant message to display
        "session_status": str,     # collecting | analyzing | complete
        "prediction": dict | None, # set when analysis completes
        "emergency": bool,
    }
    """
    # ── 1. Load or create session ──────────────────────────────────────────────
    session = await _load_or_create_session(session_id, user_id, supabase)
    session_id = session["id"]

    if session["session_status"] == "complete":
        # Session already done — start a fresh one
        session = await _create_session(user_id, supabase)
        session_id = session["id"]

    # ── 2. Append user message ─────────────────────────────────────────────────
    messages: list = session.get("messages") or []
    user_msg = ChatMessage(role="user", content=user_message.strip())
    messages.append(user_msg.to_dict())

    log.info("chat.user_message", session_id=session_id, message_count=len(messages))

    # ── 3. Call Claude ─────────────────────────────────────────────────────────
    raw_reply = await _call_claude(messages)

    # ── 4. Parse special tags ──────────────────────────────────────────────────
    is_emergency = _check_emergency(raw_reply)
    extracted = _parse_extract_block(raw_reply)
    clean_reply = _strip_tags(raw_reply)

    # ── 5. Update session state ────────────────────────────────────────────────
    assistant_msg = ChatMessage(role="assistant", content=clean_reply)
    messages.append(assistant_msg.to_dict())

    # Merge any newly extracted symptoms
    accumulated_symptoms = list(session.get("extracted_symptoms") or [])
    accumulated_severity = dict(session.get("severity_scores") or {})
    duration_days = session.get("duration_days")

    if extracted:
        new_symptoms = extracted.get("symptoms", [])
        for s in new_symptoms:
            if s not in accumulated_symptoms:
                accumulated_symptoms.append(s)
        accumulated_severity.update(extracted.get("severity", {}))
        if extracted.get("duration_days"):
            duration_days = extracted["duration_days"]

    # Determine new status
    status = session["session_status"]
    prediction_result = None

    if is_emergency:
        status = "complete"

    elif extracted and extracted.get("symptoms") and len(accumulated_symptoms) >= 1:
        # Ready to run prediction
        status = "analyzing"

        # ── 6. Run RAG+ML prediction ───────────────────────────────────────────
        try:
            prediction_result = await _run_prediction(
                symptoms=accumulated_symptoms,
                severity=accumulated_severity,
                duration_days=duration_days,
                free_text=extracted.get("free_text", ""),
                user_id=user_id,
                supabase=supabase,
            )

            # Append the prediction summary as the next assistant message
            prediction_summary = _format_prediction_for_chat(prediction_result)
            summary_msg = ChatMessage(role="assistant", content=prediction_summary)
            messages.append(summary_msg.to_dict())
            clean_reply = clean_reply + "\n\n" + prediction_summary

            status = "complete"

            # Update prediction_id in session
            session["prediction_id"] = prediction_result.get("prediction_id")

        except Exception as e:
            log.error("chat.prediction_failed", session_id=session_id, error=str(e))
            # Don't crash the chat — just note it
            error_msg = (
                "\n\n⚠️ I wasn't able to run the full analysis right now. "
                "Your symptoms have been saved. Please try the symptom analysis again shortly."
            )
            clean_reply += error_msg
            status = "collecting"

    # ── 7. Save session to DB ──────────────────────────────────────────────────
    await _save_session(
        session_id=session_id,
        user_id=user_id,
        messages=messages,
        extracted_symptoms=accumulated_symptoms,
        severity_scores=accumulated_severity,
        duration_days=duration_days,
        status=status,
        prediction_id=session.get("prediction_id"),
        supabase=supabase,
    )

    log.info(
        "chat.turn_complete",
        session_id=session_id,
        status=status,
        symptoms_count=len(accumulated_symptoms),
        emergency=is_emergency,
    )

    return {
        "session_id": session_id,
        "reply": clean_reply,
        "session_status": status,
        "extracted_symptoms": accumulated_symptoms,
        "prediction": prediction_result,
        "emergency": is_emergency,
    }


# ── Claude API call ────────────────────────────────────────────────────────────

async def _call_claude(messages: list) -> str:
    """Call Anthropic Claude with full conversation history."""
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    # Convert stored messages to API format (strip timestamps)
    api_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m["role"] in ("user", "assistant")
    ]

    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=600,   # keep replies short for chat UX
        system=CHAT_SYSTEM_PROMPT,
        messages=api_messages,
    )
    return response.content[0].text


# ── Tag parsers ────────────────────────────────────────────────────────────────

_EXTRACT_PATTERN = re.compile(r"<EXTRACT>(.*?)</EXTRACT>", re.DOTALL)
_EMERGENCY_PATTERN = re.compile(r"<EMERGENCY>\s*true\s*</EMERGENCY>", re.IGNORECASE)


def _parse_extract_block(text: str) -> Optional[dict]:
    """Extract the JSON payload from <EXTRACT>...</EXTRACT> if present."""
    match = _EXTRACT_PATTERN.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(1).strip())
        # Validate minimally
        if not isinstance(data.get("symptoms"), list):
            return None
        return data
    except (json.JSONDecodeError, AttributeError):
        log.warning("chat.extract_parse_failed", raw=match.group(1)[:200])
        return None


def _check_emergency(text: str) -> bool:
    return bool(_EMERGENCY_PATTERN.search(text))


def _strip_tags(text: str) -> str:
    """Remove all <EXTRACT> and <EMERGENCY> blocks from user-facing reply."""
    text = _EXTRACT_PATTERN.sub("", text)
    text = re.sub(r"<EMERGENCY>.*?</EMERGENCY>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


# ── Prediction integration ─────────────────────────────────────────────────────

async def _run_prediction(
    symptoms: list,
    severity: dict,
    duration_days: Optional[int],
    free_text: str,
    user_id: str,
    supabase,
) -> dict:
    """
    Call the existing prediction pipeline with extracted symptoms.
    Reuses prediction_service.analyze_symptoms — no duplication.
    """
    from app.models.symptom import SymptomAnalysisRequest
    from app.services.prediction_service import analyze_symptoms

    # Fetch user profile for age/gender context
    age, gender = None, None
    try:
        profile = supabase.table("profiles").select("age,gender").eq("id", user_id).single().execute()
        if profile.data:
            age = profile.data.get("age")
            gender = profile.data.get("gender")
    except Exception:
        pass

    request = SymptomAnalysisRequest(
        symptoms=symptoms,
        severity=severity or None,
        duration_days=duration_days,
        free_text=free_text or None,
        age=age,
        gender=gender,
    )

    result = await analyze_symptoms(
        request=request,
        user_id=user_id,
        supabase=supabase,
    )

    # Convert Pydantic model to dict for JSON serialisation
    return result.model_dump(mode="json")


# ── Format prediction for chat display ────────────────────────────────────────

def _format_prediction_for_chat(prediction: dict) -> str:
    """Turn a PredictionResponse dict into a readable chat message."""
    lines = ["---", "🔬 **Health Analysis Complete**", ""]

    predictions = prediction.get("predictions", [])
    if not predictions:
        return (
            "---\n"
            "I wasn't able to identify a specific condition from the symptoms described. "
            "Please consult a healthcare professional for a proper evaluation.\n"
            f"_{prediction.get('disclaimer', '')}_"
        )

    for pred in predictions[:2]:   # show top 2
        confidence = pred["confidence"].upper()
        score_pct = round(pred["confidence_score"] * 100)
        disease = pred["disease"]
        explanation = pred.get("explanation", "")

        emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(confidence, "⚪")
        lines.append(f"{emoji} **{disease}** — {confidence} likelihood ({score_pct}%)")
        if explanation:
            lines.append(f"   _{explanation[:200]}_")
        lines.append("")

    # Recommended tests
    tests = prediction.get("recommended_tests", [])
    if tests:
        lines.append("📋 **Recommended Tests:**")
        for t in tests[:4]:
            lines.append(f"   • {t}")
        lines.append("")

    # Emergency
    if prediction.get("emergency"):
        lines.append("🚨 **URGENT:** " + (prediction.get("emergency_reason") or "Please seek immediate medical attention."))
        lines.append("")

    lines.append(f"⚠️ _{prediction.get('disclaimer', 'This is a preliminary assessment only. Please consult a doctor.')}_")

    return "\n".join(lines)


# ── DB helpers ─────────────────────────────────────────────────────────────────

async def _load_or_create_session(
    session_id: Optional[str],
    user_id: str,
    supabase,
) -> dict:
    """Load an existing session or create a new one."""
    if session_id:
        try:
            result = (
                supabase.table("chat_sessions")
                .select("*")
                .eq("id", session_id)
                .eq("user_id", user_id)
                .single()
                .execute()
            )
            if result.data:
                return result.data
        except Exception as e:
            log.warning("chat.session_load_failed", session_id=session_id, error=str(e))

    return await _create_session(user_id, supabase)


async def _create_session(user_id: str, supabase) -> dict:
    """Insert a new chat session row and return it."""
    now = datetime.now(timezone.utc).isoformat()
    new_session = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "messages": [],
        "extracted_symptoms": [],
        "severity_scores": {},
        "session_status": "collecting",
        "created_at": now,
        "updated_at": now,
    }
    result = supabase.table("chat_sessions").insert(new_session).execute()
    return result.data[0] if result.data else new_session


async def _save_session(
    session_id: str,
    user_id: str,
    messages: list,
    extracted_symptoms: list,
    severity_scores: dict,
    duration_days: Optional[int],
    status: str,
    prediction_id: Optional[str],
    supabase,
) -> None:
    """Persist the updated session state."""
    updates = {
        "messages": messages,
        "extracted_symptoms": extracted_symptoms,
        "severity_scores": severity_scores,
        "duration_days": duration_days,
        "session_status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if prediction_id:
        updates["prediction_id"] = prediction_id

    try:
        supabase.table("chat_sessions").update(updates).eq("id", session_id).execute()
    except Exception as e:
        log.error("chat.session_save_failed", session_id=session_id, error=str(e))


# ── Session history helper (used by router) ────────────────────────────────────

async def get_session(session_id: str, user_id: str, supabase) -> Optional[dict]:
    """Fetch a single session for the current user."""
    try:
        result = (
            supabase.table("chat_sessions")
            .select("*")
            .eq("id", session_id)
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        return result.data
    except Exception:
        return None


async def get_user_sessions(user_id: str, supabase, limit: int = 20) -> list:
    """Fetch recent chat sessions for the current user."""
    try:
        result = (
            supabase.table("chat_sessions")
            .select("id, session_status, extracted_symptoms, prediction_id, created_at, updated_at")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception as e:
        log.error("chat.get_sessions_failed", user_id=user_id, error=str(e))
        return []
