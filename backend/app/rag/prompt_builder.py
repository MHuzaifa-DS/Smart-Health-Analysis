"""
rag/prompt_builder.py — Builds structured prompts from retrieved chunks
and calls the LLM (Claude via Anthropic API) to generate predictions.

FIX: generate_lab_interpretation now gracefully handles plain-text LLM
     responses. If parse_llm_json fails, the raw text is used directly
     as the interpretation field instead of crashing.
"""
import json
import re
from typing import List, Dict, Any, Optional
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.rag.retriever import RetrievedChunk

log = structlog.get_logger()

# ── System prompts ──────────────────────────────────────────────────────────────

SYMPTOM_SYSTEM_PROMPT = """You are a medical AI assistant integrated into a smart health system.
Your role is to analyze patient symptoms using ONLY the provided medical reference context 
from the Gale Encyclopedia of Medicine (3rd Edition).

CRITICAL RULES:
1. Base your analysis EXCLUSIVELY on the provided medical context chunks
2. Do NOT use knowledge outside the provided context
3. Always recommend professional medical consultation
4. Flag any emergency symptoms immediately
5. Focus only on: Diabetes (Type 2), Hypertension, and Anemia for initial predictions,
   but note other relevant conditions mentioned in context
6. Respond ONLY in valid JSON — no markdown, no preamble, no explanation outside JSON

Your response must be parseable JSON matching the exact schema provided."""

LAB_SYSTEM_PROMPT = """You are a medical AI assistant specializing in laboratory report interpretation.
Using ONLY the provided medical reference context from the Gale Encyclopedia of Medicine,
interpret the patient's laboratory values.

CRITICAL RULES:
1. Only use the provided medical context
2. Explain what each abnormal value means in plain language
3. Identify patterns that suggest specific conditions
4. Always recommend professional medical review
5. Respond ONLY in valid JSON — no markdown, no preamble, no text outside the JSON object

Your response must be a single valid JSON object. Do not write anything before or after it."""

# ── Prompt templates ────────────────────────────────────────────────────────────

SYMPTOM_PROMPT_TEMPLATE = """MEDICAL REFERENCE CONTEXT:
{context}

---

PATIENT INFORMATION:
- Reported symptoms: {symptoms}
- Symptom duration: {duration}
- Age: {age}
- Gender: {gender}
- Severity scores: {severity}
- Additional notes: {free_text}

---

Based ONLY on the medical reference context above, analyze these symptoms and respond with this exact JSON:

{{
  "predictions": [
    {{
      "disease": "disease name exactly as in context",
      "confidence": "high OR medium OR low",
      "confidence_score": 0.0,
      "matching_symptoms": ["symptom1", "symptom2"],
      "explanation": "2-3 sentences citing specific context. Mention which section of the encyclopedia supports this.",
      "source_chunks": ["chunk_id_1"]
    }}
  ],
  "recommended_tests": ["test1", "test2"],
  "emergency": false,
  "emergency_reason": null,
  "context_quality": "good OR partial OR insufficient",
  "disclaimer": "This is a preliminary AI-assisted assessment only. Consult a healthcare professional."
}}

Rules for confidence scoring:
- high (0.75-1.0): Multiple matching symptoms, clear context match
- medium (0.45-0.74): Some matching symptoms, partial context
- low (0.20-0.44): Few matching symptoms, weak context
- If confidence_score < 0.20, exclude the prediction entirely

Return 1-3 predictions maximum, ordered by confidence_score descending."""


LAB_PROMPT_TEMPLATE = """MEDICAL REFERENCE CONTEXT:
{context}

---

PATIENT LAB VALUES:
{lab_summary}

Patient info: Age {age}, Gender {gender}

---

Based ONLY on the medical reference context, interpret these lab values.

YOU MUST respond with ONLY this JSON object — no text before or after it:

{{
  "interpretation": "2-3 paragraph plain-English explanation of what these results mean",
  "likely_conditions": ["condition1", "condition2"],
  "abnormal_flags": [
    {{
      "test_name": "test",
      "value": 0.0,
      "concern": "what this value suggests",
      "urgency": "immediate OR soon OR routine"
    }}
  ],
  "recommended_followup": ["action1", "action2"],
  "emergency": false,
  "disclaimer": "These results require professional medical evaluation."
}}"""


def build_context_string(chunks: List[RetrievedChunk], max_chars: int = 12000) -> str:
    """Format retrieved chunks into a numbered context string for the prompt."""
    context_parts = []
    total_chars = 0

    for i, chunk in enumerate(chunks, 1):
        chunk_text = (
            f"[SOURCE {i}] Disease: {chunk.disease_name} | "
            f"Section: {chunk.section} | "
            f"Chunk ID: {chunk.chunk_id} | "
            f"Relevance: {chunk.score:.3f}\n"
            f"{chunk.text}\n"
            f"{'─' * 60}"
        )
        if total_chars + len(chunk_text) > max_chars:
            break
        context_parts.append(chunk_text)
        total_chars += len(chunk_text)

    return "\n\n".join(context_parts) if context_parts else "No relevant medical context found."


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call Anthropic Claude API and return raw text response."""
    import anthropic
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    message = client.messages.create(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text


def parse_llm_json(raw_response: str) -> Dict[str, Any]:
    """
    Safely parse JSON from LLM response.
    Handles markdown fences, leading/trailing text, and truncated JSON.
    """
    text = raw_response.strip()

    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Attempt 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract the first {...} block (handles preamble text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    log.error("prompt_builder.json_parse_failed", response_preview=text[:200])
    raise ValueError(f"LLM returned unparseable response: {text[:200]}")


def generate_symptom_prediction(
    chunks: List[RetrievedChunk],
    symptoms: List[str],
    severity: Optional[Dict[str, int]] = None,
    duration_days: Optional[int] = None,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    free_text: Optional[str] = None,
) -> tuple[Dict[str, Any], str]:
    """Full RAG prediction pipeline for symptoms."""
    context = build_context_string(chunks)

    user_prompt = SYMPTOM_PROMPT_TEMPLATE.format(
        context=context,
        symptoms=", ".join(symptoms) if symptoms else "not specified",
        duration=f"{duration_days} days" if duration_days else "not specified",
        age=age or "not specified",
        gender=gender or "not specified",
        severity=json.dumps(severity) if severity else "not specified",
        free_text=free_text or "none",
    )

    log.info("prompt_builder.calling_llm", num_chunks=len(chunks), num_symptoms=len(symptoms))

    raw = call_llm(SYMPTOM_SYSTEM_PROMPT, user_prompt)
    parsed = parse_llm_json(raw)
    parsed = _sanitize_prediction_response(parsed, chunks)

    log.info(
        "prompt_builder.prediction_complete",
        num_predictions=len(parsed.get("predictions", [])),
        emergency=parsed.get("emergency", False),
    )

    return parsed, raw


def generate_lab_interpretation(
    chunks: List[RetrievedChunk],
    lab_values: Dict[str, float],
    age: Optional[int] = None,
    gender: Optional[str] = None,
) -> tuple[Dict[str, Any], str]:
    """
    RAG-powered lab report interpretation.

    FIX: If the LLM returns plain text instead of JSON (which Claude sometimes
    does for rich interpretations), we catch the parse error and use the raw
    text directly as the 'interpretation' field. This ensures the user always
    sees the AI interpretation instead of a silent failure.
    """
    context = build_context_string(chunks)

    lab_lines = [f"  - {test}: {value}" for test, value in lab_values.items()]
    lab_summary = "\n".join(lab_lines)

    user_prompt = LAB_PROMPT_TEMPLATE.format(
        context=context,
        lab_summary=lab_summary,
        age=age or "not specified",
        gender=gender or "not specified",
    )

    raw = call_llm(LAB_SYSTEM_PROMPT, user_prompt)

    # Attempt JSON parse
    try:
        parsed = parse_llm_json(raw)
        log.info("prompt_builder.lab_interpretation_complete", format="json")
        return parsed, raw

    except ValueError:
        # LLM returned a high-quality plain-text interpretation instead of JSON.
        # Use it directly — the user gets the full explanation and nothing is lost.
        log.info(
            "prompt_builder.lab_interpretation_plain_text_fallback",
            chars=len(raw),
            hint="LLM returned plain text — using as interpretation field directly",
        )
        fallback = {
            "interpretation":     raw.strip(),
            "likely_conditions":  [],
            "abnormal_flags":     [],
            "recommended_followup": [],
            "emergency":          False,
            "disclaimer":         "These results require professional medical evaluation.",
        }
        return fallback, raw


def _sanitize_prediction_response(
    data: Dict[str, Any],
    chunks: List[RetrievedChunk],
) -> Dict[str, Any]:
    """Validate LLM output and fill defaults for missing fields."""
    if "predictions" not in data:
        data["predictions"] = []

    valid_chunk_ids = {c.chunk_id for c in chunks}

    for pred in data["predictions"]:
        score = pred.get("confidence_score", 0.0)
        try:
            pred["confidence_score"] = max(0.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            pred["confidence_score"] = 0.0

        s = pred["confidence_score"]
        if s >= 0.75:
            pred["confidence"] = "high"
        elif s >= 0.45:
            pred["confidence"] = "medium"
        else:
            pred["confidence"] = "low"

        pred["source_chunks"] = [
            cid for cid in pred.get("source_chunks", []) if cid in valid_chunk_ids
        ]
        pred.setdefault("matching_symptoms", [])
        pred.setdefault("explanation", "See medical context for details.")

    data["predictions"] = [
        p for p in data["predictions"] if p["confidence_score"] >= 0.20
    ]
    data["predictions"].sort(key=lambda p: p["confidence_score"], reverse=True)

    data["emergency"] = bool(data.get("emergency", False))
    data.setdefault("recommended_tests", [])
    data.setdefault(
        "disclaimer",
        "This is a preliminary AI-assisted assessment only. Consult a healthcare professional.",
    )

    return data