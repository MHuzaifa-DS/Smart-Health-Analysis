"""
routers/symptoms.py — Symptom analysis endpoints.
"""
from fastapi import APIRouter, HTTPException
from app.dependencies import CurrentUser, SupabaseClient, Pagination
from app.models.symptom import SymptomAnalysisRequest, PredictionResponse
from app.services import prediction_service
import structlog

log = structlog.get_logger()
router = APIRouter(prefix="/symptoms", tags=["Symptom Analysis"])

# Curated symptom catalogue for the frontend autocomplete
SYMPTOM_CATALOGUE = sorted([
    "fatigue", "weakness", "fever", "chills", "headache", "dizziness",
    "nausea", "vomiting", "loss of appetite", "weight loss", "weight gain",
    "frequent urination", "excessive thirst", "blurred vision", "dry mouth",
    "slow wound healing", "numbness in hands or feet", "tingling",
    "chest pain", "shortness of breath", "palpitations", "nosebleed",
    "pale skin", "cold hands and feet", "brittle nails", "hair loss",
    "jaundice", "abdominal pain", "bloating", "diarrhea", "constipation",
    "back pain", "joint pain", "muscle cramps", "swelling in legs",
    "cough", "sore throat", "difficulty swallowing", "night sweats",
    "anxiety", "depression", "confusion", "memory problems", "insomnia",
    "skin rash", "itching", "bruising easily",
])


@router.get("/list")
async def list_symptoms():
    """Return the full symptom catalogue for frontend autocomplete."""
    return {"symptoms": SYMPTOM_CATALOGUE, "total": len(SYMPTOM_CATALOGUE)}


@router.post("/analyze", response_model=PredictionResponse, status_code=201)
async def analyze_symptoms(
    request: SymptomAnalysisRequest,
    current_user: CurrentUser,
    supabase: SupabaseClient,
):
    """
    Submit symptoms for AI-powered disease prediction.

    Uses RAG (Gale Encyclopedia) + ML models to predict likely conditions.
    Results are saved to the user's health history.
    """
    # Supplement with profile data if not provided in request
    if not request.age and current_user.age:
        request.age = current_user.age
    if not request.gender and current_user.gender:
        request.gender = current_user.gender

    try:
        result = await prediction_service.analyze_symptoms(
            request=request,
            user_id=current_user.id,
            supabase=supabase,
        )
        return result
    except Exception as e:
        log.error("symptoms.analyze_error", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
