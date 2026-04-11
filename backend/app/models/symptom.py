"""models/symptom.py — Symptom and prediction request/response schemas."""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


# ── Symptom Analysis ───────────────────────────────────────────────────────────

class SymptomAnalysisRequest(BaseModel):
    symptoms: List[str] = Field(..., min_length=1, max_length=30)
    severity: Optional[Dict[str, int]] = None        # symptom → 1-10 score
    duration_days: Optional[int] = Field(None, ge=0, le=3650)
    free_text: Optional[str] = Field(None, max_length=1000)
    age: Optional[int] = Field(None, ge=1, le=149)
    gender: Optional[str] = None

    @field_validator("symptoms")
    @classmethod
    def clean_symptoms(cls, v):
        return [s.strip().lower() for s in v if s.strip()]

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v):
        if v:
            for symptom, score in v.items():
                if not (1 <= score <= 10):
                    raise ValueError(f"Severity for '{symptom}' must be 1-10")
        return v


class DiseasePrediction(BaseModel):
    disease: str
    confidence: str                        # "high" | "medium" | "low"
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    matching_symptoms: List[str]
    explanation: str
    source_chunks: List[str] = []          # Pinecone chunk IDs cited


class PredictionResponse(BaseModel):
    prediction_id: str
    predictions: List[DiseasePrediction]
    recommended_tests: List[str] = []
    emergency: bool = False
    emergency_reason: Optional[str] = None
    prediction_method: str                 # "rag_ml_combined" | "rag_only" | "ml_only"
    disclaimer: str = (
        "This is a preliminary AI-assisted assessment only. "
        "It does not constitute medical diagnosis. Please consult a qualified "
        "healthcare professional for proper evaluation."
    )
    created_at: datetime


class PredictionHistoryItem(BaseModel):
    id: str
    top_disease: str
    top_confidence: float
    risk_level: str
    prediction_method: str
    created_at: datetime


# ── Lab Report ─────────────────────────────────────────────────────────────────

class LabValue(BaseModel):
    value: float
    unit: str


class LabReportRequest(BaseModel):
    report_type: str = "blood_test"
    values: Dict[str, float]               # test_name → numeric value
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    notes: Optional[str] = None


class LabTestResult(BaseModel):
    test_name: str
    value: float
    unit: str
    status: str                            # "normal"|"low"|"high"|"critical_low"|"critical_high"
    normal_range: str
    interpretation: str
    emergency: bool = False


class LabReportResponse(BaseModel):
    report_id: str
    report_type: str
    results: List[LabTestResult]
    overall_status: str                    # "normal"|"borderline"|"abnormal"|"critical"
    likely_conditions: List[str]
    rag_interpretation: Optional[str] = None
    created_at: datetime


# ── Recommendations ────────────────────────────────────────────────────────────

class RecommendedTest(BaseModel):
    test_name: str
    reason: str
    urgency: str                           # "urgent"|"routine"|"optional"


class RecommendedSpecialist(BaseModel):
    specialty: str
    reason: str


class RecommendationResponse(BaseModel):
    recommendation_id: str
    recommended_tests: List[RecommendedTest]
    recommended_specialists: List[RecommendedSpecialist]
    health_tips: List[str]
    emergency_alert: bool
    emergency_message: Optional[str] = None
    created_at: datetime


# ── Dashboard ──────────────────────────────────────────────────────────────────

class HealthMetricPoint(BaseModel):
    recorded_at: datetime
    value: float
    unit: str


class DashboardSummary(BaseModel):
    total_checks: int
    last_check_date: Optional[datetime]
    recent_predictions: List[PredictionHistoryItem]
    recent_lab_reports: List[Dict[str, Any]]
    health_score: Optional[float] = None   # 0-100 composite score


# ── RAG Sources ────────────────────────────────────────────────────────────────

class RAGSource(BaseModel):
    chunk_id: str
    disease_name: str
    section: str
    text: str
    similarity_score: float
    page_number: Optional[int] = None
