"""
services/recommendation_service.py — Generate and persist health recommendations.
"""
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import structlog

from app.models.symptom import (
    RecommendationResponse, RecommendedTest, RecommendedSpecialist
)

log = structlog.get_logger()

# ── Recommendation knowledge base ──────────────────────────────────────────────

DISEASE_RECOMMENDATIONS: Dict[str, Dict] = {
    "diabetes": {
        "keywords": ["diabetes", "diabetic", "type 2 diabetes", "hyperglycemia"],
        "tests": [
            {"test_name": "HbA1c", "reason": "Confirms long-term blood glucose control", "urgency": "urgent"},
            {"test_name": "Fasting Blood Glucose", "reason": "Baseline blood sugar measurement", "urgency": "urgent"},
            {"test_name": "Random Blood Glucose", "reason": "Current glucose level", "urgency": "urgent"},
            {"test_name": "Kidney Function Panel (BUN, Creatinine)", "reason": "Diabetes affects kidneys over time", "urgency": "routine"},
            {"test_name": "Lipid Panel", "reason": "Diabetes increases cardiovascular risk", "urgency": "routine"},
            {"test_name": "Urine Microalbumin", "reason": "Early sign of diabetic kidney disease", "urgency": "routine"},
            {"test_name": "Eye Exam (Fundoscopy)", "reason": "Diabetic retinopathy screening", "urgency": "routine"},
        ],
        "specialists": [
            {"specialty": "Endocrinologist", "reason": "Specialist in diabetes and hormonal disorders"},
            {"specialty": "Diabetologist", "reason": "Specialized diabetes management"},
            {"specialty": "Ophthalmologist", "reason": "Eye complications of diabetes"},
        ],
        "tips": [
            "Reduce consumption of refined sugars and high-glycemic foods",
            "Exercise at least 150 minutes of moderate activity per week",
            "Monitor blood glucose levels regularly if prescribed",
            "Maintain a healthy weight — even 5-10% weight loss significantly improves control",
            "Eat smaller, more frequent meals to stabilize blood sugar",
            "Stay well-hydrated with water rather than sugary drinks",
            "Take all medications as prescribed by your doctor",
            "Inspect your feet daily for any wounds or changes",
        ],
        "emergency_threshold": 0.90,
    },
    "hypertension": {
        "keywords": ["hypertension", "high blood pressure", "cardiovascular", "blood pressure"],
        "tests": [
            {"test_name": "Blood Pressure Monitoring (24-hour)", "reason": "Ambulatory BP to confirm hypertension", "urgency": "urgent"},
            {"test_name": "Lipid Panel", "reason": "Assess cardiovascular risk", "urgency": "urgent"},
            {"test_name": "ECG (Electrocardiogram)", "reason": "Check for heart effects of high BP", "urgency": "routine"},
            {"test_name": "Kidney Function Panel", "reason": "Hypertension can damage kidneys", "urgency": "routine"},
            {"test_name": "Urinalysis", "reason": "Detect kidney damage markers", "urgency": "routine"},
            {"test_name": "Echocardiogram", "reason": "Assess heart structure if long-standing hypertension", "urgency": "optional"},
        ],
        "specialists": [
            {"specialty": "Cardiologist", "reason": "Heart and blood pressure specialist"},
            {"specialty": "Nephrologist", "reason": "If kidney involvement is suspected"},
        ],
        "tips": [
            "Reduce sodium (salt) intake to less than 2,300 mg per day",
            "Follow the DASH diet (rich in fruits, vegetables, whole grains)",
            "Limit alcohol consumption — no more than 1-2 drinks per day",
            "Exercise regularly — aerobic activity helps lower blood pressure",
            "Quit smoking if applicable — smoking greatly elevates BP",
            "Manage stress through relaxation techniques, meditation, or yoga",
            "Monitor blood pressure at home as directed by your doctor",
            "Take all antihypertensive medications as prescribed",
        ],
        "emergency_threshold": 0.88,
    },
    "anemia": {
        "keywords": ["anemia", "anaemia", "iron deficiency", "iron-deficiency"],
        "tests": [
            {"test_name": "Complete Blood Count (CBC)", "reason": "Assess severity of anemia and blood cell types", "urgency": "urgent"},
            {"test_name": "Serum Iron & TIBC", "reason": "Determine if iron-deficiency type", "urgency": "urgent"},
            {"test_name": "Serum Ferritin", "reason": "Iron storage levels — best early indicator", "urgency": "urgent"},
            {"test_name": "Reticulocyte Count", "reason": "Assesses bone marrow response", "urgency": "routine"},
            {"test_name": "Vitamin B12 & Folate", "reason": "Rule out vitamin-deficiency anemia", "urgency": "routine"},
            {"test_name": "Peripheral Blood Smear", "reason": "Identifies red blood cell abnormalities", "urgency": "routine"},
        ],
        "specialists": [
            {"specialty": "Hematologist", "reason": "Blood disorder specialist"},
            {"specialty": "Gastroenterologist", "reason": "If GI blood loss is suspected as cause"},
            {"specialty": "Gynecologist", "reason": "If heavy menstrual periods are contributing"},
        ],
        "tips": [
            "Increase iron-rich foods: red meat, spinach, lentils, beans, fortified cereals",
            "Pair iron-rich foods with Vitamin C (e.g., orange juice) to enhance absorption",
            "Avoid tea and coffee with meals — tannins inhibit iron absorption",
            "Take iron supplements as prescribed — do not take with calcium or antacids",
            "If prescribed, take Vitamin B12 or folate supplements",
            "Report any unusual bleeding (heavy periods, black stools, blood in urine) to your doctor",
            "Rest when fatigued — anemia reduces oxygen-carrying capacity",
            "Follow up with blood tests to monitor treatment response",
        ],
        "emergency_threshold": 0.85,
    },
}


def _find_disease_config(disease_name: str) -> Optional[Dict]:
    """Match a disease name to a recommendation config."""
    disease_lower = disease_name.lower()
    for key, config in DISEASE_RECOMMENDATIONS.items():
        if any(kw in disease_lower for kw in config["keywords"]):
            return config
    return None


async def generate_and_save(
    prediction_id: str,
    user_id: str,
    disease: str,
    confidence_score: float,
    emergency: bool,
    supabase,
    lab_report_id: Optional[str] = None,
) -> Optional[str]:
    """Generate recommendations for a prediction and save to DB."""
    config = _find_disease_config(disease)
    if not config:
        log.warning("recommendations.no_config_found", disease=disease)
        return None

    recommendation_id = str(uuid.uuid4())

    # Select tests by urgency based on confidence
    if confidence_score >= 0.75:
        tests = config["tests"]  # all tests
    else:
        tests = [t for t in config["tests"] if t["urgency"] in ("urgent", "routine")]

    # Emergency override
    emergency_message = None
    if emergency or confidence_score >= config.get("emergency_threshold", 0.90):
        emergency_message = (
            f"High risk indicators detected for {disease}. "
            "Please seek immediate medical evaluation. "
            "Do not delay — contact your doctor or go to an emergency department."
        )

    rec_data = {
        "id": recommendation_id,
        "user_id": user_id,
        "prediction_id": prediction_id,
        "lab_report_id": lab_report_id,
        "recommended_tests": tests,
        "recommended_specialists": config["specialists"],
        "health_tips": config["tips"],
        "emergency_alert": bool(emergency_message),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        supabase.table("recommendations").insert(rec_data).execute()
    except Exception as e:
        log.error("recommendations.db_save_failed", error=str(e))

    return recommendation_id


def get_recommendations_response(rec_data: Dict) -> RecommendationResponse:
    """Convert DB row to RecommendationResponse model."""
    tests = [RecommendedTest(**t) for t in (rec_data.get("recommended_tests") or [])]
    specialists = [RecommendedSpecialist(**s) for s in (rec_data.get("recommended_specialists") or [])]

    return RecommendationResponse(
        recommendation_id=rec_data["id"],
        recommended_tests=tests,
        recommended_specialists=specialists,
        health_tips=rec_data.get("health_tips") or [],
        emergency_alert=rec_data.get("emergency_alert", False),
        emergency_message=rec_data.get("emergency_message"),
        created_at=rec_data["created_at"],
    )
