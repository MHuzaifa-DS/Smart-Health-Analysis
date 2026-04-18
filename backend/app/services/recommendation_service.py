"""
services/recommendation_service.py — Generate and persist health recommendations.

FIX: Added GENERIC_FALLBACK config so any disease name returned by RAG
     (e.g. "Fatigue", "Influenza", "Viral Infection") still produces
     useful recommendations instead of logging no_config_found and
     returning None with no DB row written.
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
        "keywords": ["diabetes", "diabetic", "type 2 diabetes", "hyperglycemia", "blood sugar"],
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
            "Report any unusual bleeding to your doctor",
            "Rest when fatigued — anemia reduces oxygen-carrying capacity",
            "Follow up with blood tests to monitor treatment response",
        ],
        "emergency_threshold": 0.85,
    },
    "influenza": {
        "keywords": ["influenza", "flu", "viral infection", "viral illness", "respiratory infection"],
        "tests": [
            {"test_name": "Influenza Rapid Antigen Test", "reason": "Confirms influenza A or B within 15 minutes", "urgency": "urgent"},
            {"test_name": "Complete Blood Count (CBC)", "reason": "Assess severity of infection and immune response", "urgency": "routine"},
            {"test_name": "Chest X-ray", "reason": "Rule out pneumonia if symptoms are severe", "urgency": "routine"},
        ],
        "specialists": [
            {"specialty": "General Practitioner", "reason": "Primary care management of flu symptoms"},
            {"specialty": "Pulmonologist", "reason": "If respiratory symptoms are severe or worsening"},
        ],
        "tips": [
            "Rest as much as possible to help your body fight the infection",
            "Stay well-hydrated — drink water, clear broths, and electrolyte drinks",
            "Take paracetamol or ibuprofen for fever and body aches as directed",
            "Isolate yourself to avoid spreading the virus to others",
            "Seek immediate care if you experience difficulty breathing, chest pain, or confusion",
            "Most flu resolves in 7-10 days — if symptoms worsen after day 5, see a doctor",
        ],
        "emergency_threshold": 0.92,
    },
    "fatigue": {
        "keywords": [
            "fatigue", "chronic fatigue", "lethargy", "tiredness", "exhaustion",
            "general fatigue", "fatigue (general", "environmental", "constitutional",
        ],
        "tests": [
            {"test_name": "Complete Blood Count (CBC)", "reason": "Rule out anemia as a cause of fatigue", "urgency": "urgent"},
            {"test_name": "Thyroid Function Tests (TSH, T3, T4)", "reason": "Thyroid disorders are a common cause of fatigue", "urgency": "urgent"},
            {"test_name": "Fasting Blood Glucose", "reason": "Rule out diabetes-related fatigue", "urgency": "routine"},
            {"test_name": "Vitamin B12 & Folate", "reason": "Deficiencies cause significant fatigue", "urgency": "routine"},
            {"test_name": "Vitamin D Level", "reason": "Low Vitamin D is strongly linked to fatigue", "urgency": "routine"},
            {"test_name": "Liver Function Tests", "reason": "Liver issues can cause persistent tiredness", "urgency": "routine"},
            {"test_name": "Sleep Study (Polysomnography)", "reason": "Rule out sleep apnea if fatigue is chronic", "urgency": "optional"},
        ],
        "specialists": [
            {"specialty": "General Practitioner", "reason": "First point of contact for fatigue workup"},
            {"specialty": "Endocrinologist", "reason": "If thyroid or hormonal cause is suspected"},
            {"specialty": "Sleep Specialist", "reason": "If sleep disturbance is contributing"},
        ],
        "tips": [
            "Ensure you are getting 7-9 hours of quality sleep per night",
            "Eat balanced meals with adequate protein, iron, and B vitamins",
            "Limit caffeine — it can disrupt sleep quality",
            "Exercise regularly — even light walking improves energy levels",
            "Stay hydrated — even mild dehydration causes fatigue",
            "Manage stress — chronic stress is a leading cause of exhaustion",
            "If fatigue persists more than 2 weeks, see a doctor for a full workup",
        ],
        "emergency_threshold": 0.95,
    },
    "fever": {
        "keywords": ["fever", "pyrexia", "high temperature", "febrile"],
        "tests": [
            {"test_name": "Complete Blood Count (CBC)", "reason": "Identify infection or immune response", "urgency": "urgent"},
            {"test_name": "Blood Culture", "reason": "Identify bacterial infection if fever is high", "urgency": "urgent"},
            {"test_name": "Urinalysis & Urine Culture", "reason": "Rule out urinary tract infection", "urgency": "routine"},
            {"test_name": "Malaria Rapid Test", "reason": "If travel to endemic area in past 3 months", "urgency": "routine"},
            {"test_name": "Chest X-ray", "reason": "Rule out pneumonia", "urgency": "routine"},
        ],
        "specialists": [
            {"specialty": "General Practitioner", "reason": "Initial assessment and management"},
            {"specialty": "Infectious Disease Specialist", "reason": "If fever source is unidentified after workup"},
        ],
        "tips": [
            "Take paracetamol or ibuprofen to reduce fever as directed",
            "Drink plenty of fluids to prevent dehydration",
            "Rest and avoid strenuous activity",
            "Seek immediate care if fever exceeds 39.5°C (103°F) or lasts more than 3 days",
            "Seek emergency care for fever with stiff neck, confusion, or rash",
        ],
        "emergency_threshold": 0.90,
    },
}

# ── Generic fallback — used when RAG returns a disease not in the list above ──
GENERIC_FALLBACK: Dict = {
    "tests": [
        {"test_name": "Complete Blood Count (CBC)", "reason": "General health screening and infection detection", "urgency": "routine"},
        {"test_name": "Basic Metabolic Panel", "reason": "Assess kidney, liver and electrolyte function", "urgency": "routine"},
        {"test_name": "Thyroid Function Tests", "reason": "Thyroid issues can underlie many symptoms", "urgency": "routine"},
    ],
    "specialists": [
        {"specialty": "General Practitioner", "reason": "A GP can assess your symptoms and direct further investigation"},
    ],
    "tips": [
        "Rest and stay well-hydrated while your symptoms persist",
        "Track your symptoms — note when they started, severity, and any changes",
        "Avoid self-medicating without consulting a healthcare professional",
        "If symptoms worsen or new symptoms appear, seek medical attention promptly",
        "Bring a list of all your current medications to any doctor's appointment",
    ],
    "emergency_threshold": 0.95,
}


def _find_disease_config(disease_name: str) -> Dict:
    """
    Match a disease name to a recommendation config.
    Always returns a config — falls back to GENERIC_FALLBACK if no match found.
    """
    disease_lower = disease_name.lower()
    for key, config in DISEASE_RECOMMENDATIONS.items():
        if any(kw in disease_lower for kw in config["keywords"]):
            return config

    # No specific match — use generic fallback so recommendations are always saved
    log.info(
        "recommendations.using_generic_fallback",
        disease=disease_name,
        hint="Add this disease to DISEASE_RECOMMENDATIONS for specific recommendations",
    )
    return GENERIC_FALLBACK


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
    config = _find_disease_config(disease)   # always returns a config now

    recommendation_id = str(uuid.uuid4())

    # Select tests by urgency based on confidence
    if confidence_score >= 0.75:
        tests = config["tests"]
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
        "id":                     recommendation_id,
        "user_id":                user_id,
        "prediction_id":          prediction_id,
        "lab_report_id":          lab_report_id,
        "recommended_tests":      tests,
        "recommended_specialists": config["specialists"],
        "health_tips":            config["tips"],
        "emergency_alert":        bool(emergency_message),
        "emergency_message":      emergency_message,
        "created_at":             datetime.now(timezone.utc).isoformat(),
    }

    try:
        supabase.table("recommendations").insert(rec_data).execute()
        log.info("recommendations.saved", id=recommendation_id, disease=disease)
    except Exception as e:
        log.error("recommendations.db_save_failed", error=str(e))

    return recommendation_id


def get_recommendations_response(rec_data: Dict) -> RecommendationResponse:
    """Convert DB row to RecommendationResponse model."""
    tests       = [RecommendedTest(**t) for t in (rec_data.get("recommended_tests") or [])]
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