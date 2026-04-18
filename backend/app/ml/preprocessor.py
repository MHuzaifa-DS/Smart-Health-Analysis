"""
ml/preprocessor.py — Feature engineering for the three disease models.

Each disease has its own feature set based on its training dataset:
  - Diabetes:     Pima Indians Diabetes Database features
  - Hypertension: Framingham Heart Study features  
  - Anemia:       UCI Anemia Dataset features
"""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger()

# ── Feature definitions per disease ────────────────────────────────────────────

DIABETES_FEATURES = [
    "pregnancies",         # number of pregnancies (0 for males)
    "glucose",             # fasting plasma glucose (mg/dL)
    "blood_pressure",      # diastolic blood pressure (mmHg)
    "skin_thickness",      # triceps skinfold thickness (mm)
    "insulin",             # 2-hour serum insulin (μU/mL)
    "bmi",                 # body mass index (kg/m²)
    "diabetes_pedigree",   # diabetes pedigree function
    "age",                 # age in years
]

HYPERTENSION_FEATURES = [
    "age",
    "gender_male",         # 1=male, 0=female
    "smoking",             # 1=current smoker
    "cigs_per_day",        # cigarettes per day
    "bmi",
    "heart_rate",          # resting heart rate
    "glucose",             # random blood glucose
    "systolic_bp",         # systolic blood pressure
    "cholesterol",         # total cholesterol (mg/dL)
    "prevalent_hyp",       # prevalent hypertension (binary)
]

ANEMIA_FEATURES = [
    "hemoglobin",          # g/dL — most important
    "mch",                 # mean corpuscular hemoglobin (pg)
    "mchc",                # mean corpuscular hemoglobin concentration (g/dL)
    "mcv",                 # mean corpuscular volume (fL)
    # NOTE: model trained on 4 features only (Hemoglobin, MCH, MCHC, MCV)
]

# ── Symptom-to-feature mapping ─────────────────────────────────────────────────
# Maps user-reported symptoms to numeric features where possible

SYMPTOM_FEATURE_MAP: Dict[str, Dict[str, Any]] = {
    "frequent urination":   {"disease": "diabetes",     "feature": "glucose",     "signal": "high"},
    "excessive thirst":     {"disease": "diabetes",     "feature": "glucose",     "signal": "high"},
    "blurred vision":       {"disease": "diabetes",     "feature": "glucose",     "signal": "high"},
    "fatigue":              {"disease": "anemia",       "feature": "hemoglobin",  "signal": "low"},
    "pale skin":            {"disease": "anemia",       "feature": "hemoglobin",  "signal": "low"},
    "shortness of breath":  {"disease": "anemia",       "feature": "hemoglobin",  "signal": "low"},
    "headache":             {"disease": "hypertension", "feature": "systolic_bp", "signal": "high"},
    "dizziness":            {"disease": "hypertension", "feature": "systolic_bp", "signal": "high"},
    "chest pain":           {"disease": "hypertension", "feature": "systolic_bp", "signal": "high"},
    "nosebleed":            {"disease": "hypertension", "feature": "systolic_bp", "signal": "high"},
    "weight gain":          {"disease": "diabetes",     "feature": "bmi",         "signal": "high"},
    "slow healing":         {"disease": "diabetes",     "feature": "glucose",     "signal": "high"},
    "numbness":             {"disease": "diabetes",     "feature": "glucose",     "signal": "high"},
    "cold hands":           {"disease": "anemia",       "feature": "hemoglobin",  "signal": "low"},
    "brittle nails":        {"disease": "anemia",       "feature": "hemoglobin",  "signal": "low"},
}

# Default median values (from training datasets) used when features are unknown
DIABETES_DEFAULTS = {
    "pregnancies": 3,
    "glucose": 120,
    "blood_pressure": 72,
    "skin_thickness": 29,
    "insulin": 80,
    "bmi": 27.0,
    "diabetes_pedigree": 0.47,
    "age": 33,
}

HYPERTENSION_DEFAULTS = {
    "age": 49,
    "gender_male": 0,
    "smoking": 0,
    "cigs_per_day": 0,
    "bmi": 25.8,
    "heart_rate": 75,
    "glucose": 82,
    "systolic_bp": 132,
    "cholesterol": 236,
    "prevalent_hyp": 0,
}

ANEMIA_DEFAULTS = {
    "hemoglobin": 13.0,
    "mch": 27.0,
    "mchc": 32.0,
    "mcv": 80.0,
}


class FeatureBuilder:
    """Builds feature arrays for ML models from user input."""

    def build_diabetes_features(
        self,
        symptoms: List[str],
        lab_values: Optional[Dict[str, float]] = None,
        age: Optional[int] = None,
        gender: Optional[str] = None,
        bmi: Optional[float] = None,
    ) -> np.ndarray:
        features = dict(DIABETES_DEFAULTS)

        if age:
            features["age"] = age
        if gender == "female":
            features["pregnancies"] = 0  # crude adjustment
        if bmi:
            features["bmi"] = bmi

        if lab_values:
            mapping = {
                "fasting_glucose": "glucose",
                "glucose":         "glucose",
                "diastolic_bp":    "blood_pressure",
                "bmi":             "bmi",
                "insulin":         "insulin",
            }
            for lab_key, feature_key in mapping.items():
                if lab_key in lab_values:
                    features[feature_key] = lab_values[lab_key]

        # Adjust based on symptoms
        symptom_signals = self._get_symptom_signals(symptoms, "diabetes")
        if symptom_signals.get("glucose") == "high":
            features["glucose"] = max(features["glucose"], 140)

        return np.array([[features[f] for f in DIABETES_FEATURES]], dtype=float)

    def build_hypertension_features(
        self,
        symptoms: List[str],
        lab_values: Optional[Dict[str, float]] = None,
        age: Optional[int] = None,
        gender: Optional[str] = None,
    ) -> np.ndarray:
        features = dict(HYPERTENSION_DEFAULTS)

        if age:
            features["age"] = age
        if gender == "male":
            features["gender_male"] = 1

        if lab_values:
            mapping = {
                "systolic_bp":  "systolic_bp",
                "cholesterol":  "cholesterol",
                "glucose":      "glucose",
                "heart_rate":   "heart_rate",
                "bmi":          "bmi",
            }
            for lab_key, feature_key in mapping.items():
                if lab_key in lab_values:
                    features[feature_key] = lab_values[lab_key]

        symptom_signals = self._get_symptom_signals(symptoms, "hypertension")
        if symptom_signals.get("systolic_bp") == "high":
            features["systolic_bp"] = max(features["systolic_bp"], 150)

        return np.array([[features[f] for f in HYPERTENSION_FEATURES]], dtype=float)

    def build_anemia_features(
        self,
        symptoms: List[str],
        lab_values: Optional[Dict[str, float]] = None,
        age: Optional[int] = None,
        gender: Optional[str] = None,
    ) -> np.ndarray:
        features = dict(ANEMIA_DEFAULTS)

        if lab_values:
            mapping = {
                "hemoglobin": "hemoglobin",
                "mch":        "mch",
                "mchc":       "mchc",
                "mcv":        "mcv",
            }
            for lab_key, feature_key in mapping.items():
                if lab_key in lab_values:
                    features[feature_key] = lab_values[lab_key]

        symptom_signals = self._get_symptom_signals(symptoms, "anemia")
        if symptom_signals.get("hemoglobin") == "low":
            features["hemoglobin"] = min(features["hemoglobin"], 10.0)

        return np.array([[features[f] for f in ANEMIA_FEATURES]], dtype=float)

    def _get_symptom_signals(self, symptoms: List[str], disease: str) -> Dict[str, str]:
        signals = {}
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            for key, mapping in SYMPTOM_FEATURE_MAP.items():
                if key in symptom_lower and mapping["disease"] == disease:
                    signals[mapping["feature"]] = mapping["signal"]
        return signals


feature_builder = FeatureBuilder()
