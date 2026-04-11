"""
ml/lab_rules.py — Rule-based lab report analysis engine.
Compares patient values against clinical normal ranges and flags abnormalities.
"""
from typing import Dict, List, Optional, Any, Tuple


# ── Normal ranges ──────────────────────────────────────────────────────────────
# Format: (critical_low, low, high, critical_high, unit, display_name)

LAB_RANGES: Dict[str, Dict] = {
    # Blood Glucose
    "fasting_glucose": {
        "critical_low": 50,   "low": 70,    "high": 99,   "critical_high": 400,
        "unit": "mg/dL",      "display": "Fasting Blood Glucose",
        "disease_hint": "diabetes",
    },
    "random_glucose": {
        "critical_low": 50,   "low": 70,    "high": 139,  "critical_high": 400,
        "unit": "mg/dL",      "display": "Random Blood Glucose",
        "disease_hint": "diabetes",
    },
    "hba1c": {
        "critical_low": None, "low": None,  "high": 5.6,  "critical_high": 10.0,
        "unit": "%",          "display": "HbA1c (Glycated Hemoglobin)",
        "disease_hint": "diabetes",
        "ranges_text": {
            "normal":     "< 5.7% (Normal)",
            "prediabetes":"5.7% – 6.4% (Prediabetes)",
            "diabetes":   "≥ 6.5% (Diabetes)",
        },
    },

    # Blood Pressure
    "systolic_bp": {
        "critical_low": 70,   "low": 90,    "high": 120,  "critical_high": 180,
        "unit": "mmHg",       "display": "Systolic Blood Pressure",
        "disease_hint": "hypertension",
    },
    "diastolic_bp": {
        "critical_low": 40,   "low": 60,    "high": 80,   "critical_high": 120,
        "unit": "mmHg",       "display": "Diastolic Blood Pressure",
        "disease_hint": "hypertension",
    },

    # Blood Count — Hemoglobin (gender-specific)
    "hemoglobin_male": {
        "critical_low": 7.0,  "low": 13.5,  "high": 17.5, "critical_high": 20.0,
        "unit": "g/dL",       "display": "Hemoglobin (Male)",
        "disease_hint": "anemia",
    },
    "hemoglobin_female": {
        "critical_low": 7.0,  "low": 12.0,  "high": 15.5, "critical_high": 20.0,
        "unit": "g/dL",       "display": "Hemoglobin (Female)",
        "disease_hint": "anemia",
    },
    "hemoglobin": {  # default (gender-agnostic)
        "critical_low": 7.0,  "low": 12.0,  "high": 17.5, "critical_high": 20.0,
        "unit": "g/dL",       "display": "Hemoglobin",
        "disease_hint": "anemia",
    },

    # Blood Count — Other
    "wbc": {
        "critical_low": 2000, "low": 4500,  "high": 11000, "critical_high": 30000,
        "unit": "/μL",        "display": "White Blood Cells (WBC)",
        "disease_hint": None,
    },
    "platelets": {
        "critical_low": 50000,"low": 150000,"high": 400000,"critical_high": 1000000,
        "unit": "/μL",        "display": "Platelets",
        "disease_hint": None,
    },
    "mcv": {
        "critical_low": 50,   "low": 80,    "high": 100,  "critical_high": 130,
        "unit": "fL",         "display": "Mean Corpuscular Volume (MCV)",
        "disease_hint": "anemia",
    },
    "mch": {
        "critical_low": 15,   "low": 27,    "high": 33,   "critical_high": 45,
        "unit": "pg",         "display": "Mean Corpuscular Hemoglobin (MCH)",
        "disease_hint": "anemia",
    },
    "mchc": {
        "critical_low": 25,   "low": 32,    "high": 36,   "critical_high": 40,
        "unit": "g/dL",       "display": "MCHC",
        "disease_hint": "anemia",
    },

    # Lipids
    "total_cholesterol": {
        "critical_low": None, "low": None,  "high": 200,  "critical_high": 300,
        "unit": "mg/dL",      "display": "Total Cholesterol",
        "disease_hint": "hypertension",
    },
    "ldl": {
        "critical_low": None, "low": None,  "high": 100,  "critical_high": 190,
        "unit": "mg/dL",      "display": "LDL Cholesterol",
        "disease_hint": "hypertension",
    },
    "hdl_male": {
        "critical_low": 20,   "low": 40,    "high": 999,  "critical_high": None,
        "unit": "mg/dL",      "display": "HDL Cholesterol (Male)",
        "disease_hint": None,
    },
    "hdl_female": {
        "critical_low": 20,   "low": 50,    "high": 999,  "critical_high": None,
        "unit": "mg/dL",      "display": "HDL Cholesterol (Female)",
        "disease_hint": None,
    },

    # Kidney / Liver
    "creatinine": {
        "critical_low": None, "low": 0.7,   "high": 1.3,  "critical_high": 10.0,
        "unit": "mg/dL",      "display": "Serum Creatinine",
        "disease_hint": None,
    },
    "bun": {
        "critical_low": None, "low": 7,     "high": 25,   "critical_high": 80,
        "unit": "mg/dL",      "display": "Blood Urea Nitrogen (BUN)",
        "disease_hint": None,
    },
    "alt": {
        "critical_low": None, "low": 7,     "high": 56,   "critical_high": 500,
        "unit": "U/L",        "display": "ALT (Liver Enzyme)",
        "disease_hint": None,
    },
    "ast": {
        "critical_low": None, "low": 10,    "high": 40,   "critical_high": 500,
        "unit": "U/L",        "display": "AST (Liver Enzyme)",
        "disease_hint": None,
    },

    # Thyroid
    "tsh": {
        "critical_low": 0.1,  "low": 0.4,   "high": 4.0,  "critical_high": 10.0,
        "unit": "mIU/L",      "display": "Thyroid Stimulating Hormone (TSH)",
        "disease_hint": None,
    },

    # Iron Studies
    "serum_iron": {
        "critical_low": 20,   "low": 60,    "high": 170,  "critical_high": 300,
        "unit": "μg/dL",      "display": "Serum Iron",
        "disease_hint": "anemia",
    },
    "ferritin": {
        "critical_low": 5,    "low": 12,    "high": 300,  "critical_high": 1000,
        "unit": "ng/mL",      "display": "Serum Ferritin",
        "disease_hint": "anemia",
    },
}

# Disease pattern detection from lab combinations
DISEASE_PATTERNS = {
    "Type 2 Diabetes": {
        "criteria": [
            {"test": "fasting_glucose", "operator": ">=", "value": 126},
            {"test": "hba1c",           "operator": ">=", "value": 6.5},
            {"test": "random_glucose",  "operator": ">=", "value": 200},
        ],
        "logic": "OR",
        "confidence": "high",
    },
    "Prediabetes": {
        "criteria": [
            {"test": "fasting_glucose", "operator": "between", "low": 100, "high": 125},
            {"test": "hba1c",           "operator": "between", "low": 5.7, "high": 6.4},
        ],
        "logic": "OR",
        "confidence": "medium",
    },
    "Hypertension (Stage 1)": {
        "criteria": [
            {"test": "systolic_bp",  "operator": "between", "low": 130, "high": 139},
            {"test": "diastolic_bp", "operator": "between", "low": 80,  "high": 89},
        ],
        "logic": "OR",
        "confidence": "medium",
    },
    "Hypertension (Stage 2)": {
        "criteria": [
            {"test": "systolic_bp",  "operator": ">=", "value": 140},
            {"test": "diastolic_bp", "operator": ">=", "value": 90},
        ],
        "logic": "OR",
        "confidence": "high",
    },
    "Anemia": {
        "criteria": [
            {"test": "hemoglobin",        "operator": "<", "value": 12.0},
            {"test": "hemoglobin_male",   "operator": "<", "value": 13.5},
            {"test": "hemoglobin_female", "operator": "<", "value": 12.0},
        ],
        "logic": "OR",
        "confidence": "high",
    },
    "Iron Deficiency Anemia": {
        "criteria": [
            {"test": "serum_iron", "operator": "<", "value": 60},
            {"test": "ferritin",   "operator": "<", "value": 12},
        ],
        "logic": "AND",
        "confidence": "high",
    },
}


def analyze_lab_value(
    test_name: str,
    value: float,
    patient_gender: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze a single lab value against normal ranges."""
    # Apply gender-specific ranges
    effective_test = test_name
    if test_name == "hemoglobin" and patient_gender:
        effective_test = f"hemoglobin_{patient_gender}"
    if test_name in ("hdl",) and patient_gender:
        effective_test = f"hdl_{patient_gender}"

    ranges = LAB_RANGES.get(effective_test) or LAB_RANGES.get(test_name)
    if not ranges:
        return {
            "test_name": test_name,
            "value": value,
            "unit": "unknown",
            "status": "unknown",
            "normal_range": "Reference range not available",
            "interpretation": "No reference range available for this test.",
            "emergency": False,
        }

    unit = ranges["unit"]
    low = ranges.get("low")
    high = ranges.get("high")
    crit_low = ranges.get("critical_low")
    crit_high = ranges.get("critical_high")

    # Determine status
    emergency = False
    if crit_low is not None and value < crit_low:
        status = "critical_low"
        emergency = True
        interp = f"⚠️ CRITICAL LOW: {ranges['display']} is dangerously low at {value} {unit}. Immediate medical attention required."
    elif low is not None and value < low:
        status = "low"
        interp = f"{ranges['display']} is below normal ({value} {unit}). Normal range: {low}–{high} {unit}."
    elif crit_high is not None and value > crit_high:
        status = "critical_high"
        emergency = True
        interp = f"⚠️ CRITICAL HIGH: {ranges['display']} is dangerously elevated at {value} {unit}. Immediate medical attention required."
    elif high is not None and value > high:
        status = "high"
        interp = f"{ranges['display']} is above normal ({value} {unit}). Normal range: {low}–{high} {unit}."
    else:
        status = "normal"
        interp = f"{ranges['display']} is within normal range ({value} {unit})."

    normal_range = f"{low}–{high} {unit}" if (low and high) else f"See reference"

    return {
        "test_name": test_name,
        "value": value,
        "unit": unit,
        "status": status,
        "normal_range": normal_range,
        "interpretation": interp,
        "emergency": emergency,
        "disease_hint": ranges.get("disease_hint"),
    }


def detect_disease_patterns(
    lab_values: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Detect likely conditions from combinations of lab values."""
    detected = []

    for disease, pattern in DISEASE_PATTERNS.items():
        criteria_results = []

        for criterion in pattern["criteria"]:
            test = criterion["test"]
            if test not in lab_values:
                continue

            val = lab_values[test]
            op = criterion["operator"]

            if op == ">=" and val >= criterion["value"]:
                criteria_results.append(True)
            elif op == "<=" and val <= criterion["value"]:
                criteria_results.append(True)
            elif op == ">" and val > criterion["value"]:
                criteria_results.append(True)
            elif op == "<" and val < criterion["value"]:
                criteria_results.append(True)
            elif op == "between" and criterion["low"] <= val <= criterion["high"]:
                criteria_results.append(True)
            else:
                criteria_results.append(False)

        if not criteria_results:
            continue

        logic = pattern["logic"]
        matched = any(criteria_results) if logic == "OR" else all(criteria_results)

        if matched:
            detected.append({
                "condition": disease,
                "confidence": pattern["confidence"],
                "matched_criteria": sum(criteria_results),
                "total_criteria": len(criteria_results),
            })

    return detected


def compute_overall_status(results: List[Dict[str, Any]]) -> str:
    """Compute overall status from individual test results."""
    if any(r["emergency"] for r in results):
        return "critical"
    if any(r["status"] in ("critical_low", "critical_high") for r in results):
        return "critical"
    if any(r["status"] in ("high", "low") for r in results):
        return "abnormal"
    all_normal = all(r["status"] == "normal" for r in results if r["status"] != "unknown")
    return "normal" if all_normal else "borderline"


def analyze_full_report(
    lab_values: Dict[str, float],
    patient_gender: Optional[str] = None,
) -> Tuple[List[Dict], str, List[str]]:
    """
    Analyze a complete set of lab values.
    Returns: (test_results, overall_status, likely_conditions)
    """
    results = []
    for test_name, value in lab_values.items():
        result = analyze_lab_value(test_name, value, patient_gender)
        results.append(result)

    overall = compute_overall_status(results)
    conditions = detect_disease_patterns(lab_values)
    condition_names = [c["condition"] for c in conditions]

    return results, overall, condition_names
