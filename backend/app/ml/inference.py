"""
ml/inference.py — Load trained models and run predictions.
Models are loaded once at startup and kept in memory.
Acts as a cross-validation layer alongside the RAG predictions.
"""
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import structlog

log = structlog.get_logger()

MODELS_DIR = Path(__file__).parent / "models"

# Disease configuration
DISEASES = {
    "diabetes": {
        "display_name": "Type 2 Diabetes",
        "model_file":    "diabetes_model.pkl",
        "scaler_file":   "diabetes_scaler.pkl",
        "meta_file":     "diabetes_metadata.pkl",
        "requires_scaling": False,
    },
    "hypertension": {
        "display_name": "Hypertension",
        "model_file":    "hypertension_model.pkl",
        "scaler_file":   "hypertension_scaler.pkl",
        "meta_file":     "hypertension_metadata.pkl",
        "requires_scaling": False,
    },
    "anemia": {
        "display_name": "Anemia",
        "model_file":    "anemia_model.pkl",
        "scaler_file":   "anemia_scaler.pkl",
        "meta_file":     "anemia_metadata.pkl",
        "requires_scaling": True,
    },
}


class DiseasePredictor:
    """
    Loads all three ML models at startup.
    Thread-safe for FastAPI's async environment (models are read-only after load).
    """

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._scalers: Dict[str, Any] = {}
        self._metadata: Dict[str, Any] = {}
        self._loaded = False

    def load_models(self) -> None:
        """Load all models from disk. Call once at app startup."""
        import joblib
        loaded = []
        failed = []

        for disease, cfg in DISEASES.items():
            model_path = MODELS_DIR / cfg["model_file"]
            scaler_path = MODELS_DIR / cfg["scaler_file"]
            meta_path = MODELS_DIR / cfg["meta_file"]

            if not model_path.exists():
                log.warning("inference.model_not_found", disease=disease, path=str(model_path))
                failed.append(disease)
                continue

            try:
                self._models[disease] = joblib.load(model_path)
                if scaler_path.exists():
                    self._scalers[disease] = joblib.load(scaler_path)
                if meta_path.exists():
                    self._metadata[disease] = joblib.load(meta_path)
                loaded.append(disease)
                log.info("inference.model_loaded", disease=disease)
            except Exception as e:
                log.error("inference.model_load_failed", disease=disease, error=str(e))
                failed.append(disease)

        self._loaded = True
        log.info("inference.startup_complete", loaded=loaded, failed=failed)

    def is_available(self, disease: str) -> bool:
        return disease in self._models

    def predict_disease(
        self,
        disease: str,
        features: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run prediction for a single disease.
        Returns: {disease, probability, risk_level, model_version}
        """
        if not self.is_available(disease):
            return {"disease": disease, "available": False, "probability": 0.0}

        model = self._models[disease]
        cfg = DISEASES[disease]
        meta = self._metadata.get(disease, {})

        # Scale if required
        X = features
        if cfg["requires_scaling"] and disease in self._scalers:
            X = self._scalers[disease].transform(features)

        try:
            prob = float(model.predict_proba(X)[0][1])  # probability of positive class
        except Exception as e:
            log.error("inference.predict_failed", disease=disease, error=str(e))
            return {"disease": disease, "available": False, "probability": 0.0}

        threshold = meta.get("threshold", 0.5)
        risk_level = self._probability_to_risk(prob, threshold)

        return {
            "disease": disease,
            "display_name": cfg["display_name"],
            "available": True,
            "probability": prob,
            "risk_level": risk_level,
            "above_threshold": prob >= threshold,
            "model_name": meta.get("model_name", "unknown"),
            "model_version": meta.get("version", "v1"),
        }

    def predict_all(
        self,
        diabetes_features: Optional[np.ndarray] = None,
        hypertension_features: Optional[np.ndarray] = None,
        anemia_features: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Run all available predictions and return results dict."""
        results = {}

        if diabetes_features is not None and self.is_available("diabetes"):
            results["diabetes"] = self.predict_disease("diabetes", diabetes_features)

        if hypertension_features is not None and self.is_available("hypertension"):
            results["hypertension"] = self.predict_disease("hypertension", hypertension_features)

        if anemia_features is not None and self.is_available("anemia"):
            results["anemia"] = self.predict_disease("anemia", anemia_features)

        return results

    @staticmethod
    def _probability_to_risk(prob: float, threshold: float) -> str:
        if prob >= min(threshold + 0.25, 0.90):
            return "high"
        elif prob >= threshold:
            return "medium"
        else:
            return "low"

    def get_model_info(self) -> List[Dict[str, Any]]:
        """Return info about all loaded models (for health check endpoint)."""
        info = []
        for disease, cfg in DISEASES.items():
            meta = self._metadata.get(disease, {})
            info.append({
                "disease": disease,
                "available": self.is_available(disease),
                "accuracy": meta.get("accuracy"),
                "auc": meta.get("auc"),
                "version": meta.get("version", "v1"),
            })
        return info


# Singleton — imported and used across the app
predictor = DiseasePredictor()


def merge_rag_and_ml_results(
    rag_predictions: List[Dict[str, Any]],
    ml_results: Dict[str, Dict[str, Any]],
    min_rag_score: float = 0.45,
) -> tuple[List[Dict[str, Any]], str]:
    """
    Merge RAG predictions with ML predictions.

    Agreement rules:
    - RAG high + ML agrees (prob >= 0.6) → VERY HIGH confidence (boost score)
    - RAG high + ML disagrees            → downgrade to medium
    - RAG medium + ML agrees             → upgrade to high
    - RAG result only (ML unavailable)   → use RAG as-is
    - ML only (RAG below min_rag_score)  → use ML, flag as ml_only

    Returns: (merged_predictions, method_used)
    """
    if not rag_predictions and not ml_results:
        return [], "none"

    if not rag_predictions:
        return _ml_only_predictions(ml_results), "ml_only"

    DISEASE_MAP = {
        "diabetes":     ["diabetes", "type 2 diabetes", "type ii diabetes"],
        "hypertension": ["hypertension", "high blood pressure", "cardiovascular"],
        "anemia":       ["anemia", "anaemia", "iron deficiency"],
    }

    merged = []
    for rag_pred in rag_predictions:
        rag_disease = rag_pred["disease"].lower()
        matched_ml_key = None

        for ml_key, aliases in DISEASE_MAP.items():
            if any(alias in rag_disease for alias in aliases):
                matched_ml_key = ml_key
                break

        ml_result = ml_results.get(matched_ml_key, {}) if matched_ml_key else {}
        ml_prob = ml_result.get("probability", 0.0)
        ml_available = ml_result.get("available", False)

        rag_score = rag_pred["confidence_score"]

        if ml_available and ml_prob >= 0.6 and rag_score >= min_rag_score:
            # Strong agreement → boost confidence
            boosted_score = min(1.0, rag_score * 1.15)
            rag_pred["confidence_score"] = boosted_score
            rag_pred["confidence"] = "high" if boosted_score >= 0.75 else "medium"
            rag_pred["ml_validation"] = {"status": "confirmed", "ml_probability": ml_prob}
        elif ml_available and ml_prob < 0.35 and rag_score < 0.70:
            # Disagreement on low-confidence RAG → downgrade
            downgraded = max(0.0, rag_score * 0.75)
            rag_pred["confidence_score"] = downgraded
            rag_pred["confidence"] = "low"
            rag_pred["ml_validation"] = {"status": "disputed", "ml_probability": ml_prob}
        else:
            rag_pred["ml_validation"] = {
                "status": "unavailable" if not ml_available else "neutral",
                "ml_probability": ml_prob,
            }

        merged.append(rag_pred)

    # Add high-confidence ML-only predictions not covered by RAG
    for ml_key, ml_result in ml_results.items():
        if not ml_result.get("available"):
            continue
        if ml_result.get("probability", 0) < 0.70:
            continue
        # Check if already in merged
        already_covered = any(
            ml_key in p["disease"].lower() for p in merged
        )
        if not already_covered:
            merged.append({
                "disease": ml_result["display_name"],
                "confidence": "medium",
                "confidence_score": ml_result["probability"] * 0.85,
                "matching_symptoms": [],
                "explanation": f"ML model indicates elevated risk (probability: {ml_result['probability']:.2f}). Insufficient RAG context to provide detailed explanation.",
                "source_chunks": [],
                "ml_validation": {"status": "ml_only", "ml_probability": ml_result["probability"]},
            })

    merged.sort(key=lambda p: p["confidence_score"], reverse=True)

    has_rag = any(len(p.get("source_chunks", [])) > 0 for p in merged)
    has_ml = any(p.get("ml_validation", {}).get("ml_probability", 0) > 0 for p in merged)
    method = "rag_ml_combined" if (has_rag and has_ml) else ("rag_only" if has_rag else "ml_only")

    return merged, method


def _ml_only_predictions(ml_results: Dict[str, Dict]) -> List[Dict]:
    preds = []
    for disease, result in ml_results.items():
        if not result.get("available") or result.get("probability", 0) < 0.45:
            continue
        prob = result["probability"]
        preds.append({
            "disease": result["display_name"],
            "confidence": "high" if prob >= 0.75 else ("medium" if prob >= 0.50 else "low"),
            "confidence_score": prob,
            "matching_symptoms": [],
            "explanation": f"Based on ML model analysis. Probability: {prob:.2f}.",
            "source_chunks": [],
            "ml_validation": {"status": "ml_only", "ml_probability": prob},
        })
    preds.sort(key=lambda p: p["confidence_score"], reverse=True)
    return preds
