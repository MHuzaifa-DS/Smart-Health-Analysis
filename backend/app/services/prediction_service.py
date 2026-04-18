"""
services/prediction_service.py — Orchestrates the full RAG + ML prediction pipeline.

FIXES:
  FIX 1 — rag_retrievals inserted before predictions (FK order)
  FIX 2 — All inserts use get_supabase_admin() (service-role key)
  FIX 3 — likely_conditions from rule engine are NEVER lost:
           RAG conditions only ADD to the rule-engine list, never replace it.
           likely_conditions is now also saved to the lab_reports DB row.
  FIX 4 — file_url parameter added to analyze_lab_report so upload endpoint
           can persist the storage URL.
"""
import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog

from app.config import settings
from app.database import get_supabase_admin
from app.models.symptom import (
    SymptomAnalysisRequest, PredictionResponse, DiseasePrediction,
    LabReportRequest, LabReportResponse, LabTestResult,
)
from app.rag import retriever, prompt_builder
from app.ml.inference import predictor, merge_rag_and_ml_results
from app.ml.preprocessor import feature_builder
from app.ml.lab_rules import analyze_full_report
from app.services import recommendation_service

log = structlog.get_logger()


async def analyze_symptoms(
    request: SymptomAnalysisRequest,
    user_id: str,
    supabase,
) -> PredictionResponse:

    prediction_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    db = get_supabase_admin()

    # ── Step 1: RAG ────────────────────────────────────────────────────────────
    log.info("prediction.rag_retrieval_start", user_id=user_id, symptoms=request.symptoms)
    rag_predictions  = []
    rag_result       = {}
    raw_llm_response = ""
    rag_chunk_ids    = []
    rag_contexts     = []
    rag_scores       = []

    try:
        retrieved_chunks = retriever.retrieve_for_symptoms(
            symptoms=request.symptoms,
            context={
                "age":           request.age,
                "gender":        request.gender,
                "duration_days": request.duration_days,
            },
        )
        log.info("prediction.chunks_retrieved", count=len(retrieved_chunks))

        if retrieved_chunks:
            rag_result, raw_llm_response = prompt_builder.generate_symptom_prediction(
                chunks=retrieved_chunks,
                symptoms=request.symptoms,
                severity=request.severity,
                duration_days=request.duration_days,
                age=request.age,
                gender=request.gender,
                free_text=request.free_text,
            )
            rag_predictions = rag_result.get("predictions", [])
            rag_chunk_ids   = [c.chunk_id for c in retrieved_chunks]
            rag_contexts    = [{"id": c.chunk_id, "text": c.text[:500]} for c in retrieved_chunks]
            rag_scores      = [{"id": c.chunk_id, "score": c.score} for c in retrieved_chunks]
        else:
            log.warning("prediction.no_rag_chunks_returned")
    except Exception as e:
        log.error("prediction.rag_failed", error=str(e))

    # ── Step 2: ML ─────────────────────────────────────────────────────────────
    ml_results = {}
    try:
        ml_results = predictor.predict_all(
            diabetes_features=feature_builder.build_diabetes_features(
                symptoms=request.symptoms, age=request.age, gender=request.gender),
            hypertension_features=feature_builder.build_hypertension_features(
                symptoms=request.symptoms, age=request.age, gender=request.gender),
            anemia_features=feature_builder.build_anemia_features(
                symptoms=request.symptoms, age=request.age, gender=request.gender),
        )
        log.info("prediction.ml_complete", diseases=list(ml_results.keys()))
    except Exception as e:
        log.error("prediction.ml_failed", error=str(e))

    # ── Step 3: Merge ──────────────────────────────────────────────────────────
    merged_predictions, method = merge_rag_and_ml_results(
        rag_predictions=rag_predictions,
        ml_results=ml_results,
        min_rag_score=settings.rag_min_score,
    )

    is_emergency     = bool(rag_result.get("emergency")) if rag_result else False
    emergency_reason = rag_result.get("emergency_reason") if rag_result else None
    recommended_tests = rag_result.get("recommended_tests", []) if rag_result else []

    prediction_models = [
        DiseasePrediction(
            disease=p["disease"],
            confidence=p["confidence"],
            confidence_score=p["confidence_score"],
            matching_symptoms=p.get("matching_symptoms", []),
            explanation=p.get("explanation", ""),
            source_chunks=p.get("source_chunks", []),
        )
        for p in merged_predictions[:3]
    ]

    # ── Step 4: Save (FK-safe order) ───────────────────────────────────────────
    symptom_check_id = None
    retrieval_id     = str(uuid.uuid4())

    try:
        sc = db.table("symptom_checks").insert({
            "id":              str(uuid.uuid4()),
            "user_id":         user_id,
            "symptoms":        request.symptoms,
            "severity_scores": request.severity or {},
            "duration_days":   request.duration_days,
            "free_text":       request.free_text,
            "created_at":      now.isoformat(),
        }).execute()
        symptom_check_id = sc.data[0]["id"] if sc.data else None
        log.info("prediction.symptom_check_saved", id=symptom_check_id)
    except Exception as e:
        log.error("prediction.symptom_check_save_failed", error=str(e))

    try:
        db.table("rag_retrievals").insert({
            "id":                  retrieval_id,
            "prediction_id":       None,
            "query_text":          " ".join(request.symptoms),
            "retrieved_chunk_ids": rag_chunk_ids,
            "retrieved_contexts":  rag_contexts,
            "llm_raw_response":    raw_llm_response[:10000] if raw_llm_response else "",
            "retrieval_scores":    rag_scores,
            "created_at":          now.isoformat(),
        }).execute()
        log.info("prediction.rag_retrieval_saved", id=retrieval_id)
    except Exception as e:
        log.error("prediction.rag_retrieval_save_failed", error=str(e))
        retrieval_id = None

    try:
        if merged_predictions:
            top = merged_predictions[0]
            db.table("predictions").insert({
                "id":                prediction_id,
                "user_id":           user_id,
                "symptom_check_id":  symptom_check_id,
                "rag_retrieval_id":  retrieval_id,
                "disease":           top["disease"],
                "confidence_score":  top["confidence_score"],
                "risk_level":        top["confidence"],
                "model_version":     "rag_ml_v1",
                "feature_values":    request.model_dump(),
                "prediction_method": method,
                "source_chunks":     top.get("source_chunks", []),
                "created_at":        now.isoformat(),
            }).execute()
            log.info("prediction.prediction_saved", id=prediction_id)

        if retrieval_id and merged_predictions:
            db.table("rag_retrievals").update(
                {"prediction_id": prediction_id}
            ).eq("id", retrieval_id).execute()
            log.info("prediction.rag_retrieval_updated", prediction_id=prediction_id)
    except Exception as e:
        log.error("prediction.db_save_failed", error=str(e))

    try:
        if merged_predictions:
            await recommendation_service.generate_and_save(
                prediction_id=prediction_id,
                user_id=user_id,
                disease=merged_predictions[0]["disease"],
                confidence_score=merged_predictions[0]["confidence_score"],
                emergency=is_emergency,
                supabase=db,
            )
    except Exception as e:
        log.error("prediction.recommendations_save_failed", error=str(e))

    log.info("prediction.complete", method=method,
             num_predictions=len(prediction_models), emergency=is_emergency)

    return PredictionResponse(
        prediction_id=prediction_id,
        predictions=prediction_models,
        recommended_tests=recommended_tests,
        emergency=is_emergency,
        emergency_reason=emergency_reason,
        prediction_method=method,
        created_at=now,
    )


async def analyze_lab_report(
    request: LabReportRequest,
    user_id: str,
    supabase,
    file_url: Optional[str] = None,   # FIX 4: accepts file_url from upload router
) -> LabReportResponse:

    report_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    db = get_supabase_admin()

    # ── Step 1: Rule engine ────────────────────────────────────────────────────
    results, overall_status, rule_conditions = analyze_full_report(
        lab_values=request.values,
        patient_gender=request.patient_gender,
    )
    # Work with a copy so RAG can never accidentally wipe the rule engine output
    likely_conditions = list(rule_conditions)

    log.info("lab_report.rule_engine_complete",
             overall_status=overall_status, conditions=likely_conditions)

    test_results = [
        LabTestResult(**{k: v for k, v in r.items() if k in LabTestResult.model_fields})
        for r in results
    ]

    # ── Step 2: RAG interpretation ─────────────────────────────────────────────
    rag_interpretation = None
    try:
        lab_summary_parts = []
        for test, value in request.values.items():
            r = next((x for x in results if x["test_name"] == test), None)
            if r:
                lab_summary_parts.append(
                    f"{r.get('unit','')} {test}: {value} [{r['status'].upper()}]"
                )

        chunks = retriever.retrieve_for_lab_values("; ".join(lab_summary_parts))
        if chunks:
            rag_result, _ = prompt_builder.generate_lab_interpretation(
                chunks=chunks,
                lab_values=request.values,
                age=request.patient_age,
                gender=request.patient_gender,
            )
            rag_interpretation = rag_result.get("interpretation")

            # FIX 3: only MERGE rag conditions — never replace rule engine conditions
            rag_conditions = rag_result.get("likely_conditions") or []
            if rag_conditions:
                likely_conditions = sorted(set(rule_conditions) | set(rag_conditions))
            # if rag_conditions is empty → keep rule_conditions untouched

    except Exception as e:
        log.error("lab_report.rag_failed", error=str(e))
        # RAG failure → keep rule_conditions, no crash

    # ── Step 3: Save ───────────────────────────────────────────────────────────
    try:
        db.table("lab_reports").insert({
            "id":                  report_id,
            "user_id":             user_id,
            "report_type":         request.report_type,
            "file_url":            file_url,            # FIX 4
            "raw_values":          request.values,
            "interpreted_results": results,
            "overall_status":      overall_status,
            "likely_conditions":   likely_conditions,   # FIX 3
            "created_at":          now.isoformat(),
        }).execute()
        log.info("lab_report.saved", id=report_id,
                 conditions=likely_conditions, overall_status=overall_status)
    except Exception as e:
        log.error("lab_report.db_save_failed", error=str(e))

    return LabReportResponse(
        report_id=report_id,
        report_type=request.report_type,
        results=test_results,
        overall_status=overall_status,
        likely_conditions=likely_conditions,
        rag_interpretation=rag_interpretation,
        created_at=now,
    )