"""
services/prediction_service.py — Orchestrates the full RAG + ML prediction pipeline.
"""
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import structlog

from app.config import settings
from app.models.symptom import (
    SymptomAnalysisRequest, PredictionResponse, DiseasePrediction, LabReportRequest,
    LabReportResponse, LabTestResult,
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
    """
    Full symptom analysis pipeline:
    1. Retrieve relevant chunks from Pinecone
    2. Generate RAG prediction via LLM
    3. Run ML models as validation layer
    4. Merge results
    5. Save to database
    6. Return structured response
    """
    prediction_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    # ── Step 1: RAG Retrieval ───────────────────────────────────────────────────
    log.info("prediction.rag_retrieval_start", user_id=user_id, symptoms=request.symptoms)
    rag_predictions = []
    raw_llm_response = ""
    retrieved_chunks = []
    rag_chunk_ids = []
    rag_contexts = []
    rag_scores = []

    try:
        retrieved_chunks = retriever.retrieve_for_symptoms(
            symptoms=request.symptoms,
            context={
                "age": request.age,
                "gender": request.gender,
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
            rag_chunk_ids = [c.chunk_id for c in retrieved_chunks]
            rag_contexts = [{"id": c.chunk_id, "text": c.text[:500]} for c in retrieved_chunks]
            rag_scores = [{"id": c.chunk_id, "score": c.score} for c in retrieved_chunks]
        else:
            log.warning("prediction.no_rag_chunks_returned")

    except Exception as e:
        log.error("prediction.rag_failed", error=str(e))
        # Continue with ML fallback

    # ── Step 2: ML Predictions ─────────────────────────────────────────────────
    ml_results = {}
    try:
        diabetes_X = feature_builder.build_diabetes_features(
            symptoms=request.symptoms,
            age=request.age,
            gender=request.gender,
        )
        hyp_X = feature_builder.build_hypertension_features(
            symptoms=request.symptoms,
            age=request.age,
            gender=request.gender,
        )
        anemia_X = feature_builder.build_anemia_features(
            symptoms=request.symptoms,
            age=request.age,
            gender=request.gender,
        )
        ml_results = predictor.predict_all(
            diabetes_features=diabetes_X,
            hypertension_features=hyp_X,
            anemia_features=anemia_X,
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

    # Determine emergency
    is_emergency = any(p.get("emergency") for p in [rag_result] if isinstance(rag_result, dict)) if rag_predictions else False
    emergency_reason = None
    if rag_predictions and isinstance(rag_result, dict):
        emergency_reason = rag_result.get("emergency_reason")

    # Build response models
    prediction_models = []
    for p in merged_predictions[:3]:  # max 3
        prediction_models.append(
            DiseasePrediction(
                disease=p["disease"],
                confidence=p["confidence"],
                confidence_score=p["confidence_score"],
                matching_symptoms=p.get("matching_symptoms", []),
                explanation=p.get("explanation", ""),
                source_chunks=p.get("source_chunks", []),
            )
        )

    recommended_tests = []
    if rag_predictions and isinstance(rag_result, dict):
        recommended_tests = rag_result.get("recommended_tests", [])

    # ── Step 4: Save to Supabase ───────────────────────────────────────────────
    try:
        # Save symptom check
        sc = supabase.table("symptom_checks").insert({
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "symptoms": request.symptoms,
            "severity_scores": request.severity or {},
            "created_at": now.isoformat(),
        }).execute()
        symptom_check_id = sc.data[0]["id"] if sc.data else None

        # Save RAG retrieval log
        retrieval_id = str(uuid.uuid4())
        supabase.table("rag_retrievals").insert({
            "id": retrieval_id,
            "prediction_id": prediction_id,
            "query_text": " ".join(request.symptoms),
            "retrieved_chunk_ids": rag_chunk_ids,
            "retrieved_contexts": rag_contexts,
            "llm_raw_response": raw_llm_response[:10000] if raw_llm_response else "",
            "retrieval_scores": rag_scores,
            "created_at": now.isoformat(),
        }).execute()

        # Save top prediction
        if merged_predictions:
            top = merged_predictions[0]
            supabase.table("predictions").insert({
                "id": prediction_id,
                "user_id": user_id,
                "symptom_check_id": symptom_check_id,
                "disease": top["disease"],
                "confidence_score": top["confidence_score"],
                "risk_level": top["confidence"],
                "model_version": "rag_ml_v1",
                "feature_values": request.dict(),
                "rag_retrieval_id": retrieval_id,
                "prediction_method": method,
                "source_chunks": top.get("source_chunks", []),
                "created_at": now.isoformat(),
            }).execute()

            # Save recommendations
            await recommendation_service.generate_and_save(
                prediction_id=prediction_id,
                user_id=user_id,
                disease=top["disease"],
                confidence_score=top["confidence_score"],
                emergency=is_emergency,
                supabase=supabase,
            )

    except Exception as e:
        log.error("prediction.db_save_failed", error=str(e))
        # Don't fail the request — DB errors shouldn't block user

    log.info(
        "prediction.complete",
        method=method,
        num_predictions=len(prediction_models),
        emergency=is_emergency,
    )

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
) -> LabReportResponse:
    """
    Lab report analysis:
    1. Rule-based analysis (instant, deterministic)
    2. RAG-powered interpretation (rich explanation)
    3. Save to DB
    """
    report_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    # ── Step 1: Rule engine ────────────────────────────────────────────────────
    results, overall_status, likely_conditions = analyze_full_report(
        lab_values=request.values,
        patient_gender=request.patient_gender,
    )

    test_results = [
        LabTestResult(**{k: v for k, v in r.items() if k in LabTestResult.model_fields})
        for r in results
    ]

    # ── Step 2: RAG interpretation ─────────────────────────────────────────────
    rag_interpretation = None
    try:
        lab_summary_parts = []
        for test, value in request.values.items():
            result = next((r for r in results if r["test_name"] == test), None)
            if result:
                lab_summary_parts.append(
                    f"{result.get('unit', '')} {test}: {value} [{result['status'].upper()}]"
                )
        lab_summary = "; ".join(lab_summary_parts)

        chunks = retriever.retrieve_for_lab_values(lab_summary)
        if chunks:
            rag_result, _ = prompt_builder.generate_lab_interpretation(
                chunks=chunks,
                lab_values=request.values,
                age=request.patient_age,
                gender=request.patient_gender,
            )
            rag_interpretation = rag_result.get("interpretation")

            # Override likely_conditions if RAG found more
            rag_conditions = rag_result.get("likely_conditions", [])
            if rag_conditions:
                likely_conditions = list(set(likely_conditions + rag_conditions))

    except Exception as e:
        log.error("lab_report.rag_failed", error=str(e))

    # ── Step 3: Save ───────────────────────────────────────────────────────────
    try:
        supabase.table("lab_reports").insert({
            "id": report_id,
            "user_id": user_id,
            "report_type": request.report_type,
            "raw_values": request.values,
            "interpreted_results": [r for r in results],
            "overall_status": overall_status,
            "created_at": now.isoformat(),
        }).execute()
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
