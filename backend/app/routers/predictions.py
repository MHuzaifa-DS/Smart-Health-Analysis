"""
routers/predictions.py — Prediction history and detail endpoints.
"""
from fastapi import APIRouter, HTTPException
from app.dependencies import CurrentUser, SupabaseClient, Pagination
import structlog

log = structlog.get_logger()
router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.get("/history")
async def get_prediction_history(
    current_user: CurrentUser,
    supabase: SupabaseClient,
    pagination: Pagination,
):
    """Return paginated prediction history for the current user."""
    try:
        result = (
            supabase.table("predictions")
            .select(
                "id, disease, confidence_score, risk_level, prediction_method, created_at"
            )
            .eq("user_id", current_user.id)
            .order("created_at", desc=True)
            .range(pagination["skip"], pagination["skip"] + pagination["limit"] - 1)
            .execute()
        )
        return {
            "predictions": result.data,
            "skip": pagination["skip"],
            "limit": pagination["limit"],
        }
    except Exception as e:
        log.error("predictions.history_failed", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch prediction history.")


@router.get("/{prediction_id}")
async def get_prediction(
    prediction_id: str,
    current_user: CurrentUser,
    supabase: SupabaseClient,
):
    """Get a single prediction with its full detail and associated symptom check."""
    try:
        result = (
            supabase.table("predictions")
            .select("*, symptom_checks(*)")
            .eq("id", prediction_id)
            .eq("user_id", current_user.id)
            .single()
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Prediction not found.")
        return result.data
    except HTTPException:
        raise
    except Exception as e:
        log.error("predictions.get_failed", prediction_id=prediction_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch prediction.")


@router.get("/{prediction_id}/sources")
async def get_rag_sources(
    prediction_id: str,
    current_user: CurrentUser,
    supabase: SupabaseClient,
):
    """
    Return the Gale Encyclopedia chunks that were cited in this prediction.
    Useful for transparency — shows users which medical text backed the prediction.
    """
    try:
        # Get prediction to verify ownership
        pred_result = (
            supabase.table("predictions")
            .select("id, rag_retrieval_id, disease")
            .eq("id", prediction_id)
            .eq("user_id", current_user.id)
            .single()
            .execute()
        )
        if not pred_result.data:
            raise HTTPException(status_code=404, detail="Prediction not found.")

        retrieval_id = pred_result.data.get("rag_retrieval_id")
        if not retrieval_id:
            return {
                "prediction_id": prediction_id,
                "sources": [],
                "message": "No RAG context was used for this prediction (ML-only).",
            }

        # Get retrieval log
        retrieval_result = (
            supabase.table("rag_retrievals")
            .select("retrieved_contexts, retrieval_scores, query_text")
            .eq("id", retrieval_id)
            .single()
            .execute()
        )
        if not retrieval_result.data:
            return {"prediction_id": prediction_id, "sources": []}

        data = retrieval_result.data
        contexts = data.get("retrieved_contexts") or []
        scores = {s["id"]: s["score"] for s in (data.get("retrieval_scores") or [])}

        sources = []
        for ctx in contexts:
            sources.append({
                "chunk_id": ctx.get("id"),
                "text_preview": ctx.get("text", "")[:400],
                "relevance_score": round(scores.get(ctx.get("id"), 0), 4),
                "source": "Gale Encyclopedia of Medicine, 3rd Edition",
            })

        return {
            "prediction_id": prediction_id,
            "disease": pred_result.data["disease"],
            "query_text": data.get("query_text"),
            "sources": sorted(sources, key=lambda s: s["relevance_score"], reverse=True),
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error("predictions.sources_failed", prediction_id=prediction_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch RAG sources.")
