"""
routers/recommendations.py — Recommendation retrieval endpoints.
"""
from fastapi import APIRouter, HTTPException
from app.dependencies import CurrentUser, SupabaseClient
from app.models.symptom import RecommendationResponse
from app.services.recommendation_service import get_recommendations_response
import structlog

log = structlog.get_logger()
router = APIRouter(prefix="/recommendations", tags=["Recommendations"])


@router.get("/{prediction_id}", response_model=RecommendationResponse)
async def get_recommendation(
    prediction_id: str,
    current_user: CurrentUser,
    supabase: SupabaseClient,
):
    """
    Get health recommendations tied to a specific prediction.
    Includes recommended tests, specialists, and lifestyle tips.
    """
    try:
        # Verify ownership via prediction
        pred = (
            supabase.table("predictions")
            .select("id")
            .eq("id", prediction_id)
            .eq("user_id", current_user.id)
            .single()
            .execute()
        )
        if not pred.data:
            raise HTTPException(status_code=404, detail="Prediction not found.")

        result = (
            supabase.table("recommendations")
            .select("*")
            .eq("prediction_id", prediction_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="No recommendations found for this prediction.")

        return get_recommendations_response(result.data[0])

    except HTTPException:
        raise
    except Exception as e:
        log.error("recommendations.get_failed", prediction_id=prediction_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch recommendations.")
