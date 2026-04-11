"""
routers/dashboard.py — User health dashboard endpoints.
"""
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from app.dependencies import CurrentUser, SupabaseClient
import structlog

log = structlog.get_logger()
router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/summary")
async def get_dashboard_summary(
    current_user: CurrentUser,
    supabase: SupabaseClient,
):
    """
    Aggregated health dashboard for the current user.
    Returns recent predictions, lab reports, and a computed health score.
    """
    try:
        # Recent predictions (last 5)
        preds = (
            supabase.table("predictions")
            .select("id, disease, confidence_score, risk_level, prediction_method, created_at")
            .eq("user_id", current_user.id)
            .order("created_at", desc=True)
            .limit(5)
            .execute()
        )

        # Recent lab reports (last 5)
        labs = (
            supabase.table("lab_reports")
            .select("id, report_type, overall_status, likely_conditions, created_at")
            .eq("user_id", current_user.id)
            .order("created_at", desc=True)
            .limit(5)
            .execute()
        )

        # Total counts
        total_preds = (
            supabase.table("predictions")
            .select("id", count="exact")
            .eq("user_id", current_user.id)
            .execute()
        )
        total_labs = (
            supabase.table("lab_reports")
            .select("id", count="exact")
            .eq("user_id", current_user.id)
            .execute()
        )

        prediction_list = preds.data or []
        lab_list = labs.data or []

        # Compute a simple health score (0-100)
        health_score = _compute_health_score(prediction_list, lab_list)

        last_check = prediction_list[0]["created_at"] if prediction_list else None

        return {
            "user": {
                "full_name": current_user.full_name,
                "age": current_user.age,
                "gender": current_user.gender,
                "blood_type": current_user.blood_type,
            },
            "statistics": {
                "total_symptom_checks": total_preds.count or 0,
                "total_lab_reports": total_labs.count or 0,
                "last_check_date": last_check,
            },
            "health_score": health_score,
            "recent_predictions": prediction_list,
            "recent_lab_reports": lab_list,
        }

    except Exception as e:
        log.error("dashboard.summary_failed", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to load dashboard.")


@router.get("/metrics")
async def get_health_metrics(
    current_user: CurrentUser,
    supabase: SupabaseClient,
    metric_type: Optional[str] = Query(None, description="Filter by metric type"),
    days: int = Query(30, ge=1, le=365, description="Number of days to look back"),
):
    """
    Return time-series health metrics for charting on the dashboard.
    """
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        query = (
            supabase.table("health_metrics")
            .select("*")
            .eq("user_id", current_user.id)
            .gte("recorded_at", cutoff)
            .order("recorded_at", desc=False)
        )
        if metric_type:
            query = query.eq("metric_type", metric_type)

        result = query.execute()
        metrics = result.data or []

        # Group by metric_type for frontend chart consumption
        grouped: dict = {}
        for m in metrics:
            mtype = m["metric_type"]
            if mtype not in grouped:
                grouped[mtype] = []
            grouped[mtype].append({
                "date": m["recorded_at"],
                "value": m["value"],
                "unit": m["unit"],
            })

        return {
            "user_id": current_user.id,
            "period_days": days,
            "metrics": grouped,
            "metric_types": list(grouped.keys()),
        }
    except Exception as e:
        log.error("dashboard.metrics_failed", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to load health metrics.")


@router.post("/metrics")
async def record_health_metric(
    current_user: CurrentUser,
    supabase: SupabaseClient,
    metric_type: str = Query(..., description="e.g. blood_pressure, blood_sugar, weight"),
    value: float = Query(...),
    unit: str = Query(..., description="e.g. mmHg, mg/dL, kg"),
):
    """
    Manually record a health metric (for tracking over time).
    """
    valid_types = [
        "blood_sugar", "blood_pressure_systolic", "blood_pressure_diastolic",
        "weight", "heart_rate", "hemoglobin", "hba1c", "temperature",
    ]
    if metric_type not in valid_types:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid metric_type. Valid options: {valid_types}",
        )

    try:
        import uuid
        result = supabase.table("health_metrics").insert({
            "id": str(uuid.uuid4()),
            "user_id": current_user.id,
            "metric_type": metric_type,
            "value": value,
            "unit": unit,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }).execute()

        return {"status": "recorded", "metric": result.data[0] if result.data else {}}
    except Exception as e:
        log.error("dashboard.metric_record_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to record metric.")


def _compute_health_score(predictions: list, lab_reports: list) -> Optional[float]:
    """
    Simple health score (0-100) based on recent risk levels and lab statuses.
    Higher = better health indicators.
    """
    if not predictions and not lab_reports:
        return None

    score = 100.0
    risk_penalties = {"high": 20, "medium": 10, "low": 5}
    status_penalties = {"critical": 25, "abnormal": 15, "borderline": 8, "normal": 0}

    for pred in predictions[:3]:  # weight recent predictions most
        penalty = risk_penalties.get(pred.get("risk_level", "low"), 5)
        score -= penalty

    for lab in lab_reports[:3]:
        penalty = status_penalties.get(lab.get("overall_status", "normal"), 0)
        score -= penalty

    return max(0.0, min(100.0, round(score, 1)))


# Import Optional at top
from typing import Optional
