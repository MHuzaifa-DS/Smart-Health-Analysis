"""
routers/lab_reports.py — Lab report analysis endpoints.
Supports both manual value input and PDF/image upload with OCR.

FIX: upload endpoint now passes file_url to analyze_lab_report so it
     is persisted in the lab_reports DB row.
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
import json

from app.dependencies import CurrentUser, SupabaseClient, Pagination
from app.models.symptom import LabReportRequest, LabReportResponse
from app.services import prediction_service
from app.utils.ocr import extract_from_pdf_bytes, extract_from_image_bytes
import structlog

log = structlog.get_logger()
router = APIRouter(prefix="/lab-reports", tags=["Lab Reports"])

ALLOWED_MIME_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/tiff",
}
MAX_FILE_SIZE_MB = 10


@router.post("/analyze", response_model=LabReportResponse, status_code=201)
async def analyze_lab_report(
    request: LabReportRequest,
    current_user: CurrentUser,
    supabase: SupabaseClient,
):
    """
    Analyze lab values submitted manually.
    Runs rule-based analysis + RAG interpretation.

    Example request body:
    {
        "report_type": "blood_test",
        "values": {
            "fasting_glucose": 145,
            "hba1c": 7.1,
            "hemoglobin": 11.2
        }
    }
    """
    if not request.patient_age and current_user.age:
        request.patient_age = current_user.age
    if not request.patient_gender and current_user.gender:
        request.patient_gender = current_user.gender

    try:
        return await prediction_service.analyze_lab_report(
            request=request,
            user_id=current_user.id,
            supabase=supabase,
            file_url=None,
        )
    except Exception as e:
        log.error("lab_reports.analyze_error", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/upload", response_model=LabReportResponse, status_code=201)
async def upload_lab_report(
    current_user: CurrentUser,
    supabase: SupabaseClient,
    file: UploadFile = File(..., description="PDF or image of lab report"),
    report_type: str = Form(default="blood_test"),
    notes: Optional[str] = Form(default=None),
):
    """
    Upload a PDF or image lab report. OCR extracts values automatically.
    Falls back gracefully if OCR finds no values — returns 422 with message.
    """
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Allowed: PDF, PNG, JPEG.",
        )

    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum: {MAX_FILE_SIZE_MB} MB.",
        )

    log.info(
        "lab_reports.upload_received",
        user_id=current_user.id,
        filename=file.filename,
        size_mb=round(size_mb, 2),
        content_type=file.content_type,
    )

    # ── OCR extraction ─────────────────────────────────────────────────────────
    raw_text = ""
    extracted_values = {}

    if file.content_type == "application/pdf":
        raw_text, extracted_values = extract_from_pdf_bytes(file_bytes)
    else:
        raw_text, extracted_values = extract_from_image_bytes(file_bytes)

    if not extracted_values:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "Could not extract lab values from the uploaded file. "
                    "Please enter values manually using the /lab-reports/analyze endpoint."
                ),
                "raw_text_preview": raw_text[:300] if raw_text else "No text extracted.",
                "tip": "Ensure the file is clear and not password-protected.",
            },
        )

    log.info("lab_reports.ocr_success", extracted_count=len(extracted_values))

    # ── Upload file to Supabase Storage (best-effort) ──────────────────────────
    file_url = None
    try:
        from app.config import settings
        storage_path = f"{current_user.id}/{file.filename}"
        supabase.storage.from_(settings.storage_bucket).upload(
            path=storage_path,
            file=file_bytes,
            file_options={"content-type": file.content_type},
        )
        file_url = supabase.storage.from_(settings.storage_bucket).get_public_url(storage_path)
        log.info("lab_reports.storage_upload_success", path=storage_path)
    except Exception as e:
        log.warning("lab_reports.storage_upload_failed", error=str(e))

    # ── Analyze extracted values ───────────────────────────────────────────────
    lab_request = LabReportRequest(
        report_type=report_type,
        values=extracted_values,
        patient_age=current_user.age,
        patient_gender=current_user.gender,
        notes=notes,
    )

    try:
        result = await prediction_service.analyze_lab_report(
            request=lab_request,
            user_id=current_user.id,
            supabase=supabase,
            file_url=file_url,   # FIX: pass storage URL to be saved in DB
        )
        return result
    except Exception as e:
        log.error("lab_reports.analyze_after_upload_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/history")
async def get_lab_history(
    current_user: CurrentUser,
    supabase: SupabaseClient,
    pagination: Pagination,
):
    """Get current user's lab report history, most recent first."""
    try:
        result = (
            supabase.table("lab_reports")
            .select("id, report_type, overall_status, likely_conditions, created_at")
            .eq("user_id", current_user.id)
            .order("created_at", desc=True)
            .range(pagination["skip"], pagination["skip"] + pagination["limit"] - 1)
            .execute()
        )
        return {
            "reports": result.data,
            "total":   len(result.data),
            "skip":    pagination["skip"],
            "limit":   pagination["limit"],
        }
    except Exception as e:
        log.error("lab_reports.history_fetch_failed", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch lab report history.")


@router.get("/{report_id}")
async def get_lab_report(
    report_id: str,
    current_user: CurrentUser,
    supabase: SupabaseClient,
):
    """Get a single lab report by ID."""
    try:
        result = (
            supabase.table("lab_reports")
            .select("*")
            .eq("id", report_id)
            .eq("user_id", current_user.id)
            .single()
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Lab report not found.")
        return result.data
    except HTTPException:
        raise
    except Exception as e:
        log.error("lab_reports.get_failed", report_id=report_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch lab report.")