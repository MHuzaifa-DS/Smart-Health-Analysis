"""
main.py — FastAPI application entry point.
Registers all routers, startup/shutdown events, middleware, and exception handlers.
"""
import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.config import settings
from app.ml.inference import predictor
from app.routers import auth, symptoms, lab_reports, predictions, recommendations, dashboard, chat

log = structlog.get_logger()


# ── Lifespan (startup / shutdown) ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: Load ML models into memory.
    Shutdown: Cleanup (nothing heavy needed currently).
    """
    log.info("app.startup", environment=settings.environment)

    # Load all 3 ML models from disk
    predictor.load_models()

    model_info = predictor.get_model_info()
    available = [m["disease"] for m in model_info if m["available"]]
    missing = [m["disease"] for m in model_info if not m["available"]]

    if available:
        log.info("app.models_loaded", available=available)
    if missing:
        log.warning(
            "app.models_missing",
            missing=missing,
            hint="Run: python -m app.ml.train.train_models --disease all",
        )

    log.info(
        "app.ready",
        rag_index=settings.pinecone_index_name,
        llm=settings.llm_model,
    )

    yield  # Application runs here

    log.info("app.shutdown")


# ── App instance ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Smart Health Assistant API",
    description=(
        "AI-powered symptom analysis and lab report interpretation. "
        "Uses RAG over the Gale Encyclopedia of Medicine + ML models "
        "for Diabetes, Hypertension, and Anemia prediction."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ── CORS ───────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request logging middleware ─────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 1)

    log.info(
        "http.request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=duration_ms,
    )
    return response


# ── Global exception handlers ──────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Return clean 422 errors with field-level details."""
    errors = []
    for error in exc.errors():
        field = " → ".join(str(loc) for loc in error["loc"])
        errors.append({"field": field, "message": error["msg"], "type": error["type"]})
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation failed", "errors": errors},
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    log.error("app.unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal error occurred.",
            "hint": "Please try again. If this persists, contact support.",
        },
    )


# ── Routers ────────────────────────────────────────────────────────────────────

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(symptoms.router)
app.include_router(lab_reports.router)
app.include_router(predictions.router)
app.include_router(recommendations.router)
app.include_router(dashboard.router)


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """
    Public health check endpoint.
    Returns status of ML models and key integrations.
    """
    model_info = predictor.get_model_info()

    # Quick Pinecone connectivity check
    pinecone_ok = False
    pinecone_stats = {}
    try:
        from app.rag.pinecone_client import get_index_stats
        pinecone_stats = get_index_stats()
        pinecone_ok = True
    except Exception as e:
        log.warning("health.pinecone_check_failed", error=str(e))

    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": settings.environment,
        "ml_models": model_info,
        "rag": {
            "pinecone_connected": pinecone_ok,
            "index_name": settings.pinecone_index_name,
            "vector_count": pinecone_stats.get("total_vector_count", "unknown"),
            "llm_model": settings.llm_model,
        },
    }


@app.get("/", tags=["System"])
async def root():
    return {
        "name": "Smart Health Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
