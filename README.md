# Smart Health Assistant — Backend

AI-powered symptom analysis and lab report interpretation using **RAG** (Retrieval-Augmented Generation) over the Gale Encyclopedia of Medicine + ML models for Diabetes, Hypertension, and Anemia prediction.

---

## Architecture

```
User Query (symptoms / lab values)
        ↓
  NLP → Pinecone Vector Search (Gale Encyclopedia chunks)
        ↓
  Claude LLM (context + symptoms → structured JSON prediction)
        ↓
  ML Models (Random Forest / GBM / SVM — cross-validation layer)
        ↓
  Merged Result → Supabase (saved) → API Response
```

**Stack:** FastAPI · Supabase (PostgreSQL) · Pinecone · OpenAI Embeddings · Anthropic Claude · scikit-learn · XGBoost

---

## Quick Start

### 1. Prerequisites

```bash
# Python 3.11+
python --version

# Tesseract OCR (for lab report PDF upload)
# Ubuntu/Debian:
sudo apt install tesseract-ocr poppler-utils
# macOS:
brew install tesseract poppler
```

### 2. Install dependencies

```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your real keys:
# - SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_KEY
# - OPENAI_API_KEY
# - ANTHROPIC_API_KEY
# - PINECONE_API_KEY
```

### 4. Run Supabase migrations

Open your Supabase project → SQL Editor → run in order:
```
supabase/migrations/001_core_tables.sql
supabase/migrations/002_rls_and_indexes.sql
supabase/migrations/003_storage.sql
```

### 5. Train ML models

```bash
cd backend
# Uses synthetic data if Kaggle CSVs not present (good for dev)
python -m app.ml.train.train_models --disease all

# To use real datasets, place CSVs in app/ml/train/data/:
#   diabetes.csv     → https://kaggle.com/uciml/pima-indians-diabetes-database
#   framingham.csv   → https://kaggle.com/datasets/dileep070/heart-disease-prediction
#   anemia.csv       → https://kaggle.com/datasets/biswa96/anemia-detection
```

### 6. Ingest the Gale Encyclopedia (one-time, ~2-4 hours)

```bash
python -m app.rag.ingest_pipeline \
  --pdf /path/to/The-Gale-Encyclopedia-of-Medicine-3rd-Edition.pdf

# Dry run (chunk only, no API calls):
python -m app.rag.ingest_pipeline --pdf /path/to/gale.pdf --dry-run

# Resume interrupted ingestion:
python -m app.rag.ingest_pipeline --pdf /path/to/gale.pdf  # auto-resumes
```

### 7. Start the server

```bash
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/auth/register` | ❌ | Register new user |
| POST | `/auth/login` | ❌ | Login, get JWT |
| POST | `/auth/refresh` | ❌ | Refresh access token |
| GET | `/auth/me` | ✅ | Get current user |
| PUT | `/auth/profile` | ✅ | Update profile |
| POST | `/auth/logout` | ✅ | Sign out |
| GET | `/symptoms/list` | ❌ | Symptom catalogue |
| POST | `/symptoms/analyze` | ✅ | **RAG+ML prediction** |
| POST | `/lab-reports/analyze` | ✅ | Manual lab value analysis |
| POST | `/lab-reports/upload` | ✅ | PDF/image upload + OCR |
| GET | `/lab-reports/history` | ✅ | Past lab reports |
| GET | `/lab-reports/{id}` | ✅ | Single lab report |
| GET | `/predictions/history` | ✅ | Prediction history |
| GET | `/predictions/{id}` | ✅ | Prediction detail |
| GET | `/predictions/{id}/sources` | ✅ | Cited encyclopedia chunks |
| GET | `/recommendations/{id}` | ✅ | Tests + specialists + tips |
| GET | `/dashboard/summary` | ✅ | Health overview |
| GET | `/dashboard/metrics` | ✅ | Time-series metrics |
| POST | `/dashboard/metrics` | ✅ | Record a metric |
| GET | `/health` | ❌ | System health check |

---

## Example: Symptom Analysis

```bash
curl -X POST http://localhost:8000/symptoms/analyze \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["fatigue", "frequent urination", "blurred vision"],
    "severity": {"fatigue": 7, "frequent urination": 8},
    "duration_days": 14,
    "age": 45,
    "gender": "male"
  }'
```

Response:
```json
{
  "prediction_id": "uuid",
  "predictions": [
    {
      "disease": "Type 2 Diabetes",
      "confidence": "high",
      "confidence_score": 0.87,
      "matching_symptoms": ["frequent urination", "fatigue", "blurred vision"],
      "explanation": "The reported symptoms align with classic presentation of Type 2 Diabetes as described in the Gale Encyclopedia...",
      "source_chunks": ["gale_diabetes_causes_symptoms_1847_0"]
    }
  ],
  "recommended_tests": ["HbA1c", "Fasting Blood Glucose"],
  "emergency": false,
  "prediction_method": "rag_ml_combined",
  "disclaimer": "This is a preliminary AI-assisted assessment only..."
}
```

---

## Running Tests

```bash
cd backend
pytest                          # all tests
pytest tests/test_predictions.py -v
pytest --cov=app --cov-report=html
```

---

## Deployment (Render.com)

1. Push to GitHub
2. Create a new **Web Service** on Render pointing to `/backend`
3. Set Build Command: `pip install -r requirements.txt`
4. Set Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add all environment variables from `.env.example`
6. The GitHub Actions workflow auto-deploys on push to `main`

---

## Project Structure

```
backend/
├── app/
│   ├── main.py                  # FastAPI app entry point
│   ├── config.py                # Settings from .env
│   ├── database.py              # Supabase client
│   ├── dependencies.py          # Auth middleware
│   ├── models/                  # Pydantic schemas
│   ├── routers/                 # API route handlers
│   ├── services/                # Business logic
│   ├── rag/                     # RAG pipeline (chunk→embed→retrieve→LLM)
│   ├── ml/                      # ML models + training + inference
│   └── utils/                   # JWT, OCR helpers
├── tests/                       # pytest test suite
├── requirements.txt
├── Dockerfile
└── .env.example

supabase/
└── migrations/                  # SQL migrations (run in Supabase SQL editor)
```

---

## Notes

- **RAG requires Pinecone + OpenAI API keys** — without ingestion, the system falls back to ML-only mode automatically
- **ML models fall back to synthetic training data** if Kaggle datasets are not present — replace with real data for production accuracy
- All predictions include a medical disclaimer — this system is for preliminary assessment only
- Lab report OCR requires Tesseract installed on the host system
