"""
tests/conftest.py — Shared pytest fixtures and test client setup.
"""
import os
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient

# Set test environment variables BEFORE importing app
os.environ.update({
    "SUPABASE_URL":          "https://test.supabase.co",
    "SUPABASE_ANON_KEY":     "test-anon-key",
    "SUPABASE_SERVICE_KEY":  "test-service-key",
    "JWT_SECRET":            "test-secret-key-that-is-at-least-32-chars-long",
    "JWT_ALGORITHM":         "HS256",
    "JWT_EXPIRY_MINUTES":    "60",
    "OPENAI_API_KEY":        "sk-test",
    "ANTHROPIC_API_KEY":     "sk-ant-test",
    "PINECONE_API_KEY":      "test-pinecone-key",
    "PINECONE_ENVIRONMENT":  "us-east-1",
    "PINECONE_INDEX_NAME":   "test-index",
    "EMBEDDING_MODEL":       "text-embedding-ada-002",
    "LLM_MODEL":             "claude-sonnet-4-20250514",
    "LLM_MAX_TOKENS":        "1000",
    "RAG_TOP_K":             "3",
    "RAG_MIN_SCORE":         "0.7",
    "ML_FALLBACK_THRESHOLD": "0.60",
    "STORAGE_BUCKET":        "lab-reports",
    "ENVIRONMENT":           "test",
})

from app.main import app
from app.utils.jwt import create_access_token, create_refresh_token


@pytest.fixture(scope="session")
def test_user_id():
    return "00000000-0000-0000-0000-000000000001"


@pytest.fixture(scope="session")
def test_user_email():
    return "test@example.com"


@pytest.fixture(scope="session")
def test_access_token(test_user_id, test_user_email):
    return create_access_token(test_user_id, test_user_email)


@pytest.fixture(scope="session")
def auth_headers(test_access_token):
    return {"Authorization": f"Bearer {test_access_token}"}


@pytest.fixture
def mock_supabase():
    """Mock Supabase client that returns sensible defaults."""
    mock = MagicMock()

    # Profile
    mock.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = {
        "id": "00000000-0000-0000-0000-000000000001",
        "full_name": "Test User",
        "age": 35,
        "gender": "male",
        "blood_type": "O+",
        "created_at": "2024-01-01T00:00:00+00:00",
    }

    # Insert operations
    mock.table.return_value.insert.return_value.execute.return_value.data = [
        {"id": "00000000-0000-0000-0000-000000000099"}
    ]

    # Update operations
    mock.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [
        {"id": "00000000-0000-0000-0000-000000000001", "full_name": "Updated User", "age": 35, "gender": "male"}
    ]

    return mock


@pytest.fixture
def mock_predictor():
    """Mock ML predictor."""
    with patch("app.ml.inference.predictor") as mock:
        mock.is_available.return_value = True
        mock.predict_all.return_value = {
            "diabetes": {
                "disease": "diabetes",
                "display_name": "Type 2 Diabetes",
                "available": True,
                "probability": 0.72,
                "risk_level": "medium",
                "above_threshold": True,
                "model_name": "diabetes_random_forest",
                "model_version": "v1",
            },
            "hypertension": {
                "disease": "hypertension",
                "display_name": "Hypertension",
                "available": True,
                "probability": 0.31,
                "risk_level": "low",
                "above_threshold": False,
                "model_name": "hypertension_gradient_boosting",
                "model_version": "v1",
            },
            "anemia": {
                "disease": "anemia",
                "display_name": "Anemia",
                "available": True,
                "probability": 0.28,
                "risk_level": "low",
                "above_threshold": False,
                "model_name": "anemia_svm",
                "model_version": "v1",
            },
        }
        mock.get_model_info.return_value = [
            {"disease": "diabetes",     "available": True,  "accuracy": 0.82, "auc": 0.88, "version": "v1"},
            {"disease": "hypertension", "available": True,  "accuracy": 0.79, "auc": 0.84, "version": "v1"},
            {"disease": "anemia",       "available": True,  "accuracy": 0.91, "auc": 0.95, "version": "v1"},
        ]
        yield mock


@pytest.fixture
def mock_retriever():
    """Mock Pinecone retriever."""
    with patch("app.rag.retriever.retrieve_for_symptoms") as mock_sym, \
         patch("app.rag.retriever.retrieve_for_lab_values") as mock_lab:

        # Build a fake RetrievedChunk
        from unittest.mock import MagicMock
        fake_chunk = MagicMock()
        fake_chunk.chunk_id = "gale_diabetes_causes_symptoms_1847_0"
        fake_chunk.score = 0.89
        fake_chunk.disease_name = "Diabetes mellitus"
        fake_chunk.section = "causes_symptoms"
        fake_chunk.text = (
            "Diabetes mellitus — Causes and Symptoms\n\n"
            "Type 2 diabetes is characterized by frequent urination (polyuria), "
            "excessive thirst (polydipsia), blurred vision, fatigue, and slow wound healing. "
            "The condition results from insulin resistance and relative insulin deficiency."
        )
        fake_chunk.page_number = 1847

        mock_sym.return_value = [fake_chunk]
        mock_lab.return_value = [fake_chunk]
        yield mock_sym, mock_lab


@pytest.fixture
def mock_prompt_builder():
    """Mock LLM call to avoid real API calls in tests."""
    with patch("app.rag.prompt_builder.generate_symptom_prediction") as mock_sym, \
         patch("app.rag.prompt_builder.generate_lab_interpretation") as mock_lab:

        mock_sym.return_value = (
            {
                "predictions": [
                    {
                        "disease": "Type 2 Diabetes",
                        "confidence": "high",
                        "confidence_score": 0.84,
                        "matching_symptoms": ["frequent urination", "fatigue", "blurred vision"],
                        "explanation": "The reported symptoms of frequent urination and blurred vision are hallmark signs of diabetes mellitus as described in the Gale Encyclopedia.",
                        "source_chunks": ["gale_diabetes_causes_symptoms_1847_0"],
                    }
                ],
                "recommended_tests": ["HbA1c", "Fasting Blood Glucose"],
                "emergency": False,
                "emergency_reason": None,
                "disclaimer": "This is a preliminary AI-assisted assessment only.",
            },
            '{"predictions": [...]}',  # raw LLM response
        )

        mock_lab.return_value = (
            {
                "interpretation": "The elevated fasting glucose and HbA1c suggest poorly controlled diabetes.",
                "likely_conditions": ["Type 2 Diabetes"],
                "abnormal_flags": [],
                "recommended_followup": ["Endocrinologist referral"],
                "emergency": False,
                "disclaimer": "Requires professional medical evaluation.",
            },
            '{"interpretation": "..."}',
        )

        yield mock_sym, mock_lab


@pytest.fixture
def client(mock_supabase):
    """Test client with mocked Supabase."""
    with patch("app.database.get_supabase", return_value=mock_supabase), \
         patch("app.database.get_supabase_admin", return_value=mock_supabase):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
