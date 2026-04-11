"""
tests/test_predictions.py — Tests for symptom analysis and prediction endpoints.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.ml.inference import merge_rag_and_ml_results, _ml_only_predictions
from app.ml.preprocessor import feature_builder


# ── Feature builder tests ──────────────────────────────────────────────────────

class TestFeatureBuilder:
    def test_diabetes_features_shape(self):
        X = feature_builder.build_diabetes_features(
            symptoms=["frequent urination", "blurred vision"],
            age=45,
            gender="male",
        )
        assert X.shape == (1, 8)
        assert X[0, 7] == 45  # age is last feature

    def test_hypertension_features_shape(self):
        X = feature_builder.build_hypertension_features(
            symptoms=["headache", "dizziness"],
            age=55,
            gender="male",
        )
        assert X.shape == (1, 10)

    def test_anemia_features_shape(self):
        X = feature_builder.build_anemia_features(
            symptoms=["fatigue", "pale skin"],
            age=30,
            gender="female",
        )
        assert X.shape == (1, 6)
        assert X[0, 5] == 1  # gender_female = 1

    def test_lab_values_override_defaults(self):
        X = feature_builder.build_diabetes_features(
            symptoms=[],
            lab_values={"fasting_glucose": 200, "bmi": 35.0},
        )
        # glucose is index 1 in DIABETES_FEATURES
        assert X[0, 1] == 200

    def test_symptom_signals_applied(self):
        """Symptom 'frequent urination' should increase glucose estimate."""
        X_with = feature_builder.build_diabetes_features(
            symptoms=["frequent urination", "excessive thirst"]
        )
        X_without = feature_builder.build_diabetes_features(symptoms=[])
        # glucose (idx 1) should be higher when high-signal symptoms present
        assert X_with[0, 1] >= X_without[0, 1]


# ── Merge logic tests ──────────────────────────────────────────────────────────

class TestMergeResults:
    def _rag_pred(self, disease="Type 2 Diabetes", score=0.80):
        return {
            "disease": disease,
            "confidence": "high",
            "confidence_score": score,
            "matching_symptoms": ["fatigue"],
            "explanation": "Test explanation.",
            "source_chunks": ["chunk_001"],
        }

    def _ml_result(self, disease="diabetes", prob=0.72, available=True):
        return {
            disease: {
                "disease": disease,
                "display_name": "Type 2 Diabetes",
                "available": available,
                "probability": prob,
                "risk_level": "medium",
                "above_threshold": prob >= 0.5,
                "model_name": "random_forest",
                "model_version": "v1",
            }
        }

    def test_rag_and_ml_agree_boosts_confidence(self):
        preds, method = merge_rag_and_ml_results(
            rag_predictions=[self._rag_pred(score=0.80)],
            ml_results=self._ml_result(prob=0.72),
        )
        assert method == "rag_ml_combined"
        assert preds[0]["confidence_score"] > 0.80  # boosted

    def test_rag_and_ml_disagree_downgrades(self):
        preds, method = merge_rag_and_ml_results(
            rag_predictions=[self._rag_pred(score=0.55)],
            ml_results=self._ml_result(prob=0.20),
        )
        assert preds[0]["confidence_score"] < 0.55  # downgraded

    def test_no_rag_falls_back_to_ml(self):
        preds, method = merge_rag_and_ml_results(
            rag_predictions=[],
            ml_results=self._ml_result(prob=0.75),
        )
        assert method == "ml_only"
        assert len(preds) > 0

    def test_ml_only_below_threshold_excluded(self):
        preds = _ml_only_predictions({
            "diabetes": {
                "available": True,
                "probability": 0.30,  # below 0.45 threshold
                "display_name": "Type 2 Diabetes",
            }
        })
        assert len(preds) == 0

    def test_high_ml_only_included(self):
        preds = _ml_only_predictions({
            "diabetes": {
                "available": True,
                "probability": 0.80,
                "display_name": "Type 2 Diabetes",
            }
        })
        assert len(preds) == 1
        assert preds[0]["confidence"] == "high"

    def test_empty_inputs_return_none_method(self):
        preds, method = merge_rag_and_ml_results([], {})
        assert preds == []
        assert method == "none"

    def test_sorted_by_confidence_score(self):
        preds, _ = merge_rag_and_ml_results(
            rag_predictions=[
                self._rag_pred("Type 2 Diabetes", 0.55),
                self._rag_pred("Hypertension", 0.80),
            ],
            ml_results={},
        )
        scores = [p["confidence_score"] for p in preds]
        assert scores == sorted(scores, reverse=True)


# ── /symptoms/list endpoint ────────────────────────────────────────────────────

class TestSymptomList:
    def test_symptom_list_returns_array(self, client):
        response = client.get("/symptoms/list")
        assert response.status_code == 200
        data = response.json()
        assert "symptoms" in data
        assert isinstance(data["symptoms"], list)
        assert len(data["symptoms"]) > 10
        assert "fatigue" in data["symptoms"]

    def test_symptom_list_is_sorted(self, client):
        response = client.get("/symptoms/list")
        symptoms = response.json()["symptoms"]
        assert symptoms == sorted(symptoms)


# ── /symptoms/analyze endpoint ────────────────────────────────────────────────

class TestSymptomAnalyze:
    def _make_request(self, symptoms=None):
        return {
            "symptoms": symptoms or ["fatigue", "frequent urination", "blurred vision"],
            "severity": {"fatigue": 7, "frequent urination": 8},
            "duration_days": 14,
            "age": 45,
            "gender": "male",
        }

    def test_analyze_requires_auth(self, client):
        response = client.post("/symptoms/analyze", json=self._make_request())
        assert response.status_code == 401

    def test_analyze_empty_symptoms_rejected(self, client, auth_headers):
        admin_mock = MagicMock()
        admin_mock.auth.admin.get_user_by_id.return_value.user.email = "test@example.com"
        with patch("app.database.get_supabase_admin", return_value=admin_mock):
            response = client.post(
                "/symptoms/analyze",
                json={"symptoms": []},
                headers=auth_headers,
            )
        assert response.status_code == 422

    def test_analyze_success_with_mocks(
        self, client, auth_headers, mock_supabase,
        mock_retriever, mock_prompt_builder, mock_predictor
    ):
        admin_mock = MagicMock()
        admin_mock.auth.admin.get_user_by_id.return_value.user.email = "test@example.com"
        mock_supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = {
            "id": "00000000-0000-0000-0000-000000000001",
            "full_name": "Test", "age": 35, "gender": "male",
            "blood_type": "O+", "created_at": "2024-01-01T00:00:00+00:00",
        }

        with patch("app.database.get_supabase_admin", return_value=admin_mock), \
             patch("app.services.prediction_service.predictor", mock_predictor), \
             patch("app.services.prediction_service.retriever.retrieve_for_symptoms", mock_retriever[0]), \
             patch("app.services.prediction_service.prompt_builder.generate_symptom_prediction", mock_prompt_builder[0]):

            response = client.post(
                "/symptoms/analyze",
                json=self._make_request(),
                headers=auth_headers,
            )

        assert response.status_code == 201
        data = response.json()
        assert "prediction_id" in data
        assert "predictions" in data
        assert isinstance(data["predictions"], list)
        assert "disclaimer" in data
        assert data["emergency"] is False

    def test_analyze_response_schema(
        self, client, auth_headers, mock_supabase,
        mock_retriever, mock_prompt_builder, mock_predictor
    ):
        admin_mock = MagicMock()
        admin_mock.auth.admin.get_user_by_id.return_value.user.email = "test@example.com"
        mock_supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = {
            "id": "00000000-0000-0000-0000-000000000001",
            "full_name": "Test", "age": 35, "gender": "male",
            "blood_type": "O+", "created_at": "2024-01-01T00:00:00+00:00",
        }

        with patch("app.database.get_supabase_admin", return_value=admin_mock), \
             patch("app.services.prediction_service.predictor", mock_predictor), \
             patch("app.services.prediction_service.retriever.retrieve_for_symptoms", mock_retriever[0]), \
             patch("app.services.prediction_service.prompt_builder.generate_symptom_prediction", mock_prompt_builder[0]):

            response = client.post(
                "/symptoms/analyze",
                json=self._make_request(),
                headers=auth_headers,
            )

        if response.status_code == 201:
            data = response.json()
            required_fields = ["prediction_id", "predictions", "emergency", "prediction_method", "disclaimer", "created_at"]
            for field in required_fields:
                assert field in data, f"Missing field: {field}"

            if data["predictions"]:
                pred = data["predictions"][0]
                pred_fields = ["disease", "confidence", "confidence_score", "matching_symptoms", "explanation"]
                for field in pred_fields:
                    assert field in pred, f"Missing prediction field: {field}"
                assert 0.0 <= pred["confidence_score"] <= 1.0
                assert pred["confidence"] in ("high", "medium", "low")
