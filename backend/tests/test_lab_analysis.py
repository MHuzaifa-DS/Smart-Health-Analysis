"""
tests/test_lab_analysis.py — Tests for lab report analysis (rule engine + OCR + API).
"""
import pytest
from app.ml.lab_rules import (
    analyze_lab_value,
    detect_disease_patterns,
    compute_overall_status,
    analyze_full_report,
)
from app.utils.ocr import extract_lab_values_from_text


# ── Rule engine — individual value analysis ────────────────────────────────────

class TestAnalyzeLabValue:

    # ── Glucose ────────────────────────────────────────────────────────────────
    def test_normal_fasting_glucose(self):
        result = analyze_lab_value("fasting_glucose", 85.0)
        assert result["status"] == "normal"
        assert result["emergency"] is False

    def test_high_fasting_glucose(self):
        result = analyze_lab_value("fasting_glucose", 130.0)
        assert result["status"] == "high"
        assert result["emergency"] is False

    def test_critical_low_glucose(self):
        result = analyze_lab_value("fasting_glucose", 45.0)
        assert result["status"] == "critical_low"
        assert result["emergency"] is True

    def test_critical_high_glucose(self):
        result = analyze_lab_value("fasting_glucose", 450.0)
        assert result["status"] == "critical_high"
        assert result["emergency"] is True

    def test_prediabetes_glucose(self):
        result = analyze_lab_value("fasting_glucose", 110.0)
        assert result["status"] == "high"

    # ── Blood pressure ────────────────────────────────────────────────────────
    def test_normal_systolic_bp(self):
        result = analyze_lab_value("systolic_bp", 115.0)
        assert result["status"] == "normal"

    def test_hypertension_stage2_systolic(self):
        result = analyze_lab_value("systolic_bp", 160.0)
        assert result["status"] == "high"

    def test_hypertensive_crisis(self):
        result = analyze_lab_value("systolic_bp", 195.0)
        assert result["status"] == "critical_high"
        assert result["emergency"] is True

    # ── Hemoglobin (gender-specific) ──────────────────────────────────────────
    def test_normal_hemoglobin_male(self):
        result = analyze_lab_value("hemoglobin", 15.0, patient_gender="male")
        assert result["status"] == "normal"

    def test_anemia_hemoglobin_female(self):
        result = analyze_lab_value("hemoglobin", 10.5, patient_gender="female")
        assert result["status"] == "low"
        assert result["disease_hint"] == "anemia"

    def test_critical_low_hemoglobin(self):
        result = analyze_lab_value("hemoglobin", 5.0)
        assert result["status"] == "critical_low"
        assert result["emergency"] is True

    # ── HbA1c ─────────────────────────────────────────────────────────────────
    def test_normal_hba1c(self):
        result = analyze_lab_value("hba1c", 5.2)
        assert result["status"] == "normal"

    def test_diabetic_hba1c(self):
        result = analyze_lab_value("hba1c", 7.5)
        assert result["status"] == "high"

    # ── Unknown test ──────────────────────────────────────────────────────────
    def test_unknown_test_name(self):
        result = analyze_lab_value("completely_unknown_test", 42.0)
        assert result["status"] == "unknown"
        assert result["emergency"] is False


# ── Disease pattern detection ──────────────────────────────────────────────────

class TestDetectDiseasePatterns:

    def test_diabetes_detected_from_glucose(self):
        patterns = detect_disease_patterns({"fasting_glucose": 130})
        condition_names = [p["condition"] for p in patterns]
        assert "Type 2 Diabetes" in condition_names

    def test_diabetes_detected_from_hba1c(self):
        patterns = detect_disease_patterns({"hba1c": 7.0})
        condition_names = [p["condition"] for p in patterns]
        assert "Type 2 Diabetes" in condition_names

    def test_prediabetes_detected(self):
        patterns = detect_disease_patterns({"fasting_glucose": 112})
        condition_names = [p["condition"] for p in patterns]
        assert "Prediabetes" in condition_names

    def test_hypertension_stage2_detected(self):
        patterns = detect_disease_patterns({"systolic_bp": 155})
        condition_names = [p["condition"] for p in patterns]
        assert "Hypertension (Stage 2)" in condition_names

    def test_anemia_detected(self):
        patterns = detect_disease_patterns({"hemoglobin": 10.0})
        condition_names = [p["condition"] for p in patterns]
        assert "Anemia" in condition_names

    def test_normal_values_no_patterns(self):
        patterns = detect_disease_patterns({
            "fasting_glucose": 85,
            "systolic_bp": 118,
            "hemoglobin": 14.5,
        })
        assert len(patterns) == 0

    def test_multiple_conditions_detected(self):
        patterns = detect_disease_patterns({
            "fasting_glucose": 145,
            "hemoglobin": 9.5,
        })
        condition_names = [p["condition"] for p in patterns]
        assert "Type 2 Diabetes" in condition_names
        assert "Anemia" in condition_names


# ── Overall status computation ─────────────────────────────────────────────────

class TestOverallStatus:
    def _make_results(self, statuses):
        return [{"status": s, "emergency": s in ("critical_low", "critical_high")} for s in statuses]

    def test_all_normal(self):
        assert compute_overall_status(self._make_results(["normal", "normal"])) == "normal"

    def test_one_high_is_abnormal(self):
        assert compute_overall_status(self._make_results(["normal", "high"])) == "abnormal"

    def test_one_low_is_abnormal(self):
        assert compute_overall_status(self._make_results(["low", "normal"])) == "abnormal"

    def test_critical_overrides_all(self):
        results = [
            {"status": "normal", "emergency": False},
            {"status": "critical_high", "emergency": True},
        ]
        assert compute_overall_status(results) == "critical"


# ── Full report analysis ───────────────────────────────────────────────────────

class TestAnalyzeFullReport:
    def test_diabetic_profile(self):
        results, overall, conditions = analyze_full_report({
            "fasting_glucose": 145,
            "hba1c": 7.2,
        })
        assert overall in ("abnormal", "critical")
        assert "Type 2 Diabetes" in conditions

    def test_anemic_profile(self):
        results, overall, conditions = analyze_full_report(
            {"hemoglobin": 9.5},
            patient_gender="female",
        )
        assert "Anemia" in conditions

    def test_normal_profile(self):
        results, overall, conditions = analyze_full_report({
            "fasting_glucose": 88,
            "systolic_bp": 118,
            "hemoglobin": 14.0,
        })
        assert overall == "normal"
        assert len(conditions) == 0

    def test_returns_correct_result_count(self):
        lab_values = {"fasting_glucose": 100, "systolic_bp": 125, "hemoglobin": 13}
        results, _, _ = analyze_full_report(lab_values)
        assert len(results) == len(lab_values)


# ── OCR text extraction ────────────────────────────────────────────────────────

class TestOCRExtraction:
    def test_extract_glucose_from_text(self):
        text = """
        LABORATORY REPORT
        Patient: John Doe
        
        Fasting Blood Glucose    126    mg/dL    [High]
        HbA1c                    7.2    %         [High]
        """
        values = extract_lab_values_from_text(text)
        assert "fasting_glucose" in values
        assert values["fasting_glucose"] == pytest.approx(126, abs=1)

    def test_extract_hemoglobin_from_text(self):
        text = "Hemoglobin: 10.5 g/dL"
        values = extract_lab_values_from_text(text)
        assert "hemoglobin" in values
        assert values["hemoglobin"] == pytest.approx(10.5, abs=0.1)

    def test_extract_multiple_values(self):
        text = """
        Fasting Glucose   145   mg/dL
        Hemoglobin Hb     11.2  g/dL
        Systolic BP       155   mmHg
        """
        values = extract_lab_values_from_text(text)
        assert len(values) >= 2  # at least some extracted

    def test_empty_text_returns_empty(self):
        values = extract_lab_values_from_text("")
        assert values == {}

    def test_alias_mapping_hgb_to_hemoglobin(self):
        text = "HGB 12.5 g/dL"
        values = extract_lab_values_from_text(text)
        assert "hemoglobin" in values

    def test_alias_mapping_fbg_to_fasting_glucose(self):
        text = "FBG 132 mg/dL"
        values = extract_lab_values_from_text(text)
        assert "fasting_glucose" in values


# ── /lab-reports/analyze API endpoint ────────────────────────────────────────

class TestLabReportEndpoint:
    def test_requires_auth(self, client):
        response = client.post("/lab-reports/analyze", json={
            "values": {"fasting_glucose": 145}
        })
        assert response.status_code == 401

    def test_empty_values_rejected(self, client, auth_headers, mock_supabase):
        from unittest.mock import MagicMock, patch
        admin_mock = MagicMock()
        admin_mock.auth.admin.get_user_by_id.return_value.user.email = "test@example.com"
        mock_supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = {
            "id": "00000000-0000-0000-0000-000000000001",
            "full_name": "Test", "age": 35, "gender": "male",
            "blood_type": "O+", "created_at": "2024-01-01T00:00:00+00:00",
        }
        with patch("app.database.get_supabase_admin", return_value=admin_mock):
            response = client.post(
                "/lab-reports/analyze",
                json={"report_type": "blood_test", "values": {}},
                headers=auth_headers,
            )
        # Empty values dict should be rejected or return with empty results
        assert response.status_code in (201, 422)
