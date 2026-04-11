"""
tests/test_rag.py — Tests for the RAG pipeline (chunker, prompt builder, merge logic).
"""
import pytest
from app.rag.chunker import (
    chunk_encyclopedia_text,
    _split_into_disease_entries,
    _split_if_too_long,
    _extract_symptoms,
    MedicalChunk,
    MAX_CHUNK_CHARS,
)
from app.rag.prompt_builder import (
    build_context_string,
    parse_llm_json,
    _sanitize_prediction_response,
)


# ── Sample encyclopedia text fixtures ─────────────────────────────────────────

SAMPLE_DIABETES_ENTRY = """
Diabetes mellitus

Definition
Diabetes mellitus is a group of diseases characterized by high blood glucose levels.

Description
Diabetes mellitus affects millions worldwide. Type 2 diabetes is the most common form,
accounting for 90-95% of cases. It is characterized by insulin resistance.

Causes and symptoms
Causes
Type 2 diabetes develops due to a combination of genetic predisposition and lifestyle factors.
Obesity and physical inactivity are major risk factors.

Symptoms
Common symptoms include frequent urination, excessive thirst, blurred vision, fatigue,
and slow wound healing. Some patients experience numbness in the feet.

Diagnosis
Diagnosis is confirmed by fasting blood glucose >= 126 mg/dL, HbA1c >= 6.5%,
or random blood glucose >= 200 mg/dL with symptoms.

Treatment
Treatment includes lifestyle modification, oral hypoglycemic agents, and insulin therapy.
Metformin is typically the first-line medication.

Prevention
Type 2 diabetes can be delayed or prevented through weight loss, increased physical activity,
and a healthy diet low in refined carbohydrates.

KEY TERMS
Insulin—A hormone produced by the pancreas that regulates blood glucose.
Polyuria—Excessive urination, a hallmark symptom of diabetes.
HbA1c—Glycated hemoglobin; reflects average blood sugar over 2-3 months.

Resources
BOOKS
American Diabetes Association. Standards of Medical Care in Diabetes. 2023.
"""

SAMPLE_HYPERTENSION_ENTRY = """
Hypertension

Definition
Hypertension, or high blood pressure, is defined as systolic blood pressure >= 130 mmHg
or diastolic blood pressure >= 80 mmHg.

Description
Hypertension affects approximately 1 in 3 adults worldwide and is a major risk factor
for stroke, heart attack, and kidney disease.

Causes and symptoms
Causes
Essential (primary) hypertension has no single identifiable cause but involves genetic,
environmental, and lifestyle factors including obesity, high sodium intake, and stress.

Symptoms
Hypertension is often called the "silent killer" because it usually has no symptoms.
When present, symptoms may include headache, dizziness, nosebleed, and visual changes.

Diagnosis
Blood pressure is measured with a sphygmomanometer. Diagnosis requires persistent elevation
on multiple occasions.

Treatment
Treatment includes lifestyle changes and antihypertensive medications such as ACE inhibitors,
calcium channel blockers, and diuretics.

KEY TERMS
Systolic pressure—The pressure in arteries when the heart beats.
Diastolic pressure—The pressure in arteries between heartbeats.
"""


# ── Chunker tests ──────────────────────────────────────────────────────────────

class TestChunker:

    def test_chunks_from_sample_text(self):
        full_text = SAMPLE_DIABETES_ENTRY + "\n\n" + SAMPLE_HYPERTENSION_ENTRY
        chunks = chunk_encyclopedia_text(full_text)
        # Should produce multiple chunks
        assert len(chunks) >= 2

    def test_chunk_has_required_fields(self):
        chunks = chunk_encyclopedia_text(SAMPLE_DIABETES_ENTRY)
        if chunks:
            chunk = chunks[0]
            assert isinstance(chunk, MedicalChunk)
            assert chunk.chunk_id
            assert chunk.disease_name
            assert chunk.section
            assert chunk.text
            assert len(chunk.text) >= 50

    def test_no_chunk_exceeds_max_size(self):
        long_text = SAMPLE_DIABETES_ENTRY * 3
        chunks = chunk_encyclopedia_text(long_text)
        for chunk in chunks:
            assert chunk.char_count <= MAX_CHUNK_CHARS + 100  # small tolerance

    def test_disease_name_in_chunk_text(self):
        chunks = chunk_encyclopedia_text(SAMPLE_DIABETES_ENTRY)
        for chunk in chunks:
            # Disease name should appear in chunk text
            assert "Diabetes" in chunk.text or "diabetes" in chunk.text

    def test_symptoms_extracted(self):
        chunks = chunk_encyclopedia_text(SAMPLE_DIABETES_ENTRY)
        all_symptoms = []
        for chunk in chunks:
            all_symptoms.extend(chunk.symptoms_mentioned)
        assert len(all_symptoms) > 0

    def test_key_terms_chunk_created(self):
        chunks = chunk_encyclopedia_text(SAMPLE_DIABETES_ENTRY)
        sections = [c.section for c in chunks]
        assert "key_terms" in sections

    def test_causes_symptoms_chunk_created(self):
        chunks = chunk_encyclopedia_text(SAMPLE_DIABETES_ENTRY)
        sections = [c.section for c in chunks]
        assert "causes_symptoms" in sections

    def test_split_if_too_long(self):
        long_text = "A" * (MAX_CHUNK_CHARS + 500)
        parts = _split_if_too_long(long_text, "TestDisease", "causes_symptoms")
        assert len(parts) > 1
        for part in parts:
            assert len(part) <= MAX_CHUNK_CHARS + 200

    def test_symptom_extraction(self):
        text = "Patient reports fatigue, headache, and frequent urination."
        symptoms = _extract_symptoms(text)
        assert "fatigue" in symptoms
        assert "headache" in symptoms
        assert "frequent urination" in symptoms

    def test_chunk_id_is_unique(self):
        full_text = SAMPLE_DIABETES_ENTRY + "\n\n" + SAMPLE_HYPERTENSION_ENTRY
        chunks = chunk_encyclopedia_text(full_text)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_pinecone_metadata_format(self):
        chunks = chunk_encyclopedia_text(SAMPLE_DIABETES_ENTRY)
        if chunks:
            meta = chunks[0].to_pinecone_metadata()
            required_keys = ["disease_name", "section", "text", "page_number",
                             "symptoms_mentioned", "source"]
            for key in required_keys:
                assert key in meta

    def test_metadata_text_within_pinecone_limit(self):
        chunks = chunk_encyclopedia_text(SAMPLE_DIABETES_ENTRY)
        for chunk in chunks:
            meta = chunk.to_pinecone_metadata()
            assert len(meta["text"]) <= 40000  # Pinecone metadata limit


# ── Prompt builder tests ───────────────────────────────────────────────────────

class TestPromptBuilder:

    def _make_fake_chunk(self, disease="Diabetes mellitus", section="causes_symptoms", score=0.85):
        from unittest.mock import MagicMock
        chunk = MagicMock()
        chunk.chunk_id = f"gale_{disease.lower().replace(' ', '_')}_{section}_0"
        chunk.score = score
        chunk.disease_name = disease
        chunk.section = section
        chunk.text = f"{disease} — {section}\n\nSample medical content about {disease}."
        return chunk

    def test_build_context_string_includes_disease(self):
        chunks = [self._make_fake_chunk()]
        context = build_context_string(chunks)
        assert "Diabetes mellitus" in context
        assert "SOURCE 1" in context

    def test_build_context_respects_max_chars(self):
        # Many large chunks
        chunks = [self._make_fake_chunk() for _ in range(20)]
        context = build_context_string(chunks, max_chars=2000)
        assert len(context) <= 2100  # allow small overhead

    def test_build_context_empty_chunks(self):
        context = build_context_string([])
        assert "No relevant medical context" in context

    def test_parse_llm_json_valid(self):
        valid_json = '{"predictions": [], "emergency": false}'
        result = parse_llm_json(valid_json)
        assert isinstance(result, dict)
        assert "predictions" in result

    def test_parse_llm_json_strips_markdown(self):
        wrapped = '```json\n{"predictions": [], "emergency": false}\n```'
        result = parse_llm_json(wrapped)
        assert isinstance(result, dict)

    def test_parse_llm_json_invalid_raises(self):
        with pytest.raises((ValueError, Exception)):
            parse_llm_json("This is not JSON at all!!!")

    def test_sanitize_clips_confidence_score(self):
        data = {
            "predictions": [
                {
                    "disease": "Diabetes",
                    "confidence_score": 1.5,  # out of range
                    "matching_symptoms": [],
                    "source_chunks": [],
                }
            ],
            "emergency": False,
        }
        sanitized = _sanitize_prediction_response(data, [])
        assert sanitized["predictions"][0]["confidence_score"] <= 1.0

    def test_sanitize_removes_low_confidence_predictions(self):
        data = {
            "predictions": [
                {
                    "disease": "Something",
                    "confidence_score": 0.10,  # below 0.20 threshold
                    "matching_symptoms": [],
                    "source_chunks": [],
                }
            ],
            "emergency": False,
        }
        sanitized = _sanitize_prediction_response(data, [])
        assert len(sanitized["predictions"]) == 0

    def test_sanitize_derives_confidence_label(self):
        data = {
            "predictions": [
                {
                    "disease": "Diabetes",
                    "confidence_score": 0.80,
                    "matching_symptoms": [],
                    "source_chunks": [],
                }
            ],
            "emergency": False,
        }
        sanitized = _sanitize_prediction_response(data, [])
        assert sanitized["predictions"][0]["confidence"] == "high"

    def test_sanitize_sorts_by_score(self):
        data = {
            "predictions": [
                {"disease": "B", "confidence_score": 0.50, "matching_symptoms": [], "source_chunks": []},
                {"disease": "A", "confidence_score": 0.80, "matching_symptoms": [], "source_chunks": []},
            ],
            "emergency": False,
        }
        sanitized = _sanitize_prediction_response(data, [])
        scores = [p["confidence_score"] for p in sanitized["predictions"]]
        assert scores == sorted(scores, reverse=True)
