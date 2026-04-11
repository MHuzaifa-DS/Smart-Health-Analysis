-- ============================================================
-- Migration 001: Core tables
-- Run in Supabase SQL editor or via supabase CLI
-- ============================================================

-- ── Profiles ─────────────────────────────────────────────────
-- Extends Supabase auth.users with health-specific fields
CREATE TABLE IF NOT EXISTS profiles (
    id          UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    full_name   TEXT,
    age         INTEGER CHECK (age > 0 AND age < 150),
    gender      TEXT CHECK (gender IN ('male', 'female', 'other')),
    blood_type  TEXT CHECK (blood_type IN ('A+','A-','B+','B-','AB+','AB-','O+','O-')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER profiles_updated_at
    BEFORE UPDATE ON profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();


-- ── Symptom checks ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS symptom_checks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    symptoms        JSONB NOT NULL,            -- array of symptom strings
    severity_scores JSONB DEFAULT '{}',        -- {symptom: 1-10}
    duration_days   INTEGER,
    free_text       TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── RAG retrievals ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS rag_retrievals (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id       UUID,                  -- set after prediction is created
    query_text          TEXT NOT NULL,
    retrieved_chunk_ids JSONB DEFAULT '[]',    -- array of Pinecone chunk IDs
    retrieved_contexts  JSONB DEFAULT '[]',    -- [{id, text}, ...]
    llm_raw_response    TEXT,                  -- full LLM output for debugging
    retrieval_scores    JSONB DEFAULT '[]',    -- [{id, score}, ...]
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Predictions ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    symptom_check_id    UUID REFERENCES symptom_checks(id),
    rag_retrieval_id    UUID REFERENCES rag_retrievals(id),
    disease             TEXT NOT NULL,
    confidence_score    FLOAT NOT NULL CHECK (confidence_score BETWEEN 0.0 AND 1.0),
    risk_level          TEXT CHECK (risk_level IN ('low', 'medium', 'high')),
    prediction_method   TEXT DEFAULT 'rag_ml_combined'
                        CHECK (prediction_method IN ('rag_ml_combined','rag_only','ml_only')),
    model_version       TEXT DEFAULT 'v1',
    feature_values      JSONB DEFAULT '{}',    -- raw input features used
    source_chunks       JSONB DEFAULT '[]',    -- Pinecone chunk IDs cited
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Back-fill the FK from rag_retrievals → predictions
ALTER TABLE rag_retrievals
    ADD CONSTRAINT fk_rag_retrieval_prediction
    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
    ON DELETE SET NULL;

-- ── Lab reports ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS lab_reports (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    report_type         TEXT DEFAULT 'blood_test',
    file_url            TEXT,                  -- Supabase Storage URL (OCR upload)
    raw_values          JSONB NOT NULL,        -- {test_name: value}
    interpreted_results JSONB DEFAULT '[]',    -- rule engine output array
    overall_status      TEXT DEFAULT 'normal'
                        CHECK (overall_status IN ('normal','borderline','abnormal','critical')),
    likely_conditions   JSONB DEFAULT '[]',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Recommendations ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS recommendations (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                 UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    prediction_id           UUID REFERENCES predictions(id) ON DELETE SET NULL,
    lab_report_id           UUID REFERENCES lab_reports(id) ON DELETE SET NULL,
    recommended_tests       JSONB DEFAULT '[]',
    recommended_specialists JSONB DEFAULT '[]',
    health_tips             JSONB DEFAULT '[]',
    emergency_alert         BOOLEAN DEFAULT FALSE,
    emergency_message       TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Health metrics (time-series tracking) ─────────────────────
CREATE TABLE IF NOT EXISTS health_metrics (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    metric_type     TEXT NOT NULL,             -- blood_sugar, blood_pressure, etc.
    value           FLOAT NOT NULL,
    unit            TEXT NOT NULL,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Knowledge base versions ───────────────────────────────────
CREATE TABLE IF NOT EXISTS knowledge_base_versions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_document     TEXT NOT NULL,
    total_chunks        INTEGER,
    pinecone_index_name TEXT,
    embedding_model     TEXT,
    ingested_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_active           BOOLEAN DEFAULT TRUE
);
