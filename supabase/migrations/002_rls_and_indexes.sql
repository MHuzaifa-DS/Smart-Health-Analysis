-- ============================================================
-- Migration 002: Row Level Security + Indexes
-- ============================================================

-- ── Enable RLS on all user-data tables ───────────────────────
ALTER TABLE profiles             ENABLE ROW LEVEL SECURITY;
ALTER TABLE symptom_checks       ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_retrievals       ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions          ENABLE ROW LEVEL SECURITY;
ALTER TABLE lab_reports          ENABLE ROW LEVEL SECURITY;
ALTER TABLE recommendations      ENABLE ROW LEVEL SECURITY;
ALTER TABLE health_metrics       ENABLE ROW LEVEL SECURITY;


-- ── Profiles policies ─────────────────────────────────────────
CREATE POLICY "Users can view own profile"
    ON profiles FOR SELECT
    USING (auth.uid() = id);

CREATE POLICY "Users can update own profile"
    ON profiles FOR UPDATE
    USING (auth.uid() = id);

CREATE POLICY "Users can insert own profile"
    ON profiles FOR INSERT
    WITH CHECK (auth.uid() = id);


-- ── Symptom checks policies ───────────────────────────────────
CREATE POLICY "Users can manage own symptom_checks"
    ON symptom_checks FOR ALL
    USING (auth.uid() = user_id);


-- ── Predictions policies ──────────────────────────────────────
CREATE POLICY "Users can view own predictions"
    ON predictions FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Service role can insert predictions"
    ON predictions FOR INSERT
    WITH CHECK (true);   -- backend uses service role key


-- ── RAG retrievals policies ───────────────────────────────────
-- Users can see their own retrieval logs (via prediction join)
CREATE POLICY "Users can view own rag_retrievals"
    ON rag_retrievals FOR SELECT
    USING (
        prediction_id IN (
            SELECT id FROM predictions WHERE user_id = auth.uid()
        )
    );

CREATE POLICY "Service role can insert rag_retrievals"
    ON rag_retrievals FOR INSERT
    WITH CHECK (true);


-- ── Lab reports policies ──────────────────────────────────────
CREATE POLICY "Users can manage own lab_reports"
    ON lab_reports FOR ALL
    USING (auth.uid() = user_id);


-- ── Recommendations policies ──────────────────────────────────
CREATE POLICY "Users can view own recommendations"
    ON recommendations FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Service role can insert recommendations"
    ON recommendations FOR INSERT
    WITH CHECK (true);


-- ── Health metrics policies ───────────────────────────────────
CREATE POLICY "Users can manage own health_metrics"
    ON health_metrics FOR ALL
    USING (auth.uid() = user_id);


-- ── Performance indexes ───────────────────────────────────────

-- Predictions: common query patterns
CREATE INDEX IF NOT EXISTS idx_predictions_user_id
    ON predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_user_created
    ON predictions(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_disease
    ON predictions(disease);

-- Symptom checks
CREATE INDEX IF NOT EXISTS idx_symptom_checks_user_id
    ON symptom_checks(user_id);

-- Lab reports
CREATE INDEX IF NOT EXISTS idx_lab_reports_user_id
    ON lab_reports(user_id);
CREATE INDEX IF NOT EXISTS idx_lab_reports_user_created
    ON lab_reports(user_id, created_at DESC);

-- Recommendations
CREATE INDEX IF NOT EXISTS idx_recommendations_prediction_id
    ON recommendations(prediction_id);
CREATE INDEX IF NOT EXISTS idx_recommendations_user_id
    ON recommendations(user_id);

-- Health metrics: time-series queries
CREATE INDEX IF NOT EXISTS idx_health_metrics_user_type_time
    ON health_metrics(user_id, metric_type, recorded_at DESC);

-- RAG retrievals
CREATE INDEX IF NOT EXISTS idx_rag_retrievals_prediction_id
    ON rag_retrievals(prediction_id);
