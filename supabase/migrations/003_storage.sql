-- ============================================================
-- Migration 003: Supabase Storage bucket for lab report uploads
-- ============================================================

-- Create the lab-reports storage bucket
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'lab-reports',
    'lab-reports',
    false,                        -- private bucket
    10485760,                     -- 10 MB limit
    ARRAY[
        'application/pdf',
        'image/png',
        'image/jpeg',
        'image/jpg',
        'image/tiff'
    ]
)
ON CONFLICT (id) DO NOTHING;


-- ── Storage RLS policies ──────────────────────────────────────

-- Users can upload to their own folder: {user_id}/{filename}
CREATE POLICY "Users can upload own lab reports"
    ON storage.objects FOR INSERT
    WITH CHECK (
        bucket_id = 'lab-reports' AND
        auth.uid()::text = (storage.foldername(name))[1]
    );

-- Users can view their own uploads
CREATE POLICY "Users can view own lab reports"
    ON storage.objects FOR SELECT
    USING (
        bucket_id = 'lab-reports' AND
        auth.uid()::text = (storage.foldername(name))[1]
    );

-- Users can delete their own uploads
CREATE POLICY "Users can delete own lab reports"
    ON storage.objects FOR DELETE
    USING (
        bucket_id = 'lab-reports' AND
        auth.uid()::text = (storage.foldername(name))[1]
    );
