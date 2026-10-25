-- Which kinds of PII the gateway masked in a request (app/guardrails/pii.py), sorted, never
-- the values. Empty when there was none, or the request was refused before masking (a 429).
-- Inline CHECK: ADD CONSTRAINT has no IF NOT EXISTS, and this file runs on every startup.
ALTER TABLE request_logs
    ADD COLUMN IF NOT EXISTS pii_types text[] NOT NULL DEFAULT '{}'
        CHECK (pii_types <@ ARRAY['aadhaar', 'card', 'email', 'phone']);
