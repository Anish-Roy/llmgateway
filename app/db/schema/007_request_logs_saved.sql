-- What a cache hit saved: the cost of the provider call that first produced the answer, as
-- stored with the cache entry. Hits are billed $0 (cost_usd), and entries expire or get
-- replaced, so the saving is kept on the hit's own row for the dashboard.
-- NULL on every miss, and on hits logged before this column existed: what those saved is
-- unknown, not zero.
-- Inline CHECK: ADD CONSTRAINT has no IF NOT EXISTS, and this file runs on every startup.
ALTER TABLE request_logs
    ADD COLUMN IF NOT EXISTS saved_usd numeric(14, 10)
        CHECK (saved_usd IS NULL OR (saved_usd >= 0 AND cache_status <> 'miss'));
