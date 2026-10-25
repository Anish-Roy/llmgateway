-- The exact-match cache: one answer per cache key (app/cache/key.py), shared by every API key.
-- Rows are written after the response is sent and read before any provider is called.
CREATE TABLE IF NOT EXISTS response_cache (
    -- SHA-256 of the canonical request: alias, messages and every option the model sees.
    cache_key  bytea       PRIMARY KEY CHECK (octet_length(cache_key) = 32),
    -- The alias the client asked for. Already in the key; kept to see what's cached.
    alias      text        NOT NULL,
    -- The answer as first served, in the chat-completions format, usage included. A streamed
    -- answer is stored in this same form, so either kind of request can be served from it.
    response   jsonb       NOT NULL,
    -- Who answered and what that call cost: what every hit on this row saves.
    provider   text        NOT NULL,
    model      text        NOT NULL,
    cost_usd   numeric(14, 10) NOT NULL CHECK (cost_usd >= 0),
    -- Set by the gateway's clock, like request_logs.created_at. A lookup ignores expired rows;
    -- a new answer for the same key replaces the row.
    created_at timestamptz NOT NULL,
    expires_at timestamptz NOT NULL,
    CHECK (expires_at > created_at)
);

-- For deleting expired rows.
CREATE INDEX IF NOT EXISTS response_cache_expires_at ON response_cache (expires_at);
