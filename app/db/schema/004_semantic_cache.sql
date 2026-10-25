-- The semantic cache: answers found by meaning, not by the exact request. One row per answered
-- single-turn question, scoped to the API key that asked it. Rows are written after the
-- response is sent and read, on requests that opt in, after the exact cache misses.

-- docker/postgres/init.sql creates it too, but only on a fresh local volume. A hosted Postgres
-- (Neon, Supabase) never runs that file.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS semantic_cache (
    -- SHA-256 of key_id, context_hash and prompt: the same question in the same context from
    -- the same key replaces its row instead of adding another.
    entry_key    bytea       PRIMARY KEY CHECK (octet_length(entry_key) = 32),
    -- Only the key that stored an answer is served it. A semantic hit gives one question the
    -- answer to another, which may carry details from the first; across teams, that leaks.
    -- Keys are revoked, never deleted, so the reference always holds.
    key_id       bigint      NOT NULL REFERENCES api_keys (id),
    -- SHA-256 of everything in the request except the user message: the alias, the system
    -- prompt and every option the model sees. A lookup matches it exactly, so a similar
    -- question asked with another system prompt, max_tokens or response_format never hits.
    context_hash bytea       NOT NULL CHECK (octet_length(context_hash) = 32),
    -- The alias the client asked for. Already in context_hash; kept to see what's cached.
    alias        text        NOT NULL,
    -- The user message that was embedded (app/cache/embedder.py: embed_text). The numbers
    -- guard compares it with a new question's.
    prompt       text        NOT NULL,
    -- all-MiniLM-L6-v2, unit length, so cosine distance (<=>) is 1 - similarity.
    embedding    vector(384) NOT NULL,
    -- The answer as first served, usage included, in the chat-completions format: the same
    -- as response_cache, so a hit is served or replayed as SSE the same way.
    response     jsonb       NOT NULL,
    -- Who answered and what that call cost: what every hit on this row saves.
    provider     text        NOT NULL,
    model        text        NOT NULL,
    cost_usd     numeric(14, 10) NOT NULL CHECK (cost_usd >= 0),
    -- Set by the gateway's clock, like response_cache. A lookup ignores expired rows.
    created_at   timestamptz NOT NULL,
    expires_at   timestamptz NOT NULL,
    CHECK (expires_at > created_at)
);

-- Nearest neighbours by cosine distance. HNSW returns candidates first and filters by key,
-- context and expiry after, so a lookup sets hnsw.iterative_scan to keep searching until a
-- candidate passes (pgvector 0.8+). Default build settings (m 16, ef_construction 64).
CREATE INDEX IF NOT EXISTS semantic_cache_embedding
    ON semantic_cache USING hnsw (embedding vector_cosine_ops);
-- A key's entries in one context: few rows, so the planner can scan them exactly instead.
CREATE INDEX IF NOT EXISTS semantic_cache_key_context
    ON semantic_cache (key_id, context_hash);
-- For deleting expired rows.
CREATE INDEX IF NOT EXISTS semantic_cache_expires_at ON semantic_cache (expires_at);
