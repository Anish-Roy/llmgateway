-- One row per chat-completions request that got past auth: replies, streams, fallbacks,
-- failures, and the gateway's own 429s and 404s. Written after the response is sent.
CREATE TABLE IF NOT EXISTS request_logs (
    id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    -- Keys are revoked, never deleted, so every row keeps pointing at a real key.
    key_id            bigint      NOT NULL REFERENCES api_keys (id),
    created_at        timestamptz NOT NULL DEFAULT now(),
    -- The alias the client sent ("fast"), and the provider entry and model that answered.
    -- Both empty when no provider answered (every one failed, or the gateway refused first).
    requested_model   text        NOT NULL,
    provider          text,
    model             text,
    stream            boolean     NOT NULL,
    -- What the client got. A stream that fails partway is still a 200: `outcome` says so.
    status_code       integer     NOT NULL CHECK (status_code BETWEEN 100 AND 599),
    -- completed, provider_error, gateway_error and client_disconnect are the stream outcomes
    -- (StreamRecord.outcome); rejected is a request refused before any provider call (429, 404).
    outcome           text        NOT NULL CHECK (outcome IN
                          ('completed', 'provider_error', 'gateway_error', 'client_disconnect',
                           'rejected')),
    -- The provider's counts, or an estimate when it sent none (a cut stream): a lower bound.
    prompt_tokens     integer     NOT NULL DEFAULT 0 CHECK (prompt_tokens >= 0),
    completion_tokens integer     NOT NULL DEFAULT 0 CHECK (completion_tokens >= 0),
    usage_estimated   boolean     NOT NULL DEFAULT false,
    -- Until the reply was ready, or the stream closed. Time to first token: streams only.
    latency_ms        integer     NOT NULL CHECK (latency_ms >= 0),
    ttft_ms           integer     CHECK (ttft_ms >= 0),
    -- Exact decimal, not float: sums over thousands of tiny costs must not drift. 10 decimal
    -- places hold one token at the cheapest list prices ($0.02 per 1M is 2e-8).
    cost_usd          numeric(14, 10) NOT NULL DEFAULT 0 CHECK (cost_usd >= 0),
    -- Every request is a miss until the caches exist (Phases 6 and 7).
    cache_status      text        NOT NULL DEFAULT 'miss'
                          CHECK (cache_status IN ('miss', 'exact', 'semantic')),
    -- Provider calls made, retries included; 0 when the gateway refused the request first.
    attempts          integer     NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    fallback          boolean     NOT NULL DEFAULT false,
    CHECK (num_nulls(provider, model) IN (0, 2))
);

-- The dashboard's query: one key's requests over a time range.
CREATE INDEX IF NOT EXISTS request_logs_key_id_created_at
    ON request_logs (key_id, created_at);
