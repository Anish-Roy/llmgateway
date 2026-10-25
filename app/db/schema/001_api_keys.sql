-- Client API keys. Only a SHA-256 of each key is stored; the plain key is shown once, when
-- it's created. Every statement is idempotent: scripts/migrate.py runs all files every time.
CREATE TABLE IF NOT EXISTS api_keys (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name        text        NOT NULL,
    -- Hex SHA-256 of the full key. Unique, so the auth lookup is one index probe.
    key_hash    text        NOT NULL UNIQUE CHECK (key_hash ~ '^[0-9a-f]{64}$'),
    -- The first characters of the key ("gw-AbC1..."), so `list` and `revoke` can name a key
    -- without its secret.
    key_prefix  text        NOT NULL,
    rpm_limit   integer     NOT NULL CHECK (rpm_limit > 0),
    tpm_limit   integer     NOT NULL CHECK (tpm_limit > 0),
    active      boolean     NOT NULL DEFAULT true,
    created_at  timestamptz NOT NULL DEFAULT now()
);
