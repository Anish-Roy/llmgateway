-- What `model: "auto"` decided (app/routing/auto.py): the alias it picked, and the rule that
-- picked it ("default", "long_prompt", "keyword:prove"...), never text from the request. Both
-- NULL on requests for a named alias, and on auto requests refused before routing (a 429).
-- Inline CHECKs: ADD CONSTRAINT has no IF NOT EXISTS, and this file runs on every startup.
ALTER TABLE request_logs
    ADD COLUMN IF NOT EXISTS routed_alias text,
    ADD COLUMN IF NOT EXISTS route_reason text
        CHECK (route_reason ~ '^(default|long_answer|long_conversation|code_block|pasted_text|long_prompt|keyword:.+)$');
