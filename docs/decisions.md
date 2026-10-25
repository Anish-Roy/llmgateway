# Design decisions

One entry per phase: what was chosen, and why.

## Phase 0: setup

- **uv + Python 3.12.** Fast installs, one lockfile. 3.12 rather than 3.13+ because ML
  libraries (sentence-transformers, torch) support it reliably.
- **Postgres + pgvector in one container** (`pgvector/pgvector:pg17`). One database holds API
  keys, request logs and cache vectors, so there's no separate vector store to run. Mapped
  to host port 5433 to avoid clashing with a local Postgres.
- **`app/usage/` instead of `app/logging/`.** A package named `logging` would shadow Python's
  standard `logging` module.
- **Settings via pydantic-settings, keys as `SecretStr`.** Config is validated at startup,
  and API keys don't leak into logs or tracebacks.
- **Query log built by a script, not written by hand as JSONL.** Near-duplicates and traps
  are declared next to their original prompt, so the links can't drift. The file is shuffled
  with a fixed seed, so a duplicate can come before or after its original. Evaluation code
  should treat each pair as unordered.

## Phase 1: non-streaming proxy, one provider

- **Own Pydantic models for the OpenAI format, not the `openai` package's types.** The gateway
  must not depend on one vendor's SDK at runtime. Requests use `extra="allow"`: fields we don't
  model (`top_p`, `stop`, `seed`, `tools`) are forwarded, not dropped, so a client's settings
  are never silently ignored. If the provider can't do one, the client gets its honest 400.
- **`Provider` is a `Protocol` with one method, `chat(request) -> ChatResponse`.** Adapters
  translate in and out of the OpenAI shape; everything above them only sees that shape.
- **`ProviderError` carries a status and a `retryable` flag** (timeout, 429, 5xx yes; 400 no).
  Phase 2's fallback just reads the flag.
- **All errors are OpenAI-shaped** (`{"error": {message, type, param, code}}`), including
  FastAPI's validation errors, which are a 422 `{"detail": [...]}` by default. OpenAI answers
  bad input with a 400, and the SDK maps status codes to exception classes
  (`BadRequestError`, `RateLimitError`...), so the shape and status both matter.
- **One shared `httpx.AsyncClient`, created in the app lifespan.** Connection pooling, and one
  place for the timeout. The provider is a FastAPI dependency so tests swap in the mock.
- **`stream=True` returns a 400 until Phase 3**, rather than silently answering non-streamed.
- **Contract tests use the real `openai` SDK** over an in-memory ASGI transport (`httpx2`, which
  openai 3.x is built on). No server, no network, and it proves SDK compatibility directly.
- **Groq default model `llama-3.1-8b-instant`**, checked against Groq's model list on
  2026-10-24 (Groq retires models often). For now the client's `model` is passed through;
  aliases come with the registry in Phase 2.
- **A provider 401/403 becomes a 502.** It means the gateway's own provider key is bad; passing
  the 401 through would tell the client their key is wrong.
- **No auth yet.** The SDK requires an `api_key`, which the gateway ignores until Phase 4.

## Phase 2: second provider and fallback

- **The model registry is a YAML file (`config/models.yaml`), checked with Pydantic at startup.**
  Models can change without code changes, and Phase 5's pricing table can sit next to it.
  `extra="forbid"` everywhere, so a typo in a provider option stops the app instead of being
  ignored.
- **Clients send an alias; targets are `provider/model`, split on the first slash only.**
  Provider model names can contain slashes. An unknown alias gets OpenAI's 404
  `model_not_found`, which the SDK raises as `NotFoundError`.
- **The response `model` is the model that actually answered, not the alias.** OpenAI does the
  same (`gpt-4o` → `gpt-4o-2024-08-06`). `x-gateway-provider` names the registry entry that
  served the request.
- **Providers are named instances, not just types.** `flaky: {type: mock, failure_rate: 1.0}`
  sits next to `mock`, which is how the fallback demo works without breaking a real provider.
- **A provider with no API key is skipped at startup, with a warning.** Its targets are dropped
  from every alias. An alias left with none returns a 503 (a deployment problem, not a client
  one), not a 404.
- **The router passes the concrete model with `request.model_copy(update=...)`,** so the
  `Provider` protocol didn't change and adapters never see aliases.
- **Two separate questions about a failure: "try another provider?" and "retry this one?"**
  `ProviderError.retryable` (timeouts, 429, 5xx) answers the first and
  `retry_same_provider` (500/502/503 only) the second. A 429 isn't retried on the same
  provider, because its quota won't refill in a few hundred ms. A timeout isn't either,
  because a second wait would use up the time the fallback needs. A rejected gateway key
  (our 502) is marked not worth retrying: the same key fails the same way.
- **A 400 from the first provider goes straight back; a 400 from a fallback doesn't.** The
  first provider rejecting the request means it's bad. A fallback rejecting it only means
  *that* provider can't do it (say, a parameter Gemini lacks), so the router moves on.
- **When everything fails, the client gets the latest *retryable* error.** If the primary
  gave a 503 and the backup a 400, the request is fine and the gateway is just having a bad
  moment, so the client should see "try again", not "your request is invalid". If the last
  error is a 429, the openai SDK backs off before retrying.
- **One deadline for the whole request (`REQUEST_DEADLINE_S`, 45 s), as well as the 30 s
  per-call timeout.** Otherwise two providers with two attempts each could take two minutes.
  The deadline also caps the backoff sleep. Clients add their own retries on top (the openai
  SDK's default `max_retries=2`), so a gateway that keeps retrying makes the wait worse.
- **Backoff is exponential with full jitter (0–250 ms for the single retry), and `sleep` is
  injected,** so tests check the retry policy without waiting.
- **`x-gateway-attempts` and `x-gateway-fallback` headers** next to `x-gateway-provider`.
  They make the fallback demo visible from `curl -i`, and Phase 5 logs the same values.
- **Gemini through its native `generateContent` REST API, on the shared httpx client.** No
  Google SDK and no Gemini OpenAI-compatible endpoint: translating between the two formats is
  what makes this a gateway. The docs now label `generateContent` "Legacy" next to the newer
  Interactions API, but it still works and has no shutdown date (checked 2026-10-25). Moving
  to the Interactions API would only change the adapter.
- **Models are chosen from what our keys can actually use, not from docs pages.**
  `scripts/check_models.py` compares the registry with each provider's model list.
  On 2026-10-25:
  - **Groq's Llama models weren't available to our free key.** Asking for one returned a 404
    `model_not_found`. The production chat models we could use were `openai/gpt-oss-20b` and
    `-120b`.
  - **Gemini 2.5 is limited to accounts that used it before.**
  - **Result:** `fast` = gpt-oss-20b → gemini-3.5-flash-lite, `strong` = gpt-oss-120b →
    gemini-3.8-flash.
- **`strong` puts Groq first, because live testing said so.** gemini-3.8-flash timed out on
  5 of 6 calls and gave a 503 "high demand" on the sixth, so with it first every `strong`
  request waited 30 s before falling back. Fallback hid the outage, but only by adding latency.
  A per-provider timeout or Phase 3's time-to-first-token fallback would cut that wait.
- **Groq provider options `reasoning_effort` and `include_reasoning`, applied only when the
  client didn't send them.** gpt-oss reasons before answering, and the reasoning counts
  against `max_tokens`. At Groq's default effort, a 50-token request came back with an empty
  answer (`finish_reason: "length"`). At `low`, the answer fits. `include_reasoning: false`
  leaves the `reasoning` text out of the reply, matching how the Gemini adapter drops
  thought parts.
- **Thinking tokens count as completion tokens.** Gemini bills `thoughtsTokenCount` as output,
  and they count against `maxOutputTokens`. Leaving them out would make Phase 5 costs too low.
  Thought parts (`thought: true`) are left out of the reply text. Each Gemini provider entry
  can set `thinking_level`. By default Flash-Lite thinks at "minimal" and 3.8 Flash at "medium",
  so a small `max_tokens` on `strong` can run out before any answer: that comes back as an
  empty reply with `finish_reason: "length"`.
- **The Gemini adapter maps fields explicitly and rejects the rest.** It maps `temperature`,
  `max_tokens`/`max_completion_tokens`, `top_p`, `stop`, `seed` and `n=1`, all confirmed live
  on 2026-10-25. Anything else gets a 400 naming the parameter: `tools`, `logprobs`, `n>1`,
  tool messages, and the presence/frequency penalties, which Gemini 3.x rejects with "Penalty
  is not enabled for this model". Rejecting them in the adapter matters: a raw upstream 400
  would block fallback, while the adapter's own provider-specific 400 lets Groq take it. System and developer messages, wherever they appear, are joined into
  `systemInstruction`. `assistant` becomes `model`, and back-to-back turns with the same role
  are merged, because Gemini expects them to alternate.
- **Upstream errors that mean *our* config is wrong become a 502, never a client 4xx.** A bad
  Gemini key comes back as a **400** `INVALID_ARGUMENT` with reason `API_KEY_INVALID`
  (checked with a made-up key). Passed on as-is, the client would be told their request is
  invalid and fallback would never run. A model the provider doesn't serve is the same kind of
  error: Gemini's 404, and Groq's 404 / `model_not_found` / `model_decommissioned`. All of them
  move to the next provider without retrying the same one.
- **A blocked prompt (no `candidates`) is a 200 with `finish_reason: "content_filter"`,** not a
  crash and not a fallback. Any 200 the adapter can't read becomes a retryable 502.
- **The Gemini key goes in the `x-goog-api-key` header, not `?key=`,** so it stays out of URLs
  and access logs.
- **An adapter's "this provider can't do that" 400 is `provider_specific`, so it falls back even
  from the first target.** `strong` starts on Gemini, which lacks `tools`. Without the flag, a
  `tools` request would get Gemini's 400 and never reach Groq, which supports it. A real
  upstream 400 (the request is bad) still goes straight back. If every provider says
  "can't", the client gets the 400.
- **`GET /v1/models` lists aliases, not vendor models,** and only the ones this deployment can
  serve, so `client.models.list()` shows what's worth sending.
- **The "done when" check, with real providers (2026-10-25):**
  - `fast` was served by Groq.
  - A Groq model the key can't use (404) fell back to Gemini in 2 attempts, without a retry.
  - A `tools` request with Gemini first fell back to Groq, which returned a proper `tool_calls`
    reply.
  - Gemini followed a system prompt.
  - The same fallbacks are also covered offline by `tests/test_provider_fallback.py`, which
    runs the real adapters over a faked transport.
