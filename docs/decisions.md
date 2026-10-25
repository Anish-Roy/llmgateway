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
