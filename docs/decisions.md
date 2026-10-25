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
  A shorter per-provider timeout, or for streams a separate time-to-first-token timeout,
  would cut that wait. Phase 3 doesn't add one: a stream's first read still waits up to the
  30 s HTTP timeout before falling back.
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

## Phase 3: streaming

- **Fallback stops at the first token the client would see, not at the first byte from the
  provider.** The router opens the stream inside the same retry/fallback/deadline loop as
  `chat()` (the policy is shared, not copied) and reads ahead until a chunk has text, a tool call
  or a `finish_reason`. Groq and OpenAI send an empty role-only chunk first. Committing on that
  chunk would give up fallback for exactly the failures it's for, so leading empty chunks are
  held back. A 200 stream that ends without a reply counts as a retryable 502.
- **The provider is chosen before the response starts.** The route awaits the router up to the
  commit point, then returns the `StreamingResponse`. If every provider fails before the first
  token, the client gets a normal JSON error with the right status (a 429 is still
  `RateLimitError` in the SDK), not a 200 that breaks. The `x-gateway-*` headers still work.
- **After the first token, a failure is an error event and the stream ends without `[DONE]`.**
  The event is `data: {"error": {...}}`, which the openai SDK raises as `APIError` (checked in
  its `_streaming.py`). No fallback, because a second answer glued onto half of the first would
  be worse than an honest error.
- **The deadline covers the wait for the first token, not the whole answer.** A long answer can
  legitimately take more than 45 s. After the commit, a stalled provider is caught by the HTTP
  client's read timeout between chunks, and that becomes a mid-stream error.
- **Every upstream stream is closed, however the request ends.** That covers a failure while
  reading ahead (including the deadline cancelling it), a mid-stream error, and a client that
  disconnects. Starlette 1.9 doesn't close the body iterator on a disconnect: under ASGI 2.4 it
  raises `ClientDisconnect` and leaves the generator suspended until garbage collection. The
  SSE response therefore closes the provider stream in a `finally` around the whole response.
  A test runs the app with a `send` that fails after the first chunk, and it fails without
  that `finally`.
- **The usage-only chunk (`choices: []`) is sent only when the client asks** with
  `stream_options.include_usage`, as OpenAI does. Code that reads `chunk.choices[0]` is common
  and would crash on it. Providers are still expected to report usage, because Phase 5 logs it.
- **`Cache-Control: no-cache` and `X-Accel-Buffering: no`** so proxies on the hosting platforms
  pass each chunk on as it comes instead of buffering the answer.
- **The mock streams like OpenAI** (a role chunk, a word per chunk, a finish chunk, a usage
  chunk). Its `fail_after_chunks` option separates the two cases: 1 fails after the role chunk
  (still replaceable), 2 or more fails mid-answer. `demo-midstream` uses it.
- **Groq's stream format was checked live (2026-10-25) before writing the adapter.**
  - The first chunk is `role: assistant` with `content: ""`, as assumed above.
  - Errors come before the stream starts, as normal JSON (404 `model_not_found`, 400 for a
    bad `temperature`), so the existing error mapping is reused after reading the body.
  - Without `include_usage`, usage is only inside `x_groq` on the finish chunk. With it, Groq
    also sends OpenAI's `choices: []` usage chunk.
  - `include_reasoning: true` streams `delta.reasoning` (plus a `channel` field) before the
    content.
- **The Groq adapter always sends `stream_options.include_usage: true`,** merged into whatever
  `stream_options` the client sent. The SSE layer strips the usage chunk again for clients that
  didn't ask. Phase 5 needs usage on every stream, and reading the standard chunk is simpler
  than digging it out of `x_groq`.
- **Vendor fields on chunks are dropped, as in non-streamed replies.** `x_groq` (request id,
  timings, a second copy of usage) and `system_fingerprint` don't reach the client. Fields
  inside `delta` (`tool_calls`, `reasoning`) are kept.
- **Reasoning text counts as output for the commit point.** It's only streamed when the client
  turned `include_reasoning` on, and then it's what they're waiting for. Holding it back until
  the content starts would add the whole reasoning time to their time to first token, just to
  keep the fallback window open a little longer.
- **A Groq stream that closes without `[DONE]` is an error (502), not a normal end.** Otherwise a
  dropped connection would look like a complete answer. An `{"error": ...}` event inside the
  stream is a 502 too: the request was accepted, so the provider failed and another one may do
  better. A timeout between chunks (the HTTP client's read timeout) is a 504.
- **The SSE reader (`app/providers/sse.py`) is shared** by the adapters. It handles data split over
  several lines and ignores comments, which are both valid SSE even though Groq doesn't send them.
- **The live "done when" for Groq (2026-10-25):**
  - The openai SDK with `stream=True` worked on `fast` and `strong`, with the first token in
    0.5–0.65 s.
  - `include_usage` returned the usage chunk.
  - A client that disconnected after 5 chunks of a long story left no errors in the server log.
- **Gemini's stream format was checked live (2026-10-25) before writing the adapter.**
  - `streamGenerateContent` needs `?alt=sse`; without it the reply is one JSON array sent bit
    by bit.
  - Each event is a whole `GenerateContentResponse` holding the next piece of text. The last
    one has the `finishReason`, often with an empty part that only carries a
    `thoughtSignature`. There is no `[DONE]`.
  - `usageMetadata` is on every event and holds running totals (completion tokens went 2 → 6
    over a stream), so the adapter keeps the last one rather than adding them up.
    `thoughtsTokenCount` grows the same way and counts as output, as in non-streamed replies.
  - Errors (503 "high demand", 404, a 400 for a bad temperature) come before the stream as a
    normal JSON body, even though the content-type says `text/event-stream`.
  - Thought parts (`thought: true`) only appear with `includeThoughts`, which the adapter
    never sends. They're filtered anyway.
- **A Gemini stream is complete once a `finishReason` arrives.** If the connection closes before
  that, it's a 502 "ended early", the same rule as Groq's missing `[DONE]`.
- **The translator puts `role: "assistant"` on the first chunk it sends,** as OpenAI and Groq
  do. Gemini never sends a role.
- **Events with nothing for the client send no chunk.** That covers thought-only events, the
  empty part with a `thoughtSignature`, and usage-only events. Thinking time therefore can't
  commit the router, and fallback stays possible until real text arrives. Usage goes out as
  its own `choices: []` chunk after the finish, which is the shape the SSE layer filters on.
- **A blocked prompt (no candidates, a `promptFeedback.blockReason`) is a finish chunk with
  `content_filter`.** It's a real answer, as in Phase 2, so it commits rather than falling back.
- **`to_gemini` accepts `stream_options` instead of rejecting it.** It was on the "unsupported
  extras" path, so every client sending `include_usage` would have skipped Gemini with a
  provider-specific 400.
- **Both adapters are now built the same way:** each has its own `_check_key`, `_headers`
  and `_transport_errors()` context manager. These are copies rather than shared code, because
  the messages name the provider. The context manager wraps the request and, for
  streams, every read, so a timeout between events is a 504 there too.
- **The live "done when" for Gemini (2026-10-25)** used a scratch registry (`MODELS_FILE`),
  because Gemini is second in both aliases and never streams while Groq is up.
  - With Gemini first, the openai SDK streamed with the first token in about 1.9 s.
  - With a Groq model the key can't use in front, the request fell back to Gemini (2
    attempts, `x-gateway-fallback: true`), and `include_usage` returned the usage chunk.
  - A client that disconnected 1 s into a 1500-word story, mid-generation, left a clean
    server log.
  - The same fallback is covered offline in `tests/test_gemini_stream.py`: Groq returns a 503
    before the first token, then Gemini streams through the real adapters and the SDK.
- **Every stream gets a `StreamRecord`** (`RoutedStream.record`), complete once the stream
  ends or is closed. Phase 5 logs it and Phase 6 can cache its text. It holds:
  - the outcome: `completed`, `provider_error`, `gateway_error` or `client_disconnect`;
  - the error status, the finish reason and the duration;
  - the answer as streamed, even when the stream was cut;
  - the usage, and whether it was estimated;
  - the time to first token, measured from the router getting the request to the first chunk
    with output going to the client. That's what the client waits for, so it includes retries,
    fallback and the read-ahead, not just the provider's speed.
  A one-line summary is logged at INFO when the stream closes.
- **Provider usage always wins over an estimate.** Gemini's running totals ride along on every
  chunk inside the gateway, so a Gemini stream cut halfway still has Gemini's own count. The SSE
  layer strips `usage` from every chunk that has choices, so clients only see usage on the
  final chunk, as with OpenAI. Groq and the mock report usage only at the end, so a cut Groq
  stream falls back to an estimate.
- **Estimates use tiktoken `o200k_base` and are flagged `usage_estimated`.** The encoding is
  close for gpt-oss and only approximate for Gemini. It counts visible text only, so with
  `include_reasoning: false` (or Gemini thinking) the billed reasoning tokens are missing.
  Treat the estimate as a lower bound, not an exact cost.
- **tiktoken loads in a background thread at startup, and counts are chars/4 until it's
  ready.** The first `get_encoding` downloads the encoding file, which took 18.5 s on
  2026-10-25. Doing that on the request path would block the event loop. The file is cached
  in the temp directory, so a fresh container downloads it again; Phase 12's Dockerfile should
  fetch it at build time. If the download fails, counts stay at chars/4. Tests never download
  it.
- **`LOG_LEVEL` is now applied** (`logging.basicConfig` in `create_app`). Before, nothing read
  it, so the app's INFO lines never appeared and warnings only showed through Python's
  last-resort handler. httpx's per-request INFO lines are turned down to WARNING. Structured
  JSON logs are still Phase 12.
- **Phase 3 "done when", checked live (2026-10-25):**
  - The openai SDK with `stream=True` works against Groq (`fast`, `strong`) and Gemini.
  - A mid-stream failure (`demo-midstream`) reaches the client as an `APIError` after the
    partial text. It's logged as `provider_error (503)` with estimated usage.
  - A client that disconnected 0.5 s into a long `strong` answer was logged as
    `client_disconnect`, which shows the upstream close runs on a real server.


## Phase 4: API keys and rate limits

- **psycopg 3 (async) with `psycopg_pool`, not asyncpg.** The same driver works async in the
  gateway and sync in `scripts/` and the Phase 10 Streamlit dashboard, and pgvector supports it
  for Phase 7. asyncpg is async-only, so the scripts would need a second driver.
- **One pool, opened in the lifespan next to the httpx client.** Startup waits up to 10 s for a
  first connection, so a stopped Postgres or a wrong `DATABASE_URL` stops the app right away,
  like a bad registry file, instead of failing the first request. The gateway now needs
  Postgres to start.
- **The pool checks each connection before handing it out.** Hosted Postgres (Neon, Supabase)
  closes idle connections; without the check, the first request after a quiet spell would get
  a dead one. It costs one round trip per checkout, and the auth cache (step 3) keeps checkouts
  rare.
- **Tables come from SQL files run by `scripts/migrate.py`, not from `docker/postgres/init.sql`.**
  The init file runs only when the volume is first created, so an existing dev database and a
  hosted one would never get the table. No migration tool either: every file is idempotent
  (`CREATE TABLE IF NOT EXISTS`) and all of them run in name order, in one transaction, every
  time. An advisory lock makes two runs at once take turns. Enough for a handful of tables.
- **The database enforces what it can:** the key hash is unique and must be 64 lowercase hex
  characters (one key, one spelling), and limits must be positive. A bad row from the CLI fails
  at insert, not at the first request.
- **Postgres tests run in a throwaway schema** (`search_path` set on the connection) and skip
  when Postgres isn't reachable. Tests that start the app get no pool, the same way they skip
  the tiktoken download, so the rest of the suite needs no database.
- **Keys are `gw-` plus `secrets.token_urlsafe(32)`, stored as a plain SHA-256.** bcrypt and
  argon2 exist because people pick guessable passwords; a 256-bit random key can't be guessed,
  so a slow hash would only add tens of milliseconds to every request. And their per-row salt
  would rule out finding a key by its hash with one index lookup. The `gw-` prefix makes a
  leaked key easy to recognise (secret scanners match on prefixes like this).
- **The first 10 characters are stored as `key_prefix`.** `list` and `revoke` can then show
  which key is which without the secret. 7 random characters (~42 bits) tell keys apart and
  leave the other 36 characters unguessable.
- **The gateway's `KeyStore` is read-only: one method, `get_by_hash`.** Keys are created and
  revoked only by `scripts/create_key.py`, which connects to the database directly. No admin
  HTTP endpoint means no admin secret to protect and nothing for an attacker to call. A
  Postgres store and an in-memory one (for tests) implement the protocol, as with providers.
- **`get_by_hash` returns revoked keys too, marked inactive.** The auth step can then tell
  "revoked" from "never existed" in its logs. Both still get the same 401.
- **Revoking sets `active = false` instead of deleting the row,** so Phase 5's request logs
  keep pointing at a real key, and revoking twice is harmless.
- **Auth is a router-level dependency on `/v1`, so `/v1/models` needs a key too,** as on OpenAI.
  `/health` stays open for the hosting platform's checks. The key's record goes on
  `request.state` for the rate limiter and Phase 5's logs. A request with no key and a bad body
  gets the 401, not a 400 about the body.
- **Key lookups are cached in memory for 30 s (`KEY_CACHE_TTL_S`), unknown keys included.** A
  busy client costs one query per 30 s instead of one per request, and a client retrying a bad
  key doesn't reach the database at all. The price is written down rather than engineered
  away: the CLI is a separate process and can't clear the gateway's cache, so a revoke, or a
  new limit, takes effect within one TTL. Checked live: a revoked key was still accepted right
  after `revoke` and got a 401 once the TTL passed.
- **The cache is bounded (10,000 entries, oldest dropped first).** Without a bound, a stream of
  made-up keys would grow it forever. Dropping a real key's entry only costs a lookup. Strings
  that can't be keys (wrong prefix or length, someone's OpenAI key) are rejected before any
  lookup or cache entry.
- **No request coalescing.** If many requests with the same uncached key arrive at once, each
  queries the database. It happens once per key per TTL, and the pool caps the connections.
- **Every key problem is the same 401 that OpenAI sends** (`invalid_request_error`, code
  `invalid_api_key`, plus `WWW-Authenticate: Bearer`), so the SDK raises `AuthenticationError`.
  The messages differ (no key, not Bearer, incorrect, revoked) because each needs a different
  fix. Telling the holder of a revoked key that it's revoked gives nothing away: they already
  have the key. Messages show at most the first 10 characters of what was sent.
- **A database failure during the lookup is a 503, not a 401.** The key may be fine, and a 401
  would tell a client to stop using it. The failure isn't cached, so the next request tries
  again. Keys already in the cache keep working while the database is down.
- **Tests use the real authenticator over an in-memory store** (`tests/auth_helpers.authorize`),
  overriding `get_authenticator` the way `get_model_router` is overridden. One test runs the
  real startup against Postgres with keys made by the CLI's functions.
- **Rate limits are two token buckets per key, in memory** (`app/auth/rate_limit.py`): one of
  `rpm_limit` requests, one of `tpm_limit` tokens, each refilling to full over a minute,
  continuously. A full bucket allows a whole minute's worth at once, which suits chat clients
  that send a few requests together and then go quiet. A fixed window would allow twice the
  limit across a window boundary. The buckets live in this process, so the gateway runs with one
  worker, and a restart gives every key a full bucket. Sharing them between processes would
  need Redis, which is out of scope here.
- **A request's tokens are reserved before the call and settled after it.** The reply's length
  isn't known up front, so admission reserves the prompt (estimated with tiktoken, plus 4 per
  message) and the most the reply may use: `max_completion_tokens`, else `max_tokens`, else
  1024, times `n`. When the request ends, `Reservation.settle(actual)` returns what wasn't used
  or charges what was over. A request that failed before any provider answered settles at 0:
  its tokens come back, but its request slot doesn't, so hammering a failing alias is still
  limited.
- **Settling only counts once.** A stream can end several ways (completed, cut, client gone,
  failed before the first token), and step 5 may reach `settle` from more than one of them.
  Making the second call a no-op is simpler than proving each path calls it exactly once.
- **Admission is all or nothing.** Both buckets are checked before either is taken from, so a
  rejected request costs nothing. Otherwise a client retrying on 429 would drain its own bucket
  and never get back in. Nothing in `acquire` awaits, so on one event loop it can't interleave:
  no lock.
- **A reservation bigger than the whole TPM limit is cut to the limit.** Otherwise a request
  with a large `max_tokens` on a small key could never be admitted, and every retry would get a
  429. Cut down, it's admitted once the bucket is full, and settling charges the real usage.
- **Token debt stops at minus one minute's worth.** Settling a reply longer than reserved can
  take the bucket below zero. With no floor, one 2,000-token reply on a 500-TPM demo key would
  lock that key out for 4 minutes. With the floor:
  - no request ever waits more than 2 minutes;
  - a key can never go more than 2x over its TPM in any window.
  The floor at -1 minute also matches the openai SDK (checked in `_base_client.py`). It reads
  `retry-after-ms`, then `retry-after`, waits that long before retrying, and doesn't retry at
  all past 120 s. So a 429 from the gateway is always a delay the SDK will wait out.
- **Waiting exactly the `Retry-After` always works.** The wait is `(needed - level) / rate`, and
  the comparison has a 1e-9 slack, so floating-point rounding can't turn an exact wait into a
  second 429. A test advances the fake clock by 99% of the wait (still limited) and then the
  rest (admitted).
- **A changed limit applies at the key's next request** (once the auth cache has the new
  value). The bucket refills at the old rate up to that moment, then switches. Lowering a limit
  clamps the level to it. Raising one gives no free burst: the bucket refills at the new rate.
- **`estimate_tokens` ignores a `max_completion_tokens` or `n` that isn't a positive integer.**
  The gateway doesn't validate them; the provider does. Crashing on `"n": "two"` would turn the
  provider's honest 400 into a 500. `True` isn't 1 here, even though `bool` is an `int` in Python.
- **The order is auth → body validation → rate limit → router.** A request with a bad body gets
  its 400 before anything is reserved, so it costs the key nothing. Rate limiting runs in the
  route rather than as a dependency, because the estimate needs the parsed body. `/v1/models`
  isn't limited: it calls no provider, and OpenAI doesn't count it either.
- **Every way a request ends settles its reservation:**
  - a reply: its reported usage, or an estimate if the provider reported none;
  - every provider failing (including before a stream's first token): 0;
  - a stream: when it's closed, with `StreamRecord.usage`. That's the provider's count, or the
    estimate for a stream that was cut short or that the client left. `sse_response` got an
    `on_finish` hook that runs after the upstream stream is closed, inside the `finally` that
    already runs however the response ends.
  A test covers each path with a frozen clock and checks the bucket level exactly.
- **The headers are OpenAI's:** `x-ratelimit-{limit,remaining,reset}-{requests,tokens}`. Names
  and the reset format (`"1s"`, `"6m0s"`) were checked against OpenAI's rate-limit guide on
  2026-10-25. Reset is the time until the bucket is full again, rounded up to the millisecond.
  The headers come on replies and streams and on the gateway's own 429s. They are reported at
  admission, so the remaining tokens still include the reservation.
- **A request that fails after admission gets no rate-limit headers.** Its error comes from
  the shared error handlers, which don't know about the reservation. Clients that need them
  read them from the next reply or 429.
- **The gateway's 429 copies OpenAI's own: `type` is `"requests"` or `"tokens"`, `code` is
  `rate_limit_exceeded`.** That tells it apart from a provider's 429 passed through (type
  `rate_limit_exceeded`, no `x-ratelimit-*` headers). It names which limit, the key's prefix and
  the wait.
- **It sends `retry-after-ms` as well as `Retry-After`.** The openai SDK reads `retry-after-ms`
  first, so it waits the exact time instead of the next whole second. `Retry-After` is rounded
  up, so clients that only read it are never early. A test lets the SDK (`max_retries=2`) hit a
  real 429 and checks that it waited and then got in.
- **The default reservation for a request without `max_tokens` is configurable**
  (`RATE_LIMIT_DEFAULT_COMPLETION_TOKENS`, 1024). Checked live: a stream with no `max_tokens`
  held about 1,030 tokens while it ran, and the next reply showed them back once it was settled.
- **Phase 4 "done when", checked live (2026-10-25)** on a real server and Postgres, with a key
  made by the CLI (3 RPM, 20,000 TPM) and real Groq replies:
  - The smoke script and an SDK stream with `include_usage` both worked on `fast`.
  - No key and a wrong key got a 401 (`invalid_api_key`), which the SDK raises as
    `AuthenticationError`.
  - The fourth request in the minute got a 429 (`type: requests`) with `retry-after: 18`.
  - The server log had no warnings.
  - The live run doesn't show settling: at 20,000 TPM the bucket refills (333 tokens a second)
    between requests. The tests check settling exactly, with a frozen clock. The step 5 live
    check showed it with a smaller key.
  - Offline, `tests/test_auth.py` and `tests/test_rate_limit_http.py` cover the 401s and 429s
    through the openai SDK.
