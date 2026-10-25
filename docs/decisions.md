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
  lock that key out for 4 minutes. With the floor, no request ever waits more than 2 minutes:
  one minute to pay off the debt and one to refill a full reservation.
  The floor at -1 minute also matches the openai SDK (checked in `_base_client.py`). It reads
  `retry-after-ms`, then `retry-after`, waits that long before retrying, and doesn't retry at
  all past 120 s. So a 429 from the gateway is always a delay the SDK will wait out.
- **The floor bounds the wait, not the overuse.** (Corrected: this entry first said a key could
  never go more than 2x over its TPM, which is wrong.) Debt below one minute's worth is
  forgiven, so a key whose replies keep running past their reservation can use more than its
  TPM for as long as that goes on. For example, a 500-TPM key, no `max_tokens`, 8,000-token
  replies: each request is admitted with 500 reserved, settles to the -500 floor, and waits
  2 minutes. That's 8,000 tokens per 2 minutes, 8x the limit. A reply only runs past its
  reservation when the client sets no `max_tokens` (the default reserves 1024) or the prompt
  estimate is low, so with `max_tokens` set the limit holds, apart from estimation error.
  Separately, any token bucket lets a key use up to 2x its limit in the first minute after a
  quiet spell: the full bucket at once, then a minute's refill. If the overuse matters,
  raising `RATE_LIMIT_DEFAULT_COMPLETION_TOKENS`, or carrying debt over instead of forgiving it,
  would close it. Carrying debt over brings back the long lockouts.
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

## Phase 5: usage logging and cost

- **Which requests get a row: every chat-completions request that gets past auth.** That's
  replies, streams, fallbacks, every provider failing, and the gateway's own 429s and 404s.
  A 401 has no key to charge. A body the gateway can't parse is rejected before the route
  runs and costs nothing. `/v1/models` calls no provider.
- **`request_logs` stores the status code and an `outcome`.** A stream that fails partway was
  already sent as a 200, so the status alone would count it as a success. The outcomes are
  `StreamRecord`'s four, plus `rejected` for a request refused before any provider call.
- **`cost_usd` is `numeric(14, 10)`, not a float.** The dashboard sums thousands of tiny costs,
  and a float sum would drift. 10 decimal places hold one token at $0.02 per 1M (2e-8). A test
  adds that cost a thousand times and checks the sum exactly.
- **`cache_status` and `attempts` have defaults (`miss`, 0),** so Phases 6 and 7 only fill in
  `cache_status`, and a refused request needs no provider columns. `provider` and `model` are
  both set or both empty, which a check enforces.
- **One index, on `(key_id, created_at)`:** the dashboard's query is one key over a time range.
  More can come with Phase 10, once its queries exist.
- **Prices sit in `config/models.yaml`, keyed by `provider/model` like alias targets.** A price
  belongs to a model on one provider (gpt-oss-20b on Groq), and the alias lists already name
  them that way. The model actually served is what gets charged, so a fallback is billed at the
  fallback model's price.
- **The gateway won't start if a target has no price.** A missing price would quietly log every
  call to that model as free. Mock targets are the exception: free unless listed.
- **List prices, checked on 2026-10-25** (Groq's model page, Google's Gemini pricing page, paid
  tier, text). Free-tier calls are logged at list price too, so the costs mean something.
  gemini-3.8-flash is at an introductory price until 2026-12-31 and doubles on 2027-01-01. The
  config has one price per model, not a schedule: the cost is saved with each row, so updating
  the price on that date is enough, and older rows keep what they cost.
- **Prices are `Decimal`.** Pydantic turns the YAML float `0.075` into exactly `Decimal("0.075")`
  (it goes through the float's shortest string form), so the config stays plain numbers.
- **Rows are written by a fire-and-forget task, not a FastAPI `BackgroundTask`** (the plan
  said `BackgroundTask`). A background task is attached to the route's response, so it never
  runs when an error handler builds the response instead (a provider failure, a 429, a 404).
  On a stream, `StreamingResponse` runs it before our `on_close`, when the stream's record isn't
  complete, and a client disconnect can skip it. So `LogWriter.submit` starts a task and
  returns: the client never waits for the insert. Step 4 calls it from the route and from the
  stream's `on_finish`.
- **A failed insert only logs a warning.** By then the response is sent; losing one row is
  better than an error nobody can act on. Nothing retries it.
- **`LogWriter` keeps each task until it's done, and the lifespan drains them before the pool
  closes.** asyncio holds only weak references to tasks, so an unreferenced write could be
  garbage-collected mid-insert. Without the drain, the last requests before a shutdown would
  lose their rows.
- **`created_at` is when the request arrived, set by the gateway**, not the database's `now()`
  at insert time, which for a long stream would be minutes later.
- **Costs are rounded half up to 10 places in Python,** the column's scale, so a row reads back
  exactly as written and in-memory tests see what Postgres would store.
- **`UsageLog` is a Protocol with a Postgres and an in-memory store**, like `KeyStore`. It's
  write-only: the Phase 10 dashboard will read the table itself.
- **The route reports exactly one ending per request to a `Recording`:** `replied`, `streamed`
  (from the stream's `on_finish`, after the stream is closed, next to settling) or `failed`.
  Reporting twice is a no-op, as with settling. Building the row is wrapped too: a missing
  price or a bug there loses that row with a warning and never touches the response.
- **The outcome of a failed request is read from the error.** A 429 from the gateway, a 404 for
  an unknown model, and a 503 for an alias with no configured provider are `rejected` (no
  provider was called). A `ProviderError` after one or more calls is `provider_error`. Any other
  exception is a 500 `gateway_error`. A cancelled request (the client left during a plain
  reply) is `client_disconnect` with status 499, nginx's code for it, since no response went out.
- **`ProviderError.attempts`: the router writes its call count on the error it gives up with,**
  so a failed request's row says how many provider calls it made. Changing the error type or
  returning a tuple would touch every error handler; one attribute doesn't.
- **A stream's row has status 200, even if it failed partway.** That's what the client got.
  `outcome` holds the failure, and the tokens streamed up to then are logged as an estimate.
- **Apps that never start keep their rows in memory, free.** `create_app` builds an in-memory
  recorder, as it builds the rate limiter, and the lifespan replaces it with the Postgres log and
  the registry's prices. Most tests never run the lifespan; the ones that do get an in-memory
  log from `conftest`, which already gives them no pool.
- **Phase 5 "done when", checked live (2026-10-25)** on a real server and Postgres, with keys
  made by the CLI and a scratch registry adding a Gemini-only alias (`lite`). Nine requests
  made nine rows, and every row's tokens matched what the client was sent:
  - Groq replies on `fast` and `strong`, a `fast` stream (TTFT 509 ms of 543 ms), and Gemini
    Flash-Lite. Each `cost_usd` matched the cost recomputed in SQL from the listed prices, to
    the last digit. For example, 81 + 44 tokens on gpt-oss-20b cost
    (81 x $0.075 + 44 x $0.30) / 1M = $0.000019275.
  - `demo-fallback`: answered by mock, 3 attempts, `fallback` true, $0.
  - `demo-midstream`: status 200, `provider_error`, 4 completion tokens marked as estimated.
  - An unknown model (404) and the second request on a 1-RPM key (429): `rejected`,
    0 attempts, $0.
  - The server log had only the warnings the demos are meant to cause, and no failed writes.

## Phase 6: exact-match cache

- **One cache for every API key.** An exact hit is the same answer to the same request, so
  sharing leaks nothing a key couldn't get by sending that request itself, and it raises the
  hit rate. The semantic cache (Phase 7) answers *similar* prompts, which is riskier to share
  across teams: that gets its own decision.
- **The key hashes the whole forwarded request, not only alias + messages + temperature +
  max_tokens** (the plan's list). `ChatRequest` forwards fields it doesn't model (tools,
  response_format, stop, seed, top_p, a message's name), and each can change the answer: with
  the plan's four, a request with tools could get the answer to one without. Only `stream`,
  `stream_options` and `user` are left out, since the model never sees them. So a streamed and
  a plain request for the same thing share an entry.
- **Canonical JSON, then SHA-256.** Sorted keys and no spaces, so key order in a client's JSON
  doesn't matter; a null field counts as a missing one; `temperature` is a float field, so 0
  and 0.0 match. Other numbers aren't normalised (`top_p: 1` and `1.0` are different keys):
  that costs a miss, never a wrong answer. A version prefix in the hashed text lets a future
  change to the key retire old entries instead of mismatching them.
- **Only requests with a temperature at or below `CACHE_MAX_TEMPERATURE` (default 0.2) are
  cached.** Without a temperature the provider's default applies (1.0 on most), so answers are
  meant to vary. 0.2 rather than 0: low temperatures are what clients set when they want
  stable answers, and one stored sample is a fair answer to them. Requests asking for several
  choices (`n` > 1) aren't cached either.
- **`x-gateway-cache: off` (any case) skips the cache both ways:** no lookup, no store. Any
  other value keeps it on, so a typo never silently turns caching off.
- **Answers live in Postgres (`response_cache`), not Redis.** The plan allowed either. A
  primary-key lookup on the local container takes about a millisecond, well inside the
  "under ~10 ms" goal, and one fewer service keeps deploys simple. The key is the raw 32-byte
  digest (`bytea`, length checked), not hex: half the size, and nobody types it.
- **The answer is stored as a `ChatResponse` in `jsonb`, whatever kind of request produced
  it.** Streamed answers are assembled into the same shape (step 4), so either kind of request
  can be served from either kind of entry, and Phase 7 can replay semantic hits the same way.
- **Each entry keeps who answered and what that call cost.** Hits are logged at $0, since
  nothing was billed; the stored cost is what a hit saved, which Phase 10's dashboard and
  Phase 11's "cut cost by X%" need without re-deriving it.
- **TTL: `CACHE_TTL_S`, 24 h by default, from the gateway's clock.** `expires_at` is written
  with the row and a lookup ignores expired rows, so expiry needs no background job. A new
  answer for the same key replaces the row and restarts its TTL (an upsert, so two identical
  requests storing at once is fine). `delete_expired` only clears space; it will run at
  startup. 24 h because providers update models under the same name: a long TTL would keep
  serving an older model's answer.
- **The cache can't fail or slow a request.** `ResponseCache.get` turns any error into a miss
  with a warning, and so does a lookup slower than 0.5 s: at that point the database is
  struggling, and the provider is the faster answer. Stores run as kept tasks drained at
  shutdown, like request logs. That task bookkeeping is a copy of `LogWriter`'s, not shared:
  six lines, and a shared helper can come if a third writer appears.
- **Stores are a Protocol with a Postgres and an in-memory version**, like `KeyStore` and
  `UsageLog`. The clock is injectable, so TTL tests don't wait.
- **Lookup order in the route: auth, rate limit, cache, provider.** A hit settles its
  reservation at 0, so it counts against the key's requests per minute but uses none of its
  tokens. The cost of that order: `reserve` checks the token budget before the cache is asked,
  so a key at its TPM limit gets a 429 even for a request that would have been a free hit.
  Accepted, as it's simple and the same for every request. The fix, if hits should skip TPM:
  look up first, then `limiter.acquire(key, 0)` on a hit (RPM only) and `reserve` on a miss.
- **Which answers are kept (`storable`):** one choice, with text, no tool calls, ending in
  `stop` or `length` (`max_tokens` is in the key, so a cut answer is the answer to that
  request). Not content filters, empty replies or failures. An answer that came from a
  fallback provider is cached like any other: it's a real answer for that alias.
- **A hit is a new completion with the stored answer:** a fresh `id` and `created`, the
  original `model`, `choices` and `usage`. The client sees what the answer took to produce;
  the request log says it cost $0 and used no tokens.
- **The hit's row names the provider and model that first answered**, with 0 tokens, $0,
  0 attempts and `cache_status = 'exact'`. Its cost is 0 by rule, not priced at 0 tokens, so a
  model that has since left the price list can't lose the row.
- **Every reply says what the cache did, in `x-gateway-cache`:** `exact`, `miss`, or `bypass`
  (not cacheable, or the client sent `off`). The gateway headers on a hit name the original
  provider with `x-gateway-attempts: 0`.
- **Pricing a reply for the cache is guarded like building a log row:** a failure skips that
  store with a warning. The client is about to get the reply, and nothing in the cache may
  take it away.
- **An app that's never started caches nothing (`NoCacheStore`), and `conftest` swaps the
  Postgres store for it on startup.** An in-memory cache by default would turn every existing
  test that repeats a temperature-0 request into a hit and change its call counts. Tests that
  are about the cache give the app one; one test runs the real startup against Postgres.
- **Streams use the same entries as plain requests.** The key leaves `stream` out, so the
  lookup runs once, before the route splits into streamed and plain. A stored answer serves
  either kind of request, whichever kind produced it.
- **A hit on a stream replays the answer as OpenAI's chunks:** a role chunk, the whole text in
  one chunk, an empty delta with the finish_reason, then the usage-only chunk, sent only with
  `stream_options.include_usage` as on a provider stream, and `[DONE]`. One text chunk because
  the answer is all there already: splitting it would only add events, and a client
  concatenates deltas either way. The chunks go through the same `_events` as a provider's, so
  the rules on usage and errors can't drift apart. Phase 7 can replay semantic hits with it.
- **A stream is stored from its `StreamRecord` once it's closed**, in `on_finish`, next to
  settling and logging: the text, the finish_reason, the usage, and the target's model. Only
  a stream that ended `completed` is kept; one cut by a provider error or a client leaving
  would be served cut forever. A tool-call stream ends with `tool_calls`, which `storable`
  turns away, so the deltas the record doesn't keep can't be lost. Reasoning text the client
  asked for isn't kept either: a hit returns the answer only.
- **Checked with the openai SDK** against the mock: a plain hit, then a streamed hit with
  `include_usage`, read as normal completions, with one provider call in all.
- **The alias is checked before the lookup** (found by the step 5 tests). Without that, an
  alias renamed or removed from the registry kept serving its stored answers until they
  expired, and so did an alias whose providers aren't configured on this deployment. Now both
  fail as they would without a cache: a 404 and a 503.
- **Two identical requests at the same moment both call the provider.** Nothing holds the
  second until the first one's answer is stored; both store under the one key, and the last
  write stays. A test pins this down. Holding duplicates in flight (single-flight) is the fix
  if replay traffic ever shows it matters; for now it costs one extra call per burst.
- **What step 5 covers, beyond steps 3 and 4:** expiry and re-storing at the route, a stream
  the client left and one no provider answered (neither stored), removed and unserved aliases,
  a 429 never reaching the cache, a reply that can't be priced still reaching the client, and
  concurrent duplicates. Not covered by a test: the client leaving during the lookup itself.
  The route reports it like any other cancellation (`client_disconnect`), which
  `test_request_logs` covers, but a cancellation can't be sent through the test transport.
- **Phase 6 "done when", checked live (2026-10-25)** on a real server and Postgres, with a key
  made by the CLI (revoked afterwards), through the openai SDK smoke script and curl:
  - `fast` and `strong` on Groq: the first call a miss (~1 s), the same again an `exact` hit,
    and the same as a stream an `exact` hit replayed as SSE. Hit rows have `latency_ms` of
    2 to 4 ms, 0 tokens, $0 and 0 attempts, and name the model that first answered; the
    openai SDK timed them at 9 to 20 ms end to end. Each miss's cost matched the list prices:
    86 + 16 tokens on gpt-oss-20b is (86 x $0.075 + 16 x $0.30) / 1M = $0.00001125, and the
    entry stored that cost.
  - Stream first: a streamed miss, then a plain request and a stream for the same thing, both
    `exact` in about 5 ms over curl. The replay on the wire: a role chunk, the text, the
    finish_reason, the usage chunk (asked for), `[DONE]`, one id throughout.
  - No temperature and `x-gateway-cache: off` were both `bypass`, and both called Groq.
  - **Found live:** with a small `max_tokens`, gpt-oss-20b sometimes spends all of it
    reasoning and returns empty text with `finish_reason: length` (60 and even 300 tokens on
    one prompt). `storable` turns those away, so they're asked again every time rather than an
    empty answer being served for 24 h. The provider still bills them: the rows show it.
  - Not checked live: a Gemini answer through the cache. Its adapter maps Gemini's finish
    reasons to `stop` and `length`, which `storable` accepts, and nothing else in the cache
    depends on the provider.

## Phase 7: semantic cache

- **torch comes from PyTorch's CPU-only index** (`[tool.uv.sources]` in pyproject.toml). From
  PyPI, Linux gets the CUDA build, which adds several GB of GPU libraries. The CPU build is
  711 MB installed, with no `nvidia-*` packages in the lock file. That size decides whether the
  Phase 12 image fits a free tier, and the gateway embeds one short message per request, which
  a CPU does in milliseconds.
- **Only single-turn requests are embedded, for now:** an optional system (or developer)
  message, then one user message with text, and no `tools` or `functions`. `embed_text` makes
  that call and returns the user message, stripped. A later turn ("and in Rust?") means
  nothing without the turns before it, and one embedded message can't show them. Requests
  that aren't single-turn skip the semantic cache and can still hit the exact cache.
- **One function decides what's embedded, for the gateway and the experiment.** The threshold
  experiment calls `embed_text` too, so the threshold it picks is measured on exactly what the
  gateway embeds.
- **Vectors are normalised at encoding** (`normalize_embeddings=True`), so cosine similarity
  is a dot product, and pgvector's cosine distance is 1 - similarity.
- **The model loads and warms up at startup, in a thread.** Measured on this laptop (8 cores):
  - startup cost: about 11 to 13 s, most of it importing torch, plus about 1.4 s for Hugging
    Face to check for a newer model revision;
  - memory: peak RSS goes from 81 MB to about 510 MB;
  - per request: 12 ms p50, 17 to 22 ms p95 for one prompt from the query log.

  Loading at first use would put those seconds on a client's request. The warm-up embeds one
  text, because the first call is far slower than the rest.
- **A model that can't load turns the semantic cache off; it doesn't stop the gateway.** A
  first start with no network can't download the model. The gateway logs a warning and
  serves every request with the exact cache alone, the same rule as Phase 6: the cache never
  fails a request. `SEMANTIC_CACHE_ENABLED=false` skips loading entirely.
- **Tests never load the model.** conftest gives apps a `FakeEmbedder`, which returns vectors
  the test sets for chosen texts and a hash-seeded random unit vector for any other text.
  Random 384-dimension vectors are nearly orthogonal, so unrelated texts never look similar
  by accident. One test runs the real model, only when it's already in the local Hugging
  Face cache: it checks that "capital of france?" is closer to "What's the capital of
  France?" than "capital of Germany?" is. CI never downloads it.
- **The threshold experiment replays the cache, not "nearest earlier prompt".** Each prompt
  is compared only with earlier prompts that missed, because a hit isn't stored. Matching
  against every earlier prompt would let hits chain through other hits and give a higher
  hit rate than the gateway (or the Phase 11 replay) would ever see. The stored set changes
  with the threshold, so the replay runs once per threshold. Right and wrong are judged by
  meaning group (a prompt and its `dup_of` near-duplicates), not "matched its original":
  only 51 of the 96 duplicates come after their original in file order, and an original
  that hits its earlier duplicate is a right hit.
- **The rule for picking the threshold was written before the first run:** the lowest
  threshold with no false hits at it or at any threshold above it.
- **Finding: with MiniLM alone, no threshold from 0.80 to 0.97 meets that rule**
  (`docs/semantic_threshold.png`, `.csv`). Even at 0.97 the cache makes 5 false hits out of
  16, all on trap pairs. Cosine similarity can't see word order or a changed number:
  - "convert a string to an integer in JavaScript" vs "an integer to a string": 0.996;
  - "100 miles to kilometres" vs "100 kilometres to miles": 0.993;
  - the 5-machines riddle with "1 machine" vs "100 machines": 0.994;
  - the same equations with `5x + y` vs `5x - y`: 0.985.

  The best possible hit rate on this log is 25.4% (every near-duplicate caught). Single
  stage, the hit rate falls to 7.9% at 0.95, where 6 of the 30 hits are still wrong. All 32
  pairs counted false at any threshold were read: none is an unlabelled rephrasing. The
  non-trap ones are genuinely different questions, such as a rephrased "second highest
  salary" query matching the "third highest" one.
- **Tried and rejected: a cross-encoder that checks the top match**
  (`cross-encoder/quora-distilroberta-base`, trained on Quora duplicate questions; measured in a
  scratch run, never added to the project). MiniLM found the candidate at 0.80 or 0.85, and the
  cross-encoder read both prompts together and had to agree. It was worse than useless:
  - it called the traps duplicates: miles/km 0.982, string/int 0.968, `+ y`/`- y` 0.969,
    "national animal"/"national bird" of India 0.980;
  - it called real rephrasings different: "Define photosynthesis." / "what is
    photosynthesis" 0.0003, "what is docker used for" / "What is Docker?" 0.0002;
  - no verifier threshold from 0.05 to 0.95 reached zero false hits (still 3 of 49 at 0.95,
    retrieval 0.85);
  - it costs 30 ms p50 and 53 ms p95 per check on the CPU, 330 MB on disk, about 260 MB more
    memory, and 76 s to load the first time.

  Quora's "duplicate" means "the same topic", which is close but not what a cache needs. A
  word-order swap is invisible to it too.
- **Kept: a numbers guard** (`same_numbers` in app/cache/semantic.py). A match must hold the
  same numbers as the new prompt, in any order. Free (a regex), and at 0.95 it turns away 2
  wrong answers (bat-and-ball at $1.10 vs $1.20, the riddle with 1 vs 100 machines) for 1 right
  one ("three bullets" vs "3 bullet points": numbers written as words aren't understood). It
  can't see order, so miles/km, string/int and `+ y`/`- y` still get through.
- **Decision: the semantic cache is opt-in per request, at 0.95 with the numbers guard.** No
  setup measured is free of wrong answers, so the gateway can't decide for every app that a
  near answer is good enough. An FAQ bot may happily take 1 wrong answer in 7 hits to save
  calls; a coding assistant can't. Portkey makes the same call. A client asks with
  `x-gateway-cache: semantic` (exact is still checked first); without the header a request
  gets the exact cache only, and `off` still turns both off. 0.95 is the user's choice, not
  the rule's (the rule picked nothing): it's where the wrong-answer count gets low (4) before
  the hit rate falls off sharply (7.1% at 0.95, 4.5% at 0.96).
- **What a client opting in gets, measured on the query log at 0.95 with the guard:**
  - 27 semantic hits in 378 prompts: a **7.1% hit rate**, out of a possible 25.4%;
  - 4 of those 27 were wrong: a **14.8% false-hit rate**; all 4 were trap pairs, so 4 of the
    48 look-alike pairs (8.3%) got the other one's answer. Those 4: miles/km, string/int,
    `5x - y` vs `5x + y`, "at least one six" vs "exactly one six".

  The log is adversarial on purpose: 48 trap pairs against 96 near-duplicates, so one near
  pair in three is a trap. Real traffic has fewer, so its false-hit rate is likely lower, but
  that isn't measured here and the README doesn't claim it. The full table, both setups, is
  `docs/semantic_threshold.csv`.
- **A different embedding model wasn't tried.** Word-order swaps fool pooled sentence
  embeddings in general, and the plan names MiniLM. Running bge-small or mpnet through the
  same script is the cheap next check if the false-hit rate ever needs to come down.
- **Semantic entries are scoped to the API key that stored them** (`semantic_cache.key_id`),
  unlike the exact cache. An exact hit returns an answer to the very text the caller sent,
  so sharing it leaks nothing. A semantic hit hands one question the answer to *another*,
  and a single-turn prompt can carry its own data ("summarise this contract: ..."); across
  teams, that leaks. Scoping costs nothing in the measured numbers, because the experiment
  and the Phase 11 replay each run under one key. It does cost hits on a gateway shared by
  many keys asking the same things; that's the price of not leaking.
- **Migration 004 creates `semantic_cache`, a table of its own, not columns on
  `response_cache`.** The two have different keys, and only this one needs a vector.
  - `entry_key` = SHA-256 of key, context and prompt, so the same question in the same
    context replaces its row instead of piling up copies.
  - `context_hash` = SHA-256 of everything but the user message (alias, system prompt, every
    option the model sees), matched exactly. The plan said "same alias and system prompt";
    that alone would let a paraphrase sent with another `max_tokens`, `stop` or
    `response_format` get an answer made for different settings. Step 4 computes it.
  - `prompt` keeps the embedded text, because the numbers guard compares it at lookup.
  - `embedding vector(384)`, cosine distance; a test pins 384 to the embedder's width.
  - The migration runs `CREATE EXTENSION IF NOT EXISTS vector` itself.
    `docker/postgres/init.sql` only runs on a fresh local volume, and Neon or Supabase never
    run it.
- **An HNSW index on the embedding, plus btree on (key_id, context_hash) and on expires_at.**
  HNSW fetches its nearest candidates first (`ef_search`, 40) and only then applies the
  WHERE filter. With entries scoped per key, all 40 can belong to other keys, and the lookup
  then finds nothing even though a match exists. A test pins this: with iterative scan off,
  the HNSW plan finds 0 rows; with `hnsw.iterative_scan = relaxed_order` (pgvector 0.8, here
  0.8.7) it keeps searching and finds the right one. Step 4's lookup sets that. On a small
  table the planner skips HNSW for the btree index and an exact scan, which is right too.
- **The semantic cache mirrors the exact one's shape** (`app/cache/semantic_store.py`): a
  Postgres store, an in-memory store for tests, a `NoSemanticStore` that conftest swaps in so
  no existing test starts hitting, and a `SemanticCache` front whose lookups never raise and
  whose stores run in tasks drained at shutdown. A lookup returns the same `CacheEntry` the
  exact cache does, so step 5 serves and replays a semantic hit with the code it already has.
- **The store returns the nearest entry however far; the front decides.** The threshold and
  the numbers guard live in one place (`SemanticCache.get`), the same for both stores, and
  the store reports the similarity, so a test can check the arithmetic.
- **The context hash shares the exact key's idea of "what the model sees"**
  (`model_fields` in key.py, used by both). It's that, minus the user message's text, under
  its own version prefix. The user message's role and extra keys (`name`) still count.
- **A request is embedded at most once.** The lookup keeps its vector in the
  `SemanticLookup`, and the store after a miss reuses it. If the lookup timed out before the
  vector was ready, the store embeds in the background, off the request's path.
- **Embedding runs on one thread of its own** (a one-worker `ThreadPoolExecutor`). Each
  embedding is about 12 ms of CPU that torch already spreads across cores, so several at once
  only fight over them. A queue of one keeps that predictable and leaves the event loop's
  shared thread pool free. Not done: capping torch's own threads
  (`torch.set_num_threads`). In a container, torch may see the host's cores rather than the
  container's share, and that's for Phase 12 to check on the real box.
- **Vectors go to Postgres as pgvector's text form, cast with `::vector`.** No type adapter
  to register on the pool, which would have to run after migration 004 creates the type.
- **Each lookup sets `hnsw.iterative_scan` with `SET LOCAL`,** for that transaction only, so
  pooled connections don't carry it to other queries.
- **Checked against the real model and Postgres:** replaying the 378 prompts through
  `SemanticCache` (one key, one context) gave 27 hits, the same 27 the experiment counts at
  0.95 with the guard, so the gateway serves what was measured. Lookups, embedding plus query,
  took 24 ms p50, 34 ms p95 and 45 ms at most, well inside the 0.5 s lookup timeout. A
  request that opts in and misses pays that on top of the provider call.
