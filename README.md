# LLM Gateway

An OpenAI-compatible proxy in front of multiple LLM providers, with fallback, per-key rate
limits, exact and semantic caching, PII masking and a cost dashboard.

Status: Phase 6 done (Groq + Gemini behind model aliases with retry and fallback, streaming,
API keys, per-key rate limits, a request log with the cost of every call, and an exact-match
response cache). See [plan.md](plan.md) for the roadmap.

## Quick start

```bash
uv sync                                   # create .venv and install deps
cp .env.example .env                      # then add your provider keys
docker compose up -d                      # Postgres 17 + pgvector on localhost:5433
uv run python -m scripts.migrate          # create/update tables (safe to rerun)
uv run uvicorn app.main:app --reload      # http://localhost:8000/health
uv run pytest
uv run ruff check .
```

Clients ask for a model **alias**, not a vendor model. Aliases and the providers behind them
live in [config/models.yaml](config/models.yaml):

| Alias | Tried in order | |
|---|---|---|
| `fast` | Groq `openai/gpt-oss-20b` → Gemini `gemini-3.5-flash-lite` | cheap, quick |
| `strong` | Groq `openai/gpt-oss-120b` → Gemini `gemini-3.8-flash` | better answers |
| `mock` | mock provider | no provider keys needed |
| `demo-fallback` | always-failing mock → mock | fallback demo, no provider keys |
| `demo-midstream` | mock that fails mid-answer → mock | streaming failure demo, no provider keys |

`GET /v1/models` lists the aliases this deployment can serve. If a provider times out, rate
limits (429) or errors (5xx), the gateway retries once where that helps and moves to the next
entry. It doesn't fall back on a bad request (400), which every provider would reject. The
response's `model` is the model that actually answered, and the `x-gateway-provider`,
`x-gateway-attempts` and `x-gateway-fallback` headers show how it got there.

Providers retire models and change free-tier access often. Check the registry against what
your keys can actually use:

```bash
uv run python -m scripts.check_models
```

Every `/v1` request needs a gateway API key (`/health` doesn't). Keys are created with a CLI
that talks to Postgres directly; the plain key is printed once and only its hash is stored:

```bash
uv run python -m scripts.create_key create "alice" --rpm 60 --tpm 100000
uv run python -m scripts.create_key list
uv run python -m scripts.create_key revoke 1     # applies within KEY_CACHE_TTL_S (30 s)
```

Each key has a requests-per-minute and a tokens-per-minute limit. Over either, the gateway
answers 429 with `Retry-After` (the openai SDK waits and retries by itself), and every reply
carries OpenAI's `x-ratelimit-limit-*`, `x-ratelimit-remaining-*` and `x-ratelimit-reset-*`
headers for requests and tokens. A request's tokens are reserved up front (prompt plus
`max_tokens`, or 1024 without it) and corrected to the real usage when it ends. Limits are kept
in memory, so run the gateway as a single process.

Try it with the official `openai` SDK (needs `GROQ_API_KEY` and `GEMINI_API_KEY` in `.env`).
It asks once, asks the same again (a cache hit), then asks the same as a stream (a hit replayed
as SSE), and prints what the cache did and how long each call took:

```bash
GATEWAY_API_KEY=gw-... uv run python scripts/smoke_openai_sdk.py                  # model "fast"
GATEWAY_API_KEY=gw-... uv run python scripts/smoke_openai_sdk.py --model strong
GATEWAY_API_KEY=gw-... uv run python scripts/smoke_openai_sdk.py --model mock     # no provider keys needed
```

To watch fallback without any provider keys, ask for `demo-fallback`. Its first provider always fails
with a 503 and is retried once, then the mock answers:

```bash
curl -si localhost:8000/v1/chat/completions -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -H 'content-type: application/json' \
  -d '{"model": "demo-fallback", "messages": [{"role": "user", "content": "hi"}]}'
```

Look for `x-gateway-provider: mock`, `x-gateway-attempts: 3` and `x-gateway-fallback: true`.

### Streaming

`stream=True` returns server-sent events in OpenAI's chunk format, ending with `data: [DONE]`.
Add `stream_options={"include_usage": True}` to get a final chunk with token usage. Fallback
works until the first token: the gateway picks a provider before it starts the response, so a
provider that fails early is replaced, and if all of them fail you get a normal JSON error. Once
part of the answer has been sent, a failure ends the stream with an error event instead of
`[DONE]`, and the openai SDK raises it as an `APIError`. `demo-midstream` shows this:

```bash
curl -sN localhost:8000/v1/chat/completions -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -H 'content-type: application/json' \
  -d '{"model": "demo-midstream", "stream": true, "messages": [{"role": "user", "content": "hi"}]}'
```

Every provider streams: Groq passes its OpenAI-format chunks through, and Gemini's
`streamGenerateContent` events are translated into the same format.

### Caching

A request with `temperature` set at or below `CACHE_MAX_TEMPERATURE` (0.2) is answered from the
exact-match cache when the same request was answered before, within `CACHE_TTL_S` (24 h). "The
same" means the same alias, messages and every option the model sees (`max_tokens`, `tools`,
`stop`, `seed`...); `stream` and `user` don't count, so a streamed answer can serve a plain
request and the other way round. A hit on a stream is replayed as SSE chunks. One cache serves
every API key, stored in Postgres (`response_cache`).

Every reply says what happened in `x-gateway-cache`: `exact` (a hit, a few milliseconds, no
provider call, `x-gateway-attempts: 0`), `miss`, or `bypass`. Requests without a temperature,
above the limit, or with `n` > 1 are bypassed, and so is any request sent with
`x-gateway-cache: off`. A hit still counts against the key's requests per minute but uses no
tokens, and its request-log row has `cache_status = 'exact'` and $0. Only finished text answers
are stored: not errors, cut streams, tool calls or empty replies (a reasoning model that spends
all of `max_tokens` thinking returns no text, and isn't cached).

The demo commands above send no temperature, so they always reach the providers. To repeat a
cacheable request without the cache, add `-H 'x-gateway-cache: off'`.

Known limit: two identical requests that arrive at the same moment both call the provider.

## Layout

```
app/
  api/          OpenAI-compatible /v1 routes
  providers/    Groq, Gemini and mock adapters
  routing/      model registry, cheap-vs-strong router
  auth/         API keys, rate limits
  cache/        exact-match cache (semantic: Phase 7)
  guardrails/   PII masking
  usage/        request logs, cost
  config.py     settings from env / .env
  main.py       FastAPI app
data/queries.jsonl       evaluation query log (built by scripts/build_dataset.py)
dashboard/               Streamlit cost dashboard
docs/decisions.md        design decisions, one entry per phase
```

## Query log

`data/queries.jsonl` has 378 prompts for evaluating the cache and router. Each row has an
`id`, `prompt`, `category`, `difficulty` (`easy`/`hard`), and two optional links:

- `dup_of`: a near-duplicate of another prompt (same meaning). The semantic cache should hit.
- `trap_of`: looks similar to another prompt but needs a different answer. The cache must not hit.

Edit the prompts in `scripts/build_dataset.py` and regenerate with
`uv run python scripts/build_dataset.py`.
