# LLM Gateway

An OpenAI-compatible proxy in front of multiple LLM providers, with fallback, per-key rate
limits, exact and semantic caching, PII masking and a cost dashboard.

Status: Phase 2 in progress (model registry with aliases; Gemini and fallback next). See [plan.md](plan.md) for the roadmap.

## Quick start

```bash
uv sync                                   # create .venv and install deps
cp .env.example .env                      # then add your provider keys
docker compose up -d                      # Postgres 17 + pgvector on localhost:5433
uv run uvicorn app.main:app --reload      # http://localhost:8000/health
uv run pytest
uv run ruff check .
```

Clients ask for a model **alias**, not a vendor model. Aliases and the providers behind them
live in [config/models.yaml](config/models.yaml); `fast` goes to Groq, and `mock` needs no keys.

Try it with the official `openai` SDK (`fast` needs `GROQ_API_KEY` in `.env`):

```bash
uv run python scripts/smoke_openai_sdk.py                # model "fast"
uv run python scripts/smoke_openai_sdk.py --model mock   # no keys needed
```

## Layout

```
app/
  api/          OpenAI-compatible /v1 routes
  providers/    Groq, Gemini and mock adapters
  routing/      model registry, cheap-vs-strong router
  auth/         API keys, rate limits
  cache/        exact + semantic cache
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
