import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from app import __version__
from app.api.errors import install_error_handlers
from app.api.routes import router
from app.auth.authenticator import Authenticator
from app.auth.keys import PostgresKeyStore
from app.auth.rate_limit import RateLimiter
from app.config import Settings, get_settings
from app.db.pool import open_pool
from app.providers.base import Provider
from app.providers.gemini import GeminiProvider
from app.providers.groq import GroqProvider
from app.providers.mock import MockProvider
from app.routing.registry import GeminiSpec, GroqSpec, MockSpec, RegistryConfig, load_registry
from app.routing.router import ModelRouter
from app.usage.tokens import load_encoding_in_background

logger = logging.getLogger(__name__)


def build_providers(
    config: RegistryConfig, settings: Settings, client: httpx.AsyncClient
) -> dict[str, Provider]:
    """One instance per entry in the registry's `providers`. Ones missing a key are left out."""
    providers: dict[str, Provider] = {}
    for name, spec in config.providers.items():
        match spec:
            case MockSpec():
                providers[name] = MockProvider(
                    reply=spec.reply,
                    latency_s=spec.latency_s,
                    failure_rate=spec.failure_rate,
                    fail_status=spec.fail_status,
                    chunk_latency_s=spec.chunk_latency_s,
                    fail_after_chunks=spec.fail_after_chunks,
                )
            case GroqSpec():
                if not settings.groq_api_key:
                    logger.warning("GROQ_API_KEY is not set: skipping provider %r", name)
                    continue
                providers[name] = GroqProvider(
                    client,
                    settings.groq_api_key.get_secret_value(),
                    reasoning_effort=spec.reasoning_effort,
                    include_reasoning=spec.include_reasoning,
                )
            case GeminiSpec():
                if not settings.gemini_api_key:
                    logger.warning("GEMINI_API_KEY is not set: skipping provider %r", name)
                    continue
                providers[name] = GeminiProvider(
                    client, settings.gemini_api_key.get_secret_value(), spec.thinking_level
                )
    return providers


def create_app() -> FastAPI:
    settings = get_settings()
    # The app's own loggers (uvicorn configures only its own). Without a handler, Python shows
    # warnings only, so the per-stream summary lines would never appear. JSON logs: Phase 12.
    logging.basicConfig(level=settings.log_level, format="%(levelname)s:  %(name)s: %(message)s")
    # httpx logs every upstream request at INFO; the gateway's own lines say what matters.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Read and validate the registry at startup, so a bad file stops the app right away.
        config = load_registry(settings.models_file)
        # For estimating usage when a stream ends early. Never blocks startup.
        load_encoding_in_background()
        # One pooled HTTP client for all provider calls, and one Postgres pool (API keys now,
        # request logs and the caches later). Both closed on shutdown.
        async with (
            httpx.AsyncClient(timeout=settings.request_timeout_s) as client,
            open_pool(settings.database_url) as db_pool,
        ):
            app.state.db_pool = db_pool
            app.state.authenticator = Authenticator(
                PostgresKeyStore(db_pool), ttl_s=settings.key_cache_ttl_s
            )
            providers = build_providers(config, settings, client)
            app.state.model_router = ModelRouter(
                config.aliases, providers, deadline_s=settings.request_deadline_s
            )
            yield

    app = FastAPI(title="LLM Gateway", version=__version__, lifespan=lifespan)
    # In memory and nothing to open or close, so it's built here rather than in the lifespan.
    app.state.rate_limiter = RateLimiter(
        default_completion_tokens=settings.rate_limit_default_completion_tokens
    )
    install_error_handlers(app)
    app.include_router(router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "env": settings.app_env, "version": __version__}

    return app


app = create_app()
