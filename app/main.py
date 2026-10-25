import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from app import __version__
from app.api.errors import install_error_handlers
from app.api.routes import router
from app.config import Settings, get_settings
from app.providers.base import Provider
from app.providers.groq import GroqProvider
from app.providers.mock import MockProvider
from app.routing.registry import GroqSpec, MockSpec, RegistryConfig, load_registry
from app.routing.router import ModelRouter

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
                )
            case GroqSpec():
                if not settings.groq_api_key:
                    logger.warning("GROQ_API_KEY is not set: skipping provider %r", name)
                    continue
                providers[name] = GroqProvider(client, settings.groq_api_key.get_secret_value())
    return providers


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Read and validate the registry at startup, so a bad file stops the app right away.
        config = load_registry(settings.models_file)
        # One pooled HTTP client for all provider calls, closed on shutdown.
        async with httpx.AsyncClient(timeout=settings.request_timeout_s) as client:
            providers = build_providers(config, settings, client)
            app.state.model_router = ModelRouter(config.aliases, providers)
            yield

    app = FastAPI(title="LLM Gateway", version=__version__, lifespan=lifespan)
    install_error_handlers(app)
    app.include_router(router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "env": settings.app_env, "version": __version__}

    return app


app = create_app()
