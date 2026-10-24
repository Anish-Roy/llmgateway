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


def build_provider(settings: Settings, client: httpx.AsyncClient) -> Provider:
    if settings.provider == "mock":
        return MockProvider(
            latency_s=settings.mock_latency_s, failure_rate=settings.mock_failure_rate
        )
    key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else None
    return GroqProvider(client, key)


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # One pooled HTTP client for all provider calls, closed on shutdown.
        async with httpx.AsyncClient(timeout=settings.request_timeout_s) as client:
            app.state.provider = build_provider(settings, client)
            yield

    app = FastAPI(title="LLM Gateway", version=__version__, lifespan=lifespan)
    install_error_handlers(app)
    app.include_router(router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "env": settings.app_env, "version": __version__}

    return app


app = create_app()
