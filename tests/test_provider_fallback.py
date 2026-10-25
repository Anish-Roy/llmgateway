"""Phase 2's "done when": Groq fails, Gemini answers, through the real adapters and the SDK.

Both adapters run for real; only their HTTP transport is faked, so no network is used.
"""

import asyncio

import httpx
import httpx2
import openai

from app.api.routes import get_model_router
from app.main import create_app
from app.providers.gemini import GeminiProvider
from app.providers.groq import GroqProvider
from app.routing.registry import Target
from app.routing.router import ModelRouter
from tests.auth_helpers import authorize

GEMINI_OK = {
    "candidates": [{"content": {"parts": [{"text": "Hi from Gemini"}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 4},
    "modelVersion": "gemini-3.5-flash-lite",
}


def fake_upstreams(groq_status: int, calls: list[str]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        calls.append(host)
        if host == "api.groq.com":
            return httpx.Response(groq_status, json={"error": {"message": "Groq is down"}})
        return httpx.Response(200, json=GEMINI_OK)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def sdk_client(model_router: ModelRouter) -> openai.AsyncOpenAI:
    app = create_app()
    app.dependency_overrides[get_model_router] = lambda: model_router
    return openai.AsyncOpenAI(
        api_key=authorize(app),
        base_url="http://gateway.test/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app)),
        max_retries=0,
    )


def fast_router(upstream: httpx.AsyncClient) -> ModelRouter:
    return ModelRouter(
        {
            "fast": [
                Target(provider="groq", model="openai/gpt-oss-20b"),
                Target(provider="gemini", model="gemini-3.5-flash-lite"),
            ]
        },
        {
            "groq": GroqProvider(upstream, "gsk-test"),
            "gemini": GeminiProvider(upstream, "AIza-test"),
        },
        sleep=lambda _: asyncio.sleep(0),
    )


async def test_groq_outage_is_served_by_gemini():
    calls: list[str] = []
    client = sdk_client(fast_router(fake_upstreams(503, calls)))

    raw = await client.chat.completions.with_raw_response.create(
        model="fast", messages=[{"role": "user", "content": "hi"}]
    )
    resp = raw.parse()

    assert resp.choices[0].message.content == "Hi from Gemini"
    assert resp.model == "gemini-3.5-flash-lite"
    assert resp.usage.total_tokens == 7
    assert raw.headers["x-gateway-provider"] == "gemini"
    assert raw.headers["x-gateway-fallback"] == "true"
    # Groq's 503 was retried once before falling back.
    assert calls == ["api.groq.com", "api.groq.com", "generativelanguage.googleapis.com"]


async def test_groq_model_access_loss_falls_back_without_retry():
    # What Groq really returned for a model the key can't use (2026-10-25).
    calls: list[str] = []
    client = sdk_client(fast_router(fake_upstreams(404, calls)))

    raw = await client.chat.completions.with_raw_response.create(
        model="fast", messages=[{"role": "user", "content": "hi"}]
    )

    assert raw.headers["x-gateway-provider"] == "gemini"
    assert calls == ["api.groq.com", "generativelanguage.googleapis.com"]


async def test_models_endpoint_lists_aliases():
    model_router = ModelRouter(
        {
            "fast": [Target(provider="groq", model="openai/gpt-oss-20b")],
            "unconfigured": [Target(provider="nobody", model="x")],
        },
        {"groq": GroqProvider(httpx.AsyncClient(), "gsk-test")},
    )
    client = sdk_client(model_router)

    models = await client.models.list()

    # Only aliases that can actually be served.
    assert [m.id for m in models.data] == ["fast"]
