"""The Gemini adapter's streaming: streamGenerateContent events -> OpenAI chunks, faked transport.

The fake events copy what gemini-3.5-flash-lite sent on 2026-10-25 (trimmed). Each event is a
whole GenerateContentResponse with the next piece of text, usageMetadata holds running totals,
the last event has the finishReason (with an empty part carrying a thoughtSignature), and
there's no [DONE].
"""

import asyncio
import json

import httpx
import httpx2
import openai
import pytest

from app.api.routes import get_model_router
from app.api.schemas import ChatRequest
from app.main import create_app
from app.providers.base import ProviderError
from app.providers.gemini import GeminiProvider, to_gemini
from app.providers.groq import GroqProvider
from app.routing.registry import Target
from app.routing.router import ModelRouter

REQUEST = ChatRequest(
    model="gemini-3.5-flash-lite", messages=[{"role": "user", "content": "hi"}], stream=True
)


def event(*parts: dict, finish: str | None = None, usage: dict | None = None) -> dict:
    candidate = {"content": {"parts": list(parts), "role": "model"}, "index": 0}
    if finish:
        candidate["finishReason"] = finish
    return {
        "candidates": [candidate],
        "usageMetadata": usage or {"promptTokenCount": 9, "totalTokenCount": 9},
        "modelVersion": "gemini-3.5-flash-lite",
        "responseId": "RbTdapqn",
    }


GEMINI_STREAM = [
    event(
        {"text": "Amazon\n"},
        usage={"promptTokenCount": 9, "candidatesTokenCount": 2, "thoughtsTokenCount": 30},
    ),
    event(
        {"text": "Nile"},
        usage={"promptTokenCount": 9, "candidatesTokenCount": 6, "thoughtsTokenCount": 30},
    ),
    event(
        {"text": "", "thoughtSignature": "EmAK"},
        finish="STOP",
        usage={"promptTokenCount": 9, "candidatesTokenCount": 6, "thoughtsTokenCount": 30},
    ),
]


def sse(*events: dict) -> bytes:
    return b"".join(f"data: {json.dumps(e)}\r\n\r\n".encode() for e in events)


def streaming(body: bytes, status: int = 200):
    return lambda request: httpx.Response(
        status, content=body, headers={"content-type": "text/event-stream"}
    )


def provider_with(handler) -> GeminiProvider:
    return GeminiProvider(httpx.AsyncClient(transport=httpx.MockTransport(handler)), "AIza-test")


async def collect(provider: GeminiProvider, request: ChatRequest = REQUEST) -> list:
    return [c async for c in provider.stream(request)]


async def test_translates_geminis_stream():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = request.url
        seen["key"] = request.headers["x-goog-api-key"]
        return streaming(sse(*GEMINI_STREAM))(request)

    chunks = await collect(provider_with(handler))

    assert seen["url"].path == "/v1beta/models/gemini-3.5-flash-lite:streamGenerateContent"
    assert seen["url"].params["alt"] == "sse"
    assert "key" not in seen["url"].params
    assert seen["key"] == "AIza-test"

    text = [c.choices[0].delta.content for c in chunks if c.choices]
    assert text == ["Amazon\n", "Nile", None]
    assert chunks[0].choices[0].delta.role == "assistant"
    assert all(c.choices[0].delta.role is None for c in chunks[1:] if c.choices)
    assert chunks[2].choices[0].finish_reason == "stop"
    assert {c.id for c in chunks} == {"chatcmpl-RbTdapqn"}
    assert {c.model for c in chunks} == {"gemini-3.5-flash-lite"}
    # The last running total, with thinking tokens counted as output (billed that way).
    usage = chunks[-1].usage
    assert chunks[-1].choices == []
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (9, 36, 45)


async def test_thought_only_events_send_nothing():
    thought = event({"text": "**Assessing**", "thought": True})
    body = sse(thought, *GEMINI_STREAM)

    chunks = await collect(provider_with(streaming(body)))

    assert chunks[0].choices[0].delta.content == "Amazon\n"
    assert chunks[0].choices[0].delta.role == "assistant"


@pytest.mark.parametrize(
    ("reason", "expected"), [("MAX_TOKENS", "length"), ("SAFETY", "content_filter")]
)
async def test_finish_reasons_are_mapped(reason, expected):
    body = sse(GEMINI_STREAM[0], event({"text": ""}, finish=reason))

    chunks = await collect(provider_with(streaming(body)))

    assert chunks[1].choices[0].finish_reason == expected


async def test_blocked_prompt_is_an_empty_content_filter_answer():
    blocked = {"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {}}

    chunks = await collect(provider_with(streaming(sse(blocked))))

    assert len(chunks) == 1
    assert chunks[0].choices[0].finish_reason == "content_filter"
    assert chunks[0].has_output()  # a real answer: no fallback to another provider


async def test_stream_cut_before_the_finish_reason_is_an_error():
    provider = provider_with(streaming(sse(*GEMINI_STREAM[:2])))

    with pytest.raises(ProviderError) as exc_info:
        await collect(provider)

    assert exc_info.value.status == 502
    assert exc_info.value.message == "Gemini ended the stream early"


async def test_error_event_inside_the_stream():
    body = sse(GEMINI_STREAM[0], {"error": {"code": 500, "message": "boom", "status": "INTERNAL"}})
    stream = provider_with(streaming(body)).stream(REQUEST)

    assert (await anext(stream)).choices[0].delta.content == "Amazon\n"
    with pytest.raises(ProviderError) as exc_info:
        await anext(stream)

    assert (exc_info.value.status, exc_info.value.message) == (502, "boom")


@pytest.mark.parametrize(
    ("status", "error", "expected"),
    [
        # As sent live on 2026-10-25, with a text/event-stream content-type.
        (503, {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}, 503),
        (400, {"code": 400, "message": "temperature range", "status": "INVALID_ARGUMENT"}, 400),
        (404, {"code": 404, "message": "not found", "status": "NOT_FOUND"}, 502),
        (
            400,
            {
                "code": 400,
                "message": "API key not valid.",
                "status": "INVALID_ARGUMENT",
                "details": [{"reason": "API_KEY_INVALID"}],
            },
            502,
        ),
    ],
)
async def test_error_before_the_stream_starts(status, error, expected):
    body = json.dumps({"error": error}, indent=2).encode()

    with pytest.raises(ProviderError) as exc_info:
        await collect(provider_with(streaming(body, status)))

    assert exc_info.value.status == expected


async def test_unreadable_event_is_a_502():
    body = b"data: <html>oops</html>\n\n"

    with pytest.raises(ProviderError) as exc_info:
        await collect(provider_with(streaming(body)))

    assert exc_info.value.status == 502


def test_stream_options_are_accepted():
    # Rejecting it would make every include_usage client skip Gemini.
    request = REQUEST.model_copy(update={"stream_options": {"include_usage": True}})

    assert "stream_options" not in json.dumps(to_gemini(request))


async def test_groq_outage_before_the_first_token_streams_from_gemini():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        if request.url.host == "api.groq.com":
            return httpx.Response(503, json={"error": {"message": "Groq is down"}})
        return streaming(sse(*GEMINI_STREAM))(request)

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    model_router = ModelRouter(
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
    app = create_app()
    app.dependency_overrides[get_model_router] = lambda: model_router
    client = openai.AsyncOpenAI(
        api_key="unused-until-phase-4",
        base_url="http://gateway.test/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app)),
        max_retries=0,
    )

    raw = await client.chat.completions.with_raw_response.create(
        model="fast",
        messages=[{"role": "user", "content": "rivers?"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks = [c async for c in raw.parse()]

    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "Amazon\nNile"
    assert chunks[-1].usage.completion_tokens == 36
    assert raw.headers["x-gateway-provider"] == "gemini"
    assert raw.headers["x-gateway-fallback"] == "true"
    # Groq's 503 was retried once, then Gemini served it.
    assert calls == ["api.groq.com", "api.groq.com", "generativelanguage.googleapis.com"]
