"""The Groq adapter's streaming, against a fake transport. Never calls the real API.

The fake SSE bodies copy what Groq sent on 2026-10-25 (trimmed): a role chunk with empty
content, content chunks, a finish chunk carrying `x_groq`, a usage chunk, then [DONE].
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
from app.providers.groq import GroqProvider
from app.providers.mock import MockProvider
from app.routing.registry import Target
from app.routing.router import ModelRouter

REQUEST = ChatRequest(
    model="openai/gpt-oss-20b", messages=[{"role": "user", "content": "hi"}], stream=True
)
META = {
    "id": "chatcmpl-1",
    "object": "chat.completion.chunk",
    "created": 1792914138,
    "model": "openai/gpt-oss-20b",
    "system_fingerprint": "fp_1",
}
USAGE = {"prompt_tokens": 76, "completion_tokens": 51, "total_tokens": 127, "queue_time": 0.4}


def chunk(delta: dict | None = None, finish: str | None = None, **extra) -> dict:
    choice = {"index": 0, "delta": delta or {}, "logprobs": None, "finish_reason": finish}
    return {**META, "choices": [choice], **extra}


GROQ_STREAM = [
    chunk({"role": "assistant", "content": ""}, x_groq={"id": "req_1", "seed": 1}),
    chunk({"content": "Hello"}),
    chunk({"content": " there"}),
    chunk(finish="stop", x_groq={"id": "req_1", "usage": USAGE}),
    {**META, "choices": [], "usage": USAGE},
]


def sse(*events: dict | str) -> bytes:
    return b"".join(
        f"data: {e if isinstance(e, str) else json.dumps(e)}\n\n".encode() for e in events
    )


def provider_with(handler) -> GroqProvider:
    return GroqProvider(httpx.AsyncClient(transport=httpx.MockTransport(handler)), "gsk-test")


def streaming(body: bytes, status: int = 200):
    return lambda request: httpx.Response(
        status, content=body, headers={"content-type": "text/event-stream"}
    )


async def collect(provider: GroqProvider, request: ChatRequest = REQUEST) -> list:
    return [c async for c in provider.stream(request)]


async def test_parses_groqs_stream():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.read())
        return streaming(sse(*GROQ_STREAM, "[DONE]"))(request)

    chunks = await collect(provider_with(handler))

    assert seen["body"]["stream"] is True
    assert seen["body"]["stream_options"] == {"include_usage": True}
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "Hello there"
    assert chunks[0].choices[0].delta.role == "assistant"
    assert not chunks[0].has_output()  # the empty role chunk doesn't commit the router
    assert chunks[3].choices[0].finish_reason == "stop"
    assert chunks[-1].choices == []
    assert chunks[-1].usage.total_tokens == 127
    # Groq's own fields don't reach the client.
    assert all("x_groq" not in c.model_dump() for c in chunks)


async def test_usage_is_requested_even_when_the_client_set_other_stream_options():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.read())
        return streaming(sse(*GROQ_STREAM, "[DONE]"))(request)

    request = REQUEST.model_copy(
        update={"stream_options": {"include_usage": False, "something": 1}}
    )
    await collect(provider_with(handler), request)

    assert seen["body"]["stream_options"] == {"include_usage": True, "something": 1}


async def test_reasoning_chunks_count_as_output():
    # Only sent when the client turned include_reasoning on, and then it's what they see first.
    reasoning = chunk({"reasoning": "Need", "channel": "analysis"})
    chunks = await collect(provider_with(streaming(sse(GROQ_STREAM[0], reasoning, "[DONE]"))))

    assert chunks[1].choices[0].delta.model_extra["reasoning"] == "Need"
    assert chunks[1].has_output()


@pytest.mark.parametrize(
    ("status", "error", "expected"),
    [
        (429, {"message": "slow down", "type": "tokens"}, 429),
        (400, {"message": "'temperature' : number must be at most 2"}, 400),
        # A model Groq doesn't serve is our config problem: a 502 that falls back.
        (404, {"message": "no such model", "code": "model_not_found"}, 502),
    ],
)
async def test_error_before_the_stream_starts(status, error, expected):
    provider = provider_with(lambda r: httpx.Response(status, json={"error": error}))

    with pytest.raises(ProviderError) as exc_info:
        await collect(provider)

    assert exc_info.value.status == expected


async def test_error_event_inside_the_stream():
    body = sse(GROQ_STREAM[0], GROQ_STREAM[1], {"error": {"message": "overloaded"}})
    stream = provider_with(streaming(body)).stream(REQUEST)

    assert (await anext(stream)).choices[0].delta.content == ""
    assert (await anext(stream)).choices[0].delta.content == "Hello"
    with pytest.raises(ProviderError) as exc_info:
        await anext(stream)

    assert exc_info.value.status == 502
    assert exc_info.value.message == "overloaded"


async def test_stream_cut_without_done_is_an_error():
    provider = provider_with(streaming(sse(*GROQ_STREAM[:2])))

    with pytest.raises(ProviderError) as exc_info:
        await collect(provider)

    assert exc_info.value.status == 502
    assert exc_info.value.message == "Groq ended the stream early"


async def test_unreadable_chunk_is_a_502():
    provider = provider_with(streaming(sse(GROQ_STREAM[0], "<html>oops</html>")))

    with pytest.raises(ProviderError) as exc_info:
        await collect(provider)

    assert exc_info.value.status == 502


async def test_timeout_between_chunks_is_a_504():
    async def body():
        yield sse(GROQ_STREAM[0])
        raise httpx.ReadTimeout("no data")

    provider = provider_with(lambda r: httpx.Response(200, content=body()))

    with pytest.raises(ProviderError) as exc_info:
        await collect(provider)

    assert exc_info.value.status == 504


class RecordingBody(httpx.AsyncByteStream):
    """A response body that records whether httpx closed it (what frees a real connection)."""

    def __init__(self, *events: dict) -> None:
        self._events = events
        self.closed = False

    async def __aiter__(self):
        for event in self._events:
            yield sse(event)

    async def aclose(self) -> None:
        self.closed = True


async def test_closing_the_stream_early_closes_the_upstream_response():
    body = RecordingBody(*GROQ_STREAM)
    stream = provider_with(lambda r: httpx.Response(200, stream=body)).stream(REQUEST)

    await anext(stream)
    assert not body.closed
    await stream.aclose()

    assert body.closed


def sdk_client(model_router: ModelRouter) -> openai.AsyncOpenAI:
    app = create_app()
    app.dependency_overrides[get_model_router] = lambda: model_router
    return openai.AsyncOpenAI(
        api_key="unused-until-phase-4",
        base_url="http://gateway.test/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app)),
        max_retries=0,
    )


def groq_then_mock(handler) -> ModelRouter:
    return ModelRouter(
        {
            "fast": [
                Target(provider="groq", model="openai/gpt-oss-20b"),
                Target(provider="mock", model="mock-model"),
            ]
        },
        {"groq": provider_with(handler), "mock": MockProvider(reply="from mock")},
        sleep=lambda _: asyncio.sleep(0),
    )


async def test_sdk_streams_through_the_real_groq_adapter():
    client = sdk_client(groq_then_mock(streaming(sse(*GROQ_STREAM, "[DONE]"))))

    stream = await client.chat.completions.create(
        model="fast", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    chunks = [c async for c in stream]

    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "Hello there"
    # Groq always sends usage now; the client didn't ask, so it's filtered out.
    assert all(c.choices for c in chunks)


async def test_groq_outage_before_the_first_token_is_served_by_the_next_provider():
    client = sdk_client(groq_then_mock(lambda r: httpx.Response(503, json={"error": {}})))

    raw = await client.chat.completions.with_raw_response.create(
        model="fast", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    text = "".join([c.choices[0].delta.content or "" async for c in raw.parse()])

    assert text == "from mock"
    assert raw.headers["x-gateway-provider"] == "mock"
    assert raw.headers["x-gateway-fallback"] == "true"
