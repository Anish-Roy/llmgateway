"""Streaming (stream=True): the openai SDK and the raw SSE wire format, against mock providers."""

import asyncio
import json

import httpx2
import openai
import pytest
from starlette.requests import ClientDisconnect

from app.api.routes import get_model_router
from app.api.schemas import ChatChunk, ChatRequest, ChunkChoice, Delta
from app.main import create_app
from app.providers.base import ProviderError
from app.providers.mock import MockProvider
from app.routing.registry import Target
from app.routing.router import ModelRouter
from tests.auth_helpers import authorize

MESSAGES = [{"role": "user", "content": "hello there"}]


def make_app(*providers: tuple[str, MockProvider], **router_kwargs):
    """An app whose alias "fast" tries the given (name, provider) pairs in order."""
    app = create_app()
    model_router = ModelRouter(
        {"fast": [Target(provider=name, model=f"{name}-model") for name, _ in providers]},
        dict(providers),
        sleep=lambda _: asyncio.sleep(0),
        **router_kwargs,
    )
    app.dependency_overrides[get_model_router] = lambda: model_router
    app.state.test_key = authorize(app)
    return app


def sdk_client(app) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        api_key=app.state.test_key,
        base_url="http://gateway.test/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app)),
        max_retries=0,
    )


async def raw_events(app, **body) -> list[str]:
    """The `data:` payloads of a streamed response, exactly as sent."""
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://gateway.test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "fast", "messages": MESSAGES, "stream": True, **body},
            headers={"Authorization": f"Bearer {app.state.test_key}"},
        )
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.text.endswith("\n\n")
    return [e.removeprefix("data: ") for e in resp.text.strip().split("\n\n")]


async def test_sdk_streams_a_reply():
    client = sdk_client(make_app(("mock", MockProvider())))

    stream = await client.chat.completions.create(model="fast", messages=MESSAGES, stream=True)
    chunks = [chunk async for chunk in stream]

    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "Echo: hello there"
    assert {c.object for c in chunks} == {"chat.completion.chunk"}
    assert len({c.id for c in chunks}) == 1
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[-1].choices[0].finish_reason == "stop"
    # No usage chunk unless the client asks: code reading chunk.choices[0] must not break.
    assert all(c.choices for c in chunks)


async def test_include_usage_adds_a_final_usage_chunk():
    client = sdk_client(make_app(("mock", MockProvider())))

    stream = await client.chat.completions.create(
        model="fast", messages=MESSAGES, stream=True, stream_options={"include_usage": True}
    )
    chunks = [chunk async for chunk in stream]

    last = chunks[-1]
    assert last.choices == []
    assert last.usage.total_tokens == last.usage.prompt_tokens + last.usage.completion_tokens
    assert all(c.usage is None for c in chunks[:-1])


async def test_wire_format_ends_with_done():
    events = await raw_events(make_app(("mock", MockProvider(reply="hi"))))

    assert events[-1] == "[DONE]"
    assert all(json.loads(e)["object"] == "chat.completion.chunk" for e in events[:-1])


async def test_stream_headers_name_the_provider():
    client = sdk_client(make_app(("mock", MockProvider())))

    raw = await client.chat.completions.with_raw_response.create(
        model="fast", messages=MESSAGES, stream=True
    )

    assert raw.headers["content-type"].startswith("text/event-stream")
    assert raw.headers["cache-control"] == "no-cache"
    assert raw.headers["x-gateway-provider"] == "mock"
    assert raw.headers["x-gateway-fallback"] == "false"


async def test_failure_before_first_token_falls_back():
    # Fails right after the role-only chunk: nothing the client would see was sent yet,
    # so the router can still switch provider.
    primary = MockProvider(fail_after_chunks=1)
    backup = MockProvider(reply="from backup")
    client = sdk_client(make_app(("primary", primary), ("backup", backup)))

    raw = await client.chat.completions.with_raw_response.create(
        model="fast", messages=MESSAGES, stream=True
    )
    chunks = [chunk async for chunk in raw.parse()]

    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "from backup"
    assert {c.model for c in chunks} == {"backup-model"}
    assert raw.headers["x-gateway-provider"] == "backup"
    assert raw.headers["x-gateway-attempts"] == "3"  # the 503 is retried once, then fallback
    assert raw.headers["x-gateway-fallback"] == "true"
    # Both failed streams were closed, not left holding a connection.
    assert primary.closed_streams == 2


async def test_failure_mid_stream_raises_after_the_partial_answer():
    primary = MockProvider(reply="one two three", fail_after_chunks=3)
    backup = MockProvider()
    client = sdk_client(make_app(("primary", primary), ("backup", backup)))

    stream = await client.chat.completions.create(model="fast", messages=MESSAGES, stream=True)
    text = ""
    with pytest.raises(openai.APIError) as exc_info:
        async for chunk in stream:
            text += chunk.choices[0].delta.content or ""

    assert text == "one two "
    assert exc_info.value.body["type"] == "api_error"
    # The client already has part of an answer: no fallback, no second answer glued on.
    assert backup.calls == 0
    assert primary.closed_streams == 1


async def test_mid_stream_error_event_replaces_done():
    app = make_app(("primary", MockProvider(reply="one two three", fail_after_chunks=3)))

    events = await raw_events(app)

    assert "[DONE]" not in events
    error = json.loads(events[-1])["error"]
    assert error["message"] == "Mock provider failure (503)"


async def test_all_failing_before_first_token_is_a_json_error():
    client = sdk_client(make_app(("mock", MockProvider(failure_rate=1.0, fail_status=429))))

    with pytest.raises(openai.RateLimitError) as exc_info:
        await client.chat.completions.create(model="fast", messages=MESSAGES, stream=True)

    assert exc_info.value.body["type"] == "rate_limit_exceeded"


async def test_unknown_model_stream_raises_not_found():
    client = sdk_client(make_app(("mock", MockProvider())))

    with pytest.raises(openai.NotFoundError):
        await client.chat.completions.create(model="nope", messages=MESSAGES, stream=True)


class SilentProvider(MockProvider):
    """Answers 200, sends a role-only chunk, then ends without any text."""

    async def stream(self, request: ChatRequest):
        self.calls += 1
        yield ChatChunk(
            id="chatcmpl-x",
            created=0,
            model=request.model,
            choices=[ChunkChoice(delta=Delta(role="assistant"))],
        )


async def test_stream_that_ends_without_a_reply_falls_back():
    silent, backup = SilentProvider(), MockProvider(reply="from backup")
    router = ModelRouter(
        {"fast": [Target(provider="silent", model="m"), Target(provider="backup", model="m")]},
        {"silent": silent, "backup": backup},
        sleep=lambda _: asyncio.sleep(0),
    )

    stream = await router.stream(ChatRequest(model="fast", messages=MESSAGES))
    try:
        text = "".join([c.choices[0].delta.content or "" async for c in stream if c.choices])
    finally:
        await stream.aclose()

    assert text == "from backup"
    assert stream.fallback is True
    assert silent.calls == 2  # the 502 is retried once on the same provider


async def test_deadline_covers_the_wait_for_the_first_token():
    slow, backup = MockProvider(latency_s=1.0), MockProvider()
    router = ModelRouter(
        {"fast": [Target(provider="slow", model="m"), Target(provider="backup", model="m")]},
        {"slow": slow, "backup": backup},
        deadline_s=0.05,
    )

    with pytest.raises(ProviderError) as exc_info:
        await router.stream(ChatRequest(model="fast", messages=MESSAGES))

    assert exc_info.value.status == 504
    assert slow.closed_streams == 1
    assert backup.calls == 0


async def test_client_disconnect_closes_the_upstream_stream():
    provider = MockProvider(reply="one two three four")
    app = make_app(("mock", provider))
    body = json.dumps({"model": "fast", "messages": MESSAGES, "stream": True}).encode()
    received = [{"type": "http.request", "body": body, "more_body": False}]
    sent: list[dict] = []

    async def receive():
        if received:
            return received.pop(0)
        await asyncio.Event().wait()  # no disconnect message: the failed send is the signal

    async def send(message):
        if message["type"] == "http.response.body" and sent:
            # The client went away after the first chunk. Under ASGI 2.4 the server
            # reports that as an OSError from send().
            raise OSError("connection lost")
        if message["type"] == "http.response.body":
            sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"content-type", b"application/json"),
            (b"host", b"gateway.test"),
            (b"authorization", f"Bearer {app.state.test_key}".encode()),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("gateway.test", 80),
    }

    with pytest.raises(ClientDisconnect):
        await app(scope, receive, send)

    assert len(sent) == 1
    assert provider.closed_streams == 1
