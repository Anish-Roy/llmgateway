"""Contract tests: the official openai SDK, talking to the app in memory, against the mock."""

import httpx2
import openai
import pytest

from app.api.routes import get_model_router
from app.main import create_app
from app.providers.mock import MockProvider
from app.routing.registry import Target
from app.routing.router import ModelRouter


def make_client(provider: MockProvider) -> openai.AsyncOpenAI:
    app = create_app()
    model_router = ModelRouter(
        {"mock-model": [Target(provider="mock", model="mock-model")]}, {"mock": provider}
    )
    app.dependency_overrides[get_model_router] = lambda: model_router
    # The openai SDK is built on httpx2. ASGITransport calls the app directly: no server or network.
    http_client = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app))
    return openai.AsyncOpenAI(
        api_key="unused-until-phase-4",
        base_url="http://gateway.test/v1",
        http_client=http_client,
        max_retries=0,
    )


async def test_sdk_gets_a_reply():
    client = make_client(MockProvider())

    resp = await client.chat.completions.create(
        model="mock-model",
        messages=[
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "hello"},
        ],
        temperature=0,
        max_tokens=20,
        top_p=1,  # not modelled by the gateway: must be accepted and forwarded, not rejected
    )

    assert resp.object == "chat.completion"
    assert resp.id.startswith("chatcmpl-")
    assert resp.model == "mock-model"
    assert resp.choices[0].message.role == "assistant"
    assert resp.choices[0].message.content == "Echo: hello"
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage.total_tokens == resp.usage.prompt_tokens + resp.usage.completion_tokens


async def test_provider_429_raises_rate_limit_error():
    client = make_client(MockProvider(failure_rate=1.0, fail_status=429))

    with pytest.raises(openai.RateLimitError) as exc_info:
        await client.chat.completions.create(
            model="mock-model", messages=[{"role": "user", "content": "hi"}]
        )

    assert exc_info.value.body["type"] == "rate_limit_exceeded"


async def test_provider_5xx_raises_server_error():
    client = make_client(MockProvider(failure_rate=1.0, fail_status=503))

    with pytest.raises(openai.InternalServerError):
        await client.chat.completions.create(
            model="mock-model", messages=[{"role": "user", "content": "hi"}]
        )


async def test_invalid_body_raises_bad_request_with_param():
    client = make_client(MockProvider())

    with pytest.raises(openai.BadRequestError) as exc_info:
        await client.chat.completions.create(
            model="mock-model", messages=[{"role": "user", "content": "hi"}], temperature=5
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.body["param"] == "temperature"
    assert exc_info.value.body["type"] == "invalid_request_error"


async def test_empty_messages_is_bad_request():
    client = make_client(MockProvider())

    with pytest.raises(openai.BadRequestError):
        await client.chat.completions.create(model="mock-model", messages=[])


async def test_stream_true_is_rejected_until_phase_3():
    provider = MockProvider()
    client = make_client(provider)

    with pytest.raises(openai.BadRequestError) as exc_info:
        await client.chat.completions.create(
            model="mock-model", messages=[{"role": "user", "content": "hi"}], stream=True
        )

    assert exc_info.value.body["param"] == "stream"
    assert provider.calls == 0


async def test_unknown_model_raises_not_found():
    provider = MockProvider()
    client = make_client(provider)

    with pytest.raises(openai.NotFoundError) as exc_info:
        await client.chat.completions.create(
            model="gpt-nope", messages=[{"role": "user", "content": "hi"}]
        )

    assert exc_info.value.body["code"] == "model_not_found"
    assert exc_info.value.body["param"] == "model"
    assert provider.calls == 0


async def test_response_names_the_provider_that_served_it():
    client = make_client(MockProvider())

    raw = await client.chat.completions.with_raw_response.create(
        model="mock-model", messages=[{"role": "user", "content": "hi"}]
    )

    assert raw.headers["x-gateway-provider"] == "mock"
