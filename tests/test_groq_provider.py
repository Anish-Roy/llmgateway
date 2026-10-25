"""The Groq adapter against a fake transport. Never calls the real API."""

import json

import httpx
import pytest

from app.api.schemas import ChatRequest
from app.providers.base import ProviderError
from app.providers.groq import GroqProvider

REQUEST = ChatRequest(
    model="llama-3.1-8b-instant", messages=[{"role": "user", "content": "hi"}], temperature=0
)

GROQ_OK = {
    "id": "chatcmpl-abc",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "llama-3.1-8b-instant",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7, "queue_time": 0.01},
    "x_groq": {"id": "req_1"},
}


def provider_with(handler, api_key: str | None = "gsk-test") -> GroqProvider:
    return GroqProvider(httpx.AsyncClient(transport=httpx.MockTransport(handler)), api_key)


async def test_forwards_request_and_parses_reply():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = request.read()
        return httpx.Response(200, json=GROQ_OK)

    resp = await provider_with(handler).chat(REQUEST)

    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["auth"] == "Bearer gsk-test"
    assert b'"stream"' in seen["body"] and b'"max_tokens"' not in seen["body"]
    assert resp.choices[0].message.content == "Hello!"
    assert resp.usage.total_tokens == 7


@pytest.mark.parametrize(
    ("status", "retryable"), [(400, False), (429, True), (500, True), (503, True)]
)
async def test_error_status_becomes_provider_error(status, retryable):
    body = {"error": {"message": "nope", "type": "some_type", "code": "some_code"}}
    provider = provider_with(lambda r: httpx.Response(status, json=body))

    with pytest.raises(ProviderError) as exc_info:
        await provider.chat(REQUEST)

    err = exc_info.value
    assert (err.status, err.message, err.type) == (status, "nope", "some_type")
    assert err.code == "some_code"
    assert err.retryable is retryable
    assert err.retry_same_provider is (status in (500, 503))


@pytest.mark.parametrize("status", [401, 403])
async def test_rejected_gateway_key_is_not_reported_as_client_auth_error(status):
    provider = provider_with(lambda r: httpx.Response(status, json={"error": {"message": "bad"}}))

    with pytest.raises(ProviderError) as exc_info:
        await provider.chat(REQUEST)

    assert exc_info.value.status == 502
    # Worth trying another provider, but retrying Groq with the same bad key is pointless.
    assert exc_info.value.retryable
    assert not exc_info.value.retry_same_provider


async def test_unmodelled_fields_are_forwarded():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.read())
        return httpx.Response(200, json=GROQ_OK)

    request = ChatRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stop": ["\n"],
            "top_p": 0.5,
        }
    )
    await provider_with(handler).chat(request)

    assert seen["body"]["stop"] == ["\n"]
    assert seen["body"]["top_p"] == 0.5


async def test_non_json_error_body():
    provider = provider_with(lambda r: httpx.Response(502, text="Bad Gateway"))

    with pytest.raises(ProviderError) as exc_info:
        await provider.chat(REQUEST)

    assert exc_info.value.status == 502
    assert exc_info.value.message == "Bad Gateway"


async def test_timeout_becomes_504():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ProviderError) as exc_info:
        await provider_with(handler).chat(REQUEST)

    assert exc_info.value.status == 504
    assert exc_info.value.retryable
    assert not exc_info.value.retry_same_provider


async def test_connection_failure_becomes_502():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderError) as exc_info:
        await provider_with(handler).chat(REQUEST)

    assert exc_info.value.status == 502


async def test_missing_api_key():
    provider = provider_with(lambda r: httpx.Response(200, json=GROQ_OK), api_key=None)

    with pytest.raises(ProviderError) as exc_info:
        await provider.chat(REQUEST)

    assert exc_info.value.status == 500
