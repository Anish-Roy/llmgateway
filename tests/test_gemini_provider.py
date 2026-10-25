"""The Gemini adapter: OpenAI <-> generateContent translation, over a fake transport."""

import json

import httpx
import pytest

from app.api.schemas import ChatRequest
from app.providers.base import ProviderError
from app.providers.gemini import GeminiProvider, from_gemini, to_gemini

GEMINI_OK = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [{"text": "thinking about it", "thought": True}, {"text": "Hello!"}],
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 5,
        "candidatesTokenCount": 2,
        "thoughtsTokenCount": 10,
        "totalTokenCount": 17,
    },
    "modelVersion": "gemini-3.5-flash-lite",
    "responseId": "abc123",
}

# What Gemini actually returned for a made-up key (checked 2026-10-25): a 400, not a 401.
GEMINI_BAD_KEY = {
    "error": {
        "code": 400,
        "message": "API key not valid. Please pass a valid API key.",
        "status": "INVALID_ARGUMENT",
        "details": [
            {
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": "API_KEY_INVALID",
                "domain": "googleapis.com",
            }
        ],
    }
}


def req(**fields) -> ChatRequest:
    data = {"model": "gemini-3.5-flash-lite", "messages": [{"role": "user", "content": "hi"}]}
    return ChatRequest.model_validate(data | fields)


def provider_with(handler, api_key: str | None = "AIza-test", **kwargs) -> GeminiProvider:
    return GeminiProvider(
        httpx.AsyncClient(transport=httpx.MockTransport(handler)), api_key, **kwargs
    )


# --- request translation ---


def test_system_messages_become_one_system_instruction():
    body = to_gemini(
        req(
            messages=[
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hi"},
                {"role": "developer", "content": "Answer in English."},
            ]
        )
    )

    assert body["systemInstruction"] == {"parts": [{"text": "Be brief.\n\nAnswer in English."}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]


def test_assistant_becomes_model_and_repeated_roles_are_merged():
    body = to_gemini(
        req(
            messages=[
                {"role": "user", "content": "a"},
                {"role": "user", "content": "b"},
                {"role": "assistant", "content": "c"},
                {"role": "user", "content": "d"},
            ]
        )
    )

    assert body["contents"] == [
        {"role": "user", "parts": [{"text": "a"}, {"text": "b"}]},
        {"role": "model", "parts": [{"text": "c"}]},
        {"role": "user", "parts": [{"text": "d"}]},
    ]


def test_generation_settings_are_mapped():
    body = to_gemini(
        req(temperature=0.2, max_tokens=50, top_p=0.9, stop="END", seed=7, n=1),
        thinking_level="low",
    )

    assert body["generationConfig"] == {
        "temperature": 0.2,
        "maxOutputTokens": 50,
        "topP": 0.9,
        "stopSequences": ["END"],
        "seed": 7,
        "thinkingConfig": {"thinkingLevel": "low"},
    }


def test_max_completion_tokens_is_accepted_too():
    body = to_gemini(req(max_completion_tokens=30))

    assert body["generationConfig"] == {"maxOutputTokens": 30}


def test_no_settings_means_no_generation_config():
    assert "generationConfig" not in to_gemini(req())


@pytest.mark.parametrize(
    ("fields", "param"),
    [
        ({"tools": [{"type": "function", "function": {"name": "f"}}]}, "tools"),
        ({"logprobs": True}, "logprobs"),
        ({"n": 2}, "n"),
        ({"messages": [{"role": "tool", "content": "42", "tool_call_id": "x"}]}, "messages.0"),
        ({"messages": [{"role": "system", "content": "only a system prompt"}]}, "messages"),
    ],
)
def test_unsupported_input_is_an_honest_400(fields, param):
    with pytest.raises(ProviderError) as exc_info:
        to_gemini(req(**fields))

    assert exc_info.value.status == 400
    assert exc_info.value.param == param
    assert not exc_info.value.retryable
    # "Gemini can't do this", not "the request is bad": the router tries the next provider.
    assert exc_info.value.provider_specific


# --- response translation ---


def test_response_is_translated_to_openai_shape():
    resp = from_gemini(GEMINI_OK, "gemini-3.5-flash-lite")

    assert resp.id == "chatcmpl-abc123"
    assert resp.model == "gemini-3.5-flash-lite"
    # The thought part is left out of the answer.
    assert resp.choices[0].message.content == "Hello!"
    assert resp.choices[0].message.role == "assistant"
    assert resp.choices[0].finish_reason == "stop"


def test_thinking_tokens_count_as_completion_tokens():
    usage = from_gemini(GEMINI_OK, "m").usage

    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (5, 12, 17)


@pytest.mark.parametrize(
    ("gemini", "openai"),
    [("MAX_TOKENS", "length"), ("SAFETY", "content_filter"), ("OTHER", "stop")],
)
def test_finish_reasons_are_mapped(gemini, openai):
    data = GEMINI_OK | {"candidates": [GEMINI_OK["candidates"][0] | {"finishReason": gemini}]}

    assert from_gemini(data, "m").choices[0].finish_reason == openai


def test_blocked_prompt_is_a_content_filter_reply_not_a_crash():
    data = {"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {"promptTokenCount": 4}}

    resp = from_gemini(data, "gemini-3.5-flash-lite")

    assert resp.choices[0].finish_reason == "content_filter"
    assert resp.choices[0].message.content == ""
    assert resp.model == "gemini-3.5-flash-lite"


def test_candidate_with_no_parts_is_an_empty_reply():
    # E.g. thinking used up all of maxOutputTokens before any answer was written.
    data = {"candidates": [{"content": {"role": "model"}, "finishReason": "MAX_TOKENS"}]}

    resp = from_gemini(data, "m")

    assert resp.choices[0].message.content == ""
    assert resp.choices[0].finish_reason == "length"


# --- HTTP ---


async def test_request_goes_to_the_model_url_with_the_key_in_a_header():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-goog-api-key")
        seen["body"] = json.loads(request.read())
        return httpx.Response(200, json=GEMINI_OK)

    resp = await provider_with(handler).chat(req())

    assert seen["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.5-flash-lite:generateContent"
    )
    assert "key=" not in seen["url"]
    assert seen["key"] == "AIza-test"
    assert seen["body"]["contents"][0]["parts"] == [{"text": "hi"}]
    assert resp.choices[0].message.content == "Hello!"


async def test_bad_key_400_becomes_a_502_that_falls_back():
    provider = provider_with(lambda r: httpx.Response(400, json=GEMINI_BAD_KEY))

    with pytest.raises(ProviderError) as exc_info:
        await provider.chat(req())

    err = exc_info.value
    assert err.status == 502
    assert err.retryable
    assert not err.retry_same_provider


async def test_unknown_model_404_becomes_a_502_that_falls_back():
    # A 404 body may not be JSON (curl over HTTP/2 got an empty one); the mapping mustn't care.
    provider = provider_with(lambda r: httpx.Response(404, text=""))

    with pytest.raises(ProviderError) as exc_info:
        await provider.chat(req())

    assert exc_info.value.status == 502
    assert not exc_info.value.retry_same_provider


@pytest.mark.parametrize(
    ("status", "gemini_status", "retryable"),
    [
        (400, "INVALID_ARGUMENT", False),
        (429, "RESOURCE_EXHAUSTED", True),
        (500, "INTERNAL", True),
        (503, "UNAVAILABLE", True),
    ],
)
async def test_other_errors_keep_their_status(status, gemini_status, retryable):
    body = {"error": {"code": status, "message": "nope", "status": gemini_status}}
    provider = provider_with(lambda r: httpx.Response(status, json=body))

    with pytest.raises(ProviderError) as exc_info:
        await provider.chat(req())

    err = exc_info.value
    assert (err.status, err.message, err.code) == (status, "nope", gemini_status)
    assert err.retryable is retryable
    assert err.provider == "gemini"


async def test_429_keeps_the_openai_rate_limit_type():
    body = {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}}

    with pytest.raises(ProviderError) as exc_info:
        await provider_with(lambda r: httpx.Response(429, json=body)).chat(req())

    assert exc_info.value.type == "rate_limit_exceeded"


async def test_unreadable_success_body_becomes_retryable_502():
    with pytest.raises(ProviderError) as exc_info:
        await provider_with(lambda r: httpx.Response(200, text="<html>")).chat(req())

    assert exc_info.value.status == 502
    assert exc_info.value.retryable


async def test_timeout_becomes_504_without_same_provider_retry():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ProviderError) as exc_info:
        await provider_with(handler).chat(req())

    assert exc_info.value.status == 504
    assert not exc_info.value.retry_same_provider


async def test_missing_api_key():
    provider = provider_with(lambda r: httpx.Response(200, json=GEMINI_OK), api_key=None)

    with pytest.raises(ProviderError) as exc_info:
        await provider.chat(req())

    assert exc_info.value.status == 500
