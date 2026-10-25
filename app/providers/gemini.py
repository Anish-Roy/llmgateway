"""Gemini adapter: translates OpenAI chat requests to Gemini's native `generateContent` and back.

Field names checked against ai.google.dev on 2026-10-25 (see docs/decisions.md).
"""

from typing import Any, Literal

import httpx

from app.api.schemas import ChatMessage, ChatRequest, ChatResponse, Choice, Usage
from app.providers.base import ProviderError

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

ThinkingLevel = Literal["minimal", "low", "medium", "high"]

# OpenAI request fields with a Gemini equivalent. Anything else in the request's extras gets an
# honest 400 instead of being silently dropped (same rule as Phase 1).
_GENERATION_FIELDS = {
    "top_p": "topP",
    "seed": "seed",
    "presence_penalty": "presencePenalty",
    "frequency_penalty": "frequencyPenalty",
}

_FINISH_REASONS = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
}

# error.details[].reason values meaning the *gateway's* key is bad, not the client's request.
# They become a 502 so the router tries the next provider.
_CONFIG_ERROR_REASONS = {"API_KEY_INVALID", "API_KEY_SERVICE_BLOCKED"}


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str | None,
        thinking_level: ThinkingLevel | None = None,
    ) -> None:
        self._client = client
        self._api_key = api_key
        # None leaves the model's default. Thinking tokens count against maxOutputTokens, so a
        # small max_tokens can be used up before any answer is written.
        self._thinking_level = thinking_level

    async def chat(self, request: ChatRequest) -> ChatResponse:
        if not self._api_key:
            raise ProviderError(
                500,
                "GEMINI_API_KEY is not set",
                type="server_error",
                provider=self.name,
                retry_same_provider=False,
            )

        body = to_gemini(request, self._thinking_level)
        try:
            resp = await self._client.post(
                f"{GEMINI_BASE_URL}/models/{request.model}:generateContent",
                json=body,
                # A header, not ?key=, so the key never shows up in URLs or access logs.
                headers={"x-goog-api-key": self._api_key},
            )
        except httpx.TimeoutException as e:
            raise ProviderError(504, "Gemini timed out", type="timeout", provider=self.name) from e
        except httpx.TransportError as e:
            raise ProviderError(
                502,
                f"Could not reach Gemini: {e}",
                type="api_connection_error",
                provider=self.name,
            ) from e

        if resp.status_code >= 400:
            raise _error_from_response(resp)
        try:
            return from_gemini(resp.json(), request.model)
        except (ValueError, KeyError, IndexError, TypeError) as e:
            # A 200 we can't read. As a ProviderError it can fall back instead of being a 500.
            raise ProviderError(
                502, "Gemini returned an unreadable response", provider=self.name
            ) from e


def to_gemini(request: ChatRequest, thinking_level: ThinkingLevel | None = None) -> dict[str, Any]:
    """OpenAI chat request -> generateContent body."""
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []

    for i, msg in enumerate(request.messages):
        if msg.role in ("system", "developer"):
            # Gemini has one system instruction, outside the conversation. Wherever system
            # messages appear, they're joined in order.
            if msg.content:
                system_parts.append(msg.content)
            continue
        if msg.role == "tool" or (msg.model_extra or {}).get("tool_calls"):
            raise _bad_request("Tool calls are not supported by this provider", f"messages.{i}")
        role = "model" if msg.role == "assistant" else "user"
        text = msg.content or ""
        if contents and contents[-1]["role"] == role:
            # Gemini expects user and model turns to alternate: merge back-to-back turns.
            contents[-1]["parts"].append({"text": text})
        else:
            contents.append({"role": role, "parts": [{"text": text}]})

    if not contents:
        raise _bad_request("At least one user or assistant message is required", "messages")

    config: dict[str, Any] = {}
    if request.temperature is not None:
        config["temperature"] = request.temperature
    max_tokens = request.max_tokens or (request.model_extra or {}).get("max_completion_tokens")
    if max_tokens is not None:
        config["maxOutputTokens"] = max_tokens
    for field, value in (request.model_extra or {}).items():
        if value is None or field == "max_completion_tokens":
            continue
        if field in _GENERATION_FIELDS:
            config[_GENERATION_FIELDS[field]] = value
        elif field == "stop":
            config["stopSequences"] = [value] if isinstance(value, str) else value
        elif field == "n":
            if value != 1:
                raise _bad_request("Only n=1 is supported by this provider", "n")
        else:
            raise _bad_request(f"'{field}' is not supported by this provider", field)
    if thinking_level:
        config["thinkingConfig"] = {"thinkingLevel": thinking_level}

    body: dict[str, Any] = {"contents": contents}
    if system_parts:
        body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
    if config:
        body["generationConfig"] = config
    return body


def from_gemini(data: dict[str, Any], requested_model: str) -> ChatResponse:
    """generateContent response -> OpenAI chat completion."""
    usage = data.get("usageMetadata") or {}
    prompt_tokens = usage.get("promptTokenCount", 0)
    # Thinking tokens are billed as output, so they count as completion tokens. Leaving them
    # out would make Phase 5's cost numbers too low.
    completion_tokens = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)

    candidates = data.get("candidates") or []
    if candidates:
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        # Thought summaries are flagged `thought: true`; they aren't part of the answer.
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        finish_reason = _FINISH_REASONS.get(candidate.get("finishReason", "STOP"), "stop")
    else:
        # No candidates: the prompt itself was blocked (promptFeedback.blockReason).
        text, finish_reason = "", "content_filter"

    response = ChatResponse(
        model=data.get("modelVersion") or requested_model,
        choices=[
            Choice(
                message=ChatMessage(role="assistant", content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    if data.get("responseId"):
        response.id = f"chatcmpl-{data['responseId']}"
    return response


def _bad_request(message: str, param: str) -> ProviderError:
    return ProviderError(400, message, type="invalid_request_error", param=param, provider="gemini")


def _error_from_response(resp: httpx.Response) -> ProviderError:
    # Gemini's error body: {"error": {"code", "message", "status", "details": [{"reason"...}]}}.
    try:
        err = resp.json().get("error") or {}
    except ValueError:
        err = {}
    reasons = {d.get("reason") for d in err.get("details") or [] if isinstance(d, dict)}
    status = resp.status_code

    if status in (401, 403) or reasons & _CONFIG_ERROR_REASONS:
        # A bad key comes back as a *400* INVALID_ARGUMENT with reason API_KEY_INVALID. Passed
        # on, it would tell the client their request is invalid, and fallback would never run.
        return ProviderError(
            502,
            "Upstream provider rejected the gateway's credentials",
            provider="gemini",
            retry_same_provider=False,
        )
    if status == 404:
        # The registry names a model Gemini doesn't serve (retired or mistyped): our config is
        # wrong, not the client's request.
        return ProviderError(
            502,
            "Upstream provider does not serve the configured model",
            provider="gemini",
            retry_same_provider=False,
        )
    return ProviderError(
        status,
        err.get("message") or resp.text or f"Gemini returned {status}",
        type=_error_type(status),
        code=err.get("status"),
        provider="gemini",
    )


def _error_type(status: int) -> str:
    if status == 429:
        return "rate_limit_exceeded"
    if status == 504:
        return "timeout"
    if 400 <= status < 500:
        return "invalid_request_error"
    return "api_error"
