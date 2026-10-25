"""Gemini adapter: translates OpenAI chat requests to Gemini's native `generateContent` and back.

Field names checked against ai.google.dev on 2026-10-25 (see docs/decisions.md).
"""

import json
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Any, Literal

import httpx

from app.api.schemas import (
    ChatChunk,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Choice,
    ChunkChoice,
    Delta,
    Usage,
)
from app.providers.base import ProviderError
from app.providers.sse import data_events

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

ThinkingLevel = Literal["minimal", "low", "medium", "high"]

# OpenAI request fields with a Gemini equivalent (checked live 2026-10-25). Anything else in
# the request's extras gets an honest 400 instead of being silently dropped (same rule as
# Phase 1). Not the penalties: Gemini 3.x answers "Penalty is not enabled for this model", and
# a raw upstream 400 would block fallback where our own provider-specific 400 doesn't.
_GENERATION_FIELDS = {
    "top_p": "topP",
    "seed": "seed",
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
        self._check_key()
        body = to_gemini(request, self._thinking_level)
        with _transport_errors():
            resp = await self._client.post(
                f"{GEMINI_BASE_URL}/models/{request.model}:generateContent",
                json=body,
                headers=self._headers(),
            )

        if resp.status_code >= 400:
            raise _error_from_response(resp)
        try:
            return from_gemini(resp.json(), request.model)
        except (ValueError, KeyError, IndexError, TypeError) as e:
            # A 200 we can't read. As a ProviderError it can fall back instead of being a 500.
            raise ProviderError(
                502, "Gemini returned an unreadable response", provider=self.name
            ) from e

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        """streamGenerateContent events -> OpenAI chunks.

        With `alt=sse` each event is a whole GenerateContentResponse holding the next piece of
        text. There's no [DONE]: the event with a finishReason is the last one.
        """
        self._check_key()
        body = to_gemini(request, self._thinking_level)
        translator = _StreamTranslator(request.model)
        with _transport_errors():
            async with self._client.stream(
                "POST",
                f"{GEMINI_BASE_URL}/models/{request.model}:streamGenerateContent",
                # Without alt=sse the reply is one JSON array, sent bit by bit.
                params={"alt": "sse"},
                json=body,
                headers=self._headers(),
            ) as resp:
                if resp.status_code >= 400:
                    # Errors come before the stream starts, as a normal JSON body (even though
                    # the content-type says text/event-stream).
                    await resp.aread()
                    raise _error_from_response(resp)
                async for data in data_events(resp):
                    for chunk in translator.feed(_parse_event(data)):
                        yield chunk
        if not translator.finished:
            # The connection closed before the finishReason event: the answer may be cut short.
            raise ProviderError(502, "Gemini ended the stream early", provider=self.name)
        if translator.usage is not None:
            yield translator.usage_chunk()

    def _check_key(self) -> None:
        if not self._api_key:
            raise ProviderError(
                500,
                "GEMINI_API_KEY is not set",
                type="server_error",
                provider=self.name,
                retry_same_provider=False,
            )

    def _headers(self) -> dict[str, str]:
        # A header, not ?key=, so the key never shows up in URLs or access logs.
        return {"x-goog-api-key": self._api_key or ""}


@contextmanager
def _transport_errors() -> Iterator[None]:
    """Network failures as ProviderErrors. Mid-stream, a timeout is the gap between events."""
    try:
        yield
    except httpx.TimeoutException as e:
        raise ProviderError(504, "Gemini timed out", type="timeout", provider="gemini") from e
    except httpx.TransportError as e:
        raise ProviderError(
            502, f"Could not reach Gemini: {e}", type="api_connection_error", provider="gemini"
        ) from e


class _StreamTranslator:
    """Turns Gemini stream events into OpenAI chunks, keeping what spans the whole stream."""

    def __init__(self, requested_model: str) -> None:
        self._id = f"chatcmpl-{uuid.uuid4().hex}"
        self._created = int(time.time())
        self._model = requested_model
        self._sent_role = False
        self.finished = False
        # Every event repeats the running totals (checked live), so the last one wins.
        self.usage: Usage | None = None

    def feed(self, event: dict[str, Any]) -> list[ChatChunk]:
        if not self._sent_role:
            # Ids and model come from the first event and stay the same for the whole stream.
            if event.get("responseId"):
                self._id = f"chatcmpl-{event['responseId']}"
            self._model = event.get("modelVersion") or self._model
        if event.get("usageMetadata"):
            self.usage = _usage(event["usageMetadata"])

        candidates = event.get("candidates") or []
        if not candidates:
            if (event.get("promptFeedback") or {}).get("blockReason"):
                # The prompt itself was blocked: an empty answer, as in from_gemini.
                self.finished = True
                return [self._chunk(finish_reason="content_filter")]
            return []

        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        # Thought parts aren't part of the answer. An event with only thoughts (or only the
        # empty part carrying a thoughtSignature) sends no chunk, so it can't commit the router.
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        reason = candidate.get("finishReason")
        chunks = []
        if text:
            chunks.append(self._chunk(content=text))
        if reason:
            self.finished = True
            chunks.append(self._chunk(finish_reason=_FINISH_REASONS.get(reason, "stop")))
        return chunks

    def usage_chunk(self) -> ChatChunk:
        return ChatChunk(id=self._id, created=self._created, model=self._model, usage=self.usage)

    def _chunk(self, content: str | None = None, finish_reason: str | None = None) -> ChatChunk:
        # Gemini never sends a role; OpenAI puts it on the first chunk.
        delta = Delta(role=None if self._sent_role else "assistant", content=content)
        self._sent_role = True
        return ChatChunk(
            id=self._id,
            created=self._created,
            model=self._model,
            choices=[ChunkChoice(delta=delta, finish_reason=finish_reason)],
            # The running total so far, for the gateway: if the stream is cut, usage is still
            # known. The SSE layer doesn't send usage on chunks with choices.
            usage=self.usage,
        )


def _parse_event(data: str) -> dict[str, Any]:
    try:
        event = json.loads(data)
    except ValueError as e:
        raise ProviderError(502, "Gemini sent an unreadable stream event", provider="gemini") from e
    if not isinstance(event, dict):
        raise ProviderError(502, "Gemini sent an unreadable stream event", provider="gemini")
    if event.get("error"):
        # An error after the 200. The request was accepted, so treat it as the provider
        # failing: before the first token, another provider gets a chance.
        err = event["error"] if isinstance(event["error"], dict) else {}
        raise ProviderError(
            502,
            err.get("message") or "Gemini sent an error mid-stream",
            code=err.get("status"),
            provider="gemini",
        )
    return event


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
        if value is None or field in ("max_completion_tokens", "stream_options"):
            # stream_options: usage comes with every stream anyway, and the SSE layer decides
            # whether the client sees it. Rejecting it would make every include_usage client
            # skip Gemini.
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
        usage=_usage(data.get("usageMetadata") or {}),
    )
    if data.get("responseId"):
        response.id = f"chatcmpl-{data['responseId']}"
    return response


def _usage(metadata: dict[str, Any]) -> Usage:
    prompt_tokens = metadata.get("promptTokenCount", 0)
    # Thinking tokens are billed as output, so they count as completion tokens. Leaving them
    # out would make Phase 5's cost numbers too low.
    completion_tokens = metadata.get("candidatesTokenCount", 0) + metadata.get(
        "thoughtsTokenCount", 0
    )
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _bad_request(message: str, param: str) -> ProviderError:
    return ProviderError(
        400,
        message,
        type="invalid_request_error",
        param=param,
        provider="gemini",
        provider_specific=True,
    )


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
