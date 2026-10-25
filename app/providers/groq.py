import json
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager

import httpx

from app.api.schemas import ChatChunk, ChatRequest, ChatResponse
from app.providers.base import ProviderError
from app.providers.sse import data_events

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# OpenAI-style error codes for a model that doesn't exist or was retired.
_MODEL_GONE_CODES = {"model_not_found", "model_decommissioned"}


class GroqProvider:
    """Groq speaks the OpenAI format already, so this adapter mostly forwards and maps errors."""

    name = "groq"

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str | None,
        reasoning_effort: str | None = None,
        include_reasoning: bool | None = None,
    ) -> None:
        # The client is shared and owned by the app (see main.lifespan), not created per request.
        self._client = client
        self._api_key = api_key
        # Defaults for reasoning models (gpt-oss), used only when the client didn't send them.
        # Reasoning tokens count against max_tokens: at Groq's default effort a 50-token
        # request can end with an empty answer.
        self._defaults = {
            k: v
            for k, v in {
                "reasoning_effort": reasoning_effort,
                "include_reasoning": include_reasoning,
            }.items()
            if v is not None
        }

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self._check_key()
        body = self._defaults | request.model_dump(exclude_none=True)
        with _transport_errors():
            resp = await self._client.post(
                f"{GROQ_BASE_URL}/chat/completions", json=body, headers=self._headers()
            )

        if resp.status_code >= 400:
            raise _error_from_response(resp)
        try:
            return ChatResponse.model_validate(resp.json())
        except ValueError as e:
            # A 200 that isn't a chat completion (a proxy's HTML page, a changed schema).
            # ValidationError is a ValueError too. As a ProviderError it can fall back;
            # otherwise it would be an unhandled 500.
            raise ProviderError(
                502, "Groq returned an unreadable response", provider=self.name
            ) from e

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        """Groq streams OpenAI-format chunks already: parse, check and pass them on."""
        self._check_key()
        body = self._defaults | request.model_dump(exclude_none=True)
        body["stream"] = True
        # Always ask for the usage chunk, whatever the client said: Phase 5 logs usage for
        # every request. The SSE layer drops it again if the client didn't ask. Other
        # stream_options the client sent are kept.
        options = body.get("stream_options")
        body["stream_options"] = (options if isinstance(options, dict) else {}) | {
            "include_usage": True
        }

        with _transport_errors():
            async with self._client.stream(
                "POST", f"{GROQ_BASE_URL}/chat/completions", json=body, headers=self._headers()
            ) as resp:
                if resp.status_code >= 400:
                    # Errors come before the stream starts, as a normal JSON body.
                    await resp.aread()
                    raise _error_from_response(resp)
                async for data in data_events(resp):
                    if data == "[DONE]":
                        return
                    yield _parse_chunk(data)
        # The connection closed without [DONE]: the answer may be cut short. Before the first
        # token that's a retryable failure; after it, the client gets an error event.
        raise ProviderError(502, "Groq ended the stream early", provider=self.name)

    def _check_key(self) -> None:
        if not self._api_key:
            raise ProviderError(
                500,
                "GROQ_API_KEY is not set",
                type="server_error",
                provider=self.name,
                retry_same_provider=False,
            )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}


@contextmanager
def _transport_errors() -> Iterator[None]:
    """Network failures as ProviderErrors. Mid-stream, a timeout is the gap between chunks."""
    try:
        yield
    except httpx.TimeoutException as e:
        raise ProviderError(504, "Groq timed out", type="timeout", provider="groq") from e
    except httpx.TransportError as e:
        raise ProviderError(
            502, f"Could not reach Groq: {e}", type="api_connection_error", provider="groq"
        ) from e


def _parse_chunk(data: str) -> ChatChunk:
    try:
        payload = json.loads(data)
        if isinstance(payload, dict) and payload.get("error"):
            # An error sent inside the stream, after the 200. Treated as a 502: the request
            # was accepted, so this is the provider failing, and another one may do better.
            err = payload["error"] if isinstance(payload["error"], dict) else {}
            raise ProviderError(
                502,
                err.get("message") or "Groq sent an error mid-stream",
                type=err.get("type") or "api_error",
                code=err.get("code"),
                provider="groq",
            )
        # Groq's own fields (x_groq: request id, timings, a second copy of usage) are dropped
        # by the model, like in non-streamed replies.
        return ChatChunk.model_validate(payload)
    except ValueError as e:
        raise ProviderError(502, "Groq sent an unreadable stream chunk", provider="groq") from e


def _error_from_response(resp: httpx.Response) -> ProviderError:
    # Groq's error bodies are OpenAI-shaped: {"error": {"message", "type", "code", "param"}}.
    try:
        err = resp.json().get("error") or {}
    except ValueError:
        err = {}
    if resp.status_code in (401, 403):
        # The gateway's own Groq key was rejected. Passing a 401 on would tell the client
        # *their* key is wrong, so report it as an upstream failure instead.
        return ProviderError(
            502,
            "Upstream provider rejected the gateway's credentials",
            type="api_error",
            provider="groq",
            retry_same_provider=False,
        )
    if resp.status_code == 404 or err.get("code") in _MODEL_GONE_CODES:
        # The registry names a model Groq doesn't serve (Groq retires models often). That's
        # the gateway's config, not the client's request, so let the next provider try.
        return ProviderError(
            502,
            "Upstream provider does not serve the configured model",
            type="api_error",
            code=err.get("code"),
            provider="groq",
            retry_same_provider=False,
        )
    return ProviderError(
        resp.status_code,
        err.get("message") or resp.text or f"Groq returned {resp.status_code}",
        type=err.get("type") or "api_error",
        code=err.get("code"),
        param=err.get("param"),
        provider="groq",
    )
