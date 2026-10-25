import asyncio
import random
import re
import time
import uuid
from collections.abc import AsyncIterator

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


class MockProvider:
    """A fake provider for tests and demos: canned replies, adjustable latency and failures."""

    name = "mock"

    def __init__(
        self,
        reply: str | None = None,
        latency_s: float = 0.0,
        failure_rate: float = 0.0,
        fail_status: int = 503,
        seed: int | None = None,
        chunk_latency_s: float = 0.0,
        fail_after_chunks: int | None = None,
    ) -> None:
        # reply=None echoes the last user message, which makes tests easy to read.
        self.reply = reply
        self.latency_s = latency_s
        self.failure_rate = failure_rate
        self.fail_status = fail_status
        self._rng = random.Random(seed)
        # Streaming only: the wait between chunks, and a failure after this many chunks were
        # sent. The first chunk is role-only, so 1 fails before any text (fallback still
        # possible) and 2 or more fails mid-answer.
        self.chunk_latency_s = chunk_latency_s
        self.fail_after_chunks = fail_after_chunks
        self.calls = 0
        # Every request it was sent, as it got it. Tests check what reached the provider.
        self.requests: list[ChatRequest] = []
        # Streams that were closed, finished or not. Tests check upstream connections are freed.
        self.closed_streams = 0

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        await self._start()
        text = self._text(request)
        return ChatResponse(
            model=request.model,
            choices=[Choice(message=ChatMessage(role="assistant", content=text))],
            usage=_usage(request, text),
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        self.requests.append(request)
        try:
            await self._start()
            text = self._text(request)
            meta = {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "created": int(time.time()),
                "model": request.model,
            }
            # Shaped like OpenAI's (and Groq's) streams: a role-only chunk, the text a word at a
            # time, an empty delta with finish_reason, then usage with no choices.
            chunks = [ChatChunk(**meta, choices=[ChunkChoice(delta=Delta(role="assistant"))])]
            chunks += [
                ChatChunk(**meta, choices=[ChunkChoice(delta=Delta(content=word))])
                for word in re.findall(r"\S+\s*", text)
            ]
            chunks.append(ChatChunk(**meta, choices=[ChunkChoice(finish_reason="stop")]))
            chunks.append(ChatChunk(**meta, usage=_usage(request, text)))

            for sent, chunk in enumerate(chunks):
                if sent == self.fail_after_chunks:
                    raise self._failure()
                if sent and self.chunk_latency_s:
                    await asyncio.sleep(self.chunk_latency_s)
                yield chunk
        finally:
            self.closed_streams += 1

    async def _start(self) -> None:
        self.calls += 1
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        if self.failure_rate and self._rng.random() < self.failure_rate:
            raise self._failure()

    def _failure(self) -> ProviderError:
        return ProviderError(
            self.fail_status,
            f"Mock provider failure ({self.fail_status})",
            type=_error_type(self.fail_status),
            provider=self.name,
        )

    def _text(self, request: ChatRequest) -> str:
        return self.reply if self.reply is not None else _echo(request)


def _usage(request: ChatRequest, text: str) -> Usage:
    # Rough token counts (about 4 characters a token) so usage fields aren't zero.
    prompt_tokens = sum(len(m.content or "") for m in request.messages) // 4 + 1
    completion_tokens = len(text) // 4 + 1
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _echo(request: ChatRequest) -> str:
    last_user = next((m for m in reversed(request.messages) if m.role == "user"), None)
    return f"Echo: {last_user.content if last_user else ''}"


def _error_type(status: int) -> str:
    if status == 429:
        return "rate_limit_exceeded"
    if 400 <= status < 500:
        return "invalid_request_error"
    return "api_error"
