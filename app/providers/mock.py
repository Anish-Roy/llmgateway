import asyncio
import random

from app.api.schemas import ChatMessage, ChatRequest, ChatResponse, Choice, Usage
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
    ) -> None:
        # reply=None echoes the last user message, which makes tests easy to read.
        self.reply = reply
        self.latency_s = latency_s
        self.failure_rate = failure_rate
        self.fail_status = fail_status
        self._rng = random.Random(seed)
        self.calls = 0

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        if self.failure_rate and self._rng.random() < self.failure_rate:
            raise ProviderError(
                self.fail_status,
                f"Mock provider failure ({self.fail_status})",
                type=_error_type(self.fail_status),
                provider=self.name,
            )

        text = self.reply if self.reply is not None else _echo(request)
        # Rough token counts (about 4 characters a token) so usage fields aren't zero.
        prompt_tokens = sum(len(m.content or "") for m in request.messages) // 4 + 1
        completion_tokens = len(text) // 4 + 1
        return ChatResponse(
            model=request.model,
            choices=[Choice(message=ChatMessage(role="assistant", content=text))],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
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
