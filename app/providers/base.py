from typing import Protocol

from app.api.schemas import ChatRequest, ChatResponse

# Statuses worth trying on another provider. A 400 would fail the same way anywhere.
RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}
# Of those, the ones worth one more try on the *same* provider: quick, usually transient errors.
# Not 429 (its quota won't refill in a few hundred ms) or timeouts (a second wait would use up
# the time the next provider needs).
RETRY_SAME_STATUSES = {500, 502, 503}


class ProviderError(Exception):
    """A provider call failed. Carries what the gateway needs to answer in OpenAI's error format."""

    def __init__(
        self,
        status: int,
        message: str,
        type: str = "api_error",
        code: str | None = None,
        param: str | None = None,
        provider: str | None = None,
        retry_same_provider: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.type = type
        self.code = code
        self.param = param
        self.provider = provider
        # None means "decide from the status". Adapters set False for errors a retry can't fix,
        # like the gateway's own key being rejected.
        self._retry_same_provider = retry_same_provider

    @property
    def retryable(self) -> bool:
        """Worth trying on the next provider in the alias's list."""
        return self.status in RETRYABLE_STATUSES

    @property
    def retry_same_provider(self) -> bool:
        """Worth one more try on this provider before moving on."""
        if self._retry_same_provider is not None:
            return self._retry_same_provider
        return self.status in RETRY_SAME_STATUSES


class Provider(Protocol):
    name: str

    async def chat(self, request: ChatRequest) -> ChatResponse: ...
