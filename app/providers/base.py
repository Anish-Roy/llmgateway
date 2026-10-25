from collections.abc import AsyncIterator
from typing import Protocol

from app.api.schemas import ChatChunk, ChatRequest, ChatResponse

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
        provider_specific: bool = False,
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
        # True for "this provider can't do that" (e.g. Gemini and `tools`): the request is fine,
        # so another provider should get a chance even though the status is a 400.
        self.provider_specific = provider_specific

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

    def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        """The reply as OpenAI-format chunks. Usually an async generator.

        A failure raises ProviderError, either from this call or while iterating. The caller
        closes the iterator (`aclose`) when it stops early, so the upstream connection is freed.
        """
        ...
