from typing import Protocol

from app.api.schemas import ChatRequest, ChatResponse

# Statuses worth trying again or on another provider. A 400 would fail the same way anywhere.
RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}


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
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.type = type
        self.code = code
        self.param = param
        self.provider = provider

    @property
    def retryable(self) -> bool:
        return self.status in RETRYABLE_STATUSES


class Provider(Protocol):
    name: str

    async def chat(self, request: ChatRequest) -> ChatResponse: ...
