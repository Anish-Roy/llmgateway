"""Request and response models in the OpenAI chat-completions format."""

import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    # Clients may send extra keys (name, tool_call_id...). Keep them so they reach the provider.
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | None = None


class ChatRequest(BaseModel):
    # Fields we don't model (top_p, stop, seed, tools...) are kept and forwarded to the provider,
    # so a client's settings are never silently dropped. The provider rejects what it can't do.
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    stream: bool = False


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = "stop"


class ChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


class Delta(BaseModel):
    # Providers add fields we don't model (tool_calls, reasoning...). Keep them for the client.
    model_config = ConfigDict(extra="allow")

    role: Literal["assistant"] | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: Delta = Field(default_factory=Delta)
    finish_reason: str | None = None


class ChatChunk(BaseModel):
    """One server-sent event of a streamed chat completion, in OpenAI's chunk format.

    `id`, `created` and `model` are the same on every chunk of a stream. The last chunk may
    carry `usage` with an empty `choices` list, as OpenAI sends with `include_usage`.
    Top-level vendor fields (Groq's `x_groq`) are dropped, as in ChatResponse.
    """

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice] = Field(default_factory=list)
    usage: Usage | None = None

    def has_output(self) -> bool:
        """True once the chunk carries something the client sees: text, a tool call, reasoning
        text (only sent when the client asked for it) or an end."""
        for c in self.choices:
            extra = c.delta.model_extra or {}
            if c.delta.content or extra.get("tool_calls") or extra.get("reasoning"):
                return True
            if c.finish_reason:
                return True
        return False


class Model(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "llm-gateway"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[Model]


class ErrorDetail(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
