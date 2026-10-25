"""Streamed replies as server-sent events, in OpenAI's format.

Each chunk is a `data: {json}` event, and a finished stream ends with `data: [DONE]`. A failure
after the first chunk can't change the status code any more (the 200 is already sent), so it
becomes an error event, `data: {"error": {...}}`, and the stream ends without `[DONE]`. The
openai SDK raises that event as an `APIError`.
"""

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping

from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from app.api.errors import error_body
from app.api.schemas import ChatChunk, ChatResponse, ChunkChoice, Delta
from app.providers.base import ProviderError
from app.routing.router import RoutedStream, StreamRecord

logger = logging.getLogger(__name__)

SSE_HEADERS = {
    # Every chunk should reach the client as soon as it's written. no-cache keeps caches from
    # holding it; X-Accel-Buffering stops nginx-style proxies (Render, Fly...) from buffering.
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}


class _ClosingStreamingResponse(StreamingResponse):
    """A StreamingResponse that always runs `on_close` once it's done sending, however it ends.

    Starlette doesn't close the body iterator when the client disconnects (it raises
    ClientDisconnect), so the upstream provider stream would stay open until garbage collection.
    """

    def __init__(
        self,
        content: AsyncIterator[str],
        on_close: Callable[[], Awaitable[None]],
        headers: Mapping[str, str],
    ) -> None:
        super().__init__(content, media_type="text/event-stream", headers=headers)
        self._on_close = on_close

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._on_close()


def sse_response(
    stream: RoutedStream,
    *,
    include_usage: bool,
    headers: Mapping[str, str],
    on_finish: Callable[[StreamRecord], None] | None = None,
) -> StreamingResponse:
    """`on_finish` gets the stream's complete record once the stream is closed, however it
    ended (finished, failed, or the client left)."""

    async def close() -> None:
        try:
            await stream.aclose()
        finally:
            if on_finish is not None:
                on_finish(stream.record)

    return _ClosingStreamingResponse(
        _events(stream, include_usage, stream.target.provider), close, {**SSE_HEADERS, **headers}
    )


def sse_replay(
    stored: ChatResponse, *, include_usage: bool, headers: Mapping[str, str]
) -> StreamingResponse:
    """A stored answer sent as a stream, for a cache hit: the same events a provider's stream
    would have produced, all at once."""

    async def chunks() -> AsyncIterator[ChatChunk]:
        for chunk in replay_chunks(stored):
            yield chunk

    async def nothing_to_close() -> None:
        pass

    return _ClosingStreamingResponse(
        _events(chunks(), include_usage, "cache"), nothing_to_close, {**SSE_HEADERS, **headers}
    )


def replay_chunks(stored: ChatResponse) -> Iterable[ChatChunk]:
    """A one-choice answer as OpenAI streams it: a role chunk, the text, an empty delta with
    the finish_reason, then the usage with no choices. One text chunk: the whole answer is
    already here, so splitting it would only add events. A fresh id and timestamp, as for
    any new completion."""
    choice = stored.choices[0]
    meta = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "created": int(time.time()),
        "model": stored.model,
    }
    return [
        ChatChunk(**meta, choices=[ChunkChoice(delta=Delta(role="assistant", content=""))]),
        ChatChunk(**meta, choices=[ChunkChoice(delta=Delta(content=choice.message.content))]),
        ChatChunk(**meta, choices=[ChunkChoice(finish_reason=choice.finish_reason)]),
        ChatChunk(**meta, usage=stored.usage),
    ]


async def _events(
    chunks: AsyncIterator[ChatChunk], include_usage: bool, source: str
) -> AsyncIterator[str]:
    """`source` names where the chunks come from (a provider, or the cache) in logs."""
    try:
        async for chunk in chunks:
            if not chunk.choices and not include_usage:
                # The usage-only chunk is sent only when the client asked for it
                # (stream_options.include_usage), as OpenAI does. Clients that read
                # chunk.choices[0] would crash on it otherwise.
                continue
            # Usage on a chunk with choices is a provider's running total, kept for the
            # gateway's own records. OpenAI sends usage only on the final chunk.
            exclude = {"usage"} if chunk.choices else None
            yield _event(chunk.model_dump_json(exclude_none=True, exclude=exclude))
    except ProviderError as e:
        logger.warning("%s failed mid-stream: %s %s", source, e.status, e.message)
        yield _event(error_body(e.status, e.message, e.type, e.param, e.code).model_dump_json())
        return
    except Exception:
        logger.exception("unhandled error mid-stream")
        yield _event(error_body(500, "Internal gateway error", "server_error").model_dump_json())
        return
    yield _event("[DONE]")


def _event(data: str) -> str:
    return f"data: {data}\n\n"
