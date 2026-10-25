"""Streamed replies as server-sent events, in OpenAI's format.

Each chunk is a `data: {json}` event, and a finished stream ends with `data: [DONE]`. A failure
after the first chunk can't change the status code any more (the 200 is already sent), so it
becomes an error event, `data: {"error": {...}}`, and the stream ends without `[DONE]`. The
openai SDK raises that event as an `APIError`.
"""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping

from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from app.api.errors import error_body
from app.providers.base import ProviderError
from app.routing.router import RoutedStream

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
    stream: RoutedStream, *, include_usage: bool, headers: Mapping[str, str]
) -> StreamingResponse:
    return _ClosingStreamingResponse(
        _events(stream, include_usage), stream.aclose, {**SSE_HEADERS, **headers}
    )


async def _events(stream: RoutedStream, include_usage: bool) -> AsyncIterator[str]:
    try:
        async for chunk in stream:
            if not chunk.choices and not include_usage:
                # The usage-only chunk is sent only when the client asked for it
                # (stream_options.include_usage), as OpenAI does. Clients that read
                # chunk.choices[0] would crash on it otherwise.
                continue
            yield _event(chunk.model_dump_json(exclude_none=True))
    except ProviderError as e:
        logger.warning("%s failed mid-stream: %s %s", stream.target.provider, e.status, e.message)
        yield _event(error_body(e.status, e.message, e.type, e.param, e.code).model_dump_json())
        return
    except Exception:
        logger.exception("unhandled error mid-stream")
        yield _event(error_body(500, "Internal gateway error", "server_error").model_dump_json())
        return
    yield _event("[DONE]")


def _event(data: str) -> str:
    return f"data: {data}\n\n"
