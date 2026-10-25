"""Reading server-sent events from a provider's streaming response."""

from collections.abc import AsyncIterator

import httpx


async def data_events(resp: httpx.Response) -> AsyncIterator[str]:
    """The `data:` payload of each event, in order. Comments and other fields are skipped.

    An event's data can span several `data:` lines; they're joined with newlines, as the SSE
    spec says. Providers send one line per event, but a proxy may split them.
    """
    lines: list[str] = []
    async for line in resp.aiter_lines():
        if not line:
            # A blank line ends the event.
            if lines:
                yield "\n".join(lines)
                lines = []
        elif line.startswith("data:"):
            lines.append(line.removeprefix("data:").removeprefix(" "))
    if lines:
        yield "\n".join(lines)
