"""Request logs: one row per chat-completions request, written after the response is sent.

The route hands a finished `RequestLog` to `LogWriter.submit`, which writes it in a task of
its own and returns at once: the client never waits for the insert, and a failed insert only
logs a warning. A `BackgroundTask` can't do this here: it never runs when an error handler
builds the response, and on a stream it runs before the stream's record is complete.
"""

import asyncio
import dataclasses
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Literal, Protocol

from psycopg_pool import AsyncConnectionPool

from app.guardrails.pii import PiiType

logger = logging.getLogger(__name__)

# How a request ended. The first four are a stream's (StreamRecord.outcome) and mean the same
# for a plain reply; `rejected` is a request the gateway refused before calling any provider.
LogOutcome = Literal[
    "completed", "provider_error", "gateway_error", "client_disconnect", "rejected"
]
CacheStatus = Literal["miss", "exact", "semantic"]


@dataclass(frozen=True)
class RequestLog:
    """A row of `request_logs`, without its id."""

    key_id: int
    # When the request arrived, not when the row was written.
    created_at: datetime
    requested_model: str
    # The provider entry and model that answered; both None if none did.
    provider: str | None
    model: str | None
    stream: bool
    status_code: int
    outcome: LogOutcome
    prompt_tokens: int
    completion_tokens: int
    usage_estimated: bool
    latency_ms: int
    ttft_ms: int | None
    cost_usd: Decimal
    attempts: int
    fallback: bool
    cache_status: CacheStatus = "miss"
    # The kinds of PII masked in the request, sorted; never the values. A list, which psycopg
    # sends as a Postgres array.
    pii_types: list[PiiType] = field(default_factory=list)
    # On `model: "auto"` requests: the alias it picked, and the rule that picked it. None on
    # every other request, and on an auto request refused before it was routed (a 429).
    routed_alias: str | None = None
    route_reason: str | None = None


_COLUMNS = [f.name for f in dataclasses.fields(RequestLog)]
_INSERT = (
    f"INSERT INTO request_logs ({', '.join(_COLUMNS)})"
    f" VALUES ({', '.join(f'%({c})s' for c in _COLUMNS)})"
)


class UsageLog(Protocol):
    """Where request logs go. Write-only: the dashboard reads the table directly."""

    async def write(self, entry: RequestLog) -> None: ...


class PostgresUsageLog:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self.pool = pool

    async def write(self, entry: RequestLog) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(_INSERT, dataclasses.asdict(entry))


class InMemoryUsageLog:
    """A usage log for tests: no database."""

    def __init__(self) -> None:
        self.entries: list[RequestLog] = []

    async def write(self, entry: RequestLog) -> None:
        self.entries.append(entry)


class LogWriter:
    """Writes request logs off the request path.

    Each entry gets its own task. The tasks are kept until they finish (the event loop holds
    only weak references, so an unreferenced task can be garbage-collected mid-write), and
    `drain` waits for them at shutdown, before the database pool closes.
    """

    def __init__(self, store: UsageLog) -> None:
        self.store = store
        self._pending: set[asyncio.Task[None]] = set()

    def submit(self, entry: RequestLog) -> None:
        task = asyncio.create_task(self._write(entry))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def drain(self) -> None:
        """Waits for every write submitted so far."""
        while self._pending:
            await asyncio.gather(*self._pending)

    async def _write(self, entry: RequestLog) -> None:
        try:
            await self.store.write(entry)
        except Exception:
            # The response is long gone: losing a row beats failing anything else.
            logger.warning(
                "could not write the request log for key %s (%s, %s)",
                entry.key_id,
                entry.requested_model,
                entry.status_code,
                exc_info=True,
            )
