"""Where the exact-match cache keeps its answers, and the front the route talks to.

`ResponseCache` never fails or slows a request because of the cache: a lookup that errors or
takes too long is a miss, and answers are stored in a task of their own after the response is
sent, the way request logs are (app/usage/log.py).
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Protocol

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from app.api.schemas import ChatResponse

logger = logging.getLogger(__name__)

# Well above a primary-key lookup (about a millisecond), well below any provider call. Past it
# the database is struggling, and calling the provider is the faster answer.
LOOKUP_TIMEOUT_S = 0.5


@dataclass(frozen=True)
class CacheEntry:
    """A row of `response_cache`, without its key."""

    alias: str
    # The answer as first served, usage included.
    response: ChatResponse
    # Who answered, and what that call cost: what each hit saves.
    provider: str
    model: str
    cost_usd: Decimal
    created_at: datetime
    expires_at: datetime


class CacheStore(Protocol):
    async def get(self, key: bytes, now: datetime) -> CacheEntry | None:
        """The entry for `key` if it hasn't expired by `now`."""
        ...

    async def put(self, key: bytes, entry: CacheEntry) -> None:
        """Stores `entry`, replacing any entry already under `key`."""
        ...

    async def delete_expired(self, now: datetime) -> int:
        """Deletes the entries expired by `now` and returns how many."""
        ...


_COLUMNS = "alias, response, provider, model, cost_usd, created_at, expires_at"


class PostgresCacheStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self.pool = pool

    async def get(self, key: bytes, now: datetime) -> CacheEntry | None:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT {_COLUMNS} FROM response_cache WHERE cache_key = %s AND expires_at > %s",
                (key, now),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        alias, response, provider, model, cost_usd, created_at, expires_at = row
        return CacheEntry(
            alias=alias,
            response=ChatResponse.model_validate(response),
            provider=provider,
            model=model,
            cost_usd=cost_usd,
            created_at=created_at,
            expires_at=expires_at,
        )

    async def put(self, key: bytes, entry: CacheEntry) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                f"INSERT INTO response_cache (cache_key, {_COLUMNS})"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT (cache_key) DO UPDATE SET"
                " alias = EXCLUDED.alias, response = EXCLUDED.response,"
                " provider = EXCLUDED.provider, model = EXCLUDED.model,"
                " cost_usd = EXCLUDED.cost_usd, created_at = EXCLUDED.created_at,"
                " expires_at = EXCLUDED.expires_at",
                (
                    key,
                    entry.alias,
                    Jsonb(entry.response.model_dump(mode="json")),
                    entry.provider,
                    entry.model,
                    entry.cost_usd,
                    entry.created_at,
                    entry.expires_at,
                ),
            )

    async def delete_expired(self, now: datetime) -> int:
        async with self.pool.connection() as conn:
            cur = await conn.execute("DELETE FROM response_cache WHERE expires_at <= %s", (now,))
            return cur.rowcount


class InMemoryCacheStore:
    """A cache store for tests: no database."""

    def __init__(self) -> None:
        self.entries: dict[bytes, CacheEntry] = {}

    async def get(self, key: bytes, now: datetime) -> CacheEntry | None:
        entry = self.entries.get(key)
        return entry if entry is not None and entry.expires_at > now else None

    async def put(self, key: bytes, entry: CacheEntry) -> None:
        self.entries[key] = entry

    async def delete_expired(self, now: datetime) -> int:
        expired = [k for k, e in self.entries.items() if e.expires_at <= now]
        for k in expired:
            del self.entries[k]
        return len(expired)


class ResponseCache:
    """The exact-match cache as the route sees it: look up before calling a provider, store
    after answering. Neither can fail the request.

    Stores run as tasks that are kept until done and drained at shutdown, before the database
    pool closes, as `LogWriter` does with request logs.
    """

    def __init__(
        self,
        store: CacheStore,
        *,
        ttl_s: float,
        lookup_timeout_s: float = LOOKUP_TIMEOUT_S,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.store = store
        self._ttl = timedelta(seconds=ttl_s)
        self._lookup_timeout_s = lookup_timeout_s
        # Injectable so tests can expire entries without waiting.
        self._clock = clock
        self._pending: set[asyncio.Task[None]] = set()

    async def get(self, key: bytes) -> CacheEntry | None:
        """The stored answer for `key`, or None on a miss, an expired entry, or any error."""
        try:
            async with asyncio.timeout(self._lookup_timeout_s):
                return await self.store.get(key, self._clock())
        except Exception:
            # TimeoutError included. The provider can still answer: a broken cache only costs
            # money, never the request.
            logger.warning("cache lookup failed: treating it as a miss", exc_info=True)
            return None

    def put(
        self,
        key: bytes,
        *,
        alias: str,
        response: ChatResponse,
        provider: str,
        model: str,
        cost_usd: Decimal,
    ) -> None:
        """Stores an answer in the background and returns at once."""
        now = self._clock()
        entry = CacheEntry(
            alias=alias,
            response=response,
            provider=provider,
            model=model,
            cost_usd=cost_usd,
            created_at=now,
            expires_at=now + self._ttl,
        )
        task = asyncio.create_task(self._put(key, entry))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def delete_expired(self) -> None:
        """Clears out expired entries (lookups already ignore them). Errors only log."""
        try:
            deleted = await self.store.delete_expired(self._clock())
        except Exception:
            logger.warning("could not delete expired cache entries", exc_info=True)
            return
        if deleted:
            logger.info("deleted %d expired cache entries", deleted)

    async def drain(self) -> None:
        """Waits for every store started so far."""
        while self._pending:
            await asyncio.gather(*self._pending)

    async def _put(self, key: bytes, entry: CacheEntry) -> None:
        try:
            await self.store.put(key, entry)
        except Exception:
            # The response is already sent; the next identical request is a miss, that's all.
            logger.warning("could not store a cache entry for %s", entry.alias, exc_info=True)
