"""Where the semantic cache keeps its answers, and the front the route talks to.

`SemanticCache` follows the exact cache's rules (app/cache/store.py): a lookup that errors or
takes too long is a miss, and answers are stored in tasks of their own after the response is
sent. Embedding runs on one worker thread of its own, off the event loop.
"""

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Protocol

import numpy as np
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from app.api.schemas import ChatRequest, ChatResponse
from app.cache.embedder import Embedder, embed_text
from app.cache.key import cacheable
from app.cache.semantic import (
    DEFAULT_THRESHOLD,
    context_hash,
    entry_key,
    same_numbers,
    wants_semantic,
)
from app.cache.store import LOOKUP_TIMEOUT_S, CacheEntry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticEntry:
    """A row of `semantic_cache`."""

    key_id: int
    context_hash: bytes
    prompt: str
    # Unit length, `DIMENSIONS` wide.
    embedding: np.ndarray
    # What a hit serves, and what it saves: the same fields as an exact-cache entry.
    entry: CacheEntry


@dataclass(frozen=True)
class Candidate:
    """The stored question nearest to a new one, in the same key and context, and live."""

    prompt: str
    similarity: float
    entry: CacheEntry


class SemanticStore(Protocol):
    async def nearest(
        self, key_id: int, context: bytes, embedding: np.ndarray, now: datetime
    ) -> Candidate | None:
        """The most similar entry stored by `key_id` in `context` that hasn't expired by
        `now`, however far: the threshold is the caller's."""
        ...

    async def put(self, entry: SemanticEntry) -> None:
        """Stores `entry`, replacing one for the same key, context and prompt."""
        ...

    async def delete_expired(self, now: datetime) -> int:
        """Deletes the entries expired by `now` and returns how many."""
        ...


def _vector_literal(embedding: np.ndarray) -> str:
    # pgvector's text form, cast with ::vector in the query: no type adapter to register on
    # the pool, which would have to run after migration 004 creates the type.
    return "[" + ",".join(map(str, embedding.tolist())) + "]"


_ENTRY_COLUMNS = "alias, response, provider, model, cost_usd, created_at, expires_at"


class PostgresSemanticStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self.pool = pool

    async def nearest(
        self, key_id: int, context: bytes, embedding: np.ndarray, now: datetime
    ) -> Candidate | None:
        vector = _vector_literal(embedding)
        async with self.pool.connection() as conn:
            # The HNSW index finds candidates first and filters after; with entries scoped
            # per key, a first batch from other keys would leave nothing. An iterative scan
            # keeps going until a row passes (pgvector 0.8+). For this transaction only.
            await conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            cur = await conn.execute(
                f"SELECT prompt, 1 - (embedding <=> %(v)s::vector), {_ENTRY_COLUMNS}"
                " FROM semantic_cache"
                " WHERE key_id = %(key)s AND context_hash = %(ctx)s AND expires_at > %(now)s"
                " ORDER BY embedding <=> %(v)s::vector LIMIT 1",
                {"v": vector, "key": key_id, "ctx": context, "now": now},
            )
            row = await cur.fetchone()
        if row is None:
            return None
        prompt, similarity, alias, response, provider, model, cost_usd, created, expires = row
        return Candidate(
            prompt=prompt,
            similarity=float(similarity),
            entry=CacheEntry(
                alias=alias,
                response=ChatResponse.model_validate(response),
                provider=provider,
                model=model,
                cost_usd=cost_usd,
                created_at=created,
                expires_at=expires,
            ),
        )

    async def put(self, entry: SemanticEntry) -> None:
        e = entry.entry
        async with self.pool.connection() as conn:
            await conn.execute(
                "INSERT INTO semantic_cache (entry_key, key_id, context_hash, prompt, embedding,"
                f" {_ENTRY_COLUMNS})"
                " VALUES (%s, %s, %s, %s, %s::vector, %s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT (entry_key) DO UPDATE SET"
                " embedding = EXCLUDED.embedding, alias = EXCLUDED.alias,"
                " response = EXCLUDED.response, provider = EXCLUDED.provider,"
                " model = EXCLUDED.model, cost_usd = EXCLUDED.cost_usd,"
                " created_at = EXCLUDED.created_at, expires_at = EXCLUDED.expires_at",
                (
                    entry_key(entry.key_id, entry.context_hash, entry.prompt),
                    entry.key_id,
                    entry.context_hash,
                    entry.prompt,
                    _vector_literal(entry.embedding),
                    e.alias,
                    Jsonb(e.response.model_dump(mode="json")),
                    e.provider,
                    e.model,
                    e.cost_usd,
                    e.created_at,
                    e.expires_at,
                ),
            )

    async def delete_expired(self, now: datetime) -> int:
        async with self.pool.connection() as conn:
            cur = await conn.execute("DELETE FROM semantic_cache WHERE expires_at <= %s", (now,))
            return cur.rowcount


class InMemorySemanticStore:
    """A semantic store for tests: no database, an exact nearest-neighbour search."""

    def __init__(self) -> None:
        self.entries: dict[bytes, SemanticEntry] = {}

    async def nearest(
        self, key_id: int, context: bytes, embedding: np.ndarray, now: datetime
    ) -> Candidate | None:
        live = [
            e
            for e in self.entries.values()
            if e.key_id == key_id and e.context_hash == context and e.entry.expires_at > now
        ]
        if not live:
            return None
        best = max(live, key=lambda e: float(e.embedding @ embedding))
        return Candidate(best.prompt, float(best.embedding @ embedding), best.entry)

    async def put(self, entry: SemanticEntry) -> None:
        self.entries[entry_key(entry.key_id, entry.context_hash, entry.prompt)] = entry

    async def delete_expired(self, now: datetime) -> int:
        expired = [k for k, e in self.entries.items() if e.entry.expires_at <= now]
        for k in expired:
            del self.entries[k]
        return len(expired)


class NoSemanticStore:
    """Finds nothing and keeps nothing: the semantic cache of an app that was never started
    (most tests)."""

    async def nearest(
        self, key_id: int, context: bytes, embedding: np.ndarray, now: datetime
    ) -> Candidate | None:
        return None

    async def put(self, entry: SemanticEntry) -> None:
        pass

    async def delete_expired(self, now: datetime) -> int:
        return 0


@dataclass
class SemanticLookup:
    """One request's question, as the semantic cache matches it. The embedding is filled in
    by the lookup and reused to store the answer, so a request is embedded at most once."""

    key_id: int
    context: bytes
    prompt: str
    embedding: np.ndarray | None = None


class SemanticCache:
    """The semantic cache as the route sees it: look up after an exact miss, store after
    answering. Neither can fail the request.

    Without an embedder (the semantic cache is off, or the model didn't load) every request
    skips it. Embeddings run on one thread of their own: each is about 12 ms of CPU that
    torch already spreads over the cores, so running several at once would only make them
    fight over the same cores, and it keeps the event loop's shared thread pool free.
    """

    def __init__(
        self,
        store: SemanticStore,
        embedder: Embedder | None,
        *,
        ttl_s: float,
        threshold: float = DEFAULT_THRESHOLD,
        max_temperature: float = 0.2,
        lookup_timeout_s: float = LOOKUP_TIMEOUT_S,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.store = store
        self.embedder = embedder
        self._ttl = timedelta(seconds=ttl_s)
        self.threshold = threshold
        self._max_temperature = max_temperature
        self._lookup_timeout_s = lookup_timeout_s
        self._clock = clock
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embed")
        self._pending: set[asyncio.Task[None]] = set()

    def lookup_for(
        self, request: ChatRequest, header: str | None, key_id: int
    ) -> SemanticLookup | None:
        """What to match `request` on, or None if the semantic cache stays out of it: the
        client didn't opt in, the request isn't cacheable or single-turn, or there's no
        embedder."""
        if self.embedder is None or not wants_semantic(header):
            return None
        if not cacheable(request, self._max_temperature):
            return None
        prompt = embed_text(request)
        if prompt is None:
            return None
        return SemanticLookup(key_id=key_id, context=context_hash(request), prompt=prompt)

    async def get(self, lookup: SemanticLookup) -> CacheEntry | None:
        """A stored answer to a similar enough question, or None on a miss or any error.

        A hit needs the nearest stored question at or above the threshold and holding the
        same numbers. Embedding and the query share the lookup timeout.
        """
        try:
            async with asyncio.timeout(self._lookup_timeout_s):
                lookup.embedding = await self._embed(lookup.prompt)
                candidate = await self.store.nearest(
                    lookup.key_id, lookup.context, lookup.embedding, self._clock()
                )
        except Exception:
            # TimeoutError included. A broken cache only costs a provider call.
            logger.warning("semantic cache lookup failed: treating it as a miss", exc_info=True)
            return None
        if candidate is None or candidate.similarity < self.threshold:
            return None
        if not same_numbers(lookup.prompt, candidate.prompt):
            return None
        return candidate.entry

    def put(
        self,
        lookup: SemanticLookup,
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
        task = asyncio.create_task(self._put(lookup, entry))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def delete_expired(self) -> None:
        """Clears out expired entries (lookups already ignore them). Errors only log."""
        try:
            deleted = await self.store.delete_expired(self._clock())
        except Exception:
            logger.warning("could not delete expired semantic cache entries", exc_info=True)
            return
        if deleted:
            logger.info("deleted %d expired semantic cache entries", deleted)

    async def drain(self) -> None:
        """Waits for every store started so far."""
        while self._pending:
            await asyncio.gather(*self._pending)

    def close(self) -> None:
        """Stops the embedding thread once its current work is done. Call after `drain`."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def _embed(self, text: str) -> np.ndarray:
        assert self.embedder is not None
        loop = asyncio.get_running_loop()
        vectors = await loop.run_in_executor(self._executor, self.embedder.embed, [text])
        return vectors[0]

    async def _put(self, lookup: SemanticLookup, entry: CacheEntry) -> None:
        try:
            # A lookup that timed out may have left no embedding: make it now, off the
            # request's path.
            embedding = lookup.embedding
            if embedding is None:
                embedding = await self._embed(lookup.prompt)
            await self.store.put(
                SemanticEntry(lookup.key_id, lookup.context, lookup.prompt, embedding, entry)
            )
        except Exception:
            # The response is already sent; the next similar question is a miss, that's all.
            logger.warning(
                "could not store a semantic cache entry for %s", entry.alias, exc_info=True
            )
