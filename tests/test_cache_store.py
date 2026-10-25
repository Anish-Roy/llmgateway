"""The exact-match cache's stores, and the front that keeps cache trouble away from requests."""

import asyncio
import hashlib
import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import psycopg
import pytest

from app.api.schemas import ChatResponse
from app.cache.store import CacheEntry, InMemoryCacheStore, PostgresCacheStore, ResponseCache
from app.db.migrate import apply_migrations
from app.db.pool import open_pool

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
KEY = hashlib.sha256(b"a request").digest()
OTHER_KEY = hashlib.sha256(b"another request").digest()


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        return self.now


def _response(content: str = "HTTP is a protocol.") -> ChatResponse:
    return ChatResponse.model_validate(
        {
            "id": "chatcmpl-abc",
            "created": 1_761_393_600,
            "model": "openai/gpt-oss-20b",
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 81, "completion_tokens": 44, "total_tokens": 125},
        }
    )


def _entry(**overrides) -> CacheEntry:
    fields = {
        "alias": "fast",
        "response": _response(),
        "provider": "groq",
        "model": "openai/gpt-oss-20b",
        "cost_usd": Decimal("0.000019275"),
        "created_at": NOW,
        "expires_at": NOW + timedelta(hours=24),
    } | overrides
    return CacheEntry(**fields)


def _put(cache: ResponseCache, key: bytes = KEY, content: str = "HTTP is a protocol.") -> None:
    cache.put(
        key,
        alias="fast",
        response=_response(content),
        provider="groq",
        model="openai/gpt-oss-20b",
        cost_usd=Decimal("0.000019275"),
    )


# The in-memory store, through ResponseCache.


async def test_a_stored_answer_is_found_by_its_key():
    cache = ResponseCache(InMemoryCacheStore(), ttl_s=60, clock=Clock())

    _put(cache)
    await cache.drain()

    assert await cache.get(KEY) == _entry(expires_at=NOW + timedelta(seconds=60))
    assert await cache.get(OTHER_KEY) is None


async def test_an_entry_expires_after_the_ttl():
    clock = Clock()
    cache = ResponseCache(InMemoryCacheStore(), ttl_s=60, clock=clock)
    _put(cache)
    await cache.drain()

    clock.now = NOW + timedelta(seconds=59)
    assert await cache.get(KEY) is not None
    clock.now = NOW + timedelta(seconds=60)
    assert await cache.get(KEY) is None


async def test_a_new_answer_replaces_the_old_one_and_restarts_the_ttl():
    clock = Clock()
    cache = ResponseCache(InMemoryCacheStore(), ttl_s=60, clock=clock)
    _put(cache, content="old")
    await cache.drain()

    clock.now = NOW + timedelta(seconds=30)
    _put(cache, content="new")
    await cache.drain()

    entry = await cache.get(KEY)
    assert entry.response.choices[0].message.content == "new"
    assert entry.expires_at == NOW + timedelta(seconds=90)


async def test_put_returns_before_the_store_finishes():
    release = asyncio.Event()

    class SlowStore(InMemoryCacheStore):
        async def put(self, key: bytes, entry: CacheEntry) -> None:
            await release.wait()
            await super().put(key, entry)

    store = SlowStore()
    cache = ResponseCache(store, ttl_s=60)

    _put(cache)
    await asyncio.sleep(0)
    assert store.entries == {}

    release.set()
    await cache.drain()
    assert KEY in store.entries


async def test_drain_with_nothing_pending_returns():
    await ResponseCache(InMemoryCacheStore(), ttl_s=60).drain()


async def test_a_failed_lookup_is_a_miss(caplog):
    class BrokenStore(InMemoryCacheStore):
        async def get(self, key: bytes, now: datetime) -> CacheEntry | None:
            raise psycopg.OperationalError("connection refused")

    cache = ResponseCache(BrokenStore(), ttl_s=60)

    with caplog.at_level(logging.WARNING, logger="app.cache.store"):
        assert await cache.get(KEY) is None
    assert "cache lookup failed" in caplog.text


async def test_a_slow_lookup_is_a_miss(caplog):
    class HangingStore(InMemoryCacheStore):
        async def get(self, key: bytes, now: datetime) -> CacheEntry | None:
            await asyncio.Event().wait()

    cache = ResponseCache(HangingStore(), ttl_s=60, lookup_timeout_s=0.01)

    with caplog.at_level(logging.WARNING, logger="app.cache.store"):
        assert await cache.get(KEY) is None
    assert "cache lookup failed" in caplog.text


async def test_a_failed_store_is_logged_not_raised(caplog):
    class BrokenStore(InMemoryCacheStore):
        async def put(self, key: bytes, entry: CacheEntry) -> None:
            raise psycopg.OperationalError("connection refused")

    cache = ResponseCache(BrokenStore(), ttl_s=60)

    with caplog.at_level(logging.WARNING, logger="app.cache.store"):
        _put(cache)
        await cache.drain()
    assert "could not store a cache entry for fast" in caplog.text


async def test_delete_expired_removes_only_expired_entries():
    clock = Clock()
    store = InMemoryCacheStore()
    cache = ResponseCache(store, ttl_s=60, clock=clock)
    _put(cache, KEY)
    await cache.drain()
    clock.now = NOW + timedelta(seconds=30)
    _put(cache, OTHER_KEY)
    await cache.drain()

    clock.now = NOW + timedelta(seconds=60)
    await cache.delete_expired()

    assert list(store.entries) == [OTHER_KEY]


async def test_a_failed_delete_is_logged_not_raised(caplog):
    class BrokenStore(InMemoryCacheStore):
        async def delete_expired(self, now: datetime) -> int:
            raise psycopg.OperationalError("connection refused")

    with caplog.at_level(logging.WARNING, logger="app.cache.store"):
        await ResponseCache(BrokenStore(), ttl_s=60).delete_expired()
    assert "could not delete expired cache entries" in caplog.text


# The Postgres store.


@pytest.fixture
def migrated(pg_conninfo) -> str:
    with psycopg.connect(pg_conninfo) as conn:
        apply_migrations(conn)
    return pg_conninfo


async def test_postgres_store_round_trips_every_field(migrated):
    # Extra fields on the message (reasoning, tool_call_id...) survive the jsonb round trip.
    response = _response()
    response.choices[0].message.reasoning = "Define it briefly."
    entry = _entry(response=response)

    async with open_pool(migrated, max_size=1) as pool:
        store = PostgresCacheStore(pool)
        await store.put(KEY, entry)
        found = await store.get(KEY, NOW)

    assert found == entry
    assert found.response.choices[0].message.reasoning == "Define it briefly."


async def test_postgres_store_ignores_expired_and_unknown_keys(migrated):
    async with open_pool(migrated, max_size=1) as pool:
        store = PostgresCacheStore(pool)
        await store.put(KEY, _entry(expires_at=NOW + timedelta(seconds=60)))

        assert await store.get(KEY, NOW + timedelta(seconds=59)) is not None
        assert await store.get(KEY, NOW + timedelta(seconds=60)) is None
        assert await store.get(OTHER_KEY, NOW) is None


async def test_postgres_store_replaces_an_entry_under_the_same_key(migrated):
    newer = _entry(
        response=_response("new"),
        provider="gemini",
        model="gemini-3.5-flash-lite",
        cost_usd=Decimal("0.00001"),
        created_at=NOW + timedelta(hours=1),
        expires_at=NOW + timedelta(hours=25),
    )

    async with open_pool(migrated, max_size=1) as pool:
        store = PostgresCacheStore(pool)
        await store.put(KEY, _entry())
        await store.put(KEY, newer)

        assert await store.get(KEY, NOW + timedelta(hours=1)) == newer
        async with pool.connection() as conn:
            cur = await conn.execute("SELECT count(*) FROM response_cache")
            assert (await cur.fetchone())[0] == 1


async def test_postgres_store_deletes_only_expired_entries(migrated):
    async with open_pool(migrated, max_size=1) as pool:
        store = PostgresCacheStore(pool)
        await store.put(KEY, _entry(expires_at=NOW + timedelta(seconds=60)))
        await store.put(OTHER_KEY, _entry(expires_at=NOW + timedelta(seconds=120)))

        assert await store.delete_expired(NOW + timedelta(seconds=60)) == 1
        assert await store.get(KEY, NOW) is None
        assert await store.get(OTHER_KEY, NOW) is not None


@pytest.mark.parametrize(
    "bad",
    [
        # Not a SHA-256 digest.
        {"cache_key": b"short"},
        # Expires before it was made.
        {"expires_at": NOW - timedelta(seconds=1)},
        {"cost_usd": Decimal("-0.1")},
    ],
)
def test_bad_cache_rows_are_rejected(migrated, bad):
    row = {
        "cache_key": KEY,
        "alias": "fast",
        "response": '{"model": "m", "choices": []}',
        "provider": "groq",
        "model": "openai/gpt-oss-20b",
        "cost_usd": Decimal(0),
        "created_at": NOW,
        "expires_at": NOW + timedelta(hours=1),
    } | bad
    columns = ", ".join(row)
    values = ", ".join(f"%({c})s" for c in row)
    with psycopg.connect(migrated) as conn, pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(f"INSERT INTO response_cache ({columns}) VALUES ({values})", row)
