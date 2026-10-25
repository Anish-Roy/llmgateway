"""The semantic cache's stores, and the front that keeps cache trouble away from requests."""

import asyncio
import hashlib
import logging
import threading
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import psycopg
import pytest
from fastapi.testclient import TestClient

import app.main
from app.api.schemas import ChatRequest, ChatResponse
from app.cache.embedder import DIMENSIONS, FakeEmbedder
from app.cache.semantic import context_hash
from app.cache.semantic_store import (
    InMemorySemanticStore,
    NoSemanticStore,
    PostgresSemanticStore,
    SemanticCache,
    SemanticEntry,
    SemanticLookup,
)
from app.cache.store import CacheEntry
from app.db.migrate import apply_migrations
from app.db.pool import open_pool
from app.main import create_app

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
ALICE, BOB = 1, 2
QUESTION = "What's the capital of France?"
REPHRASED = "capital of france?"


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        return self.now


def _with_similarity(similarity: float) -> tuple[list[float], list[float]]:
    """Two unit vectors whose cosine similarity is `similarity`."""
    first = [1.0] + [0.0] * (DIMENSIONS - 1)
    second = [similarity, float(np.sqrt(1 - similarity**2))] + [0.0] * (DIMENSIONS - 2)
    return first, second


def _embedder(similarity: float, a: str = QUESTION, b: str = REPHRASED) -> FakeEmbedder:
    first, second = _with_similarity(similarity)
    return FakeEmbedder({a: first, b: second})


def _request(question: str = QUESTION, **overrides) -> ChatRequest:
    fields = {
        "model": "fast",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": question},
        ],
        "temperature": 0,
    }
    return ChatRequest.model_validate({**fields, **overrides})


def _response(content: str = "Paris.") -> ChatResponse:
    return ChatResponse.model_validate(
        {
            "id": "chatcmpl-abc",
            "created": 1_761_393_600,
            "model": "openai/gpt-oss-20b",
            "choices": [
                {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
        }
    )


def _cache(embedder=None, store=None, **kwargs) -> SemanticCache:
    return SemanticCache(
        store if store is not None else InMemorySemanticStore(),
        embedder if embedder is not None else _embedder(0.97),
        ttl_s=60,
        clock=kwargs.pop("clock", Clock()),
        **kwargs,
    )


def _put(cache: SemanticCache, lookup: SemanticLookup, content: str = "Paris.") -> None:
    cache.put(
        lookup,
        alias="fast",
        response=_response(content),
        provider="groq",
        model="openai/gpt-oss-20b",
        cost_usd=Decimal("0.00001125"),
    )


async def _store_answer(cache: SemanticCache, question: str = QUESTION, key_id=ALICE, **kw):
    """Asks `question` (a miss) and stores its answer, as the route will."""
    lookup = cache.lookup_for(_request(question, **kw), "semantic", key_id)
    assert await cache.get(lookup) is None
    _put(cache, lookup)
    await cache.drain()


async def _ask(cache: SemanticCache, question: str = REPHRASED, key_id=ALICE, **kw):
    lookup = cache.lookup_for(_request(question, **kw), "semantic", key_id)
    return await cache.get(lookup)


# Which requests the front looks up.


def test_a_request_that_opts_in_gets_a_lookup():
    lookup = _cache().lookup_for(_request(), " Semantic", ALICE)
    assert lookup == SemanticLookup(ALICE, context_hash(_request()), QUESTION)


@pytest.mark.parametrize("header", [None, "off", "on"])
def test_without_opting_in_there_is_no_lookup(header):
    assert _cache().lookup_for(_request(), header, ALICE) is None


@pytest.mark.parametrize(
    "request_",
    [
        # Not cacheable: no temperature, too high a one, several choices.
        _request(temperature=None),
        _request(temperature=0.7),
        _request(n=2),
        # Not single-turn.
        _request(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": QUESTION},
            ]
        ),
        _request(tools=[{"type": "function", "function": {"name": "f"}}]),
    ],
)
def test_requests_the_cache_cant_answer_get_no_lookup(request_):
    assert _cache().lookup_for(request_, "semantic", ALICE) is None


def test_without_an_embedder_there_is_no_lookup():
    cache = SemanticCache(InMemorySemanticStore(), None, ttl_s=60)
    assert cache.lookup_for(_request(), "semantic", ALICE) is None


# Hits and misses.


async def test_a_similar_question_gets_the_stored_answer():
    cache = _cache(_embedder(0.97))
    await _store_answer(cache)
    entry = await _ask(cache)
    assert entry.response.choices[0].message.content == "Paris."
    assert (entry.provider, entry.model, entry.alias) == ("groq", "openai/gpt-oss-20b", "fast")
    assert entry.cost_usd == Decimal("0.00001125")


@pytest.mark.parametrize(("offset", "hit"), [(0.0, True), (1e-4, False)])
async def test_the_threshold_is_inclusive(offset, hit):
    embedder = _embedder(0.95)
    # The similarity as float32 makes it (0.94999998...), so the boundary is exact.
    similarity = float(np.prod(embedder.embed([QUESTION, REPHRASED]), axis=0).sum())
    cache = _cache(embedder, threshold=similarity + offset)
    await _store_answer(cache)
    assert (await _ask(cache) is not None) is hit


async def test_a_question_far_below_the_threshold_is_a_miss():
    cache = _cache(_embedder(0.5))
    await _store_answer(cache)
    assert await _ask(cache) is None


async def test_a_match_with_other_numbers_is_a_miss():
    a, b = "Is 91 a prime number?", "Is 97 a prime number?"
    cache = _cache(_embedder(0.99, a, b))
    await _store_answer(cache, a)
    assert await _ask(cache, b) is None


async def test_another_keys_answer_is_never_served():
    cache = _cache()
    await _store_answer(cache, key_id=ALICE)
    assert await _ask(cache, key_id=BOB) is None
    assert await _ask(cache, key_id=ALICE) is not None


@pytest.mark.parametrize(
    "other_context",
    [
        {"model": "strong"},
        {"max_tokens": 10},
        {
            "messages": [
                {"role": "system", "content": "Answer in French."},
                {"role": "user", "content": REPHRASED},
            ]
        },
    ],
)
async def test_an_answer_from_another_context_is_never_served(other_context):
    cache = _cache()
    await _store_answer(cache)
    assert await _ask(cache, **other_context) is None


async def test_an_entry_expires_after_the_ttl():
    clock = Clock()
    cache = _cache(clock=clock)
    await _store_answer(cache)
    clock.now = NOW + timedelta(seconds=59)
    assert await _ask(cache) is not None
    clock.now = NOW + timedelta(seconds=60)
    assert await _ask(cache) is None


async def test_the_same_question_again_replaces_its_entry():
    store = InMemorySemanticStore()
    cache = _cache(store=store)
    lookup = cache.lookup_for(_request(), "semantic", ALICE)
    await cache.get(lookup)
    _put(cache, lookup, "Paris.")
    _put(cache, lookup, "Paris, France.")
    await cache.drain()
    assert len(store.entries) == 1
    assert (await _ask(cache)).response.choices[0].message.content == "Paris, France."


# Embedding: once per request, on its own thread.


async def test_a_request_is_embedded_once_for_the_lookup_and_the_store():
    embedder = _embedder(0.97)
    cache = _cache(embedder)
    await _store_answer(cache)
    assert embedder.calls == [QUESTION]


async def test_a_store_without_an_embedding_makes_one():
    # The lookup timed out before embedding, say: the store embeds in the background.
    embedder = _embedder(0.97)
    store = InMemorySemanticStore()
    cache = _cache(embedder, store)
    _put(cache, cache.lookup_for(_request(), "semantic", ALICE))
    await cache.drain()
    assert embedder.calls == [QUESTION]
    (entry,) = store.entries.values()
    assert entry.embedding.shape == (DIMENSIONS,)


async def test_embedding_runs_on_one_thread_off_the_event_loop():
    threads = set()

    class Recording(FakeEmbedder):
        def embed(self, texts):
            threads.add(threading.current_thread().name)
            return super().embed(texts)

    cache = _cache(Recording())
    await asyncio.gather(*(_ask(cache, f"question {i}") for i in range(5)))
    (name,) = threads
    assert name.startswith("embed")
    assert name != threading.current_thread().name


# Trouble never fails a request.


class BrokenStore(InMemorySemanticStore):
    async def nearest(self, *args):
        raise OSError("database down")

    async def put(self, entry):
        raise OSError("database down")

    async def delete_expired(self, now):
        raise OSError("database down")


class SlowStore(InMemorySemanticStore):
    async def nearest(self, *args):
        await asyncio.sleep(1)


async def test_a_failed_lookup_is_a_miss(caplog):
    with caplog.at_level(logging.WARNING):
        assert await _ask(_cache(store=BrokenStore())) is None
    assert "treating it as a miss" in caplog.text


async def test_a_slow_lookup_is_a_miss(caplog):
    cache = _cache(store=SlowStore(), lookup_timeout_s=0.05)
    with caplog.at_level(logging.WARNING):
        assert await _ask(cache) is None
    assert "treating it as a miss" in caplog.text


async def test_a_failing_embedder_is_a_miss(caplog):
    class Broken(FakeEmbedder):
        def embed(self, texts):
            raise RuntimeError("out of memory")

    with caplog.at_level(logging.WARNING):
        assert await _ask(_cache(Broken())) is None
    assert "treating it as a miss" in caplog.text


async def test_a_failed_store_is_logged_not_raised(caplog):
    cache = _cache(store=BrokenStore())
    with caplog.at_level(logging.WARNING):
        _put(cache, cache.lookup_for(_request(), "semantic", ALICE))
        await cache.drain()
    assert "could not store a semantic cache entry for fast" in caplog.text


async def test_put_returns_before_the_store_finishes():
    started, release = asyncio.Event(), asyncio.Event()

    class Waiting(InMemorySemanticStore):
        async def put(self, entry):
            started.set()
            await release.wait()
            await super().put(entry)

    store = Waiting()
    cache = _cache(store=store)
    lookup = cache.lookup_for(_request(), "semantic", ALICE)
    await cache.get(lookup)
    _put(cache, lookup)
    await started.wait()
    assert store.entries == {}
    release.set()
    await cache.drain()
    assert len(store.entries) == 1


async def test_delete_expired_removes_only_expired_entries():
    clock = Clock()
    store = InMemorySemanticStore()
    cache = _cache(store=store, clock=clock)
    await _store_answer(cache)
    clock.now = NOW + timedelta(seconds=30)
    await _store_answer(cache, "Who wrote Hamlet?")
    clock.now = NOW + timedelta(seconds=60)
    await cache.delete_expired()
    assert [e.prompt for e in store.entries.values()] == ["Who wrote Hamlet?"]


async def test_a_failed_delete_is_logged_not_raised(caplog):
    with caplog.at_level(logging.WARNING):
        await _cache(store=BrokenStore()).delete_expired()
    assert "could not delete expired semantic cache entries" in caplog.text


async def test_the_no_store_finds_and_keeps_nothing():
    cache = _cache(store=NoSemanticStore())
    await _store_answer(cache)
    assert await _ask(cache) is None


# At startup.


def test_startup_gives_the_semantic_cache_the_embedder():
    app_ = create_app()
    assert app_.state.semantic_cache.embedder is None
    with TestClient(app_):
        cache = app_.state.semantic_cache
        assert cache.embedder is app_.state.embedder
        assert cache.threshold == 0.95
        assert isinstance(cache.store, NoSemanticStore)


def test_a_model_that_cant_load_leaves_requests_without_lookups(monkeypatch):
    def load():
        raise OSError("no network")

    monkeypatch.setattr(app.main, "load_embedder", load)
    app_ = create_app()
    with TestClient(app_):
        assert app_.state.semantic_cache.lookup_for(_request(), "semantic", ALICE) is None


# The Postgres store.


@pytest.fixture
def migrated(pg_conninfo) -> str:
    """A migrated schema with keys ALICE and BOB (ids 1 and 2)."""
    with psycopg.connect(pg_conninfo) as conn:
        apply_migrations(conn)
        for name in ("alice", "bob"):
            conn.execute(
                "INSERT INTO api_keys (name, key_hash, key_prefix, rpm_limit, tpm_limit)"
                " VALUES (%s, %s, 'gw-test', 60, 1000)",
                (name, hashlib.sha256(name.encode()).hexdigest()),
            )
        assert [r[0] for r in conn.execute("SELECT id FROM api_keys ORDER BY id")] == [1, 2]
    return pg_conninfo


CONTEXT = context_hash(_request())


def _entry(prompt: str, vector, key_id=ALICE, context=CONTEXT, **overrides) -> SemanticEntry:
    fields = {
        "alias": "fast",
        "response": _response(),
        "provider": "groq",
        "model": "openai/gpt-oss-20b",
        "cost_usd": Decimal("0.00001125"),
        "created_at": NOW,
        "expires_at": NOW + timedelta(hours=24),
    } | overrides
    vector = np.asarray(vector, dtype=np.float32)
    return SemanticEntry(
        key_id, context, prompt, vector / np.linalg.norm(vector), CacheEntry(**fields)
    )


async def test_postgres_store_finds_the_nearest_with_its_similarity(migrated):
    near, far = _with_similarity(0.9)
    query = np.asarray(near, dtype=np.float32)
    response = _response()
    response.choices[0].message.reasoning = "Short answer."
    stored = _entry(QUESTION, near, response=response)
    async with open_pool(migrated, max_size=1) as pool:
        store = PostgresSemanticStore(pool)
        await store.put(stored)
        await store.put(_entry("something else", far))
        candidate = await store.nearest(ALICE, CONTEXT, query, NOW)
    assert candidate.prompt == QUESTION
    assert candidate.similarity == pytest.approx(1.0, abs=1e-6)
    assert candidate.entry == stored.entry
    assert candidate.entry.response.choices[0].message.reasoning == "Short answer."


async def test_postgres_store_returns_the_nearest_however_far(migrated):
    # The threshold is the front's: the store reports the similarity.
    near, far = _with_similarity(0.3)
    async with open_pool(migrated, max_size=1) as pool:
        store = PostgresSemanticStore(pool)
        await store.put(_entry(QUESTION, far))
        candidate = await store.nearest(ALICE, CONTEXT, np.asarray(near, np.float32), NOW)
    assert candidate.similarity == pytest.approx(0.3, abs=1e-5)


async def test_postgres_store_keeps_to_one_key_one_context_and_live_rows(migrated):
    vector, _ = _with_similarity(0.9)
    query = np.asarray(vector, dtype=np.float32)
    other_context = context_hash(_request(model="strong"))
    async with open_pool(migrated, max_size=1) as pool:
        store = PostgresSemanticStore(pool)
        await store.put(_entry("bob's", vector, key_id=BOB))
        await store.put(_entry("other context", vector, context=other_context))
        await store.put(
            _entry(
                "expired",
                vector,
                created_at=NOW - timedelta(days=2),
                expires_at=NOW - timedelta(days=1),
            )
        )
        assert await store.nearest(ALICE, CONTEXT, query, NOW) is None
        assert (await store.nearest(BOB, CONTEXT, query, NOW)).prompt == "bob's"
        assert (await store.nearest(ALICE, other_context, query, NOW)).prompt == "other context"


async def test_postgres_store_replaces_the_same_question(migrated):
    first, second = _with_similarity(0.5)
    newer = _entry(
        QUESTION,
        second,
        response=_response("Paris, France."),
        provider="gemini",
        created_at=NOW + timedelta(hours=1),
        expires_at=NOW + timedelta(hours=25),
    )
    async with open_pool(migrated, max_size=1) as pool:
        store = PostgresSemanticStore(pool)
        await store.put(_entry(QUESTION, first))
        await store.put(newer)
        candidate = await store.nearest(ALICE, CONTEXT, np.asarray(second, np.float32), NOW)
        async with pool.connection() as conn:
            cur = await conn.execute("SELECT count(*) FROM semantic_cache")
            assert (await cur.fetchone())[0] == 1
    assert candidate.entry == newer.entry
    assert candidate.similarity == pytest.approx(1.0, abs=1e-6)


async def test_postgres_store_deletes_only_expired_entries(migrated):
    first, second = _with_similarity(0.5)
    async with open_pool(migrated, max_size=1) as pool:
        store = PostgresSemanticStore(pool)
        await store.put(_entry("soon", first, expires_at=NOW + timedelta(seconds=60)))
        await store.put(_entry("later", second, expires_at=NOW + timedelta(seconds=120)))
        assert await store.delete_expired(NOW + timedelta(seconds=60)) == 1
        candidate = await store.nearest(ALICE, CONTEXT, np.asarray(first, np.float32), NOW)
    assert candidate.prompt == "later"


async def test_the_front_over_postgres_end_to_end(migrated):
    async with open_pool(migrated, max_size=2) as pool:
        cache = _cache(_embedder(0.97), PostgresSemanticStore(pool))
        await _store_answer(cache)
        entry = await _ask(cache)
        assert await _ask(cache, key_id=BOB) is None
        cache.close()
    assert entry.response.choices[0].message.content == "Paris."
