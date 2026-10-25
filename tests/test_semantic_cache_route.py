"""The semantic cache in the route: opting in, hits and misses, streams, and what each logs.

Each gateway is the exact-cache tests' `Gateway` plus an in-memory semantic cache with a fake
embedder that puts QUESTION and REPHRASED at similarity 0.97 (above the 0.95 threshold).
"""

import json
from decimal import Decimal

import numpy as np
import psycopg
from fastapi.testclient import TestClient

import app.main
from app.api.routes import get_semantic_cache
from app.cache.embedder import DIMENSIONS, FakeEmbedder
from app.cache.semantic_store import (
    InMemorySemanticStore,
    PostgresSemanticStore,
    SemanticCache,
)
from app.cache.store import PostgresCacheStore
from app.config import Settings
from app.db.migrate import apply_migrations
from app.db.pool import open_pool
from app.main import create_app
from app.providers.mock import MockProvider
from app.usage.log import PostgresUsageLog
from scripts.create_key import create_key
from tests.test_exact_cache import Gateway, events, sdk, streamed_text

QUESTION = "What's the capital of France?"
REPHRASED = "capital of france?"
OPT_IN = {"x-gateway-cache": "semantic"}


def _similar(a: str, b: str, similarity: float = 0.97) -> FakeEmbedder:
    first = [1.0] + [0.0] * (DIMENSIONS - 1)
    second = [similarity, float(np.sqrt(1 - similarity**2))] + [0.0] * (DIMENSIONS - 2)
    return FakeEmbedder({a: first, b: second})


class SemanticGateway(Gateway):
    def __init__(self, *providers, embedder: FakeEmbedder | None = None, **kwargs) -> None:
        super().__init__(*providers, **kwargs)
        self.embedder = embedder if embedder is not None else _similar(QUESTION, REPHRASED)
        self.semantic_store = InMemorySemanticStore()
        self.semantic = SemanticCache(
            self.semantic_store,
            self.embedder,
            ttl_s=3600,
            threshold=0.95,
            max_temperature=0.2,
            clock=self.clock,
        )
        self.app.dependency_overrides[get_semantic_cache] = lambda: self.semantic

    async def post(self, body, *, key=None, headers=None):
        resp = await super().post(body, key=key, headers=headers)
        await self.semantic.drain()
        return resp


def ask(question: str, **extra) -> dict:
    return {
        "model": "fast",
        "messages": [{"role": "user", "content": question}],
        "temperature": 0,
        **extra,
    }


async def test_a_similar_question_gets_the_stored_answer():
    gw = SemanticGateway()

    first = await gw.post(ask(QUESTION), headers=OPT_IN)
    second = await gw.post(ask(REPHRASED), headers=OPT_IN)

    assert gw.calls == 1
    assert first.headers["x-gateway-cache"] == "miss"
    assert second.headers["x-gateway-cache"] == "semantic"
    # The answer to the first question, as first served.
    assert second.json()["choices"] == first.json()["choices"]
    assert second.json()["choices"][0]["message"]["content"] == f"Echo: {QUESTION}"
    assert second.json()["id"] != first.json()["id"]
    # Who first answered; no provider was called this time.
    assert second.headers["x-gateway-provider"] == "groq"
    assert second.headers["x-gateway-attempts"] == "0"


async def test_a_semantic_hit_is_logged_as_semantic_at_no_cost_and_saves_the_first_call():
    gw = SemanticGateway()

    await gw.post(ask(QUESTION), headers=OPT_IN)
    await gw.post(ask(REPHRASED), headers=OPT_IN)
    miss, hit = await gw.rows()

    assert (miss.cache_status, miss.saved_usd) == ("miss", None)
    assert (hit.cache_status, hit.status_code, hit.outcome) == ("semantic", 200, "completed")
    assert (hit.prompt_tokens, hit.completion_tokens, hit.cost_usd, hit.attempts) == (0, 0, 0, 0)
    assert hit.saved_usd == miss.cost_usd > 0
    assert (hit.provider, hit.model, hit.requested_model) == ("groq", "m", "fast")


async def test_without_opting_in_a_similar_question_is_a_miss_and_nothing_is_embedded():
    gw = SemanticGateway()

    await gw.post(ask(QUESTION))
    second = await gw.post(ask(REPHRASED))

    assert gw.calls == 2
    assert second.headers["x-gateway-cache"] == "miss"
    assert gw.embedder.calls == []
    assert gw.semantic_store.entries == {}


async def test_an_answer_stored_without_opting_in_isnt_served_to_one_that_does():
    gw = SemanticGateway()

    await gw.post(ask(QUESTION))
    second = await gw.post(ask(REPHRASED), headers=OPT_IN)

    assert gw.calls == 2
    assert second.headers["x-gateway-cache"] == "miss"


async def test_the_exact_cache_is_checked_first_and_saves_the_embedding():
    gw = SemanticGateway()

    await gw.post(ask(QUESTION), headers=OPT_IN)
    second = await gw.post(ask(QUESTION), headers=OPT_IN)

    assert second.headers["x-gateway-cache"] == "exact"
    assert gw.embedder.calls == [QUESTION]
    assert [r.cache_status for r in await gw.rows()] == ["miss", "exact"]


async def test_a_semantic_hit_isnt_copied_into_the_exact_cache():
    # A fuzzy match must not come back later looking like an exact one.
    gw = SemanticGateway()

    await gw.post(ask(QUESTION), headers=OPT_IN)
    await gw.post(ask(REPHRASED), headers=OPT_IN)
    again = await gw.post(ask(REPHRASED), headers=OPT_IN)

    assert len(gw.store.entries) == 1
    assert again.headers["x-gateway-cache"] == "semantic"
    assert len(gw.semantic_store.entries) == 1


async def test_another_keys_answer_is_never_served():
    gw = SemanticGateway()

    await gw.post(ask(QUESTION), headers=OPT_IN)
    same = await gw.post(ask(REPHRASED), headers=OPT_IN)
    other = await gw.post(ask(REPHRASED), key=gw.other_key, headers=OPT_IN)

    assert same.headers["x-gateway-cache"] == "semantic"
    assert other.headers["x-gateway-cache"] == "miss"
    assert other.json()["choices"][0]["message"]["content"] == f"Echo: {REPHRASED}"
    assert gw.calls == 2


async def test_another_keys_identical_request_is_still_an_exact_hit():
    # The exact cache stays shared, as in Phase 6: an identical request gets the answer it
    # would have got anyway, so sharing it leaks nothing.
    gw = SemanticGateway()

    await gw.post(ask(QUESTION), headers=OPT_IN)
    other = await gw.post(ask(QUESTION), key=gw.other_key, headers=OPT_IN)

    assert other.headers["x-gateway-cache"] == "exact"


async def test_a_similar_question_in_another_context_is_a_miss():
    gw = SemanticGateway()

    await gw.post(ask(QUESTION), headers=OPT_IN)
    longer = await gw.post(ask(REPHRASED, max_tokens=500), headers=OPT_IN)

    assert longer.headers["x-gateway-cache"] == "miss"
    assert gw.calls == 2


async def test_a_match_with_other_numbers_is_a_miss():
    a, b = "Is 91 a prime number?", "Is 97 a prime number?"
    gw = SemanticGateway(embedder=_similar(a, b, 0.99))

    await gw.post(ask(a), headers=OPT_IN)
    second = await gw.post(ask(b), headers=OPT_IN)

    assert second.headers["x-gateway-cache"] == "miss"
    assert second.json()["choices"][0]["message"]["content"] == f"Echo: {b}"


async def test_a_question_below_the_threshold_is_a_miss():
    gw = SemanticGateway(embedder=_similar(QUESTION, REPHRASED, 0.9))

    await gw.post(ask(QUESTION), headers=OPT_IN)
    second = await gw.post(ask(REPHRASED), headers=OPT_IN)

    assert second.headers["x-gateway-cache"] == "miss"


async def test_requests_the_cache_cant_answer_bypass_it_unembedded():
    gw = SemanticGateway()

    no_temperature = ask(QUESTION)
    del no_temperature["temperature"]
    multi_turn = ask(QUESTION)
    multi_turn["messages"] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": QUESTION},
    ]
    bypassed = await gw.post(no_temperature, headers=OPT_IN)
    missed = await gw.post(multi_turn, headers=OPT_IN)

    assert bypassed.headers["x-gateway-cache"] == "bypass"
    # Not single-turn: the exact cache still serves it, the semantic one stays out.
    assert missed.headers["x-gateway-cache"] == "miss"
    assert gw.embedder.calls == []
    assert gw.semantic_store.entries == {}


async def test_cache_off_keeps_both_caches_out():
    gw = SemanticGateway()

    await gw.post(ask(QUESTION), headers=OPT_IN)
    off = await gw.post(ask(REPHRASED), headers={"x-gateway-cache": "off"})

    assert off.headers["x-gateway-cache"] == "bypass"
    assert gw.calls == 2
    assert gw.embedder.calls == [QUESTION]


async def test_without_an_embedder_requests_that_opt_in_are_served_as_usual():
    # The model didn't load at startup: the semantic cache is off, the exact one works.
    gw = SemanticGateway()
    gw.semantic.embedder = None

    first = await gw.post(ask(QUESTION), headers=OPT_IN)
    second = await gw.post(ask(REPHRASED), headers=OPT_IN)
    again = await gw.post(ask(REPHRASED), headers=OPT_IN)

    assert [r.headers["x-gateway-cache"] for r in (first, second, again)] == [
        "miss",
        "miss",
        "exact",
    ]


async def test_a_broken_semantic_cache_never_fails_a_request():
    class Broken(InMemorySemanticStore):
        async def nearest(self, *args):
            raise OSError("database down")

        async def put(self, entry):
            raise OSError("database down")

    gw = SemanticGateway()
    gw.semantic.store = Broken()

    first = await gw.post(ask(QUESTION), headers=OPT_IN)
    second = await gw.post(ask(REPHRASED), headers=OPT_IN)

    assert first.status_code == second.status_code == 200
    assert second.headers["x-gateway-cache"] == "miss"
    assert gw.calls == 2


async def test_a_hit_counts_as_a_request_but_uses_no_tokens():
    gw = SemanticGateway()

    first = await gw.post(ask(QUESTION), headers=OPT_IN)
    hit = await gw.post(ask(REPHRASED), headers=OPT_IN)

    assert int(hit.headers["x-ratelimit-remaining-requests"]) == (
        int(first.headers["x-ratelimit-remaining-requests"]) - 1
    )
    # The first request's tokens are settled by now; the hit took none.
    rows = await gw.rows()
    assert rows[1].prompt_tokens == rows[1].completion_tokens == 0


# Streams.


async def test_a_semantic_hit_on_a_stream_is_replayed():
    gw = SemanticGateway()

    await gw.post(ask(QUESTION), headers=OPT_IN)
    hit = await gw.post(
        ask(REPHRASED, stream=True, stream_options={"include_usage": True}), headers=OPT_IN
    )

    assert hit.headers["x-gateway-cache"] == "semantic"
    assert streamed_text(hit) == f"Echo: {QUESTION}"
    *chunks, done = events(hit)
    assert done == "[DONE]"
    assert json.loads(chunks[-1])["usage"]["total_tokens"] > 0
    assert gw.calls == 1
    assert (await gw.rows())[1].cache_status == "semantic"


async def test_a_streamed_answer_is_stored_for_similar_questions():
    gw = SemanticGateway()

    first = await gw.post(ask(QUESTION, stream=True), headers=OPT_IN)
    second = await gw.post(ask(REPHRASED), headers=OPT_IN)

    assert first.headers["x-gateway-cache"] == "miss"
    assert second.headers["x-gateway-cache"] == "semantic"
    assert second.json()["choices"][0]["message"]["content"] == f"Echo: {QUESTION}"
    # The embedding made for the lookup was reused to store it.
    assert gw.embedder.calls == [QUESTION, REPHRASED]


async def test_a_stream_cut_partway_isnt_stored():
    gw = SemanticGateway(("groq", MockProvider(fail_after_chunks=1)))

    await gw.post(ask(QUESTION, stream=True), headers=OPT_IN)

    assert gw.semantic_store.entries == {}


# The openai SDK.


async def test_the_sdk_reads_a_semantic_hit_plain_and_streamed():
    gw = SemanticGateway()
    client = sdk(gw)

    await client.chat.completions.create(
        model="fast",
        messages=[{"role": "user", "content": QUESTION}],
        temperature=0,
        extra_headers=OPT_IN,
    )
    await gw.semantic.drain()
    await gw.cache.drain()
    hit = await client.chat.completions.with_raw_response.create(
        model="fast",
        messages=[{"role": "user", "content": REPHRASED}],
        temperature=0,
        extra_headers=OPT_IN,
    )
    stream = await client.chat.completions.create(
        model="fast",
        messages=[{"role": "user", "content": REPHRASED}],
        temperature=0,
        stream=True,
        extra_headers=OPT_IN,
    )
    text = "".join([c.choices[0].delta.content or "" async for c in stream if c.choices])

    assert hit.headers["x-gateway-cache"] == "semantic"
    assert hit.parse().choices[0].message.content == f"Echo: {QUESTION}"
    # The plain hit stored nothing: the stream is a semantic hit too, not an exact one.
    assert text == f"Echo: {QUESTION}"
    assert gw.calls == 1


# The whole path, through Postgres.


def test_semantic_hits_come_from_postgres_on_the_real_startup(pg_conninfo, monkeypatch):
    with psycopg.connect(pg_conninfo) as conn:
        apply_migrations(conn)
        key, record = create_key(conn, "alice", 1000, 100_000)
        other, _ = create_key(conn, "bob", 1000, 100_000)
    # The real startup on the test schema, but conftest's fake embedder instead of MiniLM.
    monkeypatch.setattr(app.main, "open_pool", open_pool)
    monkeypatch.setattr(app.main, "PostgresUsageLog", PostgresUsageLog)
    monkeypatch.setattr(app.main, "PostgresCacheStore", PostgresCacheStore)
    monkeypatch.setattr(app.main, "PostgresSemanticStore", PostgresSemanticStore)
    monkeypatch.setattr(app.main, "load_embedder", lambda: _similar(QUESTION, REPHRASED))
    monkeypatch.setattr(
        app.main, "get_settings", lambda: Settings(_env_file=None, database_url=pg_conninfo)
    )

    def post(client, question, api_key):
        headers = {"Authorization": f"Bearer {api_key}", **OPT_IN}
        body = {**ask(question), "model": "mock"}
        return client.post("/v1/chat/completions", json=body, headers=headers)

    # Two runs: shutting down the first waits for its stores.
    with TestClient(create_app()) as client:
        first = post(client, QUESTION, key)
    with TestClient(create_app()) as client:
        hit = post(client, REPHRASED, key)
        other_key = post(client, REPHRASED, other)

    assert first.headers["x-gateway-cache"] == "miss"
    assert hit.headers["x-gateway-cache"] == "semantic"
    assert hit.json()["choices"] == first.json()["choices"]
    assert other_key.headers["x-gateway-cache"] == "miss"
    with psycopg.connect(pg_conninfo) as conn:
        entries = conn.execute("SELECT key_id, prompt FROM semantic_cache ORDER BY key_id")
        rows = conn.execute(
            "SELECT key_id, cache_status, attempts, saved_usd FROM request_logs ORDER BY id"
        ).fetchall()
        entries = entries.fetchall()
    # Bob's miss is stored under Bob.
    assert entries == [(record.id, QUESTION), (record.id + 1, REPHRASED)]
    # The mock model is free: the hit saved $0, which is known, unlike a miss's NULL.
    assert rows == [
        (record.id, "miss", 1, None),
        (record.id, "semantic", 0, Decimal(0)),
        (record.id + 1, "miss", 1, None),
    ]
