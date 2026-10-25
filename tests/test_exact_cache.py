"""The exact-match cache in the route: hits, misses, bypasses, and what each logs and costs.

Each gateway gets an in-memory cache of its own (an app's default caches nothing) and a rate
limiter on a frozen clock, so token headers are exact.
"""

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx2
import openai
import psycopg
import pytest
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

import app.main
from app.api.auth import get_authenticator
from app.api.routes import (
    get_model_router,
    get_rate_limiter,
    get_response_cache,
    get_usage_recorder,
)
from app.api.schemas import ChatRequest
from app.auth.authenticator import Authenticator
from app.auth.keys import InMemoryKeyStore
from app.auth.rate_limit import RateLimiter
from app.cache.key import cache_key
from app.cache.store import CacheEntry, InMemoryCacheStore, PostgresCacheStore, ResponseCache
from app.config import Settings
from app.db.migrate import apply_migrations
from app.db.pool import open_pool
from app.main import create_app
from app.providers.mock import MockProvider
from app.routing.registry import Price, Target
from app.routing.router import ModelRouter
from app.usage.log import InMemoryUsageLog, LogWriter, PostgresUsageLog, RequestLog
from app.usage.recorder import UsageRecorder
from scripts.create_key import create_key

# "hello there" is 3 prompt tokens to the mock, and its reply "Echo: hello there" is 5.
MESSAGES = [{"role": "user", "content": "hello there"}]
CHEAP = Price(input=Decimal("1.5"), output=Decimal("4"))


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


class Gateway:
    """An app with alias "fast" on the given mock providers (each on model "m"), an in-memory
    cache with a 1 h TTL on a clock the test moves, an in-memory request log, and two keys."""

    def __init__(
        self,
        *providers: tuple[str, MockProvider],
        rpm: int = 1_000,
        tpm: int = 100_000,
        store: InMemoryCacheStore | None = None,
        price: Callable[[Target], Price] = lambda _: CHEAP,
        other_aliases: dict[str, list[Target]] | None = None,
    ) -> None:
        self.app = create_app()
        self.providers = providers or (("groq", MockProvider()),)
        router = ModelRouter(
            {
                "fast": [Target(provider=name, model="m") for name, _ in self.providers],
                **(other_aliases or {}),
            },
            dict(self.providers),
            sleep=_no_sleep,
        )
        self.log = InMemoryUsageLog()
        self.writer = LogWriter(self.log)
        recorder = UsageRecorder(self.writer, price)
        self.store = store or InMemoryCacheStore()
        self.clock = Clock()
        self.cache = ResponseCache(self.store, ttl_s=3600, max_temperature=0.2, clock=self.clock)
        limiter = RateLimiter(clock=lambda: 1000.0)
        keys = InMemoryKeyStore()
        self.key, _ = keys.add("team-a", rpm_limit=rpm, tpm_limit=tpm)
        self.other_key, _ = keys.add("team-b", rpm_limit=rpm, tpm_limit=tpm)
        authenticator = Authenticator(keys)
        overrides = self.app.dependency_overrides
        overrides[get_model_router] = lambda: router
        overrides[get_usage_recorder] = lambda: recorder
        overrides[get_response_cache] = lambda: self.cache
        overrides[get_rate_limiter] = lambda: limiter
        overrides[get_authenticator] = lambda: authenticator

    @property
    def calls(self) -> int:
        return sum(p.calls for _, p in self.providers)

    async def post(
        self, body: dict, *, key: str | None = None, headers: dict | None = None
    ) -> httpx2.Response:
        transport = httpx2.ASGITransport(app=self.app, raise_app_exceptions=False)
        async with httpx2.AsyncClient(transport=transport, base_url="http://gateway.test") as c:
            sent = {"Authorization": f"Bearer {key or self.key}", **(headers or {})}
            resp = await c.post("/v1/chat/completions", json=body, headers=sent)
        # Stores run in the background: let them land before the next request.
        await self.cache.drain()
        return resp

    async def rows(self) -> list[RequestLog]:
        await self.writer.drain()
        return self.log.entries


async def _no_sleep(_: float) -> None:
    pass


def chat(**extra) -> dict:
    return {"model": "fast", "messages": MESSAGES, "temperature": 0, **extra}


async def test_a_repeated_request_is_served_from_the_cache():
    gw = Gateway()

    first = await gw.post(chat())
    second = await gw.post(chat())

    assert first.status_code == second.status_code == 200
    assert gw.calls == 1
    assert first.headers["x-gateway-cache"] == "miss"
    assert second.headers["x-gateway-cache"] == "exact"
    a, b = first.json(), second.json()
    assert b["choices"] == a["choices"]
    assert b["usage"] == a["usage"]
    assert b["model"] == a["model"]
    # A new completion as far as the client can tell.
    assert b["id"] != a["id"]
    assert b["id"].startswith("chatcmpl-")


async def test_a_hit_is_logged_as_exact_at_no_cost():
    gw = Gateway()

    await gw.post(chat())
    await gw.post(chat())
    miss, hit = await gw.rows()

    assert (miss.cache_status, miss.attempts) == ("miss", 1)
    # 3 x $1.5/1M + 5 x $4/1M
    assert miss.cost_usd == Decimal("0.0000245")
    assert (hit.cache_status, hit.status_code, hit.outcome) == ("exact", 200, "completed")
    assert (hit.prompt_tokens, hit.completion_tokens, hit.cost_usd) == (0, 0, Decimal(0))
    assert (hit.attempts, hit.fallback, hit.stream) == (0, False, False)
    # Who first answered, for the dashboard.
    assert (hit.provider, hit.model, hit.requested_model) == ("groq", "m", "fast")


async def test_the_entry_keeps_who_answered_and_what_it_cost():
    gw = Gateway()

    await gw.post(chat())

    (entry,) = gw.store.entries.values()
    assert (entry.alias, entry.provider, entry.model) == ("fast", "groq", "m")
    assert entry.cost_usd == Decimal("0.0000245")
    assert entry.response.choices[0].message.content == "Echo: hello there"


async def test_a_hit_names_the_original_provider_and_no_attempts():
    gw = Gateway()

    await gw.post(chat())
    hit = await gw.post(chat())

    assert hit.headers["x-gateway-provider"] == "groq"
    assert hit.headers["x-gateway-attempts"] == "0"
    assert hit.headers["x-gateway-fallback"] == "false"


async def test_a_fallback_answer_is_cached_too():
    gw = Gateway(("flaky", MockProvider(failure_rate=1.0)), ("groq", MockProvider()))

    await gw.post(chat())
    hit = await gw.post(chat())

    assert hit.headers["x-gateway-cache"] == "exact"
    assert hit.headers["x-gateway-provider"] == "groq"


async def test_one_cache_serves_every_key():
    gw = Gateway()

    await gw.post(chat(), key=gw.key)
    hit = await gw.post(chat(), key=gw.other_key)

    assert hit.headers["x-gateway-cache"] == "exact"
    assert gw.calls == 1
    _, row = await gw.rows()
    # Logged against the key that sent it.
    assert row.key_id == 2


async def test_a_different_request_is_a_miss():
    gw = Gateway()

    await gw.post(chat())
    other = await gw.post(chat(max_tokens=50))

    assert other.headers["x-gateway-cache"] == "miss"
    assert gw.calls == 2


async def test_without_a_temperature_the_cache_stays_out():
    gw = Gateway()

    first = await gw.post(chat(temperature=None))
    second = await gw.post(chat(temperature=None))

    assert first.headers["x-gateway-cache"] == second.headers["x-gateway-cache"] == "bypass"
    assert gw.calls == 2
    assert gw.store.entries == {}
    assert [r.cache_status for r in await gw.rows()] == ["miss", "miss"]


async def test_a_high_temperature_isnt_cached():
    gw = Gateway()

    await gw.post(chat(temperature=0.7))
    second = await gw.post(chat(temperature=0.7))

    assert second.headers["x-gateway-cache"] == "bypass"
    assert gw.calls == 2


async def test_cache_off_neither_stores_an_answer():
    gw = Gateway()

    off = await gw.post(chat(), headers={"x-gateway-cache": "off"})
    assert off.headers["x-gateway-cache"] == "bypass"
    assert gw.store.entries == {}

    after = await gw.post(chat())
    assert after.headers["x-gateway-cache"] == "miss"


async def test_cache_off_nor_reads_a_stored_one():
    gw = Gateway()
    await gw.post(chat())

    off = await gw.post(chat(), headers={"x-gateway-cache": "OFF"})

    assert off.headers["x-gateway-cache"] == "bypass"
    assert gw.calls == 2


async def test_a_failed_request_isnt_cached():
    gw = Gateway(("flaky", MockProvider(failure_rate=1.0)))

    first = await gw.post(chat())
    second = await gw.post(chat())

    assert first.status_code == second.status_code == 503
    assert gw.store.entries == {}
    assert [r.cache_status for r in await gw.rows()] == ["miss", "miss"]


async def test_a_reply_worth_nothing_isnt_cached():
    # An empty answer would be served again forever.
    gw = Gateway(("groq", MockProvider(reply="")))

    await gw.post(chat())
    second = await gw.post(chat())

    assert second.headers["x-gateway-cache"] == "miss"
    assert gw.store.entries == {}


async def test_a_broken_cache_never_fails_a_request():
    class BrokenStore(InMemoryCacheStore):
        async def get(self, key: bytes, now: datetime) -> CacheEntry | None:
            raise psycopg.OperationalError("connection refused")

        async def put(self, key: bytes, entry: CacheEntry) -> None:
            raise psycopg.OperationalError("connection refused")

    gw = Gateway(store=BrokenStore())

    first = await gw.post(chat())
    second = await gw.post(chat())

    assert first.status_code == second.status_code == 200
    assert second.headers["x-gateway-cache"] == "miss"
    assert gw.calls == 2


async def test_a_hit_counts_as_a_request_but_uses_no_tokens():
    gw = Gateway(rpm=3, tpm=100_000)

    first = await gw.post(chat(max_tokens=100))
    hit = await gw.post(chat(max_tokens=100))
    again = await gw.post(chat(max_tokens=100))
    limited = await gw.post(chat(max_tokens=100))

    # Each request holds its estimate while it runs (the headers are read then). The miss
    # used 8 tokens; the hit gave its whole reservation back, so the next request sees only
    # the miss's 8 spent. The clock is frozen: nothing refills.
    held = 100_000 - int(first.headers["x-ratelimit-remaining-tokens"])
    assert int(hit.headers["x-ratelimit-remaining-tokens"]) == 100_000 - 8 - held
    assert int(again.headers["x-ratelimit-remaining-tokens"]) == 100_000 - 8 - held
    # But the hits still count against the key's requests per minute.
    assert limited.status_code == 429


# Streams.


def events(resp: httpx2.Response) -> list[str]:
    """The `data:` payloads of a streamed response, exactly as sent."""
    assert resp.headers["content-type"].startswith("text/event-stream")
    return [e.removeprefix("data: ") for e in resp.text.strip().split("\n\n")]


def streamed_text(resp: httpx2.Response) -> str:
    chunks = [json.loads(e) for e in events(resp)[:-1]]
    return "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])


async def test_a_repeated_stream_is_replayed_from_the_cache():
    gw = Gateway()

    first = await gw.post(chat(stream=True))
    second = await gw.post(chat(stream=True))

    assert gw.calls == 1
    assert first.headers["x-gateway-cache"] == "miss"
    assert second.headers["x-gateway-cache"] == "exact"
    assert second.headers["x-gateway-attempts"] == "0"
    assert streamed_text(second) == streamed_text(first) == "Echo: hello there"


async def test_a_replay_is_shaped_like_a_provider_stream():
    gw = Gateway()
    await gw.post(chat())

    hit = await gw.post(chat(stream=True))
    *chunks, done = events(hit)
    chunks = [json.loads(c) for c in chunks]

    assert done == "[DONE]"
    # A role chunk, the text, then the finish_reason, all with one id.
    assert [c["choices"][0]["delta"] for c in chunks] == [
        {"role": "assistant", "content": ""},
        {"content": "Echo: hello there"},
        {},
    ]
    assert [c["choices"][0].get("finish_reason") for c in chunks] == [None, None, "stop"]
    assert len({c["id"] for c in chunks}) == 1
    assert all(c["object"] == "chat.completion.chunk" and c["model"] == "m" for c in chunks)
    # No usage unless the client asks for it, as on a provider stream.
    assert all("usage" not in c for c in chunks)


async def test_a_replay_sends_usage_when_asked():
    gw = Gateway()
    await gw.post(chat())

    hit = await gw.post(chat(stream=True, stream_options={"include_usage": True}))
    last = json.loads(events(hit)[-2])

    assert last["choices"] == []
    assert last["usage"] == {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}


async def test_a_streamed_answer_serves_a_plain_request():
    gw = Gateway()
    await gw.post(chat(stream=True, stream_options={"include_usage": True}))

    hit = await gw.post(chat())

    assert hit.headers["x-gateway-cache"] == "exact"
    assert gw.calls == 1
    body = hit.json()
    assert body["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Echo: hello there"},
            "finish_reason": "stop",
        }
    ]
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}


async def test_a_streamed_answer_is_stored_with_its_cost():
    gw = Gateway()

    await gw.post(chat(stream=True))

    (entry,) = gw.store.entries.values()
    assert (entry.provider, entry.model) == ("groq", "m")
    # 3 x $1.5/1M + 5 x $4/1M, from the stream's usage.
    assert entry.cost_usd == Decimal("0.0000245")


async def test_a_stream_hit_is_logged_as_exact():
    gw = Gateway()

    await gw.post(chat(stream=True))
    await gw.post(chat(stream=True))
    miss, hit = await gw.rows()

    assert (miss.cache_status, miss.stream) == ("miss", True)
    assert (hit.cache_status, hit.stream, hit.status_code, hit.outcome) == (
        "exact",
        True,
        200,
        "completed",
    )
    assert (hit.prompt_tokens, hit.completion_tokens, hit.cost_usd) == (0, 0, Decimal(0))
    assert hit.attempts == 0


async def test_a_stream_cut_partway_isnt_cached():
    gw = Gateway(("midstream", MockProvider(fail_after_chunks=3)))

    cut = await gw.post(chat(stream=True))
    again = await gw.post(chat(stream=True))

    assert "error" in json.loads(events(cut)[-1])
    assert again.headers["x-gateway-cache"] == "miss"
    assert gw.store.entries == {}


async def test_cache_off_keeps_a_stream_out_of_the_cache():
    gw = Gateway()

    off = await gw.post(chat(stream=True), headers={"x-gateway-cache": "off"})

    assert off.headers["x-gateway-cache"] == "bypass"
    assert gw.store.entries == {}


async def test_a_stream_the_client_left_isnt_cached():
    gw = Gateway(("groq", MockProvider(reply="one two three four")))
    body = json.dumps(chat(stream=True)).encode()
    received = [{"type": "http.request", "body": body, "more_body": False}]
    sent: list[dict] = []

    async def receive():
        if received:
            return received.pop(0)
        await asyncio.Event().wait()

    async def send(message):
        if message["type"] == "http.response.body" and sent:
            # The client is gone after the first chunk (ASGI 2.4: send() raises OSError).
            raise OSError("connection lost")
        if message["type"] == "http.response.body":
            sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"content-type", b"application/json"),
            (b"host", b"gateway.test"),
            (b"authorization", f"Bearer {gw.key}".encode()),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("gateway.test", 80),
    }

    with pytest.raises(ClientDisconnect):
        await gw.app(scope, receive, send)
    await gw.cache.drain()

    assert gw.store.entries == {}
    (row,) = await gw.rows()
    assert (row.outcome, row.cache_status) == ("client_disconnect", "miss")


async def test_a_stream_no_provider_answered_isnt_cached():
    gw = Gateway(("flaky", MockProvider(failure_rate=1.0)))

    resp = await gw.post(chat(stream=True))

    assert resp.status_code == 503
    assert gw.store.entries == {}


# Expiry, refusals and failures around the cache.


async def test_an_expired_answer_is_asked_for_again_and_stored_anew():
    gw = Gateway()
    await gw.post(chat())

    gw.clock.now += timedelta(hours=1)
    expired = await gw.post(chat())
    gw.clock.now += timedelta(minutes=30)
    fresh = await gw.post(chat())

    assert expired.headers["x-gateway-cache"] == "miss"
    # The new answer got a TTL of its own, from when it was stored.
    assert fresh.headers["x-gateway-cache"] == "exact"
    assert gw.calls == 2


async def test_an_alias_gone_from_the_registry_isnt_served_from_the_cache():
    gw = Gateway()
    await gw.post(chat())
    # Pretend the alias was renamed: its answers are still stored, under "gone".
    (entry,) = gw.store.entries.values()
    gone = ChatRequest.model_validate(chat(model="gone"))
    gw.store.entries[cache_key(gone)] = entry

    resp = await gw.post(chat(model="gone"))

    assert resp.status_code == 404
    _, row = await gw.rows()
    assert (row.outcome, row.cache_status) == ("rejected", "miss")


async def test_an_alias_without_a_provider_here_isnt_served_from_the_cache():
    orphan = {"orphan": [Target(provider="not-configured", model="m")]}
    gw = Gateway(other_aliases=orphan)
    await gw.post(chat())
    (entry,) = gw.store.entries.values()
    gw.store.entries[cache_key(ChatRequest.model_validate(chat(model="orphan")))] = entry

    resp = await gw.post(chat(model="orphan"))

    assert resp.status_code == 503


async def test_a_rate_limited_request_never_reaches_the_cache():
    class CountingStore(InMemoryCacheStore):
        lookups = 0

        async def get(self, key: bytes, now: datetime) -> CacheEntry | None:
            self.lookups += 1
            return await super().get(key, now)

    store = CountingStore()
    gw = Gateway(rpm=1, store=store)

    await gw.post(chat())
    limited = await gw.post(chat())

    assert limited.status_code == 429
    assert store.lookups == 1
    _, row = await gw.rows()
    assert (row.outcome, row.cache_status) == ("rejected", "miss")


async def test_a_reply_that_cant_be_priced_still_reaches_the_client(caplog):
    def no_price(target: Target) -> Price:
        raise KeyError(target.provider)

    gw = Gateway(price=no_price)

    with caplog.at_level(logging.WARNING):
        resp = await gw.post(chat())

    assert resp.status_code == 200
    assert gw.store.entries == {}
    assert "could not price an answer for the cache" in caplog.text


async def test_identical_requests_at_once_both_call_the_provider():
    # A known limit: nothing holds the second request until the first one's answer is
    # stored. Both answers are stored under the one key; the last one stays.
    gw = Gateway(("groq", MockProvider(latency_s=0.05)))

    first, second = await asyncio.gather(gw.post(chat()), gw.post(chat()))

    assert first.headers["x-gateway-cache"] == second.headers["x-gateway-cache"] == "miss"
    assert gw.calls == 2
    assert len(gw.store.entries) == 1


# The openai SDK, the gateway's compatibility contract.


def sdk(gw: Gateway) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        api_key=gw.key,
        base_url="http://gateway.test/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.ASGITransport(app=gw.app)),
        max_retries=0,
    )


async def test_the_sdk_reads_hits_plain_and_streamed():
    gw = Gateway()
    client = sdk(gw)

    first = await client.chat.completions.create(model="fast", messages=MESSAGES, temperature=0)
    await gw.cache.drain()
    again = await client.chat.completions.with_raw_response.create(
        model="fast", messages=MESSAGES, temperature=0
    )
    stream = await client.chat.completions.create(
        model="fast",
        messages=MESSAGES,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks = [chunk async for chunk in stream]

    assert again.headers["x-gateway-cache"] == "exact"
    assert again.parse().choices[0].message.content == first.choices[0].message.content
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == "Echo: hello there"
    assert chunks[-1].usage.total_tokens == 8
    assert gw.calls == 1


# The whole path, through Postgres.


def test_hits_come_from_postgres_on_the_real_startup(pg_conninfo, monkeypatch):
    with psycopg.connect(pg_conninfo) as conn:
        apply_migrations(conn)
        key, record = create_key(conn, "alice", 60, 10_000)
    # The real startup: a real pool, the Postgres key store, request log and cache, the
    # shipped registry, on the test schema.
    monkeypatch.setattr(app.main, "open_pool", open_pool)
    monkeypatch.setattr(app.main, "PostgresUsageLog", PostgresUsageLog)
    monkeypatch.setattr(app.main, "PostgresCacheStore", PostgresCacheStore)
    monkeypatch.setattr(
        app.main, "get_settings", lambda: Settings(_env_file=None, database_url=pg_conninfo)
    )
    headers = {"Authorization": f"Bearer {key}"}
    body = {"model": "mock", "messages": MESSAGES, "temperature": 0}

    # Two runs of the app: shutting down the first waits for its cache store, and the second
    # starts by deleting expired entries.
    with TestClient(create_app()) as client:
        first = client.post("/v1/chat/completions", json=body, headers=headers)
    with TestClient(create_app()) as client:
        second = client.post("/v1/chat/completions", json=body, headers=headers)

    assert first.headers["x-gateway-cache"] == "miss"
    assert second.headers["x-gateway-cache"] == "exact"
    assert second.json()["choices"] == first.json()["choices"]
    with psycopg.connect(pg_conninfo) as conn:
        (entries,) = conn.execute("SELECT count(*) FROM response_cache").fetchone()
        rows = conn.execute(
            "SELECT key_id, cache_status, provider, attempts FROM request_logs ORDER BY id"
        ).fetchall()
    assert entries == 1
    assert rows == [(record.id, "miss", "mock", 1), (record.id, "exact", "mock", 0)]
