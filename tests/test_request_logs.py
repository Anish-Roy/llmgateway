"""Request logs as the route writes them: one row per way a request can end, with its cost.

Mock providers count about 4 characters a token, and the conftest turns tiktoken off, so the
gateway's own estimates are chars/4 too: every token count and cost below is exact.
"""

import asyncio
import json
import logging
from decimal import Decimal

import httpx2
import psycopg
import pytest
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

import app.main
from app.api.routes import get_model_router, get_usage_recorder
from app.api.schemas import ChatRequest, ChatResponse, Usage
from app.auth.keys import ApiKey
from app.config import Settings
from app.db.migrate import apply_migrations
from app.db.pool import open_pool
from app.main import create_app
from app.providers.mock import MockProvider
from app.routing.registry import Price, Target
from app.routing.router import ModelRouter
from app.usage.cost import cost
from app.usage.log import InMemoryUsageLog, LogWriter, PostgresUsageLog, RequestLog
from app.usage.recorder import UsageRecorder
from scripts.create_key import create_key
from tests.auth_helpers import authorize

# "hello there" is 3 prompt tokens to the mock, and its reply "Echo: hello there" is 5.
MESSAGES = [{"role": "user", "content": "hello there"}]
CHEAP = Price(input=Decimal("1.5"), output=Decimal("4"))
PRICEY = Price(input=Decimal("10"), output=Decimal("30"))


class Gateway:
    """An app whose aliases each name one or more (provider name, provider) pairs, every pair
    on model "m", with its rows kept in memory."""

    def __init__(
        self,
        aliases: dict[str, list[tuple[str, MockProvider]]],
        prices: dict[str, Price] | None = None,
        *,
        rpm: int = 1_000,
        store=None,
    ) -> None:
        self.app = create_app()
        providers = {name: p for targets in aliases.values() for name, p in targets}
        router = ModelRouter(
            {alias: [Target(provider=n, model="m") for n, _ in t] for alias, t in aliases.items()},
            providers,
            sleep=_no_sleep,
        )
        self.store = store or InMemoryUsageLog()
        self.writer = LogWriter(self.store)
        prices = prices or {}
        recorder = UsageRecorder(self.writer, lambda t: prices[t.provider])
        self.app.dependency_overrides[get_model_router] = lambda: router
        self.app.dependency_overrides[get_usage_recorder] = lambda: recorder
        self.key = authorize(self.app, rpm_limit=rpm)

    async def post(self, body: dict, *, key: str | None = None) -> httpx2.Response:
        transport = httpx2.ASGITransport(app=self.app, raise_app_exceptions=False)
        async with httpx2.AsyncClient(transport=transport, base_url="http://gateway.test") as c:
            headers = {"Authorization": f"Bearer {key or self.key}"}
            return await c.post("/v1/chat/completions", json=body, headers=headers)

    async def rows(self) -> list[RequestLog]:
        await self.writer.drain()
        return self.store.entries

    async def row(self) -> RequestLog:
        rows = await self.rows()
        assert len(rows) == 1
        return rows[0]


async def _no_sleep(_: float) -> None:
    pass


def chat(model: str = "fast", **extra) -> dict:
    return {"model": model, "messages": MESSAGES, **extra}


def tokens(row: RequestLog) -> Usage:
    return Usage(
        prompt_tokens=row.prompt_tokens,
        completion_tokens=row.completion_tokens,
        total_tokens=row.prompt_tokens + row.completion_tokens,
    )


# Requests that got an answer.


async def test_a_reply_is_logged_with_its_cost():
    gw = Gateway({"fast": [("groq", MockProvider(latency_s=0.05))]}, {"groq": CHEAP})

    resp = await gw.post(chat())
    row = await gw.row()

    assert resp.status_code == 200
    assert row.key_id == 1
    assert (row.requested_model, row.provider, row.model) == ("fast", "groq", "m")
    assert (row.stream, row.status_code, row.outcome) == (False, 200, "completed")
    assert (row.prompt_tokens, row.completion_tokens, row.usage_estimated) == (3, 5, False)
    # 3 x $1.5/1M + 5 x $4/1M
    assert row.cost_usd == Decimal("0.0000245")
    assert (row.attempts, row.fallback, row.cache_status) == (1, False, "miss")
    assert row.latency_ms >= 50
    assert row.ttft_ms is None
    assert row.created_at.tzinfo is not None


async def test_a_reply_without_usage_is_logged_with_an_estimate():
    class NoUsage(MockProvider):
        async def chat(self, request: ChatRequest) -> ChatResponse:
            response = await super().chat(request)
            return response.model_copy(update={"usage": Usage()})

    gw = Gateway({"fast": [("groq", NoUsage())]}, {"groq": CHEAP})

    await gw.post(chat())
    row = await gw.row()

    # chars/4, rounded up: "hello there" (11) and "Echo: hello there" (17).
    assert (row.prompt_tokens, row.completion_tokens, row.usage_estimated) == (3, 5, True)
    assert row.cost_usd == Decimal("0.0000245")


async def test_a_completed_stream_is_logged_with_its_ttft():
    gw = Gateway({"fast": [("groq", MockProvider(chunk_latency_s=0.02))]}, {"groq": CHEAP})

    resp = await gw.post(chat(stream=True))
    row = await gw.row()

    assert resp.text.endswith("data: [DONE]\n\n")
    assert (row.stream, row.status_code, row.outcome) == (True, 200, "completed")
    assert (row.prompt_tokens, row.completion_tokens, row.usage_estimated) == (3, 5, False)
    assert row.cost_usd == Decimal("0.0000245")
    # The first word came after the role-only chunk and one wait; the rest took longer.
    assert 20 <= row.ttft_ms < row.latency_ms


async def test_a_fallback_is_charged_at_the_model_that_answered():
    gw = Gateway(
        {"fast": [("flaky", MockProvider(failure_rate=1.0)), ("mock", MockProvider())]},
        {"flaky": PRICEY, "mock": CHEAP},
    )

    resp = await gw.post(chat())
    row = await gw.row()

    assert resp.status_code == 200
    assert (row.provider, row.outcome, row.fallback) == ("mock", "completed", True)
    # flaky's 503 is retried once, then mock answers. Failed calls cost nothing.
    assert row.attempts == 3
    assert row.cost_usd == cost(CHEAP, tokens(row)) == Decimal("0.0000245")


# Streams that didn't finish.


async def test_a_stream_cut_mid_answer_is_a_200_with_a_provider_error():
    # Chunks: role, "Echo: ", "hello ", then the failure.
    gw = Gateway({"fast": [("groq", MockProvider(fail_after_chunks=3))]}, {"groq": CHEAP})

    resp = await gw.post(chat(stream=True))
    row = await gw.row()

    assert resp.status_code == 200
    assert '"error"' in resp.text
    assert (row.status_code, row.outcome, row.provider) == (200, "provider_error", "groq")
    # The client got "Echo: hello " (12 characters, 3 tokens) before the stream broke: that's
    # what's charged, marked as an estimate.
    assert (row.prompt_tokens, row.completion_tokens, row.usage_estimated) == (3, 3, True)
    assert row.cost_usd == Decimal("0.0000165")
    assert row.ttft_ms is not None


async def test_a_client_that_leaves_mid_stream_is_logged():
    gw = Gateway({"fast": [("groq", MockProvider(reply="one two three four"))]}, {"groq": CHEAP})
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
    row = await gw.row()

    assert (row.stream, row.status_code, row.outcome) == (True, 200, "client_disconnect")
    assert row.usage_estimated
    assert row.cost_usd == cost(CHEAP, tokens(row))


# Requests no provider answered: logged, at no cost.


async def test_every_provider_failing_is_logged_with_its_attempts():
    gw = Gateway({"fast": [("flaky", MockProvider(failure_rate=1.0))]}, {"flaky": PRICEY})

    resp = await gw.post(chat())
    row = await gw.row()

    assert resp.status_code == 503
    assert (row.status_code, row.outcome) == (503, "provider_error")
    assert (row.provider, row.model) == (None, None)
    assert row.attempts == 2  # one try and one retry
    assert (row.prompt_tokens, row.completion_tokens, row.cost_usd) == (0, 0, 0)


async def test_a_stream_no_provider_started_is_logged():
    gw = Gateway({"fast": [("flaky", MockProvider(failure_rate=1.0))]}, {"flaky": PRICEY})

    resp = await gw.post(chat(stream=True))
    row = await gw.row()

    assert resp.status_code == 503
    assert (row.stream, row.status_code, row.outcome, row.attempts) == (
        True,
        503,
        "provider_error",
        2,
    )
    assert row.cost_usd == 0


async def test_a_provider_400_is_logged_as_its_status():
    gw = Gateway(
        {"fast": [("groq", MockProvider(failure_rate=1.0, fail_status=400))]}, {"groq": CHEAP}
    )

    await gw.post(chat())
    row = await gw.row()

    # Not retried: another provider would reject it too.
    assert (row.status_code, row.outcome, row.attempts) == (400, "provider_error", 1)


async def test_a_gateway_429_is_logged_as_rejected():
    gw = Gateway({"fast": [("groq", MockProvider())]}, {"groq": CHEAP}, rpm=1)

    await gw.post(chat())
    resp = await gw.post(chat())
    rows = await gw.rows()

    assert resp.status_code == 429
    assert [r.status_code for r in rows] == [200, 429]
    assert (rows[1].outcome, rows[1].attempts, rows[1].provider, rows[1].cost_usd) == (
        "rejected",
        0,
        None,
        0,
    )


async def test_an_unknown_model_is_logged_as_rejected():
    gw = Gateway({"fast": [("groq", MockProvider())]}, {"groq": CHEAP})

    resp = await gw.post(chat("gpt-7"))
    row = await gw.row()

    assert resp.status_code == 404
    assert (row.requested_model, row.status_code, row.outcome, row.attempts) == (
        "gpt-7",
        404,
        "rejected",
        0,
    )


async def test_an_alias_with_no_configured_provider_is_logged_as_rejected():
    gw = Gateway({"fast": [("groq", MockProvider())]}, {"groq": CHEAP})
    # "strong" names a provider this deployment doesn't have (say, no API key for it).
    router = ModelRouter(
        {"strong": [Target(provider="gemini", model="m")]}, {"groq": MockProvider()}
    )
    gw.app.dependency_overrides[get_model_router] = lambda: router

    resp = await gw.post(chat("strong"))
    row = await gw.row()

    assert resp.status_code == 503
    assert (row.outcome, row.attempts) == ("rejected", 0)


async def test_a_gateway_bug_is_logged_as_a_gateway_error():
    class Broken(MockProvider):
        async def chat(self, request: ChatRequest) -> ChatResponse:
            raise RuntimeError("bug")

    gw = Gateway({"fast": [("groq", Broken())]}, {"groq": CHEAP})

    resp = await gw.post(chat())
    row = await gw.row()

    assert resp.status_code == 500
    assert (row.status_code, row.outcome) == (500, "gateway_error")


async def test_a_cancelled_request_is_logged_as_a_client_disconnect():
    # Cancellation can't be sent through the transport, so it's reported directly.
    writer = LogWriter(InMemoryUsageLog())
    recorder = UsageRecorder(writer, lambda t: CHEAP)
    key = ApiKey(1, "test", "gw-abc", 60, 1000, True, None)

    recorder.start(key, ChatRequest(model="fast", messages=MESSAGES)).failed(
        asyncio.CancelledError()
    )
    await writer.drain()

    [row] = writer.store.entries
    assert (row.status_code, row.outcome, row.cost_usd) == (499, "client_disconnect", 0)


# Requests that get no row.


async def test_requests_that_fail_auth_or_validation_are_not_logged():
    gw = Gateway({"fast": [("groq", MockProvider())]}, {"groq": CHEAP})

    no_key = await gw.post(chat(), key="gw-" + "x" * 43)
    bad_body = await gw.post({"model": "fast"})

    assert (no_key.status_code, bad_body.status_code) == (401, 400)
    assert await gw.rows() == []


# Logging never gets in the way of the response.


async def test_a_failed_log_write_does_not_touch_the_response(caplog):
    class BrokenLog:
        async def write(self, entry: RequestLog) -> None:
            raise psycopg.OperationalError("connection refused")

    gw = Gateway({"fast": [("groq", MockProvider())]}, {"groq": CHEAP}, store=BrokenLog())

    with caplog.at_level(logging.WARNING, logger="app.usage.log"):
        resp = await gw.post(chat())
        await gw.writer.drain()

    assert resp.status_code == 200
    assert "could not write the request log" in caplog.text


async def test_a_missing_price_loses_the_row_not_the_reply(caplog):
    gw = Gateway({"fast": [("groq", MockProvider())]}, prices={})

    with caplog.at_level(logging.WARNING, logger="app.usage.recorder"):
        resp = await gw.post(chat())

    assert resp.status_code == 200
    assert await gw.rows() == []
    assert "could not build the request log" in caplog.text


# The whole path, through Postgres.


def test_rows_reach_postgres_on_the_real_startup(pg_conninfo, monkeypatch):
    with psycopg.connect(pg_conninfo) as conn:
        apply_migrations(conn)
        key, record = create_key(conn, "alice", 60, 10_000)
    # The real startup: a real pool, the Postgres key store and request log, the shipped
    # registry and its prices, on the test schema.
    monkeypatch.setattr(app.main, "open_pool", open_pool)
    monkeypatch.setattr(app.main, "PostgresUsageLog", PostgresUsageLog)
    monkeypatch.setattr(
        app.main, "get_settings", lambda: Settings(_env_file=None, database_url=pg_conninfo)
    )
    headers = {"Authorization": f"Bearer {key}"}

    with TestClient(create_app()) as client:
        assert client.post("/v1/chat/completions", json=chat("mock"), headers=headers).is_success
        assert (
            client.post("/v1/chat/completions", json=chat("nope"), headers=headers).status_code
            == 404
        )
    # Leaving the block shut the app down, which waits for pending writes.

    with psycopg.connect(pg_conninfo) as conn:
        rows = conn.execute(
            "SELECT key_id, requested_model, provider, status_code, outcome, cost_usd"
            " FROM request_logs ORDER BY id"
        ).fetchall()

    assert rows == [
        (record.id, "mock", "mock", 200, "completed", Decimal(0)),
        (record.id, "nope", None, 404, "rejected", Decimal(0)),
    ]
