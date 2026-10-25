"""Cost calculation, the request-log stores, and the background writer."""

import asyncio
import dataclasses
import logging
from datetime import UTC, datetime
from decimal import Decimal

import psycopg
import pytest

from app.api.schemas import Usage
from app.db.migrate import apply_migrations
from app.db.pool import open_pool
from app.routing.registry import FREE, Price
from app.usage.cost import cost
from app.usage.log import InMemoryUsageLog, LogWriter, PostgresUsageLog, RequestLog

GPT_OSS_20B = Price(input=Decimal("0.075"), output=Decimal("0.30"))


def _usage(prompt: int, completion: int) -> Usage:
    return Usage(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
    )


def _entry(**overrides) -> RequestLog:
    fields = {
        "key_id": 1,
        "created_at": datetime(2026, 10, 25, 12, 0, tzinfo=UTC),
        "requested_model": "fast",
        "provider": "groq",
        "model": "openai/gpt-oss-20b",
        "stream": False,
        "status_code": 200,
        "outcome": "completed",
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "usage_estimated": False,
        "latency_ms": 840,
        "ttft_ms": None,
        "cost_usd": Decimal("0.000225"),
        "attempts": 1,
        "fallback": False,
    } | overrides
    return RequestLog(**fields)


def test_cost_charges_input_and_output_at_their_own_price():
    # 1000 x $0.075/1M + 500 x $0.30/1M = $0.000075 + $0.00015
    assert cost(GPT_OSS_20B, _usage(1000, 500)) == Decimal("0.000225")


def test_cost_of_one_token_is_kept():
    # Far below a cent, but a thousand of them add up: nothing is rounded away.
    assert cost(GPT_OSS_20B, _usage(1, 0)) == Decimal("0.000000075")


def test_cost_rounds_to_the_columns_ten_places():
    price = Price(input=Decimal("0.00015"), output=Decimal(0))

    # 1.5e-10 rounds half up to 2e-10, which is what numeric(14, 10) would store.
    assert cost(price, _usage(1, 0)) == Decimal("2E-10")


def test_free_models_cost_nothing():
    assert cost(FREE, _usage(1000, 500)) == 0


async def test_submit_returns_before_the_write_finishes():
    release = asyncio.Event()

    class SlowLog(InMemoryUsageLog):
        async def write(self, entry: RequestLog) -> None:
            await release.wait()
            await super().write(entry)

    store = SlowLog()
    writer = LogWriter(store)

    writer.submit(_entry())
    await asyncio.sleep(0)
    assert store.entries == []

    release.set()
    await writer.drain()
    assert store.entries == [_entry()]


async def test_drain_waits_for_every_pending_write():
    store = InMemoryUsageLog()
    writer = LogWriter(store)

    for i in range(5):
        writer.submit(_entry(key_id=i))
    await writer.drain()

    assert [e.key_id for e in store.entries] == [0, 1, 2, 3, 4]


async def test_a_failed_write_is_logged_not_raised(caplog):
    class BrokenLog:
        async def write(self, entry: RequestLog) -> None:
            raise psycopg.OperationalError("connection refused")

    writer = LogWriter(BrokenLog())

    with caplog.at_level(logging.WARNING, logger="app.usage.log"):
        writer.submit(_entry())
        await writer.drain()

    assert "could not write the request log for key 1" in caplog.text


async def test_drain_with_nothing_pending_returns():
    await LogWriter(InMemoryUsageLog()).drain()


@pytest.fixture
def key_id(pg_conninfo) -> int:
    """Migrates the test schema and stores one key, which logs must point at."""
    with psycopg.connect(pg_conninfo) as conn:
        apply_migrations(conn)
        (key_id,) = conn.execute(
            "INSERT INTO api_keys (name, key_hash, key_prefix, rpm_limit, tpm_limit)"
            " VALUES ('test', %s, 'gw-abcd', 60, 10000) RETURNING id",
            ("a" * 64,),
        ).fetchone()
    return key_id


async def _write_and_read_back(conninfo: str, entry: RequestLog) -> tuple:
    async with open_pool(conninfo, max_size=1) as pool:
        await PostgresUsageLog(pool).write(entry)
        async with pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT {', '.join(f.name for f in dataclasses.fields(RequestLog))}"
                " FROM request_logs"
            )
            return await cur.fetchone()


async def test_postgres_log_writes_every_column(pg_conninfo, key_id):
    entry = _entry(
        key_id=key_id,
        stream=True,
        ttft_ms=210,
        usage_estimated=True,
        outcome="client_disconnect",
        attempts=3,
        fallback=True,
        pii_types=["email", "phone"],
    )

    row = await _write_and_read_back(pg_conninfo, entry)

    assert row == dataclasses.astuple(entry)


async def test_postgres_log_writes_a_request_no_provider_answered(pg_conninfo, key_id):
    entry = _entry(
        key_id=key_id,
        provider=None,
        model=None,
        status_code=429,
        outcome="rejected",
        prompt_tokens=0,
        completion_tokens=0,
        cost_usd=Decimal(0),
        attempts=0,
    )

    row = await _write_and_read_back(pg_conninfo, entry)

    assert row == dataclasses.astuple(entry)


async def test_postgres_log_refuses_an_unknown_pii_type(pg_conninfo, key_id):
    # The column holds kinds, never values: anything else is a bug, and the row is refused.
    with pytest.raises(psycopg.errors.CheckViolation):
        await _write_and_read_back(pg_conninfo, _entry(key_id=key_id, pii_types=["a@b.com"]))
