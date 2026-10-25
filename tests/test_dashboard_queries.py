"""The dashboard's queries, on hand-made request_logs rows in a fresh schema."""

import hashlib
import subprocess
import sys
from datetime import UTC, date, datetime
from decimal import Decimal

import psycopg
import pytest

from app.db.migrate import apply_migrations
from dashboard import queries
from dashboard.queries import Filters

DAY = date(2026, 10, 20)
NOON = datetime(2026, 10, 20, 12, tzinfo=UTC)
ALL_OF_DAY = Filters(start=DAY, end=DAY)

# A paid miss: 3 x $1.5/1M + 5 x $4/1M on groq/m.
PAID = Decimal("0.0000245")


@pytest.fixture
def conn(pg_conninfo):
    with psycopg.connect(pg_conninfo) as conn:
        apply_migrations(conn)
        yield conn


def _key(conn: psycopg.Connection, name: str, prefix: str, active: bool = True) -> int:
    return conn.execute(
        "INSERT INTO api_keys (name, key_hash, key_prefix, rpm_limit, tpm_limit, active)"
        " VALUES (%s, %s, %s, 60, 10000, %s) RETURNING id",
        (name, hashlib.sha256(prefix.encode()).hexdigest(), prefix, active),
    ).fetchone()[0]


def _log(conn: psycopg.Connection, key_id: int, **overrides) -> None:
    # A paid, completed, non-streamed miss at noon on DAY, unless told otherwise.
    row = {
        "key_id": key_id,
        "created_at": NOON,
        "requested_model": "fast",
        "provider": "groq",
        "model": "m",
        "stream": False,
        "status_code": 200,
        "outcome": "completed",
        "latency_ms": 1000,
        "cost_usd": PAID,
        "attempts": 1,
    } | overrides
    columns = ", ".join(row)
    values = ", ".join(f"%({c})s" for c in row)
    conn.execute(f"INSERT INTO request_logs ({columns}) VALUES ({values})", row)


def _hit(conn: psycopg.Connection, key_id: int, status: str = "exact", **overrides) -> None:
    hit = {"cache_status": status, "cost_usd": 0, "attempts": 0, "saved_usd": PAID}
    _log(conn, key_id, **(hit | {"latency_ms": 5} | overrides))


def _rejected(conn: psycopg.Connection, key_id: int, **overrides) -> None:
    rejected = {"provider": None, "model": None, "status_code": 429, "outcome": "rejected"}
    _log(conn, key_id, **(rejected | {"cost_usd": 0, "attempts": 0, "latency_ms": 0} | overrides))


# Keys.


def test_keys_are_labelled_with_their_prefix_revoked_ones_too(conn):
    a = _key(conn, "backend", "gw-aaaa")
    b = _key(conn, "backend", "gw-bbbb", active=False)
    _log(conn, b)

    assert queries.keys(conn) == [
        queries.Key(a, "backend (gw-aaaa)", active=True, logged=False),
        queries.Key(b, "backend (gw-bbbb)", active=False, logged=True),
    ]


# The slice every query reads.


def test_the_range_is_whole_utc_days_with_the_end_day_included(conn):
    key = _key(conn, "a", "gw-aaaa")
    _log(conn, key, created_at=datetime(2026, 10, 19, 23, 59, 59, tzinfo=UTC))  # before
    _log(conn, key, created_at=datetime(2026, 10, 20, 0, 0, tzinfo=UTC))
    _log(conn, key, created_at=datetime(2026, 10, 21, 23, 59, 59, tzinfo=UTC))
    _log(conn, key, created_at=datetime(2026, 10, 22, 0, 0, tzinfo=UTC))  # after

    f = Filters(start=date(2026, 10, 20), end=date(2026, 10, 21))

    assert queries.summary(conn, f).requests == 2


def test_the_key_filter(conn):
    a = _key(conn, "a", "gw-aaaa")
    b = _key(conn, "b", "gw-bbbb")
    _log(conn, a)
    _log(conn, b)
    _log(conn, b)

    def count(key_ids):
        return queries.summary(conn, Filters(DAY, DAY, key_ids)).requests

    assert count(None) == 3
    assert count((b,)) == 2
    assert count((a, b)) == 3
    assert count(()) == 0


# The headline numbers.


def test_the_summary_counts_each_way_a_request_ended(conn):
    key = _key(conn, "a", "gw-aaaa")
    _log(conn, key)  # a paid miss
    _log(conn, key, fallback=True, attempts=2)  # a miss a second provider answered
    _hit(conn, key, "exact")
    _hit(conn, key, "semantic")
    _hit(conn, key, "exact", saved_usd=None)  # logged before saved_usd existed
    # A stream cut partway: a 200, but a provider error.
    _log(conn, key, stream=True, outcome="provider_error", usage_estimated=True)
    _log(conn, key, provider=None, model=None, status_code=500, outcome="gateway_error",
         cost_usd=0, attempts=0)  # fmt: skip
    _log(conn, key, status_code=499, outcome="client_disconnect", cost_usd=0)
    _rejected(conn, key)
    _rejected(conn, key, status_code=404, requested_model="nope")

    s = queries.summary(conn, ALL_OF_DAY)

    assert s.requests == 10
    assert s.spend_usd == 3 * PAID
    assert (s.saved_usd, s.unpriced_hits) == (2 * PAID, 1)
    assert (s.exact_hits, s.semantic_hits) == (2, 1)
    assert (s.provider_errors, s.gateway_errors, s.client_disconnects, s.rejected) == (1, 1, 1, 2)
    # The rejected never reached the cache.
    assert s.lookups == 8
    assert s.hit_rate == 3 / 8
    # Only requests that called a provider can have fallen back.
    assert (s.fallbacks, s.provider_calls) == (1, 4)
    assert s.fallback_rate == 1 / 4
    # The cut stream and the gateway's 500, out of the 8 the gateway let in.
    assert s.error_rate == 2 / 8


def test_an_empty_range_has_no_rates(conn):
    _key(conn, "a", "gw-aaaa")

    s = queries.summary(conn, ALL_OF_DAY)

    assert (s.requests, s.spend_usd, s.saved_usd) == (0, 0, 0)
    assert (s.hit_rate, s.fallback_rate, s.error_rate) == (None, None, None)


def test_sums_of_tiny_costs_are_exact(conn):
    key = _key(conn, "a", "gw-aaaa")
    for _ in range(1000):
        _log(conn, key, cost_usd=Decimal("0.00000002"))

    assert queries.summary(conn, ALL_OF_DAY).spend_usd == Decimal("0.00002")


# Who spent what.


def test_daily_spend_is_per_key_per_utc_day(conn):
    a = _key(conn, "a", "gw-aaaa")
    b = _key(conn, "b", "gw-bbbb")
    _log(conn, a)
    _hit(conn, a)
    _log(conn, b)
    # 23:30 in UTC is already the next day in India: buckets are UTC days.
    _log(conn, a, created_at=datetime(2026, 10, 21, 23, 30, tzinfo=UTC))

    rows = queries.daily_spend(conn, Filters(DAY, date(2026, 10, 21)))

    assert rows == [
        queries.DailyKeySpend(DAY, a, "a (gw-aaaa)", 2, PAID, PAID),
        queries.DailyKeySpend(DAY, b, "b (gw-bbbb)", 1, PAID, Decimal(0)),
        queries.DailyKeySpend(date(2026, 10, 21), a, "a (gw-aaaa)", 1, PAID, Decimal(0)),
    ]


def test_key_totals_put_the_biggest_spender_first(conn):
    small = _key(conn, "small", "gw-aaaa")
    big = _key(conn, "big", "gw-bbbb")
    _log(conn, small)
    _hit(conn, small)
    _log(conn, big)
    _log(conn, big)

    assert queries.key_totals(conn, ALL_OF_DAY) == [
        queries.KeyTotals(big, "big (gw-bbbb)", 2, 2 * PAID, Decimal(0), 0),
        queries.KeyTotals(small, "small (gw-aaaa)", 2, PAID, PAID, 1),
    ]


# Providers, latency and routes.


def test_the_model_mix_counts_only_requests_that_called_a_provider(conn):
    key = _key(conn, "a", "gw-aaaa")
    _log(conn, key)
    _log(conn, key, provider="gemini", model="flash", fallback=True, attempts=2)
    _hit(conn, key)  # names groq, but called nothing
    _rejected(conn, key)

    assert queries.model_mix(conn, ALL_OF_DAY) == [
        queries.ModelMix("gemini", "flash", 1, PAID, 1),
        queries.ModelMix("groq", "m", 1, PAID, 0),
    ]


def test_latency_keeps_hits_misses_and_streams_apart(conn):
    key = _key(conn, "a", "gw-aaaa")
    for ms in (100, 200, 300, 400, 500):
        _log(conn, key, latency_ms=ms)
    _hit(conn, key, latency_ms=4)
    _hit(conn, key, latency_ms=6)
    _log(conn, key, stream=True, latency_ms=2000, ttft_ms=300)
    # Neither a refusal nor a failure says how fast the gateway answers.
    _rejected(conn, key, latency_ms=1)
    _log(conn, key, outcome="provider_error", latency_ms=45_000)

    rows = {(r.cache_status, r.stream): r for r in queries.latency(conn, ALL_OF_DAY)}

    assert set(rows) == {("exact", False), ("miss", False), ("miss", True)}
    miss = rows["miss", False]
    assert (miss.requests, miss.p50_ms, miss.p95_ms) == (5, 300, 480)
    assert (miss.ttft_p50_ms, miss.ttft_p95_ms) == (None, None)
    hit = rows["exact", False]
    assert (hit.requests, hit.p50_ms) == (2, 5)
    stream = rows["miss", True]
    assert (stream.p50_ms, stream.ttft_p50_ms) == (2000, 300)


def test_routes_count_auto_requests_by_alias_and_rule(conn):
    key = _key(conn, "a", "gw-aaaa")
    auto = {"requested_model": "auto"}
    _log(conn, key, **auto, routed_alias="fast", route_reason="default")
    _log(conn, key, **auto, routed_alias="fast", route_reason="default")
    _log(conn, key, **auto, routed_alias="strong", route_reason="keyword:prove")
    _log(conn, key)  # a named alias

    assert queries.routes(conn, ALL_OF_DAY) == [
        queries.Route("fast", "default", 2, 2 * PAID),
        queries.Route("strong", "keyword:prove", 1, PAID),
    ]


# The dashboard deploys without the gateway's dependencies.


def test_the_queries_dont_load_the_gateway():
    code = (
        "import sys, dashboard.queries; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] in ('app', 'torch')))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    assert out.stdout.strip() == "[]"
