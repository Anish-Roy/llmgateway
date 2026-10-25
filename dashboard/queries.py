"""What the cost dashboard shows, as SQL over `request_logs`: a connection and filters in,
typed rows out.

Nothing here imports `app`: the gateway's packages pull in torch and sentence-transformers,
which the dashboard has no use for and its host (Streamlit Community Cloud) can't afford. The
table's own comments (app/db/schema/) are the contract.

Every function reads the same slice of the table: one or more keys over a range of UTC days,
the end day included.
"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import psycopg
from psycopg.rows import class_row

# A request the gateway turned away before any cache or provider was asked: a 429, an unknown
# model (404), an alias with no provider here (503). It never had a chance to hit the cache or
# to fail upstream, so it's left out of the hit rate, the latencies and the provider errors.
_REJECTED = "outcome = 'rejected'"
_HIT = "cache_status <> 'miss'"

_WHERE = """
    created_at >= (%(start)s::date)::timestamp AT TIME ZONE 'UTC'
    AND created_at < (%(end)s::date + 1)::timestamp AT TIME ZONE 'UTC'
    AND (%(key_ids)s::bigint[] IS NULL OR key_id = ANY(%(key_ids)s::bigint[]))
"""
# The same slice as a table, for queries that join api_keys (which has a created_at too).
_SLICE = f"(SELECT * FROM request_logs WHERE {_WHERE})"


@dataclass(frozen=True)
class Filters:
    start: date
    # Included: a range of one day is start == end.
    end: date
    # None for every key.
    key_ids: tuple[int, ...] | None = None

    def params(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "key_ids": None if self.key_ids is None else list(self.key_ids),
        }


@dataclass(frozen=True)
class Key:
    id: int
    # Names aren't unique, so the prefix tells two "backend" keys apart.
    label: str
    active: bool
    # Whether it has sent any request, ever: the dashboard gives each of those a fixed color.
    logged: bool


@dataclass(frozen=True)
class Summary:
    requests: int
    spend_usd: Decimal
    # What the hits whose saving is known saved. Hits logged before saved_usd existed are
    # counted in `unpriced_hits` instead.
    saved_usd: Decimal
    unpriced_hits: int
    exact_hits: int
    semantic_hits: int
    # Requests the cache could have answered: all but the rejected. Requests it can't serve
    # (a high temperature, `x-gateway-cache: off`) are logged as misses too, so the hit rate
    # is over everything that got that far, and reads lower than the rate over cacheable ones.
    lookups: int
    # Requests that called at least one provider, and those that needed a second one.
    provider_calls: int
    fallbacks: int
    # How requests ended, other than `completed`. A stream that failed partway is a 200 with
    # a provider_error, so these, not status codes, are the error counts.
    provider_errors: int
    gateway_errors: int
    client_disconnects: int
    rejected: int

    @property
    def hit_rate(self) -> float | None:
        return _ratio(self.exact_hits + self.semantic_hits, self.lookups)

    @property
    def fallback_rate(self) -> float | None:
        return _ratio(self.fallbacks, self.provider_calls)

    @property
    def error_rate(self) -> float | None:
        """Requests the gateway let in that didn't complete: a provider or the gateway
        failed. Rejections and clients that left aren't failures of the gateway."""
        return _ratio(self.provider_errors + self.gateway_errors, self.requests - self.rejected)


@dataclass(frozen=True)
class DailyKeySpend:
    day: date
    key_id: int
    label: str
    requests: int
    spend_usd: Decimal
    saved_usd: Decimal


@dataclass(frozen=True)
class KeyTotals:
    key_id: int
    label: str
    requests: int
    spend_usd: Decimal
    saved_usd: Decimal
    hits: int


@dataclass(frozen=True)
class ModelMix:
    provider: str
    model: str
    requests: int
    spend_usd: Decimal
    fallbacks: int


@dataclass(frozen=True)
class Latency:
    # "miss", "exact" or "semantic".
    cache_status: str
    stream: bool
    requests: int
    p50_ms: float
    p95_ms: float
    # Streams only: time to the first token. None for plain replies.
    ttft_p50_ms: float | None
    ttft_p95_ms: float | None


@dataclass(frozen=True)
class Route:
    # What `model: "auto"` picked, and the rule that picked it.
    alias: str
    reason: str
    requests: int
    spend_usd: Decimal


def keys(conn: psycopg.Connection) -> list[Key]:
    """Every key, revoked ones too: their past spend still counts."""
    with conn.cursor(row_factory=class_row(Key)) as cur:
        return cur.execute(
            "SELECT id, name || ' (' || key_prefix || ')' AS label, active,"
            " EXISTS (SELECT 1 FROM request_logs r WHERE r.key_id = k.id) AS logged"
            " FROM api_keys k ORDER BY name, id"
        ).fetchall()


def summary(conn: psycopg.Connection, f: Filters) -> Summary:
    sql = f"""
        SELECT
            count(*) AS requests,
            coalesce(sum(cost_usd), 0) AS spend_usd,
            coalesce(sum(saved_usd), 0) AS saved_usd,
            count(*) FILTER (WHERE {_HIT} AND saved_usd IS NULL) AS unpriced_hits,
            count(*) FILTER (WHERE cache_status = 'exact') AS exact_hits,
            count(*) FILTER (WHERE cache_status = 'semantic') AS semantic_hits,
            count(*) FILTER (WHERE NOT {_REJECTED}) AS lookups,
            count(*) FILTER (WHERE attempts > 0) AS provider_calls,
            count(*) FILTER (WHERE fallback) AS fallbacks,
            count(*) FILTER (WHERE outcome = 'provider_error') AS provider_errors,
            count(*) FILTER (WHERE outcome = 'gateway_error') AS gateway_errors,
            count(*) FILTER (WHERE outcome = 'client_disconnect') AS client_disconnects,
            count(*) FILTER (WHERE {_REJECTED}) AS rejected
        FROM request_logs
        WHERE {_WHERE}
    """
    with conn.cursor(row_factory=class_row(Summary)) as cur:
        return cur.execute(sql, f.params()).fetchone()


def daily_spend(conn: psycopg.Connection, f: Filters) -> list[DailyKeySpend]:
    """Spend and requests per key per UTC day. Days a key sent nothing have no row."""
    sql = f"""
        SELECT
            (r.created_at AT TIME ZONE 'UTC')::date AS day,
            r.key_id,
            k.name || ' (' || k.key_prefix || ')' AS label,
            count(*) AS requests,
            sum(r.cost_usd) AS spend_usd,
            coalesce(sum(r.saved_usd), 0) AS saved_usd
        FROM {_SLICE} r JOIN api_keys k ON k.id = r.key_id
        GROUP BY 1, 2, 3
        ORDER BY 1, 3
    """
    with conn.cursor(row_factory=class_row(DailyKeySpend)) as cur:
        return cur.execute(sql, f.params()).fetchall()


def key_totals(conn: psycopg.Connection, f: Filters) -> list[KeyTotals]:
    """Who spent what, most first."""
    sql = f"""
        SELECT
            r.key_id,
            k.name || ' (' || k.key_prefix || ')' AS label,
            count(*) AS requests,
            sum(r.cost_usd) AS spend_usd,
            coalesce(sum(r.saved_usd), 0) AS saved_usd,
            count(*) FILTER (WHERE {_HIT}) AS hits
        FROM {_SLICE} r JOIN api_keys k ON k.id = r.key_id
        GROUP BY 1, 2
        ORDER BY spend_usd DESC, label
    """
    with conn.cursor(row_factory=class_row(KeyTotals)) as cur:
        return cur.execute(sql, f.params()).fetchall()


def model_mix(conn: psycopg.Connection, f: Filters) -> list[ModelMix]:
    """Which provider and model answered the requests that called one. Cache hits name the
    provider that first answered, but called none, so they're left out."""
    sql = f"""
        SELECT
            provider,
            model,
            count(*) AS requests,
            sum(cost_usd) AS spend_usd,
            count(*) FILTER (WHERE fallback) AS fallbacks
        FROM request_logs
        WHERE {_WHERE} AND cache_status = 'miss' AND provider IS NOT NULL
        GROUP BY 1, 2
        ORDER BY requests DESC, provider, model
    """
    with conn.cursor(row_factory=class_row(ModelMix)) as cur:
        return cur.execute(sql, f.params()).fetchall()


def latency(conn: psycopg.Connection, f: Filters) -> list[Latency]:
    """p50 and p95 of completed requests, hits apart from misses and streams apart from plain
    replies: a hit takes milliseconds, a miss seconds, and mixing them describes neither.
    A stream's latency runs until it closed, so its time to first token is shown too."""
    sql = f"""
        SELECT
            cache_status,
            stream,
            count(*) AS requests,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) AS p50_ms,
            percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY ttft_ms) AS ttft_p50_ms,
            percentile_cont(0.95) WITHIN GROUP (ORDER BY ttft_ms) AS ttft_p95_ms
        FROM request_logs
        WHERE {_WHERE} AND outcome = 'completed'
        GROUP BY 1, 2
        ORDER BY 1, 2
    """
    with conn.cursor(row_factory=class_row(Latency)) as cur:
        return cur.execute(sql, f.params()).fetchall()


def routes(conn: psycopg.Connection, f: Filters) -> list[Route]:
    """How `model: "auto"` requests were split, and by which rule."""
    sql = f"""
        SELECT
            routed_alias AS alias,
            route_reason AS reason,
            count(*) AS requests,
            sum(cost_usd) AS spend_usd
        FROM request_logs
        WHERE {_WHERE} AND routed_alias IS NOT NULL
        GROUP BY 1, 2
        ORDER BY requests DESC, alias, reason
    """
    with conn.cursor(row_factory=class_row(Route)) as cur:
        return cur.execute(sql, f.params()).fetchall()


def _ratio(part: int, whole: int) -> float | None:
    return part / whole if whole else None
