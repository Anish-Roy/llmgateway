"""The replay benchmark: what the caches and the router save on the whole query log.

Replays data/queries.jsonl through a running gateway in three modes, each with a key of its own:
    baseline        everything to `strong`, caches off (`x-gateway-cache: off`);
    cache           everything to `strong`, through the exact and the semantic cache;
    cache+router    `model: "auto"`, through both caches.
Every request is the same otherwise: the prompt as a single user message, temperature 0, one
max_tokens for all.

Per mode it reports:
    cost             what the mode's key spent, from request_logs (hits are billed $0)
    saved            1 - cost / baseline cost
    hit rate         exact + semantic hits / prompts answered
    false hits       semantic hits on a prompt whose meaning group (a prompt and its labelled
                     near-duplicates) hadn't been answered earlier in the mode: nothing with its
                     meaning was stored, so it got another question's answer. A lower bound: a
                     hit can also match the wrong prompt when the right one was stored.
    to cheap         prompts `auto` routed to its cheap alias / prompts answered (cache+router
                     only). Routing comes before the cache, so hits are counted where they
                     were routed.
    fallbacks        prompts a second provider answered: a fallback model has its own price
    p50 / p95        latency as the client saw it, over the prompts answered
"""

from dataclasses import dataclass
from decimal import Decimal

from scripts.eval_semantic_cache import Query, meaning_groups


@dataclass(frozen=True)
class Mode:
    name: str
    model: str
    # The x-gateway-cache request header.
    cache: str


BASELINE = Mode("baseline", "strong", "off")
MODES = [BASELINE, Mode("cache", "strong", "semantic"), Mode("cache+router", "auto", "semantic")]


@dataclass(frozen=True)
class Result:
    """One prompt's reply, as the client saw it."""

    id: str
    status: int
    latency_ms: float
    # The x-gateway-cache response header: exact, semantic, miss or bypass.
    cache: str
    # The x-gateway-route header: the alias `auto` picked. None on a named alias.
    route: str | None
    fallback: bool

    @property
    def answered(self) -> bool:
        return self.status == 200


@dataclass(frozen=True)
class Summary:
    mode: str
    prompts: int
    answered: int
    exact_hits: int
    semantic_hits: int
    false_hits: list[str]
    # None outside cache+router.
    to_cheap: int | None
    fallbacks: int
    p50_ms: float
    p95_ms: float
    cost_usd: Decimal

    @property
    def hit_rate(self) -> float:
        return (self.exact_hits + self.semantic_hits) / self.answered if self.answered else 0.0

    @property
    def cheap_share(self) -> float | None:
        if self.to_cheap is None:
            return None
        return self.to_cheap / self.answered if self.answered else 0.0


def percentile(values: list[float], q: float) -> float:
    """Linear interpolation between the closest ranks, as Postgres' percentile_cont does, so
    these match the dashboard's latencies."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    rank = q * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def summarize(
    mode: str,
    results: list[Result],
    queries: list[Query],
    cost_usd: Decimal,
    cheap_alias: str | None = None,
) -> Summary:
    """`results` in the order the prompts were sent: the false-hit count depends on it. With
    `cheap_alias`, counts the prompts routed to it."""
    group = meaning_groups(queries)
    # A failed request stores nothing, so only answered prompts count as asked.
    asked: set[str] = set()
    false_hits = []
    for r in results:
        if r.cache == "semantic" and group[r.id] not in asked:
            false_hits.append(r.id)
        if r.answered:
            asked.add(group[r.id])

    answered = [r for r in results if r.answered]
    latencies = [r.latency_ms for r in answered]
    return Summary(
        mode=mode,
        prompts=len(results),
        answered=len(answered),
        exact_hits=sum(r.cache == "exact" for r in answered),
        semantic_hits=sum(r.cache == "semantic" for r in answered),
        false_hits=false_hits,
        to_cheap=None if cheap_alias is None else sum(r.route == cheap_alias for r in answered),
        fallbacks=sum(r.fallback for r in answered),
        p50_ms=percentile(latencies, 0.5) if latencies else 0.0,
        p95_ms=percentile(latencies, 0.95) if latencies else 0.0,
        cost_usd=cost_usd,
    )


def saved_share(cost_usd: Decimal, baseline_usd: Decimal) -> float | None:
    """The share of the baseline's cost a mode didn't spend. None if the baseline cost nothing."""
    if not baseline_usd:
        return None
    return float(1 - cost_usd / baseline_usd)


def _percent(share: float | None) -> str:
    return "-" if share is None else f"{share:.1%}"


def markdown_table(summaries: list[Summary]) -> str:
    """The results table for the README. The first summary is the baseline the others are
    compared with."""
    baseline = summaries[0].cost_usd
    lines = [
        "| Mode | Prompts | Cost (USD) | Saved | Hit rate (exact / semantic) | False hits "
        "| To cheap model | Fallbacks | p50 | p95 |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for s in summaries:
        saved = "-" if s is summaries[0] else _percent(saved_share(s.cost_usd, baseline))
        lines.append(
            f"| {s.mode} | {s.answered} | {s.cost_usd:.4f} | {saved} "
            f"| {s.hit_rate:.1%} ({s.exact_hits} / {s.semantic_hits}) | {len(s.false_hits)} "
            f"| {_percent(s.cheap_share)} | {s.fallbacks} "
            f"| {s.p50_ms:,.0f} ms | {s.p95_ms:,.0f} ms |"
        )
    return "\n".join(lines)
