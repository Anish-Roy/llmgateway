"""The replay benchmark's scoring: percentiles, hit and false-hit counts, savings, the table."""

from decimal import Decimal

import pytest

from scripts.eval_semantic_cache import Query
from scripts.replay import Result, markdown_table, percentile, saved_share, summarize


def _q(qid: str, dup_of: str | None = None) -> Query:
    return Query(qid, f"prompt {qid}", dup_of, None)


def _r(
    qid: str,
    cache: str = "miss",
    *,
    status: int = 200,
    latency_ms: float = 100,
    route: str | None = None,
    fallback: bool = False,
) -> Result:
    return Result(qid, status, latency_ms, cache, route, fallback)


def test_percentile_interpolates_like_percentile_cont():
    # percentile_cont(0.5) over 1..4 is 2.5; 0.95 is 1 + 0.95 * 3 = 3.85.
    assert percentile([4, 1, 3, 2], 0.5) == 2.5
    assert percentile([4, 1, 3, 2], 0.95) == pytest.approx(3.85)
    assert percentile([7], 0.95) == 7


def test_percentile_of_nothing_is_an_error():
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_hits_count_over_answered_prompts():
    queries = [_q("a"), _q("b", dup_of="a"), _q("c"), _q("d")]
    results = [_r("a"), _r("b", "semantic"), _r("c", "exact"), _r("d", status=503)]
    s = summarize("cache", results, queries, Decimal("0.01"))
    assert (s.prompts, s.answered, s.exact_hits, s.semantic_hits) == (4, 3, 1, 1)
    assert s.hit_rate == pytest.approx(2 / 3)


def test_a_semantic_hit_after_its_original_is_right():
    queries = [_q("a"), _q("b", dup_of="a")]
    s = summarize("cache", [_r("a"), _r("b", "semantic")], queries, Decimal(0))
    assert s.false_hits == []


def test_a_semantic_hit_before_any_of_its_group_is_false():
    queries = [_q("a"), _q("b")]
    s = summarize("cache", [_r("a"), _r("b", "semantic")], queries, Decimal(0))
    assert s.false_hits == ["b"]


def test_a_near_duplicate_of_a_near_duplicate_shares_its_group():
    queries = [_q("a"), _q("b", dup_of="a"), _q("c", dup_of="b")]
    s = summarize("cache", [_r("b"), _r("c", "semantic")], queries, Decimal(0))
    assert s.false_hits == []


def test_an_original_that_failed_stored_nothing_to_hit():
    queries = [_q("a"), _q("b", dup_of="a")]
    results = [_r("a", status=503), _r("b", "semantic")]
    s = summarize("cache", results, queries, Decimal(0))
    assert s.false_hits == ["b"]


def test_cheap_share_counts_routes_to_the_cheap_alias():
    queries = [_q("a"), _q("b"), _q("c"), _q("d")]
    results = [
        _r("a", route="fast"),
        _r("b", "exact", route="fast"),
        _r("c", route="strong"),
        _r("d", status=429, route="fast"),
    ]
    s = summarize("cache+router", results, queries, Decimal(0), cheap_alias="fast")
    assert s.to_cheap == 2
    assert s.cheap_share == pytest.approx(2 / 3)


def test_cheap_share_is_none_without_the_router():
    s = summarize("cache", [_r("a")], [_q("a")], Decimal(0))
    assert s.to_cheap is None and s.cheap_share is None


def test_latency_and_fallbacks_leave_out_failures():
    queries = [_q(x) for x in "abcd"]
    results = [
        _r("a", latency_ms=100),
        _r("b", latency_ms=300, fallback=True),
        _r("c", latency_ms=200),
        _r("d", status=500, latency_ms=45_000),
    ]
    s = summarize("baseline", results, queries, Decimal(0))
    assert s.p50_ms == 200
    assert s.fallbacks == 1


def test_a_mode_with_no_answers_still_summarizes():
    s = summarize("baseline", [_r("a", status=503)], [_q("a")], Decimal(0))
    assert (s.answered, s.hit_rate, s.p50_ms, s.p95_ms) == (0, 0.0, 0.0, 0.0)


def test_saved_share_is_against_the_baseline():
    assert saved_share(Decimal("0.25"), Decimal("1.00")) == pytest.approx(0.75)
    assert saved_share(Decimal("1.00"), Decimal("1.00")) == 0
    assert saved_share(Decimal("0.10"), Decimal(0)) is None


def test_the_table_compares_each_mode_with_the_first():
    queries = [_q("a"), _q("b", dup_of="a")]
    baseline = summarize("baseline", [_r("a"), _r("b")], queries, Decimal("0.0200"))
    cache = summarize("cache", [_r("a"), _r("b", "semantic")], queries, Decimal("0.0100"))
    lines = markdown_table([baseline, cache]).splitlines()
    assert len(lines) == 4
    assert lines[2].startswith("| baseline | 2 | 0.0200 | - | 0.0% (0 / 0) | 0 | - | 0 |")
    assert lines[3].startswith("| cache | 2 | 0.0100 | 50.0% | 50.0% (0 / 1) | 0 | - | 0 |")
