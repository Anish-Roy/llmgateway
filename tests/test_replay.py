"""The replay benchmark: its scoring (percentiles, hits, false hits, savings, the table) and
its runner (the requests it sends, resuming, where it stops), against a fake gateway."""

import json
from decimal import Decimal

import httpx
import psycopg
import pytest

from app.auth.admin import create_key
from app.db.migrate import apply_migrations
from scripts.eval_semantic_cache import Query
from scripts.replay import (
    BASELINE,
    MODES,
    ModeKey,
    Replayer,
    Result,
    Run,
    Stopped,
    load_results,
    markdown_table,
    mode_cost,
    percentile,
    saved_share,
    summarize,
)

CACHE_ROUTER = MODES[2]


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


# The runner.


class FakeGateway:
    """Answers every request; the reply's headers come from `headers`, by prompt."""

    def __init__(self, headers: dict[str, dict[str, str]] | None = None) -> None:
        self.headers = headers or {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        prompt = json.loads(request.content)["messages"][-1]["content"]
        reply = {"x-gateway-cache": "miss", "x-gateway-attempts": "1"} | self.headers.get(
            prompt, {}
        )
        status = int(reply.pop("status", 200))
        return httpx.Response(status, headers=reply, json={})


def _replayer(gateway: FakeGateway, pauses: list[float]) -> Replayer:
    http = httpx.Client(base_url="http://gw/v1", transport=httpx.MockTransport(gateway))
    return Replayer(http, max_tokens=64, pause=2.0, sleep=pauses.append)


def test_a_request_carries_the_mode_and_a_system_prompt_naming_the_run():
    gateway = FakeGateway()
    _replayer(gateway, []).send("r1", CACHE_ROUTER, "gw-k", _q("a"))

    request = gateway.requests[0]
    body = json.loads(request.content)
    assert request.url.path == "/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer gw-k"
    assert request.headers["x-gateway-cache"] == "semantic"
    assert body["model"] == "auto"
    assert body["temperature"] == 0 and body["max_tokens"] == 64
    assert body["messages"] == [
        {"role": "system", "content": "Benchmark run r1, mode cache+router."},
        {"role": "user", "content": "prompt a"},
    ]


def test_the_system_prompt_differs_by_mode_so_the_exact_cache_does_too():
    gateway = FakeGateway()
    replayer = _replayer(gateway, [])
    for mode in MODES:
        replayer.send("r1", mode, "gw-k", _q("a"))
    systems = {json.loads(r.content)["messages"][0]["content"] for r in gateway.requests}
    assert len(systems) == 3


def test_the_reply_headers_become_the_result():
    headers = {"x-gateway-cache": "exact", "x-gateway-route": "fast", "x-gateway-fallback": "true"}
    gateway = FakeGateway({"prompt a": headers})
    result = _replayer(gateway, []).send("r1", CACHE_ROUTER, "gw-k", _q("a"))
    assert (result.id, result.status, result.cache) == ("a", 200, "exact")
    assert result.route == "fast" and result.fallback


def test_it_pauses_only_after_a_provider_call():
    gateway = FakeGateway({"prompt b": {"x-gateway-cache": "exact", "x-gateway-attempts": "0"}})
    pauses: list[float] = []
    replayer = _replayer(gateway, pauses)
    replayer.send("r1", BASELINE, "gw-k", _q("a"))
    replayer.send("r1", BASELINE, "gw-k", _q("b"))
    assert pauses == [2.0]


def test_a_rerun_sends_only_the_prompts_not_yet_saved(tmp_path):
    path = tmp_path / "baseline.jsonl"
    queries = [_q("a"), _q("b"), _q("c")]
    gateway = FakeGateway()
    replayer = _replayer(gateway, [])
    assert replayer.replay("r1", BASELINE, "gw-k", queries[:2], path) == 2
    assert replayer.replay("r1", BASELINE, "gw-k", queries, path) == 1
    assert [r.id for r in load_results(path)] == ["a", "b", "c"]


def test_a_failed_request_stops_the_run_unsaved(tmp_path):
    path = tmp_path / "baseline.jsonl"
    gateway = FakeGateway({"prompt b": {"status": "503"}})
    with pytest.raises(Stopped, match="b: HTTP 503"):
        _replayer(gateway, []).replay("r1", BASELINE, "gw-k", [_q("a"), _q("b"), _q("c")], path)
    assert [r.id for r in load_results(path)] == ["a"]


def test_a_gateway_that_cant_be_reached_stops_the_run(tmp_path):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    http = httpx.Client(base_url="http://gw/v1", transport=httpx.MockTransport(refuse))
    with pytest.raises(Stopped, match="ConnectError"):
        Replayer(http, 64, 0).replay("r1", BASELINE, "gw-k", [_q("a")], tmp_path / "b.jsonl")


def test_a_fallback_answer_is_saved_then_stops_the_run(tmp_path):
    path = tmp_path / "baseline.jsonl"
    gateway = FakeGateway({"prompt a": {"x-gateway-fallback": "true"}})
    with pytest.raises(Stopped, match="fallback"):
        _replayer(gateway, []).replay("r1", BASELINE, "gw-k", [_q("a"), _q("b")], path)
    assert [(r.id, r.fallback) for r in load_results(path)] == [("a", True)]


def test_the_run_state_survives_a_restart(tmp_path):
    path = tmp_path / "replay" / "state.json"
    assert Run.load(path) is None
    run = Run.new()
    run.keys["baseline"] = ModeKey(7, "gw-secret")
    run.save(path)
    assert Run.load(path) == run


def test_a_modes_cost_is_everything_its_key_spent(pg_conninfo):
    with psycopg.connect(pg_conninfo) as conn:
        apply_migrations(conn)
        _, mine = create_key(conn, "replay-baseline", 60, 1000)
        _, other = create_key(conn, "someone-else", 60, 1000)
        for key_id, cost in [(mine.id, "0.002"), (mine.id, "0.0005"), (other.id, "1")]:
            conn.execute(
                "INSERT INTO request_logs (key_id, requested_model, provider, model, stream,"
                " status_code, outcome, latency_ms, cost_usd)"
                " VALUES (%s, 'strong', 'groq', 'm', false, 200, 'completed', 10, %s)",
                (key_id, Decimal(cost)),
            )
        assert mode_cost(conn, mine.id) == Decimal("0.0025")
        _, idle = create_key(conn, "replay-cache", 60, 1000)
        assert mode_cost(conn, idle.id) == 0
