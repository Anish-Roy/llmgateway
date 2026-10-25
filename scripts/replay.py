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

`overhead` measures what the gateway itself adds: the same prompts to the `mock` alias, whose
provider answers at once (no latency, no network), so the time the client sees is the
gateway's own: HTTP, auth, rate limit, PII masking, the cache lookups and the store. Three
ways: caches off, the exact cache (the default), and the semantic cache too, which embeds the
prompt with MiniLM. Only misses are counted: a hit skips the provider call, which here costs
nothing, so it would only make the cache look cheaper than it is. Runs on the machine the
gateway runs on; the numbers are that machine's.

A run calls the real providers (about 900 calls for the whole log) and logs them at list price;
not for CI. Each mode sends the log in file order, one request at a time, and pauses after
every provider call: for the provider's rate limit, and so a cache store has landed before the
next lookup.

uv run uvicorn app.main:app            # with GROQ_API_KEY / GEMINI_API_KEY in .env
uv run python -m scripts.replay run    # all three modes; rerun to resume
uv run python -m scripts.replay overhead    # free: the mock alias calls no provider
uv run python -m scripts.replay report

Runs resume. Each answered prompt is appended to data/replay/<run>/<mode>.jsonl as it comes
back, and a rerun skips the prompts already there, so a run stopped by a provider's daily
limit loses nothing. A run stops at the first request that fails (not saved: the rerun tries
it again) or that a fallback provider answered (saved, then stopped: a fallback model has its
own price, and a run of them usually means the first provider hit a limit).

The run's state is data/replay/state.json, kept out of git: the run id and, per mode, the key
created for it (high limits, so the gateway's own rate limit never shows). `run --new` starts
another run, with new keys. Keys give each mode its own spend in request_logs and its own
semantic cache. The exact cache is shared by all keys, so every request carries a system
prompt naming the run and the mode: an answer stored in one mode or run is never an exact hit
in another. The router reads only user messages, so the system prompt doesn't change routing.
"""

import argparse
import json
import secrets
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path

import httpx
import psycopg

from app.auth.admin import create_key
from app.config import get_settings
from app.routing.registry import load_registry
from scripts.eval_semantic_cache import DATASET, Query, load_queries, meaning_groups

RUNS = Path(__file__).resolve().parent.parent / "data" / "replay"
KEY_RPM = 10_000
KEY_TPM = 100_000_000


@dataclass(frozen=True)
class Mode:
    name: str
    model: str
    # The x-gateway-cache request header; None sends none (the exact cache only).
    cache: str | None


BASELINE = Mode("baseline", "strong", "off")
MODES = [BASELINE, Mode("cache", "strong", "semantic"), Mode("cache+router", "auto", "semantic")]
OVERHEAD_MODES = [
    Mode("overhead-off", "mock", "off"),
    Mode("overhead-exact", "mock", None),
    Mode("overhead-semantic", "mock", "semantic"),
]


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


@dataclass(frozen=True)
class Overhead:
    mode: str
    misses: int
    p50_ms: float
    p95_ms: float


def overhead(mode: str, results: list[Result]) -> Overhead:
    """The gateway's own latency, over the answered misses."""
    misses = [r.latency_ms for r in results if r.answered and r.cache not in ("exact", "semantic")]
    return Overhead(
        mode=mode,
        misses=len(misses),
        p50_ms=percentile(misses, 0.5) if misses else 0.0,
        p95_ms=percentile(misses, 0.95) if misses else 0.0,
    )


def overhead_table(rows: list[Overhead]) -> str:
    lines = ["| Gateway overhead | Requests | p50 | p95 |", "|---|--:|--:|--:|"]
    for o in rows:
        lines.append(f"| {o.mode} | {o.misses} | {o.p50_ms:,.1f} ms | {o.p95_ms:,.1f} ms |")
    return "\n".join(lines)


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


def system_prompt(run_id: str, mode: Mode) -> str:
    return f"Benchmark run {run_id}, mode {mode.name}."


@dataclass(frozen=True)
class ModeKey:
    id: int
    key: str


@dataclass
class Run:
    id: str
    keys: dict[str, ModeKey]

    @classmethod
    def new(cls) -> "Run":
        return cls(secrets.token_hex(4), {})

    @classmethod
    def load(cls, path: Path) -> "Run | None":
        if not path.exists():
            return None
        state = json.loads(path.read_text())
        keys = {mode: ModeKey(**k) for mode, k in state["keys"].items()}
        return cls(state["run"], keys)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = {mode: asdict(k) for mode, k in self.keys.items()}
        path.write_text(json.dumps({"run": self.id, "keys": keys}, indent=2) + "\n")


def results_path(runs: Path, run: Run, mode: Mode) -> Path:
    return runs / run.id / f"{mode.name}.jsonl"


def load_results(path: Path) -> list[Result]:
    """In the order they were saved, which is the order they were sent."""
    if not path.exists():
        return []
    return [Result(**json.loads(line)) for line in path.read_text().splitlines() if line]


def append_result(path: Path, result: Result) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result)) + "\n")


class Stopped(Exception):
    """A request failed, or a fallback answered it: the run stops here."""


class Replayer:
    def __init__(
        self,
        http: httpx.Client,
        max_tokens: int,
        pause: float,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.http = http
        self.max_tokens = max_tokens
        self.pause = pause
        self.sleep = sleep

    def send(self, run_id: str, mode: Mode, key: str, query: Query) -> Result:
        body = {
            "model": mode.model,
            "messages": [
                {"role": "system", "content": system_prompt(run_id, mode)},
                {"role": "user", "content": query.prompt},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {key}"}
        if mode.cache is not None:
            headers["x-gateway-cache"] = mode.cache
        started = time.perf_counter()
        r = self.http.post("/chat/completions", json=body, headers=headers)
        latency_ms = (time.perf_counter() - started) * 1000
        # Only provider calls count against the provider's rate limit; and a store lands after
        # its response, so the pause also lets the next lookup find it.
        if r.headers.get("x-gateway-attempts", "0") != "0":
            self.sleep(self.pause)
        return Result(
            id=query.id,
            status=r.status_code,
            latency_ms=round(latency_ms, 1),
            cache=r.headers.get("x-gateway-cache", "-"),
            route=r.headers.get("x-gateway-route"),
            fallback=r.headers.get("x-gateway-fallback") == "true",
        )

    def replay(self, run_id: str, mode: Mode, key: str, queries: list[Query], path: Path) -> int:
        """Sends the prompts not yet in `path`, saving each answer. Returns how many it sent;
        raises Stopped at a failure or a fallback."""
        done = {r.id for r in load_results(path)}
        sent = 0
        for q in queries:
            if q.id in done:
                continue
            try:
                result = self.send(run_id, mode, key, q)
            except httpx.HTTPError as e:
                raise Stopped(f"{mode.name} {q.id}: {e!r}") from e
            sent += 1
            if not result.answered:
                raise Stopped(f"{mode.name} {q.id}: HTTP {result.status}")
            append_result(path, result)
            print(
                f"{mode.name:>12}  {q.id}  {result.cache:<8}  {result.route or '-':<6}  "
                f"{result.latency_ms:7.0f} ms  {q.prompt[:50]}"
            )
            if result.fallback:
                raise Stopped(f"{mode.name} {q.id}: answered by a fallback provider")
        return sent


def mode_cost(conn: psycopg.Connection, key_id: int) -> Decimal:
    """Everything the key spent, failed and retried calls included."""
    row = conn.execute(
        "SELECT coalesce(sum(cost_usd), 0) FROM request_logs WHERE key_id = %s", (key_id,)
    ).fetchone()
    return row[0]


def _modes(names: list[str] | None) -> list[Mode]:
    if not names:
        return MODES
    by_name = {m.name: m for m in MODES}
    return [by_name[n] for n in names]


def run(args: argparse.Namespace, queries: list[Query], modes: list[Mode], pause: float) -> int:
    state = args.dir / "state.json"
    current = None if args.new else Run.load(state)
    if current is None:
        current = Run.new()
        current.save(state)
        print(f"Started run {current.id}.")
    replayer = Replayer(httpx.Client(base_url=args.base_url, timeout=60), args.max_tokens, pause)
    with psycopg.connect(get_settings().database_url, autocommit=True) as conn:
        for mode in modes:
            if mode.name not in current.keys:
                key, record = create_key(conn, f"replay-{current.id}-{mode.name}", KEY_RPM, KEY_TPM)
                current.keys[mode.name] = ModeKey(record.id, key)
                current.save(state)
            path = results_path(args.dir, current, mode)
            try:
                replayer.replay(current.id, mode, current.keys[mode.name].key, queries, path)
            except Stopped as e:
                print(f"\nStopped: {e}. Rerun to resume.", file=sys.stderr)
                return 1
    print(f"\nRun {current.id}: {', '.join(m.name for m in modes)} complete.")
    return 0


def report(args: argparse.Namespace, queries: list[Query]) -> int:
    current = Run.load(args.dir / "state.json")
    if current is None:
        print("No run yet: `run` or `overhead` first.", file=sys.stderr)
        return 1
    print(f"Run {current.id}")
    if BASELINE.name in current.keys:
        auto = load_registry(get_settings().models_file).auto
        summaries = []
        with psycopg.connect(get_settings().database_url) as conn:
            for mode in MODES:
                if mode.name not in current.keys:
                    continue
                results = load_results(results_path(args.dir, current, mode))
                if len(results) < len(queries):
                    print(f"{mode.name}: {len(results)} of {len(queries)} prompts so far")
                alias = auto.cheap if auto and mode.model == "auto" else None
                cost = mode_cost(conn, current.keys[mode.name].id)
                summaries.append(summarize(mode.name, results, queries, cost, alias))
        print()
        print(markdown_table(summaries))
        for s in summaries:
            if s.false_hits:
                print(f"\n{s.mode} false hits: {', '.join(s.false_hits)}")
    measured = [m for m in OVERHEAD_MODES if m.name in current.keys]
    if measured:
        rows = [
            overhead(m.name, load_results(results_path(args.dir, current, m))) for m in measured
        ]
        print()
        print(overhead_table(rows))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="replay the log; resumes the current run")
    run_parser.add_argument(
        "--mode", action="append", choices=[m.name for m in MODES], help="repeat; default all"
    )
    run_parser.add_argument(
        "--pause", type=float, default=5.0, help="seconds after a provider call"
    )
    overhead_parser = commands.add_parser(
        "overhead", help="the gateway's own latency, on the mock alias; resumes too"
    )
    for sub in (run_parser, overhead_parser):
        sub.add_argument("--base-url", default="http://localhost:8000/v1")
        sub.add_argument("--new", action="store_true", help="start another run, new keys")
        sub.add_argument("--limit", type=int, help="only the first N prompts of the log")
        # Well under auto's long_answer_max_tokens, which would send every request to strong.
        sub.add_argument("--max-tokens", type=int, default=512)
    report_parser = commands.add_parser("report", help="the results tables of the current run")
    report_parser.add_argument("--limit", type=int, help="score only the first N prompts")
    for sub in (run_parser, overhead_parser, report_parser):
        sub.add_argument("--dir", type=Path, default=RUNS, help="where runs are kept")
    args = parser.parse_args(argv)

    queries = load_queries(DATASET)[: args.limit]
    match args.command:
        case "run":
            return run(args, queries, _modes(args.mode), args.pause)
        case "overhead":
            # The mock provider has no rate limit to stay under.
            return run(args, queries, OVERHEAD_MODES, pause=0)
        case _:
            return report(args, queries)


if __name__ == "__main__":
    sys.exit(main())
