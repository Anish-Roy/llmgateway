"""Manual check: sends a mixed batch of requests through a running gateway, so the cost
dashboard has every kind of row to show. Calls the real providers; not for CI.

uv run uvicorn app.main:app            # with GROQ_API_KEY / GEMINI_API_KEY in .env
uv run python -m scripts.create_key create demo-search      # once per key; keep what it prints
uv run python -m scripts.create_key create demo-support
uv run python -m scripts.create_key create demo-tight --rpm 2
uv run python -m scripts.send_traffic --key search=gw-... --key support=gw-... \\
    --tight-key gw-...

Prompts come from data/queries.jsonl, which pairs questions with rephrasings of them. Each
team key gets its own pairs, since the semantic cache is scoped per key:
    - the question, opted in to the semantic cache: a miss that stores the answer;
    - its rephrasing, opted in too: a semantic hit when MiniLM puts it above the threshold,
      else a miss;
    - every third question again, from the next key: an exact hit (that cache is shared).
Then a few standalone prompts, half as streams (time to first token) and half as `model:
"auto"`; and on the free mock aliases a fallback, a stream cut partway and an unknown model.
The tight key, if given, sends until it's refused (429).

Everything real goes to --model at temperature 0, so the answers are cacheable, and is logged
at list price: hits then show what they saved. A run costs well under a cent at list price.
Groq's free tier allows 30 requests a minute per model: --pause keeps under it. Rerunning
within CACHE_TTL_S turns earlier misses into exact hits: use another --seed for fresh ones.
"""

import argparse
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import httpx

DATASET = Path(__file__).resolve().parent.parent / "data" / "queries.jsonl"
SEMANTIC = {"x-gateway-cache": "semantic"}


@dataclass(frozen=True)
class Pair:
    question: str
    rephrased: str


@dataclass(frozen=True)
class Sample:
    pairs: list[Pair]
    standalone: list[str]


def load_sample(pairs: int, standalone: int, seed: int) -> Sample:
    """Easy prompts only: short answers, so max_tokens isn't what cuts them. Trap prompts (alike
    in words, not in meaning) are left out: this is traffic, not the threshold experiment."""
    rows = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    by_id = {r["id"]: r for r in rows}
    easy = [r for r in rows if r["difficulty"] == "easy" and not r["trap_of"]]
    found = [
        Pair(by_id[r["dup_of"]]["prompt"], r["prompt"])
        for r in easy
        if r["dup_of"] and by_id[r["dup_of"]]["difficulty"] == "easy"
    ]
    duplicated = {r["dup_of"] for r in rows if r["dup_of"]}
    alone = [r["prompt"] for r in easy if not r["dup_of"] and r["id"] not in duplicated]
    rng = random.Random(seed)
    return Sample(rng.sample(found, pairs), rng.sample(alone, standalone))


class Traffic:
    def __init__(self, base_url: str, model: str, max_tokens: int, pause: float) -> None:
        self.http = httpx.Client(base_url=base_url, timeout=60)
        self.model = model
        self.max_tokens = max_tokens
        self.pause = pause
        self.tally: Counter[str] = Counter()

    def ask(
        self,
        who: str,
        key: str,
        prompt: str,
        *,
        model: str | None = None,
        stream: bool = False,
        headers: dict[str, str] | None = None,
    ) -> int:
        body = {
            "model": model or self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        headers = {"Authorization": f"Bearer {key}", **(headers or {})}
        started = time.perf_counter()
        with self.http.stream("POST", "/chat/completions", json=body, headers=headers) as r:
            text = r.read().decode()
        ms = (time.perf_counter() - started) * 1000
        cut = stream and r.is_success and '"error"' in text
        cache = r.headers.get("x-gateway-cache", "-")
        outcome = "cut mid-stream" if cut else cache if r.is_success else str(r.status_code)
        self.tally[outcome] += 1
        print(
            f"{who:>10}  {body['model']:<14} {'stream' if stream else 'plain':<6}  "
            f"{r.status_code}  {outcome:<14}  {r.headers.get('x-gateway-provider', '-'):<9}  "
            f"{'fallback' if r.headers.get('x-gateway-fallback') == 'true' else '':<8}  "
            f"{ms:7.0f} ms  {prompt[:50]}"
        )
        # Only provider calls count against the provider's rate limit; and a store lands
        # after its response, so the pause also lets the next lookup find it.
        if r.headers.get("x-gateway-attempts", "0") != "0":
            time.sleep(self.pause)
        return r.status_code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument(
        "--key",
        action="append",
        required=True,
        metavar="NAME=KEY",
        help="a team's gateway key (repeat for more teams)",
    )
    parser.add_argument("--tight-key", help="a key with a low --rpm, to show 429s")
    parser.add_argument("--model", default="fast", help="the alias real traffic goes to")
    parser.add_argument("--pairs", type=int, default=20, help="questions with a rephrasing")
    parser.add_argument("--standalone", type=int, default=8, help="streams and auto requests")
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--pause", type=float, default=2.5, help="seconds after a provider call")
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()

    teams = []
    for pair in args.key:
        name, sep, key = pair.partition("=")
        if not sep or not key:
            parser.error(f"--key takes NAME=KEY, not {pair!r}")
        teams.append((name, key))
    sample = load_sample(args.pairs, args.standalone, args.seed)
    traffic = Traffic(args.base_url, args.model, args.max_tokens, args.pause)

    for i, pair in enumerate(sample.pairs):
        name, key = teams[i % len(teams)]
        traffic.ask(name, key, pair.question, headers=SEMANTIC)
        traffic.ask(name, key, pair.rephrased, headers=SEMANTIC)
        if i % 3 == 0:
            other, other_key = teams[(i + 1) % len(teams)]
            traffic.ask(other, other_key, pair.question)

    for i, prompt in enumerate(sample.standalone):
        name, key = teams[i % len(teams)]
        if i % 2 == 0:
            traffic.ask(name, key, prompt, stream=True)
        else:
            traffic.ask(name, key, prompt, model="auto")

    name, key = teams[0]
    for _ in range(3):
        traffic.ask(
            name, key, "Say hello.", model="demo-fallback", headers={"x-gateway-cache": "off"}
        )
    traffic.ask(name, key, "Tell me a story.", model="demo-midstream", stream=True)
    traffic.ask(name, key, "Hello?", model="no-such-model")

    if args.tight_key:
        # Its RPM bucket starts full: the request after the last token is refused.
        for _ in range(20):
            if traffic.ask("tight", args.tight_key, "Hi.", model="mock") == 429:
                break

    print()
    print(", ".join(f"{outcome}: {n}" for outcome, n in traffic.tally.most_common()))


if __name__ == "__main__":
    main()
