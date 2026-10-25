"""How many requests still succeed when a provider fails, with fallback and without.

Sends requests through a running gateway to the aliases in config/outage.yaml, mock providers
that fail a set share of calls with a 503 before the first token:
    down       fails every call: an outage;
    flaky      fails half its calls;
    backup     the fallback, which fails 10% of its calls.
Each scenario goes to two aliases: the failing provider alone, and with `backup` behind it.
Half the requests are streamed. The caches are off, so every request reaches a provider.

A request succeeds if it gets a 200 and, streamed, ends with `data: [DONE]`: a stream that
failed after its first token would be a 200 with an error event, and counts as failed. The
mocks fail only before the first token, which is where fallback can still happen.

The gateway retries a 503 once on the same provider, then falls back. So, expected:
    down alone           0%                      down with fallback    1 - 0.1^2  = 99%
    flaky alone          1 - 0.5^2 = 75%         flaky with fallback   1 - 0.5^2 x 0.1^2 = 99.75%

Free: no real provider is called.

MODELS_FILE=config/outage.yaml SEMANTIC_CACHE_ENABLED=false \\
    uv run uvicorn app.main:app --port 8002
uv run python -m scripts.check_fallback --base-url http://localhost:8002/v1

The script creates a key with high limits for the run and revokes it at the end.
"""

import argparse
import secrets
import sys
from collections import Counter
from dataclasses import dataclass

import httpx
import psycopg

from app.auth.admin import create_key, revoke_key
from app.config import get_settings

SCENARIOS = ["down", "flaky"]
KEY_RPM = 100_000
KEY_TPM = 100_000_000


@dataclass(frozen=True)
class Outcome:
    alias: str
    stream: bool
    succeeded: bool
    fallback: bool


@dataclass(frozen=True)
class Tally:
    alias: str
    requests: int
    succeeded: int
    fell_back: int

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.requests if self.requests else 0.0


def succeeded(status: int, body: str, stream: bool) -> bool:
    """A 200, and for a stream, one that finished: a failed stream ends without [DONE]."""
    if status != 200:
        return False
    return not stream or "data: [DONE]" in body


def send(http: httpx.Client, key: str, alias: str, stream: bool, n: int) -> Outcome:
    body = {
        "model": alias,
        # A different prompt each time, so nothing could be served from a cache either way.
        "messages": [{"role": "user", "content": f"Request {n}: say hello."}],
        "temperature": 0,
        "stream": stream,
    }
    headers = {"Authorization": f"Bearer {key}", "x-gateway-cache": "off"}
    with http.stream("POST", "/chat/completions", json=body, headers=headers) as r:
        text = r.read().decode()
    return Outcome(
        alias=alias,
        stream=stream,
        succeeded=succeeded(r.status_code, text, stream),
        fallback=r.headers.get("x-gateway-fallback") == "true",
    )


def tally(alias: str, outcomes: list[Outcome]) -> Tally:
    mine = [o for o in outcomes if o.alias == alias]
    return Tally(
        alias=alias,
        requests=len(mine),
        succeeded=sum(o.succeeded for o in mine),
        fell_back=sum(o.fallback for o in mine),
    )


def markdown_table(tallies: list[Tally]) -> str:
    lines = [
        "| Alias | Requests | Succeeded | Success rate | Answered by the fallback |",
        "|---|--:|--:|--:|--:|",
    ]
    for t in tallies:
        lines.append(
            f"| {t.alias} | {t.requests} | {t.succeeded} | {t.success_rate:.2%} | {t.fell_back} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8002/v1")
    parser.add_argument("--requests", type=int, default=1000, help="per alias, half streamed")
    args = parser.parse_args(argv)

    aliases = [f"{s}-{kind}" for s in SCENARIOS for kind in ("alone", "with-fallback")]
    http = httpx.Client(base_url=args.base_url, timeout=30)
    with psycopg.connect(get_settings().database_url, autocommit=True) as conn:
        key, record = create_key(conn, f"check-fallback-{secrets.token_hex(4)}", KEY_RPM, KEY_TPM)
        try:
            outcomes = []
            for alias in aliases:
                for n in range(args.requests):
                    outcomes.append(send(http, key, alias, stream=n % 2 == 1, n=n))
                print(f"{alias}: {Counter(o.succeeded for o in outcomes if o.alias == alias)}")
        finally:
            revoke_key(conn, record.id)

    print()
    print(markdown_table([tally(alias, outcomes) for alias in aliases]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
