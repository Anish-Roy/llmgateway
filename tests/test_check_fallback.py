"""The fallback check: what counts as a success, its tallies, and the outage config it runs on."""

import json
from pathlib import Path

import httpx

from app.routing.registry import load_registry
from scripts.check_fallback import (
    SCENARIOS,
    Outcome,
    markdown_table,
    send,
    succeeded,
    tally,
)

OUTAGE = Path(__file__).resolve().parent.parent / "config" / "outage.yaml"


def test_a_plain_reply_succeeds_on_a_200():
    assert succeeded(200, "{}", stream=False)
    assert not succeeded(503, "{}", stream=False)


def test_a_stream_succeeds_only_if_it_finished():
    assert succeeded(200, 'data: {"id": 1}\n\ndata: [DONE]\n\n', stream=True)
    # Cut after its first token: a 200 with an error event and no [DONE].
    assert not succeeded(200, 'data: {"id": 1}\n\ndata: {"error": {}}\n\n', stream=True)
    assert not succeeded(503, "", stream=True)


def test_a_request_turns_the_cache_off_and_reads_the_fallback_header():
    seen: list[httpx.Request] = []

    def gateway(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, headers={"x-gateway-fallback": "true"}, text="data: [DONE]\n\n")

    http = httpx.Client(base_url="http://gw/v1", transport=httpx.MockTransport(gateway))
    outcome = send(http, "gw-k", "flaky-with-fallback", stream=True, n=3)

    assert outcome == Outcome("flaky-with-fallback", True, succeeded=True, fallback=True)
    body = json.loads(seen[0].content)
    assert body["model"] == "flaky-with-fallback" and body["stream"] is True
    assert seen[0].headers["x-gateway-cache"] == "off"
    assert seen[0].headers["authorization"] == "Bearer gw-k"


def test_tally_counts_one_alias():
    outcomes = [
        Outcome("a", False, True, False),
        Outcome("a", True, True, True),
        Outcome("a", False, False, False),
        Outcome("b", False, True, False),
    ]
    t = tally("a", outcomes)
    assert (t.requests, t.succeeded, t.fell_back) == (3, 2, 1)
    assert t.success_rate == 2 / 3


def test_the_table_has_a_row_per_alias():
    lines = markdown_table([tally("a", [Outcome("a", False, True, False)])]).splitlines()
    assert lines[2] == "| a | 1 | 1 | 100.00% | 0 |"


def test_the_outage_config_has_both_aliases_of_every_scenario():
    config = load_registry(OUTAGE)
    for scenario in SCENARIOS:
        alone = config.aliases[f"{scenario}-alone"]
        backed = config.aliases[f"{scenario}-with-fallback"]
        assert len(alone) == 1
        # The same failing provider first, then the backup.
        assert backed[0] == alone[0] and backed[1].provider == "backup"
