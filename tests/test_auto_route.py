"""`model: "auto"` in the route: which provider answers, the headers and log row that say why,
and how it shares the caches with the alias it picks."""

import httpx2

from app.api.routes import get_model_router
from app.providers.mock import MockProvider
from app.routing.auto import AutoRouter
from app.routing.registry import Target
from app.routing.router import ModelRouter
from tests.test_auto_router import spec
from tests.test_exact_cache import Gateway, events

EMAIL = "priya.sharma@example.com"


class AutoGateway(Gateway):
    """A gateway where "fast" is served by `small` and "strong" by `big`, with `auto` on."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.small, self.big = MockProvider(), MockProvider()
        router = ModelRouter(
            {
                "fast": [Target(provider="small", model="small-m")],
                "strong": [Target(provider="big", model="big-m")],
            },
            {"small": self.small, "big": self.big},
            auto=AutoRouter(spec()),
        )
        self.app.dependency_overrides[get_model_router] = lambda: router

    async def models(self) -> list[str]:
        transport = httpx2.ASGITransport(app=self.app)
        async with httpx2.AsyncClient(transport=transport, base_url="http://gateway.test") as c:
            resp = await c.get("/v1/models", headers={"Authorization": f"Bearer {self.key}"})
        return [m["id"] for m in resp.json()["data"]]


def ask(content: str, model: str = "auto", **extra) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        **extra,
    }


async def test_an_easy_request_is_served_by_the_cheap_alias():
    gw = AutoGateway()

    resp = await gw.post(ask("What is Docker?"))

    assert resp.status_code == 200
    assert (gw.small.calls, gw.big.calls) == (1, 0)
    assert resp.headers["x-gateway-route"] == "fast"
    assert resp.headers["x-gateway-route-reason"] == "default"
    assert resp.headers["x-gateway-provider"] == "small"


async def test_a_hard_request_is_served_by_the_strong_alias():
    gw = AutoGateway()

    resp = await gw.post(ask("Prove it"))

    assert (gw.small.calls, gw.big.calls) == (0, 1)
    assert resp.headers["x-gateway-route"] == "strong"
    assert resp.headers["x-gateway-route-reason"] == "keyword:prove"
    assert resp.json()["model"] == "big-m"


async def test_the_log_keeps_auto_and_says_what_it_picked():
    gw = AutoGateway()

    await gw.post(ask("Prove it"))
    await gw.post(ask("What is Docker?", model="fast"))

    auto, named = await gw.rows()
    assert (auto.requested_model, auto.routed_alias, auto.route_reason) == (
        "auto",
        "strong",
        "keyword:prove",
    )
    assert (auto.provider, auto.model) == ("big", "big-m")
    assert (named.requested_model, named.routed_alias, named.route_reason) == ("fast", None, None)


async def test_a_named_alias_gets_no_route_headers():
    resp = await AutoGateway().post(ask("What is Docker?", model="fast"))

    assert "x-gateway-route" not in resp.headers
    assert "x-gateway-route-reason" not in resp.headers


async def test_a_stream_gets_the_route_headers():
    gw = AutoGateway()

    resp = await gw.post(ask("Prove it", stream=True))

    assert resp.headers["x-gateway-route"] == "strong"
    assert events(resp)[-1] == "[DONE]"
    (row,) = await gw.rows()
    assert (row.stream, row.routed_alias) == (True, "strong")


async def test_auto_shares_cache_entries_with_the_alias_it_picks():
    gw = AutoGateway()

    await gw.post(ask("What is Docker?", model="fast"))
    hit = await gw.post(ask("What is Docker?"))

    assert hit.headers["x-gateway-cache"] == "exact"
    assert gw.small.calls == 1
    # A hit still says how it was routed, in the headers and the log.
    assert hit.headers["x-gateway-route"] == "fast"
    _, row = await gw.rows()
    assert (row.cache_status, row.routed_alias, row.route_reason) == ("exact", "fast", "default")


async def test_an_entry_for_one_alias_never_answers_the_other():
    gw = AutoGateway()

    await gw.post(ask("Prove it", model="fast"))
    resp = await gw.post(ask("Prove it"))

    assert resp.headers["x-gateway-cache"] == "miss"
    assert gw.big.calls == 1


async def test_pii_is_masked_before_routing_and_never_reaches_the_reason():
    gw = AutoGateway()

    # Raw, this is 21 tokens (82 chars); masked, 17 (67 chars).
    resp = await gw.post(ask(f"Mail {EMAIL} now please, {'x' * 40}"))

    assert resp.headers["x-gateway-route"] == "fast"
    (row,) = await gw.rows()
    assert row.route_reason == "default"
    assert EMAIL not in repr(row)


async def test_without_an_auto_router_auto_is_an_unknown_model():
    gw = Gateway()

    resp = await gw.post(ask("hi"))

    assert resp.status_code == 404
    (row,) = await gw.rows()
    assert (row.requested_model, row.routed_alias) == ("auto", None)


async def test_a_rate_limited_auto_request_is_logged_unrouted():
    gw = AutoGateway(rpm=1)

    await gw.post(ask("hi"))
    limited = await gw.post(ask("hi"))

    assert limited.status_code == 429
    assert (await gw.rows())[1].routed_alias is None


async def test_models_lists_auto():
    assert await AutoGateway().models() == ["fast", "strong", "auto"]
