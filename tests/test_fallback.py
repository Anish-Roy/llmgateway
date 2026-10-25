"""Retry and fallback in ModelRouter, with scripted providers. No network, no real waiting."""

import asyncio

import pytest

from app.api.schemas import ChatMessage, ChatRequest, ChatResponse, Choice
from app.providers.base import ProviderError
from app.routing.registry import Target
from app.routing.router import ModelRouter


class ScriptedProvider:
    """Plays back a list of outcomes, one per call: an int is an error status, "ok" a reply."""

    def __init__(self, name: str, *outcomes: int | str, delay_s: float = 0.0) -> None:
        self.name = name
        self._outcomes = list(outcomes)
        self.delay_s = delay_s
        self.calls = 0

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        outcome = self._outcomes.pop(0) if self._outcomes else "ok"
        if outcome != "ok":
            raise ProviderError(outcome, f"{self.name} failed with {outcome}", provider=self.name)
        return ChatResponse(
            model=request.model,
            choices=[Choice(message=ChatMessage(role="assistant", content=f"from {self.name}"))],
        )


class SleepRecorder:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


REQUEST = ChatRequest(model="fast", messages=[{"role": "user", "content": "hi"}])


def router_for(*providers: ScriptedProvider, **kwargs) -> tuple[ModelRouter, SleepRecorder]:
    sleep = SleepRecorder()
    aliases = {"fast": [Target(provider=p.name, model=f"{p.name}-model") for p in providers]}
    router = ModelRouter(aliases, {p.name: p for p in providers}, sleep=sleep, **kwargs)
    return router, sleep


async def test_first_provider_answers_without_retry_or_fallback():
    primary, backup = ScriptedProvider("primary", "ok"), ScriptedProvider("backup")
    router, sleep = router_for(primary, backup)

    routed = await router.chat(REQUEST)

    assert routed.target.provider == "primary"
    assert (routed.attempts, routed.fallback) == (1, False)
    assert backup.calls == 0
    assert sleep.delays == []


async def test_5xx_is_retried_once_on_the_same_provider():
    primary, backup = ScriptedProvider("primary", 503, "ok"), ScriptedProvider("backup")
    router, sleep = router_for(primary, backup)

    routed = await router.chat(REQUEST)

    assert routed.target.provider == "primary"
    assert (routed.attempts, routed.fallback) == (2, False)
    assert backup.calls == 0
    assert len(sleep.delays) == 1


async def test_5xx_twice_falls_back_to_the_next_provider():
    primary, backup = ScriptedProvider("primary", 503, 500), ScriptedProvider("backup", "ok")
    router, _ = router_for(primary, backup)

    routed = await router.chat(REQUEST)

    assert routed.target.provider == "backup"
    assert routed.response.model == "backup-model"
    assert (routed.attempts, routed.fallback) == (3, True)
    assert primary.calls == 2


@pytest.mark.parametrize("status", [429, 504])
async def test_429_and_timeouts_fall_back_without_retrying(status):
    primary, backup = ScriptedProvider("primary", status), ScriptedProvider("backup", "ok")
    router, sleep = router_for(primary, backup)

    routed = await router.chat(REQUEST)

    assert routed.target.provider == "backup"
    assert primary.calls == 1
    assert sleep.delays == []


async def test_400_is_returned_without_fallback():
    primary, backup = ScriptedProvider("primary", 400), ScriptedProvider("backup", "ok")
    router, _ = router_for(primary, backup)

    with pytest.raises(ProviderError) as exc_info:
        await router.chat(REQUEST)

    assert exc_info.value.status == 400
    # The backup would reject the same bad request, so it isn't asked.
    assert backup.calls == 0


async def test_rejected_gateway_key_falls_back_without_retrying():
    class BadKey(ScriptedProvider):
        async def chat(self, request):
            self.calls += 1
            raise ProviderError(502, "bad key", provider=self.name, retry_same_provider=False)

    primary, backup = BadKey("primary"), ScriptedProvider("backup", "ok")
    router, _ = router_for(primary, backup)

    routed = await router.chat(REQUEST)

    assert routed.target.provider == "backup"
    assert primary.calls == 1


async def test_fallback_400_reports_the_primarys_retryable_error():
    # The primary is down; the backup can't handle something in this request. The request
    # is fine for the primary, so tell the client to retry later, not that it's invalid.
    primary, backup = ScriptedProvider("primary", 503, 503), ScriptedProvider("backup", 400)
    router, _ = router_for(primary, backup)

    with pytest.raises(ProviderError) as exc_info:
        await router.chat(REQUEST)

    assert exc_info.value.status == 503
    assert exc_info.value.provider == "primary"


async def test_fallback_400_moves_on_to_a_third_provider():
    primary = ScriptedProvider("primary", 429)
    backup = ScriptedProvider("backup", 400)
    third = ScriptedProvider("third", "ok")
    router, _ = router_for(primary, backup, third)

    routed = await router.chat(REQUEST)

    assert routed.target.provider == "third"


async def test_all_failing_returns_the_latest_retryable_error():
    primary, backup = ScriptedProvider("primary", 503, 503), ScriptedProvider("backup", 429)
    router, _ = router_for(primary, backup)

    with pytest.raises(ProviderError) as exc_info:
        await router.chat(REQUEST)

    # A 429 makes the openai SDK back off before retrying.
    assert exc_info.value.status == 429
    assert (primary.calls, backup.calls) == (2, 1)


async def test_deadline_stops_a_slow_provider_and_skips_the_rest():
    primary = ScriptedProvider("primary", "ok", delay_s=1.0)
    backup = ScriptedProvider("backup", "ok")
    router, _ = router_for(primary, backup, deadline_s=0.05)

    with pytest.raises(ProviderError) as exc_info:
        await router.chat(REQUEST)

    assert exc_info.value.status == 504
    assert exc_info.value.message == "Gateway deadline exceeded"
    assert backup.calls == 0


async def test_backoff_never_sleeps_past_the_deadline():
    primary = ScriptedProvider("primary", 503, "ok")
    router, sleep = router_for(primary, backoff_s=100.0, deadline_s=1.0)

    await router.chat(REQUEST)

    assert 0 <= sleep.delays[0] <= 1.0


class CantDoIt(ScriptedProvider):
    """Rejects the request the way an adapter does for a feature it lacks (Gemini + tools)."""

    async def chat(self, request):
        self.calls += 1
        raise ProviderError(
            400,
            "'tools' is not supported",
            param="tools",
            provider=self.name,
            provider_specific=True,
        )


async def test_provider_specific_400_on_the_first_target_falls_back():
    primary, backup = CantDoIt("primary"), ScriptedProvider("backup", "ok")
    router, _ = router_for(primary, backup)

    routed = await router.chat(REQUEST)

    assert routed.target.provider == "backup"
    assert primary.calls == 1


async def test_provider_specific_400_everywhere_reaches_the_client():
    router, _ = router_for(CantDoIt("primary"), CantDoIt("backup"))

    with pytest.raises(ProviderError) as exc_info:
        await router.chat(REQUEST)

    assert exc_info.value.status == 400
    assert exc_info.value.param == "tools"
