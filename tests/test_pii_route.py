"""PII masking in the route: what the providers and both caches get.

Restoring the real values in the reply comes later (step 3), so for now the mock's echo comes
back with placeholders in it.
"""

from app.providers.mock import MockProvider
from tests.test_exact_cache import Gateway
from tests.test_semantic_cache_route import OPT_IN, SemanticGateway, _similar

EMAIL = "priya.sharma@example.com"
PHONE = "98765 43210"
CARD = "4111 1111 1111 1111"
RAW = [EMAIL, PHONE, CARD, "9876543210", "4111111111111111"]


def ask(content: str, *, system: str | None = None, **extra) -> dict:
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": content})
    return {"model": "fast", "messages": messages, "temperature": 0, **extra}


def sent(provider: MockProvider) -> list[list[str | None]]:
    """The message contents of every request `provider` got."""
    return [[m.content for m in r.messages] for r in provider.requests]


def assert_no_raw_values(*things: object) -> None:
    for thing in things:
        text = repr(thing)
        for value in RAW:
            assert value not in text


async def test_the_provider_gets_placeholders():
    provider = MockProvider()
    gw = Gateway(("groq", provider))

    resp = await gw.post(ask(f"Mail {EMAIL} or call {PHONE}, card {CARD}"))

    assert resp.status_code == 200
    assert sent(provider) == [["Mail [EMAIL_1] or call [PHONE_1], card [CARD_1]"]]


async def test_a_streamed_request_is_masked_too():
    provider = MockProvider()
    gw = Gateway(("groq", provider))

    resp = await gw.post(ask(f"Mail {EMAIL}", stream=True))

    assert resp.status_code == 200
    assert sent(provider) == [["Mail [EMAIL_1]"]]


async def test_every_message_is_masked_with_one_numbering():
    provider = MockProvider()
    gw = Gateway(("groq", provider))

    await gw.post(ask(f"Also copy {EMAIL}.", system="The admin is admin@corp.in."))

    assert sent(provider) == [["The admin is [EMAIL_1].", "Also copy [EMAIL_2]."]]


async def test_a_fallback_provider_gets_placeholders_too():
    flaky, groq = MockProvider(failure_rate=1.0), MockProvider()
    gw = Gateway(("flaky", flaky), ("groq", groq))

    resp = await gw.post(ask(f"Mail {EMAIL}"))

    assert resp.headers["x-gateway-fallback"] == "true"
    # The retry too.
    assert sent(flaky) == [["Mail [EMAIL_1]"]] * 2
    assert sent(groq) == [["Mail [EMAIL_1]"]]


async def test_a_request_without_pii_reaches_the_provider_as_sent():
    provider = MockProvider()
    gw = Gateway(("groq", provider))

    await gw.post(ask("What is HTTP?", seed=7, stop=["\n"]))

    (request,) = provider.requests
    assert request.messages[0].content == "What is HTTP?"
    assert request.model_extra == {"seed": 7, "stop": ["\n"]}


async def test_options_reach_the_provider_with_a_masked_request():
    provider = MockProvider()
    gw = Gateway(("groq", provider))

    await gw.post(ask(f"Mail {EMAIL}", seed=7, max_tokens=50))

    (request,) = provider.requests
    assert (request.model_extra, request.max_tokens, request.temperature) == ({"seed": 7}, 50, 0)


# The caches.


async def test_the_exact_cache_never_holds_the_real_values():
    gw = Gateway()

    await gw.post(ask(f"Mail {EMAIL} or call {PHONE}, card {CARD}"))
    await gw.post(ask(f"Mail {EMAIL}", stream=True))

    assert len(gw.store.entries) == 2
    assert_no_raw_values(gw.store.entries)


async def test_requests_that_differ_only_in_pii_share_an_exact_entry():
    gw = Gateway()

    await gw.post(ask(f"Write to {EMAIL} about {PHONE}"))
    hit = await gw.post(ask("Write to neha@mail.org about +91 7012345678"))

    assert hit.headers["x-gateway-cache"] == "exact"
    assert gw.calls == 1


async def test_the_semantic_cache_embeds_and_stores_placeholders_only():
    question = f"What's the capital of France? Reply to {EMAIL}"
    rephrased = "capital of france? reply to neha@mail.org"
    gw = SemanticGateway(
        embedder=_similar(
            "What's the capital of France? Reply to [EMAIL_1]",
            "capital of france? reply to [EMAIL_1]",
        )
    )

    await gw.post(ask(question), headers=OPT_IN)
    hit = await gw.post(ask(rephrased), headers=OPT_IN)

    assert hit.headers["x-gateway-cache"] == "semantic"
    assert gw.embedder.calls == [
        "What's the capital of France? Reply to [EMAIL_1]",
        "capital of france? reply to [EMAIL_1]",
    ]
    assert_no_raw_values(gw.semantic_store.entries, gw.store.entries)
