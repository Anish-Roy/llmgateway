"""PII masking in the route: what the providers and both caches get (placeholders only), and
what the client gets back (its own real values), on every path: plain and streamed, misses
and hits of both caches.

The mock provider echoes the last user message, so a reply holds the request's placeholders
the way a model's reply would.
"""

import json
import time
import uuid
from collections.abc import AsyncIterator

from app.api.schemas import ChatChunk, ChatRequest, ChunkChoice, Delta, Usage
from app.providers.base import ProviderError
from app.providers.mock import MockProvider
from tests.test_exact_cache import Gateway, events, sdk, streamed_text
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


# The client gets its own values back.


async def test_a_reply_gets_the_real_values_back():
    gw = Gateway()

    resp = await gw.post(ask(f"Mail {EMAIL} or call {PHONE}, card {CARD}"))

    reply = resp.json()["choices"][0]["message"]["content"]
    assert reply == f"Echo: Mail {EMAIL} or call {PHONE}, card {CARD}"


async def test_a_stream_gets_the_real_values_back():
    gw = Gateway()

    resp = await gw.post(ask(f"Mail {EMAIL} or call {PHONE}", stream=True))

    assert streamed_text(resp) == f"Echo: Mail {EMAIL} or call {PHONE}"
    assert events(resp)[-1] == "[DONE]"


async def test_an_exact_hit_gets_this_requests_values():
    gw = Gateway()

    first = await gw.post(ask(f"Write to {EMAIL}"))
    hit = await gw.post(ask("Write to neha@mail.org"))

    assert hit.headers["x-gateway-cache"] == "exact"
    assert first.json()["choices"][0]["message"]["content"] == f"Echo: Write to {EMAIL}"
    assert hit.json()["choices"][0]["message"]["content"] == "Echo: Write to neha@mail.org"
    assert_no_raw_values(gw.store.entries)
    assert "neha@mail.org" not in repr(gw.store.entries)


async def test_a_streamed_hit_gets_this_requests_values():
    gw = Gateway()

    await gw.post(ask(f"Write to {EMAIL}", stream=True))
    hit = await gw.post(ask("Write to neha@mail.org", stream=True))

    assert hit.headers["x-gateway-cache"] == "exact"
    assert streamed_text(hit) == "Echo: Write to neha@mail.org"
    assert_no_raw_values(gw.store.entries)


async def test_a_streamed_answer_serves_a_plain_hit_with_its_own_values():
    gw = Gateway()

    await gw.post(ask(f"Write to {EMAIL}", stream=True))
    hit = await gw.post(ask("Write to neha@mail.org"))

    assert hit.json()["choices"][0]["message"]["content"] == "Echo: Write to neha@mail.org"


async def test_a_semantic_hit_gets_this_requests_values():
    gw = SemanticGateway(
        embedder=_similar(
            "What's the capital of France? Reply to [EMAIL_1]",
            "capital of france? reply to [EMAIL_1]",
        )
    )

    await gw.post(ask(f"What's the capital of France? Reply to {EMAIL}"), headers=OPT_IN)
    hit = await gw.post(ask("capital of france? reply to neha@mail.org"), headers=OPT_IN)

    assert hit.headers["x-gateway-cache"] == "semantic"
    # The stored answer, to the first question, with the second one's email.
    content = hit.json()["choices"][0]["message"]["content"]
    assert content == "Echo: What's the capital of France? Reply to neha@mail.org"
    assert_no_raw_values(gw.semantic_store.entries)


async def test_a_placeholder_with_no_value_comes_back_as_it_is():
    # The model made it up, or the client wrote it: there's nothing to restore it to.
    gw = Gateway(("groq", MockProvider(reply="See [EMAIL_1] and [EMAIL_2].")))

    resp = await gw.post(ask(f"Mail {EMAIL}"))

    assert resp.json()["choices"][0]["message"]["content"] == f"See {EMAIL} and [EMAIL_2]."


async def test_the_sdk_gets_the_real_values_plain_and_streamed():
    gw = Gateway()
    client = sdk(gw)
    messages = [{"role": "user", "content": f"Mail {EMAIL}"}]

    plain = await client.chat.completions.create(model="fast", messages=messages)
    stream = await client.chat.completions.create(model="fast", messages=messages, stream=True)
    chunks = [chunk async for chunk in stream]

    assert plain.choices[0].message.content == f"Echo: Mail {EMAIL}"
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == f"Echo: Mail {EMAIL}"


# Placeholders split across chunks.


class ScriptedProvider(MockProvider):
    """Streams the given deltas as they are, then ends, or fails if `fail` is set."""

    def __init__(self, deltas: list[str], *, fail: bool = False) -> None:
        super().__init__()
        self.deltas = deltas
        self.fail = fail

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        self.requests.append(request)
        self.calls += 1
        meta = {"id": f"chatcmpl-{uuid.uuid4().hex}", "created": int(time.time()), "model": "m"}
        yield ChatChunk(**meta, choices=[ChunkChoice(delta=Delta(role="assistant"))])
        for delta in self.deltas:
            yield ChatChunk(**meta, choices=[ChunkChoice(delta=Delta(content=delta))])
        if self.fail:
            raise ProviderError(503, "gone", type="api_error", provider="scripted")
        yield ChatChunk(**meta, choices=[ChunkChoice(finish_reason="stop")])
        yield ChatChunk(**meta, usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10))


SPLIT = ["Sent to [EM", "AIL_1", "] and [", "PHONE_1]. Next: [EMAIL_"]


async def test_a_placeholder_split_across_chunks_is_restored():
    gw = Gateway(("groq", ScriptedProvider(SPLIT)))

    resp = await gw.post(ask(f"Mail {EMAIL}, call {PHONE}", stream=True))

    deltas = [
        json.loads(e)["choices"][0]["delta"].get("content")
        for e in events(resp)[:-1]
        if json.loads(e)["choices"]
    ]
    # No chunk carries half a placeholder. The unfinished "[EMAIL_" goes out with the end.
    assert deltas == [None, "Sent to ", "", f"{EMAIL} and ", f"{PHONE}. Next: ", "[EMAIL_"]
    assert streamed_text(resp) == f"Sent to {EMAIL} and {PHONE}. Next: [EMAIL_"


async def test_a_split_stream_is_stored_with_placeholders():
    gw = Gateway(("groq", ScriptedProvider(SPLIT)))

    await gw.post(ask(f"Mail {EMAIL}, call {PHONE}", stream=True))

    (entry,) = gw.store.entries.values()
    stored = entry.response.choices[0].message.content
    assert stored == "Sent to [EMAIL_1] and [PHONE_1]. Next: [EMAIL_"
    assert_no_raw_values(gw.store.entries)


async def test_a_held_tail_is_sent_before_a_mid_stream_error():
    gw = Gateway(("groq", ScriptedProvider(["Sent to [EMAIL_1] and [PHO"], fail=True)))

    resp = await gw.post(ask(f"Mail {EMAIL}, call {PHONE}", stream=True))

    *chunks, error = [json.loads(e) for e in events(resp)]
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert text == f"Sent to {EMAIL} and [PHO"
    assert error["error"]["message"] == "gone"


# The request log: which kinds were masked, never the values.


async def test_the_log_says_which_kinds_were_masked():
    gw = Gateway()

    await gw.post(ask(f"Mail {EMAIL}, card {CARD}, call {PHONE}, again {EMAIL}"))
    await gw.post(ask("What is HTTP?"))

    masked, plain = await gw.rows()
    assert masked.pii_types == ["card", "email", "phone"]
    assert plain.pii_types == []
    assert_no_raw_values(masked)


async def test_every_ending_logs_the_kinds_masked():
    # A stream, a cache hit and an alias that doesn't exist...
    gw = Gateway()
    await gw.post(ask(f"Mail {EMAIL}", stream=True))
    await gw.post(ask(f"Mail {EMAIL}"))
    await gw.post({**ask(f"Mail {EMAIL}"), "model": "nope"})
    # ...and every provider down.
    down = Gateway(("flaky", MockProvider(failure_rate=1.0)))
    await down.post(ask(f"Mail {EMAIL}"))

    streamed, hit, not_found = await gw.rows()
    (failed,) = await down.rows()
    assert (streamed.stream, hit.cache_status, not_found.status_code) == (True, "exact", 404)
    assert failed.outcome == "provider_error"
    assert [r.pii_types for r in (streamed, hit, not_found, failed)] == [["email"]] * 4


async def test_a_rate_limited_request_is_logged_before_masking():
    gw = Gateway(rpm=1)

    await gw.post(ask("What is HTTP?"))
    limited = await gw.post(ask(f"Mail {EMAIL}"))

    assert limited.status_code == 429
    # Refused before masking: no PII was looked for, and none reached anything.
    assert (await gw.rows())[1].pii_types == []
