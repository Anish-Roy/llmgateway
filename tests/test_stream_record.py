"""StreamRecord: what the gateway notes about each stream (TTFT, text, usage, how it ended)."""

import asyncio

import httpx

from app.api.schemas import ChatMessage, ChatRequest
from app.providers.base import ProviderError
from app.providers.gemini import GeminiProvider
from app.providers.mock import MockProvider
from app.routing.registry import Target
from app.routing.router import ModelRouter
from app.usage import tokens
from tests.test_gemini_stream import GEMINI_STREAM, sse, streaming

REQUEST = ChatRequest(
    model="fast", messages=[{"role": "user", "content": "hello there"}], stream=True
)


def router_for(*providers) -> ModelRouter:
    return ModelRouter(
        {"fast": [Target(provider=name, model=f"{name}-model") for name, _ in providers]},
        dict(providers),
        sleep=lambda _: asyncio.sleep(0),
    )


async def drain(stream, limit: int | None = None) -> None:
    """Reads the stream like the SSE layer does, then closes it like the response does."""
    try:
        seen = 0
        async for _ in stream:
            seen += 1
            if seen == limit:
                break
    except ProviderError:
        pass
    finally:
        await stream.aclose()


async def test_completed_stream():
    stream = await router_for(("mock", MockProvider(reply="one two"))).stream(REQUEST)
    await drain(stream)

    record = stream.record
    assert record.outcome == "completed"
    assert record.text == "one two"
    assert record.finish_reason == "stop"
    # The provider's own count, not an estimate.
    assert not record.usage_estimated
    assert record.usage.completion_tokens == len("one two") // 4 + 1
    assert 0 <= record.ttft_s <= record.duration_s
    assert record.error_status is None


async def test_mid_stream_failure_keeps_the_partial_answer_and_estimates_usage():
    provider = MockProvider(reply="one two three", fail_after_chunks=3)
    stream = await router_for(("mock", provider)).stream(REQUEST)
    await drain(stream)

    record = stream.record
    assert (record.outcome, record.error_status) == ("provider_error", 503)
    assert record.text == "one two "
    # The mock (like Groq) reports usage only at the end, so it's estimated from the text.
    assert record.usage_estimated
    assert record.usage.completion_tokens == tokens.count_tokens("one two ")
    assert record.usage.prompt_tokens == tokens.count_tokens("hello there")


async def test_client_disconnect():
    stream = await router_for(("mock", MockProvider(reply="one two three four"))).stream(REQUEST)
    await drain(stream, limit=2)

    assert stream.record.outcome == "client_disconnect"
    assert stream.record.text == "one "
    assert stream.record.usage_estimated


async def test_stream_closed_before_it_was_read():
    stream = await router_for(("mock", MockProvider())).stream(REQUEST)
    await stream.aclose()

    assert stream.record.outcome == "client_disconnect"
    assert stream.record.ttft_s is None
    assert stream.record.text == ""


async def test_ttft_includes_the_time_spent_on_a_failed_provider():
    flaky = MockProvider(latency_s=0.05, failure_rate=1.0, fail_status=429)
    stream = await router_for(("flaky", flaky), ("mock", MockProvider())).stream(REQUEST)
    await drain(stream)

    assert stream.fallback
    assert stream.record.ttft_s >= 0.05


async def test_cut_gemini_stream_keeps_geminis_running_usage():
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(streaming(sse(*GEMINI_STREAM[:2]))))
    stream = await router_for(("gemini", GeminiProvider(upstream, "AIza-test"))).stream(REQUEST)
    await drain(stream)

    record = stream.record
    assert (record.outcome, record.error_status) == ("provider_error", 502)
    assert record.text == "Amazon\nNile"
    # Gemini's own total from the last event (6 answer + 30 thinking tokens), not an estimate.
    assert not record.usage_estimated
    assert record.usage.completion_tokens == 36


def test_count_tokens_falls_back_to_chars_over_4(monkeypatch):
    monkeypatch.setattr(tokens, "_encoding", None)

    assert tokens.count_tokens("") == 0
    assert tokens.count_tokens("abcde") == 2


def test_count_tokens_uses_the_loaded_encoding(monkeypatch):
    class FakeEncoding:
        def encode(self, text, disallowed_special):
            assert disallowed_special == ()  # "<|endoftext|>" in text mustn't raise
            return text.split()

    monkeypatch.setattr(tokens, "_encoding", FakeEncoding())

    usage = tokens.estimate_usage(
        [ChatMessage(role="system", content="be brief"), ChatMessage(role="user", content="hi")],
        "a b c",
    )
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (3, 3, 6)


def test_failed_encoding_load_keeps_the_fallback(monkeypatch):
    def offline(name):
        raise OSError("no network")

    monkeypatch.setattr(tokens, "_encoding", None)
    monkeypatch.setattr(tokens.tiktoken, "get_encoding", offline)

    tokens.load_encoding()

    assert tokens._encoding is None
    assert tokens.count_tokens("abcdefgh") == 2
