"""Rate limits as clients see them: 429s, headers, and tokens settled on every way a request
ends. Against the mock provider, through the openai SDK where it matters."""

import time

import httpx2
import openai
import pytest

from app.api.rate_limit_headers import duration
from app.api.routes import get_model_router
from app.auth.rate_limit import RateLimiter
from app.main import create_app
from app.providers.mock import MockProvider
from app.routing.registry import Target
from app.routing.router import ModelRouter
from tests.auth_helpers import authorize

# The in-memory store's first key.
KEY_ID = 1
MESSAGES = [{"role": "user", "content": "hello there"}]


class FrozenClock:
    """Buckets never refill unless a test moves time on, so levels can be checked exactly."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_app(provider: MockProvider | None = None, *, rpm: int, tpm: int, clock=None):
    app = create_app()
    model_router = ModelRouter(
        {"mock": [Target(provider="mock", model="mock-model")]},
        {"mock": provider or MockProvider()},
    )
    app.dependency_overrides[get_model_router] = lambda: model_router
    if clock is not None:
        app.state.rate_limiter = RateLimiter(clock=clock)
    app.state.test_key = authorize(app, rpm_limit=rpm, tpm_limit=tpm)
    return app


def sdk(app, max_retries: int = 0) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        api_key=app.state.test_key,
        base_url="http://gateway.test/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app)),
        max_retries=max_retries,
    )


def token_level(app) -> float:
    return app.state.rate_limiter._tokens[KEY_ID].level


def request_level(app) -> float:
    return app.state.rate_limiter._requests[KEY_ID].level


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (0, "0s"),
        (0.0171, "18ms"),
        (0.999, "999ms"),
        (1, "1s"),
        (1.5, "1.5s"),
        (20, "20s"),
        (59.9986, "59.999s"),
        (59.9994, "1m0s"),  # rounds up, never down
        (60, "1m0s"),
        (360, "6m0s"),
        (61.25, "1m1.25s"),
    ],
)
def test_durations_are_formatted_like_openai(seconds, text):
    assert duration(seconds) == text


async def test_replies_carry_openai_rate_limit_headers():
    app = make_app(rpm=5, tpm=1000, clock=FrozenClock())

    raw = await sdk(app).chat.completions.with_raw_response.create(
        model="mock", messages=MESSAGES, max_tokens=100
    )

    h = raw.headers
    assert (h["x-ratelimit-limit-requests"], h["x-ratelimit-remaining-requests"]) == ("5", "4")
    assert h["x-ratelimit-reset-requests"] == "12s"  # one request back per 12 s
    # Reported at admission: the reservation (prompt 3 + 4, plus max_tokens 100) is held.
    assert (h["x-ratelimit-limit-tokens"], h["x-ratelimit-remaining-tokens"]) == ("1000", "893")
    assert h["x-ratelimit-reset-tokens"] == "6.42s"


async def test_streams_carry_rate_limit_headers_too():
    app = make_app(rpm=5, tpm=1000)

    async with sdk(app).chat.completions.with_streaming_response.create(
        model="mock", messages=MESSAGES, stream=True
    ) as raw:
        assert raw.headers["x-ratelimit-remaining-requests"] == "4"
        assert "x-ratelimit-remaining-tokens" in raw.headers


async def test_too_many_requests_is_a_rate_limit_error():
    app = make_app(rpm=2, tpm=100_000, clock=FrozenClock())
    client = sdk(app)
    for _ in range(2):
        await client.chat.completions.create(model="mock", messages=MESSAGES)

    with pytest.raises(openai.RateLimitError) as exc_info:
        await client.chat.completions.create(model="mock", messages=MESSAGES)

    err = exc_info.value
    assert err.status_code == 429
    assert err.code == "rate_limit_exceeded"
    assert err.type == "requests"
    assert "requests per minute (RPM): limit 2, remaining 0" in err.message
    assert "try again in 30s" in err.message
    assert app.state.test_key[:10] in err.message
    h = err.response.headers
    assert (h["retry-after"], h["retry-after-ms"]) == ("30", "30000")
    assert h["x-ratelimit-remaining-requests"] == "0"


async def test_too_many_tokens_is_a_rate_limit_error():
    app = make_app(rpm=100, tpm=600, clock=FrozenClock())
    # Another request still running holds 500 tokens (finished ones give back what they
    # didn't use, so a second request after one would fit).
    app.state.rate_limiter.acquire(_key(app), 500)

    with pytest.raises(openai.RateLimitError) as exc_info:
        await sdk(app).chat.completions.create(model="mock", messages=MESSAGES, max_tokens=500)

    assert exc_info.value.type == "tokens"
    assert "tokens per minute (TPM)" in exc_info.value.message


async def test_a_provider_429_is_told_apart_from_the_gateways():
    app = make_app(MockProvider(failure_rate=1.0, fail_status=429), rpm=100, tpm=100_000)

    with pytest.raises(openai.RateLimitError) as exc_info:
        await sdk(app).chat.completions.create(model="mock", messages=MESSAGES)

    assert exc_info.value.type not in ("requests", "tokens")
    assert "x-ratelimit-remaining-requests" not in exc_info.value.response.headers


async def test_the_sdk_waits_out_retry_after_and_gets_in():
    # 600 rpm refills one request per 0.1 s, a real wait the SDK sleeps through.
    app = make_app(rpm=600, tpm=1_000_000)
    app.state.rate_limiter._requests.clear()
    client = sdk(app, max_retries=2)
    for _ in range(600):
        app.state.rate_limiter.acquire(_key(app), 0)

    started = time.monotonic()
    resp = await client.chat.completions.create(model="mock", messages=MESSAGES)

    assert resp.choices[0].message.content == "Echo: hello there"
    assert 0.05 < time.monotonic() - started < 2


def _key(app):
    from app.api.auth import get_authenticator

    store = app.dependency_overrides[get_authenticator]().store
    return next(iter(store._keys.values()))


# Settling: the reservation is swapped for real usage, however the request ends.


async def test_a_reply_is_charged_its_real_usage():
    app = make_app(rpm=100, tpm=10_000, clock=FrozenClock())

    resp = await sdk(app).chat.completions.create(model="mock", messages=MESSAGES, max_tokens=500)

    assert token_level(app) == 10_000 - resp.usage.total_tokens


async def test_a_failed_request_gets_its_tokens_back_but_still_counts():
    app = make_app(
        MockProvider(failure_rate=1.0, fail_status=503), rpm=100, tpm=10_000, clock=FrozenClock()
    )

    with pytest.raises(openai.InternalServerError):
        await sdk(app).chat.completions.create(model="mock", messages=MESSAGES)

    assert token_level(app) == 10_000
    assert request_level(app) == 99


async def test_a_bad_request_is_rejected_before_it_counts():
    app = make_app(rpm=100, tpm=10_000, clock=FrozenClock())

    with pytest.raises(openai.BadRequestError):
        await sdk(app).chat.completions.create(model="mock", messages=[])

    # Validation runs before the handler: nothing was reserved.
    assert KEY_ID not in app.state.rate_limiter._requests


async def test_a_completed_stream_is_charged_its_real_usage():
    app = make_app(rpm=100, tpm=10_000, clock=FrozenClock())

    stream = await sdk(app).chat.completions.create(
        model="mock",
        messages=MESSAGES,
        stream=True,
        max_tokens=500,
        stream_options={"include_usage": True},
    )
    usage = [chunk.usage async for chunk in stream if chunk.usage][-1]

    assert token_level(app) == 10_000 - usage.total_tokens


async def test_a_stream_cut_mid_answer_is_charged_its_estimate():
    provider = MockProvider(reply="one two three four", fail_after_chunks=3)
    app = make_app(provider, rpm=100, tpm=10_000, clock=FrozenClock())

    stream = await sdk(app).chat.completions.create(
        model="mock", messages=MESSAGES, stream=True, max_tokens=500
    )
    with pytest.raises(openai.APIError):
        async for _ in stream:
            pass

    # Charged for what was streamed (an estimate: no provider usage), not the 500 reserved.
    used = 10_000 - token_level(app)
    assert 0 < used < 50


async def test_a_stream_that_fails_before_the_first_token_gets_its_tokens_back():
    app = make_app(
        MockProvider(failure_rate=1.0, fail_status=503), rpm=100, tpm=10_000, clock=FrozenClock()
    )

    with pytest.raises(openai.InternalServerError):
        await sdk(app).chat.completions.create(model="mock", messages=MESSAGES, stream=True)

    assert token_level(app) == 10_000
    assert request_level(app) == 99


async def test_models_are_not_rate_limited():
    app = make_app(rpm=1, tpm=1000)
    client = sdk(app)

    for _ in range(3):
        await client.models.list()
    await client.chat.completions.create(model="mock", messages=MESSAGES)
