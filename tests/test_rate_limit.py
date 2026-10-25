"""The per-key token buckets, on a fake clock: nothing here waits."""

import dataclasses
from datetime import datetime

import pytest

from app.api.schemas import ChatRequest
from app.auth.keys import ApiKey
from app.auth.rate_limit import (
    DEFAULT_COMPLETION_TOKENS,
    RateLimiter,
    RateLimitExceeded,
    TokenBucket,
    estimate_tokens,
)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_key(id: int = 1, rpm: int = 60, tpm: int = 6000) -> ApiKey:
    return ApiKey(
        id=id,
        name=f"key{id}",
        key_prefix="gw-test123",
        rpm_limit=rpm,
        tpm_limit=tpm,
        active=True,
        created_at=datetime.now().astimezone(),
    )


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def limiter(clock):
    return RateLimiter(clock)


# The bucket itself.


def test_bucket_refills_at_its_limit_per_minute_up_to_full(clock):
    bucket = TokenBucket(60, clock.now)
    bucket.adjust(-60)

    clock.now += 10
    bucket.refill(clock.now)
    assert bucket.level == pytest.approx(10)

    clock.now += 3600
    bucket.refill(clock.now)
    assert bucket.level == 60


def test_bucket_debt_stops_at_minus_one_minute(clock):
    bucket = TokenBucket(100, clock.now)

    bucket.adjust(-1000)

    assert bucket.level == -100
    assert bucket.status().remaining == 0


def test_bucket_status_reports_time_until_full(clock):
    bucket = TokenBucket(60, clock.now)
    bucket.adjust(-30)

    status = bucket.status()

    assert (status.limit, status.remaining) == (60, 30)
    assert status.reset_s == pytest.approx(30)


# Requests per minute.


def test_rpm_allows_a_full_burst_then_limits(limiter):
    key = make_key(rpm=3)

    for _ in range(3):
        limiter.acquire(key, 10)

    with pytest.raises(RateLimitExceeded) as exc_info:
        limiter.acquire(key, 10)
    assert exc_info.value.limited == ["requests"]
    assert exc_info.value.retry_after_s == pytest.approx(20)  # one request per 20 s


def test_waiting_exactly_retry_after_is_enough(limiter, clock):
    key = make_key(rpm=7, tpm=777)
    for _ in range(7):
        limiter.acquire(key, 1)

    with pytest.raises(RateLimitExceeded) as exc_info:
        limiter.acquire(key, 1)
    wait = exc_info.value.retry_after_s

    clock.now += wait * 0.99
    with pytest.raises(RateLimitExceeded):
        limiter.acquire(key, 1)

    clock.now += wait * 0.01
    limiter.acquire(key, 1)


def test_rejected_requests_take_nothing(limiter, clock):
    key = make_key(rpm=1)
    limiter.acquire(key, 10)

    # Retrying hard while limited doesn't push the wait further out.
    for _ in range(50):
        with pytest.raises(RateLimitExceeded):
            limiter.acquire(key, 10)

    clock.now += 60
    limiter.acquire(key, 10)


# Tokens per minute.


def test_tpm_limits_by_reserved_tokens(limiter):
    key = make_key(tpm=1000)
    limiter.acquire(key, 600)

    with pytest.raises(RateLimitExceeded) as exc_info:
        limiter.acquire(key, 600)

    assert exc_info.value.limited == ["tokens"]
    # 200 tokens short, at 1000/60 per second.
    assert exc_info.value.retry_after_s == pytest.approx(12)


def test_admission_is_all_or_nothing(limiter):
    key = make_key(rpm=10, tpm=1000)
    limiter.acquire(key, 900)

    # Tokens are short, so the request slot isn't taken either.
    with pytest.raises(RateLimitExceeded):
        limiter.acquire(key, 500)
    reservation = limiter.acquire(key, 100)

    assert reservation.status.requests.remaining == 8


def test_both_limits_reported_with_the_longer_wait(limiter):
    key = make_key(rpm=1, tpm=600)
    limiter.acquire(key, 600)

    with pytest.raises(RateLimitExceeded) as exc_info:
        limiter.acquire(key, 600)

    assert exc_info.value.limited == ["requests", "tokens"]
    assert exc_info.value.retry_after_s == pytest.approx(60)


def test_a_reservation_bigger_than_the_limit_is_cut_to_it(limiter, clock):
    key = make_key(tpm=1000)

    # Would never fit; with a full bucket it's admitted, holding the whole bucket.
    reservation = limiter.acquire(key, 50_000)
    assert reservation.tokens == 1000

    with pytest.raises(RateLimitExceeded) as exc_info:
        limiter.acquire(key, 50_000)
    clock.now += exc_info.value.retry_after_s
    limiter.acquire(key, 50_000)


def test_settle_returns_unused_tokens(limiter):
    key = make_key(tpm=1000)
    reservation = limiter.acquire(key, 800)

    reservation.settle(100)

    assert limiter.acquire(key, 800).status.tokens.remaining == 100


def test_settle_charges_extra_tokens_as_debt(limiter, clock):
    key = make_key(tpm=600)
    reservation = limiter.acquire(key, 100)

    reservation.settle(900)  # level: 600 - 900 = -300

    with pytest.raises(RateLimitExceeded) as exc_info:
        limiter.acquire(key, 100)
    # 400 short at 10 per second.
    assert exc_info.value.retry_after_s == pytest.approx(40)
    assert exc_info.value.status.tokens.remaining == 0


def test_debt_never_means_more_than_two_minutes(limiter):
    key = make_key(tpm=500)
    limiter.acquire(key, 500).settle(1_000_000)

    with pytest.raises(RateLimitExceeded) as exc_info:
        limiter.acquire(key, 500)

    assert exc_info.value.retry_after_s == pytest.approx(120)


def test_a_failed_request_gets_its_tokens_back_but_not_its_request(limiter):
    key = make_key(rpm=10, tpm=1000)
    limiter.acquire(key, 1000).settle(0)

    reservation = limiter.acquire(key, 1000)

    assert reservation.status.requests.remaining == 8
    assert reservation.status.tokens.remaining == 0


def test_only_the_first_settle_counts(limiter):
    key = make_key(tpm=1000)
    reservation = limiter.acquire(key, 500)

    reservation.settle(100)
    reservation.settle(0)
    reservation.settle(5000)

    assert reservation.settled
    assert limiter.acquire(key, 100).status.tokens.remaining == 800


def test_refunds_never_overfill_the_bucket(limiter, clock):
    key = make_key(tpm=1000)
    reservation = limiter.acquire(key, 500)
    clock.now += 60  # refilled to full while the request ran

    reservation.settle(0)

    assert limiter.acquire(key, 1).status.tokens.remaining == 999


# Keys and limit changes.


def test_keys_have_separate_buckets(limiter):
    alice, bob = make_key(1, rpm=1), make_key(2, rpm=1)
    limiter.acquire(alice, 1)

    limiter.acquire(bob, 1)
    with pytest.raises(RateLimitExceeded):
        limiter.acquire(alice, 1)


def test_a_new_key_starts_full(limiter):
    status = limiter.acquire(make_key(rpm=5, tpm=500), 100).status

    assert (status.requests.limit, status.requests.remaining) == (5, 4)
    assert (status.tokens.limit, status.tokens.remaining) == (500, 400)


def test_a_lowered_limit_applies_to_the_next_request(limiter):
    key = make_key(rpm=100)
    limiter.acquire(key, 1)

    lowered = dataclasses.replace(key, rpm_limit=2)
    status = limiter.acquire(lowered, 1).status

    assert status.requests.limit == 2
    assert status.requests.remaining == 1  # clamped to 2, then one taken


def test_a_raised_limit_refills_at_the_new_rate(limiter, clock):
    key = make_key(rpm=1)
    limiter.acquire(key, 1)

    raised = dataclasses.replace(key, rpm_limit=60)
    # The empty bucket stays empty (no free burst), but refills at 1 per second now.
    with pytest.raises(RateLimitExceeded) as exc_info:
        limiter.acquire(raised, 1)
    assert exc_info.value.retry_after_s == pytest.approx(1)

    clock.now += 1
    limiter.acquire(raised, 1)


# The estimate to reserve.


def chat(**fields) -> ChatRequest:
    return ChatRequest.model_validate(
        {"model": "fast", "messages": [{"role": "user", "content": "hello"}], **fields}
    )


def test_estimate_is_prompt_plus_max_tokens():
    # "hello" is 2 tokens at chars/4, plus 4 for the message.
    assert estimate_tokens(chat(max_tokens=100)) == 6 + 100


def test_estimate_uses_a_default_without_max_tokens():
    assert estimate_tokens(chat()) == 6 + DEFAULT_COMPLETION_TOKENS
    assert estimate_tokens(chat(), default_completion=50) == 6 + 50


def test_estimate_prefers_max_completion_tokens():
    assert estimate_tokens(chat(max_tokens=100, max_completion_tokens=30)) == 6 + 30


def test_estimate_counts_every_choice():
    assert estimate_tokens(chat(max_tokens=100, n=3)) == 6 + 300


@pytest.mark.parametrize(
    "fields",
    [
        {"max_completion_tokens": "lots"},
        {"max_completion_tokens": -5},
        {"max_completion_tokens": True},
        {"n": "two"},
        {"n": 0},
        {"n": True},
        {"n": 2.5},
    ],
)
def test_estimate_ignores_values_the_provider_will_reject(fields):
    # The provider gives the honest 400; the estimate just mustn't crash.
    assert estimate_tokens(chat(max_tokens=100, **fields)) == 6 + 100


def test_estimate_counts_every_message():
    request = ChatRequest.model_validate(
        {
            "model": "fast",
            "max_tokens": 10,
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": None},
            ],
        }
    )

    assert estimate_tokens(request) == (3 + 4) + (2 + 4) + (0 + 4) + 10
