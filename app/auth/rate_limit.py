"""Per-key rate limits: requests per minute and tokens per minute, as token buckets in memory.

Each key has two buckets. One holds up to `rpm_limit` requests, the other up to `tpm_limit`
tokens, and both refill to full over a minute, continuously. A full bucket allows a burst of a
whole minute's worth at once.

A request's tokens aren't known until it's done, so admission reserves an estimate (prompt plus
the most it may generate) and `Reservation.settle` corrects it with the real usage afterwards:
unused tokens go back, extra ones are charged. Extra charges can take the bucket below zero, so
later requests wait. That debt is capped at one minute's worth (`-tpm_limit`), which bounds both
sides: a key can never go more than 2x over its TPM in any window, and no key waits more than
two minutes for a full reservation, which is also the longest `Retry-After` the openai SDK will
wait for before retrying.

Buckets live in this process only, so the gateway must run with one worker; a restart gives
every key a full bucket. Nothing here awaits, so on the event loop each call runs whole: no lock.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from app.api.schemas import ChatRequest
from app.auth.keys import ApiKey
from app.usage.tokens import count_tokens

# Floating-point slack, so waiting exactly the advertised Retry-After always succeeds.
_EPSILON = 1e-9

# Reserved for the reply when the client sets no max_tokens. Settling corrects it either way.
DEFAULT_COMPLETION_TOKENS = 1024
# Roughly what each message adds beyond its text (role, separators), as in OpenAI's counting.
_TOKENS_PER_MESSAGE = 4

Limit = Literal["requests", "tokens"]


@dataclass(frozen=True)
class BucketStatus:
    """One bucket, as the `x-ratelimit-*` headers report it."""

    limit: int
    remaining: int
    # Seconds until the bucket is full again.
    reset_s: float


@dataclass(frozen=True)
class LimitStatus:
    requests: BucketStatus
    tokens: BucketStatus


class RateLimitExceeded(Exception):
    def __init__(self, retry_after_s: float, limited: list[Limit], status: LimitStatus) -> None:
        super().__init__(f"Rate limit reached for {' and '.join(limited)}")
        # How long until the same request would be admitted.
        self.retry_after_s = retry_after_s
        self.limited = limited
        self.status = status


class TokenBucket:
    """Holds up to `capacity`, refills `capacity` per minute, and may go down to `-capacity`."""

    def __init__(self, capacity: int, now: float) -> None:
        self.capacity = capacity
        self.level = float(capacity)
        self.updated = now

    @property
    def rate(self) -> float:
        return self.capacity / 60

    def refill(self, now: float) -> None:
        self.level = min(self.capacity, self.level + (now - self.updated) * self.rate)
        self.updated = now

    def resize(self, capacity: int, now: float) -> None:
        """A key's limit changed: refill at the old rate up to now, then switch."""
        self.refill(now)
        self.capacity = capacity
        self.level = max(-capacity, min(capacity, self.level))

    def wait_for(self, amount: float) -> float:
        """Seconds until the bucket holds `amount`; 0 if it already does. Refill first."""
        if self.level + _EPSILON >= amount:
            return 0.0
        return (amount - self.level) / self.rate

    def adjust(self, delta: float) -> None:
        self.level = max(-self.capacity, min(self.capacity, self.level + delta))

    def status(self) -> BucketStatus:
        return BucketStatus(
            limit=self.capacity,
            remaining=max(0, math.floor(self.level + _EPSILON)),
            reset_s=max(0.0, (self.capacity - self.level) / self.rate),
        )


class Reservation:
    """An admitted request's hold on its key's tokens. Settle it once, when the request ends."""

    def __init__(
        self, bucket: TokenBucket, tokens: int, status: LimitStatus, clock: Callable[[], float]
    ) -> None:
        self._bucket = bucket
        self._clock = clock
        self.tokens = tokens
        # The buckets right after admission, for the response headers.
        self.status = status
        self.settled = False

    def settle(self, actual_tokens: int) -> None:
        """Swaps the reservation for what the request really used: 0 for a request that failed
        before any provider answered. Only the first call counts, so every way a request can
        end may call it."""
        if self.settled:
            return
        self.settled = True
        self._bucket.refill(self._clock())
        self._bucket.adjust(self.tokens - actual_tokens)


class RateLimiter:
    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        default_completion_tokens: int = DEFAULT_COMPLETION_TOKENS,
    ) -> None:
        self.clock = clock
        self.default_completion_tokens = default_completion_tokens
        # Key id -> bucket. One entry per key that has made a request: no eviction needed.
        self._requests: dict[int, TokenBucket] = {}
        self._tokens: dict[int, TokenBucket] = {}

    def reserve(self, key: ApiKey, request: ChatRequest) -> Reservation:
        """Admits a chat request, reserving the most it may use (see `estimate_tokens`)."""
        return self.acquire(key, estimate_tokens(request, self.default_completion_tokens))

    def acquire(self, key: ApiKey, tokens: int) -> Reservation:
        """Admits a request that may use `tokens`, or raises RateLimitExceeded.

        All or nothing: a rejected request takes nothing from either bucket, so a client
        retrying on 429 doesn't push its own wait further out. A reservation larger than the
        whole TPM limit is cut to the limit: it's admitted once the bucket is full, instead
        of never.
        """
        now = self.clock()
        requests = self._bucket(self._requests, key.id, key.rpm_limit, now)
        bucket = self._bucket(self._tokens, key.id, key.tpm_limit, now)
        reserve = min(tokens, key.tpm_limit)

        waits: dict[Limit, float] = {
            "requests": requests.wait_for(1),
            "tokens": bucket.wait_for(reserve),
        }
        limited = [name for name, wait in waits.items() if wait > 0]
        if limited:
            status = LimitStatus(requests.status(), bucket.status())
            raise RateLimitExceeded(max(waits.values()), limited, status)

        requests.adjust(-1)
        bucket.adjust(-reserve)
        return Reservation(
            bucket, reserve, LimitStatus(requests.status(), bucket.status()), self.clock
        )

    @staticmethod
    def _bucket(
        buckets: dict[int, TokenBucket], key_id: int, limit: int, now: float
    ) -> TokenBucket:
        bucket = buckets.get(key_id)
        if bucket is None:
            bucket = buckets[key_id] = TokenBucket(limit, now)
        elif bucket.capacity != limit:
            # The key's limit was changed (seen once the auth cache refreshes).
            bucket.resize(limit, now)
        else:
            bucket.refill(now)
        return bucket


def estimate_tokens(
    request: ChatRequest, default_completion: int = DEFAULT_COMPLETION_TOKENS
) -> int:
    """The most a request may use: its prompt plus the longest reply it allows, for each of
    its `n` choices. An estimate to reserve, not to bill."""
    prompt = sum(count_tokens(m.content or "") + _TOKENS_PER_MESSAGE for m in request.messages)
    extra = request.model_extra or {}
    # max_completion_tokens and n aren't validated by the gateway (the provider does that), so
    # anything but a positive integer counts as absent here and the provider gives the 400.
    completion = (
        _positive_int(extra.get("max_completion_tokens"))
        or request.max_tokens
        or default_completion
    )
    return prompt + completion * (_positive_int(extra.get("n")) or 1)


def _positive_int(value: object) -> int | None:
    # bool is an int in Python: `"n": true` is not 1.
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None
