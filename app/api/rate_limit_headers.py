"""The `x-ratelimit-*` headers, named and formatted as OpenAI sends them."""

import math

from app.auth.rate_limit import LimitStatus


def rate_limit_headers(status: LimitStatus) -> dict[str, str]:
    headers: dict[str, str] = {}
    for name, bucket in (("requests", status.requests), ("tokens", status.tokens)):
        headers[f"x-ratelimit-limit-{name}"] = str(bucket.limit)
        headers[f"x-ratelimit-remaining-{name}"] = str(bucket.remaining)
        # Time until the bucket is full again.
        headers[f"x-ratelimit-reset-{name}"] = duration(bucket.reset_s)
    return headers


def retry_after_headers(seconds: float) -> dict[str, str]:
    # The openai SDK reads retry-after-ms first; Retry-After (whole seconds, rounded up so
    # it's never too early) is the standard one other clients read.
    return {
        "retry-after-ms": str(math.ceil(seconds * 1000)),
        "Retry-After": str(math.ceil(seconds)),
    }


def duration(seconds: float) -> str:
    """OpenAI's format: "17ms", "1s", "1.5s", "6m0s". Rounded up to the millisecond."""
    ms = math.ceil(seconds * 1000 - 1e-6)
    if ms <= 0:
        return "0s"
    if ms < 1000:
        return f"{ms}ms"
    minutes, ms = divmod(ms, 60_000)
    secs = f"{ms / 1000:.3f}".rstrip("0").rstrip(".")
    return f"{minutes}m{secs}s" if minutes else f"{secs}s"
