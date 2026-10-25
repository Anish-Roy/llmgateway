"""Turns how a chat-completions request ended into a `request_logs` row.

The route starts a `Recording` when a request gets past auth and reports exactly one ending:
a reply, a stream that closed, or an error. Building the row can't fail the request: a missing
price or a bug here only logs a warning and loses that one row.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal

from app.api.schemas import ChatRequest, Usage
from app.auth.keys import ApiKey
from app.auth.rate_limit import RateLimitExceeded
from app.cache.store import CacheEntry
from app.guardrails.pii import PiiType
from app.providers.base import ProviderError
from app.routing.auto import Decision
from app.routing.registry import Price, Target
from app.routing.router import ModelNotFoundError, Routed, RoutedStream
from app.usage.cost import cost
from app.usage.log import CacheStatus, LogOutcome, LogWriter, RequestLog

logger = logging.getLogger(__name__)

# Not a real HTTP status (no response was sent), but nginx's convention for "client closed the
# request", and the table needs one.
CLIENT_CLOSED = 499


class UsageRecorder:
    def __init__(self, writer: LogWriter, price: Callable[[Target], Price]) -> None:
        self.writer = writer
        self._price = price

    def start(self, key: ApiKey, body: ChatRequest) -> "Recording":
        return Recording(self, key.id, body.model, bool(body.stream))

    def cost(self, target: Target, usage: Usage) -> Decimal:
        """What `usage` on `target` costs at its list price."""
        return cost(self._price(target), usage)


class Recording:
    """One request, from when it got past auth to how it ended."""

    def __init__(self, recorder: UsageRecorder, key_id: int, model: str, stream: bool) -> None:
        self._recorder = recorder
        self._key_id = key_id
        self._model = model
        self._stream = stream
        self._created_at = datetime.now(UTC)
        self._started = time.monotonic()
        self._done = False
        self._pii_types: list[PiiType] = []
        self._decision: Decision | None = None

    def masked(self, types: tuple[PiiType, ...]) -> None:
        """The kinds of PII masked in the request. Called as soon as it's masked, so a request
        that then fails still says what it held."""
        self._pii_types = list(types)

    def routed(self, decision: Decision | None) -> None:
        """The alias `model: "auto"` picked, and why. None for any other request."""
        self._decision = decision

    def replied(self, routed: Routed, usage: Usage, usage_estimated: bool) -> None:
        """A non-streamed reply. `usage` is the provider's, or an estimate if it sent none."""
        self._submit(
            lambda: self._entry(
                status_code=200,
                outcome="completed",
                target=routed.target,
                usage=usage,
                usage_estimated=usage_estimated,
                attempts=routed.attempts,
                fallback=routed.fallback,
            )
        )

    def streamed(self, stream: RoutedStream) -> None:
        """A stream that was closed, however it ended. Its 200 was sent with the first chunk,
        so a failure partway shows only in the outcome."""

        def entry() -> RequestLog:
            record = stream.record
            usage = record.usage
            return self._entry(
                status_code=200,
                # None only if closing the stream itself failed.
                outcome=record.outcome or "gateway_error",
                target=stream.target,
                usage=usage or Usage(),
                usage_estimated=record.usage_estimated or usage is None,
                attempts=stream.attempts,
                fallback=stream.fallback,
                ttft_ms=None if record.ttft_s is None else round(record.ttft_s * 1000),
            )

        self._submit(entry)

    def cached(self, entry: CacheEntry, cache_status: CacheStatus = "exact") -> None:
        """An answer served from the exact or the semantic cache: no provider was called, so
        nothing was billed. The row names who first answered, and what that call cost, which
        is what this hit saved, for the dashboard."""
        self._submit(
            lambda: self._entry(
                status_code=200,
                outcome="completed",
                target=Target(provider=entry.provider, model=entry.model),
                usage=Usage(),
                attempts=0,
                cache_status=cache_status,
                saved_usd=entry.cost_usd,
            )
        )

    def failed(self, exc: BaseException) -> None:
        """An error before any reply: the gateway refused it, or no provider answered."""

        def entry() -> RequestLog:
            status, attempts = _status_and_attempts(exc)
            outcome: LogOutcome
            if isinstance(exc, asyncio.CancelledError):
                outcome = "client_disconnect"
            elif status == 500 and not isinstance(exc, ProviderError):
                outcome = "gateway_error"
            elif attempts == 0:
                outcome = "rejected"
            else:
                outcome = "provider_error"
            return self._entry(
                status_code=status, outcome=outcome, target=None, usage=Usage(), attempts=attempts
            )

        self._submit(entry)

    def _entry(
        self,
        *,
        status_code: int,
        outcome: LogOutcome,
        target: Target | None,
        usage: Usage,
        attempts: int,
        usage_estimated: bool = False,
        fallback: bool = False,
        ttft_ms: int | None = None,
        cache_status: CacheStatus = "miss",
        saved_usd: Decimal | None = None,
    ) -> RequestLog:
        return RequestLog(
            key_id=self._key_id,
            created_at=self._created_at,
            requested_model=self._model,
            provider=target.provider if target else None,
            model=target.model if target else None,
            stream=self._stream,
            status_code=status_code,
            outcome=outcome,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            usage_estimated=usage_estimated,
            latency_ms=round((time.monotonic() - self._started) * 1000),
            ttft_ms=ttft_ms,
            # Failed calls cost nothing: providers don't bill an error. Nor do cache hits, and
            # the model that first answered may since have lost its price.
            cost_usd=(
                self._recorder.cost(target, usage)
                if target and cache_status == "miss"
                else Decimal(0)
            ),
            attempts=attempts,
            fallback=fallback,
            cache_status=cache_status,
            pii_types=self._pii_types,
            routed_alias=self._decision.alias if self._decision else None,
            route_reason=self._decision.reason if self._decision else None,
            saved_usd=saved_usd,
        )

    def _submit(self, build: Callable[[], RequestLog]) -> None:
        # One row per request, even if two endings are reported (as with settling).
        if self._done:
            return
        self._done = True
        try:
            entry = build()
        except Exception:
            logger.warning(
                "could not build the request log for key %s", self._key_id, exc_info=True
            )
            return
        self._recorder.writer.submit(entry)


def _status_and_attempts(exc: BaseException) -> tuple[int, int]:
    """The status the client got for `exc` (see app/api/errors.py), and the provider calls
    made before it."""
    match exc:
        case ProviderError():
            return exc.status, exc.attempts
        case ModelNotFoundError():
            return 404, 0
        case RateLimitExceeded():
            return 429, 0
        case asyncio.CancelledError():
            return CLIENT_CLOSED, 0
        case _:
            return 500, 0
