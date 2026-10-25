import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from app.api.schemas import ChatChunk, ChatRequest, ChatResponse, Usage
from app.providers.base import Provider, ProviderError
from app.routing.registry import Target
from app.usage.tokens import estimate_usage

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    """The client asked for a model alias the registry doesn't define."""

    def __init__(self, model: str) -> None:
        super().__init__(f"The model '{model}' does not exist")
        self.model = model


@dataclass
class Routed:
    response: ChatResponse
    target: Target
    # Provider calls made, counting retries. 1 means the first try worked.
    attempts: int
    # True if the answer came from a later target, not the alias's first choice.
    fallback: bool


class ModelRouter:
    """Turns a model alias into provider calls, retrying and falling back on transient errors.

    For each target in the alias's list, in order:
    - success: return it;
    - not retryable (e.g. 400): on the first target, return it to the client, since another
      provider would reject it too. On a fallback target, or when the adapter says it's
      provider-specific ("Gemini can't do tools"), it only rules out that provider: move on;
    - 500/502/503: retry the same provider once after a short backoff, then move on;
    - 429, timeouts, rejected gateway credentials: move on without retrying.
    If every target fails, the client gets the latest retryable error, so the status tells it
    to try again later (a 429 makes the SDK back off). All of it fits within `deadline_s`.
    """

    def __init__(
        self,
        aliases: dict[str, list[Target]],
        providers: dict[str, Provider],
        *,
        deadline_s: float = 45.0,
        retries: int = 1,
        backoff_s: float = 0.25,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._aliases = aliases
        # Only providers that could be built (e.g. had an API key). Targets on others are skipped.
        self._providers = providers
        self._deadline_s = deadline_s
        self._retries = retries
        self._backoff_s = backoff_s
        # Injectable so tests don't actually wait.
        self._sleep = sleep

    def available_aliases(self) -> list[str]:
        """Aliases with at least one configured provider, for GET /v1/models."""
        return [
            alias
            for alias, targets in self._aliases.items()
            if any(t.provider in self._providers for t in targets)
        ]

    def targets(self, alias: str) -> list[Target]:
        if alias not in self._aliases:
            raise ModelNotFoundError(alias)
        usable = [t for t in self._aliases[alias] if t.provider in self._providers]
        if not usable:
            # The alias exists, but none of its providers is configured on this deployment.
            raise ProviderError(
                503, f"No provider is configured for model '{alias}'", type="server_error"
            )
        return usable

    async def chat(self, request: ChatRequest) -> Routed:
        response, target, attempts, fallback = await self._route(
            request, lambda provider, req: provider.chat(req)
        )
        return Routed(response, target, attempts, fallback)

    async def stream(self, request: ChatRequest) -> "RoutedStream":
        """Opens a stream with the same retry and fallback rules as `chat`, up to the first token.

        The router reads ahead until a chunk has something the client would see (text, a tool
        call, a finish_reason) and only then commits to that provider. Leading role-only chunks
        are held back, so a provider that fails right after one can still be replaced. Once
        the first token is out, a failure can only end the stream: the client already has part
        of an answer. The deadline covers the wait for the first token, not the whole answer.
        """
        started = time.monotonic()
        (held, rest), target, attempts, fallback = await self._route(request, _open_stream)
        return RoutedStream(held, rest, target, attempts, fallback, request, started)

    async def _route[T](
        self, request: ChatRequest, call: Callable[[Provider, ChatRequest], Awaitable[T]]
    ) -> tuple[T, Target, int, bool]:
        """Runs `call` on each target in turn, with the retry/fallback policy in the class doc.

        Returns the result, the target that produced it, the number of provider calls and
        whether that target was a fallback.
        """
        targets = self.targets(request.model)
        deadline = time.monotonic() + self._deadline_s
        attempts = 0
        errors: list[ProviderError] = []

        for i, target in enumerate(targets):
            provider = self._providers[target.provider]
            routed_request = request.model_copy(update={"model": target.model})

            for retry in range(self._retries + 1):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _final_error(errors, _deadline_error())
                attempts += 1
                try:
                    result = await asyncio.wait_for(call(provider, routed_request), remaining)
                except TimeoutError:
                    # The gateway's deadline ran out mid-call: there's no time for anything else.
                    errors.append(_deadline_error(target.provider))
                    raise _final_error(errors) from None
                except ProviderError as e:
                    err = e
                else:
                    return result, target, attempts, i > 0

                logger.warning(
                    "%s/%s failed (attempt %d): %s %s",
                    target.provider,
                    target.model,
                    attempts,
                    err.status,
                    err.message,
                )
                errors.append(err)
                if not err.retryable and i == 0 and not err.provider_specific:
                    raise err
                if not (err.retryable and err.retry_same_provider) or retry == self._retries:
                    break
                await self._sleep(self._backoff(retry, deadline))

        raise _final_error(errors)

    def _backoff(self, retry: int, deadline: float) -> float:
        # Exponential with full jitter, so retries from many clients don't line up. Never sleep
        # past the deadline: the next attempt would only fail on it.
        delay = random.uniform(0, self._backoff_s * 2**retry)
        return max(0.0, min(delay, deadline - time.monotonic()))


def _deadline_error(provider: str | None = None) -> ProviderError:
    return ProviderError(504, "Gateway deadline exceeded", type="timeout", provider=provider)


def _final_error(errors: list[ProviderError], default: ProviderError | None = None) -> Exception:
    retryable = [e for e in errors if e.retryable]
    if retryable:
        return retryable[-1]
    if errors:
        return errors[-1]
    return default or _deadline_error()


Outcome = Literal["completed", "provider_error", "gateway_error", "client_disconnect"]


@dataclass
class StreamRecord:
    """What happened on one stream, for request logs (Phase 5) and caching (Phase 6)."""

    # None while the stream is still running.
    outcome: Outcome | None = None
    # From the router getting the request to the first chunk with output going to the client:
    # what the client waits for, including retries, fallback and the read-ahead.
    ttft_s: float | None = None
    duration_s: float | None = None
    # The answer as streamed (what the client got, even if the stream was cut).
    text: str = ""
    finish_reason: str | None = None
    # The provider's count when it sent one (Gemini sends running totals, so even a cut
    # stream has one). Otherwise an estimate from the text, marked as such: a lower bound,
    # because hidden reasoning is billed but never streamed.
    usage: Usage | None = None
    usage_estimated: bool = False
    error_status: int | None = None


class RoutedStream:
    """A provider's stream after the router committed to it: iterate it for the chunks.

    Call `aclose` when done, finished or not, so the upstream connection is freed even if the
    stream was never iterated (say, the client left before the response started). `record`
    is complete once the stream ends or is closed.
    """

    def __init__(
        self,
        held: list[ChatChunk],
        rest: AsyncIterator[ChatChunk],
        target: Target,
        attempts: int,
        fallback: bool,
        request: ChatRequest,
        started: float,
    ) -> None:
        self._held = held
        self._rest = rest
        self.target = target
        self.attempts = attempts
        self.fallback = fallback
        self._messages = request.messages
        self._started = started
        self.record = StreamRecord()

    async def __aiter__(self) -> AsyncIterator[ChatChunk]:
        try:
            for chunk in self._held:
                self._observe(chunk)
                yield chunk
            async for chunk in self._rest:
                self._observe(chunk)
                yield chunk
        except ProviderError as e:
            self._finish("provider_error", e.status)
            raise
        except Exception:
            self._finish("gateway_error", 500)
            raise
        self._finish("completed")

    async def aclose(self) -> None:
        await _aclose(self._rest)
        # Closed before the end: the client went away (or never started reading).
        self._finish("client_disconnect")

    def _observe(self, chunk: ChatChunk) -> None:
        record = self.record
        if record.ttft_s is None and chunk.has_output():
            record.ttft_s = time.monotonic() - self._started
        if chunk.usage is not None:
            record.usage = chunk.usage
        for choice in chunk.choices:
            record.text += choice.delta.content or ""
            record.finish_reason = choice.finish_reason or record.finish_reason

    def _finish(self, outcome: Outcome, error_status: int | None = None) -> None:
        record = self.record
        if record.outcome is not None:
            return
        record.outcome = outcome
        record.error_status = error_status
        record.duration_s = time.monotonic() - self._started
        if record.usage is None:
            record.usage = estimate_usage(self._messages, record.text)
            record.usage_estimated = True
        logger.info(
            "stream %s/%s %s: ttft %s, %.2fs, %d+%d tokens%s",
            self.target.provider,
            self.target.model,
            outcome if error_status is None else f"{outcome} ({error_status})",
            "-" if record.ttft_s is None else f"{record.ttft_s:.2f}s",
            record.duration_s,
            record.usage.prompt_tokens,
            record.usage.completion_tokens,
            " (estimated)" if record.usage_estimated else "",
        )


async def _open_stream(
    provider: Provider, request: ChatRequest
) -> tuple[list[ChatChunk], AsyncIterator[ChatChunk]]:
    """Starts a stream and reads it up to the first chunk with output.

    Returns the chunks read so far and the open iterator for the rest. On any failure,
    including the router's deadline cancelling it, the stream is closed before re-raising.
    """
    chunks = aiter(provider.stream(request))
    held: list[ChatChunk] = []
    try:
        async for chunk in chunks:
            held.append(chunk)
            if chunk.has_output():
                return held, chunks
    except BaseException:
        await _aclose(chunks)
        raise
    # A 200 that ended without saying anything: as broken as an error, and another provider
    # may do better.
    raise ProviderError(
        502, "Provider ended the stream without sending a reply", provider=provider.name
    )


async def _aclose(chunks: AsyncIterator[ChatChunk]) -> None:
    # Async generators have aclose(); a plain iterator has nothing to free.
    aclose = getattr(chunks, "aclose", None)
    if aclose is not None:
        await aclose()
