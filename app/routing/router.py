import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.api.schemas import ChatRequest, ChatResponse
from app.providers.base import Provider, ProviderError
from app.routing.registry import Target

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
      provider would reject it too. On a fallback target it only means *that* provider can't
      do it, so move on;
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
                    response = await asyncio.wait_for(provider.chat(routed_request), remaining)
                except TimeoutError:
                    # The gateway's deadline ran out mid-call: there's no time for anything else.
                    errors.append(_deadline_error(target.provider))
                    raise _final_error(errors) from None
                except ProviderError as e:
                    err = e
                else:
                    return Routed(response, target, attempts, fallback=i > 0)

                logger.warning(
                    "%s/%s failed (attempt %d): %s %s",
                    target.provider,
                    target.model,
                    attempts,
                    err.status,
                    err.message,
                )
                errors.append(err)
                if not err.retryable and i == 0:
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
