from dataclasses import dataclass

from app.api.schemas import ChatRequest, ChatResponse
from app.providers.base import Provider, ProviderError
from app.routing.registry import Target


class ModelNotFoundError(Exception):
    """The client asked for a model alias the registry doesn't define."""

    def __init__(self, model: str) -> None:
        super().__init__(f"The model '{model}' does not exist")
        self.model = model


@dataclass
class Routed:
    response: ChatResponse
    target: Target


class ModelRouter:
    """Turns a model alias into a provider call. Providers only ever see their own model names."""

    def __init__(self, aliases: dict[str, list[Target]], providers: dict[str, Provider]) -> None:
        self._aliases = aliases
        # Only providers that could be built (e.g. had an API key). Targets on others are skipped.
        self._providers = providers

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
        target = self.targets(request.model)[0]
        provider = self._providers[target.provider]
        response = await provider.chat(request.model_copy(update={"model": target.model}))
        return Routed(response, target)
