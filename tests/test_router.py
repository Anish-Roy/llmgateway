"""ModelRouter: alias -> provider/model, without going through HTTP."""

import pytest

from app.api.schemas import ChatRequest, ChatResponse
from app.providers.base import ProviderError
from app.providers.mock import MockProvider
from app.routing.registry import Target
from app.routing.router import ModelNotFoundError, ModelRouter


class RecordingProvider(MockProvider):
    """A mock that remembers the model name it was asked for."""

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.seen_model = request.model
        return await super().chat(request)


def request(model: str) -> ChatRequest:
    return ChatRequest(model=model, messages=[{"role": "user", "content": "hi"}])


async def test_alias_is_replaced_by_the_providers_model_name():
    provider = RecordingProvider()
    router = ModelRouter(
        {"fast": [Target(provider="cheap", model="llama-tiny")]}, {"cheap": provider}
    )

    routed = await router.chat(request("fast"))

    assert provider.seen_model == "llama-tiny"
    # The response reports the model that actually answered, not the alias.
    assert routed.response.model == "llama-tiny"
    assert routed.target == Target(provider="cheap", model="llama-tiny")


async def test_targets_on_unconfigured_providers_are_skipped():
    provider = MockProvider()
    router = ModelRouter(
        {
            "fast": [
                Target(provider="groq", model="llama-3.1-8b-instant"),  # no key: not built
                Target(provider="mock", model="mock-model"),
            ]
        },
        {"mock": provider},
    )

    routed = await router.chat(request("fast"))

    assert routed.target.provider == "mock"
    assert provider.calls == 1


async def test_unknown_alias_raises_model_not_found():
    router = ModelRouter({}, {})

    with pytest.raises(ModelNotFoundError):
        await router.chat(request("nope"))


async def test_alias_with_no_configured_provider_is_503():
    router = ModelRouter({"fast": [Target(provider="groq", model="llama-3.1-8b-instant")]}, {})

    with pytest.raises(ProviderError) as exc_info:
        await router.chat(request("fast"))

    assert exc_info.value.status == 503
