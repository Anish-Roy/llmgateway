"""Loading and validating config/models.yaml."""

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.routing.registry import MockSpec, RegistryConfig, Target, load_registry


def test_shipped_registry_file_is_valid():
    config = load_registry(Settings(_env_file=None).models_file)

    assert config.aliases
    for targets in config.aliases.values():
        assert all(t.provider in config.providers for t in targets)


def test_targets_are_parsed_from_provider_slash_model():
    config = RegistryConfig.model_validate(
        {
            "providers": {"groq": {"type": "groq"}},
            "aliases": {"fast": ["groq/meta-llama/llama-4-scout"]},
        }
    )

    # Only the first slash splits: the rest is the provider's own model name.
    assert config.aliases["fast"] == [Target(provider="groq", model="meta-llama/llama-4-scout")]


def test_mock_options_are_read():
    config = RegistryConfig.model_validate(
        {"providers": {"flaky": {"type": "mock", "failure_rate": 1.0}}, "aliases": {}}
    )

    assert config.providers["flaky"] == MockSpec(type="mock", failure_rate=1.0)


@pytest.mark.parametrize(
    "data",
    [
        # target without a slash
        {"providers": {"groq": {"type": "groq"}}, "aliases": {"fast": ["groq"]}},
        # target on a provider that isn't defined
        {"providers": {"groq": {"type": "groq"}}, "aliases": {"fast": ["gemini/flash"]}},
        # alias with no targets
        {"providers": {"groq": {"type": "groq"}}, "aliases": {"fast": []}},
        # typo in a provider option
        {"providers": {"m": {"type": "mock", "failure_rte": 1}}, "aliases": {}},
        # unknown provider type
        {"providers": {"x": {"type": "openai"}}, "aliases": {}},
    ],
)
def test_invalid_registry_is_rejected(data):
    with pytest.raises(ValidationError):
        RegistryConfig.model_validate(data)


def test_providers_without_a_key_are_not_built(monkeypatch):
    import httpx

    from app.main import build_providers

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    config = RegistryConfig.model_validate(
        {"providers": {"groq": {"type": "groq"}, "mock": {"type": "mock"}}, "aliases": {}}
    )

    providers = build_providers(config, Settings(_env_file=None), httpx.AsyncClient())

    assert set(providers) == {"mock"}


def test_app_serves_the_mock_alias_from_the_shipped_registry():
    from fastapi.testclient import TestClient

    from app.main import create_app

    # Entering the context runs the lifespan, so this goes through the real registry file.
    with TestClient(create_app()) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "mock", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert resp.status_code == 200
    assert resp.headers["x-gateway-provider"] == "mock"
    assert resp.json()["model"] == "mock-model"
