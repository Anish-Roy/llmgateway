"""Loading and validating config/models.yaml."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.routing.registry import FREE, MockSpec, Price, RegistryConfig, Target, load_registry
from tests.auth_helpers import authorize


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
            "pricing": {"groq/meta-llama/llama-4-scout": {"input": 0.11, "output": 0.34}},
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
        # a paid target with no price: every call to it would be logged as free
        {"providers": {"groq": {"type": "groq"}}, "aliases": {"fast": ["groq/gpt-oss"]}},
        # a price for a provider that isn't defined
        {"providers": {}, "aliases": {}, "pricing": {"groq/gpt-oss": {"input": 1, "output": 1}}},
        # a negative price
        {
            "providers": {"groq": {"type": "groq"}},
            "aliases": {"fast": ["groq/gpt-oss"]},
            "pricing": {"groq/gpt-oss": {"input": -1, "output": 1}},
        },
        # typo in a price field
        {
            "providers": {"groq": {"type": "groq"}},
            "aliases": {"fast": ["groq/gpt-oss"]},
            "pricing": {"groq/gpt-oss": {"inptu": 1, "output": 1}},
        },
    ],
)
def test_invalid_registry_is_rejected(data):
    with pytest.raises(ValidationError):
        RegistryConfig.model_validate(data)


def test_shipped_registry_prices_every_target():
    config = load_registry(Settings(_env_file=None).models_file)

    for targets in config.aliases.values():
        for t in targets:
            config.price(t)  # raises if a paid target has no price
    # A YAML float becomes the exact decimal it's written as, not 0.07499999...
    assert config.price(Target(provider="groq", model="openai/gpt-oss-20b")) == Price(
        input=Decimal("0.075"), output=Decimal("0.30")
    )


def test_mock_targets_are_free_unless_priced():
    config = RegistryConfig.model_validate(
        {
            "providers": {"mock": {"type": "mock"}, "pricey": {"type": "mock"}},
            "aliases": {"m": ["mock/mock-model", "pricey/mock-model"]},
            "pricing": {"pricey/mock-model": {"input": 1, "output": 2}},
        }
    )

    assert config.price(Target(provider="mock", model="mock-model")) == FREE
    assert config.price(Target(provider="pricey", model="mock-model")) == Price(input=1, output=2)


def test_price_of_an_unpriced_paid_target_is_an_error():
    config = RegistryConfig.model_validate({"providers": {"groq": {"type": "groq"}}, "aliases": {}})

    with pytest.raises(KeyError):
        config.price(Target(provider="groq", model="gpt-oss"))


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
    app = create_app()
    key = authorize(app)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "mock", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {key}"},
        )

    assert resp.status_code == 200
    assert resp.headers["x-gateway-provider"] == "mock"
    assert resp.json()["model"] == "mock-model"


def test_demo_fallback_alias_falls_back_to_mock():
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app()
    key = authorize(app)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "demo-fallback", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {key}"},
        )

    assert resp.status_code == 200
    assert resp.headers["x-gateway-provider"] == "mock"
    assert resp.headers["x-gateway-fallback"] == "true"
