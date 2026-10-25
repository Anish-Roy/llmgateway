"""The model registry: which providers exist, which provider/model pairs serve each alias, and
what each pair costs."""

from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator


class GroqSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["groq"]
    # For reasoning models (gpt-oss). Reasoning counts against max_tokens, and
    # include_reasoning=false keeps the reasoning text out of replies (like Gemini's thoughts).
    # A client that sends either field overrides these.
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    include_reasoning: bool | None = None


class GeminiSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["gemini"]
    # Thinking is on by default for most Gemini models and counts against max_tokens.
    # None keeps the model's default; levels vary by model.
    thinking_level: Literal["minimal", "low", "medium", "high"] | None = None


class MockSpec(BaseModel):
    # extra="forbid" everywhere in this file: a typo like `failure_rte` should fail at startup,
    # not be silently ignored.
    model_config = ConfigDict(extra="forbid")

    type: Literal["mock"]
    reply: str | None = None
    latency_s: float = Field(default=0.0, ge=0)
    failure_rate: float = Field(default=0.0, ge=0, le=1)
    fail_status: int = 503
    # Streaming only: the wait between chunks, and a failure after this many chunks.
    chunk_latency_s: float = Field(default=0.0, ge=0)
    fail_after_chunks: int | None = Field(default=None, ge=0)


ProviderSpec = Annotated[GroqSpec | GeminiSpec | MockSpec, Field(discriminator="type")]


class Target(BaseModel):
    """One provider/model pair that can serve an alias."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str


def _parse_target(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    # Split on the first slash only: model names can contain slashes (meta-llama/llama-4-...).
    provider, sep, model = value.partition("/")
    if not (provider and sep and model):
        raise ValueError(f"expected 'provider/model', got {value!r}")
    return {"provider": provider, "model": model}


TargetRef = Annotated[Target, BeforeValidator(_parse_target)]


class Price(BaseModel):
    """List price in USD per 1M tokens. Decimal, so 0.075 stays exactly 0.075."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input: Decimal = Field(ge=0)
    # Includes reasoning and thinking tokens: providers bill them as output.
    output: Decimal = Field(ge=0)


FREE = Price(input=Decimal(0), output=Decimal(0))


class RegistryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    providers: dict[str, ProviderSpec]
    aliases: dict[str, Annotated[list[TargetRef], Field(min_length=1)]]
    # "provider/model" -> price, keyed like alias targets. Mock targets may leave theirs out.
    pricing: dict[str, Price] = {}

    @model_validator(mode="after")
    def _targets_name_known_providers(self) -> "RegistryConfig":
        for alias, targets in self.aliases.items():
            for t in targets:
                if t.provider not in self.providers:
                    raise ValueError(f"alias {alias!r} uses undefined provider {t.provider!r}")
        return self

    @model_validator(mode="after")
    def _every_paid_target_has_a_price(self) -> "RegistryConfig":
        # A missing price would log every call to that model as free, and nobody would notice.
        for key in self.pricing:
            provider = _parse_target(key)["provider"]
            if provider not in self.providers:
                raise ValueError(f"pricing entry {key!r} uses undefined provider {provider!r}")
        for alias, targets in self.aliases.items():
            for t in targets:
                if f"{t.provider}/{t.model}" not in self.pricing and not self._is_mock(t):
                    raise ValueError(
                        f"alias {alias!r} uses {t.provider}/{t.model}, which has no price"
                    )
        return self

    def price(self, target: Target) -> Price:
        """What `target` costs. Mock targets without an entry are free."""
        price = self.pricing.get(f"{target.provider}/{target.model}")
        if price is not None:
            return price
        if self._is_mock(target):
            return FREE
        raise KeyError(f"no price for {target.provider}/{target.model}")

    def _is_mock(self, target: Target) -> bool:
        return isinstance(self.providers[target.provider], MockSpec)


def load_registry(path: Path) -> RegistryConfig:
    with path.open() as f:
        return RegistryConfig.model_validate(yaml.safe_load(f))
