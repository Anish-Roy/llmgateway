"""The model registry: which providers exist, and which provider/model pairs serve each alias."""

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator


class GroqSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["groq"]


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


class RegistryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    providers: dict[str, ProviderSpec]
    aliases: dict[str, Annotated[list[TargetRef], Field(min_length=1)]]

    @model_validator(mode="after")
    def _targets_name_known_providers(self) -> "RegistryConfig":
        for alias, targets in self.aliases.items():
            for t in targets:
                if t.provider not in self.providers:
                    raise ValueError(f"alias {alias!r} uses undefined provider {t.provider!r}")
        return self


def load_registry(path: Path) -> RegistryConfig:
    with path.open() as f:
        return RegistryConfig.model_validate(yaml.safe_load(f))
