"""Check that every model in config/models.yaml is one its provider currently serves.

Calls each provider's model-list endpoint with the keys from .env. Not for CI.

uv run python -m scripts.check_models
"""

import sys

import httpx

from app.config import get_settings
from app.providers.gemini import GEMINI_BASE_URL
from app.providers.groq import GROQ_BASE_URL
from app.routing.registry import GeminiSpec, GroqSpec, load_registry


def groq_models(client: httpx.Client, key: str) -> set[str]:
    resp = client.get(f"{GROQ_BASE_URL}/models", headers={"Authorization": f"Bearer {key}"})
    resp.raise_for_status()
    return {m["id"] for m in resp.json()["data"]}


def gemini_models(client: httpx.Client, key: str) -> set[str]:
    # Paged; names come back as "models/gemini-...". Keep only ones that can generateContent.
    models: set[str] = set()
    params: dict[str, str] = {"pageSize": "1000"}
    while True:
        resp = client.get(
            f"{GEMINI_BASE_URL}/models", headers={"x-goog-api-key": key}, params=params
        )
        resp.raise_for_status()
        data = resp.json()
        for m in data.get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                models.add(m["name"].removeprefix("models/"))
        if not data.get("nextPageToken"):
            return models
        params["pageToken"] = data["nextPageToken"]


def main() -> int:
    settings = get_settings()
    config = load_registry(settings.models_file)
    keys = {
        "groq": settings.groq_api_key,
        "gemini": settings.gemini_api_key,
    }
    listers = {"groq": groq_models, "gemini": gemini_models}

    available: dict[str, set[str]] = {}
    with httpx.Client(timeout=30) as client:
        for name, spec in config.providers.items():
            if not isinstance(spec, GroqSpec | GeminiSpec):
                continue
            key = keys[spec.type]
            if not key:
                print(f"{name}: no {spec.type.upper()}_API_KEY, skipped")
                continue
            try:
                available[name] = listers[spec.type](client, key.get_secret_value())
            except httpx.HTTPStatusError as e:
                print(f"{name}: listing models failed with {e.response.status_code}")
                return 1

    missing = 0
    for alias, targets in config.aliases.items():
        for t in targets:
            if t.provider not in available:
                continue
            ok = t.model in available[t.provider]
            missing += not ok
            print(f"{'ok     ' if ok else 'MISSING'}  {alias:14} {t.provider}/{t.model}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
