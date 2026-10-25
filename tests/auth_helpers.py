from fastapi import FastAPI

from app.api.auth import get_authenticator
from app.auth.authenticator import Authenticator
from app.auth.keys import InMemoryKeyStore


def authorize(app: FastAPI, rpm_limit: int = 1_000, tpm_limit: int = 10_000_000) -> str:
    """Gives `app` an in-memory key store holding one key, and returns that key. Limits are
    high by default, so only tests about rate limits ever hit them."""
    store = InMemoryKeyStore()
    key, _ = store.add("test", rpm_limit=rpm_limit, tpm_limit=tpm_limit)
    authenticator = Authenticator(store)
    app.dependency_overrides[get_authenticator] = lambda: authenticator
    return key
