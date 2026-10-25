"""API key checks: the authenticator and its cache, and what clients see."""

import httpx2
import openai
import psycopg
import pytest
from fastapi.testclient import TestClient

import app.main
from app.api.auth import get_authenticator
from app.api.routes import get_model_router
from app.auth.admin import create_key, revoke_key
from app.auth.authenticator import Authenticator, AuthError, KeyStoreUnavailable
from app.auth.keys import InMemoryKeyStore, generate_key
from app.config import Settings
from app.db.migrate import apply_migrations
from app.db.pool import open_pool
from app.main import create_app
from app.providers.mock import MockProvider
from app.routing.registry import Target
from app.routing.router import ModelRouter


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class CountingStore(InMemoryKeyStore):
    """Counts lookups, so tests can see what the cache saved. `fail` makes lookups raise as
    if Postgres were down."""

    def __init__(self) -> None:
        super().__init__()
        self.lookups = 0
        self.fail = False

    async def get_by_hash(self, key_hash):
        self.lookups += 1
        if self.fail:
            raise psycopg.OperationalError("connection refused")
        return await super().get_by_hash(key_hash)


@pytest.fixture
def store():
    return CountingStore()


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def auth(store, clock):
    return Authenticator(store, ttl_s=30, clock=clock)


async def test_a_valid_key_returns_its_record(auth, store):
    key, record = store.add("alice")

    assert await auth.authenticate(f"Bearer {key}") == record
    # The scheme is case-insensitive, and stray spaces don't matter.
    assert await auth.authenticate(f"  bearer   {key} ") == record


@pytest.mark.parametrize(
    ("header", "message"),
    [
        (None, "didn't provide an API key"),
        ("", "didn't provide an API key"),
        ("Basic dXNlcjpwYXNz", "must be 'Bearer YOUR_KEY'"),
        ("Bearer", "must be 'Bearer YOUR_KEY'"),
        ("Bearer   ", "must be 'Bearer YOUR_KEY'"),
    ],
)
async def test_missing_or_badly_formed_headers_are_rejected(auth, header, message):
    with pytest.raises(AuthError, match=message):
        await auth.authenticate(header)


@pytest.mark.parametrize(
    "key", ["sk-proj-abc123", "gw-short", generate_key() + "x", "GW-" + generate_key()[3:]]
)
async def test_strings_that_cant_be_keys_are_rejected_without_a_lookup(auth, store, key):
    with pytest.raises(AuthError, match="Incorrect API key"):
        await auth.authenticate(f"Bearer {key}")

    assert store.lookups == 0


async def test_an_unknown_key_is_rejected_without_echoing_it(auth):
    key = generate_key()

    with pytest.raises(AuthError) as exc_info:
        await auth.authenticate(f"Bearer {key}")

    assert "Incorrect API key provided: " + key[:10] in exc_info.value.message
    assert key not in exc_info.value.message


async def test_a_revoked_key_is_rejected(auth, store):
    key, _ = store.add()
    store.replace(key, active=False)

    with pytest.raises(AuthError, match="has been revoked"):
        await auth.authenticate(f"Bearer {key}")


async def test_lookups_are_cached_until_the_ttl(auth, store, clock):
    key, _ = store.add()

    await auth.authenticate(f"Bearer {key}")
    clock.now += 29
    await auth.authenticate(f"Bearer {key}")
    assert store.lookups == 1

    clock.now += 2
    await auth.authenticate(f"Bearer {key}")
    assert store.lookups == 2


async def test_unknown_keys_are_cached_too(auth, store):
    key = generate_key()

    for _ in range(3):
        with pytest.raises(AuthError):
            await auth.authenticate(f"Bearer {key}")

    assert store.lookups == 1


async def test_a_revoke_applies_after_the_ttl(auth, store, clock):
    key, _ = store.add()
    await auth.authenticate(f"Bearer {key}")

    store.replace(key, active=False)
    # Still cached as active: the documented delay.
    await auth.authenticate(f"Bearer {key}")

    clock.now += 31
    with pytest.raises(AuthError, match="revoked"):
        await auth.authenticate(f"Bearer {key}")


async def test_new_limits_apply_after_the_ttl(auth, store, clock):
    key, _ = store.add(rpm_limit=60)
    await auth.authenticate(f"Bearer {key}")

    store.replace(key, rpm_limit=5)
    assert (await auth.authenticate(f"Bearer {key}")).rpm_limit == 60

    clock.now += 31
    assert (await auth.authenticate(f"Bearer {key}")).rpm_limit == 5


async def test_a_database_failure_is_unavailable_and_not_cached(auth, store):
    key, record = store.add()
    store.fail = True

    with pytest.raises(KeyStoreUnavailable):
        await auth.authenticate(f"Bearer {key}")

    store.fail = False
    assert await auth.authenticate(f"Bearer {key}") == record


async def test_the_cache_drops_its_oldest_entry_when_full(store, clock):
    auth = Authenticator(store, ttl_s=30, max_entries=2, clock=clock)
    keys = [store.add(f"k{i}")[0] for i in range(3)]

    for key in keys:
        await auth.authenticate(f"Bearer {key}")
    assert store.lookups == 3

    await auth.authenticate(f"Bearer {keys[2]}")  # still cached
    assert store.lookups == 3
    await auth.authenticate(f"Bearer {keys[0]}")  # was dropped
    assert store.lookups == 4


# What clients see.


def make_app(store: InMemoryKeyStore):
    app = create_app()
    model_router = ModelRouter(
        {"mock": [Target(provider="mock", model="mock-model")]}, {"mock": MockProvider()}
    )
    authenticator = Authenticator(store)
    app.dependency_overrides[get_model_router] = lambda: model_router
    app.dependency_overrides[get_authenticator] = lambda: authenticator
    return app


def sdk(app, api_key: str) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        api_key=api_key,
        base_url="http://gateway.test/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app)),
        max_retries=0,
    )


CHAT = {"model": "mock", "messages": [{"role": "user", "content": "hi"}]}


async def test_sdk_raises_authentication_error_for_a_wrong_key(store):
    client = sdk(make_app(store), generate_key())

    with pytest.raises(openai.AuthenticationError) as exc_info:
        await client.chat.completions.create(**CHAT)

    assert exc_info.value.status_code == 401
    assert exc_info.value.code == "invalid_api_key"
    assert exc_info.value.response.headers["www-authenticate"] == "Bearer"


async def test_sdk_raises_authentication_error_for_a_revoked_key(store):
    key, _ = store.add()
    store.replace(key, active=False)

    with pytest.raises(openai.AuthenticationError, match="revoked"):
        await sdk(make_app(store), key).chat.completions.create(**CHAT)


async def test_sdk_works_with_a_valid_key(store):
    key, _ = store.add()

    resp = await sdk(make_app(store), key).chat.completions.create(**CHAT)

    assert resp.choices[0].message.content == "Echo: hi"


def test_no_header_is_a_401_in_openai_shape(store):
    resp = TestClient(make_app(store)).post("/v1/chat/completions", json=CHAT)

    assert resp.status_code == 401
    assert resp.json() == {
        "error": {
            "message": resp.json()["error"]["message"],
            "type": "invalid_request_error",
            "param": None,
            "code": "invalid_api_key",
        }
    }
    assert "didn't provide an API key" in resp.json()["error"]["message"]


def test_auth_is_checked_before_the_body(store):
    # A bad body from someone without a key gets a 401, not a 400 about the body.
    resp = TestClient(make_app(store)).post("/v1/chat/completions", json={"model": "mock"})

    assert resp.status_code == 401


def test_models_needs_a_key_and_health_does_not(store):
    client = TestClient(make_app(store))
    key, _ = store.add()

    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": f"Bearer {key}"}).status_code == 200
    assert client.get("/health").status_code == 200


async def test_database_down_is_a_503_not_a_401():
    store = CountingStore()
    key, _ = store.add()
    store.fail = True

    with pytest.raises(openai.InternalServerError) as exc_info:
        await sdk(make_app(store), key).chat.completions.create(**CHAT)

    assert exc_info.value.status_code == 503


# The whole path, through Postgres.


def test_keys_from_the_cli_work_against_postgres(pg_conninfo, monkeypatch):
    with psycopg.connect(pg_conninfo) as conn:
        apply_migrations(conn)
        key, _ = create_key(conn, "alice", 60, 1000)
        revoked, record = create_key(conn, "bob", 60, 1000)
        revoke_key(conn, record.id)
    # The real startup: a real pool and the Postgres key store, on the test schema.
    monkeypatch.setattr(app.main, "open_pool", open_pool)
    monkeypatch.setattr(
        app.main, "get_settings", lambda: Settings(_env_file=None, database_url=pg_conninfo)
    )

    with TestClient(create_app()) as client:

        def status(api_key: str) -> int:
            headers = {"Authorization": f"Bearer {api_key}"}
            return client.post("/v1/chat/completions", json=CHAT, headers=headers).status_code

        assert status(key) == 200
        assert status(revoked) == 401
        assert status(generate_key()) == 401
