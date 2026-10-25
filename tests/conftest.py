import secrets
from contextlib import asynccontextmanager

import psycopg
import pytest
from psycopg.conninfo import make_conninfo

import app.main
from app.cache.embedder import FakeEmbedder
from app.cache.store import NoCacheStore
from app.config import Settings
from app.usage import tokens
from app.usage.log import InMemoryUsageLog


@pytest.fixture(autouse=True)
def _no_tokenizer_download(monkeypatch):
    # Tests never download tiktoken's encoding (an 18 s network fetch the first time): apps
    # started with their lifespan skip the background load, and counts are chars/4 unless a
    # test sets an encoding itself.
    monkeypatch.setattr(app.main, "load_encoding_in_background", lambda: None)
    monkeypatch.setattr(tokens, "_encoding", None)


@pytest.fixture(autouse=True)
def _no_model_download(monkeypatch):
    # Tests never load the embedding model (a 90 MB download, then seconds of torch): apps
    # started with their lifespan get a fake that embeds by hash.
    monkeypatch.setattr(app.main, "load_embedder", FakeEmbedder)


@pytest.fixture(autouse=True)
def _no_database_at_startup(monkeypatch):
    # Apps started with their lifespan get no Postgres pool, so tests run without a database.
    # Tests that need one use the `pg_conninfo` fixture.
    @asynccontextmanager
    async def no_pool(conninfo: str, max_size: int = 10):
        yield None

    monkeypatch.setattr(app.main, "open_pool", no_pool)
    # With no pool, request logs are kept in memory instead.
    monkeypatch.setattr(app.main, "PostgresUsageLog", lambda pool: InMemoryUsageLog())
    # And nothing is cached, as in an app that was never started.
    monkeypatch.setattr(app.main, "PostgresCacheStore", lambda pool: NoCacheStore())


@pytest.fixture
def pg_conninfo():
    """A connection string for a fresh, empty schema in DATABASE_URL, dropped afterwards.

    Skips the test when Postgres isn't reachable (`docker compose up -d` starts it).
    Connections made with it see only that schema (and `public`, for extensions), so tests
    never touch the dev tables and can run side by side.
    """
    base = Settings(_env_file=None).database_url
    schema = f"test_{secrets.token_hex(6)}"
    try:
        admin = psycopg.connect(base, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres not reachable at {base}: {exc}")
    with admin:
        admin.execute(f"CREATE SCHEMA {schema}")
        try:
            yield make_conninfo(base, options=f"-c search_path={schema},public")
        finally:
            admin.execute(f"DROP SCHEMA {schema} CASCADE")
