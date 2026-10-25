"""API key format, the key stores, and the create/list/revoke CLI."""

import re

import psycopg
import pytest

import scripts.create_key
from app.auth.admin import create_key, list_keys, revoke_key
from app.auth.keys import (
    InMemoryKeyStore,
    PostgresKeyStore,
    display_prefix,
    generate_key,
    hash_key,
)
from app.config import Settings
from app.db.migrate import apply_migrations
from app.db.pool import open_pool


def test_keys_are_prefixed_long_and_unique():
    keys = {generate_key() for _ in range(100)}

    assert len(keys) == 100
    for key in keys:
        assert key.startswith("gw-")
        assert len(key) == 3 + 43  # 32 random bytes, base64url without padding
        assert re.fullmatch(r"gw-[A-Za-z0-9_-]{43}", key)


def test_hash_is_the_lowercase_hex_the_table_accepts():
    key = generate_key()

    assert hash_key(key) == hash_key(key)
    assert re.fullmatch(r"[0-9a-f]{64}", hash_key(key))
    assert hash_key(key) != hash_key(key + "x")


def test_display_prefix_shows_only_the_start():
    key = generate_key()

    assert display_prefix(key) == key[:10]
    assert key.startswith(display_prefix(key))


async def test_in_memory_store_finds_keys_by_hash():
    store = InMemoryKeyStore()
    key, record = store.add("alice", rpm_limit=5, tpm_limit=500)

    assert await store.get_by_hash(hash_key(key)) == record
    assert record.rpm_limit == 5 and record.tpm_limit == 500 and record.active
    assert await store.get_by_hash(hash_key(generate_key())) is None


async def test_in_memory_store_returns_revoked_keys_as_inactive():
    store = InMemoryKeyStore()
    key, _ = store.add()

    store.replace(key, active=False)

    found = await store.get_by_hash(hash_key(key))
    assert found is not None and not found.active


@pytest.fixture
def migrated(pg_conninfo):
    with psycopg.connect(pg_conninfo) as conn:
        apply_migrations(conn)
    return pg_conninfo


def test_only_the_hash_is_stored(migrated):
    with psycopg.connect(migrated) as conn:
        key, record = create_key(conn, "alice", 60, 1000)
        row = conn.execute("SELECT * FROM api_keys WHERE id = %s", (record.id,)).fetchone()

    assert row is not None
    assert key not in [str(v) for v in row]
    assert hash_key(key) in row
    assert record.key_prefix == display_prefix(key)


async def test_postgres_store_finds_keys_by_hash(migrated):
    with psycopg.connect(migrated) as conn:
        key, record = create_key(conn, "alice", 60, 1000)
        revoked_key, revoked = create_key(conn, "bob", 60, 1000)
        revoke_key(conn, revoked.id)

    async with open_pool(migrated, max_size=1) as pool:
        store = PostgresKeyStore(pool)

        assert await store.get_by_hash(hash_key(key)) == record
        found = await store.get_by_hash(hash_key(revoked_key))
        assert found is not None and found.id == revoked.id and not found.active
        assert await store.get_by_hash(hash_key(generate_key())) is None


def test_list_and_revoke(migrated):
    with psycopg.connect(migrated) as conn:
        _, first = create_key(conn, "alice", 60, 1000)
        _, second = create_key(conn, "bob", 10, 500)

        assert revoke_key(conn, second.id).active is False
        # Revoking twice is fine; an unknown id is None.
        assert revoke_key(conn, second.id).active is False
        assert revoke_key(conn, 9999) is None

        keys = list_keys(conn)

    assert [k.id for k in keys] == [first.id, second.id]
    assert [k.active for k in keys] == [True, False]


def test_cli_create_list_revoke(migrated, monkeypatch, capsys):
    monkeypatch.setattr(
        scripts.create_key, "get_settings", lambda: Settings(_env_file=None, database_url=migrated)
    )

    assert scripts.create_key.main(["create", "alice", "--rpm", "5", "--tpm", "500"]) == 0
    key = re.search(r"gw-\S{43}", capsys.readouterr().out).group()

    assert scripts.create_key.main(["list"]) == 0
    listing = capsys.readouterr().out
    assert display_prefix(key) in listing
    assert key not in listing
    assert "active" in listing and "alice" in listing

    assert scripts.create_key.main(["revoke", "1"]) == 0
    assert scripts.create_key.main(["revoke", "99"]) == 1
    capsys.readouterr()
    scripts.create_key.main(["list"])
    assert "revoked" in capsys.readouterr().out


def test_cli_rejects_non_positive_limits(capsys):
    with pytest.raises(SystemExit):
        scripts.create_key.main(["create", "alice", "--rpm", "0"])
    assert "positive" in capsys.readouterr().err
