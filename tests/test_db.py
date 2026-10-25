"""Schema migrations and the connection pool, against a real Postgres (skipped without one)."""

import psycopg
import pytest

from app.db.migrate import apply_migrations, schema_files
from app.db.pool import open_pool

VALID_HASH = "a" * 64


def _insert_key(conn: psycopg.Connection, **overrides) -> None:
    row = {
        "name": "test",
        "key_hash": VALID_HASH,
        "key_prefix": "gw-abcd",
        "rpm_limit": 60,
        "tpm_limit": 10_000,
    } | overrides
    conn.execute(
        "INSERT INTO api_keys (name, key_hash, key_prefix, rpm_limit, tpm_limit)"
        " VALUES (%(name)s, %(key_hash)s, %(key_prefix)s, %(rpm_limit)s, %(tpm_limit)s)",
        row,
    )


def test_schema_files_run_in_name_order():
    names = [p.name for p in schema_files()]

    assert names == sorted(names)
    assert "001_api_keys.sql" in names


def test_migrations_create_the_api_keys_table(pg_conninfo):
    with psycopg.connect(pg_conninfo) as conn:
        applied = apply_migrations(conn)
        columns = {
            row[0]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns"
                " WHERE table_name = 'api_keys' AND table_schema = current_schema()"
            )
        }

    assert applied == [p.name for p in schema_files()]
    assert columns == {
        "id",
        "name",
        "key_hash",
        "key_prefix",
        "rpm_limit",
        "tpm_limit",
        "active",
        "created_at",
    }


def test_migrations_can_run_again(pg_conninfo):
    with psycopg.connect(pg_conninfo) as conn:
        apply_migrations(conn)
        _insert_key(conn)
        conn.commit()

        apply_migrations(conn)

        # A second run changes nothing, and keeps the data.
        assert conn.execute("SELECT count(*) FROM api_keys").fetchone() == (1,)


def test_new_keys_are_active(pg_conninfo):
    with psycopg.connect(pg_conninfo) as conn:
        apply_migrations(conn)
        _insert_key(conn)

        assert conn.execute("SELECT active FROM api_keys").fetchone() == (True,)


@pytest.mark.parametrize(
    "overrides",
    [
        {"key_hash": "not-a-sha256"},
        {"key_hash": "A" * 64},  # hex must be lowercase, so one key has one spelling
        {"rpm_limit": 0},
        {"tpm_limit": -1},
    ],
)
def test_bad_rows_are_rejected(pg_conninfo, overrides):
    with psycopg.connect(pg_conninfo) as conn:
        apply_migrations(conn)

        with pytest.raises(psycopg.errors.CheckViolation):
            _insert_key(conn, **overrides)


def test_a_key_hash_is_stored_once(pg_conninfo):
    with psycopg.connect(pg_conninfo) as conn:
        apply_migrations(conn)
        _insert_key(conn)

        with pytest.raises(psycopg.errors.UniqueViolation):
            _insert_key(conn, name="other")


async def test_pool_opens_and_runs_queries(pg_conninfo):
    async with open_pool(pg_conninfo, max_size=2) as pool, pool.connection() as conn:
        cur = await conn.execute("SELECT 1")

        assert await cur.fetchone() == (1,)


async def test_pool_fails_fast_when_postgres_is_unreachable(monkeypatch):
    from psycopg_pool import PoolTimeout

    import app.db.pool

    monkeypatch.setattr(app.db.pool, "CONNECT_TIMEOUT_S", 0.5)
    # Port 1 on localhost: nothing listens there, so every connection is refused at once.
    with pytest.raises(PoolTimeout):
        async with open_pool("postgresql://nobody@127.0.0.1:1/none?connect_timeout=1"):
            pass
