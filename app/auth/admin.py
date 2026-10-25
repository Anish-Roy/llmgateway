"""Creating, listing and revoking keys, for `scripts/create_key.py`.

Sync, on a plain psycopg connection: these run from the command line, not in the gateway.
"""

import psycopg
from psycopg.rows import class_row

from app.auth.keys import API_KEY_COLUMNS, ApiKey, display_prefix, generate_key, hash_key


def create_key(
    conn: psycopg.Connection, name: str, rpm_limit: int, tpm_limit: int
) -> tuple[str, ApiKey]:
    """Stores a new key and returns it in plain text with its record. The plain key exists
    only in the return value: show it once, it can't be recovered."""
    key = generate_key()
    cur = conn.cursor(row_factory=class_row(ApiKey))
    cur.execute(
        "INSERT INTO api_keys (name, key_hash, key_prefix, rpm_limit, tpm_limit)"
        f" VALUES (%s, %s, %s, %s, %s) RETURNING {API_KEY_COLUMNS}",
        (name, hash_key(key), display_prefix(key), rpm_limit, tpm_limit),
    )
    record = cur.fetchone()
    assert record is not None
    return key, record


def list_keys(conn: psycopg.Connection) -> list[ApiKey]:
    cur = conn.cursor(row_factory=class_row(ApiKey))
    return cur.execute(f"SELECT {API_KEY_COLUMNS} FROM api_keys ORDER BY id").fetchall()


def revoke_key(conn: psycopg.Connection, key_id: int) -> ApiKey | None:
    """Marks the key inactive. The row stays, so Phase 5's request logs keep their key.
    Returns None if there's no key with this id; revoking twice is fine."""
    cur = conn.cursor(row_factory=class_row(ApiKey))
    cur.execute(
        f"UPDATE api_keys SET active = false WHERE id = %s RETURNING {API_KEY_COLUMNS}",
        (key_id,),
    )
    return cur.fetchone()
