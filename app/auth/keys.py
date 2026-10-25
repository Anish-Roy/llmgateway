"""Client API keys: the key format, and where the gateway looks keys up.

A key is `gw-` plus 32 random bytes (`secrets.token_urlsafe`). Only its SHA-256 is stored. A
plain hash is enough because the key is random and long: there's nothing to guess, so a slow
password hash (bcrypt, argon2) would only add cost to every request, and its salt would make
looking a key up by its hash impossible.
"""

import dataclasses
import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from psycopg.rows import class_row
from psycopg_pool import AsyncConnectionPool

KEY_PREFIX = "gw-"
# How much of the key `list` shows: "gw-" plus 7 characters, about 42 bits. Enough to tell
# keys apart, far too little to guess the other 36.
DISPLAY_LENGTH = 10


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    """Lowercase hex SHA-256: the form `api_keys.key_hash` stores and looks up."""
    return hashlib.sha256(key.encode()).hexdigest()


def display_prefix(key: str) -> str:
    return key[:DISPLAY_LENGTH]


@dataclass(frozen=True)
class ApiKey:
    """A row of `api_keys`, without the hash."""

    id: int
    name: str
    key_prefix: str
    rpm_limit: int
    tpm_limit: int
    active: bool
    created_at: datetime


# The columns of ApiKey, in its field order, for SELECT and RETURNING.
API_KEY_COLUMNS = "id, name, key_prefix, rpm_limit, tpm_limit, active, created_at"


class KeyStore(Protocol):
    """How the gateway finds a key. Read-only: keys are created and revoked by
    `scripts/create_key.py`, never through the gateway itself."""

    async def get_by_hash(self, key_hash: str) -> ApiKey | None:
        """The key with this hash, revoked or not, or None if there's no such key."""
        ...


class PostgresKeyStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self.pool = pool

    async def get_by_hash(self, key_hash: str) -> ApiKey | None:
        async with self.pool.connection() as conn:
            cur = conn.cursor(row_factory=class_row(ApiKey))
            await cur.execute(
                f"SELECT {API_KEY_COLUMNS} FROM api_keys WHERE key_hash = %s", (key_hash,)
            )
            return await cur.fetchone()


class InMemoryKeyStore:
    """A key store for tests: no database."""

    def __init__(self) -> None:
        self._keys: dict[str, ApiKey] = {}

    def add(
        self, name: str = "test", rpm_limit: int = 60, tpm_limit: int = 100_000
    ) -> tuple[str, ApiKey]:
        """Creates a key and returns it with its record, as the CLI does."""
        key = generate_key()
        record = ApiKey(
            id=len(self._keys) + 1,
            name=name,
            key_prefix=display_prefix(key),
            rpm_limit=rpm_limit,
            tpm_limit=tpm_limit,
            active=True,
            created_at=datetime.now().astimezone(),
        )
        self._keys[hash_key(key)] = record
        return key, record

    def replace(self, key: str, **changes) -> ApiKey:
        """Changes a stored key (revoke it, change its limits), as an admin would."""
        record = dataclasses.replace(self._keys[hash_key(key)], **changes)
        self._keys[hash_key(key)] = record
        return record

    async def get_by_hash(self, key_hash: str) -> ApiKey | None:
        return self._keys.get(key_hash)
