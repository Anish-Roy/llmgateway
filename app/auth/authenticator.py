"""Checks a request's `Authorization: Bearer <key>` header against the key store.

Lookups are cached in memory for `ttl_s`, unknown keys included, so a busy client (or one
retrying a bad key) doesn't cost a database query per request. The price: a revoked key keeps
working, and a changed limit keeps its old value, for up to one TTL. The CLI that revokes keys
is a separate process and can't clear this cache.
"""

import logging
import re
import time
from collections.abc import Callable

import psycopg
from psycopg_pool import PoolTimeout

from app.auth.keys import KEY_PREFIX, ApiKey, KeyStore, display_prefix, hash_key

logger = logging.getLogger(__name__)

# What generate_key makes: "gw-" and 43 base64url characters. Anything else can't be a key,
# so it's rejected without a lookup.
_KEY_RE = re.compile(re.escape(KEY_PREFIX) + r"[A-Za-z0-9_-]{43}")


class AuthError(Exception):
    """The request has no usable key: always a 401."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class KeyStoreUnavailable(Exception):
    """The key couldn't be checked because the database is down: a 503, not a 401, since the
    key may well be fine."""


class Authenticator:
    def __init__(
        self,
        store: KeyStore,
        ttl_s: float = 30.0,
        max_entries: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.store = store
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self.clock = clock
        # key hash -> (expires at, the key or None for "no such key"), oldest first.
        self._cache: dict[str, tuple[float, ApiKey | None]] = {}

    async def authenticate(self, authorization: str | None) -> ApiKey:
        key = _bearer_key(authorization)
        if not _KEY_RE.fullmatch(key):
            raise AuthError(_incorrect(key))
        record = await self._lookup(hash_key(key))
        if record is None:
            logger.info("rejected unknown key %s...", display_prefix(key))
            raise AuthError(_incorrect(key))
        if not record.active:
            logger.info("rejected revoked key %d (%s)", record.id, record.name)
            raise AuthError(f"API key {record.key_prefix}... has been revoked.")
        return record

    async def _lookup(self, key_hash: str) -> ApiKey | None:
        now = self.clock()
        cached = self._cache.get(key_hash)
        if cached is not None and cached[0] > now:
            return cached[1]
        try:
            record = await self.store.get_by_hash(key_hash)
        except (psycopg.Error, PoolTimeout) as exc:
            # Not cached: the next request tries the database again.
            logger.warning("key lookup failed: %s", exc)
            raise KeyStoreUnavailable("Could not check the API key. Try again shortly.") from exc
        self._cache.pop(key_hash, None)
        if len(self._cache) >= self.max_entries:
            # Drop the oldest entry. A flood of made-up keys can push real ones out, which
            # costs extra lookups, never a wrong answer.
            del self._cache[next(iter(self._cache))]
        self._cache[key_hash] = (now + self.ttl_s, record)
        return record


def _bearer_key(authorization: str | None) -> str:
    if not authorization:
        raise AuthError(
            "You didn't provide an API key. Send it in an Authorization header using Bearer "
            "auth (Authorization: Bearer YOUR_KEY)."
        )
    scheme, _, key = authorization.strip().partition(" ")
    if scheme.lower() != "bearer" or not key.strip():
        raise AuthError("Authorization header must be 'Bearer YOUR_KEY'.")
    return key.strip()


def _incorrect(key: str) -> str:
    # Like OpenAI, show only the start of what was sent, never the whole key.
    return f"Incorrect API key provided: {display_prefix(key)}..."
