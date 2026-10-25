from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from psycopg_pool import AsyncConnectionPool

# How long startup waits for the first connection before giving up.
CONNECT_TIMEOUT_S = 10.0


@asynccontextmanager
async def open_pool(conninfo: str, max_size: int = 10) -> AsyncIterator[AsyncConnectionPool]:
    """A connection pool for the app's lifetime, closed on exit.

    Waits for a first connection, so a wrong DATABASE_URL or a stopped Postgres stops the app
    at startup instead of failing the first request.
    """
    pool = AsyncConnectionPool(
        conninfo,
        min_size=1,
        max_size=max_size,
        open=False,
        # Checks a connection before handing it out. Hosted Postgres (Neon, Supabase) closes
        # idle connections, and a dead one would otherwise fail a request.
        check=AsyncConnectionPool.check_connection,
    )
    await pool.open(wait=True, timeout=CONNECT_TIMEOUT_S)
    try:
        yield pool
    finally:
        await pool.close()
