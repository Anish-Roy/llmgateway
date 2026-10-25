"""Creates or updates the gateway's tables from the SQL files in `schema/`.

No migration tool: every file is idempotent (`CREATE TABLE IF NOT EXISTS`,
`ADD COLUMN IF NOT EXISTS`...) and all of them run, in name order, every time. That's enough
for a handful of tables, and it works the same on the local container and a hosted Postgres.
"""

from pathlib import Path

import psycopg

SCHEMA_DIR = Path(__file__).resolve().parent / "schema"

# Any constant works: it only has to be the same for every run, so two runs at once (say, two
# deploys) take turns instead of racing on the same CREATE.
_LOCK_ID = 4_242_001


def schema_files() -> list[Path]:
    return sorted(SCHEMA_DIR.glob("*.sql"))


def apply_migrations(conn: psycopg.Connection) -> list[str]:
    """Runs every schema file in one transaction, so a failing file leaves nothing half-done.
    Returns the file names, in the order they ran."""
    files = schema_files()
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_ID,))
        for path in files:
            # No parameters, so psycopg sends the file as one simple query: several
            # statements per file are fine.
            conn.execute(path.read_text())
    return [path.name for path in files]
