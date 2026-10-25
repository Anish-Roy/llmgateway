"""Create or update the gateway's tables in DATABASE_URL. Safe to run any number of times.

uv run python -m scripts.migrate
"""

import psycopg

from app.config import get_settings
from app.db.migrate import apply_migrations


def main() -> None:
    with psycopg.connect(get_settings().database_url) as conn:
        for name in apply_migrations(conn):
            print(f"applied {name}")


if __name__ == "__main__":
    main()
