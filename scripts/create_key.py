"""Create, list and revoke client API keys in DATABASE_URL.

uv run python -m scripts.create_key create "alice" --rpm 60 --tpm 100000
uv run python -m scripts.create_key list
uv run python -m scripts.create_key revoke 3

The plain key is printed once, by `create`. Only its hash is stored, so it can't be shown
again: a lost key is revoked and replaced. A running gateway notices a revoke within its
key-cache TTL.
"""

import argparse
import sys

import psycopg

from app.auth.admin import create_key, list_keys, revoke_key
from app.config import get_settings

DEFAULT_RPM = 60
DEFAULT_TPM = 100_000


def _positive(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create", help="create a key and print it once")
    create.add_argument("name", help="who or what the key is for")
    create.add_argument("--rpm", type=_positive, default=DEFAULT_RPM, help="requests per minute")
    create.add_argument("--tpm", type=_positive, default=DEFAULT_TPM, help="tokens per minute")

    commands.add_parser("list", help="list keys (never the keys themselves)")

    revoke = commands.add_parser("revoke", help="deactivate a key by its id")
    revoke.add_argument("id", type=int)

    args = parser.parse_args(argv)
    with psycopg.connect(get_settings().database_url) as conn:
        match args.command:
            case "create":
                key, record = create_key(conn, args.name, args.rpm, args.tpm)
                limits = f"{record.rpm_limit} rpm, {record.tpm_limit} tpm"
                print(f"Created key {record.id} ({record.name}, {limits}).")
                print(f"It is shown only this once:\n\n  {key}\n")
            case "list":
                print(f"{'id':>4}  {'prefix':10}  {'rpm':>6}  {'tpm':>9}  {'status':8}  name")
                for k in list_keys(conn):
                    status = "active" if k.active else "revoked"
                    print(
                        f"{k.id:>4}  {k.key_prefix:10}  {k.rpm_limit:>6}  {k.tpm_limit:>9}"
                        f"  {status:8}  {k.name}"
                    )
            case "revoke":
                record = revoke_key(conn, args.id)
                if record is None:
                    print(f"No key with id {args.id}", file=sys.stderr)
                    return 1
                print(f"Revoked key {record.id} ({record.name}, {record.key_prefix}...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
