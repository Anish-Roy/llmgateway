"""The semantic_cache table (migration 004), against a real Postgres (skipped without one)."""

import hashlib
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import psycopg
import pytest

from app.cache.embedder import DIMENSIONS, FakeEmbedder
from app.db.migrate import apply_migrations

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
CONTEXT = hashlib.sha256(b"fast|Be brief.").digest()


def _literal(vector: np.ndarray) -> str:
    # pgvector's text form; the gateway sends vectors the same way, so no adapter is needed.
    return "[" + ",".join(f"{x:.7f}" for x in vector) + "]"


@pytest.fixture
def migrated(pg_conninfo):
    """A migrated schema with two API keys, ids returned as `migrated.keys`."""
    with psycopg.connect(pg_conninfo) as conn:
        apply_migrations(conn)
        for name in ("alice", "bob"):
            conn.execute(
                "INSERT INTO api_keys (name, key_hash, key_prefix, rpm_limit, tpm_limit)"
                " VALUES (%s, %s, 'gw-test', 60, 1000)",
                (name, hashlib.sha256(name.encode()).hexdigest()),
            )
    return pg_conninfo


def _key_ids(conn: psycopg.Connection) -> list[int]:
    return [row[0] for row in conn.execute("SELECT id FROM api_keys ORDER BY id")]


def _row(key_id: int, prompt: str, vector: np.ndarray, **overrides) -> dict:
    return {
        "entry_key": hashlib.sha256(f"{key_id}|{prompt}".encode()).digest(),
        "key_id": key_id,
        "context_hash": CONTEXT,
        "alias": "fast",
        "prompt": prompt,
        "embedding": _literal(vector),
        "response": '{"model": "m", "choices": []}',
        "provider": "groq",
        "model": "openai/gpt-oss-20b",
        "cost_usd": Decimal("0.00001125"),
        "created_at": NOW,
        "expires_at": NOW + timedelta(hours=24),
    } | overrides


def _insert(conn: psycopg.Connection, row: dict) -> None:
    columns = ", ".join(row)
    values = ", ".join("%(embedding)s::vector" if c == "embedding" else f"%({c})s" for c in row)
    conn.execute(f"INSERT INTO semantic_cache ({columns}) VALUES ({values})", row)


def test_the_table_has_its_columns(migrated):
    with psycopg.connect(migrated) as conn:
        columns = {
            name: kind
            for name, kind in conn.execute(
                "SELECT attname, format_type(atttypid, atttypmod) FROM pg_attribute"
                " WHERE attrelid = 'semantic_cache'::regclass AND attnum > 0"
                " AND NOT attisdropped"
            )
        }
    assert set(columns) == {
        "entry_key",
        "key_id",
        "context_hash",
        "alias",
        "prompt",
        "embedding",
        "response",
        "provider",
        "model",
        "cost_usd",
        "created_at",
        "expires_at",
    }
    # The width the embedder makes: a model change has to change both.
    assert columns["embedding"] == f"vector({DIMENSIONS})"


def test_the_embedding_index_is_hnsw_by_cosine_distance(migrated):
    with psycopg.connect(migrated) as conn:
        indexes = dict(
            conn.execute(
                "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'semantic_cache'"
                " AND schemaname = current_schema()"
            ).fetchall()
        )
    assert "USING hnsw (embedding vector_cosine_ops)" in indexes["semantic_cache_embedding"]
    assert "(key_id, context_hash)" in indexes["semantic_cache_key_context"]
    assert "(expires_at)" in indexes["semantic_cache_expires_at"]


def test_migrations_can_run_again_with_rows_in_the_table(migrated):
    with psycopg.connect(migrated) as conn:
        _insert(conn, _row(_key_ids(conn)[0], "What is HTTP?", FakeEmbedder().embed(["x"])[0]))
        conn.commit()
        apply_migrations(conn)
        assert conn.execute("SELECT count(*) FROM semantic_cache").fetchone() == (1,)


@pytest.mark.parametrize(
    "bad",
    [
        {"entry_key": b"short"},
        {"context_hash": b"short"},
        {"cost_usd": Decimal("-0.1")},
        # Expires before it was made.
        {"expires_at": NOW - timedelta(seconds=1)},
    ],
)
def test_bad_rows_are_rejected(migrated, bad):
    with psycopg.connect(migrated) as conn, pytest.raises(psycopg.errors.CheckViolation):
        _insert(conn, _row(_key_ids(conn)[0], "q", FakeEmbedder().embed(["q"])[0], **bad))


def test_a_vector_of_the_wrong_width_is_rejected(migrated):
    with psycopg.connect(migrated) as conn, pytest.raises(psycopg.errors.DataException):
        _insert(conn, _row(_key_ids(conn)[0], "q", np.ones(DIMENSIONS + 1) / 20))


def test_entries_need_a_real_key(migrated):
    with psycopg.connect(migrated) as conn, pytest.raises(psycopg.errors.ForeignKeyViolation):
        _insert(conn, _row(999_999, "q", FakeEmbedder().embed(["q"])[0]))


def test_the_same_entry_key_is_stored_once(migrated):
    with psycopg.connect(migrated) as conn:
        row = _row(_key_ids(conn)[0], "q", FakeEmbedder().embed(["q"])[0])
        _insert(conn, row)
        with pytest.raises(psycopg.errors.UniqueViolation):
            _insert(conn, row)


LOOKUP = (
    "SELECT prompt, 1 - (embedding <=> %(q)s::vector) FROM semantic_cache"
    " WHERE key_id = %(key)s AND context_hash = %(ctx)s AND expires_at > %(now)s"
    " ORDER BY embedding <=> %(q)s::vector LIMIT 1"
)


@pytest.mark.parametrize(("iterative_scan", "found"), [("relaxed_order", True), ("off", False)])
def test_a_keys_nearest_entry_is_found_past_nearer_entries_of_other_keys(
    migrated, iterative_scan, found
):
    """The lookup step 4 will run: nearest by cosine distance, filtered to one key, context
    and live rows, through the HNSW index as on a big table. Bob's entry is the 61st nearest.
    HNSW's first batch (ef_search 40) holds only Alice's, and the filter drops them all: only
    an iterative scan goes on searching and finds Bob's."""
    fake = FakeEmbedder()
    query = fake.embed(["What is HTTP?"])[0]
    noise = np.random.default_rng(0).standard_normal((61, DIMENSIONS)).astype(np.float32)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    # Mixes of the query and noise, from nearest (0.999) to farthest (0.675).
    weights = np.linspace(0.95, 0.5, 61, dtype=np.float32)[:, None]
    near = weights * query + (1 - weights) * noise
    near /= np.linalg.norm(near, axis=1, keepdims=True)
    params = {"q": _literal(query), "ctx": CONTEXT, "now": NOW}
    with psycopg.connect(migrated) as conn:
        alice, bob = _key_ids(conn)
        params["key"] = bob
        for i, vector in enumerate(near[:60]):
            _insert(conn, _row(alice, f"alice {i}", vector))
        _insert(conn, _row(bob, "bob's question", near[60]))
        # An expired entry of Bob's, nearer than his live one, is never served.
        expired = {"created_at": NOW - timedelta(days=2), "expires_at": NOW - timedelta(days=1)}
        _insert(conn, _row(bob, "expired", near[0], **expired))
        # Leave the planner only the HNSW index, as a big table would: on 62 rows it would
        # rather scan the btree indexes exactly.
        conn.execute("DROP INDEX semantic_cache_key_context, semantic_cache_expires_at")
        conn.execute("SET enable_seqscan = off")
        conn.execute("SET enable_bitmapscan = off")
        conn.execute(f"SET hnsw.iterative_scan = {iterative_scan}")
        plan = "\n".join(r[0] for r in conn.execute(f"EXPLAIN {LOOKUP}", params))
        row = conn.execute(LOOKUP, params).fetchone()
        conn.rollback()
    assert "Index Scan using semantic_cache_embedding" in plan
    if found:
        assert row[0] == "bob's question"
        assert row[1] == pytest.approx(float(near[60] @ query), abs=1e-5)
    else:
        assert row is None
