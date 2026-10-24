import json
from pathlib import Path

import pytest

DATASET = Path(__file__).resolve().parent.parent / "data" / "queries.jsonl"


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    with DATASET.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_size_is_in_range(rows):
    assert 300 <= len(rows) <= 500


def test_ids_and_prompts_are_unique(rows):
    assert len({r["id"] for r in rows}) == len(rows)
    assert len({r["prompt"] for r in rows}) == len(rows)


def test_fields_are_valid(rows):
    for r in rows:
        assert r["prompt"].strip()
        assert r["difficulty"] in {"easy", "hard"}
        assert not (r["dup_of"] and r["trap_of"]), r["id"]


def test_references_point_to_original_prompts(rows):
    by_id = {r["id"]: r for r in rows}
    for r in rows:
        for ref in (r["dup_of"], r["trap_of"]):
            if ref:
                target = by_id[ref]
                assert target["dup_of"] is None and target["trap_of"] is None


def test_near_duplicate_share(rows):
    share = sum(r["dup_of"] is not None for r in rows) / len(rows)
    assert 0.20 <= share <= 0.30
