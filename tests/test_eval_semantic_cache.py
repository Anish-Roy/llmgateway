"""The threshold experiment's simulated cache, its scoring and its rule for picking."""

import numpy as np
import pytest

from app.cache.embedder import DIMENSIONS
from scripts.eval_semantic_cache import (
    DATASET,
    Hit,
    Query,
    Score,
    best_possible_hit_rate,
    embed_texts,
    load_queries,
    meaning_groups,
    pick,
    replay,
    score,
)


def _q(qid: str, prompt: str | None = None, dup_of=None, trap_of=None) -> Query:
    return Query(qid, prompt or f"prompt {qid}", dup_of, trap_of)


def _vectors(*similarities_to_first: float) -> np.ndarray:
    """Unit vectors in a plane: the first is (1, 0); each next one has the given cosine
    similarity to it. Two of them are as similar as the cosine of the angle between them."""
    rows = np.zeros((1 + len(similarities_to_first), DIMENSIONS), dtype=np.float32)
    rows[0, 0] = 1
    for i, s in enumerate(similarities_to_first, start=1):
        rows[i, 0], rows[i, 1] = s, np.sqrt(1 - s * s)
    return rows


def test_a_similar_enough_prompt_hits_the_stored_one():
    queries = [_q("a"), _q("b", dup_of="a")]
    result = replay(queries, _vectors(0.9), threshold=0.85)
    assert result.hits == [Hit("b", "a", pytest.approx(0.9))]


def test_a_prompt_below_the_threshold_misses():
    queries = [_q("a"), _q("b")]
    assert replay(queries, _vectors(0.84), threshold=0.85).hits == []


def test_the_threshold_itself_is_a_hit():
    queries = [_q("a"), _q("b")]
    vectors = np.zeros((2, DIMENSIONS), dtype=np.float32)
    vectors[:, 0] = 1
    assert len(replay(queries, vectors, threshold=1.0).hits) == 1


def test_hits_arent_stored_so_later_prompts_cant_match_them():
    # b hits a; c is close to b (cos 0.99) but not to a (0.80). With b never stored, c misses.
    queries = [_q("a"), _q("b"), _q("c")]
    vectors = _vectors(0.9, 0.8)
    assert float(vectors[1] @ vectors[2]) > 0.98
    assert [h.query for h in replay(queries, vectors, threshold=0.85).hits] == ["b"]


def test_a_hit_takes_the_most_similar_stored_prompt():
    # b and c both miss a (stored); d is closer to c than to a or b.
    queries = [_q("a"), _q("b"), _q("c"), _q("d")]
    vectors = np.zeros((4, DIMENSIONS), dtype=np.float32)
    vectors[0, 0] = vectors[1, 1] = vectors[2, 2] = 1
    vectors[3, 2], vectors[3, 0] = 0.95, np.sqrt(1 - 0.95**2)
    (hit,) = replay(queries, vectors, threshold=0.9).hits
    assert hit.matched == "c"


def test_the_same_text_again_is_an_exact_hit_not_a_semantic_one():
    queries = [_q("a", "same"), _q("b", "same")]
    result = replay(queries, _vectors(1.0), threshold=0.85)
    assert result.exact_hits == 1
    assert result.hits == []


def test_meaning_groups_follow_dup_of_to_the_original():
    groups = meaning_groups([_q("a"), _q("b", dup_of="a"), _q("c", dup_of="b"), _q("d")])
    assert groups == {"a": "a", "b": "a", "c": "a", "d": "d"}


def test_a_dup_of_loop_is_an_error():
    with pytest.raises(ValueError, match="loops"):
        meaning_groups([_q("a", dup_of="b"), _q("b", dup_of="a")])


def test_an_original_after_its_duplicate_is_a_right_hit():
    # The duplicate came first and was stored; the original then hits it. Same meaning.
    queries = [_q("b", dup_of="a"), _q("a")]
    s = score(queries, _vectors(0.9), threshold=0.85)
    assert len(s.hits) == 1
    assert s.false_hits == []


def test_a_hit_on_a_look_alike_is_a_false_hit_and_a_trap_hit():
    queries = [_q("a"), _q("b", trap_of="a"), _q("c")]
    vectors = _vectors(0.9, 0.0)
    s = score(queries, vectors, threshold=0.85)
    assert [h.query for h in s.false_hits] == ["b"]
    assert [h.query for h in s.trap_hits] == ["b"]
    assert s.trap_hit_rate == 1.0


def test_a_hit_across_groups_that_isnt_a_trap_is_only_a_false_hit():
    queries = [_q("a"), _q("b")]
    s = score(queries, _vectors(0.9), threshold=0.85)
    assert len(s.false_hits) == 1
    assert s.trap_hits == []


def test_rates():
    queries = [_q("a"), _q("b", dup_of="a"), _q("c", trap_of="a"), _q("d")]
    s = score(queries, _vectors(0.95, 0.9, 0.0), threshold=0.85)
    assert s.hit_rate == 2 / 4
    assert s.false_hit_rate == 1 / 2
    assert s.trap_hit_rate == 1 / 1


def test_best_possible_hit_rate_counts_every_prompt_after_the_first_of_its_group():
    queries = [_q("a"), _q("b", dup_of="a"), _q("c", dup_of="a"), _q("d"), _q("e", trap_of="d")]
    assert best_possible_hit_rate(queries) == 2 / 5


def _score(threshold: float, false_hits: int) -> Score:
    hit = Hit("x", "y", threshold)
    return Score(threshold, 10, 0, [hit] * 3, [hit] * false_hits, [], 1)


def test_pick_is_the_lowest_threshold_with_no_false_hits_above_it():
    scores = [_score(0.88, 2), _score(0.89, 1), _score(0.90, 0), _score(0.91, 0)]
    assert pick(scores).threshold == 0.90


def test_pick_skips_a_lucky_gap():
    # 0.89 has no false hits, but 0.90 does: 0.89 is luck, not a safe region.
    scores = [_score(0.89, 0), _score(0.90, 1), _score(0.91, 0), _score(0.92, 0)]
    assert pick(scores).threshold == 0.91


def test_pick_is_none_when_the_highest_threshold_has_false_hits():
    assert pick([_score(0.96, 0), _score(0.97, 1)]) is None


def test_every_prompt_in_the_dataset_is_one_the_gateway_would_embed():
    queries = load_queries(DATASET)
    texts = embed_texts(queries)
    assert len(texts) == len(queries)
    assert all(texts)
