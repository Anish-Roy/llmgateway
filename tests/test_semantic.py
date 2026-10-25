"""The semantic cache's rules: the opt-in header, the context a match must share, and the
guard it must pass besides the similarity threshold."""

import pytest

from app.api.schemas import ChatRequest
from app.cache.key import cache_key
from app.cache.semantic import (
    DEFAULT_THRESHOLD,
    context_hash,
    entry_key,
    same_numbers,
    wants_semantic,
)
from app.config import Settings


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("What's the capital of France?", "capital of france?"),
        # Rephrased, with the numbers moved or written differently around them.
        ("bat and ball cost 1.10 together", "A bat and a ball cost $1.10 in total."),
        ("solve 3x+2y=16 and 5x-y=18", "Solve: 3x + 2y = 16 and 5x - y = 18."),
        ("Is 97 a prime number?", "is 97 prime"),
        # The same numbers in another order still match: the guard can't see order.
        ("Convert 100 miles to kilometres.", "Convert 100 kilometres to miles."),
    ],
)
def test_texts_with_the_same_numbers_match(a, b):
    assert same_numbers(a, b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("A bat and a ball cost $1.10 in total.", "A bat and a ball cost $1.20 in total."),
        ("how long would 1 machine take", "how long would 100 machines take"),
        ("What is 25% of 200?", "What is 15% of 200?"),
        ("Is 91 a prime number?", "Is 97 a prime number?"),
        # A number on one side only.
        ("What is the square root of 2?", "What is the square root?"),
        # The same number twice is not the same as once.
        ("5 machines 5 minutes", "5 machines 2 minutes"),
    ],
)
def test_texts_with_different_numbers_dont_match(a, b):
    assert not same_numbers(a, b)


def test_the_threshold_setting_defaults_to_the_measured_value():
    assert Settings(_env_file=None).semantic_cache_threshold == DEFAULT_THRESHOLD == 0.95


@pytest.mark.parametrize("bad", [0, 1.01, -0.5])
def test_the_threshold_setting_is_a_similarity(bad):
    with pytest.raises(ValueError):
        Settings(_env_file=None, semantic_cache_threshold=bad)


# The opt-in header.


@pytest.mark.parametrize("header", ["semantic", "Semantic", " SEMANTIC "])
def test_semantic_opts_in(header):
    assert wants_semantic(header)


@pytest.mark.parametrize("header", [None, "", "off", "exact", "semantic,off"])
def test_anything_else_doesnt(header):
    assert not wants_semantic(header)


# The context hash: everything but the user message's text.


def _request(**overrides) -> ChatRequest:
    fields = {
        "model": "fast",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "What is HTTP?"},
        ],
        "temperature": 0,
    }
    return ChatRequest.model_validate({**fields, **overrides})


def _asking(question: str, **overrides) -> ChatRequest:
    request = _request(**overrides)
    request.messages[-1].content = question
    return request


def test_the_context_leaves_out_the_question():
    assert context_hash(_asking("What is HTTP?")) == context_hash(_asking("what's http"))
    assert len(context_hash(_request())) == 32


@pytest.mark.parametrize(
    "changed",
    [
        {"model": "strong"},
        {"messages": [{"role": "system", "content": "Be detailed."}, {"role": "user"}]},
        {"messages": [{"role": "user", "content": "What is HTTP?"}]},
        {"temperature": 0.1},
        {"max_tokens": 50},
        {"stop": ["\n"]},
        {"response_format": {"type": "json_object"}},
        {
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "x", "name": "alice"},
            ]
        },
    ],
)
def test_everything_else_the_model_sees_is_in_the_context(changed):
    assert context_hash(_asking("What is HTTP?", **changed)) != context_hash(_request())


@pytest.mark.parametrize("same", [{"stream": True}, {"user": "team-a"}, {"max_tokens": None}])
def test_fields_the_model_never_sees_arent(same):
    assert context_hash(_request(**same)) == context_hash(_request())


def test_the_context_hash_isnt_the_exact_key():
    # Different prefixes: one can never be mistaken for the other.
    assert context_hash(_request()) != cache_key(_request())


def test_entry_keys_differ_by_api_key_context_and_question():
    context = context_hash(_request())
    other = context_hash(_request(model="strong"))
    keys = {
        entry_key(1, context, "What is HTTP?"),
        entry_key(2, context, "What is HTTP?"),
        entry_key(1, other, "What is HTTP?"),
        entry_key(1, context, "What is HTTPS?"),
    }
    assert len(keys) == 4
    assert entry_key(1, context, "q") == entry_key(1, context, "q")
