"""The guard a semantic match must pass besides the similarity threshold."""

import pytest

from app.cache.semantic import DEFAULT_THRESHOLD, same_numbers
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
