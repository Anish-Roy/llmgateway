"""When the semantic cache may serve a stored answer to a similar question: the similarity
threshold, and the guard a match must pass besides it.

The threshold experiment (scripts/eval_semantic_cache.py) found no MiniLM threshold free of
wrong answers: cosine similarity can't see word order or a changed number. So the semantic
cache is opt-in per request, and what's measured is said plainly in docs/decisions.md.
"""

import re

# MiniLM cosine similarity. At 0.95 with the guard below, the query log's replay hit 7.1% of
# prompts, and 4 of those 27 hits were wrong (docs/semantic_threshold.csv).
DEFAULT_THRESHOLD = 0.95

# Integers and decimals: "100", "1.10", the digits in "5x" or "port 8080".
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def same_numbers(a: str, b: str) -> bool:
    """True if both texts hold the same numbers, in any order.

    Two questions a digit apart ("$1.10" / "$1.20", "1 machine" / "100 machines") embed
    almost alike but want different answers. Order is ignored, so a rephrasing may move the
    numbers around, and so "100 miles to km" still matches "100 km to miles". Numbers written
    as words ("two") or with separators ("1,000" is 1 and 000) aren't understood.
    """
    return sorted(_NUMBER.findall(a)) == sorted(_NUMBER.findall(b))
