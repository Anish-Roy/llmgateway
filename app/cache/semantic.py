"""Which requests the semantic cache may answer, what they're matched on, and when a stored
answer to a similar question may be served: the similarity threshold, and the guard a match
must pass besides it. (Its stores and the route's front are in semantic_store.py.)

The threshold experiment (scripts/eval_semantic_cache.py) found no MiniLM threshold free of
wrong answers: cosine similarity can't see word order or a changed number. So the semantic
cache is opt-in per request, and what's measured is said plainly in docs/decisions.md.
"""

import hashlib
import re

from app.api.schemas import ChatRequest
from app.cache.key import canonical_json, model_fields

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


# The `x-gateway-cache` value that asks for the semantic cache, on top of the exact one. The
# reply says `semantic` on a semantic hit.
SEMANTIC = "semantic"

# Bump it when what goes into the context hash changes, so old entries stop matching.
_CONTEXT_VERSION = "v1"


def wants_semantic(header: str | None) -> bool:
    """True if the client's `x-gateway-cache` header opts this request in."""
    return header is not None and header.strip().lower() == SEMANTIC


def context_hash(request: ChatRequest) -> bytes:
    """32 bytes: everything the model sees except the user message's text. The alias, the
    system prompt and every option (max_tokens, stop, response_format...) all count, so a
    similar question asked in another context never gets this context's answer.

    Call it on single-turn requests (those `embed_text` accepts): the last message is the
    user message the cache embeds.
    """
    fields = model_fields(request)
    # Its role and any extra keys (`name`) still count.
    fields["messages"][-1].pop("content", None)
    canonical = canonical_json(fields)
    return hashlib.sha256(f"{_CONTEXT_VERSION}:{canonical}".encode()).digest()


def entry_key(key_id: int, context: bytes, prompt: str) -> bytes:
    """The row's primary key: the same question in the same context from the same API key
    replaces its row rather than adding a copy."""
    return hashlib.sha256(f"{key_id}:".encode() + context + prompt.encode()).digest()
