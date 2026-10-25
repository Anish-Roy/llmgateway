"""Which requests the exact-match cache may answer, the key it stores them under, and which
answers it keeps.

The key is a SHA-256 of the whole request as the provider would get it, in canonical JSON. The
model alias, the messages and every sampling option the client sent (tools, stop, seed,
response_format...) are all in it: any of them can change the answer. Only fields that never
reach the model are left out, so a streamed and a plain request for the same thing share a key.

One cache serves every API key: an exact hit is the same answer to the same request.
"""

import hashlib
import json

from app.api.schemas import ChatRequest, ChatResponse

# A client sends `x-gateway-cache: off` to skip the cache, both reading and storing. The
# gateway's reply carries the same header: `exact` (a hit), `miss`, or `bypass` (not cacheable,
# or turned off).
CACHE_HEADER = "x-gateway-cache"

# A finished answer (`length`: cut at max_tokens, which is in the key). Not content filters or
# tool calls.
_STORABLE_FINISH = {"stop", "length"}

# How the request is delivered (stream, stream_options) or who sent it (user): never what the
# model sees, so never part of the key.
_NOT_IN_KEY = {"stream", "stream_options", "user"}

# Bump it when what goes into the key changes, so old entries stop matching instead of
# answering requests they weren't made for.
_KEY_VERSION = "v1"


def model_fields(request: ChatRequest) -> dict:
    """Every field of `request` the model sees, as plain JSON values. The semantic cache's
    context hash starts from the same fields, so the two caches agree on what matters."""
    # exclude_none: a field sent as null means the same as one left out. `temperature` is a
    # float field, so 0 and 0.0 dump the same.
    return request.model_dump(mode="json", exclude_none=True, exclude=_NOT_IN_KEY)


def canonical_json(fields: dict) -> str:
    """One spelling for equal values: keys sorted, no spaces."""
    return json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cache_key(request: ChatRequest) -> bytes:
    """32 bytes, the same for any two requests the model would see as identical."""
    canonical = canonical_json(model_fields(request))
    return hashlib.sha256(f"{_KEY_VERSION}:{canonical}".encode()).digest()


def cacheable(request: ChatRequest, max_temperature: float) -> bool:
    """True if one stored answer is a fair answer to every request like this one.

    Without a temperature the provider uses its own default (1.0 on most), so the answers vary
    and a stored one would hide that. Several choices (`n` > 1) are left to the provider too.
    """
    if request.temperature is None or request.temperature > max_temperature:
        return False
    n = (request.model_extra or {}).get("n")
    return n is None or n == 1


def opted_out(header: str | None) -> bool:
    """True if the client's `x-gateway-cache` header turns the cache off for this request."""
    return header is not None and header.strip().lower() == "off"


def storable(response: ChatResponse) -> bool:
    """True if `response` is a plain text answer worth serving again.

    One choice with text and nothing else. Tool calls are left out: a streamed answer is
    kept as its text alone, so a cached tool call could come back incomplete.
    """
    if len(response.choices) != 1:
        return False
    choice = response.choices[0]
    extra = choice.message.model_extra or {}
    return (
        bool(choice.message.content)
        and not extra.get("tool_calls")
        and choice.finish_reason in _STORABLE_FINISH
    )
