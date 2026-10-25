"""Token estimates for streams that ended before the provider reported usage.

Only a fallback. Provider counts are always preferred: they're what gets billed. These
estimates use OpenAI's o200k_base encoding, which is close for gpt-oss but only approximate
for Gemini. They count visible text only, so hidden reasoning or thinking tokens (billed, but
never streamed) are missing: treat the estimate as a lower bound.
"""

import logging
import math
import threading

import tiktoken

from app.api.schemas import ChatMessage, Usage

logger = logging.getLogger(__name__)

ENCODING = "o200k_base"
_encoding: tiktoken.Encoding | None = None


def load_encoding() -> None:
    """Loads the encoding. The first time, tiktoken downloads it (~18 s on 2026-10-25), so
    this runs in a background thread at startup. Until it's loaded, or if the download fails,
    counts fall back to about 4 characters a token."""
    global _encoding
    try:
        _encoding = tiktoken.get_encoding(ENCODING)
    except Exception:
        logger.warning("could not load tiktoken %s: token estimates use chars/4", ENCODING)


def load_encoding_in_background() -> None:
    # A daemon thread, so a slow first download never holds up startup or shutdown.
    threading.Thread(target=load_encoding, name="tiktoken-load", daemon=True).start()


def count_tokens(text: str) -> int:
    if _encoding is None:
        return math.ceil(len(text) / 4)
    # disallowed_special=(): model output may contain "<|endoftext|>" as plain text.
    return len(_encoding.encode(text, disallowed_special=()))


def estimate_usage(messages: list[ChatMessage], completion: str) -> Usage:
    prompt_tokens = sum(count_tokens(m.content or "") for m in messages)
    completion_tokens = count_tokens(completion)
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
