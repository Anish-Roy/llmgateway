"""What the semantic cache embeds, and the model that embeds it.

Only single-turn requests are embedded for now: an optional system message and one user
message. `embed_text` decides that, and the threshold experiment (scripts/eval_semantic_cache.py)
calls it too, so the threshold is measured on exactly what the gateway embeds.
"""

import hashlib
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from app.api.schemas import ChatRequest

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSIONS = 384

# Both say how the model should behave; "developer" is OpenAI's newer name for "system".
_INSTRUCTION_ROLES = {"system", "developer"}

# Fields that make the answer depend on more than the text: a tool call, or a function call
# (the older API), is not something a similar question can share.
_NOT_SINGLE_TURN = {"tools", "functions"}


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """One row per text, `DIMENSIONS` wide, float32, each of length 1: cosine similarity
        is then a dot product. CPU-bound and blocking: callers on the event loop run it in a
        thread."""
        ...


def embed_text(request: ChatRequest) -> str | None:
    """The text the semantic cache embeds for `request`: its user message, or None if the
    request isn't single-turn.

    Single-turn is an optional system (or developer) message, then one user message with text,
    and no tools. Anything else (earlier turns, tool results, a second system message) changes
    what the answer means in ways one embedded message can't show, so it skips the cache.
    """
    if _NOT_SINGLE_TURN & (request.model_extra or {}).keys():
        return None
    *instructions, last = request.messages
    if len(instructions) > 1 or any(m.role not in _INSTRUCTION_ROLES for m in instructions):
        return None
    if last.role != "user" or not last.content or not last.content.strip():
        return None
    # MiniLM lowercases on its own; outer spaces would only be noise.
    return last.content.strip()


class MiniLMEmbedder:
    """all-MiniLM-L6-v2 on the CPU: about 90 MB, downloaded from Hugging Face on first use and
    cached there after."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        # Imported here: torch takes seconds to import, and only an app that embeds needs it.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device="cpu")
        dims = self._model.get_embedding_dimension()
        if dims != DIMENSIONS:
            raise ValueError(f"{model_name} makes {dims}-dimension vectors, not {DIMENSIONS}")

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32, copy=False)


def load_embedder() -> MiniLMEmbedder:
    """Loads the model and embeds one text, so the first request doesn't pay for the slow first
    call. Blocking, for seconds: run it in a thread."""
    embedder = MiniLMEmbedder()
    embedder.embed(["warm up"])
    return embedder


class FakeEmbedder:
    """An embedder for tests: no model, no download.

    A text in `vectors` gets that vector (normalised); any other text gets a random unit
    vector seeded by its hash, the same every time. Random 384-dimension vectors are nearly
    orthogonal, so two different unlisted texts are never similar.
    """

    def __init__(self, vectors: dict[str, Sequence[float]] | None = None) -> None:
        self.vectors = {
            text: _unit(np.asarray(v, dtype=np.float32)) for text, v in (vectors or {}).items()
        }
        self.calls: list[str] = []

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.extend(texts)
        if not texts:
            return np.empty((0, DIMENSIONS), dtype=np.float32)
        return np.stack([self._vector(t) for t in texts])

    def _vector(self, text: str) -> np.ndarray:
        if text in self.vectors:
            return self.vectors[text]
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8])
        return _unit(np.random.default_rng(seed).standard_normal(DIMENSIONS).astype(np.float32))


def _unit(vector: np.ndarray) -> np.ndarray:
    if vector.shape != (DIMENSIONS,):
        raise ValueError(f"expected a {DIMENSIONS}-dimension vector, got shape {vector.shape}")
    return vector / np.linalg.norm(vector)
