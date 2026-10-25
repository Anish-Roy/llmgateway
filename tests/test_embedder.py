"""What the semantic cache embeds, the fake embedder tests use, and loading the model."""

import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient
from huggingface_hub import try_to_load_from_cache

import app.main
from app.api.schemas import ChatRequest
from app.cache.embedder import DIMENSIONS, MODEL_NAME, FakeEmbedder, MiniLMEmbedder, embed_text
from app.config import Settings
from app.main import create_app, start_embedder


def _request(messages, **extra) -> ChatRequest:
    return ChatRequest.model_validate({"model": "fast", "messages": messages, **extra})


USER = {"role": "user", "content": "What is HTTP?"}
SYSTEM = {"role": "system", "content": "Be brief."}


# embed_text: which requests are single-turn, and what of them is embedded.


@pytest.mark.parametrize(
    "messages",
    [
        [USER],
        [SYSTEM, USER],
        [{"role": "developer", "content": "Be brief."}, USER],
    ],
)
def test_single_turn_requests_embed_their_user_message(messages):
    assert embed_text(_request(messages)) == "What is HTTP?"


def test_outer_whitespace_is_dropped():
    assert embed_text(_request([{"role": "user", "content": "  What is HTTP?\n"}])) == (
        "What is HTTP?"
    )


@pytest.mark.parametrize(
    "messages",
    [
        # Earlier turns: the last message may only make sense after them ("and in Rust?").
        [USER, {"role": "assistant", "content": "A protocol."}, USER],
        [SYSTEM, USER, {"role": "assistant", "content": "A protocol."}, USER],
        # Two system messages, or one after the question.
        [SYSTEM, SYSTEM, USER],
        [USER, SYSTEM],
        # The last message isn't a question with text.
        [SYSTEM, {"role": "assistant", "content": "Hi"}],
        [{"role": "tool", "content": "42", "tool_call_id": "c1"}],
        [{"role": "user", "content": None}],
        [{"role": "user", "content": "   "}],
    ],
)
def test_requests_that_arent_single_turn_embed_nothing(messages):
    assert embed_text(_request(messages)) is None


@pytest.mark.parametrize(
    "extra",
    [
        {"tools": [{"type": "function", "function": {"name": "get_weather"}}]},
        {"functions": [{"name": "get_weather"}]},
    ],
)
def test_requests_with_tools_embed_nothing(extra):
    assert embed_text(_request([USER], **extra)) is None


# FakeEmbedder.


def test_fake_vectors_are_unit_length_float32_rows():
    vectors = FakeEmbedder().embed(["a", "b", "c"])
    assert vectors.shape == (3, DIMENSIONS)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, rtol=1e-6)


def test_fake_gives_a_text_the_same_vector_every_time():
    np.testing.assert_array_equal(FakeEmbedder().embed(["a"]), FakeEmbedder().embed(["a"]))


def test_fake_unlisted_texts_are_far_apart():
    a, b = FakeEmbedder().embed(["capital of France?", "capital of Germany?"])
    assert abs(float(a @ b)) < 0.3


def test_fake_listed_texts_get_their_vector_normalised():
    # cos(a, b) = 0.8 / 1.0 = 0.8, whatever the lengths given.
    first = [1.0] + [0.0] * (DIMENSIONS - 1)
    second = [0.8, 0.6] + [0.0] * (DIMENSIONS - 2)
    fake = FakeEmbedder({"a": [3 * x for x in first], "b": second})
    a, b = fake.embed(["a", "b"])
    assert float(a @ b) == pytest.approx(0.8)
    assert fake.calls == ["a", "b"]


def test_fake_rejects_a_vector_of_the_wrong_size():
    with pytest.raises(ValueError, match="384"):
        FakeEmbedder({"a": [1.0, 0.0]})


def test_fake_embeds_an_empty_list():
    assert FakeEmbedder().embed([]).shape == (0, DIMENSIONS)


# Loading at startup.


async def test_semantic_cache_off_never_loads_the_model(monkeypatch):
    def load():
        raise AssertionError("the model was loaded")

    monkeypatch.setattr(app.main, "load_embedder", load)
    assert await start_embedder(Settings(_env_file=None, semantic_cache_enabled=False)) is None


async def test_a_model_that_cant_load_turns_the_semantic_cache_off(monkeypatch, caplog):
    def load():
        raise OSError("no network to download the model")

    monkeypatch.setattr(app.main, "load_embedder", load)
    with caplog.at_level(logging.WARNING):
        assert await start_embedder(Settings(_env_file=None)) is None
    assert "semantic cache off" in caplog.text


async def test_the_loaded_model_is_returned():
    # conftest's loader: a FakeEmbedder.
    assert isinstance(await start_embedder(Settings(_env_file=None)), FakeEmbedder)


def test_startup_puts_the_embedder_on_the_app():
    app_ = create_app()
    assert app_.state.embedder is None
    with TestClient(app_):
        assert isinstance(app_.state.embedder, FakeEmbedder)


def test_a_model_that_cant_load_doesnt_stop_the_app(monkeypatch):
    def load():
        raise OSError("no network to download the model")

    monkeypatch.setattr(app.main, "load_embedder", load)
    app_ = create_app()
    with TestClient(app_) as client:
        assert app_.state.embedder is None
        assert client.get("/health").status_code == 200


# The real model, only where it's already downloaded: tests never fetch it.

_model_cached = isinstance(try_to_load_from_cache(MODEL_NAME, "config.json"), str)


@pytest.mark.skipif(not _model_cached, reason=f"{MODEL_NAME} isn't in the local Hugging Face cache")
def test_minilm_ranks_a_rephrasing_above_a_lookalike():
    # A near-duplicate and a trap pair from data/queries.jsonl.
    texts = ["What's the capital of France?", "capital of france?", "capital of Germany?"]
    question, rephrased, lookalike = MiniLMEmbedder().embed(texts)
    np.testing.assert_allclose(np.linalg.norm([question, rephrased], axis=1), 1, rtol=1e-5)
    assert float(question @ rephrased) > float(question @ lookalike)
