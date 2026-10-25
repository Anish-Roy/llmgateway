"""The exact-match cache's key, which requests it may answer and which answers it keeps."""

import pytest

from app.api.schemas import ChatRequest, ChatResponse
from app.cache.key import cache_key, cacheable, opted_out, storable


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


def test_key_is_a_sha256_digest():
    assert len(cache_key(_request())) == 32


def test_same_request_same_key():
    assert cache_key(_request()) == cache_key(_request())


@pytest.mark.parametrize(
    "same",
    [
        # Delivery and identity fields never reach the model.
        {"stream": True},
        {"stream": True, "stream_options": {"include_usage": True}},
        {"user": "team-a"},
        # 0 and 0.0 are the same temperature; a null is the same as leaving a field out.
        {"temperature": 0.0},
        {"max_tokens": None},
    ],
)
def test_fields_the_model_never_sees_dont_change_the_key(same):
    assert cache_key(_request(**same)) == cache_key(_request())


def test_key_order_in_extra_fields_doesnt_matter():
    a = _request(response_format={"type": "json_object", "strict": True})
    b = _request(response_format={"strict": True, "type": "json_object"})
    assert cache_key(a) == cache_key(b)


@pytest.mark.parametrize(
    "different",
    [
        {"model": "strong"},
        {"temperature": 0.1},
        {"max_tokens": 50},
        {"messages": [{"role": "user", "content": "What is HTTP?"}]},
        {
            "messages": [
                {"role": "system", "content": "Be detailed."},
                {"role": "user", "content": "What is HTTP?"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "What is HTTP?", "name": "alice"},
            ]
        },
        # Fields the gateway doesn't model are forwarded to the provider, so they count too.
        {"tools": [{"type": "function", "function": {"name": "lookup"}}]},
        {"response_format": {"type": "json_object"}},
        {"stop": ["\n"]},
        {"seed": 7},
        {"top_p": 0.5},
    ],
)
def test_anything_the_model_sees_changes_the_key(different):
    assert cache_key(_request(**different)) != cache_key(_request())


def test_message_order_changes_the_key():
    turns = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Bye"},
    ]
    swapped = [turns[0], turns[2], turns[1]]
    assert cache_key(_request(messages=turns)) != cache_key(_request(messages=swapped))


@pytest.mark.parametrize("temperature", [0, 0.1, 0.2])
def test_low_temperature_is_cacheable(temperature):
    assert cacheable(_request(temperature=temperature), max_temperature=0.2)


@pytest.mark.parametrize("temperature", [0.21, 0.7, 1.0, 2.0])
def test_higher_temperature_is_not(temperature):
    assert not cacheable(_request(temperature=temperature), max_temperature=0.2)


def test_no_temperature_is_not_cacheable():
    # The provider's default (1.0 on most) applies, so the answers vary.
    assert not cacheable(_request(temperature=None), max_temperature=0.2)


def test_max_temperature_zero_caches_only_greedy_requests():
    assert cacheable(_request(temperature=0), max_temperature=0)
    assert not cacheable(_request(temperature=0.01), max_temperature=0)


@pytest.mark.parametrize(("n", "expected"), [(1, True), (2, False), (5, False)])
def test_only_one_choice_is_cacheable(n, expected):
    assert cacheable(_request(n=n), max_temperature=0.2) is expected


@pytest.mark.parametrize("header", ["off", "OFF", " Off "])
def test_header_off_opts_out(header):
    assert opted_out(header)


@pytest.mark.parametrize("header", [None, "", "on", "exact", "no"])
def test_anything_else_keeps_the_cache(header):
    assert not opted_out(header)


def _reply(**choice) -> ChatResponse:
    fields = {"message": {"role": "assistant", "content": "HTTP is a protocol."}} | choice
    return ChatResponse.model_validate({"model": "m", "choices": [fields]})


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_a_finished_text_answer_is_storable(finish_reason):
    assert storable(_reply(finish_reason=finish_reason))


@pytest.mark.parametrize(
    "reply",
    [
        _reply(message={"role": "assistant", "content": ""}),
        _reply(message={"role": "assistant", "content": None}),
        _reply(
            message={
                "role": "assistant",
                "content": "Looking it up.",
                "tool_calls": [{"id": "call_1", "type": "function"}],
            }
        ),
        _reply(finish_reason="content_filter"),
        _reply(finish_reason=None),
        ChatResponse.model_validate({"model": "m", "choices": []}),
    ],
)
def test_other_answers_arent(reply):
    assert not storable(reply)


def test_several_choices_arent_storable():
    reply = _reply()
    reply.choices.append(reply.choices[0].model_copy(update={"index": 1}))
    assert not storable(reply)
