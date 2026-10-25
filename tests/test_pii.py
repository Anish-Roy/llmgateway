"""PII masking: the detectors, their checksums, and the placeholders a request is masked with."""

import time

import pytest

from app.api.schemas import ChatMessage, ChatRequest, ChatResponse, Choice
from app.cache.embedder import embed_text
from app.guardrails.pii import (
    PLACEHOLDER_NOTE,
    StreamRestorer,
    explain_placeholders,
    find_pii,
    luhn_valid,
    mask_request,
    restore,
    restore_response,
    verhoeff_valid,
)

# Pass their checksums: Verhoeff for the Aadhaar numbers, Luhn for the cards.
AADHAAR = "234567890124"
AADHAAR_2 = "498765432102"
VISA = "4111111111111111"
AMEX = "378282246310005"


def _types(text: str) -> list[tuple[str, str]]:
    return [(m.type, text[m.start : m.end]) for m in find_pii(text)]


def _request(*contents: str, **extra) -> ChatRequest:
    messages = [{"role": "user", "content": c} for c in contents]
    return ChatRequest.model_validate({"model": "fast", "messages": messages, **extra})


# The checksums.


def test_verhoeff_textbook_example():
    assert verhoeff_valid("2363")
    assert not verhoeff_valid("2364")


def test_verhoeff_on_aadhaar_length_numbers():
    assert verhoeff_valid(AADHAAR)
    assert verhoeff_valid(AADHAAR_2)
    assert not verhoeff_valid("234567890123")


def test_luhn():
    assert luhn_valid("79927398713")
    assert luhn_valid(VISA)
    assert luhn_valid(AMEX)
    assert not luhn_valid("4111111111111112")


# What's found.


@pytest.mark.parametrize(
    "text, value",
    [
        ("mail me at priya.sharma@example.co.in please", "priya.sharma@example.co.in"),
        ("RAHUL+test@Mail.com", "RAHUL+test@Mail.com"),
        ("<a_b-c@sub.domain.org>", "a_b-c@sub.domain.org"),
    ],
)
def test_finds_emails(text, value):
    assert _types(text) == [("email", value)]


@pytest.mark.parametrize(
    "value",
    [
        "9876543210",
        "98765 43210",
        "98765-43210",
        "987 654 3210",
        "+91 9876543210",
        "+91-98765-43210",
        "+919876543210",
        "919876543210",
        "09876543210",
        "6123456789",
    ],
)
def test_finds_indian_mobile_numbers(value):
    assert _types(f"call {value} today") == [("phone", value)]


@pytest.mark.parametrize("value", [AADHAAR, "2345 6789 0124", "2345-6789-0124"])
def test_finds_aadhaar_numbers(value):
    assert _types(f"my aadhaar is {value}.") == [("aadhaar", value)]


@pytest.mark.parametrize(
    "value", [VISA, "4111 1111 1111 1111", "4111-1111-1111-1111", AMEX, "3782 822463 10005"]
)
def test_finds_card_numbers(value):
    assert _types(f"card: {value}") == [("card", value)]


def test_finds_several_in_order():
    text = f"I'm a@b.com, {VISA} is my card, phone 9876543210, aadhaar {AADHAAR}"
    assert [m.type for m in find_pii(text)] == ["email", "card", "phone", "aadhaar"]


# What isn't: the done-when's negative cases.


@pytest.mark.parametrize(
    "text",
    [
        # A 12-digit order ID that fails Verhoeff is not an Aadhaar number, nor a phone.
        "order 234567890123 shipped",
        "order id 123456789012",
        # Aadhaar numbers never start with 0 or 1, even when the checksum passes.
        "ref 123456789010",
        # 16 digits that fail Luhn are not a card.
        "tracking 4111111111111112",
        "4111 1111 1111 1112",
        # Indian mobiles start with 6-9.
        "call 5876543210",
        "id 1234567890",
        # Too short or too long to be a phone.
        "pin 987654321",
        "id 98765432101",
        # A Unix timestamp in milliseconds (it passes Luhn, but no card starts with 1), and a
        # date-time.
        "at 1729843200000 sharp",
        "2026-10-25 14:30:00",
        # Not emails.
        "user@localhost",
        "the @ sign and a.b",
        # Plain numbers and versions.
        "port 8080, version 3.12.4, 100 miles",
    ],
)
def test_ignores_lookalikes(text):
    assert find_pii(text) == []


def test_ten_digit_order_id_starting_with_9_is_masked_as_a_phone():
    # A known false positive, kept on purpose: an Indian mobile has no checksum, so any ten
    # digits starting 6-9 look like one. Masking too much only costs the model some context.
    assert _types("order 9123456780 is late") == [("phone", "9123456780")]


def test_a_phone_isnt_found_inside_a_longer_number():
    assert find_pii("serial 1234987654321099") == []


def test_overlaps_go_to_the_stronger_detector():
    # The digits of an email, or a card's last ten, are never masked twice.
    assert _types("9876543210@example.com") == [("email", "9876543210@example.com")]
    assert _types("5555 9876 5432 1012") == []  # fails Luhn: not a card, and not a phone
    assert _types(f"x {VISA} y") == [("card", VISA)]


# Masking a request.


def test_masks_with_numbered_placeholders():
    masked = mask_request(_request(f"Email a@b.com or c@d.com, call 9876543210. {AADHAAR}"))
    assert masked.request.messages[0].content == (
        "Email [EMAIL_1] or [EMAIL_2], call [PHONE_1]. [AADHAAR_1]"
    )
    assert masked.values == {
        "[EMAIL_1]": "a@b.com",
        "[EMAIL_2]": "c@d.com",
        "[PHONE_1]": "9876543210",
        "[AADHAAR_1]": AADHAAR,
    }
    assert masked.types == ("aadhaar", "email", "phone")


def test_numbering_runs_across_messages():
    request = ChatRequest.model_validate(
        {
            "model": "fast",
            "messages": [
                {"role": "system", "content": "The admin is admin@corp.com."},
                {"role": "user", "content": "Forward it to me@home.org."},
            ],
        }
    )
    masked = mask_request(request)
    assert [m.content for m in masked.request.messages] == [
        "The admin is [EMAIL_1].",
        "Forward it to [EMAIL_2].",
    ]


def test_same_value_gets_the_same_placeholder():
    masked = mask_request(
        _request("call 98765 43210", "again: +91 9876543210 or A@B.com", "a@b.com")
    )
    assert [m.content for m in masked.request.messages] == [
        "call [PHONE_1]",
        "again: [PHONE_1] or [EMAIL_1]",
        "[EMAIL_1]",
    ]
    # The spelling first seen is the one restored.
    assert masked.values == {"[PHONE_1]": "98765 43210", "[EMAIL_1]": "A@B.com"}


def test_different_values_give_the_same_masked_request():
    # So two requests that differ only in their PII share a cache key.
    a = mask_request(_request("Write to ravi@x.com about 9876543210"))
    b = mask_request(_request("Write to neha@y.org about 7012345678"))
    assert a.request == b.request
    assert a.values != b.values


def test_nothing_found_returns_the_request_unchanged():
    request = _request("What is HTTP?")
    masked = mask_request(request)
    assert masked.request is request
    assert masked.values == {}
    assert masked.types == ()


def test_keeps_everything_but_content():
    request = ChatRequest.model_validate(
        {
            "model": "fast",
            "temperature": 0,
            "seed": 7,
            "messages": [
                {"role": "user", "content": "hi a@b.com", "name": "ravi"},
                {"role": "assistant", "content": None},
            ],
        }
    )
    masked = mask_request(request).request
    assert masked.model_dump(exclude={"messages"}) == request.model_dump(exclude={"messages"})
    assert masked.messages[0].model_extra == {"name": "ravi"}
    assert masked.messages[0].content == "hi [EMAIL_1]"
    assert masked.messages[1].content is None


def test_doesnt_change_the_original_request():
    request = _request("hi a@b.com")
    mask_request(request)
    assert request.messages[0].content == "hi a@b.com"


# Time: every request is scanned, so a long one must not take quadratic time.


@pytest.mark.parametrize(
    "text",
    [
        # A long word with no "@" once rescanned itself from every character.
        "a" * 200_000,
        "1" * 200_000,
        "a@" + "b." * 100_000,
        # Many matches once checked each against every earlier one.
        "a@b.com 9876543210 4111111111111111 234567890124 " * 4_000,
    ],
    ids=["letters", "digits", "dotted-domain", "many-matches"],
)
def test_long_texts_take_linear_time(text):
    started = time.perf_counter()
    find_pii(text)
    # About 0.03 s each; quadratic took seconds at this size.
    assert time.perf_counter() - started < 1


# Restoring.

VALUES = {"[EMAIL_1]": "a@b.com", "[EMAIL_2]": "c@d.org", "[PHONE_1]": "98765 43210"}


def test_restore_puts_the_values_back():
    text = "Write to [EMAIL_1], cc [EMAIL_2], or call [PHONE_1]. [EMAIL_1] again."
    assert restore(text, VALUES) == (
        "Write to a@b.com, cc c@d.org, or call 98765 43210. a@b.com again."
    )


def test_restore_leaves_unknown_placeholders_and_lookalikes():
    text = "[EMAIL_3] [PHONE_2] [CARD_1] [EMAIL_1 ] [email_1] [EMAIL_]"
    assert restore(text, VALUES) == text


def test_restore_without_values_is_a_no_op():
    assert restore("[EMAIL_1]", {}) == "[EMAIL_1]"


def test_mask_then_restore_round_trips():
    original = f"Mail priya@x.in, call +91 98765 43210, card {VISA}, aadhaar {AADHAAR}."
    masked = mask_request(_request(original))
    assert restore(masked.request.messages[0].content, masked.values) == original


def _response(*contents: str | None) -> ChatResponse:
    return ChatResponse(
        model="m",
        choices=[
            Choice(index=i, message=ChatMessage(role="assistant", content=c))
            for i, c in enumerate(contents)
        ],
    )


def test_restore_response_returns_a_restored_copy():
    stored = _response("Sent to [EMAIL_1].", None)
    restored = restore_response(stored, VALUES)
    assert [c.message.content for c in restored.choices] == ["Sent to a@b.com.", None]
    # The original, which a cache may hold, keeps its placeholders.
    assert stored.choices[0].message.content == "Sent to [EMAIL_1]."
    assert restored.id == stored.id


def test_restore_response_without_values_is_the_same_response():
    stored = _response("hi")
    assert restore_response(stored, {}) is stored


def _streamed(restorer: StreamRestorer, deltas: list[str]) -> list[str]:
    return [restorer.feed(d) for d in deltas] + [restorer.flush()]


STREAMED = "To [EMAIL_1] and [PHONE_1]: hi [EMAIL_2]! [x] [EMAIL_9] [EMA"


@pytest.mark.parametrize("cut", range(len(STREAMED) + 1))
def test_a_stream_split_anywhere_restores_like_the_whole_text(cut):
    sent = _streamed(StreamRestorer(VALUES), [STREAMED[:cut], STREAMED[cut:]])
    assert "".join(sent) == restore(STREAMED, VALUES)


def test_a_stream_a_character_at_a_time_restores_like_the_whole_text():
    sent = _streamed(StreamRestorer(VALUES), list(STREAMED))
    assert "".join(sent) == restore(STREAMED, VALUES)
    # Nothing leaks half a placeholder: each one arrives whole.
    assert "a@b.com" in sent and "98765 43210" in sent


def test_a_split_placeholder_is_held_until_it_closes():
    restorer = StreamRestorer(VALUES)
    assert restorer.feed("Mail [EM") == "Mail "
    assert restorer.feed("AIL_") == ""
    assert restorer.feed("1] now") == "a@b.com now"


def test_text_that_cant_become_a_placeholder_isnt_held():
    restorer = StreamRestorer(VALUES)
    assert restorer.feed("see [x") == "see [x"
    assert restorer.feed("list[0") == "list[0"
    # [EMAIL_9] isn't one of this request's.
    assert restorer.feed("[EMAIL_9") == "[EMAIL_9"


def test_an_unfinished_tail_is_flushed_as_it_is():
    restorer = StreamRestorer(VALUES)
    assert restorer.feed("cut [EMAIL_") == "cut "
    assert restorer.flush() == "[EMAIL_"
    assert restorer.flush() == ""


def test_a_restorer_without_values_passes_text_through():
    restorer = StreamRestorer({})
    assert restorer.feed("[EMA") == "[EMA"
    assert restorer.flush() == ""


# The note that tells the model what the placeholders are.


def test_the_note_becomes_the_system_message_when_there_is_none():
    request = explain_placeholders(_request("hi [EMAIL_1]"))
    assert [(m.role, m.content) for m in request.messages] == [
        ("system", PLACEHOLDER_NOTE),
        ("user", "hi [EMAIL_1]"),
    ]


@pytest.mark.parametrize("role", ["system", "developer"])
def test_the_note_joins_the_clients_system_message(role):
    request = ChatRequest.model_validate(
        {
            "model": "fast",
            "messages": [
                {"role": role, "content": "Be brief.", "name": "ops"},
                {"role": "user", "content": "hi [EMAIL_1]"},
            ],
        }
    )
    explained = explain_placeholders(request)
    first, user = explained.messages
    assert (first.role, first.content) == (role, f"{PLACEHOLDER_NOTE}\n\nBe brief.")
    assert first.model_extra == {"name": "ops"}
    assert user.content == "hi [EMAIL_1]"
    # The original is left as it was.
    assert request.messages[0].content == "Be brief."


def test_the_note_joins_only_the_first_system_message():
    request = ChatRequest.model_validate(
        {
            "model": "fast",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "A"},
                {"role": "system", "content": "B"},
            ],
        }
    )
    contents = [m.content for m in explain_placeholders(request).messages]
    assert contents == ["hi", f"{PLACEHOLDER_NOTE}\n\nA", "B"]


def test_the_note_fills_an_empty_system_message():
    request = ChatRequest.model_validate(
        {
            "model": "fast",
            "messages": [{"role": "system", "content": ""}, {"role": "user", "content": "x"}],
        }
    )
    assert explain_placeholders(request).messages[0].content == PLACEHOLDER_NOTE


def test_the_note_keeps_a_single_turn_request_single_turn():
    with_system = ChatRequest.model_validate(
        {
            "model": "fast",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hi [EMAIL_1]"},
            ],
        }
    )
    for request in (_request("hi [EMAIL_1]"), with_system):
        assert embed_text(explain_placeholders(request)) == "hi [EMAIL_1]"
