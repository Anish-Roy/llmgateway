"""PII masking: finds personal data in a request's messages and swaps it for placeholders.

Four kinds are found: emails, Indian mobile numbers, Aadhaar numbers (Verhoeff checksum) and
card numbers (Luhn checksum). The route masks every request before the caches and providers
see it, so the real values never reach a provider, a cache table or a log; the reply gets them
back on the way out, by `restore` for a whole reply and `StreamRestorer` for a stream.

Each value becomes a numbered placeholder, `[EMAIL_1]`, `[PHONE_2]`, numbered per kind in the
order first seen across all the messages. The same value always gets the same placeholder, so
two requests that differ only in their PII mask to the same request and share a cache entry.

A masked request also gets `PLACEHOLDER_NOTE` in its system message (`explain_placeholders`).
Without it, gpt-oss often treats a placeholder as a blank to fill, or as private data to
refuse, and the value is lost (docs/decisions.md has the measurements).

Only message `content` is scanned, and only reply text is restored: tool-call arguments and
other extra fields are left as they are, both ways.
"""

import bisect
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from app.api.schemas import ChatMessage, ChatRequest, ChatResponse

PiiType = Literal["email", "phone", "aadhaar", "card"]

# Verhoeff's tables: the dihedral group D5's multiplication (_D) and the position
# permutations (_P).
_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(digits: str) -> bool:
    """True if the last digit is the Verhoeff check digit of the others, as on Aadhaar."""
    check = 0
    for i, d in enumerate(reversed(digits)):
        check = _D[check][_P[i % 8][int(d)]]
    return check == 0


def luhn_valid(digits: str) -> bool:
    """True if the number passes the Luhn check every card number carries."""
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


# Every pattern refuses to start or end next to a digit, so none is found inside a longer
# number: an order ID's last ten digits aren't a phone. The email pattern only starts where a
# run of address characters does: tried from inside one, it would rescan the run from every
# character, and a long word (or base64) with no "@" would take quadratic time.
_EMAIL = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"
)
# Ten digits starting 6-9, written whole, 5+5 or 3+3+4, after +91, 0091, 91 or 0 if any.
_PHONE = re.compile(
    r"(?<!\d)(?:(?:\+|00)?91[ -]?|0)?(?:[6-9]\d{4}[ -]?\d{5}|[6-9]\d{2}[ -]?\d{3}[ -]?\d{4})(?!\d)"
)
# Twelve digits, whole or 4+4+4 with one kind of separator. UIDAI never issues one starting
# with 0 or 1.
_AADHAAR = re.compile(r"(?<!\d)[2-9]\d{3}([ -]?)\d{4}\1\d{4}(?!\d)")
# 13 to 19 digits, with single spaces or dashes anywhere between them. One random number in 10
# passes Luhn, so the first digit must be a network's too: 2 and 5 Mastercard, 3 Amex, JCB and
# Diners, 4 Visa, 6 Discover and RuPay, 8 RuPay. Millisecond timestamps (1...) aren't cards.
_CARD = re.compile(r"(?<!\d)[2-68](?:[ -]?\d){12,18}(?!\d)")


@dataclass(frozen=True)
class _Detector:
    type: PiiType
    pattern: re.Pattern[str]
    valid: Callable[[str], bool]
    # Two spellings of one value share a placeholder: "+91 98765 43210" is "9876543210".
    normal: Callable[[str], str]


# In priority order: where two matches overlap, the earlier detector's wins. An email can hold
# digits, and a checksum makes a card or an Aadhaar number surer than a phone, which has none.
_DETECTORS = [
    _Detector("email", _EMAIL, lambda _: True, str.casefold),
    _Detector("card", _CARD, lambda s: luhn_valid(_digits(s)), _digits),
    _Detector("aadhaar", _AADHAAR, lambda s: verhoeff_valid(_digits(s)), _digits),
    _Detector("phone", _PHONE, lambda _: True, lambda s: _digits(s)[-10:]),
]
_BY_TYPE = {d.type: d for d in _DETECTORS}


@dataclass(frozen=True)
class PiiMatch:
    type: PiiType
    start: int
    end: int


def find_pii(text: str) -> list[PiiMatch]:
    """Every piece of PII in `text`, in order, none overlapping."""
    # Sorted by start. The spans never overlap, so they're sorted by end too, and a new match
    # can only overlap its neighbours: checking those keeps a text full of matches fast.
    found: list[PiiMatch] = []
    for detector in _DETECTORS:
        for m in detector.pattern.finditer(text):
            if not detector.valid(m.group()):
                continue
            i = bisect.bisect_left(found, m.start(), key=lambda f: f.start)
            if i > 0 and found[i - 1].end > m.start():
                continue
            if i < len(found) and found[i].start < m.end():
                continue
            found.insert(i, PiiMatch(detector.type, m.start(), m.end()))
    return found


@dataclass(frozen=True)
class Masked:
    """A request with its PII swapped for placeholders, and what each placeholder stands for."""

    request: ChatRequest
    # "[EMAIL_1]" -> "priya@example.com": the spelling first seen.
    values: dict[str, str]

    @property
    def types(self) -> tuple[PiiType, ...]:
        """The kinds masked, sorted: what the request log records, never the values."""
        return tuple(sorted({_type_of(p) for p in self.values}))


def _type_of(placeholder: str) -> PiiType:
    return placeholder[1:].split("_")[0].lower()  # type: ignore[return-value]


class _Masker:
    """Hands out placeholders for one request, so numbering runs across its messages."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self._seen: dict[tuple[PiiType, str], str] = {}
        self._counts: dict[PiiType, int] = {}

    def mask(self, text: str) -> str:
        parts = []
        end = 0
        for m in find_pii(text):
            parts.append(text[end : m.start])
            parts.append(self._placeholder(m.type, text[m.start : m.end]))
            end = m.end
        parts.append(text[end:])
        return "".join(parts)

    def _placeholder(self, type: PiiType, value: str) -> str:
        key = (type, _BY_TYPE[type].normal(value))
        if key not in self._seen:
            self._counts[type] = self._counts.get(type, 0) + 1
            placeholder = f"[{type.upper()}_{self._counts[type]}]"
            self._seen[key] = placeholder
            self.values[placeholder] = value
        return self._seen[key]


def mask_request(request: ChatRequest) -> Masked:
    """`request` with the PII in every message's content masked. The same request object when
    there's none, and never a changed original."""
    masker = _Masker()
    messages = [
        m.model_copy(update={"content": masker.mask(m.content)}) if m.content else m
        for m in request.messages
    ]
    if not masker.values:
        return Masked(request, {})
    return Masked(request.model_copy(update={"messages": messages}), masker.values)


# The text measured live (scripts/check_pii_live.py): with it, every model tried returned
# every value; without it, gpt-oss-20b returned 13 to 18 of 22. Change it only after measuring.
PLACEHOLDER_NOTE = (
    "Text in square brackets such as [EMAIL_1] or [PHONE_1] stands for a real value the user "
    "gave, hidden for privacy. Treat it as that value, and wherever the value belongs in your "
    "reply, copy the placeholder exactly, brackets included."
)

_INSTRUCTION_ROLES = ("system", "developer")


def explain_placeholders(request: ChatRequest) -> ChatRequest:
    """`request` with `PLACEHOLDER_NOTE` ahead of its first system (or developer) message, or
    as a new system message first if it has none. Merged, not added: a second system message
    would take a single-turn request out of the semantic cache."""
    messages = list(request.messages)
    for i, m in enumerate(messages):
        if m.role in _INSTRUCTION_ROLES:
            content = f"{PLACEHOLDER_NOTE}\n\n{m.content}" if m.content else PLACEHOLDER_NOTE
            messages[i] = m.model_copy(update={"content": content})
            break
    else:
        messages.insert(0, ChatMessage(role="system", content=PLACEHOLDER_NOTE))
    return request.model_copy(update={"messages": messages})


# What a placeholder looks like. Only those in a request's `values` are restored: one the
# model made up, or "[EMAIL_7]" when there were two emails, stays as it is.
_PLACEHOLDER = re.compile(r"\[(?:EMAIL|PHONE|AADHAAR|CARD)_\d+\]")


def restore(text: str, values: dict[str, str]) -> str:
    """`text` with each of the request's placeholders back to the value it stands for."""
    if not values:
        return text
    return _PLACEHOLDER.sub(lambda m: values.get(m.group(), m.group()), text)


def restore_response(response: ChatResponse, values: dict[str, str]) -> ChatResponse:
    """A copy of `response` with the real values back in its text. Never the response itself:
    the caches may hold that object, and they must keep the placeholders."""
    if not values:
        return response
    restored = response.model_copy(deep=True)
    for choice in restored.choices:
        if choice.message.content:
            choice.message.content = restore(choice.message.content, values)
    return restored


class StreamRestorer:
    """Restores a stream's text one delta at a time.

    A placeholder may arrive split across deltas ("[EMA", "IL_1]"), so a tail that could
    still become one of the request's placeholders is held back until it either does or
    can't. Call `flush` at the end for whatever is still held.
    """

    def __init__(self, values: dict[str, str]) -> None:
        self._values = values
        self._held = ""

    def feed(self, text: str) -> str:
        """What can be sent now of everything fed so far."""
        if not self._values:
            return text
        text = self._held + text
        # A placeholder holds no "[", so only the last one can start an unfinished one.
        start = text.rfind("[")
        tail = text[start:] if start >= 0 else ""
        if tail and any(p.startswith(tail) and p != tail for p in self._values):
            self._held = tail
            text = text[:start]
        else:
            self._held = ""
        return restore(text, self._values)

    def flush(self) -> str:
        """The held tail, as it is: the stream ended before it became a placeholder."""
        held, self._held = self._held, ""
        return held
