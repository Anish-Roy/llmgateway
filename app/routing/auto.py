"""`model: "auto"`: the gateway picks the cheap alias or the strong one for each request.

Rules, not a model: each one is cheap to check and says why it matched, so every decision can
be logged with its reason. The first rule that matches sends the request to the strong alias;
a request no rule matches goes to the cheap one. They read the request as the providers get
it, after PII masking, and only its last user message, apart from the turn count and
max_tokens: a long system prompt or earlier turns don't make a short question hard.

Decisions are deterministic, so `auto` shares cache entries with the alias it picks: the same
prompt always lands on the same alias, and its answers there are as good as asking for that
alias directly. The one exception: token counts use chars/4 until tiktoken has loaded (a few
seconds after startup), so a prompt right at `long_prompt_tokens` may go either way then.
"""

import re
from dataclasses import dataclass

from app.api.schemas import ChatRequest
from app.routing.registry import AutoSpec
from app.usage.tokens import count_tokens

CODE_FENCE = "```"
# Response headers on `auto` requests: the alias picked, and the rule that picked it.
ROUTE_HEADER = "x-gateway-route"
ROUTE_REASON_HEADER = "x-gateway-route-reason"


@dataclass(frozen=True)
class Decision:
    alias: str
    # Which rule matched: a fixed name, or "keyword:" and the configured keyword. Never text
    # from the request, so logging it can't leak what PII masking missed.
    reason: str


class AutoRouter:
    def __init__(self, spec: AutoSpec) -> None:
        self.spec = spec
        # One pattern for every keyword; the match tells which one it was.
        self._keywords: re.Pattern[str] | None = None
        if spec.keywords:
            alternatives = "|".join(
                r"\s+".join(re.escape(word) for word in keyword.split())
                for keyword in sorted(spec.keywords, key=len, reverse=True)
            )
            self._keywords = re.compile(rf"\b(?:{alternatives})\b", re.IGNORECASE)

    def decide(self, request: ChatRequest) -> Decision:
        spec = self.spec
        text = _last_user_text(request)
        reason = None
        if request.max_tokens is not None and request.max_tokens >= spec.long_answer_max_tokens:
            reason = "long_answer"
        elif sum(m.role == "user" for m in request.messages) >= spec.long_conversation_turns:
            reason = "long_conversation"
        elif CODE_FENCE in text:
            reason = "code_block"
        elif sum(bool(line.strip()) for line in text.splitlines()) >= spec.pasted_lines:
            reason = "pasted_text"
        elif count_tokens(text) >= spec.long_prompt_tokens:
            reason = "long_prompt"
        elif self._keywords is not None and (match := self._keywords.search(text)):
            # The keyword as configured (lowercased), not as the request spelled it.
            reason = f"keyword:{' '.join(match.group(0).lower().split())}"
        if reason is None:
            return Decision(spec.cheap, "default")
        return Decision(spec.strong, reason)


def _last_user_text(request: ChatRequest) -> str:
    for message in reversed(request.messages):
        if message.role == "user":
            return message.content or ""
    return ""
