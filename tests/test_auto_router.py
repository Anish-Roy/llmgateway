"""The `auto` router's rules, its config, and ModelRouter.resolve, without going through HTTP.

Tests count tokens at chars/4, rounded up (conftest leaves tiktoken unloaded): 20 tokens is
77 to 80 chars.
"""

import pytest
from pydantic import ValidationError

from app.api.schemas import ChatRequest
from app.config import Settings
from app.providers.mock import MockProvider
from app.routing.auto import AutoRouter, Decision
from app.routing.registry import AutoSpec, RegistryConfig, Target, load_registry
from app.routing.router import ModelRouter


def spec(**overrides) -> AutoSpec:
    fields = {
        "cheap": "fast",
        "strong": "strong",
        "long_answer_max_tokens": 2048,
        "long_conversation_turns": 3,
        "pasted_lines": 3,
        "long_prompt_tokens": 20,
        "keywords": ["prove", "step by step"],
    } | overrides
    return AutoSpec(**fields)


def request(*user: str, system: str | None = None, **extra) -> ChatRequest:
    """A conversation with these user messages, an assistant reply between each."""
    messages = [{"role": "system", "content": system}] if system else []
    for i, content in enumerate(user):
        if i:
            messages.append({"role": "assistant", "content": "ok"})
        messages.append({"role": "user", "content": content})
    return ChatRequest(model="auto", messages=messages, **extra)


def decide(*user: str, auto: AutoRouter | None = None, **extra) -> Decision:
    return (auto or AutoRouter(spec())).decide(request(*user, **extra))


def test_a_short_plain_question_goes_to_the_cheap_alias():
    assert decide("What is Docker?") == Decision("fast", "default")


def test_a_large_max_tokens_goes_to_the_strong_alias():
    assert decide("hi", max_tokens=2048) == Decision("strong", "long_answer")
    assert decide("hi", max_tokens=2047) == Decision("fast", "default")


def test_a_long_conversation_goes_to_the_strong_alias():
    assert decide("hi", "and?", "go on") == Decision("strong", "long_conversation")
    assert decide("hi", "and?") == Decision("fast", "default")


def test_a_code_fence_goes_to_the_strong_alias():
    assert decide("what does ```x = 1``` do") == Decision("strong", "code_block")


def test_pasted_lines_go_to_the_strong_alias():
    # Blank lines don't count.
    assert decide("why?\n\nx = 1\ny = 2") == Decision("strong", "pasted_text")
    assert decide("why?\n\n\nx = 1") == Decision("fast", "default")


def test_a_long_prompt_goes_to_the_strong_alias():
    assert decide("a" * 80) == Decision("strong", "long_prompt")
    assert decide("a" * 76) == Decision("fast", "default")


def test_a_keyword_goes_to_the_strong_alias_and_is_named():
    assert decide("Prove it") == Decision("strong", "keyword:prove")
    assert decide("go step by step") == Decision("strong", "keyword:step by step")


def test_keywords_match_any_case_and_spacing_but_only_whole_words():
    assert decide("go STEP\n by   Step") == Decision("strong", "keyword:step by step")
    assert decide("improve it") == Decision("fast", "default")
    assert decide("proven") == Decision("fast", "default")


def test_the_reason_never_holds_the_requests_own_text():
    auto = AutoRouter(spec(keywords=["Step By Step"]))

    # The keyword as configured, lowercased: never the request's spelling.
    assert decide("STEP BY STEP", auto=auto).reason == "keyword:step by step"


def test_a_keyword_with_regex_characters_is_matched_literally():
    auto = AutoRouter(spec(keywords=["trade-offs", "c++"]))

    assert decide("list the trade-offs", auto=auto).alias == "strong"
    assert decide("tradeXoffs", auto=auto).alias == "fast"


def test_only_the_last_user_message_counts():
    # A long system prompt or a long earlier turn doesn't make a short question hard.
    assert decide("hi", system="a" * 400) == Decision("fast", "default")
    assert decide("prove " + "a" * 100, "thanks") == Decision("fast", "default")


def test_rules_are_checked_in_order():
    # A request several rules match is named after the first.
    assert decide("prove " + "a" * 100, max_tokens=4096).reason == "long_answer"
    assert decide("prove " + "a" * 100).reason == "long_prompt"


def test_a_request_with_no_user_message_goes_to_the_cheap_alias():
    only_system = ChatRequest(model="auto", messages=[{"role": "system", "content": "hi"}])

    assert AutoRouter(spec()).decide(only_system) == Decision("fast", "default")


def test_no_keywords_means_no_keyword_rule():
    assert decide("prove it", auto=AutoRouter(spec(keywords=[]))) == Decision("fast", "default")


# The config.


def test_the_shipped_config_routes_easy_and_hard_prompts_apart():
    config = load_registry(Settings(_env_file=None).models_file)
    assert config.auto is not None
    auto = AutoRouter(config.auto)

    assert decide("What is Docker?", auto=auto).alias == config.auto.cheap
    assert decide("Prove that the square root of 2 is irrational.", auto=auto).alias == (
        config.auto.strong
    )


def _config(auto: dict, aliases: dict | None = None) -> RegistryConfig:
    return RegistryConfig.model_validate(
        {
            "providers": {"mock": {"type": "mock"}},
            "aliases": aliases or {"fast": ["mock/a"], "strong": ["mock/b"]},
            "auto": auto,
        }
    )


AUTO = spec().model_dump()


def test_auto_must_pick_between_defined_aliases():
    assert _config(AUTO).auto == spec()
    with pytest.raises(ValidationError, match="undefined alias 'strong'"):
        _config(AUTO, aliases={"fast": ["mock/a"]})


def test_auto_cant_be_an_alias_name():
    with pytest.raises(ValidationError, match="can't be an alias"):
        _config(AUTO, aliases={"fast": ["mock/a"], "strong": ["mock/b"], "auto": ["mock/a"]})


@pytest.mark.parametrize(
    "change",
    [
        {"long_promt_tokens": 20},  # a typo
        {"long_prompt_tokens": 0},
        {"pasted_lines": 1},
        {"long_conversation_turns": 1},
    ],
)
def test_a_bad_auto_section_is_rejected(change):
    with pytest.raises(ValidationError):
        _config(AUTO | change)


# ModelRouter.resolve and /v1/models.


def _router(auto: AutoRouter | None, providers=("a", "b")) -> ModelRouter:
    return ModelRouter(
        {
            "fast": [Target(provider="a", model="small")],
            "strong": [Target(provider="b", model="big")],
        },
        {name: MockProvider() for name in providers},
        auto=auto,
    )


def test_resolve_puts_the_picked_alias_in_the_request():
    router = _router(AutoRouter(spec()))

    resolved, decision = router.resolve(request("Prove it", temperature=0))

    assert decision == Decision("strong", "keyword:prove")
    assert resolved.model == "strong"
    assert resolved.temperature == 0


def test_resolve_leaves_other_requests_alone():
    router = _router(AutoRouter(spec()))
    fast = request("hi").model_copy(update={"model": "fast"})

    assert router.resolve(fast) == (fast, None)


def test_without_an_auto_router_auto_is_left_to_fail_as_unknown():
    auto = request("hi")

    assert _router(None).resolve(auto) == (auto, None)


def test_auto_is_listed_only_when_both_its_aliases_are_usable():
    assert _router(AutoRouter(spec())).available_aliases() == ["fast", "strong", "auto"]
    assert _router(AutoRouter(spec()), providers=("a",)).available_aliases() == ["fast"]
    assert _router(None).available_aliases() == ["fast", "strong"]
