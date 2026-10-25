"""The `auto` router's offline check: how much of the query log it sends to the cheap alias,
and how often it gets a prompt's difficulty wrong.

Every prompt in data/queries.jsonl is labelled easy or hard. Each one is sent through the
router as a single-turn request, as `auto` would get it, and scored:
    to cheap         share of prompts the router sends to the cheap alias
    hard -> cheap    hard prompts sent to the cheap alias: the quality risk
    easy -> strong   easy prompts sent to the strong alias: money spent for nothing
    est. saving      cost saved against sending everything to the strong alias, if every
                     request used the same tokens on either one (Phase 11 measures it for real)

The rules can be tuned to fit this dataset, so its meaning groups (a prompt and its labelled
rephrasings, `dup_of`) are split about 70/30 by a hash of the group, and thresholds are chosen
on the train split and reported on the test split too. A rephrasing never lands on the other
side of its original. The split doesn't make the keyword list unbiased: it was written by
someone who had read the whole dataset. Read the test numbers as optimistic.

Also printed:
    - a sweep of `long_prompt_tokens`, the one threshold with a real trade-off;
    - the plan's first keyword list (write code, prove, step by step, analyze), with the same
      other rules, for comparison;
    - which rule decided, and the routing per category;
    - with --show-misses, every misrouted prompt and the rule that sent it there.

uv run python -m scripts.eval_router                 # tiktoken downloads its encoding once
"""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from app.api.schemas import ChatRequest
from app.config import Settings
from app.routing.auto import AutoRouter, Decision
from app.routing.registry import AutoSpec, RegistryConfig, load_registry
from app.usage import tokens
from scripts.eval_semantic_cache import DATASET, load_queries, meaning_groups

# The keywords in plan.md's first version of the rules.
PLAN_KEYWORDS = ["write code", "prove", "step by step", "analyze"]
SWEEP = [10, 15, 20, 25, 30, 40, 50, 60, 80, 100_000]
TEST_SHARE = 3  # in 10


@dataclass(frozen=True)
class Prompt:
    id: str
    text: str
    category: str
    hard: bool
    test: bool


@dataclass(frozen=True)
class Score:
    prompts: int
    to_cheap: int
    hard: int
    hard_to_cheap: int
    easy: int
    easy_to_strong: int

    @property
    def cheap_share(self) -> float:
        return self.to_cheap / self.prompts


def load_prompts(path: Path) -> list[Prompt]:
    groups = meaning_groups(load_queries(path))
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [
        Prompt(
            r["id"],
            r["prompt"],
            r["category"],
            r["difficulty"] == "hard",
            _in_test_split(groups[r["id"]]),
        )
        for r in rows
    ]


def _in_test_split(group: str) -> bool:
    return hashlib.sha256(group.encode()).digest()[0] % 10 < TEST_SHARE


def decide(router: AutoRouter, prompt: Prompt) -> Decision:
    request = ChatRequest(model="auto", messages=[{"role": "user", "content": prompt.text}])
    return router.decide(request)


def score(router: AutoRouter, prompts: list[Prompt]) -> Score:
    cheap = router.spec.cheap
    to_cheap = [p for p in prompts if decide(router, p).alias == cheap]
    hard = [p for p in prompts if p.hard]
    return Score(
        prompts=len(prompts),
        to_cheap=len(to_cheap),
        hard=len(hard),
        hard_to_cheap=sum(p.hard for p in to_cheap),
        easy=len(prompts) - len(hard),
        easy_to_strong=len(prompts) - len(hard) - sum(not p.hard for p in to_cheap),
    )


def saving_per_cheap_request(config: RegistryConfig, spec: AutoSpec) -> Decimal:
    """The share of a strong request's cost a cheap one saves, at the first target's list
    price, for a request with as many output tokens as input ones."""

    def price(alias: str) -> Decimal:
        p = config.price(config.aliases[alias][0])
        return p.input + p.output

    return 1 - price(spec.cheap) / price(spec.strong)


def row(label: str, s: Score, saving: Decimal) -> str:
    return (
        f"{label:<22} {s.prompts:>4}  {s.cheap_share:>6.1%}  "
        f"{s.hard_to_cheap:>3}/{s.hard:<3} {s.hard_to_cheap / max(s.hard, 1):>6.1%}  "
        f"{s.easy_to_strong:>3}/{s.easy:<3} {s.easy_to_strong / max(s.easy, 1):>6.1%}  "
        f"{s.cheap_share * float(saving):>6.1%}"
    )


HEADER = (
    f"{'':<22} {'n':>4}  {'cheap':>6}  {'hard -> cheap':>14}  {'easy -> strong':>14}  {'saving':>6}"
)


def report(config: RegistryConfig, spec: AutoSpec, prompts: list[Prompt], misses: bool) -> None:
    router = AutoRouter(spec)
    saving = saving_per_cheap_request(config, spec)
    train = [p for p in prompts if not p.test]
    test = [p for p in prompts if p.test]
    print(
        f"{len(prompts)} prompts ({sum(p.hard for p in prompts)} hard): {len(train)} train, "
        f"{len(test)} test. A cheap request saves {saving:.0%} of a strong one.\n"
    )

    print("Shipped rules (config/models.yaml)")
    print(HEADER)
    for label, subset in (("train", train), ("test", test), ("all", prompts)):
        print(row(label, score(router, subset), saving))

    print("\nPlan's first keywords, other rules as shipped")
    print(HEADER)
    baseline = AutoRouter(spec.model_copy(update={"keywords": PLAN_KEYWORDS}))
    for label, subset in (("train", train), ("test", test)):
        print(row(label, score(baseline, subset), saving))

    print("\nlong_prompt_tokens sweep, other rules as shipped (train | test)")
    print(f"{'tokens':<8}{HEADER[22:]}")
    for threshold in SWEEP:
        swept = AutoRouter(spec.model_copy(update={"long_prompt_tokens": threshold}))
        name = "off" if threshold == SWEEP[-1] else str(threshold)
        marker = " *" if threshold == spec.long_prompt_tokens else ""
        print(row(f"{name}{marker} train", score(swept, train), saving))
        print(row(f"{name}{marker} test", score(swept, test), saving))

    decisions = {p.id: decide(router, p) for p in prompts}
    print("\nWhich rule decided (all prompts)")
    reasons = Counter(d.reason.split(":")[0] for d in decisions.values())
    for reason, n in reasons.most_common():
        print(f"  {reason:<18} {n:>4}")

    print("\nTo cheap, per category and label (all prompts)")
    by_category: dict[tuple[str, bool], list[bool]] = {}
    for p in prompts:
        by_category.setdefault((p.category, p.hard), []).append(decisions[p.id].alias == spec.cheap)
    for (category, hard), cheap in sorted(by_category.items()):
        label = f"{category} ({'hard' if hard else 'easy'})"
        print(f"  {label:<24} {sum(cheap):>3}/{len(cheap):<3}")

    if misses:
        print("\nMisrouted prompts")
        for p in prompts:
            d = decisions[p.id]
            if p.hard == (d.alias == spec.cheap):
                kind = "hard -> cheap " if p.hard else "easy -> strong"
                split = "test " if p.test else "train"
                text = p.text.replace("\n", " / ")[:90]
                print(f"  {kind}  {split}  {d.reason:<22} {p.id}  {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--show-misses", action="store_true")
    args = parser.parse_args()

    config = load_registry(Settings(_env_file=None).models_file)
    if config.auto is None:
        raise SystemExit("config/models.yaml has no `auto` section")
    # The gateway counts with tiktoken once it has loaded; so does this.
    tokens.load_encoding()
    if tokens._encoding is None:
        raise SystemExit("tiktoken's encoding didn't load: counts would be chars/4")
    report(config, config.auto, load_prompts(args.dataset), args.show_misses)


if __name__ == "__main__":
    main()
