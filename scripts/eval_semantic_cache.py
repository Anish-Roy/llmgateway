"""The semantic cache's threshold experiment: how many repeated questions it catches, and how
many different questions it wrongly answers, at each similarity threshold.

Replays data/queries.jsonl, in file order, through a simulated cache. Each prompt is embedded
as the gateway would embed it (`embed_text`, then all-MiniLM-L6-v2) and checked against the
prompts already stored:
    - the same text as a stored prompt: an exact hit (the exact cache answers first);
    - otherwise, the most similar stored prompt at or above the threshold: a semantic hit;
    - otherwise a miss, and it's stored. Hits are not stored, as in the gateway, so a later
      prompt can't match a hit, only the prompt that first got the answer.

A semantic hit is right if both prompts are in the same meaning group (a prompt and its
labelled near-duplicates, `dup_of`), and a false hit if not. A trap hit is a false hit on a
labelled look-alike pair (`trap_of`): the cache's worst mistake, a confident wrong answer.

Two setups: MiniLM's similarity alone, and the gateway's, where the match must also hold the
same numbers as the new prompt (`same_numbers`). For each, at each threshold from 0.80 to 0.97:
    hit rate         semantic hits / prompts
    false-hit rate   false hits / semantic hits
    trap-hit rate    trap pairs hit / trap pairs
The best possible hit rate (every near-duplicate caught, nothing else) is drawn for reference.

The rule for choosing the threshold, set before the first run: the lowest threshold with no
false hits at it or at any threshold above it. Lowest, because it catches the most repeats;
"and above", so a lucky gap in the data (no false hits at one threshold, some just above it)
can't be the pick. A false hit costs a user a wrong answer; a miss costs one provider call.

The result: no threshold meets that rule in either setup, because cosine similarity can't see
word order ("string to integer" vs "integer to string": 0.996). So the semantic cache is
opt-in per request, at DEFAULT_THRESHOLD, with the measured false-hit rate documented
(docs/decisions.md). The chart marks that default.

The labels are the dataset's: a pair counted false may be a real rephrasing nobody labelled.
The script prints every false-hit pair so they can be read, not just counted.

uv run python -m scripts.eval_semantic_cache        # downloads the model on first use
"""

import argparse
import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.api.schemas import ChatRequest
from app.cache.embedder import DIMENSIONS, MiniLMEmbedder, embed_text
from app.cache.semantic import DEFAULT_THRESHOLD, same_numbers

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "queries.jsonl"
CHART = ROOT / "docs" / "semantic_threshold.png"
TABLE = ROOT / "docs" / "semantic_threshold.csv"
THRESHOLDS = [round(0.80 + step / 100, 2) for step in range(18)]  # 0.80 to 0.97


# A check a match must pass besides the threshold, given both prompts' text.
Guard = Callable[[str, str], bool]


@dataclass(frozen=True)
class Query:
    id: str
    prompt: str
    dup_of: str | None
    trap_of: str | None


@dataclass(frozen=True)
class Hit:
    """A prompt answered with the stored answer of `matched`."""

    query: str
    matched: str
    similarity: float


@dataclass(frozen=True)
class Replay:
    exact_hits: int
    hits: list[Hit]


@dataclass(frozen=True)
class Score:
    threshold: float
    queries: int
    exact_hits: int
    hits: list[Hit]
    false_hits: list[Hit]
    trap_hits: list[Hit]
    trap_pairs: int

    @property
    def hit_rate(self) -> float:
        return len(self.hits) / self.queries

    @property
    def false_hit_rate(self) -> float:
        return len(self.false_hits) / len(self.hits) if self.hits else 0.0

    @property
    def trap_hit_rate(self) -> float:
        return len(self.trap_hits) / self.trap_pairs if self.trap_pairs else 0.0


def load_queries(path: Path) -> list[Query]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [Query(r["id"], r["prompt"], r["dup_of"], r["trap_of"]) for r in rows]


def embed_texts(queries: list[Query]) -> list[str]:
    """What the gateway would embed for each prompt, sent as a single-turn request."""
    texts = []
    for q in queries:
        request = ChatRequest(
            model="eval", messages=[{"role": "user", "content": q.prompt}], temperature=0
        )
        text = embed_text(request)
        if text is None:
            raise ValueError(f"{q.id} isn't something the semantic cache would embed")
        texts.append(text)
    return texts


def meaning_groups(queries: list[Query]) -> dict[str, str]:
    """Each prompt's group: the id of the original its `dup_of` chain leads to (its own id if
    it's an original)."""
    dup_of = {q.id: q.dup_of for q in queries}

    def root(qid: str) -> str:
        seen = set()
        while dup_of.get(qid):
            if qid in seen:
                raise ValueError(f"dup_of loops through {qid}")
            seen.add(qid)
            qid = dup_of[qid]
        return qid

    return {q.id: root(q.id) for q in queries}


def trap_pairs(queries: list[Query]) -> set[frozenset[str]]:
    return {frozenset((q.id, q.trap_of)) for q in queries if q.trap_of}


def best_possible_hit_rate(queries: list[Query]) -> float:
    """Every prompt after the first of its meaning group is a hit, and nothing else is."""
    return (len(queries) - len(set(meaning_groups(queries).values()))) / len(queries)


def replay(
    queries: list[Query], vectors: np.ndarray, threshold: float, guard: Guard | None = None
) -> Replay:
    """Runs the prompts, in order, through a cache answering at `threshold`. `vectors` are
    unit-length rows, one per prompt. With a guard, the most similar stored prompt must also
    pass it (given the new prompt's text and the stored one's), or the prompt misses."""
    stored = np.empty((len(queries), DIMENSIONS), dtype=np.float32)
    stored_queries: list[Query] = []
    stored_texts: set[str] = set()
    exact_hits = 0
    hits = []
    for query, vector in zip(queries, vectors, strict=True):
        if query.prompt in stored_texts:
            exact_hits += 1
            continue
        if stored_queries:
            similarities = stored[: len(stored_queries)] @ vector
            best = int(np.argmax(similarities))
            match = stored_queries[best]
            if similarities[best] >= threshold and (
                guard is None or guard(query.prompt, match.prompt)
            ):
                hits.append(Hit(query.id, match.id, float(similarities[best])))
                continue
        stored[len(stored_queries)] = vector
        stored_queries.append(query)
        stored_texts.add(query.prompt)
    return Replay(exact_hits, hits)


def score(
    queries: list[Query], vectors: np.ndarray, threshold: float, guard: Guard | None = None
) -> Score:
    groups = meaning_groups(queries)
    traps = trap_pairs(queries)
    result = replay(queries, vectors, threshold, guard)
    false_hits = [h for h in result.hits if groups[h.query] != groups[h.matched]]
    return Score(
        threshold=threshold,
        queries=len(queries),
        exact_hits=result.exact_hits,
        hits=result.hits,
        false_hits=false_hits,
        trap_hits=[h for h in false_hits if frozenset((h.query, h.matched)) in traps],
        trap_pairs=len(traps),
    )


def pick(scores: list[Score]) -> Score | None:
    """The lowest threshold with no false hits at it or at any threshold above it; None if
    even the highest has false hits."""
    chosen = None
    for s in sorted(scores, key=lambda s: s.threshold, reverse=True):
        if s.false_hits:
            break
        chosen = s
    return chosen


# The two cache setups compared: MiniLM's similarity alone, and the gateway's, which also
# requires the same numbers in both prompts.
SETUPS: dict[str, Guard | None] = {"minilm": None, "minilm+numbers": same_numbers}


def write_table(results: dict[str, list[Score]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "setup",
                "threshold",
                "hits",
                "false_hits",
                "trap_hits",
                "hit_rate",
                "false_hit_rate",
                "trap_hit_rate",
            ]
        )
        for setup, scores in results.items():
            for s in scores:
                writer.writerow(
                    [
                        setup,
                        f"{s.threshold:.2f}",
                        len(s.hits),
                        len(s.false_hits),
                        len(s.trap_hits),
                        f"{s.hit_rate:.4f}",
                        f"{s.false_hit_rate:.4f}",
                        f"{s.trap_hit_rate:.4f}",
                    ]
                )


def plot(alone: list[Score], guarded: list[Score], best_possible: float, path: Path) -> None:
    # Only this script draws; the gateway and its tests never import matplotlib.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The dataviz reference palette, light mode. Colour is the setup, the same in both panels:
    # the gateway's in series-1 blue, MiniLM alone in muted grey for comparison.
    blue, ink, muted, grid, axis, surface = (
        "#2a78d6",
        "#52514e",
        "#898781",
        "#e1e0d9",
        "#c3c2b7",
        "#fcfcfb",
    )
    x = [s.threshold for s in guarded]
    chosen = next(s for s in guarded if s.threshold == DEFAULT_THRESHOLD)
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(8, 6.5), facecolor=surface)
    fig.suptitle(
        "Semantic cache: no threshold is free of wrong answers",
        color=ink,
        x=0.08,
        ha="left",
        fontsize=13,
    )
    setups = [
        (guarded, blue, "-", "o", "MiniLM + same numbers (the gateway)"),
        (alone, muted, "--", "o", "MiniLM alone"),
    ]
    for scores, color, ls, marker, label in setups:
        top.plot(
            x,
            [100 * s.hit_rate for s in scores],
            color=color,
            ls=ls,
            lw=2,
            marker=marker,
            ms=4,
            label=label,
        )
        bottom.plot(
            x,
            [100 * s.false_hit_rate for s in scores],
            color=color,
            ls=ls,
            lw=2,
            marker=marker,
            ms=4,
            label=label,
        )
    top.axhline(100 * best_possible, color=ink, lw=1, ls=":")
    top.annotate(
        f"best possible: every near-duplicate ({100 * best_possible:.1f}%)",
        xy=(x[-1], 100 * best_possible),
        xytext=(0, 5),
        textcoords="offset points",
        ha="right",
        color=ink,
        fontsize=8,
    )
    top.set_ylabel("Hit rate (% of prompts)", color=ink)
    top.set_ylim(0, 35)

    # The false-hit rate's denominator shrinks as the threshold rises: show the gateway's
    # counts, so a rise on few hits (3 of 13) isn't read as the cache getting worse.
    for s in guarded[::2]:
        bottom.annotate(
            f"{len(s.false_hits)}/{len(s.hits)}",
            xy=(s.threshold, 100 * s.false_hit_rate),
            xytext=(0, -13),
            textcoords="offset points",
            ha="center",
            color=muted,
            fontsize=7,
        )
    bottom.set_ylabel("False-hit rate (% of hits wrong)", color=ink)
    bottom.set_xlabel("MiniLM cosine similarity threshold", color=ink)
    bottom.set_ylim(0, 45)

    for ax in (top, bottom):
        ax.set_facecolor(surface)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(axis)
        ax.tick_params(colors=muted)
        ax.axvline(chosen.threshold, color=ink, lw=1, ls=":")
    # In the bottom panel's empty top band, where it crosses no line; it names both panels.
    bottom.legend(frameon=False, loc="upper center", labelcolor=ink, fontsize=9, ncols=2)
    top.annotate(
        f"default {chosen.threshold:.2f}, opt-in only:\n"
        f"{100 * chosen.hit_rate:.1f}% hit, {len(chosen.false_hits)} of {len(chosen.hits)} wrong",
        xy=(chosen.threshold, 100 * chosen.hit_rate),
        xytext=(chosen.threshold - 0.005, 21),
        textcoords="data",
        ha="right",
        color=ink,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": muted, "lw": 0.8},
    )
    bottom.set_xticks(x[::2])
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--chart", type=Path, default=CHART)
    parser.add_argument("--table", type=Path, default=TABLE)
    args = parser.parse_args()

    queries = load_queries(args.dataset)
    vectors = MiniLMEmbedder().embed(embed_texts(queries))
    results = {
        setup: [score(queries, vectors, t, guard) for t in THRESHOLDS]
        for setup, guard in SETUPS.items()
    }
    best_possible = best_possible_hit_rate(queries)
    prompts = {q.id: q.prompt for q in queries}
    traps = trap_pairs(queries)

    print(
        f"{len(queries)} prompts, {len(traps)} trap pairs, "
        f"exact hits {results['minilm'][0].exact_hits}"
    )
    print(f"best possible hit rate: {100 * best_possible:.1f}%")
    for setup, scores in results.items():
        chosen = pick(scores)
        print(f"\n{setup}\nthreshold  hits  false  traps  hit rate  false-hit rate  trap-hit rate")
        for s in scores:
            mark = "  <- rule" if s is chosen else ""
            mark += "  <- default" if s.threshold == DEFAULT_THRESHOLD else ""
            print(
                f"   {s.threshold:.2f}   {len(s.hits):4}  {len(s.false_hits):5}  "
                f"{len(s.trap_hits):5}  {100 * s.hit_rate:7.1f}%  "
                f"{100 * s.false_hit_rate:13.1f}%  {100 * s.trap_hit_rate:12.1f}%{mark}"
            )
        if chosen is None:
            print("no threshold up to 0.97 is free of false hits: the rule picks nothing")

        # A pair's similarity doesn't depend on the threshold: list each once.
        pairs = {(h.query, h.matched): h for s in scores for h in s.false_hits}
        print(f"\nevery false-hit pair, most similar first ({len(pairs)}):")
        for h in sorted(pairs.values(), key=lambda h: -h.similarity):
            label = "TRAP " if frozenset((h.query, h.matched)) in traps else "     "
            print(
                f"  {h.similarity:.3f} {label}{prompts[h.query]!r}\n"
                f"              -> {prompts[h.matched]!r}"
            )

    write_table(results, args.table)
    plot(results["minilm"], results["minilm+numbers"], best_possible, args.chart)
    print(f"\nwrote {args.table.relative_to(ROOT)} and {args.chart.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
