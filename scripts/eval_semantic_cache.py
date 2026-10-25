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

For each threshold from 0.80 to 0.97:
    hit rate         semantic hits / prompts
    false-hit rate   false hits / semantic hits
    trap-hit rate    trap pairs hit / trap pairs
The best possible hit rate (every near-duplicate caught, nothing else) is drawn for reference.

The rule for choosing the threshold, set before the first run: the lowest threshold with no
false hits at it or at any threshold above it. Lowest, because it catches the most repeats;
"and above", so a lucky gap in the data (no false hits at one threshold, some just above it)
can't be the pick. A false hit costs a user a wrong answer; a miss costs one provider call.

The labels are the dataset's: a pair counted false may be a real rephrasing nobody labelled.
The script prints every false-hit pair so they can be read, not just counted.

uv run python -m scripts.eval_semantic_cache        # downloads the model on first use
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.api.schemas import ChatRequest
from app.cache.embedder import DIMENSIONS, MiniLMEmbedder, embed_text

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "queries.jsonl"
CHART = ROOT / "docs" / "semantic_threshold.png"
TABLE = ROOT / "docs" / "semantic_threshold.csv"
THRESHOLDS = [round(0.80 + step / 100, 2) for step in range(18)]  # 0.80 to 0.97


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


def replay(queries: list[Query], vectors: np.ndarray, threshold: float) -> Replay:
    """Runs the prompts, in order, through a cache answering at `threshold`. `vectors` are
    unit-length rows, one per prompt."""
    stored = np.empty((len(queries), DIMENSIONS), dtype=np.float32)
    stored_ids: list[str] = []
    stored_texts: set[str] = set()
    exact_hits = 0
    hits = []
    for query, vector in zip(queries, vectors, strict=True):
        if query.prompt in stored_texts:
            exact_hits += 1
            continue
        if stored_ids:
            similarities = stored[: len(stored_ids)] @ vector
            best = int(np.argmax(similarities))
            if similarities[best] >= threshold:
                hits.append(Hit(query.id, stored_ids[best], float(similarities[best])))
                continue
        stored[len(stored_ids)] = vector
        stored_ids.append(query.id)
        stored_texts.add(query.prompt)
    return Replay(exact_hits, hits)


def score(queries: list[Query], vectors: np.ndarray, threshold: float) -> Score:
    groups = meaning_groups(queries)
    traps = trap_pairs(queries)
    result = replay(queries, vectors, threshold)
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


def write_table(scores: list[Score], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "threshold",
                "hits",
                "false_hits",
                "trap_hits",
                "hit_rate",
                "false_hit_rate",
                "trap_hit_rate",
            ]
        )
        for s in scores:
            writer.writerow(
                [
                    f"{s.threshold:.2f}",
                    len(s.hits),
                    len(s.false_hits),
                    len(s.trap_hits),
                    f"{s.hit_rate:.4f}",
                    f"{s.false_hit_rate:.4f}",
                    f"{s.trap_hit_rate:.4f}",
                ]
            )


def plot(scores: list[Score], chosen: Score | None, best_possible: float, path: Path) -> None:
    # Only this script draws; the gateway and its tests never import matplotlib.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The dataviz reference palette, light mode: series 1 to 3, then ink and hairlines.
    blue, orange, aqua, ink, muted, grid, surface = (
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#52514e",
        "#898781",
        "#e1e0d9",
        "#fcfcfb",
    )
    x = [s.threshold for s in scores]
    fig, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 6.5), height_ratios=[1, 1], facecolor=surface
    )
    fig.suptitle(
        "Semantic cache: choosing the similarity threshold",
        color=ink,
        x=0.08,
        ha="left",
        fontsize=13,
    )

    top.plot(
        x,
        [100 * s.hit_rate for s in scores],
        color=blue,
        lw=2,
        marker="o",
        ms=4,
        label="Hit rate (% of prompts)",
    )
    top.axhline(
        100 * best_possible,
        color=muted,
        lw=1.5,
        ls="--",
        label=f"Best possible: every near-duplicate ({100 * best_possible:.1f}%)",
    )
    top.set_ylabel("% of prompts", color=ink)
    top.set_ylim(0, max(40, 100 * max(s.hit_rate for s in scores) + 5))

    bottom.plot(
        x,
        [100 * s.false_hit_rate for s in scores],
        color=orange,
        lw=2,
        marker="o",
        ms=4,
        label="False-hit rate (% of hits that were wrong; false/hits)",
    )
    bottom.plot(
        x,
        [100 * s.trap_hit_rate for s in scores],
        color=aqua,
        lw=2,
        marker="s",
        ms=4,
        label="Trap pairs hit (% of look-alike pairs)",
    )
    bottom.set_ylabel("%", color=ink)
    bottom.set_xlabel("Cosine similarity threshold", color=ink)
    bottom.set_ylim(0, max(60, 100 * max(s.trap_hit_rate for s in scores) + 10))
    # The false-hit rate's denominator shrinks as the threshold rises: show the counts, so a
    # rise on few hits (5 of 16) isn't read as the cache getting worse.
    for s in scores[::2]:
        bottom.annotate(
            f"{len(s.false_hits)}/{len(s.hits)}",
            xy=(s.threshold, 100 * s.false_hit_rate),
            xytext=(0, -13),
            textcoords="offset points",
            ha="center",
            color=muted,
            fontsize=7,
        )

    for ax in (top, bottom):
        ax.set_facecolor(surface)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        ax.tick_params(colors=muted)
        if chosen is not None:
            ax.axvline(chosen.threshold, color=ink, lw=1, ls=":")
        ax.legend(frameon=False, loc="upper right", labelcolor=ink, fontsize=9)
    if chosen is not None:
        top.annotate(
            f"chosen: {chosen.threshold:.2f}\n{100 * chosen.hit_rate:.1f}% hit, 0 false hits",
            xy=(chosen.threshold, 100 * chosen.hit_rate),
            xytext=(8, -34),
            textcoords="offset points",
            color=ink,
            fontsize=9,
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
    scores = [score(queries, vectors, t) for t in THRESHOLDS]
    chosen = pick(scores)
    best_possible = best_possible_hit_rate(queries)
    prompts = {q.id: q.prompt for q in queries}
    traps = trap_pairs(queries)

    print(f"{len(queries)} prompts, {len(traps)} trap pairs, exact hits {scores[0].exact_hits}")
    print(f"best possible hit rate: {100 * best_possible:.1f}%\n")
    print("threshold  hits  false  traps  hit rate  false-hit rate  trap-hit rate")
    for s in scores:
        mark = "  <- chosen" if s is chosen else ""
        print(
            f"   {s.threshold:.2f}   {len(s.hits):4}  {len(s.false_hits):5}  "
            f"{len(s.trap_hits):5}  {100 * s.hit_rate:7.1f}%  {100 * s.false_hit_rate:13.1f}%  "
            f"{100 * s.trap_hit_rate:12.1f}%{mark}"
        )

    # A pair's similarity doesn't depend on the threshold: list each once.
    pairs = {(h.query, h.matched): h for s in scores for h in s.false_hits}
    print(f"\nevery false-hit pair, most similar first ({len(pairs)}):")
    for h in sorted(pairs.values(), key=lambda h: -h.similarity):
        label = "TRAP " if frozenset((h.query, h.matched)) in traps else "     "
        print(
            f"  {h.similarity:.3f} {label}{prompts[h.query]!r}\n"
            f"              -> {prompts[h.matched]!r}"
        )

    write_table(scores, args.table)
    plot(scores, chosen, best_possible, args.chart)
    print(f"\nwrote {args.table.relative_to(ROOT)} and {args.chart.relative_to(ROOT)}")
    if chosen is None:
        print("no threshold up to 0.97 is free of false hits: the rule picks nothing")
    else:
        print(f"chosen threshold: {chosen.threshold:.2f} ({100 * chosen.hit_rate:.1f}% hit rate)")


if __name__ == "__main__":
    main()
