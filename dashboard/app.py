"""The cost dashboard: who spent what, and what the gateway saved.

    uv run --env-file .env --group dashboard streamlit run dashboard/app.py

Reads `request_logs` through `queries.py` and nothing else. DATABASE_URL comes from the
environment, or from Streamlit's secrets when deployed.
"""

import os
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

import altair as alt
import pandas as pd
import psycopg
import streamlit as st

import queries

# The reference categorical palette (light, dark), in its fixed order. A key keeps its slot
# whatever the filters show: slots go to keys in id order, among those that ever sent a
# request, so a new key takes the next slot and never repaints an older one. Keys past the
# last slot share "Other keys"; the per-key table lists them one by one.
SERIES = [
    ("#2a78d6", "#3987e5"),
    ("#eb6834", "#d95926"),
    ("#1baf7a", "#199e70"),
    ("#eda100", "#c98500"),
    ("#e87ba4", "#d55181"),
    ("#008300", "#008300"),
    ("#6250d6", "#9085e9"),
]
OTHER = ("#9a9893", "#6f6e69")
OTHER_KEYS = "Other keys"

# How long query results are reused before Postgres is asked again.
CACHE_TTL_S = 60

st.set_page_config(page_title="LLM Gateway costs", layout="wide")


def _database_url() -> str | None:
    if url := os.environ.get("DATABASE_URL"):
        return url
    try:
        return st.secrets.get("DATABASE_URL")
    except Exception:  # no secrets file at all
        return None


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def _keys(url: str) -> list[queries.Key]:
    with psycopg.connect(url) as conn:
        return queries.keys(conn)


@st.cache_data(ttl=CACHE_TTL_S, show_spinner="Reading the request log...")
def _load(url: str, f: queries.Filters) -> dict:
    with psycopg.connect(url) as conn:
        return {
            "summary": queries.summary(conn, f),
            "daily": queries.daily_spend(conn, f),
            "keys": queries.key_totals(conn, f),
            "mix": queries.model_mix(conn, f),
            "latency": queries.latency(conn, f),
            "routes": queries.routes(conn, f),
        }


def _frame(rows: list) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in rows])
    # Exact decimals in Postgres; charts want floats.
    for column in ("spend_usd", "saved_usd"):
        if column in df:
            df[column] = df[column].astype(float)
    return df


def _usd(value) -> str:
    value = float(value)
    return f"${value:,.2f}" if value >= 1 else f"${value:,.4f}"


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value:.1%}"


def _ms(value: float | None) -> str:
    return "–" if value is None else f"{value:,.0f} ms"


def _colors(keys: list[queries.Key], dark: bool) -> dict[str, str]:
    """Each key's label and its fixed color; every key past the palette maps to OTHER_KEYS."""
    side = 1 if dark else 0
    logged = sorted((k for k in keys if k.logged), key=lambda k: k.id)
    colors = {k.label: SERIES[i][side] for i, k in enumerate(logged[: len(SERIES)])}
    colors[OTHER_KEYS] = OTHER[side]
    return colors


def _bars(df: pd.DataFrame, x: alt.X, y: alt.Y, color: alt.Color, tooltip: list, **kw):
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=2, **kw)
        .encode(x=x, y=y, color=color, tooltip=tooltip)
        .properties(height=280)
    )


def _days(start, end) -> alt.X:
    """A day axis over the whole range, one band per day (`day` as an ISO date string), so a
    range with one busy day shows one bar among empty days, not one bar as wide as the chart."""
    days = [(start + timedelta(days=i)).isoformat() for i in range((end - start).days + 1)]
    return alt.X(
        "day:O",
        title=None,
        scale=alt.Scale(domain=days, paddingInner=0.2),
        axis=alt.Axis(labelExpr="utcFormat(toDate(datum.value), '%b %d')", labelAngle=0),
    )


# Filters, in one row above everything.

url = _database_url()
st.title("LLM Gateway costs")
if not url:
    st.error("Set DATABASE_URL (or add it to .streamlit/secrets.toml) to read the request log.")
    st.stop()

all_keys = _keys(url)
labels = {k.id: k.label + ("" if k.active else " · revoked") for k in all_keys}
today = datetime.now(UTC).date()

left, middle, right = st.columns([2, 4, 1], vertical_alignment="bottom")
picked = left.date_input(
    "Days (UTC)", value=(today - timedelta(days=29), today), max_value=today, format="YYYY-MM-DD"
)
chosen = middle.multiselect(
    "API keys", options=list(labels), format_func=labels.get, placeholder="All keys"
)
if right.button("Refresh", width="stretch"):
    st.cache_data.clear()
    st.rerun()

if not isinstance(picked, tuple) or len(picked) != 2:
    st.info("Pick the last day of the range too.")
    st.stop()
start, end = picked
f = queries.Filters(start=start, end=end, key_ids=tuple(chosen) or None)
data = _load(url, f)
s: queries.Summary = data["summary"]
dark = st.context.theme.type == "dark"
colors = _colors(all_keys, dark)

if not s.requests:
    st.info("No requests in this range.")
    st.stop()

# The headline: who spent what, and what the gateway saved.

tiles = st.columns(3) + st.columns(3)
tiles[0].metric("Spend", _usd(s.spend_usd), help="At list prices, free-tier calls included.")
tiles[1].metric(
    "Saved by the cache",
    _usd(s.saved_usd),
    help="What each cache hit's answer cost when a provider first gave it.",
)
tiles[2].metric("Requests", f"{s.requests:,}")
tiles[3].metric(
    "Cache hit rate",
    _pct(s.hit_rate),
    help="Hits over requests that got past rate limits and model checks. Requests the cache "
    "can't serve (temperature above the cache limit, or `x-gateway-cache: off`) count as "
    "misses, so this is a floor on the rate over cacheable requests.",
)
tiles[4].metric(
    "Fallback rate",
    _pct(s.fallback_rate),
    help="Requests a second provider answered, over requests that called any provider.",
)
tiles[5].metric(
    "Error rate",
    _pct(s.error_rate),
    help="Requests that a provider or the gateway failed, streams cut partway included, over "
    "requests the gateway let in. Rejections (429s, unknown models) are counted apart.",
)
notes = [f"{s.exact_hits:,} exact and {s.semantic_hits:,} semantic hits"]
if s.unpriced_hits:
    notes.append(
        f"{s.unpriced_hits:,} hits were logged before savings were recorded and aren't in "
        "“Saved by the cache”"
    )
st.caption(" · ".join(notes) + ".")

# Who spent what.

st.subheader("Spend per key")
daily = _frame(data["daily"])
daily["day"] = daily["day"].map(lambda d: d.isoformat())
daily["key"] = daily["label"].where(daily["label"].isin(colors), OTHER_KEYS)
by_day = daily.groupby(["day", "key"], as_index=False)[["requests", "spend_usd"]].sum()
shown = [label for label in colors if label in set(by_day["key"])]
key_color = alt.Color(
    "key:N",
    title=None,
    scale=alt.Scale(domain=shown, range=[colors[label] for label in shown]),
    sort=shown,
    legend=alt.Legend(orient="top", columns=3, labelLimit=280),
)
st.altair_chart(
    _bars(
        by_day,
        _days(start, end),
        alt.Y("spend_usd:Q", title="Spend (USD)", axis=alt.Axis(format="$~g")),
        key_color,
        [
            alt.Tooltip("day:N", title="Day"),
            alt.Tooltip("key:N", title="Key"),
            alt.Tooltip("spend_usd:Q", title="Spend", format="$.6f"),
            alt.Tooltip("requests:Q", title="Requests"),
        ],
    ),
    width="stretch",
)

totals = _frame(data["keys"])
st.dataframe(
    totals[["label", "requests", "spend_usd", "saved_usd", "hits"]],
    hide_index=True,
    width="stretch",
    column_config={
        "label": "Key",
        "requests": "Requests",
        "spend_usd": st.column_config.NumberColumn("Spend", format="$%.6f"),
        "saved_usd": st.column_config.NumberColumn("Saved by the cache", format="$%.6f"),
        "hits": "Cache hits",
    },
)

# What the gateway saved.

st.subheader("Spent and saved per day")
per_day = daily.groupby("day", as_index=False)[["spend_usd", "saved_usd"]].sum()
both = per_day.melt("day", var_name="what", value_name="usd")
both["what"] = both["what"].map({"spend_usd": "Spent", "saved_usd": "Saved by the cache"})
side = 1 if dark else 0
st.altair_chart(
    _bars(
        both,
        _days(start, end),
        alt.Y("usd:Q", title="USD", axis=alt.Axis(format="$~g")),
        alt.Color(
            "what:N",
            title=None,
            scale=alt.Scale(
                domain=["Spent", "Saved by the cache"], range=[SERIES[0][side], SERIES[2][side]]
            ),
            legend=alt.Legend(orient="top"),
        ),
        [
            alt.Tooltip("day:N", title="Day"),
            alt.Tooltip("what:N", title=" "),
            alt.Tooltip("usd:Q", title="USD", format="$.6f"),
        ],
    ).encode(xOffset="what:N"),
    width="stretch",
)

# Providers, failures and speed.

mix_col, outcome_col = st.columns([3, 2])
with mix_col:
    st.subheader("Who answered")
    mix = _frame(data["mix"])
    if mix.empty:
        st.caption("No provider was called in this range.")
    else:
        mix["target"] = mix["provider"] + " / " + mix["model"]
        st.altair_chart(
            _bars(
                mix,
                alt.X("requests:Q", title="Requests"),
                alt.Y("target:N", title=None, sort="-x", axis=alt.Axis(labelLimit=320)),
                alt.value(SERIES[0][side]),
                [
                    alt.Tooltip("target:N", title="Provider / model"),
                    alt.Tooltip("requests:Q", title="Requests"),
                    alt.Tooltip("fallbacks:Q", title="As a fallback"),
                    alt.Tooltip("spend_usd:Q", title="Spend", format="$.6f"),
                ],
            ).properties(height=60 + 32 * len(mix)),
            width="stretch",
        )
        st.caption(
            f"Cache hits aren't counted here: they called no provider. "
            f"{s.fallbacks:,} requests were answered by a fallback."
        )

with outcome_col:
    st.subheader("How requests ended")
    failed = s.provider_errors + s.gateway_errors + s.client_disconnects + s.rejected
    st.dataframe(
        pd.DataFrame(
            {
                "Outcome": [
                    "Completed",
                    "Provider failed",
                    "Gateway failed",
                    "Client left",
                    "Rejected (429, unknown model)",
                ],
                "Requests": [
                    s.requests - failed,
                    s.provider_errors,
                    s.gateway_errors,
                    s.client_disconnects,
                    s.rejected,
                ],
            }
        ),
        hide_index=True,
        width="stretch",
    )

st.subheader("Latency")
latency = {(r.cache_status, r.stream): r for r in data["latency"]}
miss, stream = latency.get(("miss", False)), latency.get(("miss", True))
hits = [r for (status, _), r in latency.items() if status != "miss"]
cells = st.columns(3)
cells[0].metric(
    "Miss, p50 / p95",
    f"{_ms(miss and miss.p50_ms)} / {_ms(miss and miss.p95_ms)}",
    help="Plain (non-streamed) replies a provider answered.",
)
cells[1].metric(
    "Stream first token, p50 / p95",
    f"{_ms(stream and stream.ttft_p50_ms)} / {_ms(stream and stream.ttft_p95_ms)}",
    help="Time to the first token of streams a provider answered.",
)
fastest = max(hits, key=lambda r: r.requests) if hits else None
cells[2].metric(
    "Cache hit, p50 / p95",
    f"{_ms(fastest and fastest.p50_ms)} / {_ms(fastest and fastest.p95_ms)}",
    help="The most common kind of hit; the table has each.",
)
with st.expander("Every latency group"):
    st.dataframe(
        _frame(data["latency"]),
        hide_index=True,
        width="stretch",
        column_config={
            "cache_status": "Cache",
            "stream": "Stream",
            "requests": "Requests",
            **{
                c: st.column_config.NumberColumn(c.replace("_ms", " (ms)"), format="%.0f")
                for c in ("p50_ms", "p95_ms", "ttft_p50_ms", "ttft_p95_ms")
            },
        },
    )

if data["routes"]:
    st.subheader("model: auto")
    st.dataframe(
        _frame(data["routes"]),
        hide_index=True,
        width="stretch",
        column_config={
            "alias": "Sent to",
            "reason": "Rule",
            "requests": "Requests",
            "spend_usd": st.column_config.NumberColumn("Spend", format="$%.6f"),
        },
    )

st.caption(
    f"Days are UTC, {start} to {end} included. Costs are list prices. "
    f"Results refresh every {CACHE_TTL_S} s."
)
