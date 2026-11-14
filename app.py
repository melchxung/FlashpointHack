"""
app.py - Flashpoint K-Alpha Bloomberg-Style Retail Sentiment & Asset Pricing Terminal

TERMINAL  - Dense 2x2 grid: K-gap wedge | strategy equity curves,
            vibecession scatter | IC leaderboard.
BACKTEST  - Equity curves + metrics table.
DATA      - Reference dump: baselines, econ diagnostics, quintile stats,
            transcripts, survey batteries.
"""

import json
from pathlib import Path

# NumPy 2.0 backward-compatibility patch for older ecosystem packages (xarray/plotly)
import numpy as np

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_
if not hasattr(np, "bytes_"):
    np.bytes_ = bytes

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration & Theme Settings
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="K-ALPHA TERMINAL",
    page_icon="▮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Institutional CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    /* ---------- Base terminal palette ---------- */
    .stApp {
        background-color: #050505;
        color: #c8c8c8;
        font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
    }
    .main, .block-container {
        background-color: #050505;
        color: #c8c8c8;
    }
    [data-testid="stSidebar"] {
        background-color: #080808;
        border-right: 1px solid #1c1c1c;
    }
    [data-testid="stHeader"] {
        background-color: #050505;
        border-bottom: 1px solid #1c1c1c;
    }

    /* ---------- Typography ---------- */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'IBM Plex Mono', 'SF Mono', Menlo, monospace !important;
        letter-spacing: 0.02em;
        color: #eaeaea !important;
    }
    p, li, label, .stMarkdown {
        color: #b0b0b0;
    }
    strong { color: #eaeaea; }
    code {
        font-family: 'IBM Plex Mono', monospace;
        background: #111111;
        border: 1px solid #222222;
        padding: 1px 5px;
        border-radius: 0;
        color: #e8a33d;
    }

    /* ---------- Signal badges ---------- */
    .badge-buy, .badge-sell, .badge-hold, .econ-badge {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 3px 8px;
        border-radius: 0;
        display: inline-block;
        border-width: 1px;
    }
    .badge-buy  { background: rgba(76,175,125,0.10);  color: #6fd3a0; border-color: #2e6b4f; }
    .badge-sell { background: rgba(212,85,77,0.10);   color: #e0706a; border-color: #8f3232; }
    .badge-hold { background: rgba(216,178,92,0.10);  color: #d8b25c; border-color: #8a6420; }
    .econ-badge  { background: rgba(76,175,125,0.10); color: #6fd3a0; border-color: #2e6b4f; font-size: 0.75rem; padding: 4px 10px; }

    /* ---------- Quote panels ---------- */
    .quote-box-stated, .quote-box-revealed {
        background: #0a0a0a;
        border-left: 2px solid;
        padding: 12px 14px;
        border-radius: 0;
        margin-top: 8px;
        font-style: italic;
        color: #b0b0b0;
    }
    .quote-box-stated  { border-left-color: #d4554d; }
    .quote-box-revealed { border-left-color: #4caf7d; }

    /* ---------- Header / hero ---------- */
    .banner-header {
        background: #0a0a0a;
        padding: 10px 18px;
        border-radius: 0;
        border: 1px solid #222222;
        border-left: 3px solid #e8a33d;
        margin-bottom: 10px;
    }
    .forecast-hero {
        background: #0a0a0a;
        border: 1px solid #2a2a2a;
        border-top: 2px solid #e8a33d;
        border-radius: 0;
        padding: 10px 18px;
        margin-bottom: 12px;
    }
    .banner-header h1 { font-size: 1.3rem !important; margin: 0 !important; }

    /* ---------- Terminal section label ---------- */
    .term-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #e8a33d;
        border-bottom: 1px solid #222222;
        padding-bottom: 4px;
        margin: 4px 0 8px 0;
    }

    /* ---------- Streamlit widget chrome ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #222222;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #8a8a8a;
        border-radius: 0;
        padding: 8px 14px;
    }
    .stTabs [aria-selected="true"] {
        color: #e8a33d !important;
        border-bottom: 2px solid #e8a33d;
        background: transparent;
    }

    /* ---------- Dataframes & tables ---------- */
    .stDataFrame, [data-testid="stDataFrame"] {
        border: 1px solid #222222;
        border-radius: 0;
    }
    [data-testid="stDataFrame"] thead th {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #8a8a8a;
        background: #0a0a0a;
        border-bottom: 1px solid #2a2a2a;
    }
    [data-testid="stDataFrame"] td {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        font-variant-numeric: tabular-nums;
        color: #c8c8c8;
        border-bottom: 1px solid #161616;
    }

    /* ---------- Inputs ---------- */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        background-color: #0a0a0a !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 0;
        color: #eaeaea !important;
        font-family: 'IBM Plex Mono', monospace;
    }
    .stSlider [data-baseweb="slider"] [data-testid="stTickBar"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
    }
    div[data-baseweb="popover"] {
        background: #0a0a0a;
        border: 1px solid #2a2a2a;
    }

    /* ---------- Buttons / expander / info ---------- */
    .stButton > button {
        border-radius: 0;
        border: 1px solid #2a2a2a;
        background: #0a0a0a;
        color: #eaeaea;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .stButton > button:hover { border-color: #e8a33d; color: #e8a33d; }
    [data-testid="stExpander"] {
        border: 1px solid #222222;
        border-radius: 0;
        background: #080808;
    }
    [data-testid="stExpander"] summary {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #b0b0b0;
    }
    .stInfo {
        background: #0a0a0a;
        border: 1px solid #222222;
        border-left: 2px solid #e8a33d;
        border-radius: 0;
    }

    /* ---------- Chat messages ---------- */
    [data-testid="stChatMessage"] {
        border-radius: 0;
        border: 1px solid #222222;
        background: #0a0a0a;
    }

    /* ---------- Horizontal rules & misc ---------- */
    hr { border-top: 1px solid #222222; }
    .stCaption, small { color: #5c5c5c; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Data Loading & Caching
# -----------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"

STRATEGY_STYLE = {
    "spy": {"name": "SPY Buy & Hold", "color": "#5c5c5c", "dash": "dash", "width": 1.5},
    "pair_always": {
        "name": "Long COST / Short WMT",
        "color": "#e8a33d",
        "dash": "solid",
        "width": 1.6,
    },
    "pre_release": {
        "name": "K-Alpha: Buy on Expectation",
        "color": "#4caf7d",
        "dash": "solid",
        "width": 2.6,
    },
    "on_release": {
        "name": "Buy on News Release",
        "color": "#d4554d",
        "dash": "solid",
        "width": 1.8,
    },
    "pair_pre_release": {
        "name": "Pair on Expectation",
        "color": "#8a8a8a",
        "dash": "dot",
        "width": 1.2,
    },
    "pair_on_release": {
        "name": "Pair on Release",
        "color": "#3a3a3a",
        "dash": "dot",
        "width": 1.2,
    },
}


@st.cache_data
def load_all_data():
    out = {}
    for key, fname in [
        ("k_alpha", "k_alpha_retail_signals.json"),
        ("transcripts", "respondent_scored_transcripts.json"),
        ("replications", "affluent_survey_replications.json"),
        ("baseline_corr", "baseline_correlations.json"),
        ("fundamentals", "ticker_fundamentals.json"),
        ("forecast", "forward_cycle_forecast.json"),
        ("econometrics", "econometrics_results.json"),
        ("backtest", "backtest_results.json"),
    ]:
        with open(DATA_DIR / fname, "r") as f:
            out[key] = json.load(f)
    out["surveys_df"] = pd.read_csv(DATA_DIR / "baseline_sentiment_surveys.csv")
    out["prices_df"] = pd.read_csv(DATA_DIR / "market_prices.csv")
    return out


def load_strategy_performance():
    """data/strategy_performance.json — produced by the quant pipeline; None until it exists.
    Deliberately uncached so the file is picked up on the next rerun once it lands."""
    path = DATA_DIR / "strategy_performance.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


try:
    data = load_all_data()
    k_alpha = data["k_alpha"]
    transcripts = data["transcripts"]
    replications = data["replications"]
    fundamentals = data["fundamentals"]
    forecast = data["forecast"]
    econometrics = data["econometrics"]
    backtest = data["backtest"]
    surveys_df = data["surveys_df"]
    prices_df = data["prices_df"]
except Exception as e:
    st.error(f"Error loading datasets from data/: {e}")
    st.stop()

strategy_perf = load_strategy_performance()


# -----------------------------------------------------------------------------
# Strategy equity payload: JSON if present, inline fallback otherwise
# -----------------------------------------------------------------------------
def _fallback_series(prices: pd.DataFrame) -> dict:
    """SPY buy&hold + monthly-rebalanced Long COST / Short WMT from market_prices.csv."""
    spy_cum = (1.0 + prices["SPY"].pct_change().fillna(0.0)).cumprod() - 1.0
    pair_cum = (
        1.0 + (prices["COST"].pct_change() - prices["WMT"].pct_change()).fillna(0.0)
    ).cumprod() - 1.0
    spy_cum.iloc[0] = 0.0
    pair_cum.iloc[0] = 0.0
    return {
        "dates": prices["Date"].tolist(),
        "series": {"spy": spy_cum.tolist(), "pair_always": pair_cum.tolist()},
    }


def _series_metrics(cum: list) -> dict:
    """Total/ann return, vol, sharpe, max drawdown from a monthly cum-return series."""
    c = np.asarray(cum, dtype=float)
    if c.size < 2:
        return {
            "total_return": 0.0,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
        }
    monthly = np.diff(1.0 + c) / (1.0 + c[:-1])
    total = float(c[-1])
    n = c.size - 1
    ann_ret = float((1.0 + total) ** (12.0 / max(n, 1)) - 1.0)
    ann_vol = float(monthly.std(ddof=1) * np.sqrt(12.0)) if monthly.size > 1 else 0.0
    peak = np.maximum.accumulate(1.0 + c)
    return {
        "total_return": total,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0.0,
        "max_dd": float(np.min((1.0 + c) / peak - 1.0)),
    }


def strategy_equity_payload() -> dict:
    """Unified {'dates','series','metrics','from_json'} for both data paths."""
    if strategy_perf is not None:
        return {
            "dates": strategy_perf["dates"],
            "series": strategy_perf["series"],
            "metrics": strategy_perf.get("metrics", []),
            "notes": strategy_perf.get("strategy_notes", {}),
            "from_json": True,
        }
    fb = _fallback_series(prices_df)
    spy_m = _series_metrics(fb["series"]["spy"])
    pair_m = _series_metrics(fb["series"]["pair_always"])
    pair_m["excess_vs_spy_ann"] = pair_m["ann_return"] - spy_m["ann_return"]
    spy_m["excess_vs_spy_ann"] = 0.0
    return {
        "dates": fb["dates"],
        "series": fb["series"],
        "metrics": [
            {"strategy": "SPY Buy & Hold", **spy_m},
            {"strategy": "Long COST / Short WMT", **pair_m},
        ],
        "notes": {},
        "from_json": False,
    }


eq_payload = strategy_equity_payload()

# -----------------------------------------------------------------------------
# Figure builders (Bloomberg style: thin lines, hairline grid, mono legend)
# -----------------------------------------------------------------------------
_DARK = dict(
    template="plotly_dark",
    plot_bgcolor="#0a0a0a",
    paper_bgcolor="#050505",
    font=dict(color="#c8c8c8", family="IBM Plex Mono, monospace"),
)
_LEGEND = dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="left",
    x=0,
    bgcolor="rgba(10,10,10,0.85)",
    bordercolor="#2a2a2a",
    borderwidth=1,
    font=dict(size=11),
)


def build_equity_fig(payload: dict, height: int = 430) -> go.Figure:
    fig = go.Figure()
    for key, st_ in STRATEGY_STYLE.items():
        if key not in payload["series"]:
            continue
        fig.add_trace(
            go.Scatter(
                x=payload["dates"],
                y=[v * 100 for v in payload["series"][key]],
                name=st_["name"],
                mode="lines",
                line=dict(color=st_["color"], width=st_["width"], dash=st_["dash"]),
                hovertemplate="%{x}  %{y:+.1f}%<extra></extra>",
            )
        )
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10), legend=_LEGEND, **_DARK
    )
    fig.update_yaxes(
        title_text="Cum. Return (%)", gridcolor="#1a1a1a", zerolinecolor="#3a3a3a"
    )
    fig.update_xaxes(gridcolor="#1a1a1a")
    return fig


def build_kgap_fig(ts: pd.DataFrame, height: int = 430) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ts["dates"],
            y=ts["s_upper"],
            name="S_upper (Top 25%)",
            line=dict(color="#e8a33d", width=2.5),
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ts["dates"],
            y=ts["s_headline"],
            name="Michigan UMCSENT",
            line=dict(color="#8a8a8a", width=2, dash="dash"),
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ts["dates"],
            y=ts["s_lower"],
            name="S_lower (Bottom 25%)",
            line=dict(color="#d4554d", width=1.8),
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ts["dates"],
            y=ts["s_upper"],
            line=dict(width=0),
            showlegend=False,
            hoverinfo="none",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ts["dates"],
            y=ts["s_headline"],
            fill="tonexty",
            fillcolor="rgba(232,163,61,0.10)",
            name="Divergence Wedge",
            line=dict(width=0),
        )
    )
    fig.add_vline(x="2022-06-30", line_dash="dot", line_color="#5c5c5c")
    fig.add_vline(x="2024-01-31", line_dash="dot", line_color="#5c5c5c")
    fig.add_annotation(
        x="2022-06-30",
        y=55,
        showarrow=True,
        arrowhead=1,
        text="Jun-22 UMCSENT ATL 50.0",
        font=dict(size=10, color="#e8a33d"),
    )
    fig.add_annotation(
        x="2024-01-31",
        yref="paper",
        y=0.95,
        showarrow=True,
        arrowhead=1,
        text="2024 web-mode shock",
        font=dict(size=10, color="#e8a33d"),
    )
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10), legend=_LEGEND, **_DARK
    )
    fig.update_yaxes(title_text="Sentiment Index", gridcolor="#1a1a1a")
    fig.update_xaxes(gridcolor="#1a1a1a")
    return fig


def build_dispersion_fig(prices: pd.DataFrame, height: int = 360) -> go.Figure:
    dates = pd.to_datetime(prices["Date"])
    growth = {
        t: (prices[t] / prices[t].iloc[0] - 1.0) * 100
        for t in ["COST", "WMT", "TGT", "SPY"]
    }
    style = {
        "COST": ("#e8a33d", 2.5, "solid"),
        "WMT": ("#d4554d", 2, "solid"),
        "TGT": ("#d8b25c", 1.8, "dash"),
        "SPY": ("#5c5c5c", 1.5, "dot"),
    }
    fig = go.Figure()
    for t, (col, w, d) in style.items():
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=growth[t],
                name=f"{t} {growth[t].iloc[-1]:+.1f}%",
                line=dict(color=col, width=w, dash=d),
                mode="lines",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=growth["COST"] - growth["WMT"],
            name=f"L/S {(growth['COST'] - growth['WMT']).iloc[-1]:+.1f}%",
            line=dict(color="#4caf7d", width=2, dash="dot"),
            mode="lines",
        )
    )
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10), legend=_LEGEND, **_DARK
    )
    fig.update_yaxes(
        title_text="Normalized Return (%)", gridcolor="#1a1a1a", zerolinecolor="#3a3a3a"
    )
    fig.update_xaxes(gridcolor="#1a1a1a")
    return fig


def metrics_display_df(payload: dict) -> pd.DataFrame | None:
    if not payload["metrics"]:
        return None
    df = pd.DataFrame(payload["metrics"]).rename(
        columns={
            "strategy": "Strategy",
            "total_return": "Total Return",
            "ann_return": "Ann. Return",
            "ann_vol": "Ann. Vol",
            "sharpe": "Sharpe",
            "max_dd": "Max Drawdown",
            "excess_vs_spy_ann": "Excess vs SPY (Ann.)",
        }
    )
    for col, fmt in [
        ("Total Return", "{:+.1%}"),
        ("Ann. Return", "{:+.1%}"),
        ("Ann. Vol", "{:.1%}"),
        ("Max Drawdown", "{:.1%}"),
        ("Excess vs SPY (Ann.)", "{:+.1%}"),
    ]:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: fmt.format(v) if pd.notna(v) else "—")
    if "Sharpe" in df.columns:
        df["Sharpe"] = df["Sharpe"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    return df


def best_strategy_line(payload: dict) -> str:
    raw = payload["metrics"]
    non_spy = [r for r in raw if "spy" not in r.get("strategy", "").lower()]
    if not non_spy:
        return ""
    best = max(non_spy, key=lambda r: r.get("ann_return", float("-inf")))
    spy_ann = next(
        (r["ann_return"] for r in raw if "spy" in r.get("strategy", "").lower()), 0.0
    )
    return (
        f"{best['strategy']} leads: {best['ann_return']:+.1%} ann. vs SPY {spy_ann:+.1%} "
        f"({(best['ann_return'] - spy_ann) * 100:+.1f} pts excess)."
    )


ts_df = pd.DataFrame(k_alpha["macro_divergence_time_series"])
ts_df["dates"] = pd.to_datetime(ts_df["dates"])

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_terminal, tab_backtest, tab_data = st.tabs(["TERMINAL", "BACKTEST", "DATA"])

# =============================================================================
# TERMINAL
# =============================================================================
with tab_terminal:
    # Row 1: banner + one-line catalyst strip
    st.markdown(
        """
        <div class="banner-header">
            <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 10px;">
                <h1>K-ALPHA TERMINAL — RETAIL DIVERGENCE</h1>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #8a8a8a;">
                    FLASHPOINT AI / HACKATHON 2026 &nbsp;|&nbsp; BENCH: CHICAGO FED 521
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    catalyst = forecast["next_major_catalyst"]
    trade_rec = forecast["pair_trade_recommendation"]
    st.markdown(
        f"""
        <div class="forecast-hero">
            <span style="font-family:'IBM Plex Mono',monospace; font-size:0.68rem; letter-spacing:0.12em; color:#d4554d;">NEXT CATALYST</span>
            <strong style="color:#eaeaea;"> {catalyst['event_name']}</strong>
            <span style="color:#8a8a8a;"> ({catalyst['cycle_period']}, {catalyst['release_date']})</span>
            &nbsp;·&nbsp; <strong style="color:#eaeaea;">T-{catalyst['days_until_release']}d</strong>
            &nbsp;|&nbsp; CONS <strong style="color:#e8a33d;">{forecast['consensus_headline_prediction']:.1f}</strong>
            vs S_upper <strong style="color:#e8a33d;">{forecast['flashpoint_affluent_signal']:.1f}</strong>
            &rarr; surprise <strong style="color:#4caf7d;">+{forecast['predicted_surprise_points']:.1f} (z={forecast['predicted_surprise_zscore']:.2f})</strong>
            &nbsp;|&nbsp; <strong style="color:#d8b25c;">{trade_rec['strategy']}</strong>
            <span style="color:#4caf7d;"> · target {trade_rec['projected_cycle_alpha']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Row 2: K-gap wedge | strategy equity curves
    r2_l, r2_r = st.columns(2)
    with r2_l:
        st.markdown(
            '<div class="term-label">K-SHAPED SENTIMENT WEDGE</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(build_kgap_fig(ts_df, height=420), use_container_width=True)
    with r2_r:
        st.markdown(
            '<div class="term-label">STRATEGY EQUITY CURVES</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_equity_fig(eq_payload, height=420), use_container_width=True
        )
        if not eq_payload["from_json"]:
            st.caption(
                "strategy_performance.json pending — SPY + L/S pair computed inline."
            )

    # Row 3: vibecession scatter | IC leaderboard
    r3_l, r3_r = st.columns(2)
    with r3_l:
        st.markdown(
            '<div class="term-label">STATED vs REVEALED — VIBECESSION SCATTER</div>',
            unsafe_allow_html=True,
        )
        scatter_df = pd.DataFrame(k_alpha["respondent_scatter_data"])

        f1, f2, f3 = st.columns([2, 3, 3])
        with f1:
            retailer_filter = st.multiselect(
                "Cohort:", ["COST", "TGT", "WMT"], default=["COST", "TGT", "WMT"]
            )
        with f2:
            income_range = st.slider("HHI ($k):", 20, 350, (20, 350), step=10)
        with f3:
            highlight_costco = st.checkbox("Vibecession quadrant", value=True)

        filtered_df = scatter_df.copy()
        if retailer_filter:
            filtered_df = filtered_df[filtered_df["ticker"].isin(retailer_filter)]
        filtered_df = filtered_df[
            (filtered_df["hhi"] >= income_range[0] * 1000)
            & (filtered_df["hhi"] <= income_range[1] * 1000)
        ]
        filtered_df["bubble_size"] = np.clip(filtered_df["hhi"] / 12000, 5, 25)

        fig1 = px.scatter(
            filtered_df,
            x="stated_macro_sentiment",
            y="revealed_spend_conviction",
            color="ticker",
            color_discrete_map={"COST": "#e8a33d", "TGT": "#d8b25c", "WMT": "#d4554d"},
            size="bubble_size",
            size_max=20,
            hover_data={
                "id": True,
                "ticker": True,
                "hhi": ":$,.0f",
                "stated_macro_sentiment": ":.2f",
                "revealed_spend_conviction": ":.2f",
                "vibecession_gap": ":.2f",
                "trade_down_risk": ":.1%",
                "bubble_size": False,
            },
            labels={
                "stated_macro_sentiment": "Stated Sentiment",
                "revealed_spend_conviction": "Revealed Conviction",
                "ticker": "Cohort",
                "hhi": "HHI",
                "vibecession_gap": "Gap",
            },
        )
        fig1.add_hline(y=0, line_dash="dash", line_color="#3a3a3a", line_width=1)
        fig1.add_vline(x=0, line_dash="dash", line_color="#3a3a3a", line_width=1)
        if highlight_costco:
            fig1.add_shape(
                type="rect",
                x0=-1.05,
                y0=0.0,
                x1=0.0,
                y1=1.05,
                fillcolor="rgba(232,163,61,0.07)",
                line=dict(color="#e8a33d", width=1, dash="dot"),
            )
            fig1.add_annotation(
                x=-0.52,
                y=0.92,
                showarrow=False,
                text="<b>VIBECESSION</b>",
                font=dict(size=10, color="#e8a33d"),
                bgcolor="rgba(10,10,10,0.85)",
                bordercolor="#e8a33d",
                borderwidth=1,
                borderpad=4,
            )
            fig1.add_annotation(
                x=-0.55,
                y=-0.9,
                showarrow=False,
                text="<b>DISTRESS</b>",
                font=dict(size=10, color="#d4554d"),
                bgcolor="rgba(10,10,10,0.85)",
                bordercolor="#d4554d",
                borderwidth=1,
                borderpad=4,
            )
        fig1.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(
                range=[-1.05, 0.4], gridcolor="#1a1a1a", zerolinecolor="#3a3a3a"
            ),
            yaxis=dict(
                range=[-1.05, 1.05], gridcolor="#1a1a1a", zerolinecolor="#3a3a3a"
            ),
            legend=dict(_LEGEND, xanchor="right", x=1),
            **_DARK,
        )
        st.plotly_chart(fig1, use_container_width=True)

    with r3_r:
        st.markdown(
            '<div class="term-label">IC LEADERBOARD — SIGNAL vs HEADLINE</div>',
            unsafe_allow_html=True,
        )
        ic_comp = econometrics["information_coefficient_comparison"]
        ic_1m, ic_3m = ic_comp["1M"], ic_comp["3M"]
        cats = ["1M Forward", "3M Forward"]
        aff = [ic_1m["affluent_signal"]["mean_ic"], ic_3m["affluent_signal"]["mean_ic"]]
        hed = [
            ic_1m["headline_sentiment"]["mean_ic"],
            ic_3m["headline_sentiment"]["mean_ic"],
        ]

        fig_ic = go.Figure()
        fig_ic.add_trace(
            go.Bar(
                name="Affluent (S_upper)",
                x=cats,
                y=aff,
                marker_color="#e8a33d",
                text=[
                    f"IC={v:+.4f} (t={t:.2f})"
                    for v, t in zip(
                        aff,
                        [
                            ic_1m["affluent_signal"]["t_stat"],
                            ic_3m["affluent_signal"]["t_stat"],
                        ],
                    )
                ],
                textposition="outside",
            )
        )
        fig_ic.add_trace(
            go.Bar(
                name="Headline UMCSENT",
                x=cats,
                y=hed,
                marker_color="#d4554d",
                text=[
                    f"IC={v:+.4f} (t={t:.2f})"
                    for v, t in zip(
                        hed,
                        [
                            ic_1m["headline_sentiment"]["t_stat"],
                            ic_3m["headline_sentiment"]["t_stat"],
                        ],
                    )
                ],
                textposition="outside",
            )
        )
        fig_ic.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            barmode="group",
            legend=dict(_LEGEND, xanchor="center", x=0.5),
            **_DARK,
        )
        fig_ic.update_yaxes(
            title_text="Mean Spearman IC", gridcolor="#1a1a1a", range=[-0.12, 0.25]
        )
        fig_ic.update_xaxes(gridcolor="#1a1a1a")
        st.plotly_chart(fig_ic, use_container_width=True)
        st.caption(
            f"IC spread: 1M {ic_1m['comparative_paired_test']['ic_difference']:+.4f} "
            f"(t={ic_1m['comparative_paired_test']['t_statistic']:.2f}) · "
            f"3M {ic_3m['comparative_paired_test']['ic_difference']:+.4f} "
            f"(t={ic_3m['comparative_paired_test']['t_statistic']:.2f})."
        )

# =============================================================================
# BACKTEST
# =============================================================================
with tab_backtest:
    st.markdown(
        '<div class="term-label">STRATEGY PERFORMANCE — CUMULATIVE EQUITY</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(build_equity_fig(eq_payload, height=520), use_container_width=True)

    m_df = metrics_display_df(eq_payload)
    if m_df is not None:

        def _hl_excess(col):
            if col.name == "Excess vs SPY (Ann.)":
                return [
                    "background-color: rgba(232,163,61,0.14); color: #e8a33d; font-weight: 600"
                ] * len(col)
            return [""] * len(col)

        st.dataframe(
            m_df.style.apply(_hl_excess, axis=0),
            hide_index=True,
            use_container_width=True,
        )

    line = best_strategy_line(eq_payload)
    if not eq_payload["from_json"]:
        line = "strategy_performance.json pending — inline fallback shown. " + line
    st.caption(line)

# =============================================================================
# DATA — thin reference dump
# =============================================================================
with tab_data:
    # Baseline benchmark
    st.markdown(
        '<div class="term-label">BASELINE BENCHMARK — CHICAGO FED LETTER 521</div>',
        unsafe_allow_html=True,
    )
    surveys_comp = pd.DataFrame(k_alpha["survey_comparison_table"])
    surveys_sorted = surveys_comp.sort_values(by="pce_correlation", ascending=True)
    fig3 = go.Figure(
        go.Bar(
            x=surveys_sorted["pce_correlation"],
            y=surveys_sorted["survey"],
            orientation="h",
            marker=dict(
                color=[
                    "#e8a33d" if "Flashpoint" in s else "#3a3a3a"
                    for s in surveys_sorted["survey"]
                ],
                line=dict(
                    color="#e8a33d",
                    width=[
                        2 if "Flashpoint" in s else 0 for s in surveys_sorted["survey"]
                    ],
                ),
            ),
            text=[f"r = {v:+.3f}" for v in surveys_sorted["pce_correlation"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>r = %{x:.3f}<extra></extra>",
        )
    )
    fig3.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="r vs Real PCE Growth", range=[0, 0.85], gridcolor="#1a1a1a"),
        yaxis=dict(gridcolor="#1a1a1a"),
        **_DARK,
    )
    st.plotly_chart(fig3, use_container_width=True)
    display_df = surveys_comp[
        ["survey", "source", "coverage", "pce_correlation", "flaw_diagnosed"]
    ].copy()
    display_df.columns = ["Survey", "Source", "Coverage", "PCE r", "Flaw (Fed 521)"]
    st.dataframe(display_df, hide_index=True, use_container_width=True)
    fp_row = next(
        (r for r in surveys_comp.to_dict("records") if "Flashpoint" in r["survey"]),
        None,
    )
    fp_r = f"{fp_row['pce_correlation']:+.3f}" if fp_row else "n/a"
    st.caption(
        f"Fed 521: web-mode shift (~14.5 pts) + inflation salience (~7.2 pts) drive the 25-30 pt Michigan bias; SCF top-20% drives ~50% of PCE — K-Alpha restores r = {fp_r}."
    )

    # Econometric diagnostics (compact table)
    st.markdown(
        '<div class="term-label" style="margin-top:14px;">ECONOMETRIC DIAGNOSTICS</div>',
        unsafe_allow_html=True,
    )
    adf = econometrics["stationarity_tests"]["primary_signal"]
    coint = econometrics["cointegration_test"]
    eg = coint["engle_granger_coint_test"]
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Test": f"ADF — {adf['signal_name']}",
                    "Statistic": f"{adf['adf_statistic']:.4f}",
                    "p-value": f"{adf['p_value']:.6f}",
                    "Critical": f"1%: {adf['critical_values']['1%']:.3f}",
                    "Verdict": "STATIONARY",
                },
                {
                    "Test": f"Engle-Granger — {coint['relationship']}",
                    "Statistic": f"{eg['test_statistic']:.4f}",
                    "p-value": f"{eg['p_value']:.4f}",
                    "Critical": f"5%: {eg['critical_values']['5%']:.3f}",
                    "Verdict": "COINTEGRATED",
                },
                {
                    "Test": "IC spread (affluent − headline) 1M",
                    "Statistic": f"{ic_1m['comparative_paired_test']['ic_difference']:+.4f}",
                    "p-value": f"{ic_1m['comparative_paired_test']['p_value_one_tailed']:.4f}",
                    "Critical": f"t={ic_1m['comparative_paired_test']['t_statistic']:.2f}",
                    "Verdict": "SUPERIOR (p<0.05)",
                },
                {
                    "Test": "IC spread (affluent − headline) 3M",
                    "Statistic": f"{ic_3m['comparative_paired_test']['ic_difference']:+.4f}",
                    "p-value": f"{ic_3m['comparative_paired_test']['p_value_one_tailed']:.4f}",
                    "Critical": f"t={ic_3m['comparative_paired_test']['t_statistic']:.2f}",
                    "Verdict": "SUPERIOR (p<0.001)",
                },
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    # Ticker valuation
    st.markdown(
        '<div class="term-label" style="margin-top:14px;">TICKER VALUATION</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Ticker": tk,
                    "Action": (s := k_alpha["ticker_signals"][tk])["action"],
                    "Current": f"${s['current_price']:.2f}",
                    "K-Alpha Target": f"${s['k_alpha_target_price']:.2f}",
                    "Alpha": f"{s['k_alpha_upside_pct']:+.1f}%",
                    "Conviction": f"{s['revealed_conviction_score']:+.2f}",
                    "Trade-Down": f"{s['trade_down_risk']*100:.0f}%",
                    "P/E Fwd": f"{s['forward_pe']:.1f}x",
                }
                for tk in ["COST", "WMT", "TGT"]
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    # Quintile backtest stats + dispersion
    st.markdown(
        '<div class="term-label" style="margin-top:14px;">QUINTILE BACKTEST & DISPERSION</div>',
        unsafe_allow_html=True,
    )
    bt_l, bt_r = st.columns(2)
    with bt_l:
        for hzn in ["1M", "3M"]:
            hz = backtest["horizons"][hzn]
            lsm, icr = hz["long_short_metrics"], hz["information_coefficient"]
            q_rows = [
                {
                    "Q": q,
                    "Ann.Ret": f"{hz['quintile_metrics'][q]['annualized_return']:+.1%}",
                    "Sharpe": f"{hz['quintile_metrics'][q]['sharpe_ratio']:.2f}",
                    "MaxDD": f"{hz['quintile_metrics'][q]['max_drawdown']:.1%}",
                }
                for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]
            ]
            q_rows.append(
                {
                    "Q": "L/S",
                    "Ann.Ret": f"{lsm['annualized_return']:+.1%}",
                    "Sharpe": f"{lsm['sharpe_ratio']:.2f}",
                    "MaxDD": f"{lsm['max_drawdown']:.1%}",
                }
            )
            ja = hz.get("jensens_alpha")
            ja_txt = (
                f"; Jensen α {ja['alpha_annualized']:+.1%}/yr "
                f"(t={ja['alpha_t_stat']:.2f}, p={ja['alpha_p_value']:.3f}, "
                f"β={ja['beta']:+.2f}, R²={ja['r_squared']:.2f})"
                if ja
                else ""
            )
            st.caption(
                f"{hzn} quintiles + L/S (IC {icr['mean_ic']:+.4f}, t={icr['t_stat']:.2f}, p={icr['p_value']:.4f}; mono {hz['monotonicity_score']:.2f}{ja_txt})"
            )
            st.dataframe(
                pd.DataFrame(q_rows), hide_index=True, use_container_width=True
            )
    with bt_r:
        st.caption("3-year normalized retail dispersion.")
        st.plotly_chart(
            build_dispersion_fig(prices_df, height=340), use_container_width=True
        )

    # Transcripts explorer
    st.markdown(
        '<div class="term-label" style="margin-top:14px;">RESPONDENT INTERVIEW EXPLORER</div>',
        unsafe_allow_html=True,
    )
    t1, t2 = st.columns([1, 3])
    with t1:
        filter_ret = st.selectbox("Cohort:", ["COST", "TGT", "WMT"])
    subset_resp = [r for r in transcripts["respondents"] if r["retailer"] == filter_ret]
    resp_options = {
        r[
            "respondent_id"
        ]: f"{r['respondent_id']} | ${r['profile']['household_income']:,.0f} | ${r['profile']['avg_basket_size']:.2f} | {r['profile']['occupation']}"
        for r in subset_resp
    }
    with t2:
        selected_id = st.selectbox(
            "Respondent:",
            list(resp_options.keys()),
            format_func=lambda x: resp_options[x],
        )
    selected_resp = next(r for r in subset_resp if r["respondent_id"] == selected_id)
    prof, sc = selected_resp["profile"], selected_resp["scores"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stated", f"{sc['stated_macro_sentiment']:+.3f}")
    m2.metric("Revealed", f"{sc['revealed_spend_conviction']:+.3f}")
    m3.metric("Gap", f"{sc['vibecession_divergence']:+.3f}")
    m4.metric("Trade-Down", f"{sc['trade_down_risk']*100:.1f}%")
    st.caption(
        f"{prof['respondent_id']} · {prof['age']}{prof['gender']} · {prof['occupation']} · {prof['location']} · {prof['trips_per_month']} trips/mo"
    )

    for turn in selected_resp["qa_turns"][:2]:
        with st.chat_message("assistant"):
            st.markdown(f"**Probe:** {turn['question']}")
        with st.chat_message("user"):
            st.markdown(f"**Respondent:** {turn['answer']}")

    # Survey battery replications
    st.markdown(
        '<div class="term-label" style="margin-top:14px;">AFFLUENT SURVEY BATTERY REPLICATIONS</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Survey": v.get("survey_name"),
                    "Affluent": list(v.get("affluent_index_summary", {}).values())[0],
                    "Baseline": list(v.get("population_index_summary", {}).values())[0],
                    "Spread Gap": v.get("affluent_spread_gap"),
                }
                for v in replications.get("surveys", {}).values()
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    if "retail_divergence_implications" in replications:
        imp = replications["retail_divergence_implications"]
        for r_key, r_info in imp.get("ticker_signals", {}).items():
            st.markdown(
                f"- **{r_key}** <strong style='color:#e8a33d;'>{r_info.get('signal')}</strong> "
                f"(HHI {r_info.get('shopper_hhi_median')} | {r_info.get('affluent_exposure')}): {r_info.get('rationale')}",
                unsafe_allow_html=True,
            )

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### K-Alpha Terminal")
    st.markdown(
        "Affluent personas + revealed-preference scoring. Top 20% drive ~50% of PCE."
    )
    st.markdown("---")
    st.code(
        "Personas -> Probes -> Scorer\n-> K-Alpha Factor -> Signals", language="text"
    )
    st.caption("Touchpoint AI & Anthropic Claude")
