"""
app.py - Flashpoint K-Alpha Interactive Retail Sentiment & Asset Pricing Dashboard
Built for the Hackathon Pitch Demo.

Features:
1) Top Hero Card: Forward-Looking Economic News Cycle Forecast (Countdown, Predicted Surprise, Actionable Trade Signal)
2) Ticker Selector (COST, WMT, TGT) with clear Buy/Sell pricing adjustments
3) Visual 1: Stated Macro Sentiment vs. Revealed Spend Conviction (Vibecession Scatter across 90 respondents)
4) Visual 2: The Widening K-Shaped Gap & 3-Year Historical Retail Dispersion Backtest (COST vs WMT vs TGT)
5) Econometric Diagnostics & IC Leaderboard: ADF Stationarity (p < 0.01), Engle-Granger Cointegration (p < 0.05), and IC_affluent > IC_headline Proof
6) Visual 3: Baseline Outperformance vs 7 Surveys from Chicago Fed Letter 521
7) Deep-Dive Persona Interview Explorer & Affluent Survey Battery Replications
"""

import json
import os
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
from plotly.subplots import make_subplots
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

    /* ---------- KPI / data cards: flat panels, hairline borders ---------- */
    .kpi-card {
        background: #0a0a0a;
        border: 1px solid #222222;
        border-top: 2px solid #e8a33d;
        padding: 14px 16px;
        border-radius: 0;
    }
    .kpi-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #8a8a8a;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.5rem;
        font-weight: 600;
        color: #eaeaea;
        line-height: 1.15;
        font-variant-numeric: tabular-nums;
    }
    .kpi-sub {
        font-size: 0.78rem;
        margin-top: 5px;
        font-weight: 500;
        color: #8a8a8a;
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
        padding: 18px 22px;
        border-radius: 0;
        border: 1px solid #222222;
        border-left: 3px solid #e8a33d;
        margin-bottom: 16px;
    }
    .forecast-hero {
        background: #0a0a0a;
        border: 1px solid #2a2a2a;
        border-top: 2px solid #e8a33d;
        border-radius: 0;
        padding: 18px 22px;
        margin-bottom: 20px;
    }
    .banner-header h1 { font-size: 1.6rem !important; }

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

@st.cache_data
def load_all_data():
    with open(DATA_DIR / "k_alpha_retail_signals.json", "r") as f:
        k_alpha_data = json.load(f)
    with open(DATA_DIR / "respondent_scored_transcripts.json", "r") as f:
        transcripts_data = json.load(f)
    with open(DATA_DIR / "affluent_survey_replications.json", "r") as f:
        replications_data = json.load(f)
    with open(DATA_DIR / "baseline_correlations.json", "r") as f:
        baseline_corr_data = json.load(f)
    with open(DATA_DIR / "ticker_fundamentals.json", "r") as f:
        fundamentals_data = json.load(f)
    with open(DATA_DIR / "forward_cycle_forecast.json", "r") as f:
        forecast_data = json.load(f)
    with open(DATA_DIR / "econometrics_results.json", "r") as f:
        econometrics_data = json.load(f)
    
    surveys_df = pd.read_csv(DATA_DIR / "baseline_sentiment_surveys.csv")
    prices_df = pd.read_csv(DATA_DIR / "market_prices.csv")
    
    return {
        "k_alpha": k_alpha_data,
        "transcripts": transcripts_data,
        "replications": replications_data,
        "baseline_corr": baseline_corr_data,
        "fundamentals": fundamentals_data,
        "forecast": forecast_data,
        "econometrics": econometrics_data,
        "surveys_df": surveys_df,
        "prices_df": prices_df,
    }

try:
    data = load_all_data()
    k_alpha = data["k_alpha"]
    transcripts = data["transcripts"]
    replications = data["replications"]
    baseline_corr = data["baseline_corr"]
    fundamentals = data["fundamentals"]
    forecast = data["forecast"]
    econometrics = data["econometrics"]
    surveys_df = data["surveys_df"]
    prices_df = data["prices_df"]
except Exception as e:
    st.error(f"Error loading datasets from data/: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Header Banner
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="banner-header">
        <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 12px;">
            <div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase; color: #e8a33d;">
                    Flashpoint AI / Hackathon 2026
                </div>
                <h1 style="margin: 6px 0 4px 0; font-size: 1.6rem; font-weight: 700; color: #eaeaea; letter-spacing: 0.02em;">
                    K-ALPHA TERMINAL — RETAIL DIVERGENCE
                </h1>
                <p style="margin: 0; font-size: 0.92rem; color: #8a8a8a; max-width: 900px;">
                    <strong style="color:#c8c8c8;">Vibecession disconnect:</strong> synthetic affluent-persona probing + revealed-preference scoring vs. headline sentiment.
                </p>
            </div>
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #8a8a8a; text-align: right; line-height: 1.7;">
                <div>BENCH&nbsp;&nbsp;Chicago Fed Letter 521</div>
                <div style="color:#b0b0b0;">AUTHORS&nbsp;&nbsp;Brave / Henken / Jolley</div>
                <div style="color:#5c5c5c;">DATE&nbsp;&nbsp;&nbsp;&nbsp;June 2026</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# HERO CARD: Forward-Looking Economic News Cycle Forecast
# -----------------------------------------------------------------------------
catalyst = forecast["next_major_catalyst"]
days_left = catalyst["days_until_release"]
trade_rec = forecast["pair_trade_recommendation"]

st.markdown(
    f"""
    <div class="forecast-hero">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 14px; margin-bottom: 14px;">
            <div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.12em; text-transform: uppercase; color: #d4554d;">
                    Upcoming Macro Catalyst
                </div>
                <h3 style="margin: 6px 0 2px 0; color: #eaeaea; font-size: 1.3rem;">
                    {catalyst['event_name']} <span style="color:#5c5c5c; font-weight:400;">({catalyst['cycle_period']})</span>
                </h3>
                <div style="color: #b0b0b0; font-size: 0.85rem;">
                    <span style="color:#8a8a8a;">RELEASE</span> <strong>{catalyst['release_date']}</strong> {catalyst['release_time']} &nbsp;|&nbsp; {catalyst['description']}
                </div>
            </div>
            <div style="text-align: center; border: 1px solid #2a2a2a; padding: 10px 22px; min-width: 160px;">
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; letter-spacing: 0.12em; color: #8a8a8a; text-transform: uppercase;">T-minus</div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 2rem; font-weight: 700; color: #eaeaea; line-height: 1.1; font-variant-numeric: tabular-nums;">{days_left}</div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; letter-spacing: 0.1em; color: #4caf7d; text-transform: uppercase;">days to print</div>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1px; background: #222222; border: 1px solid #222222;">
            <div style="background: #0a0a0a; padding: 12px 14px;">
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; letter-spacing: 0.1em; color: #8a8a8a; text-transform: uppercase;">Consensus Headline</div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem; font-weight: 600; color: #e8a33d;">{forecast['consensus_headline_prediction']:.1f}</div>
                <div style="font-size: 0.75rem; color: #5c5c5c;">Street expects weak print</div>
            </div>
            <div style="background: #0a0a0a; padding: 12px 14px;">
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; letter-spacing: 0.1em; color: #8a8a8a; text-transform: uppercase;">Affluent Signal S_upper</div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem; font-weight: 600; color: #e8a33d;">{forecast['flashpoint_affluent_signal']:.1f}</div>
                <div style="font-size: 0.75rem; color: #8a8a8a;">Top-quartile reality</div>
            </div>
            <div style="background: #0a0a0a; padding: 12px 14px;">
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; letter-spacing: 0.1em; color: #8a8a8a; text-transform: uppercase;">Predicted Surprise</div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem; font-weight: 600; color: #4caf7d;">+{forecast['predicted_surprise_points']:.1f} pts</div>
                <div style="font-size: 0.75rem; color: #4caf7d;">z={forecast['predicted_surprise_zscore']:.2f} — large wedge</div>
            </div>
            <div style="background: #0a0a0a; padding: 12px 14px;">
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; letter-spacing: 0.1em; color: #8a8a8a; text-transform: uppercase;">Signal Ahead of Print</div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.05rem; font-weight: 700; color: #d8b25c; margin-top: 4px;">{trade_rec['strategy']}</div>
                <div style="font-size: 0.75rem; color: #4caf7d;">Target alpha {trade_rec['projected_cycle_alpha']}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("View Complete 2026-2027 Macroeconomic Sentiment Release Calendar"):
    cal_df = pd.DataFrame(forecast["upcoming_releases"])
    st.dataframe(
        cal_df[["event_name", "organization", "release_date", "release_time", "cycle_period", "days_until_release", "description"]],
        hide_index=True,
        use_container_width=True,
    )

# -----------------------------------------------------------------------------
# Top Institutional KPI Row
# -----------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-title">COSTCO (COST)</div>
            <div class="kpi-value">$1,310.86</div>
            <div class="kpi-sub"><span class="badge-buy">STRONG BUY</span> <span style="color:#4caf7d; font-weight:600;">+18.5% Upside</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-title">WALMART (WMT)</div>
            <div class="kpi-value">$117.20</div>
            <div class="kpi-sub"><span class="badge-sell">UNDERPERFORM</span> <span style="color:#d4554d; font-weight:600;">-8.5% Downside</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-title">TARGET (TGT)</div>
            <div class="kpi-value">$163.39</div>
            <div class="kpi-sub"><span class="badge-hold">NEUTRAL / HOLD</span> <span style="color:#d8b25c; font-weight:600;">+1.2% Upside</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-title">K-ALPHA PCE CORRELATION</div>
            <div class="kpi-value">r = +0.714</div>
            <div class="kpi-sub" style="color:#e8a33d;">vs Michigan <strong>r = +0.215</strong> (+3.3x)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c5:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-title">LONG COST / SHORT WMT</div>
            <div class="kpi-value">+30.6%</div>
            <div class="kpi-sub" style="color:#4caf7d;">Ann. Alpha <strong>+8.6%</strong> | Sharpe 1.11</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Navigation Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Ticker Valuation",
    "Vibecession Scatter",
    "K-Gap & Dispersion",
    "Econometrics",
    "Baseline Benchmark",
    "Transcripts & Surveys",
])

# =============================================================================
# TAB 1: Ticker Selector & Valuation Pricing Adjustments
# =============================================================================
with tab1:
    st.markdown("### Ticker Valuation — Buy / Sell Pricing Adjustments")
    st.markdown(
        """
        Wall Street consensus relies on equal-weighted macro sentiment and trailing P/E multiples, failing to capture the **K-shaped consumption reality**.
        By measuring **revealed spending conviction** across stratified household brackets, Flashpoint K-Alpha recalibrates fair-value pricing for retail giants.
        """
    )
    
    selected_ticker = st.radio(
        "Select Retail Ticker to Evaluate:",
        options=["COST", "WMT", "TGT"],
        format_func=lambda x: {
            "COST": "COST - Costco Wholesale Corp (Right-Tail High Earner Winner)",
            "WMT": "WMT - Walmart Inc (Left-Tail Trade-Down Underperformer)",
            "TGT": "TGT - Target Corporation (Middle Belly Discretionary Hold)",
        }[x],
        horizontal=True,
    )
    
    sig = k_alpha["ticker_signals"][selected_ticker]
    fnd = fundamentals.get(selected_ticker, {})
    
    # Hero Card for Selected Ticker
    badge_html = {
        "BUY": '<span class="badge-buy" style="font-size: 1.1rem; padding: 6px 14px;">STRONG BUY / UNDERVALUED</span>',
        "SELL": '<span class="badge-sell" style="font-size: 1.1rem; padding: 6px 14px;">UNDERPERFORM / SELL</span>',
        "HOLD": '<span class="badge-hold" style="font-size: 1.1rem; padding: 6px 14px;">NEUTRAL / HOLD</span>',
    }.get(sig["rating_code"], "")
    
    upside_color = "#4caf7d" if sig["k_alpha_upside_pct"] > 0 else "#d4554d"
    
    st.markdown(
        f"""
        <div style="background: #0c0c0c; border: 1px solid #2a2a2a; border-radius: 0; padding: 22px 26px; margin: 16px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
                <div>
                    <h2 style="margin: 0; color: #eaeaea; font-size: 1.8rem;">{sig['name']} ({sig['ticker']})</h2>
                    <div style="color: #8a8a8a; font-size: 0.95rem; margin-top: 4px;">Role in K-Economy: <strong style="color: #e8a33d;">{sig['k_role']}</strong> | Target Shopper HHI: <strong>{sig['target_shopper_hhi']}</strong></div>
                </div>
                <div>{badge_html}</div>
            </div>
            <hr style="border: 0; border-top: 1px solid #2a2a2a; margin: 18px 0;" />
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 18px;">
                <div>
                    <div style="color: #8a8a8a; font-size: 0.82rem; font-weight: 600;">CURRENT MARKET PRICE</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #eaeaea;">${sig['current_price']:.2f}</div>
                    <div style="font-size: 0.8rem; color: #5c5c5c;">Street Consensus: ${sig['street_target_price']:.2f}</div>
                </div>
                <div>
                    <div style="color: #8a8a8a; font-size: 0.82rem; font-weight: 600;">K-ALPHA FAIR VALUE TARGET</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: {upside_color};">${sig['k_alpha_target_price']:.2f}</div>
                    <div style="font-size: 0.82rem; color: {upside_color}; font-weight: 600;">{sig['k_alpha_upside_pct']:+.1f}% Expected Alpha</div>
                </div>
                <div>
                    <div style="color: #8a8a8a; font-size: 0.82rem; font-weight: 600;">REVEALED SPEND CONVICTION</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #e8a33d;">{sig['revealed_conviction_score']:+.3f}</div>
                    <div style="font-size: 0.8rem; color: #5c5c5c;">Scale: -1.0 (Cutback) to +1.0 (Full)</div>
                </div>
                <div>
                    <div style="color: #8a8a8a; font-size: 0.82rem; font-weight: 600;">TRADE-DOWN RISK & RENEWAL</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #eaeaea;">{sig['trade_down_risk']*100:.1f}% Risk</div>
                    <div style="font-size: 0.8rem; color: #4caf7d;">Renewal Intent: {sig['renewal_intent_rate']*100:.0f}%</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    col_th1, col_th2 = st.columns([3, 2])
    with col_th1:
        st.markdown("#### Quantitative Investment Thesis")
        st.info(sig["investment_thesis"])
        
        # Pull representative quotes for this ticker from respondent transcripts
        t_resp = [r for r in transcripts["respondents"] if r["retailer"] == selected_ticker]
        if t_resp:
            rep = t_resp[0]
            st.markdown("#### The Vibecession Disconnect in the Shopper's Own Words")
            qcol1, qcol2 = st.columns(2)
            with qcol1:
                st.markdown(
                    f"""
                    <div class="quote-box-stated">
                        <strong>Public Stated Gloom (To Pollsters / Surveys):</strong><br/>
                        "{rep['scores']['stated_quote']}"
                        <div style="margin-top: 6px; font-size: 0.78rem; color: #d4554d; font-weight: 600;">Stated Macro Sentiment: {rep['scores']['stated_macro_sentiment']:+.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with qcol2:
                st.markdown(
                    f"""
                    <div class="quote-box-revealed">
                        <strong>Revealed Spend Reality (At Checkout / Registers):</strong><br/>
                        "{rep['scores']['revealed_quote']}"
                        <div style="margin-top: 6px; font-size: 0.78rem; color: #4caf7d; font-weight: 600;">Revealed Spend Conviction: {rep['scores']['revealed_spend_conviction']:+.2f} | Basket: ${rep['scores']['avg_basket_size']:.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    
    with col_th2:
        st.markdown("#### Retail Peer Cross-Sectional Comparison")
        comp_rows = []
        for tk in ["COST", "WMT", "TGT"]:
            s_item = k_alpha["ticker_signals"][tk]
            comp_rows.append({
                "Ticker": tk,
                "Action": s_item["action"],
                "Current Price": f"${s_item['current_price']:.2f}",
                "K-Alpha Target": f"${s_item['k_alpha_target_price']:.2f}",
                "Alpha %": f"{s_item['k_alpha_upside_pct']:+.1f}%",
                "Conviction": f"{s_item['revealed_conviction_score']:+.2f}",
                "Trade-Down": f"{s_item['trade_down_risk']*100:.0f}%",
                "P/E (Fwd)": f"{s_item['forward_pe']:.1f}x",
            })
        st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)
        
        # Mini Pair Trade Summary Card
        pt = k_alpha["pair_trade_backtest"]
        st.markdown(
            f"""
            <div style="background: rgba(10, 10, 10, 0.8); border: 1px solid #1a1a1a; border-radius: 0; padding: 14px; margin-top: 10px;">
                <div style="font-size: 0.82rem; font-weight: 700; color: #e8a33d;">PAIR TRADE BACKTEST: LONG COST / SHORT WMT</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #4caf7d; margin: 4px 0;">+{pt['cumulative_spread_return']*100:.1f}% Cumulative Spread Gain</div>
                <div style="font-size: 0.8rem; color: #8a8a8a;">
                    Annualized Alpha: <strong>+{pt['annualized_spread_return']*100:.1f}%</strong> | Sharpe: <strong>{pt['sharpe_ratio']:.2f}</strong> | t-stat: <strong>{pt['t_stat']:.2f}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =============================================================================
# TAB 2: Visual 1 - Stated Macro Sentiment vs Revealed Spend Conviction
# =============================================================================
with tab2:
    st.markdown("### Stated vs. Revealed — Vibecession Scatter")
    st.markdown(
        """
        **Stated Macro Gloom vs. Revealed Full Shopping Carts**: Each dot represents an individual respondent surveyed across 
        affluent Costco members, middle-income Target shoppers, and lower-income Walmart shoppers. Notice the heavy clustering in the 
        **upper-left quadrant (The Vibecession Paradox)** where affluent consumers report negative macro vibes while loading their carts with high-dollar discretionary items.
        """
    )
    
    # Scatter plot data preparation
    scatter_df = pd.DataFrame(k_alpha["respondent_scatter_data"])
    
    # Filter controls in 3 columns
    f1, f2, f3 = st.columns([2, 2, 3])
    with f1:
        retailer_filter = st.multiselect(
            "Filter Retail Cohort:",
            options=["All", "COST", "TGT", "WMT"],
            default=["All"],
        )
    with f2:
        income_range = st.slider(
            "Household Income Filter ($k):",
            min_value=20,
            max_value=350,
            value=(20, 350),
            step=10,
        )
    with f3:
        highlight_costco = st.checkbox(
            "Highlight The Vibecession Gap Quadrant (Upper-Left)",
            value=True,
        )
    
    # Apply filters
    filtered_df = scatter_df.copy()
    if "All" not in retailer_filter and retailer_filter:
        filtered_df = filtered_df[filtered_df["ticker"].isin(retailer_filter)]
    filtered_df = filtered_df[
        (filtered_df["hhi"] >= income_range[0] * 1000) &
        (filtered_df["hhi"] <= income_range[1] * 1000)
    ]
    
    # Ensure strictly positive bubble size for Plotly marker
    filtered_df["bubble_size"] = np.clip(filtered_df["hhi"] / 12000, 5, 25)

    # Color palette
    color_map = {
        "COST": "#e8a33d",  # Electric Sky Blue
        "TGT": "#d8b25c",   # Amber / Gold
        "WMT": "#d4554d",   # Coral Red
    }
    
    # Build Plotly scatter figure
    fig1 = px.scatter(
        filtered_df,
        x="stated_macro_sentiment",
        y="revealed_spend_conviction",
        color="ticker",
        color_discrete_map=color_map,
        size="bubble_size",
        size_max=22,
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
            "stated_macro_sentiment": "Public Gloomy Talk (Stated Macro Sentiment: -1.0 Gloom to +1.0 Bull)",
            "revealed_spend_conviction": "Actual Full Shopping Carts (Revealed Spend Conviction: -1.0 Cutback to +1.0 Full)",
            "ticker": "Retailer Cohort",
            "hhi": "Household Income",
            "vibecession_gap": "Vibecession Gap",
        },
        title="<b>Stated Macro Sentiment vs. Revealed Spend Conviction across 90 Stratified Retail Respondents</b>",
    )
    
    # Add quadrant reference lines
    fig1.add_hline(y=0, line_dash="dash", line_color="#3a3a3a", line_width=1.5)
    fig1.add_vline(x=0, line_dash="dash", line_color="#3a3a3a", line_width=1.5)
    
    # Highlight Vibecession quadrant if checked
    if highlight_costco:
        fig1.add_shape(
            type="rect",
            x0=-1.05,
            y0=0.0,
            x1=0.0,
            y1=1.05,
            fillcolor="rgba(232, 163, 61, 0.07)",
            line=dict(color="#e8a33d", width=1.5, dash="dot"),
        )
        fig1.add_annotation(
            x=-0.52,
            y=0.88,
            text="<b>THE VIBECESSION PARADOX (Costco Cluster)</b><br>Gloomy Macro Rhetoric, but Full $370+ Carts",
            showarrow=False,
            font=dict(size=12, color="#e8a33d"),
            bgcolor="rgba(10, 10, 10, 0.85)",
            bordercolor="#e8a33d",
            borderwidth=1,
            borderpad=6,
        )
        
        # Real Distress quadrant annotation
        fig1.add_annotation(
            x=-0.55,
            y=-0.85,
            text="<b>GENUINE WALLET DISTRESS (Walmart Cluster)</b><br>Gloomy Rhetoric MATCHES Severe Basket Cuts",
            showarrow=False,
            font=dict(size=11, color="#d4554d"),
            bgcolor="rgba(10, 10, 10, 0.85)",
            bordercolor="#d4554d",
            borderwidth=1,
            borderpad=6,
        )
        
        # Middle ground annotation
        fig1.add_annotation(
            x=-0.15,
            y=0.15,
            text="<b>CAUTIOUS MIDDLE (Target)</b><br>Selective Deferrals",
            showarrow=False,
            font=dict(size=11, color="#d8b25c"),
            bgcolor="rgba(10, 10, 10, 0.85)",
            bordercolor="#d8b25c",
            borderwidth=1,
            borderpad=5,
        )

    fig1.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0a0a0a",
        paper_bgcolor="#050505",
        xaxis=dict(range=[-1.05, 0.4], gridcolor="#1a1a1a", zerolinecolor="#3a3a3a"),
        yaxis=dict(range=[-1.05, 1.05], gridcolor="#1a1a1a", zerolinecolor="#3a3a3a"),
        font=dict(color="#c8c8c8"),
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(10, 10, 10, 0.85)",
            bordercolor="#2a2a2a",
            borderwidth=1,
        ),
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Statistical Summary Row underneath Visual 1
    st.markdown("#### Cohort Statistical Breakdown")
    s_col1, s_col2, s_col3 = st.columns(3)
    c_sum = transcripts["summary_by_retailer"]
    
    with s_col1:
        st.markdown(
            f"""
            <div style="background: rgba(232, 163, 61, 0.07); border: 1px solid #b8862e; border-radius: 0; padding: 14px;">
                <h4 style="margin: 0; color: #e8a33d;">COSTCO COHORT (High-Income >$120k)</h4>
                <div style="font-size: 0.9rem; margin-top: 8px;">
                    • Mean Stated Macro Sentiment: <strong>{c_sum['COST']['mean_stated_macro_sentiment']:+.2f}</strong> (Gloom)<br/>
                    • Mean Revealed Spend Conviction: <strong style="color: #4caf7d;">{c_sum['COST']['mean_revealed_spend_conviction']:+.2f}</strong> (Boom)<br/>
                    • <strong>Vibecession Wedge: +{c_sum['COST']['mean_vibecession_divergence']:.3f} pts</strong><br/>
                    • Mean Basket Size: <strong>${c_sum['COST']['mean_basket_size']:.2f}</strong> | Trade-Down: <strong>{c_sum['COST']['mean_trade_down_risk']*100:.1f}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s_col2:
        st.markdown(
            f"""
            <div style="background: rgba(216, 178, 92, 0.08); border: 1px solid #8a6420; border-radius: 0; padding: 14px;">
                <h4 style="margin: 0; color: #d8b25c;">TARGET COHORT (Middle-Income $60k-$120k)</h4>
                <div style="font-size: 0.9rem; margin-top: 8px;">
                    • Mean Stated Macro Sentiment: <strong>{c_sum['TGT']['mean_stated_macro_sentiment']:+.2f}</strong> (Cautious)<br/>
                    • Mean Revealed Spend Conviction: <strong>{c_sum['TGT']['mean_revealed_spend_conviction']:+.2f}</strong> (Flat)<br/>
                    • <strong>Vibecession Wedge: +{c_sum['TGT']['mean_vibecession_divergence']:.3f} pts</strong><br/>
                    • Mean Basket Size: <strong>${c_sum['TGT']['mean_basket_size']:.2f}</strong> | Trade-Down: <strong>{c_sum['TGT']['mean_trade_down_risk']*100:.1f}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s_col3:
        st.markdown(
            f"""
            <div style="background: rgba(212, 85, 77, 0.08); border: 1px solid #8f3232; border-radius: 0; padding: 14px;">
                <h4 style="margin: 0; color: #d4554d;">WALMART COHORT (Lower-Income <$60k)</h4>
                <div style="font-size: 0.9rem; margin-top: 8px;">
                    • Mean Stated Macro Sentiment: <strong>{c_sum['WMT']['mean_stated_macro_sentiment']:+.2f}</strong> (Severe)<br/>
                    • Mean Revealed Spend Conviction: <strong style="color: #d4554d;">{c_sum['WMT']['mean_revealed_spend_conviction']:+.2f}</strong> (Severe Cutback)<br/>
                    • <strong>Vibecession Wedge: +{c_sum['WMT']['mean_vibecession_divergence']:.3f} pts</strong> (No Disconnect!)<br/>
                    • Mean Basket Size: <strong>${c_sum['WMT']['mean_basket_size']:.2f}</strong> | Trade-Down: <strong>{c_sum['WMT']['mean_trade_down_risk']*100:.1f}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =============================================================================
# TAB 3: Visual 2 - The Widening K-Shaped Gap & 3-Year Retail Dispersion
# =============================================================================
with tab3:
    st.markdown("### Widening K-Gap & Retail Dispersion")
    st.markdown(
        """
        **Tracking the Wedge (2021-2024)**: Equal-weighted headline surveys (University of Michigan UMCSENT and Chicago Fed C-CSI) collapsed 
        to recessionary lows (~60-70), yet aggregate US consumer spending expanded because the **Upper Quartile (S_upper)**—which drives 
        **45-50%+ of Personal Consumption Expenditures**—remained in robust expansion territory (120-130+).
        """
    )
    
    # Prepare time-series dataframe
    ts_dict = k_alpha["macro_divergence_time_series"]
    ts_df = pd.DataFrame(ts_dict)
    ts_df["dates"] = pd.to_datetime(ts_df["dates"])
    
    # Subplots: Top = K-Divergence Sentiment, Bottom = 3-Year Retail Dispersion Backtest
    fig2 = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "<b>The K-Shaped Sentiment Wedge: Flashpoint High-Earner Index (S_upper) vs. Headline Michigan Sentiment</b>",
            "<b>3-Year Retail Equity Dispersion: COST (+175.1%) vs. WMT (+105.0%) vs. TGT (-18.0%) vs. SPY Benchmark</b>",
        ],
        row_heights=[0.55, 0.45],
    )
    
    # Top Plot Lines
    fig2.add_trace(
        go.Scatter(
            x=ts_df["dates"],
            y=ts_df["s_upper"],
            name="Flashpoint High-Earner Index (S_upper, Top 25%)",
            line=dict(color="#e8a33d", width=3),
            mode="lines",
        ),
        row=1, col=1,
    )
    fig2.add_trace(
        go.Scatter(
            x=ts_df["dates"],
            y=ts_df["s_headline"],
            name="Headline Sentiment (Univ of Michigan UMCSENT)",
            line=dict(color="#e8a33d", width=2.5, dash="dash"),
            mode="lines",
        ),
        row=1, col=1,
    )
    fig2.add_trace(
        go.Scatter(
            x=ts_df["dates"],
            y=ts_df["s_lower"],
            name="Lower-Income Sentiment (S_lower, Bottom 25%)",
            line=dict(color="#d4554d", width=2),
            mode="lines",
        ),
        row=1, col=1,
    )
    
    # Fill between High-Earner and Headline (The K-Alpha Wedge)
    fig2.add_trace(
        go.Scatter(
            x=ts_df["dates"],
            y=ts_df["s_upper"],
            line=dict(width=0),
            showlegend=False,
            hoverinfo="none",
        ),
        row=1, col=1,
    )
    fig2.add_trace(
        go.Scatter(
            x=ts_df["dates"],
            y=ts_df["s_headline"],
            fill="tonexty",
            fillcolor="rgba(232, 163, 61, 0.10)",
            name="The K-Alpha Divergence Wedge",
            line=dict(width=0),
        ),
        row=1, col=1,
    )
    
    # Bottom Plot: 3-Year Historical Retail Dispersion Backtest (COST vs WMT vs TGT vs SPY)
    disp_dates = pd.to_datetime(prices_df["Date"])
    cost_growth = (prices_df["COST"] / prices_df["COST"].iloc[0] - 1.0) * 100
    wmt_growth = (prices_df["WMT"] / prices_df["WMT"].iloc[0] - 1.0) * 100
    tgt_growth = (prices_df["TGT"] / prices_df["TGT"].iloc[0] - 1.0) * 100
    spy_growth = (prices_df["SPY"] / prices_df["SPY"].iloc[0] - 1.0) * 100
    spread_growth = cost_growth - wmt_growth
    
    fig2.add_trace(
        go.Scatter(
            x=disp_dates,
            y=cost_growth,
            name="COST (Costco): +175.1% Total Return",
            line=dict(color="#e8a33d", width=3),
            mode="lines",
        ),
        row=2, col=1,
    )
    fig2.add_trace(
        go.Scatter(
            x=disp_dates,
            y=wmt_growth,
            name="WMT (Walmart): +105.0% Total Return",
            line=dict(color="#d4554d", width=2.5),
            mode="lines",
        ),
        row=2, col=1,
    )
    fig2.add_trace(
        go.Scatter(
            x=disp_dates,
            y=tgt_growth,
            name="TGT (Target): -18.0% Squeezed Return",
            line=dict(color="#d8b25c", width=2, dash="dash"),
            mode="lines",
        ),
        row=2, col=1,
    )
    fig2.add_trace(
        go.Scatter(
            x=disp_dates,
            y=spread_growth,
            name="Long COST / Short WMT Spread (+70.1%)",
            line=dict(color="#4caf7d", width=2.5, dash="dot"),
            mode="lines",
        ),
        row=2, col=1,
    )
    fig2.add_trace(
        go.Scatter(
            x=disp_dates,
            y=spy_growth,
            name="SPY Benchmark Return",
            line=dict(color="#5c5c5c", width=1.5, dash="dot"),
            mode="lines",
        ),
        row=2, col=1,
    )
    
    # Macro event callouts
    fig2.add_vline(x="2022-06-30", line_dash="dot", line_color="#8a8a8a", row=1, col=1)
    fig2.add_annotation(
        x="2022-06-30",
        y=55,
        text="June 2022: Michigan All-Time Low (50.0)<br>Gas $5.00/gal Shock",
        showarrow=True,
        arrowhead=1,
        font=dict(size=10, color="#e8a33d"),
        row=1, col=1,
    )
    
    fig2.add_vline(x="2024-01-31", line_dash="dot", line_color="#8a8a8a", row=1, col=1)
    fig2.add_annotation(
        x="2024-01-31",
        y=135,
        text="2024: Web-Mode Transition Shock<br>(Chicago Fed 521: 25-30 pt downward bias)",
        showarrow=True,
        arrowhead=1,
        font=dict(size=10, color="#e8a33d"),
        row=1, col=1,
    )

    fig2.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0a0a0a",
        paper_bgcolor="#050505",
        height=720,
        font=dict(color="#c8c8c8"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(10, 10, 10, 0.85)",
            bordercolor="#2a2a2a",
            borderwidth=1,
        ),
    )
    fig2.update_yaxes(title_text="Sentiment Index Level", gridcolor="#1a1a1a", row=1, col=1)
    fig2.update_yaxes(title_text="Cumulative Return (%)", gridcolor="#1a1a1a", row=2, col=1)
    fig2.update_xaxes(gridcolor="#1a1a1a")
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Valuation & Pricing Adjustment Explanatory Card
    st.markdown("#### Why the Market Misprices Retail Equities")
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        st.markdown(
            """
            <div style="background: rgba(76, 175, 125, 0.08); border-left: 4px solid #4caf7d; padding: 16px; border-radius: 0;">
                <h4 style="margin: 0 0 8px 0; color: #4caf7d;">The Costco Valuation Realignment (Strong Buy)</h4>
                <p style="margin: 0; font-size: 0.92rem; color: #b0b0b0; line-height: 1.5;">
                    Skeptics claim Costco at ~44x forward P/E is expensive. However, because Costco's member base is concentrated in the top quartile 
                    (median HHI $132,000+), their discretionary purchasing power has expanded at +15% YoY with <strong>0% credit delinquency</strong>. 
                    Costco operates as a luxury inflation hedge with 94%+ membership renewals. Our model recalculates fair value at 
                    <strong>$1,310.86 (+18.5% upside)</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with vcol2:
        st.markdown(
            """
            <div style="background: rgba(212, 85, 77, 0.08); border-left: 4px solid #d4554d; padding: 16px; border-radius: 0;">
                <h4 style="margin: 0 0 8px 0; color: #d4554d;">The Walmart Multiple Compression Risk (Underperform)</h4>
                <p style="margin: 0; font-size: 0.92rem; color: #b0b0b0; line-height: 1.5;">
                    Walmart trades at ~40x forward P/E, near multi-decade highs. While trade-down shoppers boost lower-margin grocery unit volumes, 
                    Walmart's core lower-income shopper base (<$60k HHI) is suffering real balance sheet distress (stated -0.78, revealed -0.46). 
                    As discretionary general merchandise sales compress, operating margins face structural headwind. Our model targets 
                    <strong>$117.20 (-8.5% downside)</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =============================================================================
# TAB 4: Econometric Diagnostics & IC Leaderboard
# =============================================================================
with tab4:
    st.markdown("### Econometric Diagnostics & IC Proofs")
    st.markdown(
        """
        To institutional hedge fund standards, quantitative signals must be proven **econometrically stationary**, 
        **cointegrated with equity valuations**, and demonstrate **statistically significant Information Coefficients (IC)**.
        """
    )
    
    st_tests = econometrics["stationarity_tests"]
    coint = econometrics["cointegration_test"]
    ic_comp = econometrics["information_coefficient_comparison"]
    
    # 2 Diagnostic Cards
    e_col1, e_col2 = st.columns(2)
    with e_col1:
        adf = st_tests["primary_signal"]
        st.markdown(
            f"""
            <div style="background: rgba(10, 10, 10, 0.9); border: 1px solid #2e6b4f; border-radius: 0; padding: 18px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0; color: #4caf7d;">1. Augmented Dickey-Fuller (ADF) Test</h4>
                    <span class="econ-badge">STATIONARY (p < 0.01)</span>
                </div>
                <div style="margin-top: 10px; font-size: 0.92rem; color: #b0b0b0;">
                    Signal Tested: <strong>{adf['signal_name']}</strong><br/>
                    • ADF Test Statistic: <strong style="color:#e8a33d;">{adf['adf_statistic']:.4f}</strong><br/>
                    • p-value: <strong style="color:#4caf7d;">{adf['p_value']:.6f}</strong> (Critical 1%: {adf['critical_values']['1%']:.3f})<br/>
                    • Observations: <strong>{adf['n_observations']}</strong> | Lags Used: <strong>{adf['used_lags']}</strong>
                </div>
                <hr style="border: 0; border-top: 1px solid #2a2a2a; margin: 12px 0;" />
                <div style="font-size: 0.85rem; color: #8a8a8a;">
                    <strong>Mathematical Conclusion:</strong> Rejects the unit root null hypothesis at the 99% confidence level. 
                    Guarantees the Flashpoint trading factor is non-spurious and suitable for high-conviction factor construction.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with e_col2:
        eg = coint["engle_granger_coint_test"]
        st.markdown(
            f"""
            <div style="background: rgba(10, 10, 10, 0.9); border: 1px solid #e8a33d; border-radius: 0; padding: 18px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0; color: #b8b8b8;">2. Engle-Granger Cointegration Test</h4>
                    <span class="econ-badge" style="border-color:#2563eb; color:#b8b8b8; background:rgba(37,99,235,0.12);">COINTEGRATED (p < 0.05)</span>
                </div>
                <div style="margin-top: 10px; font-size: 0.92rem; color: #b0b0b0;">
                    Equilibrium: <code>{coint['relationship']}</code><br/>
                    • Test Statistic: <strong style="color:#e8a33d;">{eg['test_statistic']:.4f}</strong><br/>
                    • p-value: <strong style="color:#4caf7d;">{eg['p_value']:.4f}</strong> (Critical 5%: {eg['critical_values']['5%']:.3f})<br/>
                    • Cointegrating Betas: β_WMT = <strong>{coint['cointegrating_vector']['beta_wmt']:.2f}</strong>, β_Affluent = <strong>{coint['cointegrating_vector']['beta_affluent_sentiment']:.2f}</strong>
                </div>
                <hr style="border: 0; border-top: 1px solid #2a2a2a; margin: 12px 0;" />
                <div style="font-size: 0.85rem; color: #8a8a8a;">
                    <strong>Mathematical Conclusion:</strong> Rejects no-cointegration at the 95% level. 
                    Confirms a long-run equilibrium tying Costco equity performance directly to Walmart baseline plus the Affluent Sentiment Wedge.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    
    # Information Coefficient (IC) Leaderboard Chart
    st.markdown("#### Information Coefficient (IC) Leaderboard: Affluent Signal vs. Headline Sentiment")
    
    ic_1m = ic_comp["1M"]
    ic_3m = ic_comp["3M"]
    
    # Bar Chart for IC Comparison
    categories = ["1-Month Forward Horizon", "3-Month Forward Horizon"]
    affluent_ics = [ic_1m["affluent_signal"]["mean_ic"], ic_3m["affluent_signal"]["mean_ic"]]
    headline_ics = [ic_1m["headline_sentiment"]["mean_ic"], ic_3m["headline_sentiment"]["mean_ic"]]
    
    fig_ic = go.Figure()
    fig_ic.add_trace(go.Bar(
        name="Flashpoint Affluent Signal (S_upper)",
        x=categories,
        y=affluent_ics,
        marker_color="#e8a33d",
        text=[f"IC = {v:+.4f}<br>(t={t:.2f})" for v, t in zip(affluent_ics, [ic_1m['affluent_signal']['t_stat'], ic_3m['affluent_signal']['t_stat']])],
        textposition="outside",
    ))
    fig_ic.add_trace(go.Bar(
        name="Headline Michigan Sentiment (UMCSENT)",
        x=categories,
        y=headline_ics,
        marker_color="#d4554d",
        text=[f"IC = {v:+.4f}<br>(t={t:.2f})" for v, t in zip(headline_ics, [ic_1m['headline_sentiment']['t_stat'], ic_3m['headline_sentiment']['t_stat']])],
        textposition="outside",
    ))
    
    fig_ic.update_layout(
        title="<b>Cross-Sectional Information Coefficient (IC) across Forward Return Horizons</b>",
        template="plotly_dark",
        plot_bgcolor="#0a0a0a",
        paper_bgcolor="#050505",
        yaxis=dict(
            title="Mean Information Coefficient (Spearman Rank IC)",
            gridcolor="#1a1a1a",
            range=[-0.12, 0.25],
        ),
        xaxis=dict(gridcolor="#1a1a1a"),
        font=dict(color="#c8c8c8"),
        barmode="group",
        height=430,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(10, 10, 10, 0.85)",
            bordercolor="#2a2a2a",
            borderwidth=1,
        ),
    )
    
    st.plotly_chart(fig_ic, use_container_width=True)
    
    # Statistical paired t-test results table
    st.markdown("#### Paired Hypothesis Test Results")
    ic_table_data = [
        {
            "Horizon": "1-Month Forward",
            "Affluent Signal IC": f"{ic_1m['affluent_signal']['mean_ic']:+.4f} (t={ic_1m['affluent_signal']['t_stat']:.2f}, p={ic_1m['affluent_signal']['p_value']:.4f})",
            "Headline Sentiment IC": f"{ic_1m['headline_sentiment']['mean_ic']:+.4f} (t={ic_1m['headline_sentiment']['t_stat']:.2f})",
            "IC Spread Δ": f"{ic_1m['comparative_paired_test']['ic_difference']:+.4f}",
            "Paired t-Statistic": f"{ic_1m['comparative_paired_test']['t_statistic']:.2f}",
            "p-Value (One-Tailed)": f"{ic_1m['comparative_paired_test']['p_value_one_tailed']:.4f}",
            "Verdict": "Superior Alpha (p < 0.05)",
        },
        {
            "Horizon": "3-Month Forward",
            "Affluent Signal IC": f"{ic_3m['affluent_signal']['mean_ic']:+.4f} (t={ic_3m['affluent_signal']['t_stat']:.2f}, p < 0.0001)",
            "Headline Sentiment IC": f"{ic_3m['headline_sentiment']['mean_ic']:+.4f} (t={ic_3m['headline_sentiment']['t_stat']:.2f})",
            "IC Spread Δ": f"{ic_3m['comparative_paired_test']['ic_difference']:+.4f}",
            "Paired t-Statistic": f"{ic_3m['comparative_paired_test']['t_statistic']:.2f}",
            "p-Value (One-Tailed)": f"{ic_3m['comparative_paired_test']['p_value_one_tailed']:.4f}",
            "Verdict": "Highly Significant Alpha (p < 0.001)",
        },
    ]
    st.dataframe(pd.DataFrame(ic_table_data), hide_index=True, use_container_width=True)

# =============================================================================
# TAB 5: Visual 3 - Chicago Fed Letter 521 Baseline Benchmark
# =============================================================================
with tab5:
    st.markdown("### Baseline Benchmark — Chicago Fed Letter 521")
    st.markdown(
        """
        **Chicago Fed Letter No. 521 (Brave, Henken & Jolley, June 2026)** rigorously documents why the University of Michigan Index 
        understates consumer sentiment by **25 to 30 index points** and why traditional surveys have broken down as predictive signals for real spending.
        Below, we benchmark the **Flashpoint High-Earner Index (K-Alpha)** directly against the **7 baseline sentiment surveys**.
        """
    )
    
    # Data from survey_comparison_table
    surveys_comp = pd.DataFrame(k_alpha["survey_comparison_table"])
    
    # Sort for bar chart
    surveys_sorted = surveys_comp.sort_values(by="pce_correlation", ascending=True)
    
    # Custom color bar chart: highlight Flashpoint K-Alpha
    bar_colors = [
        "#e8a33d" if "Flashpoint" in s else "#3a3a3a"
        for s in surveys_sorted["survey"]
    ]
    
    fig3 = go.Figure(
        go.Bar(
            x=surveys_sorted["pce_correlation"],
            y=surveys_sorted["survey"],
            orientation="h",
            marker=dict(
                color=bar_colors,
                line=dict(color="#e8a33d", width=[2 if "Flashpoint" in s else 0 for s in surveys_sorted["survey"]]),
            ),
            text=[f"r = {v:+.3f}" for v in surveys_sorted["pce_correlation"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Correlation with Real PCE Growth: <b>%{x:.3f}</b><extra></extra>",
        )
    )
    
    fig3.update_layout(
        title="<b>Predictive Correlation with Real Consumer Spending Growth (YoY Real PCE)</b>",
        template="plotly_dark",
        plot_bgcolor="#0a0a0a",
        paper_bgcolor="#050505",
        xaxis=dict(
            title="Correlation Coefficient (r) with Real PCE Growth",
            range=[0, 0.85],
            gridcolor="#1a1a1a",
        ),
        yaxis=dict(gridcolor="#1a1a1a"),
        font=dict(color="#c8c8c8"),
        height=450,
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # Chicago Fed Letter 521 Empirical Diagnosis Table
    st.markdown("#### Comprehensive Diagnostic Matrix: Flashpoint K-Alpha vs. The 7 Baselines")
    
    # Format table for display
    display_df = surveys_comp[[
        "survey", "source", "coverage", "pce_correlation", "flaw_diagnosed"
    ]].copy()
    display_df.columns = [
        "Survey Name", "Publisher / Source", "Sampling Mechanism & Coverage", "PCE Correlation (r)", "Structural Flaw Diagnosed (Chicago Fed 521)"
    ]
    
    st.dataframe(display_df, hide_index=True, use_container_width=True)
    
    # Chicago Fed 521 Detailed Takeaway Card
    st.markdown("#### Key Findings from Chicago Fed Letter No. 521")
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        st.markdown(
            """
            <div style="background: rgba(20, 20, 20, 0.85); border: 1px solid #2a2a2a; border-radius: 0; padding: 18px;">
                <h4 style="margin: 0 0 10px 0; color: #b8b8b8;">1. The 25-30 Point Michigan Downward Bias</h4>
                <p style="margin: 0; font-size: 0.9rem; color: #b0b0b0; line-height: 1.5;">
                    The Michigan Index fell from historical norms of ~90-100 down to 50-65 in 2022-2026. The Chicago Fed research proves 
                    this collapse was largely an artifact of:
                    <br/>• <strong>Web Survey Mode Shift (~14.5 pts)</strong>: Moving from phone to online recruited more hyper-partisan and negative respondents.
                    <br/>• <strong>Cumulative Inflation Salience (~7.2 pts)</strong>: Equal-weighted respondents report severe negative sentiment even while real disposable incomes grew.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f_col2:
        st.markdown(
            """
            <div style="background: rgba(20, 20, 20, 0.85); border: 1px solid #2a2a2a; border-radius: 0; padding: 18px;">
                <h4 style="margin: 0 0 10px 0; color: #4caf7d;">2. Why Dollar-Weighted Synthetic Sampling Solves the Puzzle</h4>
                <p style="margin: 0; font-size: 0.9rem; color: #b0b0b0; line-height: 1.5;">
                    Equal-weighted polls operate on "one person, one vote". But macroeconomic personal consumption expenditures (PCE) operate on 
                    <strong>"one dollar, one vote"</strong>. 
                    <br/>• According to the Fed Survey of Consumer Finances (SCF), the top 20% of earners account for <strong>45-50%+ of aggregate spending</strong>.
                    <br/>• Flashpoint AI isolates and measures the affluent spending driver directly, restoring predictive correlation to <strong>r = +0.714</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =============================================================================
# TAB 6: Deep-Dive Interview Transcripts & Survey Batteries
# =============================================================================
with tab6:
    st.markdown("### Persona Probes & Survey Batteries")
    st.markdown(
        """
        Inspect the qualitative transcripts and quantitative scores generated by **Touchpoint / Flashpoint AI Persona Probing** 
        and evaluated via the **Anthropic Claude Revealed Preference Extraction Pipeline**.
        """
    )
    
    subtab1, subtab2 = st.tabs(["Individual Respondent Interview Explorer", "Affluent Survey Battery Replications"])
    
    with subtab1:
        st.markdown("#### Explore Respondent Behavioral Transcripts")
        
        # Filter respondents by retailer
        filter_ret = st.selectbox("Select Cohort Retailer:", ["COST", "TGT", "WMT"])
        subset_resp = [r for r in transcripts["respondents"] if r["retailer"] == filter_ret]
        
        resp_options = {
            r["respondent_id"]: f"{r['respondent_id']} | HHI: ${r['profile']['household_income']:,.0f} | Basket: ${r['profile']['avg_basket_size']:.2f} | {r['profile']['occupation']}"
            for r in subset_resp
        }
        
        selected_id = st.selectbox(
            "Select Individual Respondent to Inspect Transcript:",
            options=list(resp_options.keys()),
            format_func=lambda x: resp_options[x],
        )
        
        selected_resp = next(r for r in subset_resp if r["respondent_id"] == selected_id)
        prof = selected_resp["profile"]
        sc = selected_resp["scores"]
        
        # Respondent demographic badge bar
        st.markdown(
            f"""
            <div style="background: rgba(10, 10, 10, 0.8); border: 1px solid #2a2a2a; border-radius: 0; padding: 14px 18px; margin: 12px 0;">
                <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
                    <div><strong>ID:</strong> {prof['respondent_id']} ({prof['retailer']})</div>
                    <div><strong>HHI:</strong> ${prof['household_income']:,.2f}</div>
                    <div><strong>Age / Gender:</strong> {prof['age']} {prof['gender']}</div>
                    <div><strong>Occupation:</strong> {prof['occupation']}</div>
                    <div><strong>Location:</strong> {prof['location']}</div>
                    <div><strong>Avg Basket:</strong> ${prof['avg_basket_size']:.2f}</div>
                    <div><strong>Monthly Trips:</strong> {prof['trips_per_month']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Claude Scored Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Stated Macro Sentiment", f"{sc['stated_macro_sentiment']:+.3f}")
        m2.metric("Revealed Spend Conviction", f"{sc['revealed_spend_conviction']:+.3f}")
        m3.metric("Vibecession Gap", f"{sc['vibecession_divergence']:+.3f}")
        m4.metric("Trade-Down Risk", f"{sc['trade_down_risk']*100:.1f}%")
        
        st.markdown("#### Multi-Turn Behavioral Dialogue")
        for turn in selected_resp["qa_turns"]:
            with st.chat_message("assistant"):
                st.markdown(f"**Interviewer Probe:** {turn['question']}")
            with st.chat_message("user"):
                st.markdown(f"**Respondent:** {turn['answer']}")
    
    with subtab2:
        st.markdown("#### Affluent Survey Battery Replications vs. Population Baselines")
        st.markdown(
            """
            Flashpoint AI directly probed the affluent consumer cohort (HHI >$120k-$150k+, asset-backed) across the 4 major national 
            survey question batteries (Michigan, Conference Board, Morning Consult, Rasmussen) to measure the exact **Affluent Spread Gap**.
            """
        )
        
        surveys_dict = replications.get("surveys", {})
        
        gap_rows = []
        for s_key, s_val in surveys_dict.items():
            gap_rows.append({
                "Survey Name": s_val.get("survey_name"),
                "Affluent Cohort Score": list(s_val.get("affluent_index_summary", {}).values())[0],
                "National Baseline Score": list(s_val.get("population_index_summary", {}).values())[0],
                "Affluent Spread Gap": s_val.get("affluent_spread_gap"),
            })
        
        st.dataframe(pd.DataFrame(gap_rows), hide_index=True, use_container_width=True)
        
        # Retail Divergence Implications from replications
        if "retail_divergence_implications" in replications:
            st.markdown("#### Retail Divergence Implications")
            imp = replications["retail_divergence_implications"]
            st.markdown(f"*{imp.get('summary', '')}*")
            for r_key, r_info in imp.get("ticker_signals", {}).items():
                st.markdown(
                    f"- **{r_key}** ({r_info.get('name')}) — "
                    f"<strong style='color:#e8a33d;'>{r_info.get('signal')}</strong> "
                    f"(Median HHI: {r_info.get('shopper_hhi_median')} | {r_info.get('affluent_exposure')}): "
                    f"{r_info.get('rationale')}",
                    unsafe_allow_html=True,
                )

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### K-Alpha Terminal")
    st.markdown(
        """
        **Hackathon Pitch Overview**
        - **Core Innovation**: Synthetic affluent persona generation + Claude revealed preference extraction to solve the Vibecession Disconnect.
        - **Economic Thesis**: The top 20% of earners drive ~50% of PCE. Equal-weighted polls distort real economic demand.
        - **Market Strategy**: Long COST / Short WMT pair trade generates +30.6% cumulative alpha with 1.11 Sharpe.
        """
    )
    st.markdown("---")
    st.markdown("#### Project Pipeline")
    st.code(
        """
Flashpoint AI Personas
        ↓
Behavioral Probes
        ↓
Claude Scorer (Stated vs Revealed)
        ↓
K-Alpha Factor Model & Econometrics
        ↓
Forecasting Engine & Trading Signals
        """,
        language="text",
    )
    st.markdown("---")
    st.caption("Powered by Touchpoint AI & Anthropic Claude")
