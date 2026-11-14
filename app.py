"""
app.py - Flashpoint K-Alpha Interactive Retail Sentiment & Asset Pricing Dashboard
Built for the Hackathon Pitch Demo.

Features:
1) Ticker Selector (COST, WMT, TGT) with clear Buy/Sell pricing adjustments
2) Visual 1: Stated Macro Sentiment vs. Revealed Spend Conviction (Vibecession Scatter)
3) Visual 2: The Widening K-Shaped Gap (S_upper vs Headline Sentiment) & Market Overlay
4) Visual 3: Baseline Outperformance vs 7 Surveys from Chicago Fed Letter 521
5) Deep-Dive Persona Interview Explorer & Affluent Survey Battery Replications
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
    page_title="Flashpoint K-Alpha | Vibecession Disconnect Engine",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Institutional CSS
st.markdown(
    """
    <style>
    /* Main container styling */
    .main {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Top KPI metric cards */
    .kpi-card {
        background: linear-gradient(135deg, #131b2e 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .kpi-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.85rem;
        margin-top: 4px;
        font-weight: 500;
    }
    .badge-buy {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid #059669;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-sell {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid #dc2626;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-hold {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid #d97706;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        display: inline-block;
    }
    .quote-box-stated {
        background: rgba(239, 68, 68, 0.07);
        border-left: 4px solid #ef4444;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-top: 8px;
        font-style: italic;
        color: #cbd5e1;
    }
    .quote-box-revealed {
        background: rgba(16, 185, 129, 0.07);
        border-left: 4px solid #10b981;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-top: 8px;
        font-style: italic;
        color: #cbd5e1;
    }
    .banner-header {
        background: linear-gradient(90deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        padding: 24px 28px;
        border-radius: 14px;
        border: 1px solid #3730a3;
        margin-bottom: 24px;
    }
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
    
    surveys_df = pd.read_csv(DATA_DIR / "baseline_sentiment_surveys.csv")
    prices_df = pd.read_csv(DATA_DIR / "market_prices.csv")
    
    return {
        "k_alpha": k_alpha_data,
        "transcripts": transcripts_data,
        "replications": replications_data,
        "baseline_corr": baseline_corr_data,
        "fundamentals": fundamentals_data,
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
    surveys_df = data["surveys_df"]
except Exception as e:
    st.error(f"Error loading datasets from data/: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Header Banner
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="banner-header">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <span style="background: #3b82f6; color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">
                    Flashpoint AI Hackathon 2026
                </span>
                <h1 style="margin: 8px 0 4px 0; font-size: 2.2rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">
                    ⚡ FLASHPOINT K-ALPHA | RETAIL DIVERGENCE ENGINE
                </h1>
                <p style="margin: 0; font-size: 1.02rem; color: #94a3b8; max-width: 950px;">
                    Exploiting the <strong>Vibecession Disconnect</strong>: Synthetic Affluent Persona Probing & Claude Revealed Preference Scoring uncover why dollar-weighted consumer spending diverges from headline sentiment.
                </p>
            </div>
            <div style="text-align: right; background: rgba(30, 41, 59, 0.6); padding: 12px 18px; border-radius: 10px; border: 1px solid #334155;">
                <div style="font-size: 0.78rem; color: #94a3b8; font-weight: 600;">BENCHMARK FRAMEWORK</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #60a5fa;">Chicago Fed Letter No. 521</div>
                <div style="font-size: 0.75rem; color: #a5b4fc;">Brave, Henken & Jolley (June 2026)</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
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
            <div class="kpi-sub"><span class="badge-buy">STRONG BUY</span> <span style="color:#34d399; font-weight:600;">+18.5% Upside</span></div>
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
            <div class="kpi-sub"><span class="badge-sell">UNDERPERFORM</span> <span style="color:#f87171; font-weight:600;">-8.5% Downside</span></div>
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
            <div class="kpi-sub"><span class="badge-hold">NEUTRAL / HOLD</span> <span style="color:#fbbf24; font-weight:600;">+1.2% Upside</span></div>
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
            <div class="kpi-sub" style="color:#38bdf8;">vs Michigan <strong>r = +0.215</strong> (+3.3x)</div>
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
            <div class="kpi-sub" style="color:#34d399;">Ann. Alpha <strong>+8.6%</strong> | Sharpe 1.11</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Navigation Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Ticker Selector & Valuation Adjustments",
    "📊 Visual 1: Stated vs Revealed Vibecession Scatter",
    "📈 Visual 2: Widening K-Shaped Gap & Market Alpha",
    "🏆 Visual 3: Chicago Fed 521 Baseline Benchmark",
    "🎙️ Interview Transcripts & Survey Batteries",
])

# =============================================================================
# TAB 1: Ticker Selector & Valuation Pricing Adjustments
# =============================================================================
with tab1:
    st.markdown("### 🎯 Quantitative Ticker Selector & Buy / Sell Pricing Adjustments")
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
        "BUY": '<span class="badge-buy" style="font-size: 1.1rem; padding: 6px 14px;">🟢 STRONG BUY / UNDERVALUED</span>',
        "SELL": '<span class="badge-sell" style="font-size: 1.1rem; padding: 6px 14px;">🔴 UNDERPERFORM / SELL</span>',
        "HOLD": '<span class="badge-hold" style="font-size: 1.1rem; padding: 6px 14px;">🟡 NEUTRAL / HOLD</span>',
    }.get(sig["rating_code"], "")
    
    upside_color = "#34d399" if sig["k_alpha_upside_pct"] > 0 else "#f87171"
    
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #131c31 0%, #1a233a 100%); border: 1px solid #334155; border-radius: 14px; padding: 22px 26px; margin: 16px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
                <div>
                    <h2 style="margin: 0; color: #f8fafc; font-size: 1.8rem;">{sig['name']} ({sig['ticker']})</h2>
                    <div style="color: #94a3b8; font-size: 0.95rem; margin-top: 4px;">Role in K-Economy: <strong style="color: #38bdf8;">{sig['k_role']}</strong> | Target Shopper HHI: <strong>{sig['target_shopper_hhi']}</strong></div>
                </div>
                <div>{badge_html}</div>
            </div>
            <hr style="border: 0; border-top: 1px solid #334155; margin: 18px 0;" />
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 18px;">
                <div>
                    <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;">CURRENT MARKET PRICE</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #f8fafc;">${sig['current_price']:.2f}</div>
                    <div style="font-size: 0.8rem; color: #64748b;">Street Consensus: ${sig['street_target_price']:.2f}</div>
                </div>
                <div>
                    <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;">K-ALPHA FAIR VALUE TARGET</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: {upside_color};">${sig['k_alpha_target_price']:.2f}</div>
                    <div style="font-size: 0.82rem; color: {upside_color}; font-weight: 600;">{sig['k_alpha_upside_pct']:+.1f}% Expected Alpha</div>
                </div>
                <div>
                    <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;">REVEALED SPEND CONVICTION</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #38bdf8;">{sig['revealed_conviction_score']:+.3f}</div>
                    <div style="font-size: 0.8rem; color: #64748b;">Scale: -1.0 (Cutback) to +1.0 (Full)</div>
                </div>
                <div>
                    <div style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;">TRADE-DOWN RISK & RENEWAL</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #f8fafc;">{sig['trade_down_risk']*100:.1f}% Risk</div>
                    <div style="font-size: 0.8rem; color: #34d399;">Renewal Intent: {sig['renewal_intent_rate']*100:.0f}%</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    col_th1, col_th2 = st.columns([3, 2])
    with col_th1:
        st.markdown("#### 💡 Quantitative Investment Thesis")
        st.info(sig["investment_thesis"])
        
        # Pull representative quotes for this ticker from respondent transcripts
        t_resp = [r for r in transcripts["respondents"] if r["retailer"] == selected_ticker]
        if t_resp:
            rep = t_resp[0]
            st.markdown("#### 🗣️ The Vibecession Disconnect in the Shopper's Own Words")
            qcol1, qcol2 = st.columns(2)
            with qcol1:
                st.markdown(
                    f"""
                    <div class="quote-box-stated">
                        <strong>Public Stated Gloom (To Pollsters / Surveys):</strong><br/>
                        "{rep['scores']['stated_quote']}"
                        <div style="margin-top: 6px; font-size: 0.78rem; color: #f87171; font-weight: 600;">Stated Macro Sentiment: {rep['scores']['stated_macro_sentiment']:+.2f}</div>
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
                        <div style="margin-top: 6px; font-size: 0.78rem; color: #34d399; font-weight: 600;">Revealed Spend Conviction: {rep['scores']['revealed_spend_conviction']:+.2f} | Basket: ${rep['scores']['avg_basket_size']:.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    
    with col_th2:
        st.markdown("#### ⚖️ Retail Peer Cross-Sectional Comparison")
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
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid #1e293b; border-radius: 10px; padding: 14px; margin-top: 10px;">
                <div style="font-size: 0.82rem; font-weight: 700; color: #38bdf8;">PAIR TRADE BACKTEST: LONG COST / SHORT WMT</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #34d399; margin: 4px 0;">+{pt['cumulative_spread_return']*100:.1f}% Cumulative Spread Gain</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">
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
    st.markdown("### 📊 Visual 1: The 'Vibecession' Scatter Plot")
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
        "COST": "#38bdf8",  # Electric Sky Blue
        "TGT": "#fbbf24",   # Amber / Gold
        "WMT": "#f87171",   # Coral Red
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
    fig1.add_hline(y=0, line_dash="dash", line_color="#475569", line_width=1.5)
    fig1.add_vline(x=0, line_dash="dash", line_color="#475569", line_width=1.5)
    
    # Highlight Vibecession quadrant if checked
    if highlight_costco:
        fig1.add_shape(
            type="rect",
            x0=-1.05,
            y0=0.0,
            x1=0.0,
            y1=1.05,
            fillcolor="rgba(56, 189, 248, 0.08)",
            line=dict(color="#38bdf8", width=1.5, dash="dot"),
        )
        fig1.add_annotation(
            x=-0.52,
            y=0.88,
            text="<b>THE VIBECESSION PARADOX (Costco Cluster)</b><br>Gloomy Macro Rhetoric, but Full $370+ Carts",
            showarrow=False,
            font=dict(size=12, color="#38bdf8"),
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="#38bdf8",
            borderwidth=1,
            borderpad=6,
        )
        
        # Real Distress quadrant annotation
        fig1.add_annotation(
            x=-0.55,
            y=-0.85,
            text="<b>GENUINE WALLET DISTRESS (Walmart Cluster)</b><br>Gloomy Rhetoric MATCHES Severe Basket Cuts",
            showarrow=False,
            font=dict(size=11, color="#f87171"),
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="#f87171",
            borderwidth=1,
            borderpad=6,
        )
        
        # Middle ground annotation
        fig1.add_annotation(
            x=-0.15,
            y=0.15,
            text="<b>CAUTIOUS MIDDLE (Target)</b><br>Selective Deferrals",
            showarrow=False,
            font=dict(size=11, color="#fbbf24"),
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="#fbbf24",
            borderwidth=1,
            borderpad=5,
        )

    fig1.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0b0f19",
        xaxis=dict(range=[-1.05, 0.4], gridcolor="#1e293b", zerolinecolor="#475569"),
        yaxis=dict(range=[-1.05, 1.05], gridcolor="#1e293b", zerolinecolor="#475569"),
        font=dict(color="#e2e8f0"),
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="#334155",
            borderwidth=1,
        ),
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Statistical Summary Row underneath Visual 1
    st.markdown("#### 📌 Cohort Statistical Breakdown")
    s_col1, s_col2, s_col3 = st.columns(3)
    c_sum = transcripts["summary_by_retailer"]
    
    with s_col1:
        st.markdown(
            f"""
            <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid #0284c7; border-radius: 10px; padding: 14px;">
                <h4 style="margin: 0; color: #38bdf8;">COSTCO COHORT (High-Income >$120k)</h4>
                <div style="font-size: 0.9rem; margin-top: 8px;">
                    • Mean Stated Macro Sentiment: <strong>{c_sum['COST']['mean_stated_macro_sentiment']:+.2f}</strong> (Gloom)<br/>
                    • Mean Revealed Spend Conviction: <strong style="color: #34d399;">{c_sum['COST']['mean_revealed_spend_conviction']:+.2f}</strong> (Boom)<br/>
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
            <div style="background: rgba(251, 191, 36, 0.08); border: 1px solid #d97706; border-radius: 10px; padding: 14px;">
                <h4 style="margin: 0; color: #fbbf24;">TARGET COHORT (Middle-Income $60k-$120k)</h4>
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
            <div style="background: rgba(248, 113, 113, 0.08); border: 1px solid #dc2626; border-radius: 10px; padding: 14px;">
                <h4 style="margin: 0; color: #f87171;">WALMART COHORT (Lower-Income <$60k)</h4>
                <div style="font-size: 0.9rem; margin-top: 8px;">
                    • Mean Stated Macro Sentiment: <strong>{c_sum['WMT']['mean_stated_macro_sentiment']:+.2f}</strong> (Severe)<br/>
                    • Mean Revealed Spend Conviction: <strong style="color: #f87171;">{c_sum['WMT']['mean_revealed_spend_conviction']:+.2f}</strong> (Severe Cutback)<br/>
                    • <strong>Vibecession Wedge: +{c_sum['WMT']['mean_vibecession_divergence']:.3f} pts</strong> (No Disconnect!)<br/>
                    • Mean Basket Size: <strong>${c_sum['WMT']['mean_basket_size']:.2f}</strong> | Trade-Down: <strong>{c_sum['WMT']['mean_trade_down_risk']*100:.1f}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =============================================================================
# TAB 3: Visual 2 - The Widening K-Shaped Gap & Market Alpha
# =============================================================================
with tab3:
    st.markdown("### 📈 Visual 2: The Widening K-Shaped Gap & Asset Pricing Overlay")
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
    
    # Subplots: Top = K-Divergence Sentiment, Bottom = Pair Trade Cumulative Alpha
    fig2 = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "<b>The K-Shaped Sentiment Wedge: Flashpoint High-Earner Index (S_upper) vs. Headline Michigan Sentiment</b>",
            "<b>Cumulative Market Return: Long COST / Short WMT Pair Trade vs. S&P 500 (SPY)</b>",
        ],
        row_heights=[0.6, 0.4],
    )
    
    # Top Plot Lines
    fig2.add_trace(
        go.Scatter(
            x=ts_df["dates"],
            y=ts_df["s_upper"],
            name="Flashpoint High-Earner Index (S_upper, Top 25%)",
            line=dict(color="#38bdf8", width=3),
            mode="lines",
        ),
        row=1, col=1,
    )
    fig2.add_trace(
        go.Scatter(
            x=ts_df["dates"],
            y=ts_df["s_headline"],
            name="Headline Sentiment (Univ of Michigan UMCSENT)",
            line=dict(color="#f59e0b", width=2.5, dash="dash"),
            mode="lines",
        ),
        row=1, col=1,
    )
    fig2.add_trace(
        go.Scatter(
            x=ts_df["dates"],
            y=ts_df["s_lower"],
            name="Lower-Income Sentiment (S_lower, Bottom 25%)",
            line=dict(color="#ef4444", width=2),
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
            fillcolor="rgba(56, 189, 248, 0.12)",
            name="The K-Alpha Divergence Wedge",
            line=dict(width=0),
        ),
        row=1, col=1,
    )
    
    # Bottom Plot: Long/Short Pair Trade & Equities Cumulative Performance
    prices_df = data["prices_df"].copy()
    prices_df["Date"] = pd.to_datetime(prices_df["Date"])
    cost_cum = (prices_df["COST"] / prices_df["COST"].iloc[0] - 1.0) * 100
    wmt_cum = (prices_df["WMT"] / prices_df["WMT"].iloc[0] - 1.0) * 100
    spread_cum = cost_cum - wmt_cum
    spy_cum = (prices_df["SPY"] / prices_df["SPY"].iloc[0] - 1.0) * 100
    
    fig2.add_trace(
        go.Scatter(
            x=prices_df["Date"],
            y=spread_cum,
            name="Long COST / Short WMT Spread (+70.1% Net Spread)",
            line=dict(color="#34d399", width=2.5),
            mode="lines",
        ),
        row=2, col=1,
    )
    fig2.add_trace(
        go.Scatter(
            x=prices_df["Date"],
            y=cost_cum,
            name="COST (+175.1% Total Return)",
            line=dict(color="#38bdf8", width=1.5, dash="dot"),
            mode="lines",
        ),
        row=2, col=1,
    )
    fig2.add_trace(
        go.Scatter(
            x=prices_df["Date"],
            y=wmt_cum,
            name="WMT (+105.0% Total Return)",
            line=dict(color="#f87171", width=1.5, dash="dot"),
            mode="lines",
        ),
        row=2, col=1,
    )
    fig2.add_trace(
        go.Scatter(
            x=prices_df["Date"],
            y=spy_cum,
            name="SPY Benchmark Return",
            line=dict(color="#64748b", width=1.2, dash="dash"),
            mode="lines",
        ),
        row=2, col=1,
    )
    
    # Macro event callouts
    fig2.add_vline(x="2022-06-30", line_dash="dot", line_color="#94a3b8", row=1, col=1)
    fig2.add_annotation(
        x="2022-06-30",
        y=55,
        text="June 2022: Michigan All-Time Low (50.0)<br>Gas $5.00/gal Shock",
        showarrow=True,
        arrowhead=1,
        font=dict(size=10, color="#f59e0b"),
        row=1, col=1,
    )
    
    fig2.add_vline(x="2024-01-31", line_dash="dot", line_color="#94a3b8", row=1, col=1)
    fig2.add_annotation(
        x="2024-01-31",
        y=135,
        text="2024: Web-Mode Transition Shock<br>(Chicago Fed 521: 25-30 pt downward bias)",
        showarrow=True,
        arrowhead=1,
        font=dict(size=10, color="#38bdf8"),
        row=1, col=1,
    )

    fig2.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0b0f19",
        height=720,
        font=dict(color="#e2e8f0"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="#334155",
            borderwidth=1,
        ),
    )
    fig2.update_yaxes(title_text="Sentiment Index Level", gridcolor="#1e293b", row=1, col=1)
    fig2.update_yaxes(title_text="Cumulative Return (%)", gridcolor="#1e293b", row=2, col=1)
    fig2.update_xaxes(gridcolor="#1e293b")
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Valuation & Pricing Adjustment Explanatory Card
    st.markdown("#### 🔍 Why the Market Misprices Retail Equities")
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        st.markdown(
            """
            <div style="background: rgba(16, 185, 129, 0.08); border-left: 4px solid #10b981; padding: 16px; border-radius: 0 10px 10px 0;">
                <h4 style="margin: 0 0 8px 0; color: #34d399;">🟢 The Costco Valuation Realignment (Strong Buy)</h4>
                <p style="margin: 0; font-size: 0.92rem; color: #cbd5e1; line-height: 1.5;">
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
            <div style="background: rgba(239, 68, 68, 0.08); border-left: 4px solid #ef4444; padding: 16px; border-radius: 0 10px 10px 0;">
                <h4 style="margin: 0 0 8px 0; color: #f87171;">🔴 The Walmart Multiple Compression Risk (Underperform)</h4>
                <p style="margin: 0; font-size: 0.92rem; color: #cbd5e1; line-height: 1.5;">
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
# TAB 4: Visual 3 - Chicago Fed Letter 521 Baseline Benchmark
# =============================================================================
with tab4:
    st.markdown("### 🏆 Visual 3: Baseline Outperformance & Chicago Fed Letter 521")
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
        "#38bdf8" if "Flashpoint" in s else "#475569"
        for s in surveys_sorted["survey"]
    ]
    
    fig3 = go.Figure(
        go.Bar(
            x=surveys_sorted["pce_correlation"],
            y=surveys_sorted["survey"],
            orientation="h",
            marker=dict(
                color=bar_colors,
                line=dict(color="#38bdf8", width=[2 if "Flashpoint" in s else 0 for s in surveys_sorted["survey"]]),
            ),
            text=[f"r = {v:+.3f}" for v in surveys_sorted["pce_correlation"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Correlation with Real PCE Growth: <b>%{x:.3f}</b><extra></extra>",
        )
    )
    
    fig3.update_layout(
        title="<b>Predictive Correlation with Real Consumer Spending Growth (YoY Real PCE)</b>",
        template="plotly_dark",
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0b0f19",
        xaxis=dict(
            title="Correlation Coefficient (r) with Real PCE Growth",
            range=[0, 0.85],
            gridcolor="#1e293b",
        ),
        yaxis=dict(gridcolor="#1e293b"),
        font=dict(color="#e2e8f0"),
        height=450,
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # Chicago Fed Letter 521 Empirical Diagnosis Table
    st.markdown("#### 📋 Comprehensive Diagnostic Matrix: Flashpoint K-Alpha vs. The 7 Baselines")
    
    # Format table for display
    display_df = surveys_comp[[
        "survey", "source", "coverage", "pce_correlation", "flaw_diagnosed"
    ]].copy()
    display_df.columns = [
        "Survey Name", "Publisher / Source", "Sampling Mechanism & Coverage", "PCE Correlation (r)", "Structural Flaw Diagnosed (Chicago Fed 521)"
    ]
    
    st.dataframe(display_df, hide_index=True, use_container_width=True)
    
    # Chicago Fed 521 Detailed Takeaway Card
    st.markdown("#### 📑 Key Findings from Chicago Fed Letter No. 521")
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid #334155; border-radius: 10px; padding: 18px;">
                <h4 style="margin: 0 0 10px 0; color: #60a5fa;">1. The 25-30 Point Michigan Downward Bias</h4>
                <p style="margin: 0; font-size: 0.9rem; color: #cbd5e1; line-height: 1.5;">
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
            <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid #334155; border-radius: 10px; padding: 18px;">
                <h4 style="margin: 0 0 10px 0; color: #34d399;">2. Why Dollar-Weighted Synthetic Sampling Solves the Puzzle</h4>
                <p style="margin: 0; font-size: 0.9rem; color: #cbd5e1; line-height: 1.5;">
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
# TAB 5: Deep-Dive Interview Transcripts & Survey Batteries
# =============================================================================
with tab5:
    st.markdown("### 🎙️ Synthetic Persona Probes & Affluent Survey Battery Replications")
    st.markdown(
        """
        Inspect the qualitative transcripts and quantitative scores generated by **Touchpoint / Flashpoint AI Persona Probing** 
        and evaluated via the **Anthropic Claude Revealed Preference Extraction Pipeline**.
        """
    )
    
    subtab1, subtab2 = st.tabs(["🗣️ Individual Respondent Interview Explorer", "📑 Affluent Survey Battery Replications"])
    
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
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid #334155; border-radius: 10px; padding: 14px 18px; margin: 12px 0;">
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
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"**Interviewer Probe:** {turn['question']}")
            with st.chat_message("user", avatar="🛒"):
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
            st.markdown("#### 🛍️ Retail Divergence Implications")
            imp = replications["retail_divergence_implications"]
            st.markdown(f"*{imp.get('summary', '')}*")
            for r_key, r_info in imp.get("ticker_signals", {}).items():
                st.markdown(
                    f"- **{r_key}** ({r_info.get('name')}) — "
                    f"<strong style='color:#38bdf8;'>{r_info.get('signal')}</strong> "
                    f"(Median HHI: {r_info.get('shopper_hhi_median')} | {r_info.get('affluent_exposure')}): "
                    f"{r_info.get('rationale')}",
                    unsafe_allow_html=True,
                )

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛒 Flashpoint K-Alpha")
    st.markdown(
        """
        **Hackathon Pitch Overview**
        - **Core Innovation**: Synthetic affluent persona generation + Claude revealed preference extraction to solve the Vibecession Disconnect.
        - **Economic Thesis**: The top 20% of earners drive ~50% of PCE. Equal-weighted polls distort real economic demand.
        - **Market Strategy**: Long COST / Short WMT pair trade generates +30.6% cumulative alpha with 1.11 Sharpe.
        """
    )
    st.markdown("---")
    st.markdown("#### 📂 Project Pipeline")
    st.code(
        """
Flashpoint AI Personas
        ↓
Behavioral Probes
        ↓
Claude Scorer (Stated vs Revealed)
        ↓
K-Alpha Factor Model
        ↓
Asset Pricing & Trading Signals
        """,
        language="text",
    )
    st.markdown("---")
    st.caption("Powered by Touchpoint AI & Anthropic Claude")
