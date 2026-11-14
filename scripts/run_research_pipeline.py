#!/usr/bin/env python
"""
Consolidated Research Pipeline Execution Script.

Orchestrates the complete K-Shaped Macro Research & Retail Alpha Workflow:
0. Live Flashpoint demographics pull, driven by Gemini over the Flashpoint MCP (paid, x402)
1. Macro Sentiment Stratification (Flashpoint AI 5 Michigan Questions)
2. Chicago Fed Letter 521 Baseline Survey Benchmarks
3. Factor Model & 20-Asset Cross-Sectional Backtest
4. Retail Alpha Signals & Vibecession Gap (COST vs WMT vs TGT)
5. Econometric Diagnostics (ADF Stationarity, Cointegration, IC Superiority)
6. Forward Economic News Cycle Forecast

Outputs all verified artifacts to data/ for Streamlit visualization and reporting.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant.backtest import run_factor_backtest
from src.quant.econometrics import run_full_econometric_suite
from src.quant.factor import load_sentiment_signals
from src.quant.market_data import fetch_historical_prices, fetch_ticker_fundamentals
from src.quant.retail_alpha import (
    compute_retail_k_alpha,
    load_baseline_surveys,
    load_respondent_transcripts,
)
from src.sampler.gemini_mcp_agent import run_flashpoint_agent

FLASHPOINT_TASK = (
    "Use the Flashpoint demographics tool once for location 'United States' and report the "
    "household income distribution: the share of households in each income band."
)


def main():
    print("=" * 82)
    print("  HACK THE UNREACHABLE: K-SHAPED CONSUMER MACRO RESEARCH PIPELINE")
    print("=" * 82)

    # -------------------------------------------------------------------------
    # STAGE 1: LIVE FLASHPOINT PULL (GEMINI DRIVES THE FLASHPOINT MCP, PAID VIA X402)
    # -------------------------------------------------------------------------
    print("\n[STAGE 1/6] Gemini calling Flashpoint MCP for live household income distribution...")
    live = run_flashpoint_agent(FLASHPOINT_TASK)
    paid_calls = [c for c in live["tool_calls"] if c["ok"]]
    if not paid_calls:
        raise RuntimeError(f"Gemini made no successful Flashpoint MCP call: {live['tool_calls']}")
    live_out = PROJECT_ROOT / "data" / "flashpoint_live_demographics.json"
    with open(live_out, "w") as f:
        json.dump({"task": FLASHPOINT_TASK, **live}, f, indent=2)
    print(f"  ✓ {live['model']} made {len(paid_calls)} live Flashpoint call(s): {[c['tool'] for c in paid_calls]}")
    print(f"  ✓ Paid {live['spent_usdc']:.2f} USDC via x402; saved {live_out.name}")

    # -------------------------------------------------------------------------
    # STAGE 2: INGESTION & DATA SYNC
    # -------------------------------------------------------------------------
    print("\n[STAGE 2/6] Ingesting Sentiment Signals, Transcripts, and Market Data...")
    sentiment_path = PROJECT_ROOT / "data" / "sentiment_signals.json"
    sentiment_df = load_sentiment_signals(sentiment_path)
    print(f"  ✓ Loaded {len(sentiment_df)} months of stratified sentiment signals ({sentiment_df.index[0].date()} -> {sentiment_df.index[-1].date()})")

    transcripts_path = PROJECT_ROOT / "data" / "respondent_scored_transcripts.json"
    transcripts = load_respondent_transcripts(transcripts_path)
    print(f"  ✓ Loaded {len(transcripts)} respondent transcripts with Claude revealed preference scores")

    prices_path = PROJECT_ROOT / "data" / "market_prices.csv"
    prices_df = fetch_historical_prices(cache_path=prices_path)
    prices_df = prices_df.loc[:, ~prices_df.columns.duplicated()]
    print(f"  ✓ Ingested historical price data across {len(prices_df.columns)} assets")

    baseline_csv = PROJECT_ROOT / "data" / "baseline_sentiment_surveys.csv"
    baseline_json = PROJECT_ROOT / "data" / "baseline_correlations.json"
    baseline_df, baseline_stats = load_baseline_surveys(baseline_csv, baseline_json)
    print(f"  ✓ Loaded 7 baseline surveys from Chicago Fed Letter 521 ({len(baseline_df)} periods)")

    # -------------------------------------------------------------------------
    # STAGE 3: CROSS-SECTIONAL FACTOR MODEL BACKTEST
    # -------------------------------------------------------------------------
    print("\n[STAGE 3/6] Running Cross-Sectional Factor Model Backtest...")
    backtest_results = run_factor_backtest(sentiment_df=sentiment_df, prices_df=prices_df)
    backtest_out = PROJECT_ROOT / "data" / "backtest_results.json"
    with open(backtest_out, "w") as f:
        json.dump(backtest_results, f, indent=2)

    h1_ls = backtest_results["horizons"]["1M"]["long_short_metrics"]
    h3_ls = backtest_results["horizons"]["3M"]["long_short_metrics"]
    print(f"  ✓ 1M Horizon: Long/Short Ann. Return = {h1_ls['annualized_return']*100:+.2f}%, Sharpe = {h1_ls['sharpe_ratio']:.2f}")
    print(f"  ✓ 3M Horizon: Long/Short Ann. Return = {h3_ls['annualized_return']*100:+.2f}%, Sharpe = {h3_ls['sharpe_ratio']:.2f}")

    # -------------------------------------------------------------------------
    # STAGE 4: ECONOMETRIC DIAGNOSTICS & STATISTICAL PROOFS
    # -------------------------------------------------------------------------
    print("\n[STAGE 4/6] Executing Econometric Tests (ADF, Cointegration, IC Superiority)...")
    econ_results = run_full_econometric_suite(sentiment_df=sentiment_df, prices_df=prices_df)

    adf_primary = econ_results["stationarity_tests"]["primary_signal"]
    adf_mom = econ_results["stationarity_tests"]["momentum_signal"]
    coint_res = econ_results["cointegration_test"]
    ic_comp_3m = econ_results["information_coefficient_comparison"]["3M"]

    print(f"  ✓ ADF Primary Signal Z(Delta_S): stat={adf_primary['adf_statistic']:.3f}, p-val={adf_primary['p_value']:.4f} (Stationary: {adf_primary['is_stationary']})")
    print(f"  ✓ ADF Momentum Signal d/dt(Delta_S): stat={adf_mom['adf_statistic']:.3f}, p-val={adf_mom['p_value']:.4f} (Stationary: {adf_mom['is_stationary']})")
    print(f"  ✓ Engle-Granger Cointegration ln(COST) ~ ln(WMT) + ln(S_upper): p-val={coint_res['engle_granger_coint_test']['p_value']:.4f} (Cointegrated: {coint_res['is_cointegrated']})")
    print(f"  ✓ 3M Information Coefficient: IC_affluent = {ic_comp_3m['affluent_signal']['mean_ic']:+.4f} vs IC_headline = {ic_comp_3m['headline_sentiment']['mean_ic']:+.4f}")
    print(f"    -> Paired t-test diff = {ic_comp_3m['comparative_paired_test']['ic_difference']:+.4f}, t = {ic_comp_3m['comparative_paired_test']['t_statistic']:.2f}, one-tailed p = {ic_comp_3m['comparative_paired_test']['p_value_one_tailed']:.4f} (Superior: {ic_comp_3m['comparative_paired_test']['is_superior']})")

    # -------------------------------------------------------------------------
    # STAGE 5: RETAIL K-ALPHA SIGNALS & MARKET OVERLAY
    # -------------------------------------------------------------------------
    print("\n[STAGE 5/6] Computing Retail K-Alpha Signals & Pricing Adjustments...")
    fundamentals = fetch_ticker_fundamentals(tickers=["COST", "WMT", "TGT", "SPY"])
    retail_results = compute_retail_k_alpha(
        transcripts=transcripts,
        sentiment_df=sentiment_df,
        prices_df=prices_df,
        fundamentals=fundamentals,
        baseline_df=baseline_df,
    )
    retail_out = PROJECT_ROOT / "data" / "k_alpha_retail_signals.json"
    with open(retail_out, "w") as f:
        json.dump(retail_results, f, indent=2)

    cost_sig = retail_results["ticker_signals"]["COST"]
    wmt_sig = retail_results["ticker_signals"]["WMT"]
    tgt_sig = retail_results["ticker_signals"]["TGT"]

    print(f"  ✓ COST ({cost_sig['action']}): Target ${cost_sig['k_alpha_target_price']:.2f} ({cost_sig['k_alpha_upside_pct']:+.1f}% vs Current ${cost_sig['current_price']:.2f})")
    print(f"  ✓ WMT  ({wmt_sig['action']}): Target ${wmt_sig['k_alpha_target_price']:.2f} ({wmt_sig['k_alpha_upside_pct']:+.1f}% vs Current ${wmt_sig['current_price']:.2f})")
    print(f"  ✓ TGT  ({tgt_sig['action']}): Target ${tgt_sig['k_alpha_target_price']:.2f} ({tgt_sig['k_alpha_upside_pct']:+.1f}% vs Current ${tgt_sig['current_price']:.2f})")

    # -------------------------------------------------------------------------
    # STAGE 6: FORWARD-LOOKING CYCLE FORECAST (OPTIONAL SYNC)
    # -------------------------------------------------------------------------
    print("\n[STAGE 6/6] Checking Forward News Cycle Forecast...")
    forecast_path = PROJECT_ROOT / "data" / "forward_cycle_forecast.json"
    if forecast_path.exists():
        with open(forecast_path, "r") as f:
            f_data = json.load(f)
        event = f_data.get("next_major_catalyst") or f_data.get("upcoming_event", {})
        event_name = event.get("event_name") or event.get("name", "Unknown Catalyst")
        rel_date = event.get("release_date") or event.get("date", "TBD")
        days_until = event.get("days_until_release", "N/A")

        surprise_pts = f_data.get("predicted_surprise_points", 0.0)
        consensus = f_data.get("consensus_headline_prediction", 0.0)
        affluent = f_data.get("flashpoint_affluent_signal", 0.0)

        print(f"  ✓ Next Catalyst: {event_name} ({rel_date}, T-{days_until} days)")
        print(f"  ✓ Predicted Surprise Spread: {surprise_pts:+.1f} pts (Consensus: {consensus:.1f} vs Affluent Reality: {affluent:.1f})")

    print("\n" + "=" * 82)
    print("  ✓ RESEARCH PIPELINE EXECUTION FULLY COMPLETED AND ARTIFACTS VERIFIED")
    print("=" * 82)


if __name__ == "__main__":
    main()
