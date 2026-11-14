#!/usr/bin/env python
"""
Retail K-Alpha Signal Generation and Market Overlay Script.

Quantifies the K-shaped divergence across retail leaders:
- COST (Costco): Right-tail high-earner winner -> Strong Buy
- WMT (Walmart): Left-tail trade-down underperformer -> Underperform / Sell
- TGT (Target): Middle-belly discretionary -> Neutral / Hold

Outputs comprehensive signals to data/k_alpha_retail_signals.json
for consumption by Streamlit (app.py) and verification.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant.factor import load_sentiment_signals
from src.quant.market_data import fetch_historical_prices, fetch_ticker_fundamentals
from src.quant.retail_alpha import (
    compute_retail_k_alpha,
    load_baseline_surveys,
    load_respondent_transcripts,
)


def main():
    print("=" * 80)
    print("  RETAIL K-ALPHA DIVERGENCE & MARKET OVERLAY (COST vs WMT vs TGT)")
    print("=" * 80)

    # 1. Load Data
    print("\n[1/5] Loading respondent transcripts, sentiment signals, and prices...")
    transcripts_path = PROJECT_ROOT / "data" / "respondent_scored_transcripts.json"
    transcripts = load_respondent_transcripts(transcripts_path)
    print(f"      Loaded {len(transcripts)} respondent transcripts across 3 retail cohorts.")

    sentiment_path = PROJECT_ROOT / "data" / "sentiment_signals.json"
    sentiment_df = load_sentiment_signals(sentiment_path)
    print(f"      Loaded {len(sentiment_df)} months of stratified sentiment time-series.")

    prices_path = PROJECT_ROOT / "data" / "market_prices.csv"
    prices_df = fetch_historical_prices(tickers=["COST", "WMT", "TGT", "SPY"], cache_path=prices_path)
    print(f"      Loaded {len(prices_df)} price periods for COST, WMT, TGT, SPY.")

    # 2. Fundamentals & Baselines
    print("\n[2/5] Ingesting ticker fundamentals and Chicago Fed Letter 521 baselines...")
    fundamentals = fetch_ticker_fundamentals(tickers=["COST", "WMT", "TGT", "SPY"])
    baseline_csv = PROJECT_ROOT / "data" / "baseline_sentiment_surveys.csv"
    baseline_json = PROJECT_ROOT / "data" / "baseline_correlations.json"
    baseline_df, baseline_stats = load_baseline_surveys(baseline_csv, baseline_json)

    # 3. Compute Retail K-Alpha
    print("\n[3/5] Computing Divergence Alpha and ticker price adjustments...")
    results = compute_retail_k_alpha(
        transcripts=transcripts,
        sentiment_df=sentiment_df,
        prices_df=prices_df,
        fundamentals=fundamentals,
        baseline_df=baseline_df,
    )

    # 4. Save results to JSON
    output_path = PROJECT_ROOT / "data" / "k_alpha_retail_signals.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[4/5] Successfully exported retail signals to: {output_path}")

    # 5. Display Executive Summary
    print("\n[5/5] RETAIL ALPHA EXECUTIVE SUMMARY:")
    print("-" * 80)
    print(f"{'Cohort':<18} | {'Avg HHI':<10} | {'Stated Sentiment':<16} | {'Revealed Spend':<16} | {'Vibecession Gap':<15}")
    print("-" * 80)
    for c_name, c_data in results["cohort_summary"].items():
        print(
            f"{c_name:<18} | ${c_data['avg_hhi']:>8,d} | {c_data['mean_stated_macro_sentiment']:>+16.3f} | "
            f"{c_data['mean_revealed_spend_conviction']:>+16.3f} | {c_data['vibecession_gap']:>+15.3f}"
        )
    print("-" * 80)

    print("\n>>> TICKER ACTIONABLE RATINGS & PRICING ADJUSTMENTS <<<")
    print("-" * 80)
    for ticker in ["COST", "WMT", "TGT"]:
        sig = results["ticker_signals"][ticker]
        perf = sig.get("performance", {})
        print(f"\n• Ticker: {ticker} ({sig['name']})")
        print(f"  K-Shaped Role:      {sig['k_role']}")
        print(f"  Signal Action:      [{sig['action']}] (Rating: {sig['rating_code']})")
        print(f"  Current Price:      ${sig['current_price']:.2f}")
        print(f"  Street Target:      ${sig['street_target_price']:.2f}")
        print(f"  K-Alpha Target:     ${sig['k_alpha_target_price']:.2f} ({sig['k_alpha_upside_pct']:+.1f}% vs Current)")
        print(f"  Forward P/E:        {sig['forward_pe']:.1f}x")
        print(f"  Historical Return:  Total = {perf.get('total_return', 0)*100:+.1f}%, Ann. = {perf.get('annualized_return', 0)*100:+.1f}%, Sharpe = {perf.get('sharpe_ratio', 0):.2f}")
        print(f"  Thesis:             {sig['investment_thesis']}")

    pair = results.get("pair_trade_backtest", {})
    if pair:
        print("\n" + "-" * 80)
        print(">>> PAIR TRADE OVERLAY: Long COST (Right Tail) / Short WMT (Left Tail) <<<")
        print(f"  Annualized Spread Return: {pair['annualized_spread_return']*100:+.2f}%")
        print(f"  Annualized Spread Vol:    {pair['annualized_volatility']*100:.2f}%")
        print(f"  Pair Sharpe Ratio:        {pair['sharpe_ratio']:.2f}")
        print(f"  t-statistic on Alpha:     {pair['t_stat']:.2f} (p-value: {pair['p_value']:.4f})")
        print(f"  Cumulative Spread Gain:   {pair['cumulative_spread_return']*100:+.2f}%")

    print("\n" + "-" * 80)
    print(">>> BENCHMARK: K-Alpha vs 7 Baseline Surveys (Chicago Fed Letter 521) <<<")
    print(f"{'Survey Name':<38} | {'Correlation w/ Real PCE':<24} | {'Directionality':<20}")
    print("-" * 80)
    for row in results["survey_comparison_table"]:
        print(f"{row['survey']:<38} | {row['pce_correlation']:>+24.3f} | {row['directionality']:<20}")
    print("-" * 80)

    print("\nRetail K-Alpha execution successfully completed!")


if __name__ == "__main__":
    main()
