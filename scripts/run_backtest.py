#!/usr/bin/env python
"""
Backtest Execution Script for K-Shaped Macro Factor Model.

Loads sentiment signals from Touchpoint AI / Michigan survey,
evaluates cross-sectional equity factor performance over 1M and 3M horizons,
and writes comprehensive backtest metrics to data/backtest_results.json.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant.backtest import run_factor_backtest
from src.quant.factor import load_sentiment_signals
from src.quant.market_data import fetch_historical_prices
from src.quant.universe import MODEL_TICKERS, WINNER_TICKERS, LOSER_TICKERS


def main():
    print("=" * 78)
    print("  K-SHAPED CONSUMER MACRO EQUITY FACTOR BACKTEST")
    print("=" * 78)

    # 1. Load data
    print("\n[1/4] Loading sentiment signals and asset prices...")
    sentiment_path = PROJECT_ROOT / "data" / "sentiment_signals.json"
    sentiment_df = load_sentiment_signals(sentiment_path)
    prices_df = fetch_historical_prices(cache_path=PROJECT_ROOT / "data" / "market_prices.csv")

    print(f"      Sentiment periods: {len(sentiment_df)} months ({sentiment_df.index[0].date()} to {sentiment_df.index[-1].date()})")
    print(f"      Universe size: {len(MODEL_TICKERS)} assets ({len(WINNER_TICKERS)} K-winners, {len(LOSER_TICKERS)} K-losers)")
    print(f"      Prices periods: {len(prices_df)} months")

    # 2. Run Backtest
    print("\n[2/4] Executing cross-sectional ranking and quintile backtesting...")
    results = run_factor_backtest(sentiment_df=sentiment_df, prices_df=prices_df)

    # 3. Output results
    output_path = PROJECT_ROOT / "data" / "backtest_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[3/4] Saved backtest metrics to: {output_path}")

    # 4. Display Performance Summary Table
    print("\n[4/4] PERFORMANCE SUMMARY:")
    print("-" * 78)
    macro = results["macro_factors_summary"]
    print(f"Macro Sentiment Spread (Delta_S = S_upper - S_median):")
    print(f"  Mean Spread: {macro['mean_delta_s']} pts | Max Spread: {macro['max_delta_s']} pts | Current: {macro['current_delta_s']} pts")
    print(f"  Upper Quartile Sentiment Mean: {macro['mean_upper_sentiment']} | Median Sentiment Mean: {macro['mean_median_sentiment']}")
    print("-" * 78)

    for h in ["1M", "3M"]:
        h_data = results["horizons"][h]
        ls = h_data["long_short_metrics"]
        ic = h_data["information_coefficient"]
        bm = h_data.get("benchmark_metrics", {})

        print(f"\n>>> HORIZON: {h} Forward Returns <<<")
        print(f"{'Portfolio / Metric':<25} | {'Ann. Ret':<10} | {'Ann. Vol':<10} | {'Sharpe':<8} | {'Max DD':<8} | {'t-stat':<8}")
        print("-" * 78)
        print(f"{'Long/Short Spread (Q5-Q1)':<25} | {ls['annualized_return']*100:>9.2f}% | {ls['annualized_volatility']*100:>9.2f}% | {ls['sharpe_ratio']:>8.2f} | {ls['max_drawdown']*100:>7.2f}% | {ls['t_stat']:>8.2f}")

        if bm:
            print(f"{'Benchmark (SPY)':<25} | {bm.get('annualized_return', 0)*100:>9.2f}% | {bm.get('annualized_volatility', 0)*100:>9.2f}% | {bm.get('sharpe_ratio', 0):>8.2f} | {bm.get('max_drawdown', 0)*100:>7.2f}% | {'N/A':>8}")

        print("\nQuintile Breakdown:")
        for q in range(1, 6):
            qm = h_data["quintile_metrics"][f"Q{q}"]
            tag = " (Short Leg / Losers)" if q == 1 else " (Long Leg / Winners)" if q == 5 else ""
            print(f"  Quintile {q}{tag:<22} : Ann. Return = {qm['annualized_return']*100:>6.2f}%, Vol = {qm['annualized_volatility']*100:>6.2f}%, Sharpe = {qm['sharpe_ratio']:>6.2f}")

        print(f"\nFactor Information Coefficient (Rank IC):")
        print(f"  Mean IC: {ic['mean_ic']:+.4f} | IC IR: {ic['ic_ir']:.2f} | t-stat: {ic['t_stat']:.2f} (p-val: {ic['p_value']:.4f})")
        print(f"  Monotonicity Score: {h_data['monotonicity_score']*100:.0f}%")
        print("-" * 78)

    print("\nBacktest execution successfully completed!")


if __name__ == "__main__":
    main()
