#!/usr/bin/env python3
"""
Run Forward-Looking Economic News Cycle Forecaster.

Computes the macroeconomic news calendar release schedule,
forecasts the upcoming sentiment release surprise (Consensus vs Affluent),
and outputs actionable trade signals to data/forward_cycle_forecast.json.
"""

import sys
import os
import json
from pathlib import Path
from datetime import date

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.forecast.cycle_forecaster import EconomicNewsCycleForecaster, generate_current_cycle_forecast


def main():
    print("=" * 78)
    print("  ECONOMIC NEWS CYCLE FORECASTER & AHEAD-OF-THE-PRINT TRADING SIGNALS")
    print("=" * 78)

    forecaster = EconomicNewsCycleForecaster()
    forecast = forecaster.generate_cycle_forecast(
        reference_date=date(2026, 11, 14),
        consensus_headline_estimate=70.5,
    )

    data_dir = REPO_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / "forward_cycle_forecast.json"

    with open(json_path, "w") as f:
        json.dump(forecast.to_dict(), f, indent=2)

    print(f"\n[1] Reference Date: {forecast.reference_date}")
    print(f"    Cycle: {forecast.cycle_name}")

    print("\n[2] UPCOMING MACROECONOMIC SENTIMENT RELEASES:")
    print("-" * 78)
    print(f"{'Release Event':<42} | {'Date':<10} | {'Time':<12} | {'Countdown'}")
    print("-" * 78)
    for evt in forecast.upcoming_releases[:4]:
        imm = " [NEXT]" if evt["is_imminent"] else ""
        print(f"{evt['event_name'][:42]:<42} | {evt['release_date']} | {evt['release_time']:<12} | {evt['days_until_release']:>3} days{imm}")
    print("-" * 78)

    print("\n[3] FORWARD RELEASE PREDICTION & SPREAD ANALYSIS:")
    print(f"    - Consensus Wall Street Headline Estimate: {forecast.consensus_headline_prediction:.1f}")
    print(f"    - Flashpoint Affluent Revealed Reality (S_upper): {forecast.flashpoint_affluent_signal:.1f}")
    print(f"    - Predicted Release Surprise Spread:       +{forecast.predicted_surprise_points:.1f} pts")
    print(f"    - Normalized Surprise Z-Score:              +{forecast.predicted_surprise_zscore:.2f}")
    print(f"    - Economic Regime:                          {forecast.surprise_regime}")

    print("\n[4] ACTIONABLE TICKER SIGNALS AHEAD OF THE PRINT:")
    print("-" * 78)
    print(f"{'Ticker':<6} | {'Recommendation':<22} | {'Target Adj':<10} | {'Conviction'}")
    print("-" * 78)
    for ticker, sig in forecast.ticker_signals.items():
        print(f"{ticker:<6} | {sig['recommendation']:<22} | {sig['price_target_adjustment']:<10} | {sig['conviction_score']:.0%}")
        print(f"       -> Catalyst: {sig['catalyst_timeline']}")
        print(f"       -> Thesis:   {sig['strategic_thesis'][:70]}...")
    print("-" * 78)

    pt = forecast.pair_trade_recommendation
    print(f"\n[5] RECOMMENDED PAIR TRADE:")
    print(f"    Strategy:      {pt['strategy']}")
    print(f"    Allocation:    {pt['recommended_allocation']}")
    print(f"    Target Alpha:  {pt['projected_cycle_alpha']} (Sharpe: {pt['annualized_sharpe_ratio']})")
    print(f"    Risk Mgmt:     Stop-Loss {pt['stop_loss_threshold']} / Take-Profit {pt['take_profit_threshold']}")

    print(f"\n[6] Saved complete forecast payload to:")
    print(f"    {json_path} ({os.path.getsize(json_path) / 1024:.1f} KB)")
    print("\n" + "=" * 78)
    print("  ✓ NEWS CYCLE FORECAST GENERATION COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
