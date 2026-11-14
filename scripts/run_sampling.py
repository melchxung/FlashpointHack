#!/usr/bin/env python3
"""
Run Synthetic Persona Sampling & Generate K-Shaped Sentiment Signals.

Fields the 5-question University of Michigan sentiment battery across
stratified consumer cohorts (Upper Quartile, Middle Income, Lower Quartile)
from 2021-01 to 2024-12 using the Flashpoint AI sampling engine.
Outputs results to data/sentiment_signals.json and data/sentiment_signals.csv.
"""

import sys
import os
import json
from pathlib import Path
import pandas as pd

# Ensure repository root is on Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.sampler.client import FlashpointClient
from src.sampler.engine import SentimentSamplingEngine


def main():
    print("=" * 78)
    print("  K-SHAPED CONSUMER SENTIMENT SAMPLING ENGINE (FLASHPOINT AI / TOUCHPOINT)")
    print("=" * 78)

    # 1. Initialize client and engine
    client = FlashpointClient(fallback_to_simulator=True)
    mode = "Live Flashpoint API" if client.has_credentials else "High-Fidelity Synthetic Simulator (Zero-API Key Fallback)"
    print(f"\n[1] Client Mode: {mode}")

    engine = SentimentSamplingEngine(
        client=client,
        weight_upper=0.48,   # Top 20-25% drives ~48% of aggregate PCE
        weight_middle=0.36,  # Middle 40% drives ~36% of aggregate PCE
        weight_lower=0.16,   # Bottom 35% drives ~16% of aggregate PCE
    )

    print(f"    Macro PCE Consumption Weights:")
    print(f"    - Upper Quartile (Top 20-25%): {engine.weight_upper:.1%}")
    print(f"    - Middle Income (40-70%):       {engine.weight_middle:.1%}")
    print(f"    - Lower Quartile (Bottom 35%):  {engine.weight_lower:.1%}")

    # 2. Run historical time-series sampling (2021-01 to 2024-12: 48 monthly periods)
    print("\n[2] Executing monthly sampling across 48 periods (2021-01 -> 2024-12)...")
    print("    Sampling 50 synthetic respondents per cohort per period (7,200 total completions)...")

    result = engine.sample_historical_time_series(
        start_date="2021-01",
        end_date="2024-12",
        num_respondents_per_cohort=50,
    )

    df = result.to_dataframe()

    # 3. Save outputs
    data_dir = REPO_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / "sentiment_signals.json"
    csv_path = data_dir / "sentiment_signals.csv"

    engine.save_to_json(result, str(json_path))
    engine.save_to_csv(result, str(csv_path))

    print(f"\n[3] Successfully exported signals to:")
    print(f"    - JSON: {json_path} ({os.path.getsize(json_path) / 1024:.1f} KB)")
    print(f"    - CSV:  {csv_path} ({os.path.getsize(csv_path) / 1024:.1f} KB)")

    # 4. Display Summary Analytics
    print("\n[4] SUMMARY STATISTICS (2021 - 2024):")
    print("-" * 78)
    print(f"{'Metric':<32} | {'Mean':>8} | {'Min':>8} | {'Max':>8} | {'Std Dev':>8}")
    print("-" * 78)
    for col in ["S_upper", "S_middle", "S_lower", "S_equal", "S_K", "Delta_S"]:
        s = df[col]
        print(f"{col:<32} | {s.mean():>8.2f} | {s.min():>8.2f} | {s.max():>8.2f} | {s.std():>8.2f}")
    print("-" * 78)

    # 5. Display Key Divergence Highlights
    print("\n[5] K-SHAPED REGIME HIGHLIGHTS:")
    sample_dates = ["2021-06", "2022-06", "2023-06", "2024-06"]
    for d in sample_dates:
        row = df.loc[d]
        if hasattr(row, "iloc") and isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        s_u = float(row['S_upper'])
        s_l = float(row['S_lower'])
        delta = float(row['Delta_S'])
        s_k = float(row['S_K'])
        s_eq = float(row['S_equal'])
        print(
            f"    Date {d}:  S_upper = {s_u:5.1f}  |  "
            f"S_lower = {s_l:5.1f}  |  "
            f"Delta_S (Spread) = {delta:+5.1f}  |  "
            f"S_K (Spending-Wt) = {s_k:5.1f}  |  "
            f"S_equal = {s_eq:5.1f}"
        )

    # Validate assertions
    assert json_path.exists(), "JSON output missing"
    assert csv_path.exists(), "CSV output missing"
    assert len(result.dates) == 48, f"Expected 48 periods, got {len(result.dates)}"
    assert all(d > 0 for d in df["Delta_S"]), "Upper sentiment should consistently exceed lower sentiment"

    print("\n" + "=" * 78)
    print("  ✓ SAMPLING PIPELINE EXECUTION COMPLETE AND VERIFIED")
    print("=" * 78)


if __name__ == "__main__":
    main()
