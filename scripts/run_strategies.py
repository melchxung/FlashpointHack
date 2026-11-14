#!/usr/bin/env python
"""
Strategy Performance Runner.

Runs the four monthly strategies in src/quant/strategies.py over
data/market_prices.csv (2021-01 -> 2024-12), writes
data/strategy_performance.json, and prints the metrics table.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant.strategies import build_strategy_performance

OUT_PATH = PROJECT_ROOT / "data" / "strategy_performance.json"


def main() -> None:
    payload = build_strategy_performance()

    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    # --- Metrics table to stdout ---
    rows = payload["metrics"]
    header = f"{'STRATEGY':<46}{'TOTAL':>9}{'ANN':>9}{'VOL':>9}{'SHARPE':>8}{'MAXDD':>9}{'XS-SPY':>9}"
    print(header)
    print("-" * len(header))
    for m in rows:
        print(
            f"{m['strategy']:<46}"
            f"{m['total_return']:>8.1%} "
            f"{m['ann_return']:>8.1%} "
            f"{m['ann_vol']:>8.1%} "
            f"{m['sharpe']:>7.2f} "
            f"{m['max_dd']:>8.1%} "
            f"{m['excess_vs_spy_ann']:>+8.1%}"
        )

    n = len(payload["dates"])
    for k, s in payload["series"].items():
        assert len(s) == n and s[0] == 0.0, f"contract violation in series '{k}'"
    print(f"\nWrote {OUT_PATH} ({n} periods, {len(payload['series'])} series)")


if __name__ == "__main__":
    main()
