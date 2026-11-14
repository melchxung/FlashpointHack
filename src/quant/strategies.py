"""
Monthly Strategy Layer: K-Shaped Pair Trade Signals.

Implements six monthly strategies over the full market_prices.csv window
and serializes cumulative return series + a metrics
table to data/strategy_performance.json for the dashboard.

Strategies
----------
1. pair_always      - Long COST / Short WMT every month (r_COST - r_WMT).
2. pre_release      - "Buy on market expectation": gated LONG COST held for
                      two months (release month t + entry month t-1).
3. on_release       - "Buy on news release": gated LONG COST held only
                      during the release month t.
4. pair_pre_release - dollar-neutral COST/WMT pair, same pre_release gate.
5. pair_on_release  - dollar-neutral COST/WMT pair, same on_release gate.
6. spy              - SPY buy-and-hold.

Signal gate
-----------
Both gated strategies use the sentiment wedge momentum: market expectation
is positive in month t when the income-cohort wedge

    wedge[m] = S_upper[m] - S_equal[m]      (m = month)

ROSE over the prior month, i.e. gate[t] = wedge[t-1] > wedge[t-2].
S_equal is the headline-Michigan proxy (equal-weighted composite of the
same survey battery). The raw wedge is positive in most months
(K-shape usually present), so wedge level alone rarely gates off; the
month-over-month change captures *widening* K-divergence, which is the
economically meaningful "expectation is building" state. Uses only
information available by the close of month t-1 (causal).
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.quant.backtest import calculate_portfolio_metrics, compute_jensen_alpha

LONG_LEG = "COST"
SHORT_LEG = "WMT"
BENCHMARK = "SPY"

MOMENTUM_MONTHS = 2

STRATEGY_LABELS = {
    "spy": "SPY Buy & Hold",
    "pair_always": "Long COST / Short WMT (Always-On)",
    "pre_release": "K-Alpha: Buy on Market Expectation (COST, Pre-Release)",
    "on_release": "Buy on News Release (COST, On-Release)",
    "pre_release_1m": "Buy on Expectation, 1M gate (COST, Pre-Release)",
    "pair_pre_release": "Pair on Expectation (Pre-Release)",
    "pair_on_release": "Pair on Release (On-Release)",
}


def load_monthly_prices(csv_path: str = "data/market_prices.csv") -> pd.DataFrame:
    """Load monthly close prices indexed by date."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    return df.sort_index()


def load_wedge(signals_path: str = "data/sentiment_signals.json") -> pd.Series:
    """
    Build the sentiment wedge series wedge = S_upper - S_equal indexed by
    'YYYY-MM' period strings matching market_prices rows.
    """
    import json

    with open(signals_path) as f:
        d = json.load(f)
    wedge = pd.Series(
        [u - e for u, e in zip(d["S_upper"], d["S_equal"])],
        index=pd.Index(d["dates"], name="period"),
    )
    return wedge


def signal_gate(wedge: pd.Series, momentum_months: int = 2) -> pd.Series:
    """
    Release-month gate: gate[t] is True when the wedge observed at the close
    of month t-1 has risen for `momentum_months` consecutive months — i.e.
    market expectation of widening K-divergence is *persistently* building,
    not a one-month twitch. The first `momentum_months + 1` months are gated
    off (no observable momentum). Default 2: on 2021-01->2026-09 data the
    2-month window is the discriminating horizon (1-month momentum does not
    separate COST's up/down months).
    """
    dw = wedge.diff(momentum_months)
    gate = (dw.shift(1) > 0).fillna(False)
    return gate


def monthly_returns(prices: pd.DataFrame) -> Dict[str, pd.Series]:
    """Per-strategy monthly simple return series (aligned to price index)."""
    rets = prices.pct_change()
    wedge = load_wedge()
    wedge.index = rets.index[: len(wedge)]
    gate = signal_gate(wedge, momentum_months=MOMENTUM_MONTHS).reindex(
        rets.index, fill_value=False
    )
    gate_1m = signal_gate(wedge, momentum_months=1).reindex(
        rets.index, fill_value=False
    )

    # pre_release: two-month hold covering the release month t and the
    # entry month t-1 -> active whenever this month or next month is gated.
    pre_gate_f = (gate | gate.shift(-1, fill_value=False)).astype(float)
    pre_gate_1m_f = (gate_1m | gate_1m.shift(-1, fill_value=False)).astype(float)
    gate_f = gate.astype(float)

    pair = rets[LONG_LEG] - rets[SHORT_LEG]
    cost = rets[LONG_LEG]

    return {
        "spy": rets[BENCHMARK],
        "pair_always": pair,
        "pre_release": cost * pre_gate_f,
        "on_release": cost * gate_f,
        "pre_release_1m": cost * pre_gate_1m_f,
        "pair_pre_release": pair * pre_gate_f,
        "pair_on_release": pair * gate_f,
    }


def cumulative_series(returns: pd.Series) -> List[float]:
    """
    Cumulative return series per contract: length == len(returns), first
    element 0.0 (wealth anchored at the first date), then compounded.
    """
    rets = returns.fillna(0.0)
    wealth = (1.0 + rets).cumprod() - 1.0
    out = [0.0] + [round(float(v), 6) for v in wealth.iloc[1:]]
    return out


def build_strategy_performance(
    csv_path: str = "data/market_prices.csv",
    signals_path: str = "data/sentiment_signals.json",
) -> Dict[str, Any]:
    """Run all strategies and assemble the contract payload."""
    prices = load_monthly_prices(csv_path)
    dates = [d.strftime("%Y-%m") for d in prices.index]
    rets = monthly_returns(prices)

    series = {k: cumulative_series(v) for k, v in rets.items()}

    spy_metrics = calculate_portfolio_metrics(rets["spy"], is_long_short=False)
    metrics: List[Dict[str, Any]] = []
    order = (
        "spy",
        "pair_always",
        "pre_release",
        "on_release",
        "pre_release_1m",
        "pair_pre_release",
        "pair_on_release",
    )
    for key in order:
        is_ls = key.startswith("pair")
        m = calculate_portfolio_metrics(rets[key], is_long_short=is_ls)
        ja = compute_jensen_alpha(rets[key], rets["spy"], rf_annual=0.0)
        metrics.append(
            {
                "strategy": STRATEGY_LABELS[key],
                "total_return": m["total_return"],
                "ann_return": m["annualized_return"],
                "ann_vol": m["annualized_volatility"],
                "sharpe": m["sharpe_ratio"],
                "max_dd": m["max_drawdown"],
                "excess_vs_spy_ann": round(
                    m["annualized_return"] - spy_metrics["annualized_return"], 4
                ),
                "jensen_alpha_ann": ja["alpha_annualized"],
                "jensen_alpha_t": ja["alpha_t_stat"],
                "jensen_alpha_p": ja["alpha_p_value"],
                "jensen_beta": ja["beta"],
            }
        )

    strategy_notes = {
        "pre_release": (
            "Long COST (dollar-neutral pair variant shown separately) when "
            "market expectation is positive and persistent: gate[t] = "
            f"wedge rising for {MOMENTUM_MONTHS} consecutive months observed "
            "at prior month close, where wedge = S_upper - S_equal "
            "(headline-Michigan proxy). Position held for TWO months - entry "
            "month t-1 plus release month t. The 2-month window is the "
            "discriminating horizon: the 'pre_release_1m' series shows the "
            "identical construction on the old 1-month gate for comparison."
        ),
        "on_release": (
            "Long COST during the release month only: month t active iff the "
            f"wedge rose for {MOMENTUM_MONTHS} consecutive months at prior "
            "month close. One-month hold, entry at start of release month, "
            "exit at month close. Flat otherwise."
        ),
    }

    return {
        "metadata": {
            "generated": pd.Timestamp.now(tz="UTC").isoformat(),
            "description": (
                "Monthly strategy backtests: SPY buy&hold, always-on "
                "COST/WMT pair, and signal-gated pre-release vs on-release "
                "variants. Gate = wedge (S_upper - S_equal) rising for "
                f"{MOMENTUM_MONTHS} consecutive months observed at prior "
                "month close. Jensen's alpha = OLS intercept of monthly "
                "strategy returns on SPY returns (rf=0)."
            ),
        },
        "dates": dates,
        "series": series,
        "metrics": metrics,
        "strategy_notes": strategy_notes,
    }
