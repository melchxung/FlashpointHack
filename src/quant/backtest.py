"""
Cross-Sectional Factor Backtester and Portfolio Analytics.

Constructs:
- Quintile portfolios (Q1 to Q5) and Long/Short Spread (Q5 - Q1)
- 1-Month and 3-Month forward return analysis
- Sharpe ratio, Sortino ratio, Max Drawdown, t-statistics, and cumulative alpha
- Monotonicity checks and factor decay evaluation
- Full JSON serialization for reporting
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.quant.factor import (
    assign_quintiles,
    calculate_derived_macro_factors,
    compute_cross_sectional_factor_scores,
    compute_information_coefficients,
    load_sentiment_signals,
)
from src.quant.market_data import compute_forward_returns, fetch_historical_prices
from src.quant.universe import ASSET_MAP, MODEL_TICKERS, WINNER_TICKERS, LOSER_TICKERS


def calculate_portfolio_metrics(
    returns: pd.Series,
    rf_annual: float = 0.02,
    periods_per_year: int = 12,
    is_long_short: bool = False,
) -> Dict[str, float]:
    """
    Compute institutional performance and risk statistics for a return series.
    """
    clean_ret = returns.dropna()
    n = len(clean_ret)
    if n < 3:
        return {
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "t_stat": 0.0,
            "p_value": 1.0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "periods": n,
        }

    # Cumulative growth
    cum_wealth = (1.0 + clean_ret).cumprod()
    total_return = float(cum_wealth.iloc[-1] - 1.0)
    years = n / periods_per_year

    # Annualized return (geometric)
    if cum_wealth.iloc[-1] > 0 and years > 0:
        annualized_return = float(cum_wealth.iloc[-1] ** (1.0 / years) - 1.0)
    else:
        annualized_return = float(clean_ret.mean() * periods_per_year)

    # Annualized volatility
    vol_annual = float(clean_ret.std(ddof=1) * np.sqrt(periods_per_year))

    # Excess return for Sharpe
    rf_period = 0.0 if is_long_short else (rf_annual / periods_per_year)
    excess_ret = clean_ret - rf_period
    mean_excess = float(excess_ret.mean())

    sharpe = float((mean_excess * np.sqrt(periods_per_year)) / (clean_ret.std(ddof=1) if clean_ret.std(ddof=1) > 1e-6 else 1e-6))

    # Downside deviation for Sortino
    neg_ret = clean_ret[clean_ret < 0]
    downside_std = float(neg_ret.std(ddof=1) * np.sqrt(periods_per_year)) if len(neg_ret) > 1 else vol_annual
    sortino = float((annualized_return - (0.0 if is_long_short else rf_annual)) / (downside_std if downside_std > 1e-6 else 1e-6))

    # Drawdown
    peak = cum_wealth.cummax()
    drawdown = (cum_wealth - peak) / peak
    max_dd = float(abs(drawdown.min()))
    calmar = float(annualized_return / max_dd) if max_dd > 1e-4 else 0.0

    # Alpha t-stat against zero mean
    t_res = stats.ttest_1samp(clean_ret, 0.0)
    t_stat = float(t_res.statistic) if not np.isnan(t_res.statistic) else 0.0
    p_val = float(t_res.pvalue) if not np.isnan(t_res.pvalue) else 1.0

    # Win rate
    win_rate = float((clean_ret > 0).mean())

    return {
        "annualized_return": round(annualized_return, 4),
        "annualized_volatility": round(vol_annual, 4),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "max_drawdown": round(max_dd, 4),
        "calmar_ratio": round(calmar, 3),
        "t_stat": round(t_stat, 3),
        "p_value": round(p_val, 4),
        "win_rate": round(win_rate, 4),
        "total_return": round(total_return, 4),
        "periods": int(n),
    }


def build_quintile_returns(
    quintiles: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    num_quantiles: int = 5,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Construct time series of monthly returns for each quintile (Q1..Q5)
    and the Long/Short Spread portfolio (Q5 - Q1).
    """
    common_dates = quintiles.index.intersection(fwd_returns.index)
    q_returns = {q: [] for q in range(1, num_quantiles + 1)}
    dates = []

    for date in common_dates:
        q_row = quintiles.loc[date].dropna()
        r_row = fwd_returns.loc[date].dropna()
        common_tickers = q_row.index.intersection(r_row.index)

        if len(common_tickers) < num_quantiles:
            continue

        dates.append(date)
        for q in range(1, num_quantiles + 1):
            q_assets = common_tickers[q_row[common_tickers] == q]
            if len(q_assets) > 0:
                ret = r_row[q_assets].mean()
            else:
                ret = 0.0
            q_returns[q].append(ret)

    q_df = pd.DataFrame(q_returns, index=dates)
    q_df.columns = [f"Q{q}" for q in q_df.columns]

    # Long/Short Spread: Long Q5, Short Q1
    ls_series = q_df[f"Q{num_quantiles}"] - q_df["Q1"]
    ls_series.name = "Long_Short_Spread"

    return q_df, ls_series


def run_factor_backtest(
    sentiment_df: Optional[pd.DataFrame] = None,
    prices_df: Optional[pd.DataFrame] = None,
    start_date: str = "2021-01-01",
    end_date: str = "2024-12-31",
) -> Dict[str, Any]:
    """
    Run full cross-sectional factor backtest pipeline.
    Produces 1-month and 3-month forward horizon performance, IC analysis,
    and cumulative return profiles.
    """
    # 1. Load or fetch data
    if prices_df is None:
        prices_df = fetch_historical_prices(start_date=start_date, end_date=end_date)
    if sentiment_df is None:
        sentiment_df = load_sentiment_signals()

    # 2. Derive macro factors
    macro_factors = calculate_derived_macro_factors(sentiment_df)

    # 3. Compute cross-sectional scores and quintile assignments
    factor_scores = compute_cross_sectional_factor_scores(macro_factors, prices_df)
    quintiles = assign_quintiles(factor_scores, num_quantiles=5)

    # 4. Compute forward returns
    fwd_returns_dict = compute_forward_returns(prices_df, horizons=(1, 3))

    results: Dict[str, Any] = {
        "metadata": {
            "start_date": str(prices_df.index[0].date()),
            "end_date": str(prices_df.index[-1].date()),
            "total_assets": len(MODEL_TICKERS),
            "k_winners_count": len(WINNER_TICKERS),
            "k_losers_count": len(LOSER_TICKERS),
            "rebalance_frequency": "Monthly",
            "quintiles": 5,
        },
        "macro_factors_summary": {
            "mean_delta_s": round(float(macro_factors["delta_s"].mean()), 2),
            "max_delta_s": round(float(macro_factors["delta_s"].max()), 2),
            "min_delta_s": round(float(macro_factors["delta_s"].min()), 2),
            "current_delta_s": round(float(macro_factors["delta_s"].iloc[-1]), 2),
            "mean_upper_sentiment": round(float(macro_factors["s_upper"].mean()), 2),
            "mean_median_sentiment": round(float(macro_factors["s_median"].mean()), 2),
            "mean_lower_sentiment": round(float(macro_factors["s_lower"].mean()), 2),
            "spending_weighted_sentiment": round(float(macro_factors["s_k"].mean()), 2),
        },
        "horizons": {},
    }

    # 5. Evaluate each horizon (1-Month and 3-Month)
    for h in [1, 3]:
        fwd_ret = fwd_returns_dict[h]
        q_df, ls_series = build_quintile_returns(quintiles, fwd_ret, num_quantiles=5)

        # SPY Benchmark comparison
        spy_fwd = fwd_ret["SPY"].reindex(q_df.index).dropna() if "SPY" in fwd_ret.columns else pd.Series()
        spy_metrics = calculate_portfolio_metrics(spy_fwd, periods_per_year=12 // h) if not spy_fwd.empty else {}

        # Long/Short Metrics
        ls_metrics = calculate_portfolio_metrics(ls_series, periods_per_year=12 // h, is_long_short=True)

        # Quintile Metrics
        quintile_metrics = {}
        for col in q_df.columns:
            m = calculate_portfolio_metrics(q_df[col], periods_per_year=12 // h)
            quintile_metrics[col] = m

        # Information Coefficient (IC)
        ic_df = compute_information_coefficients(factor_scores, fwd_ret)
        mean_ic = float(ic_df["IC"].mean()) if not ic_df.empty else 0.0
        std_ic = float(ic_df["IC"].std(ddof=1)) if not ic_df.empty else 1.0
        ir_ic = float(mean_ic / std_ic) if std_ic > 1e-4 else 0.0
        t_ic = float(stats.ttest_1samp(ic_df["IC"].dropna(), 0.0).statistic) if not ic_df.empty else 0.0
        p_ic = float(stats.ttest_1samp(ic_df["IC"].dropna(), 0.0).pvalue) if not ic_df.empty else 1.0

        # Cumulative performance series
        cum_q = ((1.0 + q_df).cumprod() - 1.0).round(4).to_dict()
        cum_ls = ((1.0 + ls_series).cumprod() - 1.0).round(4).to_dict()
        cum_spy = ((1.0 + spy_fwd).cumprod() - 1.0).round(4).to_dict() if not spy_fwd.empty else {}

        # Monotonicity check
        mean_q_rets = [quintile_metrics[f"Q{q}"]["annualized_return"] for q in range(1, 6)]
        monotonic_pairs = sum(1 for i in range(len(mean_q_rets) - 1) if mean_q_rets[i + 1] >= mean_q_rets[i])
        monotonicity_score = monotonic_pairs / (len(mean_q_rets) - 1)

        results["horizons"][f"{h}M"] = {
            "long_short_metrics": ls_metrics,
            "benchmark_metrics": spy_metrics,
            "quintile_metrics": quintile_metrics,
            "information_coefficient": {
                "mean_ic": round(mean_ic, 4),
                "std_ic": round(std_ic, 4),
                "ic_ir": round(ir_ic, 3),
                "t_stat": round(t_ic, 3),
                "p_value": round(p_ic, 4),
            },
            "monotonicity_score": round(monotonicity_score, 2),
            "cumulative_returns": {
                "quintiles": {k: {str(dt.date()): val for dt, val in v.items()} for k, v in cum_q.items()},
                "long_short": {str(dt.date()): val for dt, val in cum_ls.items()},
                "benchmark": {str(dt.date()): val for dt, val in cum_spy.items()},
            },
        }

    return results
