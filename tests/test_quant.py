"""
Unit tests for the Quantitative Factor and Cross-Sectional Asset Pricing Model.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.quant.backtest import (
    build_quintile_returns,
    calculate_portfolio_metrics,
    run_factor_backtest,
)
from src.quant.factor import (
    assign_quintiles,
    calculate_derived_macro_factors,
    compute_cross_sectional_factor_scores,
    compute_information_coefficients,
    load_sentiment_signals,
)
from src.quant.market_data import (
    compute_forward_returns,
    fetch_historical_prices,
)
def _sentiment(start_date: str, end_date: str) -> pd.DataFrame:
    """Real Flashpoint sentiment signals sliced to a date range (no simulated data)."""
    return load_sentiment_signals().loc[start_date:end_date]


from src.quant.universe import (
    ALL_TICKERS,
    ASSET_MAP,
    MODEL_TICKERS,
    WINNER_TICKERS,
    LOSER_TICKERS,
    UNIVERSE_ASSETS,
    get_universe_metadata,
)


def test_universe_definitions():
    """Verify that universe assets are properly categorized without duplicates."""
    tickers = [a.ticker for a in UNIVERSE_ASSETS]
    assert len(tickers) == len(set(tickers)), "Duplicate tickers found in universe!"
    assert len(WINNER_TICKERS) >= 8, "Expected at least 8 K-winners."
    assert len(LOSER_TICKERS) >= 6, "Expected at least 6 K-losers."
    assert "SPY" in ALL_TICKERS, "SPY benchmark must be in ALL_TICKERS."
    assert "SPY" not in MODEL_TICKERS, "SPY benchmark should not be in MODEL_TICKERS."

    for ticker, asset in ASSET_MAP.items():
        if asset.asset_type == "k_winner":
            assert asset.k_exposure_score > 0, f"Winner {ticker} must have positive exposure score."
        elif asset.asset_type == "k_loser":
            assert asset.k_exposure_score < 0, f"Loser {ticker} must have negative exposure score."

    metadata = get_universe_metadata()
    assert len(metadata) == len(UNIVERSE_ASSETS)


def test_market_data_forward_returns():
    """Verify forward return alignment without lookahead bias on cached real prices."""
    prices = fetch_historical_prices(tickers=["AAPL", "DLTR", "SPY"])
    assert not prices.empty
    assert (prices > 0).all().all()

    # Forward return test
    fwd = compute_forward_returns(prices, horizons=(1, 3))
    assert 1 in fwd
    assert 3 in fwd

    # At row t, 1M forward return should equal (Price_{t+1} / Price_t) - 1
    p0 = prices.iloc[0]["AAPL"]
    p1 = prices.iloc[1]["AAPL"]
    expected_fwd_1m = (p1 / p0) - 1.0
    actual_fwd_1m = fwd[1].iloc[0]["AAPL"]
    assert np.isclose(expected_fwd_1m, actual_fwd_1m, atol=1e-6)

    # Last row of 1M fwd return must be NaN (future price unknown)
    assert np.isnan(fwd[1].iloc[-1]["AAPL"])
    # Last 3 rows of 3M fwd return must be NaN
    assert np.isnan(fwd[3].iloc[-3:]["AAPL"]).all()


def test_factor_spread_and_acceleration():
    """Verify delta_s spread factor and acceleration mathematics."""
    s_df = _sentiment("2022-01-01", "2023-12-31")
    assert "s_upper" in s_df.columns
    assert "s_median" in s_df.columns
    assert "s_lower" in s_df.columns
    assert "s_k" in s_df.columns

    macro = calculate_derived_macro_factors(s_df)
    # Delta_S = S_upper - S_median
    expected_delta_s = s_df["s_upper"] - s_df["s_median"]
    assert np.allclose(macro["delta_s"], expected_delta_s)

    # Spending momentum 1M = diff(s_upper, 1)
    expected_mom1 = s_df["s_upper"].diff(1).fillna(0.0)
    assert np.allclose(macro["spending_momentum_1m"], expected_mom1)

    # Acceleration = diff(momentum, 1)
    expected_acc = expected_mom1.diff(1).fillna(0.0)
    assert np.allclose(macro["spending_accel"], expected_acc)


def test_cross_sectional_ranking_and_quintiles():
    """Verify cross-sectional ranking assigns exactly 5 balanced quintiles."""
    prices = fetch_historical_prices(start_date="2023-01-01", end_date="2023-12-31")
    s_df = _sentiment("2023-01-01", "2023-12-31")
    macro = calculate_derived_macro_factors(s_df)

    scores = compute_cross_sectional_factor_scores(macro, prices)
    quintiles = assign_quintiles(scores, num_quantiles=5)

    assert scores.shape == quintiles.shape
    # For any date with complete rows, quintile values must span 1 to 5
    ranked = quintiles.dropna(how="all")
    assert len(ranked) > 0, "No date was ranked"
    for date in ranked.index:
        row = ranked.loc[date].dropna()
        unique_q = sorted(row.unique().tolist())
        assert unique_q == [1, 2, 3, 4, 5], f"Quintiles must span 1 to 5 on {date}"


def test_factor_ranking_depends_on_sentiment():
    """The ranking must move with the sentiment series, not be fixed by static labels."""
    prices = fetch_historical_prices(start_date="2021-01-01", end_date="2024-12-31")
    s_df = _sentiment("2021-01-01", "2024-12-31")

    def ranked_quintiles(sentiment):
        scores = compute_cross_sectional_factor_scores(calculate_derived_macro_factors(sentiment), prices)
        return assign_quintiles(scores, num_quantiles=5).dropna(how="all")

    real = ranked_quintiles(s_df)
    assert len(real.drop_duplicates()) > 1, "Ranking is identical in every month"

    # Same values in a different time order is a different signal, so the ranking must change
    shuffled = pd.DataFrame(
        s_df.sample(frac=1.0, random_state=0).to_numpy(), index=s_df.index, columns=s_df.columns
    )
    other = ranked_quintiles(shuffled)
    shared = real.index.intersection(other.index)
    assert not real.loc[shared].equals(other.loc[shared]), "Ranking ignores the sentiment input"


def test_portfolio_metrics_math():
    """Verify statistical metrics (Sharpe, max drawdown, t-statistic)."""
    # Deterministic return series: ~1% monthly gain with low volatility
    steady_rets = pd.Series([0.01 + 0.001 * (i % 3) for i in range(24)])
    m = calculate_portfolio_metrics(steady_rets, rf_annual=0.0, periods_per_year=12, is_long_short=True)
    assert m["annualized_return"] > 0.12
    assert m["max_drawdown"] == 0.0
    assert m["win_rate"] == 1.0

    # Series with a known drawdown
    rets_with_dd = pd.Series([0.10, -0.20, 0.05])
    m_dd = calculate_portfolio_metrics(rets_with_dd)
    assert m_dd["max_drawdown"] > 0.10


def test_information_coefficient():
    """Verify Rank IC computation between factor scores and forward returns."""
    # Synthetic perfect correlation test
    dates = pd.date_range("2023-01-01", periods=5, freq="ME")
    assets = ["A", "B", "C", "D", "E"]
    scores = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 5.0]] * 5,
        index=dates,
        columns=assets,
    )
    # Forward returns perfectly aligned with scores
    fwd_ret = pd.DataFrame(
        [[0.01, 0.02, 0.03, 0.04, 0.05]] * 5,
        index=dates,
        columns=assets,
    )
    ic_df = compute_information_coefficients(scores, fwd_ret)
    assert len(ic_df) == 5
    assert np.isclose(ic_df["IC"], 1.0).all(), "Spearman rank IC for monotonic alignment must be 1.0"


def test_end_to_end_backtest():
    """Verify end-to-end backtest returns full results dictionary."""
    results = run_factor_backtest()
    assert "metadata" in results
    assert "macro_factors_summary" in results
    assert "horizons" in results
    assert "1M" in results["horizons"]
    assert "3M" in results["horizons"]

    h1 = results["horizons"]["1M"]
    assert "long_short_metrics" in h1
    assert "quintile_metrics" in h1
    assert "information_coefficient" in h1
    assert np.isfinite(h1["long_short_metrics"]["sharpe_ratio"])
