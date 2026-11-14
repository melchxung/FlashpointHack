"""
Comprehensive Verification Test Suite for K-Shaped Consumer Sentiment & Factor Pricing Model.

Validates:
1. Missing API Keys and Client Fallbacks
2. Distribution Weights Normalization & Boundedness
3. Cross-Sectional Ranking Math, Ties, and Degenerate Cases
4. Forward Return Lookahead Bias Prevention & Temporal Alignment
5. End-to-End Pipeline Execution, Schema Invariants, & Reproducibility
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import requests

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
    generate_synthetic_prices,
)
from src.quant.universe import ALL_TICKERS, MODEL_TICKERS, WINNER_TICKERS, LOSER_TICKERS
from src.sampler.client import FlashpointClient
from src.sampler.engine import SentimentSamplingEngine
from src.sampler.personas import PersonaArchetype


# ==============================================================================
# 1. MISSING API KEYS & CLIENT FALLBACKS
# ==============================================================================

class TestMissingApiKeysAndFallbacks:
    """Validate client robustness when credentials are absent, invalid, or remote fails."""

    def test_missing_credentials_defaults_to_simulator(self):
        """When api_token is None, client should report no credentials and fallback seamlessly."""
        client = FlashpointClient(api_token=None, fallback_to_simulator=True)
        assert not client.has_credentials
        res = client.create_and_run_panel(archetype=PersonaArchetype.UPPER_QUARTILE, num_respondents=10)
        assert res["status"] == "success"
        assert res["source"] == "synthetic"
        assert len(res["responses"]) == 10

    def test_blank_or_whitespace_token(self):
        """Empty string or whitespace tokens must be treated as absent."""
        for blank_tok in ["", "   ", "\t\n"]:
            client = FlashpointClient(api_token=blank_tok, fallback_to_simulator=True)
            assert not client.has_credentials
            res = client.create_and_run_panel(archetype=PersonaArchetype.LOWER_QUARTILE, num_respondents=5)
            assert res["status"] == "success"

    def test_missing_credentials_raises_if_fallback_disabled(self):
        """When fallback_to_simulator=False and no keys present, raise ValueError."""
        client = FlashpointClient(api_token=None, fallback_to_simulator=False)
        assert not client.has_credentials
        with pytest.raises(ValueError, match="No Flashpoint API token configured"):
            client.create_and_run_panel(archetype=PersonaArchetype.UPPER_QUARTILE, num_respondents=5)

    @patch("requests.post")
    def test_mcp_rate_limit_and_retry_fallback(self, mock_post):
        """When remote returns 429 then fails, client must retry and fallback to simulator."""
        # 1st call: 429 Rate Limit; 2nd call: 500 Server Error; 3rd call: Network Exception
        r429 = MagicMock()
        r429.status_code = 429
        r429.headers = {"Retry-After": "0"}

        r500 = MagicMock()
        r500.status_code = 500
        r500.text = "Internal Server Error"

        mock_post.side_effect = [r429, r500, requests.exceptions.ConnectionError("Connection failed")]

        client = FlashpointClient(
            api_token="dummy_token_xyz",
            max_retries=3,
            fallback_to_simulator=True,
            timeout=1.0,
        )
        assert client.has_credentials
        # Should exhaust retries and fall back to simulator
        res = client.create_and_run_panel(archetype=PersonaArchetype.MIDDLE_INCOME, num_respondents=10)
        assert res["status"] == "success"
        assert res["source"] == "synthetic"
        assert mock_post.call_count == 3


# ==============================================================================
# 2. DISTRIBUTION WEIGHTS NORMALIZATION & INVARIANTS
# ==============================================================================

class TestDistributionWeightsNormalization:
    """Validate macro consumption weighting normalization and mathematical bounds."""

    def test_unnormalized_weights_are_normalized(self):
        """Integer percentage weights (48, 36, 16) must normalize to exactly 1.0."""
        engine = SentimentSamplingEngine(
            weight_upper=48.0,
            weight_middle=36.0,
            weight_lower=16.0,
        )
        total_w = engine.weight_upper + engine.weight_middle + engine.weight_lower
        assert np.isclose(total_w, 1.0)
        assert np.isclose(engine.weight_upper, 0.48)
        assert np.isclose(engine.weight_middle, 0.36)
        assert np.isclose(engine.weight_lower, 0.16)

    def test_arbitrary_scale_weights_normalization(self):
        """Arbitrary weight scalings (e.g. 10x) must normalize correctly."""
        engine = SentimentSamplingEngine(
            weight_upper=4.8,
            weight_middle=3.6,
            weight_lower=1.6,
        )
        assert np.isclose(engine.weight_upper + engine.weight_middle + engine.weight_lower, 1.0)

    def test_spending_weighted_sentiment_boundedness(self):
        """S_K must always be strictly bounded between min(cohorts) and max(cohorts)."""
        engine = SentimentSamplingEngine()
        record, _ = engine.sample_single_period(date_str="2023-06", num_respondents_per_cohort=20)
        min_cohort = min(record.S_upper, record.S_middle, record.S_lower)
        max_cohort = max(record.S_upper, record.S_middle, record.S_lower)
        assert min_cohort <= record.S_K <= max_cohort
        assert min_cohort <= record.S_equal <= max_cohort


# ==============================================================================
# 3. CROSS-SECTIONAL RANKING MATH & DEGENERATE CASES
# ==============================================================================

class TestCrossSectionalRankingMath:
    """Validate cross-sectional ranking with edge cases (ties, identical scores, few assets)."""

    def test_identical_tied_scores_handling(self):
        """When all assets have identical factor scores, quintiles must still form valid 1..5 partitions."""
        dates = pd.date_range("2023-01-01", periods=3, freq="ME")
        tickers = [f"STK_{i}" for i in range(20)]
        # All scores identical (0.0)
        scores = pd.DataFrame(0.0, index=dates, columns=tickers)
        quintiles = assign_quintiles(scores, num_quantiles=5)

        for date in dates:
            row = quintiles.loc[date].dropna()
            # method='first' in rank ensures all 1..5 are assigned
            assert sorted(row.unique().tolist()) == [1, 2, 3, 4, 5]
            # Each quintile should have exactly 4 assets (20 / 5)
            counts = row.value_counts()
            for q in range(1, 6):
                assert counts[q] == 4

    def test_fewer_assets_than_quintiles(self):
        """When number of assets is less than num_quantiles, row is skipped gracefully."""
        dates = pd.date_range("2023-01-01", periods=2, freq="ME")
        tickers = ["A", "B", "C"]  # only 3 assets for 5 quintiles
        scores = pd.DataFrame(1.0, index=dates, columns=tickers)
        quintiles = assign_quintiles(scores, num_quantiles=5)
        # Should be empty or all NaN because len(row) < num_quantiles
        assert quintiles.isna().all().all()

    def test_quintile_monotonicity_under_clear_signal(self):
        """Under a strong signal, Q5 must have significantly higher factor scores than Q1."""
        dates = pd.date_range("2023-01-01", periods=5, freq="ME")
        tickers = [f"STK_{i}" for i in range(20)]
        # Monotonic scores from 1 to 20
        scores = pd.DataFrame(
            [list(range(1, 21))] * 5,
            index=dates,
            columns=tickers,
        )
        quintiles = assign_quintiles(scores, num_quantiles=5)

        # STK_0 (lowest score) must be Q1, STK_19 (highest score) must be Q5
        assert (quintiles["STK_0"] == 1).all()
        assert (quintiles["STK_19"] == 5).all()


# ==============================================================================
# 4. FORWARD RETURN LOOKAHEAD BIAS PREVENTION
# ==============================================================================

class TestLookaheadBiasPrevention:
    """Ensure strict temporal alignment: factor scores at t predict returns from t to t+k."""

    def test_forward_return_definition_and_boundary_nans(self):
        """
        Verify forward return math:
        R_{t, t+k} = (P_{t+k} / P_t) - 1.
        The last k rows MUST be NaN because future prices are unobservable.
        """
        dates = pd.date_range("2023-01-31", periods=12, freq="ME")
        prices = pd.DataFrame({"AAPL": np.linspace(100, 220, 12)}, index=dates)

        fwd = compute_forward_returns(prices, horizons=(1, 3))

        # Check 1M forward return
        for t in range(11):
            expected_ret = (prices.iloc[t + 1]["AAPL"] / prices.iloc[t]["AAPL"]) - 1.0
            actual_ret = fwd[1].iloc[t]["AAPL"]
            assert np.isclose(expected_ret, actual_ret)

        # Boundary NaN check: last 1 row for 1M, last 3 rows for 3M
        assert np.isnan(fwd[1].iloc[-1]["AAPL"])
        assert np.isnan(fwd[3].iloc[-3:]["AAPL"]).all()
        assert not np.isnan(fwd[3].iloc[-4]["AAPL"])

    def test_backtest_does_not_trade_on_boundary_nans(self):
        """Quintile portfolio returns must drop dates where forward returns are NaN."""
        dates = pd.date_range("2023-01-31", periods=10, freq="ME")
        tickers = [f"T_{i}" for i in range(10)]
        prices = pd.DataFrame(100.0, index=dates, columns=tickers)

        fwd = compute_forward_returns(prices, horizons=(1, 3))
        # Build dummy quintiles
        scores = pd.DataFrame(
            [list(range(10))] * 10,
            index=dates,
            columns=tickers,
        )
        quintiles = assign_quintiles(scores, num_quantiles=5)

        # Build returns for 1M
        q_df_1m, _ = build_quintile_returns(quintiles, fwd[1], num_quantiles=5)
        # Should have exactly 9 dates (10 - 1)
        assert len(q_df_1m) == 9
        assert dates[-1] not in q_df_1m.index

        # Build returns for 3M
        q_df_3m, _ = build_quintile_returns(quintiles, fwd[3], num_quantiles=5)
        # Should have exactly 7 dates (10 - 3)
        assert len(q_df_3m) == 7
        assert dates[-1] not in q_df_3m.index
        assert dates[-2] not in q_df_3m.index
        assert dates[-3] not in q_df_3m.index

    def test_derived_factors_are_strictly_backward_looking(self):
        """calculate_derived_macro_factors must not use any future sentiment."""
        dates = pd.date_range("2023-01-31", periods=12, freq="ME")
        s_df = pd.DataFrame(
            {
                "s_upper": np.linspace(80, 120, 12),
                "s_median": np.linspace(60, 70, 12),
                "s_lower": np.linspace(40, 45, 12),
                "s_k": np.linspace(70, 95, 12),
            },
            index=dates,
        )
        macro_full = calculate_derived_macro_factors(s_df)

        # Truncate s_df to first 6 periods and compute factors
        s_df_sub = s_df.iloc[:6].copy()
        macro_sub = calculate_derived_macro_factors(s_df_sub)

        # Factors at period 5 (index 5) must be identical whether future rows (6..11) exist or not
        for col in ["delta_s", "spending_momentum_1m", "spending_momentum_3m", "spending_accel"]:
            val_full = macro_full.iloc[5][col]
            val_sub = macro_sub.iloc[5][col]
            assert np.isclose(val_full, val_sub), f"Factor {col} at index 5 leaked future data!"


# ==============================================================================
# 5. END-TO-END PIPELINE & REPRODUCIBILITY
# ==============================================================================

class TestEndToEndPipelineAndReproducibility:
    """Validate full scripts execution, artifact schema, and numerical reproducibility."""

    def test_sampling_engine_seed_reproducibility(self):
        """Running sampling engine twice with same seed produces identical sentiment series."""
        engine1 = SentimentSamplingEngine()
        engine1.client.simulator.seed = 999
        res1 = engine1.sample_single_period("2024-01", num_respondents_per_cohort=20, seed_offset=1)

        engine2 = SentimentSamplingEngine()
        engine2.client.simulator.seed = 999
        res2 = engine2.sample_single_period("2024-01", num_respondents_per_cohort=20, seed_offset=1)

        assert res1[0].S_upper == res2[0].S_upper
        assert res1[0].S_lower == res2[0].S_lower
        assert res1[0].S_K == res2[0].S_K
        assert res1[0].Delta_S == res2[0].Delta_S

    def test_backtest_reproducibility(self):
        """Running factor backtest twice on identical data produces identical metrics."""
        res1 = run_factor_backtest()
        res2 = run_factor_backtest()

        m1_1 = res1["horizons"]["1M"]["long_short_metrics"]
        m1_2 = res2["horizons"]["1M"]["long_short_metrics"]
        assert m1_1["annualized_return"] == m1_2["annualized_return"]
        assert m1_1["sharpe_ratio"] == m1_2["sharpe_ratio"]
        assert m1_1["max_drawdown"] == m1_2["max_drawdown"]

    def test_sentiment_signals_artifact_schema(self):
        """Verify data/sentiment_signals.json exists and conforms to required schema."""
        signals_path = Path("data/sentiment_signals.json")
        assert signals_path.exists(), "data/sentiment_signals.json missing!"

        with open(signals_path, "r") as f:
            data = json.load(f)

        for req_key in ["dates", "S_upper", "S_lower", "S_middle", "S_equal", "S_K", "Delta_S", "metadata"]:
            assert req_key in data, f"Required key {req_key} missing from sentiment_signals.json"

        n = len(data["dates"])
        assert n >= 36, "Expected at least 3 years (36 months) of data"
        assert len(data["S_upper"]) == n
        assert len(data["S_lower"]) == n
        assert len(data["S_K"]) == n
        assert len(data["Delta_S"]) == n

    def test_backtest_results_artifact_schema_and_metrics(self):
        """Verify data/backtest_results.json exists and satisfies performance hurdles."""
        bt_path = Path("data/backtest_results.json")
        assert bt_path.exists(), "data/backtest_results.json missing!"

        with open(bt_path, "r") as f:
            res = json.load(f)

        assert "metadata" in res
        assert "horizons" in res
        assert "1M" in res["horizons"]
        assert "3M" in res["horizons"]

        h1 = res["horizons"]["1M"]
        ls_m = h1["long_short_metrics"]
        # Invariants: Long/Short positive return and positive Sharpe
        assert ls_m["annualized_return"] > 0.10, "Expected Long/Short return > 10%"
        assert ls_m["sharpe_ratio"] > 0.5, "Expected Long/Short Sharpe > 0.5"
        assert h1["information_coefficient"]["mean_ic"] > 0.0, "Rank IC must be positive"


# ==============================================================================
# 6. ECONOMETRICS & FORWARD FORECASTING INVARIANTS
# ==============================================================================

class TestEconometricsAndForecastingInvariants:
    """Validate statistical properties, stationarity proofs, and forecast schema."""

    def test_econometrics_results_artifact_schema_and_pvalues(self):
        """Verify data/econometrics_results.json has statistically significant proofs."""
        econ_path = Path("data/econometrics_results.json")
        assert econ_path.exists(), "data/econometrics_results.json missing!"

        with open(econ_path, "r") as f:
            econ = json.load(f)

        assert "stationarity_tests" in econ
        assert "cointegration_test" in econ
        assert "information_coefficient_comparison" in econ
        assert "retail_dispersion_3year" in econ

        # ADF stationarity p < 0.05
        primary_adf = econ["stationarity_tests"]["primary_signal"]
        assert primary_adf["is_stationary"] is True
        assert primary_adf["p_value"] < 0.05

        # Cointegration p < 0.05
        coint = econ["cointegration_test"]
        assert coint["is_cointegrated"] is True
        assert coint["engle_granger_coint_test"]["p_value"] < 0.05

        # IC comparison 3M: affluent superior with p < 0.01
        ic_comp = econ["information_coefficient_comparison"]["3M"]
        assert ic_comp["comparative_paired_test"]["is_superior"] is True
        assert ic_comp["comparative_paired_test"]["p_value_one_tailed"] < 0.01
        assert ic_comp["comparative_paired_test"]["t_statistic"] > 2.0

    def test_forward_cycle_forecast_artifact(self):
        """Verify data/forward_cycle_forecast.json has upcoming catalysts and actionable signals."""
        fc_path = Path("data/forward_cycle_forecast.json")
        assert fc_path.exists(), "data/forward_cycle_forecast.json missing!"

        with open(fc_path, "r") as f:
            fc = json.load(f)

        assert "cycle_name" in fc
        assert "next_major_catalyst" in fc
        assert "upcoming_releases" in fc
        assert "ticker_signals" in fc
        assert "pair_trade_recommendation" in fc

        # Check ticker recommendations
        assert fc["ticker_signals"]["COST"]["recommendation"] == "STRONG BUY / OVERWEIGHT"
        assert fc["ticker_signals"]["WMT"]["recommendation"] == "UNDERPERFORM / SHORT"
        assert fc["ticker_signals"]["TGT"]["recommendation"] == "NEUTRAL / HOLD"

        # Check surprise points are positive
        assert fc["predicted_surprise_points"] > 20.0

    def test_baseline_sentiment_surveys_3year_window(self):
        """Verify data/baseline_sentiment_surveys.csv has the required 7 surveys and zero nulls."""
        csv_path = Path("data/baseline_sentiment_surveys.csv")
        assert csv_path.exists(), "data/baseline_sentiment_surveys.csv missing!"

        df = pd.read_csv(csv_path)
        required_surveys = [
            "michigan",
            "ccsi",
            "conference_board",
            "morning_consult",
            "rasmussen",
            "sf_fed_news",
            "goldman_sachs",
        ]
        for s in required_surveys:
            assert s in df.columns, f"Survey {s} missing from baseline_sentiment_surveys.csv"

        # Check 36-month window (Nov 2023 - Oct 2026) has zero NaNs in survey columns
        recent_36 = df.tail(36)
        for s in required_surveys:
            assert not recent_36[s].isna().any(), f"Survey {s} has NaNs in recent 36-month window"

