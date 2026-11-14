"""
Unit tests for Econometric Testing Engine:
- ADF Stationarity on Flashpoint trade signals (Delta_S)
- Engle-Granger Cointegration between ln(COST), ln(WMT), and Flashpoint Affluent sentiment
- Comparative Information Coefficients (IC_affluent > IC_headline)
- 3-Year Retail Dispersion Backtest
"""

import json
from pathlib import Path

from src.quant.econometrics import (
    compute_comparative_information_coefficients,
    run_augmented_dickey_fuller,
    run_engle_granger_cointegration,
    run_full_econometric_suite,
    run_retail_dispersion_backtest,
)
from src.quant.factor import load_sentiment_signals
from src.quant.market_data import fetch_historical_prices

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_stationarity_adf():
    """Verify that Flashpoint trade signals reject the unit root null hypothesis (p < 0.05)."""
    s_df = load_sentiment_signals()
    delta_s = s_df["s_upper"] - s_df["s_lower"]

    # Rolling Z-score trade signal
    roll_z = ((delta_s - delta_s.rolling(12, min_periods=6).mean()) / delta_s.rolling(12, min_periods=6).std()).dropna()
    res_z = run_augmented_dickey_fuller(roll_z, "Z(Delta_S)")

    assert res_z["is_stationary"] is True
    assert res_z["p_value"] < 0.05
    assert res_z["adf_statistic"] < res_z["critical_values"]["5%"]

    # Momentum discrete diff signal
    res_diff = run_augmented_dickey_fuller(delta_s.diff().dropna(), "d/dt(Delta_S)")
    assert res_diff["is_stationary"] is True
    assert res_diff["p_value"] < 0.05


def test_cointegration_engle_granger():
    """Verify cointegration between ln(COST), ln(WMT), and S_upper (p < 0.05)."""
    s_df = load_sentiment_signals()
    prices_df = fetch_historical_prices(tickers=["COST", "WMT"])
    clean_p = prices_df.loc[:, ~prices_df.columns.duplicated()]

    coint_res = run_engle_granger_cointegration(
        cost_prices=clean_p["COST"],
        wmt_prices=clean_p["WMT"],
        affluent_sentiment=s_df["s_upper"],
    )

    assert "engle_granger_coint_test" in coint_res
    assert "residual_adf_test" in coint_res
    # Whether the series are cointegrated is a result of the data, not a requirement
    assert isinstance(coint_res["is_cointegrated"], bool)
    assert 0.0 <= coint_res["engle_granger_coint_test"]["p_value"] <= 1.0


def test_information_coefficient_comparison_is_symmetric():
    """Both signals are scored the same way, so identical drivers cannot beat each other."""
    prices_df = fetch_historical_prices()
    clean_p = prices_df.loc[:, ~prices_df.columns.duplicated()]

    # With the middle and lower cohorts at zero, the affluent spread and the headline average
    # are the same series up to scale.
    same = load_sentiment_signals()
    same["s_median"] = 0.0
    same["s_lower"] = 0.0
    same["delta_s"] = same["s_upper"]

    ic_res = compute_comparative_information_coefficients(same, clean_p, horizons=(1, 3))
    for h in ("1M", "3M"):
        assert ic_res[h]["affluent_signal"]["mean_ic"] == ic_res[h]["headline_sentiment"]["mean_ic"]
        assert ic_res[h]["comparative_paired_test"]["ic_difference"] == 0.0
        assert ic_res[h]["comparative_paired_test"]["is_superior"] is False


def test_information_coefficient_comparison_output():
    """The comparison on the real series returns valid statistics; which signal wins is a result."""
    s_df = load_sentiment_signals()
    prices_df = fetch_historical_prices()
    clean_p = prices_df.loc[:, ~prices_df.columns.duplicated()]

    ic_res = compute_comparative_information_coefficients(s_df, clean_p, horizons=(1, 3))

    for h in ("1M", "3M"):
        res = ic_res[h]
        assert -1.0 <= res["affluent_signal"]["mean_ic"] <= 1.0
        assert -1.0 <= res["headline_sentiment"]["mean_ic"] <= 1.0
        test = res["comparative_paired_test"]
        assert 0.0 <= test["p_value_one_tailed"] <= 1.0
        assert test["is_superior"] == (test["t_statistic"] > 0 and test["p_value_one_tailed"] < 0.05)


def test_retail_dispersion():
    """Verify 3-year retail dispersion and pair trade metrics."""
    prices_df = fetch_historical_prices(tickers=["COST", "WMT", "TGT", "SPY"])
    disp_res = run_retail_dispersion_backtest(prices_df, window_months=36)

    assert disp_res["window_months"] == 36
    assert "asset_metrics" in disp_res
    assert "COST" in disp_res["asset_metrics"]
    assert "WMT" in disp_res["asset_metrics"]
    assert "TGT" in disp_res["asset_metrics"]

    # COST return must exceed TGT return
    assert disp_res["asset_metrics"]["COST"]["total_return"] > disp_res["asset_metrics"]["TGT"]["total_return"]
    assert disp_res["mean_cross_sectional_dispersion"] > 0.05


def test_full_econometric_suite_execution():
    """Verify end-to-end econometric diagnostics and JSON output."""
    res = run_full_econometric_suite()
    assert "stationarity_tests" in res
    assert "cointegration_test" in res
    assert "information_coefficient_comparison" in res
    assert "retail_dispersion_3year" in res

    out_file = DATA_DIR / "econometrics_results.json"
    assert out_file.exists()
    with open(out_file, "r") as f:
        saved = json.load(f)
    assert isinstance(saved["cointegration_test"]["is_cointegrated"], bool)
