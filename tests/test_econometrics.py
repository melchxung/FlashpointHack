"""
Unit tests for Econometric Testing Engine:
- ADF Stationarity on Flashpoint trade signals (Delta_S)
- Engle-Granger Cointegration between ln(COST), ln(WMT), and Flashpoint Affluent sentiment
- Comparative Information Coefficients (IC_affluent > IC_headline)
- 3-Year Retail Dispersion Backtest
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
    assert coint_res["is_cointegrated"] is True
    assert coint_res["engle_granger_coint_test"]["p_value"] < 0.05


def test_information_coefficients_superiority():
    """Mathematically prove IC_affluent > IC_headline with paired t-test."""
    s_df = load_sentiment_signals()
    prices_df = fetch_historical_prices()
    clean_p = prices_df.loc[:, ~prices_df.columns.duplicated()]

    ic_res = compute_comparative_information_coefficients(s_df, clean_p, horizons=(1, 3))

    assert "1M" in ic_res
    assert "3M" in ic_res

    # 3M horizon comparison
    h3 = ic_res["3M"]
    aff_ic = h3["affluent_signal"]["mean_ic"]
    head_ic = h3["headline_sentiment"]["mean_ic"]
    diff = h3["comparative_paired_test"]["ic_difference"]
    p_val_one_tailed = h3["comparative_paired_test"]["p_value_one_tailed"]

    assert aff_ic > 0.10, "Affluent signal must exhibit strong positive IC (>0.10)"
    assert aff_ic > head_ic, "Affluent IC must strictly exceed headline IC"
    assert diff > 0.15, "Expected IC divergence spread > 0.15"
    assert p_val_one_tailed < 0.01, "Superiority must be statistically significant (p < 0.01)"
    assert h3["comparative_paired_test"]["is_superior"] is True


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
    assert saved["cointegration_test"]["is_cointegrated"] is True
