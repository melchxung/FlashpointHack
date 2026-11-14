"""
tests/test_frontend.py - Unit and Integration Tests for Streamlit app.py
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def test_dashboard_data_artifacts_exist():
    required_files = [
        "k_alpha_retail_signals.json",
        "respondent_scored_transcripts.json",
        "affluent_survey_replications.json",
        "baseline_correlations.json",
        "baseline_sentiment_surveys.csv",
        "market_prices.csv",
        "ticker_fundamentals.json",
        "forward_cycle_forecast.json",
        "econometrics_results.json",
    ]
    for filename in required_files:
        path = DATA_DIR / filename
        assert path.exists(), f"Missing required file: {filename}"
        assert path.stat().st_size > 0, f"File is empty: {filename}"


def test_ticker_signals_structure():
    with open(DATA_DIR / "k_alpha_retail_signals.json", "r") as f:
        data = json.load(f)

    ticker_signals = data.get("ticker_signals", {})
    assert "COST" in ticker_signals
    assert "WMT" in ticker_signals
    assert "TGT" in ticker_signals

    # COST should be Strong Buy
    assert "Buy" in ticker_signals["COST"]["action"]
    assert ticker_signals["COST"]["k_alpha_upside_pct"] > 0
    assert ticker_signals["COST"]["revealed_conviction_score"] > 0.5

    # WMT should be Underperform / Sell
    assert "Sell" in ticker_signals["WMT"]["action"] or "Underperform" in ticker_signals["WMT"]["action"]
    assert ticker_signals["WMT"]["k_alpha_upside_pct"] < 0
    assert ticker_signals["WMT"]["trade_down_risk"] > 0.7

    # TGT should be Neutral / Hold
    assert "Hold" in ticker_signals["TGT"]["action"] or "Neutral" in ticker_signals["TGT"]["action"]


def test_respondent_scatter_data_integrity():
    with open(DATA_DIR / "k_alpha_retail_signals.json", "r") as f:
        data = json.load(f)

    scatter = data.get("respondent_scatter_data", [])
    assert len(scatter) >= 75

    for item in scatter:
        assert "stated_macro_sentiment" in item
        assert "revealed_spend_conviction" in item
        assert "vibecession_gap" in item
        assert "hhi" in item
        assert -1.1 <= item["stated_macro_sentiment"] <= 1.1
        assert -1.1 <= item["revealed_spend_conviction"] <= 1.1
        assert item["hhi"] > 0


def test_survey_comparison_table_and_baselines():
    with open(DATA_DIR / "k_alpha_retail_signals.json", "r") as f:
        data = json.load(f)

    table = data.get("survey_comparison_table", [])
    assert len(table) >= 8  # Flashpoint + 7 Chicago Fed Letter 521 baselines

    # Find Flashpoint K-Alpha
    k_alpha_entry = next((s for s in table if "Flashpoint" in s["survey"]), None)
    assert k_alpha_entry is not None
    assert k_alpha_entry["pce_correlation"] > 0.65

    # Find Michigan UMCSENT
    michigan_entry = next((s for s in table if "Michigan" in s["survey"] or "UMCSENT" in s["survey"]), None)
    assert michigan_entry is not None
    assert michigan_entry["pce_correlation"] < 0.35

    # K-Alpha must outperform Michigan significantly
    assert k_alpha_entry["pce_correlation"] > michigan_entry["pce_correlation"] * 2.0


def test_app_load_all_data_callable():
    import app
    loaded = app.load_all_data()
    assert "k_alpha" in loaded
    assert "transcripts" in loaded
    assert "replications" in loaded
    assert "baseline_corr" in loaded
    assert "fundamentals" in loaded
    assert "forecast" in loaded
    assert "econometrics" in loaded
    assert "surveys_df" in loaded
    assert "prices_df" in loaded

    # Check forecast hero card fields
    fc = loaded["forecast"]
    assert "next_major_catalyst" in fc
    assert fc["next_major_catalyst"]["days_until_release"] >= 0
    assert fc["predicted_surprise_points"] > 0

    # Check econometrics diagnostics
    econ = loaded["econometrics"]
    assert econ["stationarity_tests"]["primary_signal"]["p_value"] < 0.05
    assert 0.0 <= econ["cointegration_test"]["engle_granger_coint_test"]["p_value"] <= 1.0
    assert isinstance(
        econ["information_coefficient_comparison"]["3M"]["comparative_paired_test"]["is_superior"], bool
    )
    assert "retail_dispersion_3year" in econ
