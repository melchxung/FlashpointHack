"""
Unit tests for the Retail K-Alpha Divergence Engine and Market Overlay.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.quant.factor import load_sentiment_signals
from src.quant.market_data import fetch_historical_prices, fetch_ticker_fundamentals
from src.quant.retail_alpha import (
    compute_retail_k_alpha,
    load_baseline_surveys,
    load_respondent_transcripts,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_load_respondent_transcripts():
    """Verify loading and standardization of respondent transcripts."""
    transcripts = load_respondent_transcripts()
    assert len(transcripts) >= 75
    first = transcripts[0]
    assert "respondent_id" in first
    assert "retail_cohort" in first
    assert "primary_ticker" in first
    assert "stated_macro_sentiment" in first
    assert "revealed_spend_conviction" in first
    assert "vibecession_gap" in first
    assert -1.0 <= first["stated_macro_sentiment"] <= 1.0
    assert -1.0 <= first["revealed_spend_conviction"] <= 1.0


def test_load_baseline_surveys():
    """Verify Chicago Fed Letter 521 baseline survey loading and correlations."""
    df, stats_summary = load_baseline_surveys()
    assert not df.empty
    expected_surveys = [
        "real_pce_growth",
        "michigan",
        "ccsi",
        "conference_board",
        "morning_consult",
        "rasmussen",
        "sf_fed_news",
        "goldman_sachs",
    ]
    for s in expected_surveys:
        assert s in df.columns
    assert "survey_correlations" in stats_summary


def test_ticker_fundamentals():
    """Verify fundamentals fetching for COST, WMT, TGT, SPY."""
    funds = fetch_ticker_fundamentals(["COST", "WMT", "TGT", "SPY"])
    for ticker in ["COST", "WMT", "TGT", "SPY"]:
        assert ticker in funds
        f = funds[ticker]
        assert f["forward_pe"] > 0
        assert f["target_mean_price"] > 0
        assert f["current_price"] > 0


def test_compute_retail_k_alpha():
    """Verify full retail alpha computation pipeline."""
    transcripts = load_respondent_transcripts()
    sentiment_df = load_sentiment_signals()
    prices_df = fetch_historical_prices(tickers=["COST", "WMT", "TGT", "SPY"])
    funds = fetch_ticker_fundamentals(["COST", "WMT", "TGT", "SPY"])
    baseline_df, _ = load_baseline_surveys()

    results = compute_retail_k_alpha(
        transcripts=transcripts,
        sentiment_df=sentiment_df,
        prices_df=prices_df,
        fundamentals=funds,
        baseline_df=baseline_df,
    )

    assert "cohort_summary" in results
    assert "ticker_signals" in results
    assert "COST" in results["ticker_signals"]
    assert "WMT" in results["ticker_signals"]
    assert "TGT" in results["ticker_signals"]

    # COST must be Buy/Undervalued
    assert results["ticker_signals"]["COST"]["rating_code"] == "BUY"
    assert results["ticker_signals"]["COST"]["k_alpha_upside_pct"] > 0

    # WMT must be Sell/Underperform
    assert results["ticker_signals"]["WMT"]["rating_code"] == "SELL"
    assert results["ticker_signals"]["WMT"]["k_alpha_upside_pct"] < 0

    # TGT must be Hold/Neutral
    assert results["ticker_signals"]["TGT"]["rating_code"] == "HOLD"

    # Survey comparison table must contain Flashpoint High-Earner and 7 baselines
    surveys = [row["survey"] for row in results["survey_comparison_table"]]
    assert len(surveys) == 8
    assert "Flashpoint High-Earner Index (K-Alpha)" in surveys[0]
