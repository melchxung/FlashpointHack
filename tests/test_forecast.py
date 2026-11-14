"""
Unit Tests for Forward-Looking Economic News Cycle Forecaster.
"""

from datetime import date

from src.forecast.cycle_forecaster import (
    CycleForecastResult,
    EconomicNewsCycleForecaster,
    generate_current_cycle_forecast,
    get_monthly_release_dates,
    get_release_calendar,
)


def test_monthly_release_dates_calculation():
    # November 2026
    dates = get_monthly_release_dates(2026, 11)
    assert dates["michigan_prelim"] == date(2026, 11, 13)  # 2nd Friday
    assert dates["conference_board"] == date(2026, 11, 24) # Last Tuesday
    assert dates["michigan_final"] == date(2026, 11, 27)   # Last Friday

    # December 2026
    dates_dec = get_monthly_release_dates(2026, 12)
    assert dates_dec["michigan_prelim"] == date(2026, 12, 11) # 2nd Friday
    assert dates_dec["conference_board"] == date(2026, 12, 29) # Last Tuesday
    assert dates_dec["michigan_final"] == date(2026, 12, 25)   # Last Friday


def test_get_release_calendar():
    ref_date = date(2026, 11, 14)
    events = get_release_calendar(reference_date=ref_date, lookahead_months=2)
    assert len(events) >= 3
    # First future event should be Conference Board on 2026-11-24
    assert events[0].is_imminent is True
    assert events[0].release_date == "2026-11-24"
    assert events[0].days_until_release == 10


def test_forecaster_engine():
    forecaster = EconomicNewsCycleForecaster()
    forecast = forecaster.generate_cycle_forecast(
        reference_date=date(2026, 11, 14),
        consensus_headline_estimate=70.0,
    )

    assert isinstance(forecast, CycleForecastResult)
    assert forecast.consensus_headline_prediction == 70.0
    # The surprise is the latest affluent reading minus consensus; its size is a result of the data
    assert forecast.predicted_surprise_points == round(forecast.flashpoint_affluent_signal - 70.0, 1)

    # Verify ticker signals
    assert "COST" in forecast.ticker_signals
    assert "WMT" in forecast.ticker_signals
    assert "TGT" in forecast.ticker_signals

    assert forecast.ticker_signals["COST"]["recommendation"] == "STRONG BUY / OVERWEIGHT"
    assert forecast.ticker_signals["WMT"]["recommendation"] == "UNDERPERFORM / SHORT"
    assert forecast.ticker_signals["TGT"]["recommendation"] == "NEUTRAL / HOLD"

    # Verify pair trade
    assert forecast.pair_trade_recommendation["strategy"] == "Long COST / Short WMT (Dollar-Neutral)"
    assert forecast.pair_trade_recommendation["annualized_sharpe_ratio"] > 1.0


def test_generate_current_cycle_forecast_file(tmp_path):
    target = tmp_path / "forecast_test.json"
    payload = generate_current_cycle_forecast(output_path=str(target))
    assert target.exists()
    assert "cycle_name" in payload
    assert "next_major_catalyst" in payload
    assert "upcoming_releases" in payload
    assert "ticker_signals" in payload
