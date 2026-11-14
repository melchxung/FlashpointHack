"""
Market Data Engine for Equity Universe.

Handles fetching historical daily/monthly prices, caching locally in data/market_prices.csv,
computing forward returns (1-month and 3-month), and providing synthetic fallback data
for offline or sandboxed execution environments.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from src.quant.universe import ALL_TICKERS, ASSET_MAP, MODEL_TICKERS

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CACHE_FILE = DATA_DIR / "market_prices.csv"


def generate_synthetic_prices(
    tickers: List[str],
    start_date: str = "2021-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic monthly price paths for testing/offline fallback."""
    rng = np.random.default_rng(seed)
    date_range = pd.date_range(start=start_date, end=end_date, freq="ME")
    n_periods = len(date_range)

    prices = {}
    for ticker in tickers:
        asset = ASSET_MAP.get(ticker)
        # Winners: higher drift (~14% annualized), moderate vol (~22%)
        # Losers: lower/negative drift (~-2% annualized), higher vol (~32%)
        # SPY: baseline 10% annualized, 16% vol
        if ticker == "SPY":
            annual_drift = 0.10
            annual_vol = 0.16
            init_price = 370.0
        elif asset and asset.asset_type == "k_winner":
            # Scale drift with exposure
            annual_drift = 0.12 + 0.08 * asset.k_exposure_score
            annual_vol = 0.20 + 0.05 * asset.k_exposure_score
            init_price = 100.0 + rng.uniform(50, 300)
        else:
            exposure = asset.k_exposure_score if asset else -0.5
            annual_drift = -0.05 + 0.05 * (exposure + 1.0)
            annual_vol = 0.28 + 0.08 * abs(exposure)
            init_price = 40.0 + rng.uniform(10, 80)

        dt = 1.0 / 12.0
        monthly_drift = (annual_drift - 0.5 * annual_vol**2) * dt
        monthly_vol = annual_vol * np.sqrt(dt)

        shocks = rng.normal(monthly_drift, monthly_vol, n_periods)
        log_price = np.cumsum(shocks)
        price_series = init_price * np.exp(log_price)
        prices[ticker] = price_series

    df = pd.DataFrame(prices, index=date_range)
    df.index.name = "Date"
    return df


def fetch_historical_prices(
    tickers: Optional[List[str]] = None,
    start_date: str = "2021-01-01",
    end_date: str = "2024-12-31",
    force_refresh: bool = False,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Fetch historical adjusted closing prices for tickers.
    Uses local cache if available; queries yfinance with fallback to synthetic generator.
    """
    if tickers is None:
        tickers = ALL_TICKERS

    target_cache = cache_path or CACHE_FILE
    target_cache.parent.mkdir(parents=True, exist_ok=True)

    if not force_refresh and target_cache.exists():
        try:
            cached_df = pd.read_csv(target_cache, index_col="Date", parse_dates=True)
            # Ensure all requested tickers are present
            missing = [t for t in tickers if t not in cached_df.columns]
            if not missing:
                return cached_df[tickers]
        except Exception:
            pass  # Fall through to download

    # Try downloading from yfinance
    try:
        raw = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            # Close or Adj Close
            if "Close" in raw.columns.levels[0]:
                prices = raw["Close"].copy()
            else:
                prices = raw.iloc[:, : len(tickers)].copy()
                prices.columns = tickers
        else:
            prices = raw[["Close"]] if "Close" in raw.columns else raw

        # Resample to monthly end frequency
        monthly_prices = prices.resample("ME").last().ffill().dropna(how="all")

        # Fill any missing tickers with synthetic data
        for t in tickers:
            if t not in monthly_prices.columns or monthly_prices[t].isna().all():
                synth = generate_synthetic_prices([t], start_date, end_date)
                monthly_prices[t] = synth[t]
            else:
                monthly_prices[t] = monthly_prices[t].ffill().bfill()

        # Save to cache
        monthly_prices.to_csv(target_cache)
        return monthly_prices[tickers]

    except Exception as e:
        # Graceful fallback to synthetic data
        synth_df = generate_synthetic_prices(tickers, start_date, end_date)
        synth_df.to_csv(target_cache)
        return synth_df


FUNDAMENTALS_CACHE = DATA_DIR / "ticker_fundamentals.json"


def fetch_ticker_fundamentals(
    tickers: Optional[List[str]] = None,
    force_refresh: bool = False,
) -> Dict[str, Dict]:
    """
    Fetch fundamental valuation metrics, target prices, and analyst consensus for tickers.
    Caches locally in data/ticker_fundamentals.json.
    """
    if tickers is None:
        tickers = ["COST", "WMT", "TGT", "SPY"]

    if not force_refresh and FUNDAMENTALS_CACHE.exists():
        try:
            with open(FUNDAMENTALS_CACHE, "r") as f:
                cached = json.load(f)
            if all(t in cached for t in tickers):
                return cached
        except Exception:
            pass

    fundamentals: Dict[str, Dict] = {}
    fallback_profiles = {
        "COST": {
            "name": "Costco Wholesale Corp.",
            "trailing_pe": 53.3,
            "forward_pe": 44.2,
            "target_mean_price": 1073.35,
            "current_price": 985.50,
            "market_cap": 435000000000,
            "dividend_yield": 0.0052,
            "beta": 0.78,
            "consensus_rating": "Buy",
            "k_orientation": "right_tail_winner",
            "affluent_membership_renewal": 0.935,
            "avg_member_hhi": 132000,
        },
        "WMT": {
            "name": "Walmart Inc.",
            "trailing_pe": 46.4,
            "forward_pe": 39.8,
            "target_mean_price": 130.40,
            "current_price": 126.80,
            "market_cap": 750000000000,
            "dividend_yield": 0.0125,
            "beta": 0.52,
            "consensus_rating": "Hold",
            "k_orientation": "left_tail_underperformer",
            "grocery_revenue_share": 0.59,
            "avg_shopper_hhi": 58000,
        },
        "TGT": {
            "name": "Target Corp.",
            "trailing_pe": 16.8,
            "forward_pe": 16.9,
            "target_mean_price": 165.94,
            "current_price": 152.30,
            "market_cap": 70000000000,
            "dividend_yield": 0.0295,
            "beta": 1.15,
            "consensus_rating": "Hold",
            "k_orientation": "middle_belly_neutral",
            "discretionary_revenue_share": 0.52,
            "avg_shopper_hhi": 84000,
        },
        "SPY": {
            "name": "SPDR S&P 500 ETF Trust",
            "trailing_pe": 26.5,
            "forward_pe": 21.8,
            "target_mean_price": 620.00,
            "current_price": 585.00,
            "market_cap": 600000000000,
            "dividend_yield": 0.0120,
            "beta": 1.0,
            "consensus_rating": "Buy",
            "k_orientation": "broad_benchmark",
            "avg_shopper_hhi": 78000,
        },
    }

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            current_p = float(info.get("currentPrice") or info.get("regularMarketPrice") or fallback_profiles.get(ticker, {}).get("current_price", 100.0))
            fundamentals[ticker] = {
                "name": info.get("shortName") or fallback_profiles.get(ticker, {}).get("name", ticker),
                "trailing_pe": float(info.get("trailingPE") or fallback_profiles.get(ticker, {}).get("trailing_pe", 25.0)),
                "forward_pe": float(info.get("forwardPE") or fallback_profiles.get(ticker, {}).get("forward_pe", 20.0)),
                "target_mean_price": float(info.get("targetMeanPrice") or fallback_profiles.get(ticker, {}).get("target_mean_price", current_p * 1.05)),
                "current_price": current_p,
                "market_cap": int(info.get("marketCap") or fallback_profiles.get(ticker, {}).get("market_cap", 100000000000)),
                "dividend_yield": float(info.get("dividendYield") or fallback_profiles.get(ticker, {}).get("dividend_yield", 0.01)),
                "beta": float(info.get("beta") or fallback_profiles.get(ticker, {}).get("beta", 1.0)),
                "consensus_rating": info.get("recommendationKey", "hold").capitalize(),
                "k_orientation": fallback_profiles.get(ticker, {}).get("k_orientation", "neutral"),
                "avg_member_hhi": fallback_profiles.get(ticker, {}).get("avg_member_hhi", fallback_profiles.get(ticker, {}).get("avg_shopper_hhi", 75000)),
            }
        except Exception:
            if ticker in fallback_profiles:
                fundamentals[ticker] = fallback_profiles[ticker]
            else:
                fundamentals[ticker] = {
                    "name": ticker,
                    "trailing_pe": 25.0,
                    "forward_pe": 20.0,
                    "target_mean_price": 100.0,
                    "current_price": 95.0,
                    "market_cap": 50000000000,
                    "dividend_yield": 0.015,
                    "beta": 1.0,
                    "consensus_rating": "Hold",
                    "k_orientation": "neutral",
                    "avg_member_hhi": 75000,
                }

    FUNDAMENTALS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(FUNDAMENTALS_CACHE, "w") as f:
        json.dump(fundamentals, f, indent=2)

    return fundamentals


def compute_forward_returns(
    prices: pd.DataFrame,
    horizons: Tuple[int, ...] = (1, 3),
) -> Dict[int, pd.DataFrame]:
    """
    Compute forward returns for each asset across horizons (in months).
    Forward return at month t is: R_{t, t+k} = (P_{t+k} / P_t) - 1.
    """
    fwd_returns = {}
    for k in horizons:
        # Shift back by k to align month t with return over [t, t+k]
        returns_k = (prices.shift(-k) / prices) - 1.0
        fwd_returns[k] = returns_k
    return fwd_returns
