"""
Market Data Engine for Equity Universe.

Handles fetching historical daily/monthly prices, caching locally in data/market_prices.csv,
computing forward returns (1-month and 3-month). All data is live or cached real data;
missing data raises instead of being simulated.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from src.quant.universe import ALL_TICKERS

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CACHE_FILE = DATA_DIR / "market_prices.csv"


def fetch_historical_prices(
    tickers: Optional[List[str]] = None,
    start_date: str = "2021-01-01",
    end_date: str = "2026-10-31",
    force_refresh: bool = False,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Fetch historical adjusted closing prices for tickers.
    Uses local cache if available; otherwise downloads from yfinance and raises if any ticker is missing.
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

    # Live download only: a failed or incomplete download is an error, never simulated
    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError(f"yfinance returned no price data for {tickers}")
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    missing = [t for t in tickers if t not in prices.columns or prices[t].isna().all()]
    if missing:
        raise RuntimeError(f"yfinance returned no prices for: {missing}")

    monthly_prices = prices.resample("ME").last().ffill().bfill()[tickers]
    monthly_prices.to_csv(target_cache)
    return monthly_prices


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
    for ticker in tickers:
        info = yf.Ticker(ticker).info
        current_p = info.get("currentPrice") or info.get("regularMarketPrice")
        target_p = info.get("targetMeanPrice")
        if not current_p or not target_p:
            raise RuntimeError(f"yfinance returned no price or analyst target for {ticker}")
        fundamentals[ticker] = {
            "name": info.get("shortName") or ticker,
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "target_mean_price": float(target_p),
            "current_price": float(current_p),
            "market_cap": info.get("marketCap"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "consensus_rating": (info.get("recommendationKey") or "hold").capitalize(),
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
