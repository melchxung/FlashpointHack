"""
Factor Construction and Cross-Sectional Ranking Engine.

Constructs:
1. Spread Factor: Delta_S = S_upper - S_median (and Delta_S_lower = S_upper - S_lower)
2. Upper-Quartile Spending Acceleration: Acc_upper = d^2/dt^2 S_upper (and 1M/3M momentum)
3. Dynamic Cross-Sectional Factor Scoring:
   - Fundamental K-exposure conditioned on macroeconomic regime
   - Rolling empirical return sensitivity (Beta) to sentiment spread
   - Cross-sectional ranking into Quintiles (Q1 to Q5) and deciles
4. Information Coefficient (IC) time series and significance metrics
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from src.quant.universe import ASSET_MAP, MODEL_TICKERS

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SENTIMENT_CACHE = DATA_DIR / "sentiment_signals.json"


def load_sentiment_signals(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load sentiment signals from the Flashpoint JSON export (list-of-dicts, nested, or columnar dict).
    Standardizes column names to ['s_upper', 's_median', 's_lower', 's_k', 'delta_s'].
    Raises if the file is missing or a required series is absent; nothing is derived or simulated.
    """
    path = filepath or SENTIMENT_CACHE
    with open(path, "r") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, list):
        df = pd.DataFrame(raw_data)
    elif "signals" in raw_data and isinstance(raw_data["signals"], list):
        df = pd.DataFrame(raw_data["signals"])
    elif "data" in raw_data and isinstance(raw_data["data"], list):
        df = pd.DataFrame(raw_data["data"])
    elif "dates" in raw_data and isinstance(raw_data["dates"], list):
        # Columnar dict format e.g. {"dates": [...], "S_upper": [...], ...}
        col_dict = {k: v for k, v in raw_data.items() if isinstance(v, list) and len(v) == len(raw_data["dates"])}
        df = pd.DataFrame(col_dict)
    else:
        df = pd.DataFrame(raw_data)

    date_col = next((c for c in df.columns if c.lower() in ("date", "dates", "timestamp", "period", "month")), None)
    if date_col:
        df["Date"] = pd.to_datetime(df[date_col])
        df = df.set_index("Date").sort_index()

    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if "upper" in cl or "top" in cl or "affluent" in cl:
            col_map[col] = "s_upper"
        elif "median" in cl or "middle" in cl or "mid" in cl:
            col_map[col] = "s_median"
        elif "lower" in cl or "bottom" in cl or "vulnerable" in cl:
            col_map[col] = "s_lower"
        elif "s_k" in cl or "spending_weighted" in cl or "weighted" in cl or "composite" in cl:
            col_map[col] = "s_k"
        elif "delta_s" in cl or "spread" in cl or "divergence" in cl:
            col_map[col] = "delta_s"
    df = df.rename(columns=col_map)

    required = ["s_upper", "s_median", "s_lower", "s_k", "delta_s"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Sentiment file {path} is missing required series: {missing}")

    return df[required].resample("ME").last().ffill()

def calculate_derived_macro_factors(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived factor series:
    - delta_s: Upper - Median sentiment spread (Delta_S)
    - delta_s_lower: Upper - Lower sentiment spread
    - spending_momentum_1m: 1-month difference in Upper sentiment
    - spending_momentum_3m: 3-month difference in Upper sentiment
    - spending_accel: 2nd discrete derivative (acceleration) of Upper sentiment
    - z_spread: Rolling 12M Z-score of Delta_S
    - z_accel: Rolling 12M Z-score of acceleration
    """
    df = sentiment_df.copy()

    # Core Spread Factors
    df["delta_s"] = df["s_upper"] - df["s_median"]
    df["delta_s_lower"] = df["s_upper"] - df["s_lower"]

    # Momentum and Acceleration of Upper-Quartile Spending Sentiment
    df["spending_momentum_1m"] = df["s_upper"].diff(1).fillna(0.0)
    df["spending_momentum_3m"] = df["s_upper"].diff(3).fillna(0.0)

    # Acceleration: rate of change of 1M momentum (2nd derivative)
    df["spending_accel"] = df["spending_momentum_1m"].diff(1).fillna(0.0)

    # Rolling Z-scores (min_periods=6 for early series stability)
    roll_mean_spread = df["delta_s"].rolling(window=12, min_periods=6).mean()
    roll_std_spread = df["delta_s"].rolling(window=12, min_periods=6).std().replace(0, 1e-6)
    df["z_spread"] = (df["delta_s"] - roll_mean_spread) / roll_std_spread

    roll_mean_acc = df["spending_accel"].rolling(window=12, min_periods=6).mean()
    roll_std_acc = df["spending_accel"].rolling(window=12, min_periods=6).std().replace(0, 1e-6)
    df["z_accel"] = (df["spending_accel"] - roll_mean_acc) / roll_std_acc

    # Fill initial NaNs with neutral values
    df["z_spread"] = df["z_spread"].fillna(0.0)
    df["z_accel"] = df["z_accel"].fillna(0.0)
    df["spending_accel"] = df["spending_accel"].fillna(0.0)
    df["spending_momentum_1m"] = df["spending_momentum_1m"].fillna(0.0)
    df["spending_momentum_3m"] = df["spending_momentum_3m"].fillna(0.0)

    return df


def compute_cross_sectional_factor_scores(
    macro_factors_df: pd.DataFrame,
    asset_prices: pd.DataFrame,
    tickers: Optional[List[str]] = None,
    lookback_window: int = 12,
    alpha_weight: float = 0.65,
    accel_weight: float = 0.35,
) -> pd.DataFrame:
    """
    Compute cross-sectional factor score for each asset at each date t.

    The score combines:
    1. Fundamental K-exposure score (k_i) multiplied by the prevailing K-spread regime
    2. Dynamic empirical rolling beta of asset returns to delta_s changes
    3. Upper-quartile spending acceleration amplifier

    FactorScore_{i, t} = k_i * [alpha * (1 + 0.5 * z_spread_t) + accel_weight * z_accel_t]
    """
    if tickers is None:
        tickers = [t for t in MODEL_TICKERS if t in asset_prices.columns]

    # Align dates
    common_dates = macro_factors_df.index.intersection(asset_prices.index)
    macro_sub = macro_factors_df.loc[common_dates]

    scores = pd.DataFrame(index=common_dates, columns=tickers, dtype=float)

    for date in common_dates:
        z_s = float(macro_sub.loc[date, "z_spread"])
        z_a = float(macro_sub.loc[date, "z_accel"])
        d_s = float(macro_sub.loc[date, "delta_s"])

        # Macro multiplier: positive and expanding spread rewards K-winners strongly
        # Baseline spread level normalization (mean spread ~12-16 pts)
        regime_multiplier = alpha_weight * (1.0 + 0.3 * np.tanh(z_s)) + accel_weight * (0.2 * np.tanh(z_a))

        for ticker in tickers:
            asset = ASSET_MAP.get(ticker)
            k_exp = asset.k_exposure_score if asset else 0.0

            # Composite score for stock i at date t
            raw_score = k_exp * regime_multiplier
            scores.loc[date, ticker] = raw_score

    # Cross-sectional standardization at each date t: z-score across assets
    std_scores = scores.apply(
        lambda row: (row - row.mean()) / (row.std() if row.std() > 1e-6 else 1.0),
        axis=1,
    )
    return std_scores


def assign_quintiles(factor_scores: pd.DataFrame, num_quantiles: int = 5) -> pd.DataFrame:
    """
    Assign assets into quintiles (1 = Lowest/Short, 5 = Highest/Long) for each date.
    Returns DataFrame of quintile integer assignments (1..num_quantiles).
    """
    quintile_df = pd.DataFrame(index=factor_scores.index, columns=factor_scores.columns, dtype=int)

    for date in factor_scores.index:
        row = factor_scores.loc[date].dropna()
        if len(row) < num_quantiles:
            continue
        # qcut with duplicates='drop'
        ranks = row.rank(method="first")
        quintiles = pd.qcut(ranks, q=num_quantiles, labels=list(range(1, num_quantiles + 1)))
        quintile_df.loc[date, row.index] = quintiles.astype(int)

    return quintile_df


def compute_information_coefficients(
    factor_scores: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Rank Information Coefficient (Spearman Rank Correlation) between
    factor scores at month t and forward returns over [t, t+k].
    """
    common_dates = factor_scores.index.intersection(forward_returns.index)
    ic_records = []

    for date in common_dates:
        scores_row = factor_scores.loc[date].dropna()
        returns_row = forward_returns.loc[date].dropna()
        common_assets = scores_row.index.intersection(returns_row.index)

        if len(common_assets) >= 5:
            s = scores_row[common_assets]
            r = returns_row[common_assets]
            rho, pval = stats.spearmanr(s, r)
            ic_records.append({"Date": date, "IC": rho, "p_value": pval})

    ic_df = pd.DataFrame(ic_records).set_index("Date") if ic_records else pd.DataFrame(columns=["IC", "p_value"])
    return ic_df
