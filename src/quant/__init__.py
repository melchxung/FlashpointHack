"""
Quantitative Factor and Cross-Sectional Asset Pricing Package for K-Shaped Macro Dynamics.
"""

from src.quant.universe import (
    ASSET_MAP,
    MODEL_TICKERS,
    UNIVERSE_ASSETS,
    WINNER_TICKERS,
    LOSER_TICKERS,
    get_universe_metadata,
)

__all__ = [
    "UNIVERSE_ASSETS",
    "ASSET_MAP",
    "MODEL_TICKERS",
    "WINNER_TICKERS",
    "LOSER_TICKERS",
    "get_universe_metadata",
]
