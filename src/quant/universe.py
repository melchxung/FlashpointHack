"""
Asset Universe Definition for K-Shaped Consumer Economy.

The K-shaped economic recovery separates households into two diverging paths:
1. Upper-Quartile (>80th percentile / >$150k income, asset-backed via equities & real estate):
   Insulated from inflation, driving discretionary luxury, premium travel, wealth management,
   and premium tech.
2. Median-and-Lower-Quartile (<50th percentile income, wage-dependent, high debt-to-income):
   Squeezed by cumulative inflation, depleted excess pandemic savings, rising credit delinquencies,
   and forced into trade-down discount retailers and subprime credit distress.
"""

from dataclasses import dataclass
from typing import Dict, List, Literal

AssetType = Literal["k_winner", "k_loser", "benchmark"]


@dataclass(frozen=True)
class EquityAsset:
    ticker: str
    name: str
    asset_type: AssetType
    sector: str
    sub_industry: str
    target_tier: str  # e.g., "Top 20%", "Bottom 50%", "Broad Market"
    k_exposure_score: float  # +1.0 for high luxury/wealth, -1.0 for discount/subprime distress
    rationale: str


UNIVERSE_ASSETS: List[EquityAsset] = [
    # --- K-SHAPED WINNERS (Top Quartile Beneficiaries) ---
    EquityAsset(
        ticker="LVMUY",
        name="LVMH Moët Hennessy Louis Vuitton",
        asset_type="k_winner",
        sector="Consumer Discretionary",
        sub_industry="Luxury Goods & Haute Couture",
        target_tier="Top 1-5% Wealth",
        k_exposure_score=1.0,
        rationale="Global luxury conglomerate with extreme pricing power and high-net-worth clientele.",
    ),
    EquityAsset(
        ticker="HESAY",
        name="Hermès International",
        asset_type="k_winner",
        sector="Consumer Discretionary",
        sub_industry="Ultra-Luxury Leather & Apparel",
        target_tier="Top 1% Wealth",
        k_exposure_score=1.0,
        rationale="Ultra-luxury Veblen goods with multi-year waiting lists; zero price elasticity.",
    ),
    EquityAsset(
        ticker="RACE",
        name="Ferrari N.V.",
        asset_type="k_winner",
        sector="Consumer Discretionary",
        sub_industry="Luxury Supercars",
        target_tier="Top 0.1% Ultra-High Net Worth",
        k_exposure_score=1.0,
        rationale="Elite order backlog spanning years; fully detached from broad consumer distress.",
    ),
    EquityAsset(
        ticker="AAPL",
        name="Apple Inc.",
        asset_type="k_winner",
        sector="Information Technology",
        sub_industry="Premium Consumer Hardware & Services",
        target_tier="Top 25% Affluent Consumer",
        k_exposure_score=0.75,
        rationale="Premium ecosystem pricing power and affluent, sticky customer base with high services spend.",
    ),
    EquityAsset(
        ticker="AMZN",
        name="Amazon.com Inc.",
        asset_type="k_winner",
        sector="Consumer Discretionary",
        sub_industry="E-Commerce & Cloud",
        target_tier="Top 25% Prime & Cloud",
        k_exposure_score=0.6,
        rationale="Prime household spend skew heavily affluent; resilient e-commerce wallet share.",
    ),
    EquityAsset(
        ticker="BKNG",
        name="Booking Holdings Inc.",
        asset_type="k_winner",
        sector="Consumer Discretionary",
        sub_industry="Premium Online Travel & Lodging",
        target_tier="Top 20% Leisure Travel",
        k_exposure_score=0.85,
        rationale="High-end experiential travel demand booming among asset-backed households.",
    ),
    EquityAsset(
        ticker="MAR",
        name="Marriott International",
        asset_type="k_winner",
        sector="Consumer Discretionary",
        sub_industry="Luxury Hotels & Resorts",
        target_tier="Top 15% Travel & Business Elite",
        k_exposure_score=0.8,
        rationale="Strong luxury/resort RevPAR outperformance driven by affluent travel and loyalty points.",
    ),
    EquityAsset(
        ticker="AXP",
        name="American Express Co.",
        asset_type="k_winner",
        sector="Financials",
        sub_industry="Premium Consumer Credit & Services",
        target_tier="Top 15% High-Spending Cardholders",
        k_exposure_score=0.9,
        rationale="Cardholder base has low default rates, high average ticket size, and premium fee resilience.",
    ),
    EquityAsset(
        ticker="V",
        name="Visa Inc.",
        asset_type="k_winner",
        sector="Financials",
        sub_industry="Payment Transaction Networks",
        target_tier="Broad with High Dollar Volume Skew",
        k_exposure_score=0.65,
        rationale="Nominal dollar spending capture with affluent cross-border travel transaction leverage.",
    ),
    EquityAsset(
        ticker="MS",
        name="Morgan Stanley",
        asset_type="k_winner",
        sector="Financials",
        sub_industry="Wealth Management & Private Banking",
        target_tier="Top 5% Wealth & High Net Worth",
        k_exposure_score=0.85,
        rationale="Wealth management fee revenues surge with equity market capitalization and asset prices.",
    ),
    EquityAsset(
        ticker="RH",
        name="RH (Restoration Hardware)",
        asset_type="k_winner",
        sector="Consumer Discretionary",
        sub_industry="Luxury Home Furnishings",
        target_tier="Top 5% Real Estate Owners",
        k_exposure_score=0.8,
        rationale="High-end home furnishing catering to multi-million-dollar residential real estate owners.",
    ),
    EquityAsset(
        ticker="RL",
        name="Ralph Lauren Corp.",
        asset_type="k_winner",
        sector="Consumer Discretionary",
        sub_industry="Premium Apparel & Lifestyle",
        target_tier="Top 20% Lifestyle Consumer",
        k_exposure_score=0.7,
        rationale="Successful elevated ASP (average selling price) pivot targeting affluent consumers.",
    ),
    EquityAsset(
        ticker="COST",
        name="Costco Wholesale Corp.",
        asset_type="k_winner",
        sector="Consumer Staples",
        sub_industry="Affluent Membership Warehouse",
        target_tier="Top 25% High-Earner Suburban ($130k+ median)",
        k_exposure_score=0.85,
        rationale="High-income member base ($130k+ average household income), record 93%+ renewal rates, and massive bulk discretionary spending resilience.",
    ),

    # --- K-SHAPED LOSERS / DEFENSIVES / SUBPRIME SENSITIVE (Lower & Median Quartile Distress) ---
    EquityAsset(
        ticker="DLTR",
        name="Dollar Tree Inc.",
        asset_type="k_loser",
        sector="Consumer Staples",
        sub_industry="Discount Retail & Dollar Stores",
        target_tier="Bottom 40% Income",
        k_exposure_score=-0.85,
        rationale="Core low-income customer base face severe real wage compression and cuts to SNAP benefits.",
    ),
    EquityAsset(
        ticker="DG",
        name="Dollar General Corp.",
        asset_type="k_loser",
        sector="Consumer Staples",
        sub_industry="Rural Discount Retail",
        target_tier="Bottom 30% Income (<$35k)",
        k_exposure_score=-0.9,
        rationale="Deeply exposed to financially fragile rural and lower-income households struggling with necessities.",
    ),
    EquityAsset(
        ticker="BBY",
        name="Best Buy Co. Inc.",
        asset_type="k_loser",
        sector="Consumer Discretionary",
        sub_industry="Consumer Electronics Retail",
        target_tier="Middle 50% Mass Discretionary",
        k_exposure_score=-0.7,
        rationale="Big-ticket middle-income discretionary electronics deferred due to credit card APR strain.",
    ),
    EquityAsset(
        ticker="ALLY",
        name="Ally Financial Inc.",
        asset_type="k_loser",
        sector="Financials",
        sub_industry="Auto Lending & Consumer Finance",
        target_tier="Middle to Subprime Borrowers",
        k_exposure_score=-0.85,
        rationale="Surging auto loan net charge-offs (NCOs) and 60+ day delinquencies in mass-market credit.",
    ),
    EquityAsset(
        ticker="KSS",
        name="Kohl's Corp.",
        asset_type="k_loser",
        sector="Consumer Discretionary",
        sub_industry="Department Stores",
        target_tier="Middle 40% Income",
        k_exposure_score=-0.75,
        rationale="Middle-class department store traffic erosion and proprietary credit card receivable losses.",
    ),
    EquityAsset(
        ticker="FIVE",
        name="Five Below Inc.",
        asset_type="k_loser",
        sector="Consumer Discretionary",
        sub_industry="Value Discretionary Retail",
        target_tier="Teens & Value-Seeking Families",
        k_exposure_score=-0.65,
        rationale="Value discretionary impulse purchases cut by budget-constrained lower/middle income families.",
    ),
    EquityAsset(
        ticker="OLLI",
        name="Ollie's Bargain Outlet Holdings",
        asset_type="k_loser",
        sector="Consumer Discretionary",
        sub_industry="Extreme Value Closeout Retail",
        target_tier="Bottom 50% Value Seekers",
        k_exposure_score=-0.5,
        rationale="Trade-down beneficiary initially, but vulnerable to extreme purchasing power exhaustion.",
    ),
    EquityAsset(
        ticker="CACC",
        name="Credit Acceptance Corp.",
        asset_type="k_loser",
        sector="Financials",
        sub_industry="Subprime Auto Financing",
        target_tier="Bottom 20% Subprime Credit",
        k_exposure_score=-0.95,
        rationale="High default and repossession risk on deep subprime vehicle installment loans.",
    ),
    EquityAsset(
        ticker="WMT",
        name="Walmart Inc.",
        asset_type="k_loser",
        sector="Consumer Staples",
        sub_industry="Hypermarket & Discount Groceries",
        target_tier="Middle & Lower 50% Mass Market",
        k_exposure_score=-0.65,
        rationale="High volume of low-margin grocery and staple transactions, but heavily exposed to wage-compressed shoppers reducing general merchandise margin baskets.",
    ),
    EquityAsset(
        ticker="TGT",
        name="Target Corp.",
        asset_type="k_loser",
        sector="Consumer Discretionary",
        sub_industry="General Merchandise & Apparel",
        target_tier="Middle 40% Discretionary Shopper",
        k_exposure_score=-0.45,
        rationale="Severe discretionary pull-back in home, electronics, and casual apparel as middle-class budgets are consumed by inflation.",
    ),

    # --- BENCHMARK ---
    EquityAsset(
        ticker="SPY",
        name="SPDR S&P 500 ETF Trust",
        asset_type="benchmark",
        sector="Broad Market",
        sub_industry="Large Cap Equity Benchmark",
        target_tier="Market Cap Weighted",
        k_exposure_score=0.0,
        rationale="Standard broad market equity benchmark.",
    ),
]


ASSET_MAP: Dict[str, EquityAsset] = {asset.ticker: asset for asset in UNIVERSE_ASSETS}

WINNER_TICKERS: List[str] = [a.ticker for a in UNIVERSE_ASSETS if a.asset_type == "k_winner"]
LOSER_TICKERS: List[str] = [a.ticker for a in UNIVERSE_ASSETS if a.asset_type == "k_loser"]
ALL_TICKERS: List[str] = [a.ticker for a in UNIVERSE_ASSETS]
MODEL_TICKERS: List[str] = [a.ticker for a in UNIVERSE_ASSETS if a.asset_type != "benchmark"]


def get_universe_metadata() -> List[Dict]:
    """Return dictionary list of all universe assets for reporting and inspection."""
    return [
        {
            "ticker": a.ticker,
            "name": a.name,
            "asset_type": a.asset_type,
            "sector": a.sector,
            "sub_industry": a.sub_industry,
            "target_tier": a.target_tier,
            "k_exposure_score": a.k_exposure_score,
            "rationale": a.rationale,
        }
        for a in UNIVERSE_ASSETS
    ]
