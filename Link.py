import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np

# Verified University of Michigan Consumer Sentiment Friday release dates (10:00 AM ET)
UMICH_RELEASES = [
    # 2025 Releases
    "2025-05-16", "2025-05-30", "2025-08-15", "2025-08-29",
    "2025-11-14", "2025-11-26",
    # 2026 Releases (including May 44.8 record low & Oct 48.0 trough)
    "2026-01-09", "2026-01-23", "2026-02-06", "2026-02-20",
    "2026-03-13", "2026-03-27", "2026-04-10", "2026-04-24",
    "2026-05-08", "2026-05-22", "2026-06-12", "2026-06-26",
    "2026-07-10", "2026-07-24", "2026-08-14", "2026-08-28",
    "2026-09-11", "2026-09-25", "2026-10-09", "2026-10-23",
    "2026-11-13"
]

TICKERS = ["COST", "WMT", "TGT", "XRT"]
BENCHMARK = "SPY"

def run_sentiment_event_study(start="2025-01-01", end="2026-11-14"):
    data = yf.download(TICKERS + [BENCHMARK], start=start, end=end, progress=False)["Close"]
    returns = data.pct_change().dropna()

    results = []
    for ticker in TICKERS:
        beta, alpha = np.polyfit(returns[BENCHMARK].values, returns[ticker].values, 1)
        ar = returns[ticker] - (alpha + beta * returns[BENCHMARK])

        for dt_str in UMICH_RELEASES:
            dt = pd.to_datetime(dt_str)
            if dt not in ar.index:
                continue
            idx = ar.index.get_loc(dt)
            if idx + 15 >= len(ar):
                continue

            ar_0 = ar.iloc[idx]
            car_1_5 = ar.iloc[idx+1 : idx+6].sum()
            car_1_15 = ar.iloc[idx+1 : idx+16].sum()

            results.append({
                "Ticker": ticker,
                "Release_Date": dt_str,
                "AR_Day0 (%)": round(ar_0 * 100, 3),
                "CAR_1_to_5d (%)": round(car_1_5 * 100, 3),
                "CAR_1_to_15d (%)": round(car_1_15 * 100, 3),
            })

    df_res = pd.DataFrame(results)
    # Isolate days where sentiment release triggered a Day-0 selloff (AR_Day0 < 0)
    neg_shocks = df_res[df_res["AR_Day0 (%)"] < 0]
    summary_neg = neg_shocks.groupby("Ticker")[["AR_Day0 (%)", "CAR_1_to_5d (%)", "CAR_1_to_15d (%)"]].mean()
    return df_res, neg_shocks, summary_neg

if __name__ == "__main__":
    all_events, neg_shocks, summary_neg = run_sentiment_event_study()
    print("=== Average Reaction & Drift on Negative Sentiment Shock Days (AR_Day0 < 0) ===")
    print(summary_neg)
    print("\n=== COST Negative Sentiment Shock Days ===")
    print(neg_shocks[neg_shocks["Ticker"] == "COST"])
