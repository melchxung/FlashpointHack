# FlashpointHack

A research pipeline that tests one idea: does the sentiment gap between affluent and other
US consumers (the "K-shape") predict stock returns?

Sentiment comes from [Flashpoint AI](https://docs.flashpoint.ai) synthetic panels. Google Gemini
issues every Flashpoint call over the Flashpoint MCP server, and an agent wallet pays per call
in USDC (x402).

## Result

On the data in this repo, the signal shows no edge.

| Test | Result |
|---|---|
| Long/short backtest, 1-month horizon | -12.2% annualised, Sharpe -0.43 |
| Long/short backtest, 3-month horizon | -10.5% annualised, Sharpe -0.38 |
| Jensen's alpha (1M / 3M, vs SPY excess returns) | -6.4%/yr (t = -0.61, p = 0.54) / -7.4%/yr (t = -1.15, p = 0.26) — no significant alpha |
| Rank IC, 1-month | +0.013 (p = 0.71, not significant) |
| Affluent signal vs headline sentiment (3-month IC) | not superior (one-tailed p = 0.59) |
| COST / WMT / affluent sentiment cointegration | not cointegrated (p = 0.29) |

## How it works

1. Three Flashpoint panels of 30 AI personas each (affluent, middle and lower income) answer the
   five University of Michigan sentiment questions for every month from January 2021 to
   September 2026 (69 months; October 2026 is missing one cohort answer and is excluded).
2. The answers become a sentiment index per cohort. The affluent-minus-middle gap is the signal.
3. For each stock, the code measures how its monthly return has moved with changes in that gap
   over the past 12 months (a rolling beta), using data up to that month only.
4. The beta is multiplied by a regime signal: positive when the gap is wide and accelerating
   against its own past year, negative when it is narrow and slowing.
5. Stocks are ranked by that product each month. The backtest buys the top fifth, shorts the
   bottom fifth, and measures the return over the next one and three months.

## Setup

Tested on Python 3.14.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

The pipeline reads the saved data in `data/` and makes no network or paid calls. Two runs
produce identical output files.

```bash
.venv/bin/python scripts/run_research_pipeline.py   # backtest, econometrics, forecast
.venv/bin/python -m pytest -q                       # tests
.venv/bin/streamlit run app.py                      # dashboard
```

## Live mode (optional, costs money)

Live calls need the keys listed in `.env.example`, copied into a local `.env`, and a wallet
holding USDC on Base. The client caps spend at $0.60 per call and $5.00 per session.

```bash
.venv/bin/python scripts/setup_flashpoint_wallet.py               # create or load the agent wallet
.venv/bin/python scripts/run_research_pipeline.py --live          # one Flashpoint call, about $0.05
.venv/bin/python scripts/run_flashpoint_sampling.py               # rebuild the sentiment series, about $3
RUN_LIVE_TESTS=1 .venv/bin/python -m pytest -q tests/test_gemini.py
```

`run_flashpoint_sampling.py` saves every answer as it arrives, so a rerun resumes without paying twice.

## Data

| File | Source |
|---|---|
| `data/sentiment_signals.json`, `data/flashpoint_sentiment_raw.json` | Flashpoint synthetic panels. The personas answer in hindsight, so this is simulated history, not observed survey data. |
| `data/flashpoint_live_demographics.json` | US Census data returned by Flashpoint's `demographics` tool. |
| `data/market_prices.csv`, `data/ticker_fundamentals.json` | Yahoo Finance, cached. |
| `data/baseline_sentiment_surveys.csv` | Michigan (FRED), Chicago Fed C-CSI and SF Fed news sentiment are real. The Conference Board, Morning Consult, Rasmussen and Goldman Sachs columns are modelled from those series in `scripts/compile_baseline_surveys.py`. |
| `data/affluent_survey_replications.json` | Static simulated survey-battery replication (affluent cohort vs population baselines). Not part of the 2021–2026-09 monthly series. |
| `data/respondent_scored_transcripts.json` | Locally generated shopper interviews with rule-based scores. Simulated. |

## Limitations

- The COST, WMT and TGT price targets come from the simulated transcripts and fundamentals. They
  do not depend on the sentiment series.
- `src/forecast/cycle_forecaster.py` reports fixed illustrative figures for the pair trade
  (Sharpe, beat rate). They are not computed from data.
- `src/sampler/client.py` and `src/sampler/claude_scorer.py` still contain local simulator
  fallbacks. The research pipeline does not use them.

## Keys

Secrets live only in `.env`, which is gitignored. Never commit it.
