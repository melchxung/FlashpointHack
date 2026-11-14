"""
Unit tests for Gemini Scorer & Flashpoint MCP integration.
"""

import pytest

from src.sampler.gemini_scorer import GeminiScorer
from src.sampler.gemini_mcp_client import GeminiFlashpointMCPClient


def test_gemini_scorer_requires_key():
    # No heuristic fallback: a missing key must raise
    with pytest.raises(RuntimeError):
        GeminiScorer(api_key="")


def test_gemini_scorer_live():
    scorer = GeminiScorer()
    transcript = {
        "respondent_id": "test_resp_live_gemini",
        "retailer": "COST",
        "cohort": "high_income_costco",
        "transcript_text": "I spent $450 at Costco on steaks and wine. Inflation on TV is scary, but our budget is solid.",
        "avg_basket_size": 450.0,
        "trips_per_month": 3.0,
        "renewal_intent": "yes",
    }
    res = scorer.score_transcript(transcript)
    assert res.scoring_engine == scorer.model_name
    assert -1.0 <= res.stated_macro_sentiment <= 1.0
    assert -1.0 <= res.revealed_spend_conviction <= 1.0
    assert 0.0 <= res.trade_down_risk <= 1.0


def test_flashpoint_mcp_lists_tools_live():
    tools = {t["name"] for t in GeminiFlashpointMCPClient().list_tools()}
    assert {"demographics", "panel", "tool_survey"} <= tools


def test_gemini_drives_flashpoint_mcp_without_paying():
    # Live Gemini -> Flashpoint MCP round trip with payment disabled: the paid tool must
    # answer with its x402 requirement, and no USDC may be spent.
    from src.sampler.gemini_mcp_agent import run_flashpoint_agent

    mcp = GeminiFlashpointMCPClient(auto_pay=False)
    out = run_flashpoint_agent(
        "What is the household income distribution for ZIP code 94110? Use Flashpoint demographics.",
        mcp=mcp,
    )
    demo_calls = [c for c in out["tool_calls"] if c["tool"] == "demographics"]
    assert demo_calls, "Gemini did not call the Flashpoint demographics tool"
    assert all(not c["ok"] and "x402 payment" in c["error"] for c in demo_calls)
    assert out["spent_usdc"] == 0.0
