"""
Unit tests for Gemini Scorer & Flashpoint MCP integration.
"""

from src.sampler.gemini_scorer import GeminiScorer
from src.sampler.gemini_mcp_client import GeminiFlashpointMCPClient


def test_gemini_scorer_fallback():
    # Explicit empty key to test fallback logic
    scorer = GeminiScorer(api_key="")
    assert not scorer.is_live

    transcript = {
        "respondent_id": "test_resp_01",
        "retailer": "COST",
        "cohort": "high_income_costco",
        "transcript_text": "I spent $420 at Costco on prime beef and wine. Economy feels uncertain on TV.",
        "avg_basket_size": 420.0,
        "trips_per_month": 2.5,
        "renewal_intent": "yes",
    }

    result = scorer.score_transcript(transcript, idx=0)
    assert result.respondent_id == "test_resp_01"
    assert result.retailer == "COST"
    assert -1.0 <= result.stated_macro_sentiment <= 1.0
    assert -1.0 <= result.revealed_spend_conviction <= 1.0
    assert 0.0 <= result.trade_down_risk <= 1.0
    assert result.vibecession_divergence > 0.5  # Costco affluent gap should be strongly positive
    assert result.scoring_engine == "gemini_semantic_fallback"


def test_gemini_scorer_live_if_configured():
    scorer = GeminiScorer()
    if scorer.is_live:
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
        assert -1.0 <= res.stated_macro_sentiment <= 1.0
        assert -1.0 <= res.revealed_spend_conviction <= 1.0
        assert res.vibecession_divergence > 0.0


def test_gemini_mcp_client_simulation():
    client = GeminiFlashpointMCPClient()
    panel_res = client.call_mcp_tool("panel", {"size": 25, "filters": {"hhi_min": 150000}})
    assert panel_res is not None
    sim = panel_res.get("simulated_response", panel_res)
    assert sim["cohort_id"] == "coh_affluent_costco_mcp_01"
    assert sim["matched_count"] == 25
