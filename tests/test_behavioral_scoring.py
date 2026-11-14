"""
Unit Tests for Retail Behavioral Probe, Claude Scorer & Survey Replicator.
"""

import os
import json
import pytest
import numpy as np

from src.sampler.behavioral_probe import (
    generate_retail_respondents,
    ConversationalTranscript,
)
from src.sampler.claude_scorer import ClaudeScorer, ClaudeExtractionResult
from src.sampler.survey_replicator import run_affluent_survey_replications


def test_generate_retail_respondents_distribution():
    transcripts = generate_retail_respondents(n_per_cohort=10, random_state=np.random.RandomState(42))
    assert len(transcripts) == 30
    retailers = [t.retailer for t in transcripts]
    assert retailers.count("COST") == 10
    assert retailers.count("TGT") == 10
    assert retailers.count("WMT") == 10

    # Test COST specific traits
    cost_t = next(t for t in transcripts if t.retailer == "COST")
    assert cost_t.profile["household_income"] >= 120_000
    assert cost_t.profile["avg_basket_size"] >= 200.0
    assert len(cost_t.qa_turns) == 4
    assert "Kirkland" in cost_t.transcript_text or "receipt" in cost_t.transcript_text.lower()


def test_claude_scorer_fallback_scoring():
    scorer = ClaudeScorer()
    transcripts = generate_retail_respondents(n_per_cohort=2, random_state=np.random.RandomState(42))
    results = scorer.score_batch(transcripts)
    assert len(results) == 6

    for r in results:
        assert isinstance(r, ClaudeExtractionResult)
        assert -1.0 <= r.stated_macro_sentiment <= 1.0
        assert -1.0 <= r.revealed_spend_conviction <= 1.0
        assert 0.0 <= r.trade_down_risk <= 1.0
        assert abs(r.vibecession_divergence - (r.revealed_spend_conviction - r.stated_macro_sentiment)) < 1e-4

    # COST respondents should show high positive vibecession divergence
    cost_res = [r for r in results if r.retailer == "COST"]
    for cr in cost_res:
        assert cr.vibecession_divergence > 1.0
        assert cr.revealed_spend_conviction > 0.5
        assert cr.stated_macro_sentiment < 0.0


def test_affluent_survey_replications():
    res = run_affluent_survey_replications(seed=42)
    assert "survey_batteries" in res
    assert "michigan_ics" in res["survey_batteries"]
    assert "conference_board_cci" in res["survey_batteries"]
    assert "morning_consult_ics" in res["survey_batteries"]
    assert "rasmussen_rci" in res["survey_batteries"]

    # Each affluent score must strictly exceed baseline
    for k, b in res["survey_batteries"].items():
        assert b["affluent_score"] > b["baseline_population_score"]
        assert b["divergence_wedge"] > 20.0

    assert len(res["time_series"]) == 48
