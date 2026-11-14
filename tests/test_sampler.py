"""
Unit & Integration Tests for Flashpoint AI Persona Sampler & Sentiment Engine.
"""

import os
import json
import tempfile
import pytest
import numpy as np
import pandas as pd

from src.sampler.survey import (
    MICHIGAN_QUESTIONS,
    SurveyBattery,
    QuestionMetrics,
)
from src.sampler.personas import (
    PersonaArchetype,
    CohortSpecification,
    UPPER_QUARTILE_SPEC,
    MIDDLE_INCOME_SPEC,
    LOWER_QUARTILE_SPEC,
    COHORT_REGISTRY,
    generate_synthetic_persona,
)
from src.sampler.simulator import SyntheticPersonaSimulator
from src.sampler.client import FlashpointClient
from src.sampler.engine import SentimentSamplingEngine, SamplingResult, PeriodSentimentRecord


class TestSurveyBattery:
    """Tests for Michigan survey core questions and formula index calculation."""

    def test_five_core_questions_defined(self):
        expected_keys = {"PAGO", "PAGED", "BUS12", "BUS5", "DUR"}
        assert set(MICHIGAN_QUESTIONS.keys()) == expected_keys
        for code, q in MICHIGAN_QUESTIONS.items():
            assert q.code == code
            assert q.favorable_code == 1
            assert q.neutral_code == 2
            assert q.unfavorable_code == 3
            assert len(q.options) == 3

    def test_relative_score_all_favorable(self):
        responses = [1, 1, 1, 1]
        metrics = SurveyBattery.calculate_relative_score(responses)
        assert metrics.pct_favorable == 100.0
        assert metrics.pct_neutral == 0.0
        assert metrics.pct_unfavorable == 0.0
        assert metrics.relative_score == 200.0

    def test_relative_score_all_unfavorable(self):
        responses = [3, 3, 3, 3]
        metrics = SurveyBattery.calculate_relative_score(responses)
        assert metrics.pct_favorable == 0.0
        assert metrics.pct_neutral == 0.0
        assert metrics.pct_unfavorable == 100.0
        assert metrics.relative_score == 0.0

    def test_relative_score_all_neutral(self):
        responses = [2, 2, 2, 2]
        metrics = SurveyBattery.calculate_relative_score(responses)
        assert metrics.pct_favorable == 0.0
        assert metrics.pct_neutral == 100.0
        assert metrics.pct_unfavorable == 0.0
        assert metrics.relative_score == 100.0

    def test_relative_score_empty(self):
        metrics = SurveyBattery.calculate_relative_score([])
        assert metrics.relative_score == 100.0
        assert metrics.n_responses == 0

    def test_compute_indices_baseline_100(self):
        # When all relative scores are 100:
        # Sum = 500
        # ICS = 500 / 6.7558 + 2.0 = 74.01 + 2.0 = 76.01
        # ICC = 200 / 2.6416 + 2.0 = 75.71 + 2.0 = 77.71
        # ICE = 300 / 4.1134 + 2.0 = 72.93 + 2.0 = 74.93
        scores = {"PAGO": 100.0, "PAGED": 100.0, "BUS12": 100.0, "BUS5": 100.0, "DUR": 100.0}
        indices = SurveyBattery.compute_indices(scores)
        assert abs(indices["ICS"] - 76.01) < 0.1
        assert abs(indices["ICC"] - 77.71) < 0.1
        assert abs(indices["ICE"] - 74.93) < 0.1


class TestPersonas:
    """Tests for cohort specifications and synthetic persona profiles."""

    def test_cohort_registry_weights(self):
        assert UPPER_QUARTILE_SPEC.target_spending_share == 0.48
        assert MIDDLE_INCOME_SPEC.target_spending_share == 0.36
        assert LOWER_QUARTILE_SPEC.target_spending_share == 0.16
        total_pce_share = (
            UPPER_QUARTILE_SPEC.target_spending_share +
            MIDDLE_INCOME_SPEC.target_spending_share +
            LOWER_QUARTILE_SPEC.target_spending_share
        )
        assert abs(total_pce_share - 1.0) < 1e-6

    def test_generate_upper_persona(self):
        rng = np.random.RandomState(42)
        p = generate_synthetic_persona(PersonaArchetype.UPPER_QUARTILE, "test_u_1", rng)
        assert p.household_income >= 150_000
        assert p.liquid_net_worth >= 800_000
        assert p.credit_score >= 750
        assert p.archetype == PersonaArchetype.UPPER_QUARTILE

    def test_generate_lower_persona(self):
        rng = np.random.RandomState(42)
        p = generate_synthetic_persona(PersonaArchetype.LOWER_QUARTILE, "test_l_1", rng)
        assert p.household_income <= 60_000
        assert p.liquid_net_worth < 50_000
        assert p.credit_score < 700
        assert p.credit_card_debt >= 4_000
        assert p.archetype == PersonaArchetype.LOWER_QUARTILE


class TestSyntheticPersonaSimulator:
    """Tests for synthetic sampling simulator."""

    def test_run_cohort_panel_schema(self):
        sim = SyntheticPersonaSimulator(seed=123)
        res = sim.run_cohort_panel(
            archetype=PersonaArchetype.UPPER_QUARTILE,
            num_respondents=10,
            date_str="2023-06",
        )
        assert res["status"] == "success"
        assert res["source"] == "synthetic"
        assert res["completions"] == 10
        assert len(res["responses"]) == 10
        assert 0.90 <= res["fit_score"] <= 1.0

        first_resp = res["responses"][0]
        assert "persona" in first_resp
        assert "answers" in first_resp
        answers = first_resp["answers"]
        for q in ["PAGO", "PAGED", "BUS12", "BUS5", "DUR"]:
            assert answers[q] in [1, 2, 3]

    def test_reproducibility_with_seed(self):
        sim1 = SyntheticPersonaSimulator(seed=999)
        sim2 = SyntheticPersonaSimulator(seed=999)
        res1 = sim1.run_cohort_panel(PersonaArchetype.LOWER_QUARTILE, 5, "2022-01")
        res2 = sim2.run_cohort_panel(PersonaArchetype.LOWER_QUARTILE, 5, "2022-01")
        assert res1["responses"][0]["answers"] == res2["responses"][0]["answers"]


class TestFlashpointClient:
    """Tests for API client fallback behavior."""

    def test_client_fallback_without_keys(self):
        # When no API key is provided, should default to synthetic simulator
        client = FlashpointClient(api_token=None, fallback_to_simulator=True)
        assert not client.has_credentials
        res = client.create_and_run_panel(
            archetype=PersonaArchetype.UPPER_QUARTILE,
            num_respondents=5,
            date_str="2024-01",
        )
        assert res["status"] == "success"
        assert res["source"] == "synthetic"
        assert res["completions"] == 5

    def test_client_raises_when_no_credentials_and_fallback_disabled(self):
        client = FlashpointClient(api_token=None, fallback_to_simulator=False)
        with pytest.raises(ValueError, match="No Flashpoint API token configured"):
            client.create_and_run_panel(PersonaArchetype.UPPER_QUARTILE)


class TestSentimentSamplingEngine:
    """Tests for sampling engine orchestration and index calculations."""

    def test_weight_normalization(self):
        # Weights that do not sum to 1.0 should be automatically normalized
        engine = SentimentSamplingEngine(weight_upper=48, weight_middle=36, weight_lower=16)
        assert abs(engine.weight_upper - 0.48) < 1e-4
        assert abs(engine.weight_middle - 0.36) < 1e-4
        assert abs(engine.weight_lower - 0.16) < 1e-4

    def test_sample_single_period(self):
        engine = SentimentSamplingEngine()
        record, summaries = engine.sample_single_period("2023-01", num_respondents_per_cohort=15)
        assert record.date == "2023-01"
        assert record.S_upper > record.S_lower
        assert record.Delta_S == round(record.S_upper - record.S_lower, 2)
        assert record.completions_total == 45
        assert len(summaries) == 3

    def test_time_series_generation_and_export(self):
        engine = SentimentSamplingEngine()
        # Run a 3-month mini-series
        result = engine.sample_historical_time_series(
            start_date="2023-01",
            end_date="2023-03",
            num_respondents_per_cohort=10,
        )
        assert len(result.dates) == 3
        assert len(result.S_upper) == 3
        assert len(result.Delta_S) == 3

        # Test DataFrame conversion
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "S_upper" in df.columns
        assert "Delta_S" in df.columns

        # Test JSON serialization and save
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            engine.save_to_json(result, tmp_path)
            assert os.path.exists(tmp_path)
            with open(tmp_path) as f:
                loaded = json.load(f)
            assert "dates" in loaded
            assert "S_upper" in loaded
            assert "Delta_S" in loaded
            assert len(loaded["dates"]) == 3
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
