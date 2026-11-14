"""
Sentiment Sampling Engine & K-Shaped Divergence Index Calculator.

Orchestrates sampling across stratified synthetic panels (Upper, Middle, Lower income cohorts),
computes University of Michigan Index of Consumer Sentiment (ICS, ICC, ICE) metrics,
and evaluates spending-weighted sentiment (S_K) vs. standard equal-weighted sentiment.
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

from src.sampler.personas import (
    PersonaArchetype,
    CohortSpecification,
    COHORT_REGISTRY,
    UPPER_QUARTILE_SPEC,
    MIDDLE_INCOME_SPEC,
    LOWER_QUARTILE_SPEC,
)
from src.sampler.survey import SurveyBattery, QuestionMetrics
from src.sampler.client import FlashpointClient

logger = logging.getLogger(__name__)


@dataclass
class PeriodSentimentRecord:
    """Detailed sentiment metrics for a single observation date."""
    date: str
    S_upper: float
    S_middle: float
    S_lower: float
    S_equal: float
    S_K: float
    Delta_S: float
    ICC_upper: float
    ICE_upper: float
    ICC_lower: float
    ICE_lower: float
    ICC_middle: float
    ICE_middle: float
    relative_scores_upper: Dict[str, float]
    relative_scores_lower: Dict[str, float]
    relative_scores_middle: Dict[str, float]
    completions_total: int


@dataclass
class SamplingResult:
    """Complete sentiment time series and summary metadata."""
    dates: List[str]
    S_upper: List[float]
    S_lower: List[float]
    S_middle: List[float]
    S_equal: List[float]
    S_K: List[float]
    Delta_S: List[float]
    spending_weights: Dict[str, float]
    period_records: List[PeriodSentimentRecord]
    responses_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "metadata": {
                "description": "K-Shaped Consumer Sentiment Index & Spread Signal",
                "survey_battery": "University of Michigan 5 Core Questions (ICS/ICC/ICE)",
                "methodology": "Synthetic Persona Stratified Sampling (Flashpoint AI / Touchpoint)",
                "spending_weights": self.spending_weights,
                "n_periods": len(self.dates),
                "date_range": [self.dates[0], self.dates[-1]] if self.dates else [],
            },
            "dates": self.dates,
            "S_upper": self.S_upper,
            "S_lower": self.S_lower,
            "S_middle": self.S_middle,
            "S_equal": self.S_equal,
            "S_K": self.S_K,
            "Delta_S": self.Delta_S,
            "responses_summary": self.responses_summary,
            "period_records": [asdict(r) for r in self.period_records],
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert time series to pandas DataFrame with Date index."""
        df = pd.DataFrame({
            "date": pd.to_datetime(self.dates),
            "S_upper": self.S_upper,
            "S_middle": self.S_middle,
            "S_lower": self.S_lower,
            "S_equal": self.S_equal,
            "S_K": self.S_K,
            "Delta_S": self.Delta_S,
        })
        df.set_index("date", inplace=True)
        return df


class SentimentSamplingEngine:
    """
    Core engine managing persona sampling, question tabulation, and sentiment index calculation.
    """

    def __init__(
        self,
        client: Optional[FlashpointClient] = None,
        weight_upper: float = 0.48,   # Top 20-25% drives ~48% of aggregate PCE
        weight_middle: float = 0.36,  # Middle 40% drives ~36% of aggregate PCE
        weight_lower: float = 0.16,   # Bottom 35% drives ~16% of aggregate PCE
    ):
        self.client = client or FlashpointClient()
        # Normalize spending weights to ensure they sum to exactly 1.0
        total_w = weight_upper + weight_middle + weight_lower
        self.weight_upper = weight_upper / total_w
        self.weight_middle = weight_middle / total_w
        self.weight_lower = weight_lower / total_w

    def sample_single_period(
        self,
        date_str: str,
        num_respondents_per_cohort: int = 50,
        seed_offset: int = 0,
    ) -> Tuple[PeriodSentimentRecord, Dict[str, Any]]:
        """
        Field the 5-question battery across all three cohorts for a single period.
        Calculate Michigan indices and divergence spreads.
        """
        cohort_results = {}
        cohort_relative_scores = {}
        cohort_indices = {}
        total_completions = 0
        panel_summaries = {}

        archetypes = [
            PersonaArchetype.UPPER_QUARTILE,
            PersonaArchetype.MIDDLE_INCOME,
            PersonaArchetype.LOWER_QUARTILE,
        ]

        for idx, arch in enumerate(archetypes):
            panel_data = self.client.create_and_run_panel(
                archetype=arch,
                num_respondents=num_respondents_per_cohort,
                date_str=date_str,
                seed_offset=seed_offset * 10 + idx,
            )
            cohort_results[arch] = panel_data
            total_completions += panel_data.get("completions", 0)
            panel_summaries[arch.value] = {
                "panel_id": panel_data.get("panel_id"),
                "fit_score": panel_data.get("fit_score"),
                "completions": panel_data.get("completions"),
            }

            # Extract answers per question
            responses_list = panel_data.get("responses", [])
            q_responses: Dict[str, List[int]] = {
                "PAGO": [], "PAGED": [], "BUS12": [], "BUS5": [], "DUR": []
            }
            for resp in responses_list:
                ans = resp.get("answers", {})
                for q_code in q_responses:
                    if q_code in ans:
                        q_responses[q_code].append(int(ans[q_code]))

            # Calculate relative score for each question
            rel_scores = {}
            for q_code, vals in q_responses.items():
                metrics = SurveyBattery.calculate_relative_score(vals)
                rel_scores[q_code] = metrics.relative_score

            cohort_relative_scores[arch] = rel_scores
            cohort_indices[arch] = SurveyBattery.compute_indices(rel_scores)

        # Extract primary ICS indices
        s_upper = cohort_indices[PersonaArchetype.UPPER_QUARTILE]["ICS"]
        s_middle = cohort_indices[PersonaArchetype.MIDDLE_INCOME]["ICS"]
        s_lower = cohort_indices[PersonaArchetype.LOWER_QUARTILE]["ICS"]

        # Calculate K-spread divergence: Delta_S = S_upper - S_lower
        delta_s = round(s_upper - s_lower, 2)

        # Equal-weighted sentiment (Standard headline proxy)
        s_equal = round(0.25 * s_upper + 0.50 * s_middle + 0.25 * s_lower, 2)

        # Spending-weighted sentiment (Dollar-weighted PCE proxy)
        s_k = round(
            self.weight_upper * s_upper +
            self.weight_middle * s_middle +
            self.weight_lower * s_lower,
            2
        )

        record = PeriodSentimentRecord(
            date=date_str,
            S_upper=s_upper,
            S_middle=s_middle,
            S_lower=s_lower,
            S_equal=s_equal,
            S_K=s_k,
            Delta_S=delta_s,
            ICC_upper=cohort_indices[PersonaArchetype.UPPER_QUARTILE]["ICC"],
            ICE_upper=cohort_indices[PersonaArchetype.UPPER_QUARTILE]["ICE"],
            ICC_lower=cohort_indices[PersonaArchetype.LOWER_QUARTILE]["ICC"],
            ICE_lower=cohort_indices[PersonaArchetype.LOWER_QUARTILE]["ICE"],
            ICC_middle=cohort_indices[PersonaArchetype.MIDDLE_INCOME]["ICC"],
            ICE_middle=cohort_indices[PersonaArchetype.MIDDLE_INCOME]["ICE"],
            relative_scores_upper=cohort_relative_scores[PersonaArchetype.UPPER_QUARTILE],
            relative_scores_lower=cohort_relative_scores[PersonaArchetype.LOWER_QUARTILE],
            relative_scores_middle=cohort_relative_scores[PersonaArchetype.MIDDLE_INCOME],
            completions_total=total_completions,
        )

        return record, panel_summaries

    def sample_historical_time_series(
        self,
        start_date: str = "2021-01",
        end_date: str = "2024-12",
        num_respondents_per_cohort: int = 50,
    ) -> SamplingResult:
        """
        Generate monthly time series of sentiment signals across the specified date range.
        """
        dates = pd.date_range(start=start_date, end=end_date, freq="MS").strftime("%Y-%m").tolist()
        logger.info(f"Generating sentiment time series across {len(dates)} periods ({dates[0]} to {dates[-1]})...")

        period_records: List[PeriodSentimentRecord] = []
        s_upper_series: List[float] = []
        s_lower_series: List[float] = []
        s_middle_series: List[float] = []
        s_equal_series: List[float] = []
        s_k_series: List[float] = []
        delta_s_series: List[float] = []

        total_panel_completions = 0
        fit_scores = []

        for idx, date_str in enumerate(dates):
            record, panel_summaries = self.sample_single_period(
                date_str=date_str,
                num_respondents_per_cohort=num_respondents_per_cohort,
                seed_offset=idx,
            )
            period_records.append(record)
            s_upper_series.append(record.S_upper)
            s_lower_series.append(record.S_lower)
            s_middle_series.append(record.S_middle)
            s_equal_series.append(record.S_equal)
            s_k_series.append(record.S_K)
            delta_s_series.append(record.Delta_S)

            total_panel_completions += record.completions_total
            for summary in panel_summaries.values():
                fit = summary.get("fit_score")
                if fit is not None:
                    fit_scores.append(fit)

        responses_summary = {
            "total_periods": len(dates),
            "total_synthetic_respondents": total_panel_completions,
            "respondents_per_cohort_per_period": num_respondents_per_cohort,
            "average_fit_score": round(float(np.mean(fit_scores)), 3) if fit_scores else 0.95,
            "cohort_definitions": {
                "upper_quartile": {
                    "income_range": "$150,000 - $450,000+",
                    "net_worth": "$800,000 - $5,000,000+",
                    "spending_share": self.weight_upper,
                },
                "middle_income": {
                    "income_range": "$65,000 - $120,000",
                    "net_worth": "$60,000 - $300,000",
                    "spending_share": self.weight_middle,
                },
                "lower_quartile": {
                    "income_range": "$24,000 - $58,000",
                    "net_worth": "<$35,000",
                    "spending_share": self.weight_lower,
                },
            },
        }

        return SamplingResult(
            dates=dates,
            S_upper=s_upper_series,
            S_lower=s_lower_series,
            S_middle=s_middle_series,
            S_equal=s_equal_series,
            S_K=s_k_series,
            Delta_S=delta_s_series,
            spending_weights={
                "upper": round(self.weight_upper, 3),
                "middle": round(self.weight_middle, 3),
                "lower": round(self.weight_lower, 3),
            },
            period_records=period_records,
            responses_summary=responses_summary,
        )

    def save_to_json(self, result: SamplingResult, filepath: str) -> None:
        """Save sampling result to JSON file."""
        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Saved sentiment signals to {filepath}")

    def save_to_csv(self, result: SamplingResult, filepath: str) -> None:
        """Save sentiment signals time series to CSV file."""
        df = result.to_dataframe()
        df.to_csv(filepath)
        logger.info(f"Saved sentiment signals CSV to {filepath}")
