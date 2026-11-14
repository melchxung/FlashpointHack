"""
University of Michigan Index of Consumer Sentiment (ICS/ICC/ICE) Survey Battery.

Implements the official 5 core Michigan questions, standardized codings,
relative score calculations, and baseline formulas anchored to the 1966 base period.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass(frozen=True)
class MichiganQuestion:
    """Definition of a core Michigan sentiment survey question."""
    code: str
    component: str  # 'ICC' (Current Economic Conditions) or 'ICE' (Consumer Expectations)
    prompt: str
    options: Dict[int, str]
    favorable_code: int = 1
    neutral_code: int = 2
    unfavorable_code: int = 3


# The 5 Core University of Michigan Questions
MICHIGAN_QUESTIONS: Dict[str, MichiganQuestion] = {
    "PAGO": MichiganQuestion(
        code="PAGO",
        component="ICC",
        prompt=(
            "We are interested in how people are getting along financially these days. "
            "Would you say that you (and your family living there) are better off or worse off "
            "financially than you were a year ago?"
        ),
        options={
            1: "Better off",
            2: "Same / Uncertain",
            3: "Worse off"
        },
        favorable_code=1,
        neutral_code=2,
        unfavorable_code=3,
    ),
    "PAGED": MichiganQuestion(
        code="PAGED",
        component="ICE",
        prompt=(
            "Now looking ahead—do you think that a year from now you (and your family living there) "
            "will be better off financially, or worse off, or just about the same as now?"
        ),
        options={
            1: "Better off",
            2: "Same / Uncertain",
            3: "Worse off"
        },
        favorable_code=1,
        neutral_code=2,
        unfavorable_code=3,
    ),
    "BUS12": MichiganQuestion(
        code="BUS12",
        component="ICE",
        prompt=(
            "Now turning to business conditions in the country as a whole—do you think that "
            "during the next 12 months we'll have good times financially, or bad times, or what?"
        ),
        options={
            1: "Good times",
            2: "Uncertain / Depends",
            3: "Bad times"
        },
        favorable_code=1,
        neutral_code=2,
        unfavorable_code=3,
    ),
    "BUS5": MichiganQuestion(
        code="BUS5",
        component="ICE",
        prompt=(
            "Looking ahead, which would you say is more likely—that in the country as a whole "
            "we'll have continuous good times during the next 5 years or so, or that we will "
            "have periods of widespread unemployment or depression, or what?"
        ),
        options={
            1: "Continuous good times",
            2: "Uncertain / Depends",
            3: "Unemployment / Depression"
        },
        favorable_code=1,
        neutral_code=2,
        unfavorable_code=3,
    ),
    "DUR": MichiganQuestion(
        code="DUR",
        component="ICC",
        prompt=(
            "About the big things people buy for their homes—such as furniture, a refrigerator, "
            "stove, television, and things like that. Generally speaking, do you think now is a "
            "good or a bad time for people to buy major household items?"
        ),
        options={
            1: "Good time to buy",
            2: "Uncertain / Neither",
            3: "Bad time to buy"
        },
        favorable_code=1,
        neutral_code=2,
        unfavorable_code=3,
    ),
}


@dataclass
class QuestionResponse:
    """Individual response to a single question."""
    question_code: str
    response_code: int  # 1, 2, or 3
    response_text: str
    persona_id: Optional[str] = None


@dataclass
class QuestionMetrics:
    """Aggregated metrics for a question across a cohort."""
    question_code: str
    pct_favorable: float
    pct_neutral: float
    pct_unfavorable: float
    relative_score: float  # (pct_favorable - pct_unfavorable) + 100
    n_responses: int


class SurveyBattery:
    """Helper class to evaluate Michigan sentiment responses and calculate official indices."""

    # Official Base Constants (1966 base period = 100)
    ICS_DIVISOR = 6.7558
    ICC_DIVISOR = 2.6416
    ICE_DIVISOR = 4.1134
    CORRECTION_CONSTANT = 2.0

    @classmethod
    def calculate_relative_score(cls, responses: List[int]) -> QuestionMetrics:
        """
        Calculate relative score X_i = (% Favorable - % Unfavorable) + 100.
        responses: List of integers (1 = Favorable, 2 = Neutral, 3 = Unfavorable)
        """
        if not responses:
            return QuestionMetrics(
                question_code="",
                pct_favorable=0.0,
                pct_neutral=0.0,
                pct_unfavorable=0.0,
                relative_score=100.0,
                n_responses=0,
            )

        n = len(responses)
        n_fav = sum(1 for r in responses if r == 1)
        n_neu = sum(1 for r in responses if r == 2)
        n_unfav = sum(1 for r in responses if r == 3)

        pct_fav = (n_fav / n) * 100.0
        pct_neu = (n_neu / n) * 100.0
        pct_unfav = (n_unfav / n) * 100.0
        rel_score = (pct_fav - pct_unfav) + 100.0

        return QuestionMetrics(
            question_code="",
            pct_favorable=round(pct_fav, 2),
            pct_neutral=round(pct_neu, 2),
            pct_unfavorable=round(pct_unfav, 2),
            relative_score=round(rel_score, 2),
            n_responses=n,
        )

    @classmethod
    def compute_indices(cls, relative_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Compute ICS, ICC, and ICE indices from relative scores {PAGO, PAGED, BUS12, BUS5, DUR}.
        """
        pago = relative_scores.get("PAGO", 100.0)
        paged = relative_scores.get("PAGED", 100.0)
        bus12 = relative_scores.get("BUS12", 100.0)
        bus5 = relative_scores.get("BUS5", 100.0)
        dur = relative_scores.get("DUR", 100.0)

        sum_all = pago + paged + bus12 + bus5 + dur
        ics = (sum_all / cls.ICS_DIVISOR) + cls.CORRECTION_CONSTANT

        sum_icc = pago + dur
        icc = (sum_icc / cls.ICC_DIVISOR) + cls.CORRECTION_CONSTANT

        sum_ice = paged + bus12 + bus5
        ice = (sum_ice / cls.ICE_DIVISOR) + cls.CORRECTION_CONSTANT

        return {
            "ICS": round(ics, 2),
            "ICC": round(icc, 2),
            "ICE": round(ice, 2),
        }
