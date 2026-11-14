"""
Synthetic Persona Archetypes & Stratification Matrix.

Defines the demographic, financial balance sheet, psychographic, and behavioral
profiles for:
1. Upper-Quartile Affluent Consumers (Top 20-25% income, asset-backed, discretionary spenders)
2. Middle-Income Baseline Consumers (40th-70th percentile)
3. Lower/Median-Quartile Vulnerable Consumers (Bottom 35-40% income, rent/debt squeezed)

Includes macroeconomic consumption weighting based on Federal Reserve SCF and BEA data.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np


class PersonaArchetype(str, Enum):
    UPPER_QUARTILE = "upper_quartile"
    MIDDLE_INCOME = "middle_income"
    LOWER_QUARTILE = "lower_quartile"


@dataclass
class CohortSpecification:
    """Specification defining a demographic cohort and its economic parameters."""
    archetype: PersonaArchetype
    display_name: str
    income_min: float
    income_max: float
    target_spending_share: float  # Share of aggregate US PCE (e.g. 0.48 for upper)
    population_share: float       # Share of US household population
    typical_net_worth_range: str
    housing_tenure: str           # 'own_low_rate', 'own_mixed', 'rent'
    debt_profile: str
    interest_rate_sensitivity: str  # 'net_creditor' vs 'net_debtor'
    discretionary_elasticity: str
    base_probabilities: Dict[str, Dict[int, float]]  # Default probability for [1: fav, 2: neu, 3: unfav]


@dataclass
class PersonaProfile:
    """Individual synthetic respondent profile within a cohort."""
    persona_id: str
    archetype: PersonaArchetype
    age: int
    gender: str
    location: str
    occupation: str
    education: str
    household_income: float
    liquid_net_worth: float
    homeowner: bool
    mortgage_rate: Optional[float]
    credit_card_debt: float
    credit_score: int
    risk_tolerance: str  # 'high', 'moderate', 'low'
    inflation_sentiment: str
    survey_attitude: str  # 'engaged', 'skeptical', 'neutral'


# Baseline Question Response Distributions by Cohort
# (Options: 1=Favorable, 2=Neutral/Uncertain, 3=Unfavorable)

UPPER_BASE_PROBABILITIES: Dict[str, Dict[int, float]] = {
    "PAGO": {1: 0.72, 2: 0.18, 3: 0.10},   # 72% better off
    "PAGED": {1: 0.68, 2: 0.22, 3: 0.10},  # 68% expect better off
    "BUS12": {1: 0.65, 2: 0.20, 3: 0.15},  # 65% good business times
    "BUS5": {1: 0.60, 2: 0.25, 3: 0.15},   # 60% continuous good times
    "DUR": {1: 0.70, 2: 0.15, 3: 0.15},    # 70% good time to buy
}

MIDDLE_BASE_PROBABILITIES: Dict[str, Dict[int, float]] = {
    "PAGO": {1: 0.38, 2: 0.32, 3: 0.30},   # balanced/slightly optimistic
    "PAGED": {1: 0.42, 2: 0.35, 3: 0.23},
    "BUS12": {1: 0.40, 2: 0.30, 3: 0.30},
    "BUS5": {1: 0.36, 2: 0.34, 3: 0.30},
    "DUR": {1: 0.38, 2: 0.28, 3: 0.34},
}

LOWER_BASE_PROBABILITIES: Dict[str, Dict[int, float]] = {
    "PAGO": {1: 0.15, 2: 0.25, 3: 0.60},   # 60% worse off financially
    "PAGED": {1: 0.20, 2: 0.35, 3: 0.45},  # 45% expect worse off
    "BUS12": {1: 0.20, 2: 0.25, 3: 0.55},  # 55% bad times ahead
    "BUS5": {1: 0.15, 2: 0.30, 3: 0.55},   # 55% widespread unemployment
    "DUR": {1: 0.15, 2: 0.20, 3: 0.65},    # 65% bad time to buy durables
}


# Pre-defined cohort specifications
UPPER_QUARTILE_SPEC = CohortSpecification(
    archetype=PersonaArchetype.UPPER_QUARTILE,
    display_name="Affluent Upper-Quartile (Top 20-25%)",
    income_min=150_000,
    income_max=450_000,
    target_spending_share=0.48,  # Drives 48% of aggregate US PCE
    population_share=0.20,
    typical_net_worth_range="$800,000 - $5,000,000+",
    housing_tenure="own_low_rate",
    debt_profile="Minimal revolving debt, 780+ FICO",
    interest_rate_sensitivity="net_creditor",  # Enjoys 5%+ cash yields & asset inflation
    discretionary_elasticity="inelastic",
    base_probabilities=UPPER_BASE_PROBABILITIES,
)

MIDDLE_INCOME_SPEC = CohortSpecification(
    archetype=PersonaArchetype.MIDDLE_INCOME,
    display_name="Middle-Income Baseline (40th-70th Percentile)",
    income_min=65_000,
    income_max=120_000,
    target_spending_share=0.36,  # Drives 36% of aggregate US PCE
    population_share=0.45,
    typical_net_worth_range="$60,000 - $300,000",
    housing_tenure="own_mixed",
    debt_profile="Manageable mortgage, moderate credit card debt",
    interest_rate_sensitivity="neutral",
    discretionary_elasticity="selective_moderation",
    base_probabilities=MIDDLE_BASE_PROBABILITIES,
)

LOWER_QUARTILE_SPEC = CohortSpecification(
    archetype=PersonaArchetype.LOWER_QUARTILE,
    display_name="Lower/Median Vulnerable (Bottom 35-40%)",
    income_min=24_000,
    income_max=58_000,
    target_spending_share=0.16,  # Drives 16% of aggregate US PCE
    population_share=0.35,
    typical_net_worth_range="<$35,000",
    housing_tenure="rent",
    debt_profile="High credit card debt (>22% APR), auto loans, living paycheck to paycheck",
    interest_rate_sensitivity="net_debtor",  # Crushed by rate hikes & inflation
    discretionary_elasticity="highly_elastic",
    base_probabilities=LOWER_BASE_PROBABILITIES,
)

COHORT_REGISTRY: Dict[PersonaArchetype, CohortSpecification] = {
    PersonaArchetype.UPPER_QUARTILE: UPPER_QUARTILE_SPEC,
    PersonaArchetype.MIDDLE_INCOME: MIDDLE_INCOME_SPEC,
    PersonaArchetype.LOWER_QUARTILE: LOWER_QUARTILE_SPEC,
}


def generate_synthetic_persona(
    archetype: PersonaArchetype,
    persona_id: str,
    random_state: Optional[np.random.RandomState] = None
) -> PersonaProfile:
    """
    Generate an individual realistic synthetic persona profile within a specified archetype.
    """
    rng = random_state if random_state is not None else np.random.RandomState()
    spec = COHORT_REGISTRY[archetype]

    genders = ["Female", "Male", "Non-Binary"]
    gender = rng.choice(genders, p=[0.50, 0.48, 0.02])

    if archetype == PersonaArchetype.UPPER_QUARTILE:
        age = int(rng.randint(32, 66))
        locations = [
            "Scarsdale, NY", "San Francisco, CA", "Greenwich, CT", "Seattle, WA",
            "Austin, TX", "Naples, FL", "Boston, MA", "Denver, CO", "Chicago, IL"
        ]
        occupations = [
            "Senior Software Architect", "Corporate Attorney", "Private Equity Director",
            "Specialist Physician", "VP of Product", "Management Consultant", "Tech Entrepreneur"
        ]
        educations = ["Master's Degree", "Doctorate / Professional (JD, MD)", "Bachelor's Degree"]
        location = rng.choice(locations)
        occupation = rng.choice(occupations)
        education = rng.choice(educations, p=[0.45, 0.35, 0.20])
        income = float(rng.uniform(160_000, 480_000))
        net_worth = float(rng.uniform(900_000, 4_500_000))
        homeowner = rng.choice([True, False], p=[0.88, 0.12])
        mortgage_rate = float(rng.uniform(2.625, 3.5)) if homeowner else None
        credit_card_debt = float(rng.uniform(0, 3_500))  # Paid monthly
        credit_score = int(rng.randint(770, 835))
        risk_tolerance = rng.choice(["high", "moderate"], p=[0.60, 0.40])
        inflation_sentiment = "Asset values and dividend yields offset price increases; travel unconstrained."
        survey_attitude = rng.choice(["engaged", "confident"], p=[0.75, 0.25])

    elif archetype == PersonaArchetype.MIDDLE_INCOME:
        age = int(rng.randint(26, 62))
        locations = [
            "Columbus, OH", "Phoenix, AZ", "Charlotte, NC", "Indianapolis, IN",
            "Salt Lake City, UT", "Tampa, FL", "Minneapolis, MN", "Kansas City, MO"
        ]
        occupations = [
            "Registered Nurse", "High School Teacher", "Operations Manager",
            "Civil Engineer", "Accountant", "Logistics Supervisor", "Sales Representative"
        ]
        educations = ["Bachelor's Degree", "Associate Degree", "Some College"]
        location = rng.choice(locations)
        occupation = rng.choice(occupations)
        education = rng.choice(educations, p=[0.55, 0.25, 0.20])
        income = float(rng.uniform(65_000, 115_000))
        net_worth = float(rng.uniform(50_000, 260_000))
        homeowner = rng.choice([True, False], p=[0.65, 0.35])
        mortgage_rate = float(rng.uniform(3.75, 6.25)) if homeowner else None
        credit_card_debt = float(rng.uniform(2_500, 9_000))
        credit_score = int(rng.randint(680, 755))
        risk_tolerance = rng.choice(["moderate", "low"], p=[0.70, 0.30])
        inflation_sentiment = "Careful with discretionary spending; choosing store brands and looking for deals."
        survey_attitude = "engaged"

    else:  # LOWER_QUARTILE
        age = int(rng.randint(22, 60))
        locations = [
            "Cleveland, OH", "Memphis, TN", "Detroit, MI", "Bakersfield, CA",
            "San Antonio, TX", "Jackson, MS", "Macon, GA", "Scranton, PA"
        ]
        occupations = [
            "Retail Associate", "Warehouse Material Handler", "Food Service Worker",
            "Home Health Aide", "Customer Support Rep", "Rideshare Driver", "Security Guard"
        ]
        educations = ["High School Diploma / GED", "Some College", "Vocational Training"]
        location = rng.choice(locations)
        occupation = rng.choice(occupations)
        education = rng.choice(educations, p=[0.60, 0.30, 0.10])
        income = float(rng.uniform(26_000, 54_000))
        net_worth = float(rng.uniform(-8_000, 25_000))
        homeowner = rng.choice([True, False], p=[0.22, 0.78])
        mortgage_rate = float(rng.uniform(6.5, 7.8)) if homeowner else None
        credit_card_debt = float(rng.uniform(5_000, 14_000))  # Revolving @ 24%
        credit_score = int(rng.randint(560, 660))
        risk_tolerance = "low"
        inflation_sentiment = "Under severe stress from food and rent; skipping large purchases and maxing cards."
        survey_attitude = rng.choice(["skeptical", "frustrated", "stressed"], p=[0.40, 0.40, 0.20])

    return PersonaProfile(
        persona_id=persona_id,
        archetype=archetype,
        age=age,
        gender=gender,
        location=location,
        occupation=occupation,
        education=education,
        household_income=round(income, 2),
        liquid_net_worth=round(net_worth, 2),
        homeowner=homeowner,
        mortgage_rate=round(mortgage_rate, 3) if mortgage_rate else None,
        credit_card_debt=round(credit_card_debt, 2),
        credit_score=credit_score,
        risk_tolerance=risk_tolerance,
        inflation_sentiment=inflation_sentiment,
        survey_attitude=survey_attitude,
    )
