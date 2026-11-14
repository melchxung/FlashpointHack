"""
Flashpoint / Touchpoint AI Persona Sampling & K-Shaped Sentiment Engine.

This module provides tools for surveying stratified synthetic consumer panels
using the University of Michigan 5-question sentiment battery (ICS / ICC / ICE).
It models the divergent economic realities of affluent upper-quartile vs.
mass/lower-quartile consumers, computing both raw and spending-weighted sentiment indices.
"""

from src.sampler.survey import (
    MICHIGAN_QUESTIONS,
    MichiganQuestion,
    QuestionResponse,
    SurveyBattery,
)
from src.sampler.personas import (
    PersonaArchetype,
    PersonaProfile,
    CohortSpecification,
    UPPER_QUARTILE_SPEC,
    MIDDLE_INCOME_SPEC,
    LOWER_QUARTILE_SPEC,
)
from src.sampler.client import FlashpointClient
from src.sampler.simulator import SyntheticPersonaSimulator
from src.sampler.engine import SentimentSamplingEngine, SamplingResult

__all__ = [
    "MICHIGAN_QUESTIONS",
    "MichiganQuestion",
    "QuestionResponse",
    "SurveyBattery",
    "PersonaArchetype",
    "PersonaProfile",
    "CohortSpecification",
    "UPPER_QUARTILE_SPEC",
    "MIDDLE_INCOME_SPEC",
    "LOWER_QUARTILE_SPEC",
    "FlashpointClient",
    "SyntheticPersonaSimulator",
    "SentimentSamplingEngine",
    "SamplingResult",
]
