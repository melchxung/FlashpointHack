"""
Retail Behavioral Interview Probe & Conversational Transcript Generator.

Simulates in-depth conversational shopping interviews across three key retail cohorts:
1. High-Income Costco Members (HHI >$120k, Asset-Backed):
   - Stated: Macroeconomic complaints ("inflation is terrible", "government spending", "gas prices").
   - Revealed: $300-$450 cart sizes, premium steaks, organic produce, Kirkland wine, 98% renewal intent, frequent weekend runs.
   - The Vibecession archetype: "Public gloomy talk vs. Actual overflowing shopping carts".
2. Middle-Income Target Shoppers (HHI $60k-$120k):
   - Stated: Moderate economic anxiety, watchful of inflation.
   - Revealed: $90-$180 cart sizes, trading down from national brands to Good & Gather, trimming discretionary home decor/apparel, maintaining Target Circle.
3. Lower-Income Walmart Shoppers (HHI <$60k):
   - Stated: Acute distress, feeling pinched by grocery inflation and utilities.
   - Revealed: $35-$90 cart sizes, strict shopping lists, switching to Great Value, deferred durables, living paycheck to paycheck.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class RespondentProfile:
    respondent_id: str
    retailer: str  # 'COST', 'TGT', 'WMT'
    cohort: str    # 'high_income_costco', 'middle_income_target', 'lower_income_walmart'
    household_income: float
    age: int
    gender: str
    location: str
    occupation: str
    membership_type: str
    avg_basket_size: float
    trips_per_month: float
    renewal_intent: str  # 'definitely_renew', 'likely_renew', 'uncertain', 'cancel'


@dataclass
class ConversationalTranscript:
    respondent_id: str
    retailer: str
    cohort: str
    profile: Dict[str, Any]
    transcript_text: str
    qa_turns: List[Dict[str, str]]


def generate_retail_respondents(
    n_per_cohort: int = 30,
    random_state: Optional[np.random.RandomState] = None
) -> List[ConversationalTranscript]:
    """
    Generate conversational behavioral interviews across 3 cohorts (default 90 total).
    """
    rng = random_state if random_state is not None else np.random.RandomState(42)
    transcripts: List[ConversationalTranscript] = []

    # 1. High-Income Costco Cohort (HHI >$120k)
    costco_locations = ["Scottsdale, AZ", "Bellevue, WA", "Naperville, IL", "Westchester, NY", "Plano, TX", "Alpharetta, GA"]
    costco_occupations = ["Senior VP Tech", "Orthopedic Surgeon", "Corporate Attorney", "Management Consultant", "Commercial Real Estate Broker", "Engineering Director"]
    costco_genders = ["Female", "Male"]

    for i in range(n_per_cohort):
        resp_id = f"COST_{i+1:03d}"
        hhi = float(rng.uniform(135_000, 360_000))
        age = int(rng.randint(34, 63))
        gender = rng.choice(costco_genders)
        loc = rng.choice(costco_locations)
        occ = rng.choice(costco_occupations)
        basket = float(rng.uniform(265.0, 480.0))
        trips = float(rng.uniform(2.5, 5.5))
        renewal = "definitely_renew"

        prof = RespondentProfile(
            respondent_id=resp_id,
            retailer="COST",
            cohort="high_income_costco",
            household_income=round(hhi, 2),
            age=age,
            gender=gender,
            location=loc,
            occupation=occ,
            membership_type="Costco Executive ($130/yr)",
            avg_basket_size=round(basket, 2),
            trips_per_month=round(trips, 1),
            renewal_intent=renewal,
        )

        # Conversational script with Vibecession divergence
        qa = [
            {
                "question": "How are you feeling about the national economy, inflation, and your household outlook right now?",
                "answer": (
                    f"Honestly, the country feels like it's in terrible shape. Inflation is out of control, "
                    f"gas and utility bills are frustrating, and with all the government spending and high interest rates, "
                    f"I'm pretty gloomy about where the broader economy is heading in the next 12 to 24 months."
                ),
            },
            {
                "question": "What was your last register receipt total here at Costco, and how has your overall spending changed?",
                "answer": (
                    f"My receipt today was ${basket:.2f}. We bought USDA Prime ribeye steaks, organic olive oil, two cases of "
                    f"Kirkland Prosecco, fresh berries, and a Dyson vacuum on promotion. Honestly, our actual monthly spending "
                    f"hasn't slowed down at all—if anything, we're spending 15% more this year because we love the quality and value."
                ),
            },
            {
                "question": "Are you cutting back on discretionary spending or trading down to cheaper store alternatives?",
                "answer": (
                    "Not really cutting back. We buy Kirkland Signature because the quality is just as good or better than name brands, "
                    "like the Kirkland French vodka and Italian Parmigiano. We just booked our ski trip to Aspen through Costco Travel. "
                    "We're definitely not pinching pennies on lifestyle."
                ),
            },
            {
                "question": "Do you plan to renew your Costco Executive membership next year, and how often do you shop here?",
                "answer": (
                    f"Oh, 100% renewal without question. The 2% cash back check basically pays for the membership twice over. "
                    f"We come here about {trips:.1f} times a month, usually on Saturday mornings. It's an indispensable routine for our family."
                ),
            },
        ]

        full_text = "\n\n".join([f"Interviewer: {t['question']}\nRespondent: {t['answer']}" for t in qa])

        transcripts.append(ConversationalTranscript(
            respondent_id=resp_id,
            retailer="COST",
            cohort="high_income_costco",
            profile=asdict(prof),
            transcript_text=full_text,
            qa_turns=qa,
        ))

    # 2. Middle-Income Target Cohort (HHI $60k-$120k)
    target_locations = ["Columbus, OH", "Charlotte, NC", "Overland Park, KS", "Tampa, FL", "Denver, CO", "Minneapolis, MN"]
    target_occupations = ["Registered Nurse", "Middle School Principal", "Accountant", "Logistics Coordinator", "Marketing Specialist", "Operations Lead"]
    
    for i in range(n_per_cohort):
        resp_id = f"TGT_{i+1:03d}"
        hhi = float(rng.uniform(65_000, 115_000))
        age = int(rng.randint(28, 55))
        gender = rng.choice(costco_genders, p=[0.65, 0.35])
        loc = rng.choice(target_locations)
        occ = rng.choice(target_occupations)
        basket = float(rng.uniform(85.0, 175.0))
        trips = float(rng.uniform(1.8, 4.0))
        renewal = rng.choice(["definitely_renew", "likely_renew", "uncertain"], p=[0.45, 0.45, 0.10])

        prof = RespondentProfile(
            respondent_id=resp_id,
            retailer="TGT",
            cohort="middle_income_target",
            household_income=round(hhi, 2),
            age=age,
            gender=gender,
            location=loc,
            occupation=occ,
            membership_type="Target Circle 360 / RedCard Holder",
            avg_basket_size=round(basket, 2),
            trips_per_month=round(trips, 1),
            renewal_intent=renewal,
        )

        qa = [
            {
                "question": "How are you feeling about the national economy, inflation, and your household outlook right now?",
                "answer": (
                    "It's decent but definitely tighter than two years ago. We are watching our family budget closely. "
                    "Prices haven't really come down, they've just stopped rising as fast. We feel steady with our jobs, but cautious."
                ),
            },
            {
                "question": "What was your last register receipt total here at Target, and how has your overall spending changed?",
                "answer": (
                    f"Today came out to ${basket:.2f}. We bought household essentials, paper towels, school supplies, "
                    f"and some snacks. I'd say our total monthly shopping is fairly flat. We stick more strictly to our list "
                    f"instead of impulsively grabbing extra apparel or home seasonal decor like we used to."
                ),
            },
            {
                "question": "Are you cutting back on discretionary spending or trading down to cheaper store alternatives?",
                "answer": (
                    "Yes, definitely on the pantry side. We switched from Cheerios and Kraft to Target's Good & Gather "
                    "and Up & Up brands for essentials, which saves maybe 20-25%. We are holding off on replacing patio furniture "
                    "and buying fewer throw pillows and decorative items in the dollar spot."
                ),
            },
            {
                "question": "Do you plan to renew your membership or stay a regular Target shopper?",
                "answer": (
                    f"We love the Target app and 5% RedCard discount. We'll keep shopping here about {trips:.1f} times a month, "
                    f"mostly using drive-up pickup so we avoid impulse buys inside the aisles."
                ),
            },
        ]

        full_text = "\n\n".join([f"Interviewer: {t['question']}\nRespondent: {t['answer']}" for t in qa])

        transcripts.append(ConversationalTranscript(
            respondent_id=resp_id,
            retailer="TGT",
            cohort="middle_income_target",
            profile=asdict(prof),
            transcript_text=full_text,
            qa_turns=qa,
        ))

    # 3. Lower-Income Walmart Cohort (HHI <$60k)
    walmart_locations = ["Memphis, TN", "Bakersfield, CA", "Cleveland, OH", "Macon, GA", "Scranton, PA", "San Antonio, TX"]
    walmart_occupations = ["Warehouse Associate", "Food Service Crew", "Home Care Worker", "Delivery Driver", "Retail Cashier", "Hospital Orderly"]

    for i in range(n_per_cohort):
        resp_id = f"WMT_{i+1:03d}"
        hhi = float(rng.uniform(28_000, 56_000))
        age = int(rng.randint(23, 62))
        gender = rng.choice(costco_genders, p=[0.55, 0.45])
        loc = rng.choice(walmart_locations)
        occ = rng.choice(walmart_occupations)
        basket = float(rng.uniform(38.0, 92.0))
        trips = float(rng.uniform(1.2, 3.2))
        renewal = rng.choice(["likely_renew", "uncertain", "cancel"], p=[0.30, 0.40, 0.30])

        prof = RespondentProfile(
            respondent_id=resp_id,
            retailer="WMT",
            cohort="lower_income_walmart",
            household_income=round(hhi, 2),
            age=age,
            gender=gender,
            location=loc,
            occupation=occ,
            membership_type="Walmart+ / Regular In-Store Shopper",
            avg_basket_size=round(basket, 2),
            trips_per_month=round(trips, 1),
            renewal_intent=renewal,
        )

        qa = [
            {
                "question": "How are you feeling about the national economy, inflation, and your household outlook right now?",
                "answer": (
                    "It's really hard right now. Everything from rent to eggs and bread is way too high. "
                    "Between credit card bills at 24% interest and car insurance going up, we are stretched to the absolute limit. "
                    "I don't see things getting better anytime soon."
                ),
            },
            {
                "question": "What was your last register receipt total here at Walmart, and how has your overall spending changed?",
                "answer": (
                    f"My basket today was ${basket:.2f}. We bought eggs, generic bread, store brand canned beans, and milk. "
                    f"We have cut our total spending down to the bare bone. By the end of the month we have to use SNAP or "
                    f"put food on credit cards."
                ),
            },
            {
                "question": "Are you cutting back on discretionary spending or trading down to cheaper store alternatives?",
                "answer": (
                    "We buy 100% Great Value brand for everything now. We stopped buying beef completely, just chicken thighs "
                    "or canned tuna. Any kind of discretionary shopping—clothes, new electronics, dining out—is completely out of the question."
                ),
            },
            {
                "question": "Do you plan to renew any store subscriptions like Walmart+, and how often do you shop?",
                "answer": (
                    f"I probably won't keep Walmart+ because every extra dollar counts right now. We shop about {trips:.1f} times a month, "
                    f"usually right after paycheck hits on Friday. We can't afford to buy in bulk."
                ),
            },
        ]

        full_text = "\n\n".join([f"Interviewer: {t['question']}\nRespondent: {t['answer']}" for t in qa])

        transcripts.append(ConversationalTranscript(
            respondent_id=resp_id,
            retailer="WMT",
            cohort="lower_income_walmart",
            profile=asdict(prof),
            transcript_text=full_text,
            qa_turns=qa,
        ))

    return transcripts
