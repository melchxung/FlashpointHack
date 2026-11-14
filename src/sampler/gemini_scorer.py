"""
Gemini Scorer & MCP Agent Interface for Flashpoint AI.

Enables Google Gemini (gemini-3.5-flash-lite by default) to:
1. Extract structured behavioral economics metrics from conversational retail interviews:
   - stated_macro_sentiment (-1.0 to 1.0)
   - revealed_spend_conviction (-1.0 to 1.0)
   - trade_down_risk (0.0 to 1.0)
   - vibecession_divergence (revealed - stated)
2. Act as the intelligent autonomous agent driving Flashpoint AI's MCP endpoint
   (https://api.flashpoint.ai/mcp) via standard tool-calling schemas.

Scoring is live only: missing credentials or any API/parse failure raises.
"""

import os
import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

# Auto-load required keys from .env
from src.utils.dotenv import load_dotenv_keys
load_dotenv_keys(keys=["GEMINI_API_KEY", "GOOGLE_API_KEY"])


@dataclass
class GeminiExtractionResult:
    respondent_id: str
    retailer: str
    cohort: str
    stated_macro_sentiment: float
    revealed_spend_conviction: float
    trade_down_risk: float
    vibecession_divergence: float
    avg_basket_size: float
    trips_per_month: float
    renewal_intent: str
    stated_quote: str
    revealed_quote: str
    behavioral_synthesis: str
    scoring_engine: str  # Gemini model name that produced the scores


class GeminiScorer:
    """
    Google Gemini behavioral scoring engine and MCP Tool Agent.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.5-flash-lite",
    ):
        if api_key is None:
            api_key = (
                os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GEMINI_API_Key")
                or os.environ.get("GOOGLE_API_KEY")
            )
        if not api_key or not api_key.strip():
            raise RuntimeError("GEMINI_API_KEY is not set; Gemini scoring requires a live key.")

        import google.generativeai as genai
        genai.configure(api_key=api_key.strip())
        self.model_name = model_name
        self.client = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={"response_mime_type": "application/json"},
        )
        logger.info(f"Initialized Google Gemini client with model {self.model_name}")

    def score_transcript(self, transcript_dict: Dict[str, Any], idx: int = 0) -> GeminiExtractionResult:
        """
        Score a single interview transcript with live Gemini inference.
        Raises if the API call or JSON parsing fails.
        """
        transcript_text = transcript_dict.get("transcript_text") or "\n".join(
            f"Q: {turn['question']}\nA: {turn['answer']}"
            for turn in transcript_dict.get("qa_turns", [])
        )
        if not transcript_text.strip():
            raise ValueError(f"No transcript text to score for {transcript_dict.get('respondent_id')}")

        prompt = f"""
You are a Senior Quantitative Behavioral Economist. Analyze this verbatim shopping interview transcript:
Respondent ID: {transcript_dict.get('respondent_id')}
Retailer: {transcript_dict.get('retailer')}
Cohort: {transcript_dict.get('cohort')}
Transcript:
{transcript_text}

Output a JSON object with:
{{
  "stated_macro_sentiment": float between -1.0 (deep gloom) and 1.0 (euphoric),
  "revealed_spend_conviction": float between -1.0 (retrenchment) and 1.0 (unconstrained spend),
  "trade_down_risk": float between 0.0 and 1.0,
  "stated_quote": "verbatim quote regarding inflation/economy",
  "revealed_quote": "verbatim quote regarding basket spend/renewal",
  "behavioral_synthesis": "one-sentence summary of the behavioral divergence"
}}
"""
        response = self.client.generate_content(prompt)
        data = json.loads(response.text)

        stated = max(-1.0, min(1.0, float(data["stated_macro_sentiment"])))
        revealed = max(-1.0, min(1.0, float(data["revealed_spend_conviction"])))
        trade_down = max(0.0, min(1.0, float(data["trade_down_risk"])))
        divergence = round(revealed - stated, 3)

        return GeminiExtractionResult(
            respondent_id=transcript_dict.get("respondent_id", f"resp_{idx}"),
            retailer=transcript_dict.get("retailer", "COST"),
            cohort=transcript_dict.get("cohort", "high_income_costco"),
            stated_macro_sentiment=stated,
            revealed_spend_conviction=revealed,
            trade_down_risk=trade_down,
            vibecession_divergence=divergence,
            avg_basket_size=transcript_dict.get("avg_basket_size", 350.0),
            trips_per_month=transcript_dict.get("trips_per_month", 2.5),
            renewal_intent=transcript_dict.get("renewal_intent", "yes"),
            stated_quote=data.get("stated_quote", ""),
            revealed_quote=data.get("revealed_quote", ""),
            behavioral_synthesis=data.get("behavioral_synthesis", ""),
            scoring_engine=self.model_name,
        )
