"""
Flashpoint AI & Touchpoint API Modular Client.

Handles communication with Flashpoint's REST API and MCP endpoints:
- Automatic fallback to high-fidelity SyntheticPersonaSimulator when API keys are absent
  or in offline/test environments.
- Exponential backoff and retry handling for rate limits (HTTP 429) and network timeouts.
- Idempotency support for synthetic panel dispatch.
"""

import os
import time
import uuid
import logging
from typing import Dict, List, Optional, Any
import requests

from src.sampler.personas import PersonaArchetype, COHORT_REGISTRY
from src.sampler.simulator import SyntheticPersonaSimulator

logger = logging.getLogger(__name__)


class FlashpointClient:
    """
    Client for Touchpoint AI / Flashpoint AI API with graceful synthetic simulation fallback.
    """

    def __init__(
        self,
        api_token: Optional[str] = None,
        base_url: str = "https://api.flashpoint.ai/api/v1",
        mcp_url: str = "https://api.flashpoint.ai/mcp",
        timeout: float = 30.0,
        max_retries: int = 3,
        fallback_to_simulator: bool = True,
        simulator_seed: int = 42,
    ):
        self.api_token = (
            api_token
            or os.environ.get("FLASHPOINT_API_TOKEN")
            or os.environ.get("TOUCHPOINT_API_KEY")
        )
        self.base_url = base_url.rstrip("/")
        self.mcp_url = mcp_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.fallback_to_simulator = fallback_to_simulator
        self.simulator = SyntheticPersonaSimulator(seed=simulator_seed)

    @property
    def has_credentials(self) -> bool:
        """Check if an API token or key is configured."""
        return bool(self.api_token and self.api_token.strip())

    def _get_headers(self, idempotency_key: Optional[str] = None) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token.strip()}"
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return headers

    def create_and_run_panel(
        self,
        archetype: PersonaArchetype,
        num_respondents: int = 50,
        date_str: str = "2024-01",
        survey_id: str = "srv_michigan_5q_k_divergence",
        seed_offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Deploy and execute a synthetic panel against the Michigan survey battery.
        If credentials are absent or remote calls fail, automatically falls back
        to the local high-fidelity synthetic simulator.
        """
        spec = COHORT_REGISTRY[archetype]

        if not self.has_credentials:
            if not self.fallback_to_simulator:
                raise ValueError("No Flashpoint API token configured and fallback_to_simulator is False.")
            logger.info(
                f"[FlashpointClient] No live API token found. Using high-fidelity synthetic simulator "
                f"for cohort '{archetype.value}' ({num_respondents} respondents, date={date_str})."
            )
            return self.simulator.run_cohort_panel(
                archetype=archetype,
                num_respondents=num_respondents,
                date_str=date_str,
                seed_offset=seed_offset,
            )

        # Attempt live API call via MCP tool interface
        payload = {
            "jsonrpc": "2.0",
            "id": f"req_{uuid.uuid4().hex[:12]}",
            "method": "tools/call",
            "params": {
                "name": "panel",
                "arguments": {
                    "action": "create_and_run",
                    "name": f"{spec.display_name} - {date_str}",
                    "num_respondents": num_respondents,
                    "research_objective": (
                        f"Field 5-question Michigan sentiment battery on {spec.display_name} consumers."
                    ),
                    "context": (
                        f"Target cohort: income {spec.typical_net_worth_range}, "
                        f"housing tenure: {spec.housing_tenure}, "
                        f"interest sensitivity: {spec.interest_rate_sensitivity}."
                    ),
                    "audience": {
                        "demographics": {
                            "household_income": f"${spec.income_min:,.0f} - ${spec.income_max:,.0f}",
                            "housing": spec.housing_tenure,
                        }
                    },
                    "survey_id": survey_id,
                },
            },
        }

        idempotency_key = str(uuid.uuid4())
        headers = self._get_headers(idempotency_key)

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.mcp_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    data = response.json()
                    result = data.get("result", {})
                    if result.get("status") == "success":
                        result["archetype"] = archetype.value
                        result["date"] = date_str
                        return result
                    else:
                        logger.warning(f"Live API call returned unsuccessful result: {data}")

                elif response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 2 ** attempt))
                    logger.warning(f"Rate limited (429). Retrying after {retry_after}s...")
                    time.sleep(retry_after)
                    continue

                else:
                    logger.warning(
                        f"Live API request failed with status {response.status_code}: {response.text}"
                    )

            except Exception as e:
                logger.warning(f"Network error on attempt {attempt}/{self.max_retries}: {e}")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        if self.fallback_to_simulator:
            logger.info(
                f"[FlashpointClient] Live call failed after {self.max_retries} attempts. "
                f"Falling back to high-fidelity simulator for cohort '{archetype.value}'."
            )
            return self.simulator.run_cohort_panel(
                archetype=archetype,
                num_respondents=num_respondents,
                date_str=date_str,
                seed_offset=seed_offset,
            )
        else:
            raise RuntimeError(f"Failed to execute panel on live Flashpoint API after {self.max_retries} attempts.")
