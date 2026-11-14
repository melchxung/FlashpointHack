"""
Gemini Flashpoint MCP Client.

Demonstrates how Google Gemini connects to Flashpoint AI's Model Context Protocol (MCP)
server at https://api.flashpoint.ai/mcp using JSON-RPC 2.0 over HTTP POST.
"""

import os
import json
import requests
from typing import Dict, List, Optional, Any
from pathlib import Path

# Auto-load required keys from .env
from src.utils.dotenv import load_dotenv_keys
load_dotenv_keys(keys=["GEMINI_API_KEY", "GOOGLE_API_KEY", "FLASHPOINT_BEARER_TOKEN"])


class GeminiFlashpointMCPClient:
    """
    Client connecting Google Gemini function calling to Flashpoint AI's MCP endpoint.
    """

    MCP_ENDPOINT = "https://api.flashpoint.ai/mcp"

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        flashpoint_token: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
    ):
        self.gemini_key = gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.flashpoint_token = flashpoint_token or os.environ.get("FLASHPOINT_BEARER_TOKEN")
        self.model_name = model_name

    def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a JSON-RPC 2.0 tool execution against the Flashpoint MCP endpoint.
        """
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
            "id": 1,
        }

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Gemini-MCP-Client/1.0",
        }
        if self.flashpoint_token:
            headers["Authorization"] = f"Bearer {self.flashpoint_token}"

        try:
            resp = requests.post(self.MCP_ENDPOINT, json=payload, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            else:
                return {
                    "error": f"MCP HTTP {resp.status_code}",
                    "details": resp.text,
                    "simulated_response": self._simulate_mcp_tool_response(tool_name, arguments),
                }
        except Exception as e:
            return {
                "error": f"Connection failed: {e}",
                "simulated_response": self._simulate_mcp_tool_response(tool_name, arguments),
            }

    def _simulate_mcp_tool_response(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback simulation for offline or sandbox environments."""
        if tool_name == "panel":
            return {
                "cohort_id": "coh_affluent_costco_mcp_01",
                "matched_count": arguments.get("size", 30),
                "hhi_min": arguments.get("filters", {}).get("hhi_min", 120000),
                "status": "active",
            }
        elif tool_name == "tool_survey":
            return {
                "survey_id": "surv_retail_alpha_01",
                "respondents_sampled": 30,
                "mean_revealed_spend_score": 0.813,
                "mean_stated_macro_gloom": -0.540,
                "vibecession_spread": 1.353,
            }
        return {"result": "success", "tool": tool_name, "args": arguments}
