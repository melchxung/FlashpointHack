"""
Flashpoint MCP client.

Calls Flashpoint AI's Model Context Protocol server at https://api.flashpoint.ai/mcp
using JSON-RPC 2.0 over HTTP POST. Paid tools answer HTTP 402 with x402 payment
requirements; the client pays them with an EIP-3009 USDC authorization signed by the
agent wallet via the x402 SDK, subject to per-call and per-session spend caps. Every failure raises;
there are no simulated responses.
"""

import base64
import json
import os
import threading
import time
from typing import Dict, Optional, Any

import requests
from eth_account import Account
from eth_account.messages import encode_defunct
from x402.http.utils import encode_payment_signature_header
from x402.mechanisms.evm.exact.client import ExactEvmScheme
from x402.mechanisms.evm.signers import EthAccountSigner
from x402.schemas import PaymentPayload, PaymentRequirements

# Auto-load required keys from .env
from src.utils.dotenv import load_dotenv_keys
load_dotenv_keys(keys=["FLASHPOINT_BEARER_TOKEN", "FLASHPOINT_PRIVATE_KEY"])

CHALLENGE_URL = "https://api.flashpoint.ai/auth/wallet/challenge"
TOKEN_URL = "https://api.flashpoint.ai/auth/wallet/token"

BASE_NETWORK = "eip155:8453"
USDC_BASE = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
USDC_DECIMALS = 6

# Spend guards (USDC). A single call above the per-call cap, or a session above the
# session cap, raises instead of paying.
MAX_PAYMENT_PER_CALL_USDC = 0.60
MAX_SPEND_PER_SESSION_USDC = 5.00

class PaymentRequired(Exception):
    """Raised on HTTP 402 so the caller can pay and retry."""

    def __init__(self, requirements: Dict[str, Any]):
        super().__init__("Flashpoint MCP requires x402 payment")
        self.requirements = requirements


class GeminiFlashpointMCPClient:
    """
    Client for Flashpoint AI's MCP endpoint (live calls only).
    """

    MCP_ENDPOINT = "https://api.flashpoint.ai/mcp"
    TIMEOUT = 600  # panel and chat tools can run for minutes

    def __init__(self, flashpoint_token: Optional[str] = None, private_key: Optional[str] = None,
                 auto_pay: bool = True, session_cap_usdc: float = MAX_SPEND_PER_SESSION_USDC):
        self.flashpoint_token = flashpoint_token or os.environ.get("FLASHPOINT_BEARER_TOKEN")
        self.private_key = private_key or os.environ.get("FLASHPOINT_PRIVATE_KEY")
        self.auto_pay = auto_pay
        self.session_cap_usdc = min(session_cap_usdc, MAX_SPEND_PER_SESSION_USDC)
        self.session_id: Optional[str] = None
        self.spent_usdc = 0.0
        self._spend_lock = threading.Lock()
        self._request_id = 0
        self._ensure_token()

    def _token_expired(self) -> bool:
        """True if there is no wallet JWT or it expires within a minute."""
        if not self.flashpoint_token:
            return True
        try:
            segment = self.flashpoint_token.split(".")[1]
            claims = json.loads(base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4)))
        except (IndexError, ValueError):
            return True
        return claims.get("exp", 0) < time.time() + 60

    def _ensure_token(self) -> None:
        """Wallet JWTs last one hour: sign a fresh EIP-191 challenge when the stored one is stale."""
        if not self._token_expired():
            return
        if not self.private_key:
            raise RuntimeError(
                "FLASHPOINT_BEARER_TOKEN is missing or expired and FLASHPOINT_PRIVATE_KEY is not set"
            )
        account = Account.from_key(self.private_key)
        try:
            challenge = requests.post(CHALLENGE_URL, json={"address": account.address}, timeout=self.TIMEOUT)
            if challenge.status_code != 200:
                raise RuntimeError(f"Flashpoint challenge failed (HTTP {challenge.status_code}): {challenge.text}")
            body = challenge.json()
            signature = account.sign_message(encode_defunct(text=body["message_to_sign"])).signature.hex()
            if not signature.startswith("0x"):
                signature = "0x" + signature
            token = requests.post(
                TOKEN_URL,
                json={"address": account.address, "nonce": body["nonce"], "signature": signature},
                timeout=self.TIMEOUT,
            )
        except requests.RequestException as e:
            raise RuntimeError(f"Flashpoint wallet authentication failed: {e}") from e
        if token.status_code != 200:
            raise RuntimeError(f"Flashpoint token exchange failed (HTTP {token.status_code}): {token.text}")
        self.flashpoint_token = token.json()["token"]
        self.session_id = None

    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.flashpoint_token}",
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": "Gemini-MCP-Client/1.0",
        }
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        if extra:
            headers.update(extra)
        return headers

    def _post(self, method: str, params: Optional[Dict[str, Any]] = None,
              extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        self._request_id += 1
        payload = {"jsonrpc": "2.0", "method": method, "id": self._request_id}
        if params is not None:
            payload["params"] = params

        try:
            resp = requests.post(self.MCP_ENDPOINT, json=payload,
                                 headers=self._headers(extra_headers), timeout=self.TIMEOUT)
        except requests.RequestException as e:
            raise RuntimeError(f"Flashpoint MCP connection failed ({method}): {e}") from e

        if resp.status_code == 402:
            raise PaymentRequired(self._payment_requirements(resp))
        if resp.status_code != 200:
            raise RuntimeError(f"Flashpoint MCP HTTP {resp.status_code} on {method}: {resp.text}")

        if method == "initialize" and resp.headers.get("mcp-session-id"):
            self.session_id = resp.headers["mcp-session-id"]

        body = self._parse_body(resp.text, resp.headers.get("Content-Type", ""))
        if "error" in body:
            raise RuntimeError(f"Flashpoint MCP JSON-RPC error on {method}: {json.dumps(body['error'])}")
        return body.get("result", {})

    @staticmethod
    def _payment_requirements(resp: requests.Response) -> Dict[str, Any]:
        """x402 requirements come in the JSON body, or base64 in the payment-required header."""
        try:
            body = resp.json()
            if body.get("accepts"):
                return body
        except ValueError:
            pass
        header = resp.headers.get("payment-required")
        if header:
            return json.loads(base64.b64decode(header))
        raise RuntimeError(f"Flashpoint MCP HTTP 402 without x402 requirements: {resp.text[:300]}")

    @staticmethod
    def _parse_body(text: str, content_type: str) -> Dict[str, Any]:
        """Parse a JSON body or a single-message text/event-stream body."""
        if "text/event-stream" in content_type or text.lstrip().startswith("event:"):
            data_lines = [line[len("data:"):].strip() for line in text.splitlines() if line.startswith("data:")]
            if not data_lines:
                raise RuntimeError(f"Flashpoint MCP returned an event stream with no data: {text[:200]}")
            return json.loads(data_lines[-1])
        return json.loads(text)

    def _ensure_session(self) -> None:
        self._ensure_token()
        if self.session_id is None:
            self._post(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "retail-k-alpha-agent", "version": "1.0"},
                },
            )

    def _payment_header(self, requirements: Dict[str, Any]) -> str:
        """Build the x402 v2 PAYMENT-SIGNATURE header: an EIP-3009 USDC authorization signed by the agent wallet."""
        if not self.private_key:
            raise RuntimeError("FLASHPOINT_PRIVATE_KEY is not set; cannot pay for paid Flashpoint tools")

        accepted = next(
            (a for a in requirements.get("accepts", [])
             if a.get("scheme") == "exact" and a.get("network") == BASE_NETWORK),
            None,
        )
        if accepted is None:
            raise RuntimeError(f"No exact/{BASE_NETWORK} payment option in requirements: {requirements}")
        if accepted["asset"].lower() != USDC_BASE.lower():
            raise RuntimeError(f"Refusing to pay in unexpected asset {accepted['asset']}")
        if accepted.get("extra", {}).get("assetTransferMethod") != "eip3009":
            raise RuntimeError("Refusing non-EIP-3009 payment method")

        amount_units = int(accepted["amount"])
        amount_usdc = amount_units / 10 ** USDC_DECIMALS
        if amount_usdc > MAX_PAYMENT_PER_CALL_USDC:
            raise RuntimeError(f"Per-call cap exceeded: call costs {amount_usdc} USDC, cap is {MAX_PAYMENT_PER_CALL_USDC}")
        # Reserve the spend under a lock so parallel calls cannot overrun the session cap.
        with self._spend_lock:
            if self.spent_usdc + amount_usdc > self.session_cap_usdc:
                raise RuntimeError(
                    f"Session cap exceeded: spent {self.spent_usdc:.2f} + {amount_usdc} > {self.session_cap_usdc} USDC"
                )
            self.spent_usdc += amount_usdc

        # Sign with the official x402 SDK so the payload matches what the facilitator verifies.
        requirement = PaymentRequirements.model_validate({
            "scheme": accepted["scheme"],
            "network": accepted["network"],
            "asset": accepted["asset"],
            "amount": accepted["amount"],
            "payTo": accepted["payTo"],
            "maxTimeoutSeconds": accepted.get("maxTimeoutSeconds", 600),
            "extra": accepted.get("extra", {}),
        })
        inner = ExactEvmScheme(EthAccountSigner(Account.from_key(self.private_key))).create_payment_payload(requirement)
        payload = PaymentPayload(
            x402_version=2,
            payload=inner,
            accepted=requirement,
            resource=requirements.get("resource"),
        )
        return encode_payment_signature_header(payload)

    def list_tools(self) -> list:
        """Free call: the tools Flashpoint MCP currently advertises."""
        self._ensure_session()
        return self._post("tools/list", {}).get("tools", [])

    def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a Flashpoint MCP tool and return its result. Pays x402 when the tool requires it.
        Raises on any failure.
        """
        self._ensure_session()
        params = {"name": tool_name, "arguments": arguments}
        try:
            return self._post("tools/call", params)
        except PaymentRequired as required:
            if not self.auto_pay:
                raise RuntimeError(f"Flashpoint tool '{tool_name}' requires x402 payment (auto_pay is off)")
            header = self._payment_header(required.requirements)
            try:
                return self._post("tools/call", params, extra_headers={"PAYMENT-SIGNATURE": header})
            except PaymentRequired as rejected:
                raise RuntimeError(
                    f"Flashpoint rejected the x402 payment for '{tool_name}': {rejected.requirements.get('error')}"
                ) from rejected
