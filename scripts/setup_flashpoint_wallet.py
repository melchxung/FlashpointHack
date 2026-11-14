#!/usr/bin/env python
"""
Automated Wallet Setup & Authentication for Flashpoint AI (EIP-191 MCP Auth).

This script:
1. Connects an Ethereum / Base wallet (or generates a dedicated local agent wallet).
2. Requests a challenge nonce from Flashpoint AI (POST /auth/wallet/challenge).
3. Signs the message using EIP-191 personal_sign with the private key.
4. Exchanges the signature for an active 1-hour Bearer JWT (POST /auth/wallet/token).
5. Automatically saves the FLASHPOINT_BEARER_TOKEN and wallet credentials into .env.
6. Tests the live Flashpoint MCP connection (POST /mcp initialize & tools/list).
"""

import os
import sys
import json
import argparse
from pathlib import Path
import requests
from eth_account import Account
from eth_account.messages import encode_defunct

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

FLASHPOINT_CHALLENGE_URL = "https://api.flashpoint.ai/auth/wallet/challenge"
FLASHPOINT_TOKEN_URL = "https://api.flashpoint.ai/auth/wallet/token"
FLASHPOINT_MCP_URL = "https://api.flashpoint.ai/mcp"


def load_env_vars() -> dict:
    env = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip("'\"")
    return env


def save_env_vars(updates: dict):
    env = load_env_vars()
    env.update(updates)
    lines = [f"{k}={v}" for k, v in env.items()]
    ENV_PATH.write_text("\n".join(lines) + "\n")
    print(f" [✓] Updated .env with keys: {list(updates.keys())}")


def setup_wallet(private_key_hex: str = None) -> Account:
    if private_key_hex:
        if private_key_hex.startswith("0x"):
            private_key_hex = private_key_hex[2:]
        acc = Account.from_key(private_key_hex)
        print(f" [✓] Loaded wallet address: {acc.address}")
    else:
        # Check if already saved in .env
        env = load_env_vars()
        saved_key = env.get("FLASHPOINT_PRIVATE_KEY")
        if saved_key:
            acc = Account.from_key(saved_key)
            print(f" [✓] Loaded existing agent wallet from .env: {acc.address}")
        else:
            acc = Account.create()
            print(f" [★] Generated new autonomous agent wallet: {acc.address}")
            save_env_vars({
                "FLASHPOINT_WALLET_ADDRESS": acc.address,
                "FLASHPOINT_PRIVATE_KEY": acc.key.hex(),
            })
    return acc


def authenticate_flashpoint(acc: Account) -> dict:
    print("\n[1/3] Requesting challenge nonce from Flashpoint AI...")
    resp = requests.post(FLASHPOINT_CHALLENGE_URL, json={"address": acc.address}, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"Challenge request failed (HTTP {resp.status_code}): {resp.text}")
    chal = resp.json()
    nonce = chal["nonce"]
    msg_to_sign = chal["message_to_sign"]
    print(f"      Nonce received: {nonce[:12]}...")

    print("[2/3] Signing EIP-191 challenge with wallet private key...")
    encoded = encode_defunct(text=msg_to_sign)
    signed = acc.sign_message(encoded)
    sig_hex = signed.signature.hex()
    if not sig_hex.startswith("0x"):
        sig_hex = "0x" + sig_hex

    print("[3/3] Exchanging signature for 1-hour Flashpoint Bearer JWT...")
    payload = {
        "address": acc.address,
        "nonce": nonce,
        "signature": sig_hex,
    }
    t_resp = requests.post(FLASHPOINT_TOKEN_URL, json=payload, timeout=10)
    if t_resp.status_code != 200:
        raise RuntimeError(f"Token exchange failed (HTTP {t_resp.status_code}): {t_resp.text}")
    
    data = t_resp.json()
    token = data["token"]
    user_id = data.get("user_id", "")
    team_id = data.get("team_id", "")
    print(f" [✓] Authentication SUCCESS! Provisioned User ID: {user_id}")

    save_env_vars({
        "FLASHPOINT_BEARER_TOKEN": token,
        "FLASHPOINT_WALLET_ADDRESS": acc.address,
        "FLASHPOINT_USER_ID": user_id,
        "FLASHPOINT_TEAM_ID": team_id,
    })
    return data


def test_mcp_connection(token: str):
    print("\n[Testing Live Flashpoint MCP Session]")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    init_payload = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "retail-k-alpha-agent", "version": "1.0"},
        },
        "id": 1,
    }
    r = requests.post(FLASHPOINT_MCP_URL, json=init_payload, headers=headers, timeout=10)
    if r.status_code != 200:
        print(f" [!] MCP Initialize returned HTTP {r.status_code}: {r.text}")
        return

    session_id = r.headers.get("mcp-session-id")
    print(f" [✓] MCP Session Established! Session ID: {session_id}")

    # List tools
    headers["mcp-session-id"] = session_id
    r_tools = requests.post(
        FLASHPOINT_MCP_URL,
        json={"jsonrpc": "2.0", "method": "tools/list", "id": 2},
        headers=headers,
        timeout=10,
    )
    if r_tools.status_code == 200:
        data_lines = [ln[len("data:"):].strip() for ln in r_tools.text.splitlines() if ln.startswith("data:")]
        body = json.loads(data_lines[-1]) if data_lines else r_tools.json()
        names = [t["name"] for t in body.get("result", {}).get("tools", [])]
        print(f" [✓] Live Tools Discovered on Flashpoint MCP: {', '.join(names)}")
    else:
        print(f" [!] Tools list returned: {r_tools.status_code}")


def main():
    parser = argparse.ArgumentParser(description="Authenticate Flashpoint AI via EIP-191 Wallet JWT")
    parser.add_argument("--key", help="Optional private key hex string (0x...). If omitted, generates or uses agent wallet in .env")
    args = parser.parse_args()

    print("=" * 70)
    print("  FLASHPOINT AI AUTONOMOUS WALLET AUTHENTICATION (EIP-191)")
    print("=" * 70)

    acc = setup_wallet(args.key)
    data = authenticate_flashpoint(acc)
    test_mcp_connection(data["token"])

    print("\n" + "=" * 70)
    print("  ALL SET! Your agent wallet is authenticated with Flashpoint AI.")
    print("=" * 70)


if __name__ == "__main__":
    main()
