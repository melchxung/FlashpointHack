#!/usr/bin/env python
"""
Flashpoint Sentiment Sampling.

Builds the monthly K-shaped sentiment series from Flashpoint synthetic panels. Gemini issues
every Flashpoint MCP call: one `panel` create per income cohort, then one `chat` call per cohort
per month asking that panel the five Michigan questions as of that month. Answer counts are read
from Flashpoint's raw reply, never from Gemini's text.

Raw answers are saved after every call, so a rerun resumes without paying twice. There is no
simulator fallback: a call that keeps failing raises.

The panels are AI personas answering in hindsight, so the series is simulated history, not
observed survey data.
"""

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sampler.engine import PeriodSentimentRecord, SamplingResult
from src.sampler.gemini_mcp_agent import run_flashpoint_agent
from src.sampler.gemini_mcp_client import GeminiFlashpointMCPClient
from src.sampler.personas import COHORT_REGISTRY, PersonaArchetype
from src.sampler.survey import MICHIGAN_QUESTIONS, SurveyBattery

RAW_PATH = PROJECT_ROOT / "data" / "flashpoint_sentiment_raw.json"
SIGNALS_PATH = PROJECT_ROOT / "data" / "sentiment_signals.json"

RESPONDENTS_PER_COHORT = 30
SPENDING_WEIGHTS = {"upper": 0.48, "middle": 0.36, "lower": 0.16}
COHORTS = [PersonaArchetype.UPPER_QUARTILE, PersonaArchetype.MIDDLE_INCOME, PersonaArchetype.LOWER_QUARTILE]
SHORT = {
    PersonaArchetype.UPPER_QUARTILE: "upper",
    PersonaArchetype.MIDDLE_INCOME: "middle",
    PersonaArchetype.LOWER_QUARTILE: "lower",
}
MAX_ATTEMPTS = 3

_save_lock = threading.Lock()


def month_range(start: str, end: str) -> List[str]:
    y, m = map(int, start.split("-"))
    end_y, end_m = map(int, end.split("-"))
    months = []
    while (y, m) <= (end_y, end_m):
        months.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


def load_raw() -> Dict[str, Any]:
    if RAW_PATH.exists():
        return json.loads(RAW_PATH.read_text())
    return {"panels": {}, "answers": {}}


def save_raw(raw: Dict[str, Any]) -> None:
    with _save_lock:
        RAW_PATH.write_text(json.dumps(raw, indent=2))


def tool_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    """The JSON object a Flashpoint tool returns inside its MCP text content."""
    if result.get("isError"):
        raise RuntimeError(f"Flashpoint tool returned an error: {json.dumps(result)[:400]}")
    return json.loads(result["content"][0]["text"])


def create_panel(mcp: GeminiFlashpointMCPClient, archetype: PersonaArchetype) -> Dict[str, Any]:
    spec = COHORT_REGISTRY[archetype]
    name = f"K-Alpha {spec.display_name}"
    context = (
        f"US households, income ${spec.income_min:,.0f}-${spec.income_max:,.0f}, net worth "
        f"{spec.typical_net_worth_range}, housing tenure: {spec.housing_tenure}, debt profile: {spec.debt_profile}."
    )
    task = (
        "Call the Flashpoint `panel` tool exactly once with these arguments and nothing else: "
        f"action='create', name='{name}', num_respondents={RESPONDENTS_PER_COHORT}, "
        "research_objective='US consumer sentiment tracking with the five University of Michigan core questions.', "
        f"context='{context}'. Then report the panel_id."
    )

    def check(tool: str, args: Dict[str, Any]) -> None:
        if tool != "panel" or args.get("action") != "create" or args.get("num_respondents") != RESPONDENTS_PER_COHORT:
            raise ValueError(f"Refused: expected panel create with {RESPONDENTS_PER_COHORT} respondents, got {tool} {args}")

    out = run_flashpoint_agent(task, mcp=mcp, tools={"panel"}, check_call=check, max_turns=3)
    done = [c for c in out["tool_calls"] if c["ok"]]
    if not done:
        raise RuntimeError(f"Panel creation failed for {archetype.value}: {out['tool_calls']}")
    result = tool_payload(done[0]["result"])["result"]
    return {"panel_id": result["panel_id"], "name": result["name"], "num_respondents": result["num_respondents"]}


def question_block() -> str:
    return "\n".join(
        f"{code}: {q.prompt} Options: " + "; ".join(f"{k} = {v}" for k, v in q.options.items())
        for code, q in MICHIGAN_QUESTIONS.items()
    )


def parse_counts(reply: str, n: int) -> Dict[str, Dict[str, int]]:
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in Flashpoint reply: {reply[:200]}")
    data = json.loads(match.group(0))
    counts = {}
    for code in MICHIGAN_QUESTIONS:
        row = {k: int(data[code][k]) for k in ("1", "2", "3")}
        if min(row.values()) < 0 or sum(row.values()) != n:
            raise ValueError(f"{code} counts {row} do not sum to {n}")
        counts[code] = row
    return counts


def ask_panel(mcp: GeminiFlashpointMCPClient, panel: Dict[str, Any], month: str) -> Dict[str, Any]:
    y, m = map(int, month.split("-"))
    label = date(y, m, 1).strftime("%B %Y")
    message = (
        f"Ask my existing synthetic panel '{panel['name']}' (panel_id {panel['panel_id']}) the five survey "
        f"questions below. Every persona answers as themselves as of {label}, knowing only what had happened "
        "up to that month. Each persona picks exactly one option per question.\n\n" + question_block() + "\n\n"
        'Reply with ONLY a JSON object of the form {"PAGO": {"1": n, "2": n, "3": n}, "PAGED": {...}, '
        '"BUS12": {...}, "BUS5": {...}, "DUR": {...}} where n is the number of personas who chose that option. '
        "No other text."
    )
    task = (
        "Call the Flashpoint `chat` tool exactly once. Pass the text between <message> tags verbatim as the "
        "`message` argument and do not set conversation_id. Then reply with the tool's answer.\n"
        f"<message>{message}</message>"
    )

    def check(tool: str, args: Dict[str, Any]) -> None:
        sent = args.get("message", "")
        required = [panel["panel_id"], label, "ONLY a JSON object", *MICHIGAN_QUESTIONS]
        if tool != "chat" or args.get("conversation_id") or any(part not in sent for part in required):
            raise ValueError("Refused: the chat message must be passed verbatim, without a conversation_id")

    errors = []
    for _ in range(MAX_ATTEMPTS):
        out = run_flashpoint_agent(task, mcp=mcp, tools={"chat"}, check_call=check, max_turns=3)
        for call in out["tool_calls"]:
            if not call["ok"]:
                errors.append(call["error"][:200])
                continue
            try:
                payload = tool_payload(call["result"])
                counts = parse_counts(payload["reply"], panel["num_respondents"])
            except (ValueError, KeyError, TypeError, RuntimeError) as e:
                errors.append(str(e)[:200])
                continue
            return {"counts": counts, "conversation_id": payload.get("conversation_id"), "gemini_model": out["model"]}
    raise RuntimeError(f"No usable Flashpoint answer for {panel['name']} {month} after {MAX_ATTEMPTS} attempts: {errors}")


def build_signals(raw: Dict[str, Any], months: List[str]) -> Dict[str, Any]:
    records = []
    for month in months:
        scores, indices = {}, {}
        for arch in COHORTS:
            counts = raw["answers"][f"{month}|{arch.value}"]["counts"]
            scores[arch] = {
                code: SurveyBattery.calculate_relative_score(
                    [1] * row["1"] + [2] * row["2"] + [3] * row["3"]
                ).relative_score
                for code, row in counts.items()
            }
            indices[arch] = SurveyBattery.compute_indices(scores[arch])
        up, mid, low = (indices[a] for a in COHORTS)
        records.append(PeriodSentimentRecord(
            date=month,
            S_upper=up["ICS"],
            S_middle=mid["ICS"],
            S_lower=low["ICS"],
            S_equal=round(0.25 * up["ICS"] + 0.50 * mid["ICS"] + 0.25 * low["ICS"], 2),
            S_K=round(
                SPENDING_WEIGHTS["upper"] * up["ICS"]
                + SPENDING_WEIGHTS["middle"] * mid["ICS"]
                + SPENDING_WEIGHTS["lower"] * low["ICS"],
                2,
            ),
            Delta_S=round(up["ICS"] - low["ICS"], 2),
            ICC_upper=up["ICC"], ICE_upper=up["ICE"],
            ICC_lower=low["ICC"], ICE_lower=low["ICE"],
            ICC_middle=mid["ICC"], ICE_middle=mid["ICE"],
            relative_scores_upper=scores[PersonaArchetype.UPPER_QUARTILE],
            relative_scores_lower=scores[PersonaArchetype.LOWER_QUARTILE],
            relative_scores_middle=scores[PersonaArchetype.MIDDLE_INCOME],
            completions_total=RESPONDENTS_PER_COHORT * len(COHORTS),
        ))

    result = SamplingResult(
        dates=months,
        S_upper=[r.S_upper for r in records],
        S_lower=[r.S_lower for r in records],
        S_middle=[r.S_middle for r in records],
        S_equal=[r.S_equal for r in records],
        S_K=[r.S_K for r in records],
        Delta_S=[r.Delta_S for r in records],
        spending_weights=SPENDING_WEIGHTS,
        period_records=records,
        responses_summary={
            "total_periods": len(months),
            "total_synthetic_respondents": len(months) * len(COHORTS) * RESPONDENTS_PER_COHORT,
            "respondents_per_cohort_per_period": RESPONDENTS_PER_COHORT,
            "flashpoint_panels": {a.value: raw["panels"][a.value] for a in COHORTS},
        },
    ).to_dict()
    result["metadata"]["methodology"] = (
        "Flashpoint AI synthetic panels (AI personas) asked the five Michigan questions as of each month "
        "via the Flashpoint MCP `chat` tool, driven by Gemini. Simulated in hindsight, not observed survey data."
    )
    result["metadata"]["source"] = "flashpoint_mcp_live"
    return result


def main():
    parser = argparse.ArgumentParser(description="Sample K-shaped sentiment from Flashpoint synthetic panels")
    parser.add_argument("--start", default="2021-01")
    parser.add_argument("--end", default="2026-10")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-spend", type=float, default=5.0, help="USDC cap for this run")
    args = parser.parse_args()

    months = month_range(args.start, args.end)
    raw = load_raw()
    mcp = GeminiFlashpointMCPClient(session_cap_usdc=args.max_spend)

    for arch in COHORTS:
        if arch.value not in raw["panels"]:
            print(f"Creating Flashpoint panel for {arch.value}...")
            raw["panels"][arch.value] = create_panel(mcp, arch)
            save_raw(raw)
        print(f"  ✓ panel {arch.value}: {raw['panels'][arch.value]['panel_id']}")

    todo = [(m, a) for m in months for a in COHORTS if f"{m}|{a.value}" not in raw["answers"]]
    print(f"Asking panels: {len(todo)} of {len(months) * len(COHORTS)} cohort-months to run ({args.workers} workers)...")

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(ask_panel, mcp, raw["panels"][a.value], m): (m, a) for m, a in todo}
        for i, future in enumerate(as_completed(futures), 1):
            month, arch = futures[future]
            try:
                raw["answers"][f"{month}|{arch.value}"] = future.result()
                save_raw(raw)
            except Exception as e:
                failures.append(f"{month} {arch.value}: {e}")
            if i % 12 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done, {len(failures)} failed, paid {mcp.spent_usdc:.2f} USDC this run")

    if failures:
        raise RuntimeError(f"{len(failures)} cohort-months failed; rerun to resume. First: {failures[0]}")

    signals = build_signals(raw, months)
    SIGNALS_PATH.write_text(json.dumps(signals, indent=2))
    print(f"  ✓ Wrote {SIGNALS_PATH.name}: {len(months)} months from Flashpoint panels (paid {mcp.spent_usdc:.2f} USDC this run)")


if __name__ == "__main__":
    main()
