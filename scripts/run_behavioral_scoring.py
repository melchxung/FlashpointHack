#!/usr/bin/env python3
"""
Run Retail Behavioral Interviews, Claude Scoring & Affluent Survey Replications.

Generates 90 conversational shopping transcripts across Costco (COST), Target (TGT),
and Walmart (WMT), scores them via the Anthropic Claude extraction pipeline,
replicates the 4 major survey batteries for affluent consumers, and saves results to:
- data/respondent_scored_transcripts.json
- data/affluent_survey_replications.json
"""

import sys
import os
import json
from pathlib import Path
from dataclasses import asdict
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.sampler.behavioral_probe import generate_retail_respondents
from src.sampler.claude_scorer import ClaudeScorer
from src.sampler.survey_replicator import run_affluent_survey_replications


def main():
    print("=" * 78)
    print("  RETAIL BEHAVIORAL INTERVIEW SCORING & AFFLUENT REPLICATION PIPELINE")
    print("=" * 78)

    # 1. Generate 90 Conversational Shopping Transcripts
    print("\n[1] Generating 90 Stratified Retail Interviews (30 COST, 30 TGT, 30 WMT)...")
    transcripts = generate_retail_respondents(n_per_cohort=30, random_state=np.random.RandomState(42))
    print(f"    ✓ Generated {len(transcripts)} respondent transcripts with full Q&A dialogues.")

    # 2. Run Claude Revealed Preference Extraction Pipeline
    scorer = ClaudeScorer()
    engine_name = "Live Anthropic Claude 3.5 Sonnet" if scorer.is_live else "Claude Semantic Rule Fallback (Zero-API Key)"
    print(f"\n[2] Scoring Transcripts via {engine_name}...")
    scores = scorer.score_batch(transcripts)
    print(f"    ✓ Extracted structured behavioral metrics for all {len(scores)} respondents.")

    # 3. Assemble JSON Payload
    data_dir = REPO_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    transcripts_json_path = data_dir / "respondent_scored_transcripts.json"

    records = []
    for t, s in zip(transcripts, scores):
        records.append({
            "respondent_id": t.respondent_id,
            "retailer": t.retailer,
            "cohort": t.cohort,
            "profile": t.profile,
            "qa_turns": t.qa_turns,
            "scores": asdict(s),
        })

    # Summary by retailer
    df_scores = pd.DataFrame([asdict(s) for s in scores])
    summary_by_retailer = {}
    for ret in ["COST", "TGT", "WMT"]:
        sub = df_scores[df_scores["retailer"] == ret]
        summary_by_retailer[ret] = {
            "n_respondents": len(sub),
            "mean_stated_macro_sentiment": round(float(sub["stated_macro_sentiment"].mean()), 3),
            "mean_revealed_spend_conviction": round(float(sub["revealed_spend_conviction"].mean()), 3),
            "mean_vibecession_divergence": round(float(sub["vibecession_divergence"].mean()), 3),
            "mean_trade_down_risk": round(float(sub["trade_down_risk"].mean()), 3),
            "mean_basket_size": round(float(sub["avg_basket_size"].mean()), 2),
            "mean_trips_per_month": round(float(sub["trips_per_month"].mean()), 1),
        }

    transcripts_payload = {
        "metadata": {
            "title": "Flashpoint Retail Behavioral Shopping Interviews & Claude Scoring",
            "total_respondents": len(records),
            "retailers": ["COST", "TGT", "WMT"],
            "scoring_engine": engine_name,
        },
        "summary_by_retailer": summary_by_retailer,
        "respondents": records,
    }

    with open(transcripts_json_path, "w") as f:
        json.dump(transcripts_payload, f, indent=2)
    print(f"\n[3] Saved scored transcripts to: {transcripts_json_path} ({os.path.getsize(transcripts_json_path) / 1024:.1f} KB)")

    # 4. Run Affluent Survey Replications across 4 Major Batteries
    print("\n[4] Running Flashpoint AI Affluent Replications (UMCSENT, Conf Board, Morning Consult, Rasmussen)...")
    replications_data = run_affluent_survey_replications(seed=42)
    replications_json_path = data_dir / "affluent_survey_replications.json"

    with open(replications_json_path, "w") as f:
        json.dump(replications_data, f, indent=2)
    print(f"    ✓ Saved affluent survey replications to: {replications_json_path} ({os.path.getsize(replications_json_path) / 1024:.1f} KB)")

    # 5. Display Summary Analytics
    print("\n" + "=" * 78)
    print("  RETAIL COHORT BEHAVIORAL SCORING SUMMARY")
    print("=" * 78)
    print(f"{'Retailer / Archetype':<26} | {'Stated Macro':>12} | {'Revealed Spend':>14} | {'Vibecession Gap':>15} | {'Trade-Down':>10}")
    print("-" * 78)
    for ret, stats in summary_by_retailer.items():
        name_map = {"COST": "COST (Costco Affluent)", "TGT": "TGT (Target Middle)", "WMT": "WMT (Walmart Value)"}
        print(
            f"{name_map[ret]:<26} | "
            f"{stats['mean_stated_macro_sentiment']:>+12.2f} | "
            f"{stats['mean_revealed_spend_conviction']:>+14.2f} | "
            f"{stats['mean_vibecession_divergence']:>+15.2f} | "
            f"{stats['mean_trade_down_risk']:>10.2f}"
        )
    print("-" * 78)

    print("\n" + "=" * 78)
    print("  AFFLUENT SURVEY BATTERY REPLICATIONS VS. POPULATION BASELINES")
    print("=" * 78)
    print(f"{'Survey Battery':<32} | {'Affluent':>10} | {'Baseline':>10} | {'Divergence Wedge':>16}")
    print("-" * 78)
    for code, b in replications_data["survey_batteries"].items():
        print(f"{b['survey_name'][:32]:<32} | {b['affluent_score']:>10.1f} | {b['baseline_population_score']:>10.1f} | {b['divergence_wedge']:>+16.1f} pts")
    print("-" * 78)

    print("\n" + "=" * 78)
    print("  ✓ PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 78)


if __name__ == "__main__":
    main()
