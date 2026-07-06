"""
calibrate_judge.py — Validate the ConsistencyJudge BEFORE trusting it.

The judge is itself a manipulation-check instrument; an unvalidated judge
makes Check 1 meaningless. This script runs the judge on hand-labeled
calibration cases and reports accuracy + a confusion matrix.

Gate: do not proceed to episode validation unless accuracy >= 7/8 here,
AND both CONTRADICTS cases are caught (a judge that misses contradictions
silently admits broken episodes into the dataset).

Usage:
    export OPENAI_API_KEY=...
    python calibrate_judge.py calibration_cases.json --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from judges import LLMClient, CONSISTENCY_SYSTEM, sha256_file, sha256_text, utc_now


def judge_calibration_case(client: LLMClient, case: dict) -> str:
    chars = "\n".join(
        f"- {c['name']}: moral core = {c['moral_core']}" for c in case["characters"]
    )
    user = (
        f"HIDDEN GOAL of the author:\n{case['hidden_goal']}\n\n"
        f"CHARACTERS:\n{chars}\n\n"
        f"STORY SO FAR:\n{case['story_so_far']}\n\n"
        f"NEW STEP:\n{case['new_step']}\n\n"
        f"Classify the NEW STEP."
    )
    out = client.json_call(CONSISTENCY_SYSTEM, user,
                           purpose="calibration_consistency")
    label = str(out.get("label", "")).strip().upper()
    return label if label in {"ADVANCES", "NEUTRAL", "CONTRADICTS"} else "INVALID"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cases", help="path to calibration_cases.json")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--repeats", type=int, default=1,
                   help="repeat each case N times to check judge stability")
    p.add_argument("--log", default="calibration_calls.jsonl",
                   help="JSONL call log path")
    args = p.parse_args()

    with open(args.cases, "r", encoding="utf-8") as f:
        cases = json.load(f)

    client = LLMClient(model=args.model, log_path=args.log)

    rows = []
    confusion: Counter = Counter()
    for case in cases:
        preds = [judge_calibration_case(client, case) for _ in range(args.repeats)]
        majority = Counter(preds).most_common(1)[0][0]
        stable = len(set(preds)) == 1
        gold = case["gold_label"]
        correct = majority == gold
        confusion[(gold, majority)] += 1
        rows.append({
            "case_id": case["case_id"],
            "gold": gold,
            "pred": majority,
            "all_preds": preds,
            "stable": stable,
            "correct": correct,
        })
        mark = "OK " if correct else "MISS"
        print(f"[{mark}] {case['case_id']}: gold={gold} pred={majority}"
              + ("" if stable else f"  (UNSTABLE: {preds})"))

    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    contradicts_cases = [r for r in rows if r["gold"] == "CONTRADICTS"]
    contradicts_recall = (
        sum(r["correct"] for r in contradicts_cases) / len(contradicts_cases)
        if contradicts_cases else float("nan")
    )

    print(f"\nAccuracy: {acc:.2f} ({sum(r['correct'] for r in rows)}/{n})")
    print(f"CONTRADICTS recall: {contradicts_recall:.2f}")
    print("\nConfusion (gold -> pred):")
    for (g, pr), c in sorted(confusion.items()):
        print(f"  {g:12s} -> {pr:12s} : {c}")

    gate = acc >= 7 / 8 and contradicts_recall == 1.0
    print(f"\nGATE {'PASSED' if gate else 'FAILED'}: "
          f"{'judge is usable for Check 1' if gate else 'fix the judge prompt before validating episodes'}")

    with open("calibration_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "run_config": {
                "created_at": utc_now(),
                "cases_path": args.cases,
                "cases_sha256": sha256_file(args.cases),
                "judge_model": args.model,
                "repeats": args.repeats,
                "gate_rule": "accuracy >= 7/8 AND CONTRADICTS recall == 1.0",
                "call_log": args.log,
                "consistency_prompt_sha256": sha256_text(CONSISTENCY_SYSTEM),
            },
            "rows": rows,
            "accuracy": acc,
            "contradicts_recall": contradicts_recall,
            "gate_passed": gate,
        }, f, ensure_ascii=False, indent=2)
    print(f"All calls logged to: {args.log}")


if __name__ == "__main__":
    main()
