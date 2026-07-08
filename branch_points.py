"""
branch_points.py — Localize WHERE goal-hypotheses differentiate, from the
archived Check-2 trials. Zero API calls.

For every episode and prefix length t, the leakage judge's n trials give a
pick distribution over the k candidate goals. Two step-level signals:

  entropy(t)   — normalized Shannon entropy of the pick distribution
                 (1.0 = maximal uncertainty, 0.0 = judge fully committed)
  jump(t)      — accuracy(t) - accuracy(t-1): information about the TRUE
                 goal released by step t

The BRANCH STEP of an episode is argmax_t jump(t): the step that most
separates the true goal from the alternatives. Entropy distinguishes two
kinds of pre-branch uncertainty: spread (high entropy: judge undecided)
vs. committed-wrong (low entropy, low accuracy: misdirection working).

Usage:
    python branch_points.py checks_report_v3.json [--usable-only]
    python branch_points.py checks_report_v3.json --out branch_points.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter


def entropy_norm(picks: list, k: int) -> float:
    """Normalized Shannon entropy of pick counts (invalid picks dropped)."""
    valid = [p for p in picks if p is not None and 0 <= p < k]
    if not valid:
        return float("nan")
    n = len(valid)
    h = 0.0
    for cnt in Counter(valid).values():
        q = cnt / n
        h -= q * math.log2(q)
    return h / math.log2(k)


def analyze_episode(res: dict) -> dict:
    curve = res["check2"]["curve"]
    k = len(curve[0]["trials"][0]["candidates"])
    rows = []
    for point in curve:
        picks = [t.get("chosen_index") for t in point["trials"]]
        rows.append({
            "prefix_len": point["prefix_len"],
            "accuracy": point["accuracy"],
            "entropy": round(entropy_norm(picks, k), 3),
        })
    jumps = [(rows[i]["prefix_len"],
              rows[i]["accuracy"] - rows[i - 1]["accuracy"])
             for i in range(1, len(rows))]
    branch_step, branch_jump = max(jumps, key=lambda x: x[1]) if jumps else (None, 0.0)
    # committed-wrong regime: low entropy AND low accuracy before the branch
    pre = [r for r in rows if branch_step and r["prefix_len"] < branch_step]
    committed_wrong = [r["prefix_len"] for r in pre
                       if not math.isnan(r["entropy"])
                       and r["entropy"] <= 0.5 and r["accuracy"] <= 0.2]
    return {
        "episode_id": res["episode_id"],
        "usable": res["usable"],
        "branch_step": branch_step,
        "branch_jump": round(branch_jump, 2),
        "committed_wrong_steps": committed_wrong,
        "curve": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("report")
    p.add_argument("--usable-only", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    with open(args.report, "r", encoding="utf-8") as f:
        rep = json.load(f)
    results = [analyze_episode(r) for r in rep["results"]
               if r["check2"].get("curve")
               and (r["usable"] or not args.usable_only)]

    print(f"{'episode':24s} {'branch@':8s} {'jump':6s} entropy/accuracy per prefix")
    for r in results:
        prof = "  ".join(f"t{row['prefix_len']}:H={row['entropy']:.2f}/a={row['accuracy']:.1f}"
                         for row in r["curve"])
        cw = (f"  COMMITTED-WRONG@{r['committed_wrong_steps']}"
              if r["committed_wrong_steps"] else "")
        flag = "" if r["usable"] else " [excluded]"
        print(f"{r['episode_id']:24s} step {r['branch_step']}   "
              f"+{r['branch_jump']:.2f}  {prof}{cw}{flag}")

    used = [r for r in results if r["usable"]]
    dist = Counter(r["branch_step"] for r in used)
    print(f"\nBRANCH-STEP DISTRIBUTION (usable episodes, n={len(used)}):")
    for step in sorted(d for d in dist if d is not None):
        print(f"  step {step}: {'#' * dist[step]} ({dist[step]})")
    n_cw = sum(1 for r in used if r["committed_wrong_steps"])
    print(f"episodes with a committed-wrong regime before the branch: {n_cw}/{len(used)}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"source_report": args.report, "results": results},
                      f, ensure_ascii=False, indent=2)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
