"""
analyze_predictors.py — Paired analysis of predictor conditions.

Usage:
    python analyze_predictors.py predictions_v3b.json
    python analyze_predictors.py predictions_v3b.json --report checks_report_v3.json

Reports, per model:
  - goal-choice accuracy by condition
  - paired 2x2 cells vs cot_matched for every other condition
  - McNemar EXACT test (two-sided binomial on discordant pairs)
  - mean reasoning length by condition (budget check)
  - sensitivity: same analysis excluding 'ceiling' episodes
    (usable episodes with raw step-1 accuracy >= 0.6, from --report)
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

CONTROL = "cot_matched"


def mcnemar_exact(b: int, c: int):
    """Two-sided exact McNemar on discordant counts (b = condition-only
    correct, c = control-only correct)."""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def analyze(records, label=""):
    models = sorted({r["model"] for r in records})
    conds = sorted({r["condition"] for r in records})
    print(f"\n===== {label} ({len({r['episode_id'] for r in records})} episodes) =====")
    for model in models:
        recs = [r for r in records if r["model"] == model]
        print(f"\n--- {model}")
        # accuracy + budget
        acc = defaultdict(lambda: [0, 0]); rlen = defaultdict(list)
        for r in recs:
            acc[r["condition"]][1] += 1
            acc[r["condition"]][0] += int(r["goal_correct"])
            rlen[r["condition"]].append(r["reasoning_len_chars"])
        for c in conds:
            if c in acc:
                k, n = acc[c]
                print(f"  {c:16s}: {k}/{n} = {k/n:.2f}   "
                      f"(mean rlen {sum(rlen[c])/len(rlen[c]):.0f})")
        # paired vs control
        byep = defaultdict(dict)
        for r in recs:
            byep[r["episode_id"]][r["condition"]] = r["goal_correct"]
        for c in conds:
            if c == CONTROL or c == "direct":
                continue
            b = sum(1 for d in byep.values()
                    if d.get(c) and not d.get(CONTROL))
            cc = sum(1 for d in byep.values()
                     if d.get(CONTROL) and not d.get(c))
            both = sum(1 for d in byep.values()
                       if d.get(c) and d.get(CONTROL))
            neither = sum(1 for d in byep.values()
                          if c in d and CONTROL in d
                          and not d[c] and not d[CONTROL])
            p = mcnemar_exact(b, cc)
            pstr = f"p={p:.3f}" if p is not None else "p=NA (no discordant)"
            print(f"  PAIRED {c} vs {CONTROL}: "
                  f"{c}-only={b}  {CONTROL}-only={cc}  "
                  f"both={both}  neither={neither}  ({pstr})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("predictions")
    p.add_argument("--report", default=None,
                   help="checks report, to identify ceiling episodes")
    args = p.parse_args()

    with open(args.predictions, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data["records"]

    analyze(records, label="ALL USABLE EPISODES")

    if args.report:
        with open(args.report, "r", encoding="utf-8") as f:
            rep = json.load(f)
        ceiling = {r["episode_id"] for r in rep["results"]
                   if r["usable"] and r["check2"]["step1_accuracy"] >= 0.6}
        if ceiling:
            print(f"\nCeiling episodes excluded in sensitivity: {sorted(ceiling)}")
            analyze([r for r in records if r["episode_id"] not in ceiling],
                    label="SENSITIVITY: ceiling episodes excluded")


if __name__ == "__main__":
    main()