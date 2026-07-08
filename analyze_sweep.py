"""
analyze_sweep.py — Analysis of the exploratory prefix sweep against the
branch points measured by branch_points.py.

Reports, per model x condition:
  1. goal accuracy vs prefix length
  2. branch-aligned accuracy: pooled over episodes at offset
     (prefix - branch_step), i.e., trajectory relative to the moment the
     story releases the differentiating information
  3. RECOVERY  = P(correct at first post-branch prefix | wrong at last
                 pre-branch prefix)          [pre-stated metric]
  4. LOCK-IN  = P(same wrong choice repeated at first post-branch prefix
                 | wrong at last pre-branch prefix)   [pre-stated metric]
Episodes whose branch step <= min prefix have no pre-branch observation
and are excluded from recovery/lock-in denominators (reported).

Usage:
    python analyze_sweep.py sweep_v1.json branch_points_v3.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("sweep")
    p.add_argument("branch_report")
    args = p.parse_args()

    with open(args.sweep, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data["records"]
    prefixes = sorted({r["prefix_len"] for r in records})
    conds = data["run_config"]["conditions"]
    models = data["run_config"]["models"]

    with open(args.branch_report, "r", encoding="utf-8") as f:
        branch = {r["episode_id"]: r["branch_step"]
                  for r in json.load(f)["results"] if r["usable"]}

    idx = defaultdict(dict)  # (model, cond, episode) -> {prefix: record}
    for r in records:
        idx[(r["model"], r["condition"], r["episode_id"])][r["prefix_len"]] = r

    print("=== 1. ACCURACY vs PREFIX ===")
    header = "model/condition".ljust(34) + "".join(f"t={t}   " for t in prefixes)
    print(header)
    for m in models:
        for c in conds:
            row = []
            for t in prefixes:
                cell = [r["goal_correct"] for r in records
                        if r["model"] == m and r["condition"] == c
                        and r["prefix_len"] == t]
                row.append(f"{sum(cell)/len(cell):.2f}" if cell else "  - ")
            print(f"{m} {c}".ljust(34) + "  ".join(row))

    print("\n=== 2. BRANCH-ALIGNED ACCURACY (offset = prefix - branch) ===")
    offsets = list(range(-4, 5))
    print("model/condition".ljust(34)
          + "".join(f"{o:+d}    " for o in offsets))
    for m in models:
        for c in conds:
            row = []
            for o in offsets:
                cell = []
                for eid, b in branch.items():
                    if b is None:
                        continue
                    rec = idx[(m, c, eid)].get(b + o)
                    if rec:
                        cell.append(rec["goal_correct"])
                row.append(f"{sum(cell)/len(cell):.2f}({len(cell):2d})"
                           if cell else "   -    ")
            print(f"{m} {c}".ljust(34) + " ".join(row))

    print("\n=== 3/4. RECOVERY and LOCK-IN at the branch (pre-stated) ===")
    excluded = [e for e, b in branch.items() if b is None or b <= min(prefixes)]
    print(f"(episodes without a pre-branch observation, excluded: "
          f"{len(excluded)}: {sorted(excluded)})")
    for m in models:
        print(f"--- {m}")
        for c in conds:
            wrong_pre = recovered = locked = 0
            for eid, b in branch.items():
                if b is None or b <= min(prefixes):
                    continue
                cells = idx[(m, c, eid)]
                pre_ts = [t for t in prefixes if t < b and t in cells]
                post_ts = [t for t in prefixes if t >= b and t in cells]
                if not pre_ts or not post_ts:
                    continue
                pre, post = cells[max(pre_ts)], cells[min(post_ts)]
                if not pre["goal_correct"]:
                    wrong_pre += 1
                    if post["goal_correct"]:
                        recovered += 1
                    elif post["choice_index"] == pre["choice_index"]:
                        locked += 1
            if wrong_pre:
                print(f"  {c:14s}: wrong pre-branch n={wrong_pre:2d} | "
                      f"RECOVERY {recovered}/{wrong_pre} = "
                      f"{recovered/wrong_pre:.2f} | "
                      f"LOCK-IN {locked}/{wrong_pre} = {locked/wrong_pre:.2f}")
            else:
                print(f"  {c:14s}: no wrong-pre-branch cases")

    print("\n=== exploratory: mean confidence vs prefix ===")
    for m in models:
        for c in conds:
            row = []
            for t in prefixes:
                cell = [r["confidence"] for r in records
                        if r["model"] == m and r["condition"] == c
                        and r["prefix_len"] == t
                        and isinstance(r["confidence"], (int, float))]
                row.append(f"{sum(cell)/len(cell):.2f}" if cell else "  - ")
            print(f"{m} {c}".ljust(34) + "  ".join(row))


if __name__ == "__main__":
    main()
