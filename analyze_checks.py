"""
analyze_checks.py — Summarize a finished checks report and recommend the
prediction horizon per the pre-registered rule.

Usage:
    python analyze_checks.py checks_report_v3.json
    python analyze_checks.py checks_report_v3.json --priors seed_priors_report.json

Outputs (printed + written to <report>_analysis.json):
  1. Usable rate, with a failure taxonomy: check1 / check2-lift / check2-final.
  2. Rotation balance after attrition: usable count by goal position g1..g4.
     (Rotation only balances residual prior asymmetry if attrition does not
     systematically remove one position.)
  3. Dead-member exposure: usable episodes whose true goal had prior=0.00
     (stricter gate applied) and whose distractor set contained dead members
     (final gate mildly inflated) — documented, not gated.
  4. Pooled inferability curve over USABLE episodes only.
  5. Horizon recommendation, pre-registered rule: t* = earliest prefix at
     which pooled mean accuracy exceeds chance (1/k) but remains below
     final_min; prediction horizon = t* .. T.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("report", help="checks_report_*.json")
    p.add_argument("--priors", default=None,
                   help="seed_priors_report.json (for dead-member exposure)")
    args = p.parse_args()

    with open(args.report, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = data.get("run_config", {})
    results = data.get("results", data if isinstance(data, list) else [])
    k = cfg.get("k", 4)
    chance = 1.0 / k
    final_min = cfg.get("final_min", 0.8)

    dead_by_family: dict = {}
    if args.priors:
        with open(args.priors, "r", encoding="utf-8") as f:
            pr = json.load(f)
        for r in pr.get("results", []):
            if "pick_shares" in r:
                dead_by_family[r["family_id"]] = set(r.get("dead_members", []))

    # ---- 1. usable + failure taxonomy -------------------------------
    usable, fail_c1, fail_lift, fail_final, fail_both2 = [], [], [], [], []
    for r in results:
        c1, c2 = r["check1"], r["check2"]
        if r["usable"]:
            usable.append(r)
            continue
        if not c1["passed_check1"]:
            fail_c1.append(r["episode_id"])
        if not c2["passed_check2"]:
            leaky = not c2["not_leaky"]
            uninferable = not c2["eventually_inferable"]
            if leaky and uninferable:
                fail_both2.append(r["episode_id"])
            elif leaky:
                fail_lift.append(r["episode_id"])
            elif uninferable:
                fail_final.append(r["episode_id"])

    n = len(results)
    print(f"USABLE: {len(usable)}/{n}")
    print(f"  fail check1 (consistency): {len(fail_c1)}  {fail_c1}")
    print(f"  fail check2 leak/lift    : {len(fail_lift)}  {fail_lift}")
    print(f"  fail check2 final<{final_min}: {len(fail_final)}  {fail_final}")
    if fail_both2:
        print(f"  fail check2 both         : {len(fail_both2)}  {fail_both2}")

    # ---- 2. rotation balance after attrition ------------------------
    by_pos = defaultdict(lambda: [0, 0])  # pos -> [usable, total]
    for r in results:
        eid = r["episode_id"]
        pos = eid.rsplit("_g", 1)[-1] if "_g" in eid else "?"
        by_pos[pos][1] += 1
        by_pos[pos][0] += int(r["usable"])
    print("\nROTATION BALANCE (usable/total by goal position):")
    for pos in sorted(by_pos):
        u, t = by_pos[pos]
        print(f"  g{pos}: {u}/{t}")
    counts = [by_pos[p][0] for p in sorted(by_pos)]
    if counts and max(counts) - min(counts) >= max(2, 0.5 * max(counts)):
        print("  WARNING: attrition is position-skewed; residual prior "
              "asymmetry may not be balanced in the usable set.")

    # ---- 3. dead-member exposure ------------------------------------
    if dead_by_family:
        true_dead, distractor_dead = [], []
        for r in usable:
            c2 = r["check2"]
            fam = r["episode_id"].rsplit("_g", 1)[0]
            dead = dead_by_family.get(fam, set())
            if c2.get("prior") == 0.0:
                true_dead.append(r["episode_id"])
            if dead:
                distractor_dead.append(r["episode_id"])
        print(f"\nDEAD-MEMBER EXPOSURE (usable set): "
              f"true-goal-dead={len(true_dead)}, "
              f"family-has-dead-members={len(distractor_dead)}")

    # ---- 4. pooled inferability curve (usable only) -----------------
    acc_by_prefix = defaultdict(list)
    for r in usable:
        for pt in r["check2"]["curve"]:
            acc_by_prefix[pt["prefix_len"]].append(pt["accuracy"])
    pooled = {pl: sum(v) / len(v) for pl, v in sorted(acc_by_prefix.items())}
    print("\nPOOLED INFERABILITY CURVE (usable episodes):")
    for pl, m in pooled.items():
        bar = "#" * int(m * 40)
        print(f"  prefix {pl}: {m:.2f} {bar}")

    # ---- 5. horizon recommendation (pre-registered rule) ------------
    t_star = None
    for pl, m in pooled.items():
        if chance < m < final_min:
            t_star = pl
            break
    T = max(pooled) if pooled else None
    print("\nHORIZON (pre-registered rule: earliest prefix with "
          f"chance<mean<{final_min}):")
    if t_star is not None:
        print(f"  t* = {t_star}  ->  predict steps {t_star + 1}..{T} "
              f"(horizon length {T - t_star})")
    else:
        print("  no prefix satisfies the rule — inspect the pooled curve; "
              "likely all-early-saturation (goal inferable immediately) or "
              "no usable episodes.")

    out = {
        "report": args.report,
        "usable": len(usable), "total": n,
        "failure_taxonomy": {"check1": fail_c1, "check2_lift": fail_lift,
                             "check2_final": fail_final, "check2_both": fail_both2},
        "rotation_balance": {f"g{p}": by_pos[p] for p in sorted(by_pos)},
        "pooled_curve": pooled,
        "t_star": t_star, "T": T,
        "chance": chance, "final_min": final_min,
    }
    out_path = args.report.rsplit(".", 1)[0] + "_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nAnalysis -> {out_path}")


if __name__ == "__main__":
    main()
