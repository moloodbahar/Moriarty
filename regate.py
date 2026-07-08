"""
regate.py — Recompute Check-2 gating from STORED curves with a new priors
file. Zero API calls: the inferability curves in the checks report are
reused; only the prior subtracted in the lift gate changes.

Purpose: robustness of the usable set to prior-measurement noise. The
pilot's lift gating used n=12 priors (sigma up to ~0.14); re-measuring
priors at n=24 and re-gating tests whether the 23-episode set (and hence
downstream conclusions) is stable.

Usage:
    python check_seed_priors.py seeds_v3_1.json --n-trials 24 \
        --out seed_priors_v3_1_n24.json
    python regate.py checks_report_v3.json seed_priors_v3_1_n24.json \
        --episodes episodes_v3.json
"""

from __future__ import annotations

import argparse
import json

from judges import sha256_file, utc_now


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("report", help="existing checks report (curves reused)")
    p.add_argument("priors", help="new priors report")
    p.add_argument("--episodes", required=True,
                   help="episodes file (to map episode -> family -> goal)")
    p.add_argument("--lift-max", type=float, default=None,
                   help="default: value recorded in the report's run_config")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    with open(args.report, "r", encoding="utf-8") as f:
        rep = json.load(f)
    cfg = rep["run_config"]
    lift_max = args.lift_max if args.lift_max is not None else cfg["lift_max"]
    final_min = cfg["final_min"]
    step1_max = cfg.get("step1_max", 0.5)

    with open(args.priors, "r", encoding="utf-8") as f:
        pri = json.load(f)
    priors_map = {r["family_id"]: {g.strip(): s for g, s in r["pick_shares"].items()}
                  for r in pri["results"] if "pick_shares" in r}

    with open(args.episodes, "r", encoding="utf-8") as f:
        eps = {e["episode_id"]: e for e in json.load(f)}

    changed, results = [], []
    for r in rep["results"]:
        eid = r["episode_id"]
        ep = eps[eid]
        fam = ep["meta"].get("family_id")
        goal = ep["hidden_goal"].strip()
        old_prior = r["check2"].get("prior")
        new_prior = priors_map.get(fam, {}).get(goal)
        step1 = r["check2"]["step1_accuracy"]
        final = r["check2"]["final_accuracy"]
        if new_prior is not None:
            not_leaky = (step1 - new_prior) <= lift_max
            mode = "lift"
        else:
            not_leaky = step1 <= step1_max
            mode = "raw"
        new_c2 = not_leaky and (final >= final_min)
        new_usable = r["check1"]["passed_check1"] and new_c2
        if new_usable != r["usable"]:
            changed.append((eid, r["usable"], new_usable,
                            old_prior, new_prior, step1))
        results.append({"episode_id": eid, "usable": new_usable,
                        "old_usable": r["usable"], "prior_old": old_prior,
                        "prior_new": new_prior, "gating_mode": mode,
                        "step1": step1, "final": final,
                        "check1": r["check1"]["passed_check1"]})

    n_old = sum(r["old_usable"] for r in results)
    n_new = sum(r["usable"] for r in results)
    print(f"usable: {n_old} -> {n_new}  (report={args.report}, "
          f"new priors={args.priors})")
    if changed:
        print("\nFLIPPED EPISODES:")
        for eid, old, new, po, pn, s1 in changed:
            print(f"  {eid}: {old}->{new}  prior {po}->{pn}  step1={s1:.2f}")
    else:
        print("no episode flipped — usable set is stable under the new priors")

    out = args.out or args.report.rsplit(".", 1)[0] + "_regated.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"run_config": {
                       "created_at": utc_now(),
                       "base_report": args.report,
                       "base_report_sha256": sha256_file(args.report),
                       "new_priors": args.priors,
                       "new_priors_sha256": sha256_file(args.priors),
                       "episodes": args.episodes,
                       "episodes_sha256": sha256_file(args.episodes),
                       "lift_max": lift_max, "final_min": final_min,
                       "note": "curves reused from base report; no new "
                               "leakage measurements",
                   },
                   "results": results,
                   "flipped": [c[0] for c in changed]},
                  f, ensure_ascii=False, indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
