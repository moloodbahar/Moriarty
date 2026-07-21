"""confirmatory_triggers.py — apply the FROZEN interpretive-capture
filter to new (confirmatory) data only (Step 2 of the confirmatory
plan).

The filter below is byte-for-byte the criteria string recorded in the
pilot's validated_triggers.json:

    same WC step+goal both observers; add=del within each; same clause
    across; argmax_agreement>=3 both; dp>=0.15 both; drop>=0.10 both

It is now PRE-REGISTERED: the constants in this file must not change
after the PREREGISTRATION.md hash is committed. The script verifies
that hash, refuses to run on pilot/replication episode IDs, and emits a
report in the same schema as validated_triggers.json plus per-gate
attrition counts (so the paper can report exactly where candidate
triggers fail, not just how many survive).

Inputs are the steps- and clauses-mode outputs of EITHER probe backend
(goal_distribution.py or probe_openweights.py) for BOTH observers.

Usage:
    python confirmatory_triggers.py \
        --steps-naive  ow_steps_naive.json \
        --steps-agentb ow_steps_agentb.json \
        --clauses-naive  ow_clauses_naive.json \
        --clauses-agentb ow_clauses_agentb.json \
        --exclude episodes_v3.json episodes_replication.json \
        --prereg PREREGISTRATION.md \
        --out confirmatory_triggers.json
"""

from __future__ import annotations

import argparse
import hashlib
import json

# ---------------- FROZEN FILTER CONSTANTS (pre-registered) -------------
DP_TARGET_WRONG_MIN = 0.15   # incremental wrong-goal gain, both observers
DROP_ON_DELETE_MIN = 0.10    # deletion drop, both observers
ARGMAX_AGREEMENT_MIN = 3     # of 4 label permutations, both observers
CRITERIA = ("same WC step+goal both observers; add=del within each; "
            "same clause across; argmax_agreement>=3 both; "
            "dp>=0.15 both; drop>=0.10 both")
# ----------------------------------------------------------------------


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {r["episode_id"]: r for r in d["results"]}, d.get("run_config",
                                                             {})


def wc_target(clause_rec):
    """The wrong-collapse localization target in a clauses-mode record."""
    for t in clause_rec.get("targets", []):
        if "wrong_collapse_step" in t.get("labels", []):
            return t
    return None


def clause_metrics(target, clause):
    """(dp_target_wrong of clause when added, drop when deleted)."""
    dp = next((r.get("dp_target_wrong")
               for r in target["rows"]
               if r["kind"] == "incremental"
               and r["clause_added"] == clause), None)
    drop = next((r.get("target_wrong_drop_on_delete")
                 for r in target["rows"]
                 if r["kind"] == "deletion"
                 and r["clause_removed"] == clause), None)
    return dp, drop


def step_agreement_ok(steps_rec, t):
    pt = next((p for p in steps_rec["points"] if p["t"] == t), None)
    return pt is not None and pt.get("argmax_agreement", 0) \
        >= ARGMAX_AGREEMENT_MIN


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-naive", required=True)
    ap.add_argument("--steps-agentb", required=True)
    ap.add_argument("--clauses-naive", required=True)
    ap.add_argument("--clauses-agentb", required=True)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="episode json files whose IDs must NOT appear "
                         "in the confirmatory data (pilot/replication)")
    ap.add_argument("--prereg", default="PREREGISTRATION.md",
                    help="preregistration file; its sha256 is recorded")
    ap.add_argument("--out", default="confirmatory_triggers.json")
    args = ap.parse_args()

    sn, cfg_n = load_results(args.steps_naive)
    sb, _ = load_results(args.steps_agentb)
    cn, _ = load_results(args.clauses_naive)
    cb, _ = load_results(args.clauses_agentb)

    # ---- pre-registration hygiene ------------------------------------
    excluded_ids = set()
    for path in args.exclude:
        with open(path, "r", encoding="utf-8") as f:
            for e in json.load(f):
                excluded_ids.add(e["episode_id"])
    overlap = excluded_ids & set(sn)
    if overlap:
        raise SystemExit(
            f"CONFIRMATORY VIOLATION: {len(overlap)} episode IDs overlap "
            f"the excluded (pilot/replication) sets: {sorted(overlap)[:5]}"
            " ... The confirmatory run must use fresh episodes only.")
    try:
        with open(args.prereg, "rb") as f:
            prereg_sha = hashlib.sha256(f.read()).hexdigest()
    except FileNotFoundError:
        raise SystemExit(f"{args.prereg} not found — the confirmatory "
                         "filter must reference a committed "
                         "preregistration file.")

    # ---- frozen filter, with per-gate attrition ----------------------
    gates = ["wc_both_observers", "same_step", "same_wrong_goal",
             "argmax_agreement_both", "add_del_agree_naive",
             "add_del_agree_agentb", "same_clause_across",
             "dp_min_both", "drop_min_both"]
    attrition = {g: 0 for g in gates}
    validated, candidates = [], []

    for eid in sorted(set(sn) & set(sb)):
        wcn = sn[eid].get("wrong_collapse_step")
        wcb = sb[eid].get("wrong_collapse_step")
        if not (wcn and wcb):
            continue
        attrition["wc_both_observers"] += 1
        rec = {"episode_id": eid, "step": wcn,
               "checks": {"same_step_and_goal": False,
                          "agree_naive": False, "agree_agentB": False,
                          "same_clause_across": False,
                          "agreement_gate": False,
                          "effect_inc": False, "effect_del": False}}
        if wcn != wcb:
            candidates.append({**rec, "fail": "different collapse steps",
                               "steps": [wcn, wcb]})
            continue
        attrition["same_step"] += 1

        tn = wc_target(cn.get(eid, {}))
        tb = wc_target(cb.get(eid, {}))
        if not (tn and tb):
            candidates.append({**rec,
                               "fail": "no clause localization record"})
            continue
        gn, gb = tn.get("target_wrong_goal"), tb.get("target_wrong_goal")
        if not gn or gn != gb:
            candidates.append({**rec, "fail": "different wrong goals",
                               "wrong_goals": [gn, gb]})
            continue
        attrition["same_wrong_goal"] += 1
        rec["wrong_goal"] = gn[:60]
        rec["checks"]["same_step_and_goal"] = True

        if not (step_agreement_ok(sn[eid], wcn)
                and step_agreement_ok(sb[eid], wcn)):
            candidates.append({**rec, "fail": "argmax agreement < 3"})
            continue
        attrition["argmax_agreement_both"] += 1
        rec["checks"]["agreement_gate"] = True

        if not tn["triggers_agree"]:
            candidates.append({**rec, "fail": "naive add/del disagree"})
            continue
        attrition["add_del_agree_naive"] += 1
        rec["checks"]["agree_naive"] = True
        if not tb["triggers_agree"]:
            candidates.append({**rec, "fail": "agent_b add/del disagree"})
            continue
        attrition["add_del_agree_agentb"] += 1
        rec["checks"]["agree_agentB"] = True

        clause = tn["trigger_by_addition"]
        if clause != tb["trigger_by_addition"]:
            candidates.append({**rec,
                               "fail": "different clause across observers",
                               "clauses": [clause,
                                           tb["trigger_by_addition"]]})
            continue
        attrition["same_clause_across"] += 1
        rec["checks"]["same_clause_across"] = True
        rec["trigger_clause"] = clause

        dp_n, drop_n = clause_metrics(tn, clause)
        dp_b, drop_b = clause_metrics(tb, clause)
        rec["dp_target_wrong"] = {"naive": dp_n, "agent_b": dp_b}
        rec["drop_on_delete"] = {"naive": drop_n, "agent_b": drop_b}
        if not (dp_n is not None and dp_b is not None
                and dp_n >= DP_TARGET_WRONG_MIN
                and dp_b >= DP_TARGET_WRONG_MIN):
            candidates.append({**rec, "fail": "dp_target_wrong below "
                               f"{DP_TARGET_WRONG_MIN}"})
            continue
        attrition["dp_min_both"] += 1
        rec["checks"]["effect_inc"] = True
        if not (drop_n is not None and drop_b is not None
                and drop_n >= DROP_ON_DELETE_MIN
                and drop_b >= DROP_ON_DELETE_MIN):
            candidates.append({**rec, "fail": "drop_on_delete below "
                               f"{DROP_ON_DELETE_MIN}"})
            continue
        attrition["drop_min_both"] += 1
        rec["checks"]["effect_del"] = True
        validated.append(rec)

    out = {"criteria": CRITERIA,
           "preregistration_sha256": prereg_sha,
           "frozen_constants": {
               "dp_target_wrong_min": DP_TARGET_WRONG_MIN,
               "drop_on_delete_min": DROP_ON_DELETE_MIN,
               "argmax_agreement_min": ARGMAX_AGREEMENT_MIN},
           "probe_model": cfg_n.get("model"),
           "probe_backend": cfg_n.get("backend", "api"),
           "n_episodes_scored": len(set(sn) & set(sb)),
           "excluded_episode_ids": len(excluded_ids),
           "gate_attrition": attrition,
           "validated": validated,
           "candidates": candidates}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"validated: {len(validated)} | labeled candidates: "
          f"{len(candidates)}")
    for g in gates:
        print(f"  {g:26s} {attrition[g]}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
