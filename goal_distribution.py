"""
goal_distribution.py — Exact interpretation-distribution curves via logprobs.

THE METHOD CHANGE: instead of estimating p(goal | prefix) by repeated
sampling (5 or 30 MCQ trials), read the EXACT distribution from the API:
the model answers with a single letter and `top_logprobs=20` on that one
token yields the full 4-way posterior in ONE call. No sampling noise, no
0.2-granularity, no bootstrap. Label-position bias is handled by
averaging over 4 cyclic goal->label permutations (the spread across
permutations is the honest error bar and is recorded).

Measures the same object as branch_points.py (naive-observer
interpretation of the story; no character cores shown), but exactly.

Per point it records the full distribution and computes:
  q      = p(true goal)
  H      = normalized entropy
  W      = max wrong-goal probability (dominant-wrong)
  JSD    = Jensen-Shannon divergence from the previous point

Three-point decomposition per episode (definitions from the analysis doc):
  uncertainty_peak      = argmax_t H_t
  interpretation_branch = argmax_t JSD_t
  resolution_branch     = argmax_t (q_t - q_{t-1}), requiring the gain
                          >= 0.30 and H decreasing
  committed_wrong steps = q<=0.20 and W>=0.70 and H<=0.50

Modes:
  --mode steps    prefix-level curves, prefixes 1..T
                  cost: n_eps x T x 4 perms calls (21x6x4 = 504)
  --mode clauses  inside each episode's interpretation_branch step:
                  incremental clause sweep + per-clause deletion ablation
                  (requires a steps-mode output as --steps-report)
                  cost: ~(2m+1) x 4 perms per episode, m = clauses

Usage:
    python goal_distribution.py episodes.json --report checks_report.json \
        --mode steps --out goal_dist_steps.json
    python goal_distribution.py episodes.json --report checks_report.json \
        --mode clauses --steps-report goal_dist_steps.json \
        --out goal_dist_clauses.json

Model: gpt-4o-mini (OpenAI logprobs). Gemini flash is skipped here: this
is an instrument, not a cross-model claim; use the open-weights script
for token-level and layer-level versions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re

from judges import CallLogger, load_episodes, sha256_file, utc_now

LABELS = ["A", "B", "C", "D"]
T0_TEXT = "(the story has not started yet)"

SYSTEM = """You are given the opening of a story and four candidate hidden \
goals the story's author might be pursuing. Answer which goal is most \
likely with a SINGLE LETTER (A, B, C, or D) and nothing else."""


def one_distribution(client, prefix_text: str, goals_in_label_order: list,
                     logger: CallLogger, purpose: str,
                     chars_block: str = "") -> dict:
    """One API call -> exact probability over the 4 labels."""
    listing = "\n".join(f"{LABELS[i]}. {g}"
                        for i, g in enumerate(goals_in_label_order))
    user = (f"STORY OPENING:\n{prefix_text}\n\n"
            + (f"CHARACTERS:\n{chars_block}\n\n" if chars_block else "")
            + f"CANDIDATE HIDDEN GOALS:\n{listing}\n\n"
            f"Which is most likely? Answer with a single letter.")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": user}],
        max_tokens=1, temperature=0.0,
        logprobs=True, top_logprobs=20,
    )
    tops = resp.choices[0].logprobs.content[0].top_logprobs
    p = {lab: 0.0 for lab in LABELS}
    for t in tops:
        tok = t.token.strip().upper()
        if tok in p:
            p[tok] += math.exp(t.logprob)
    coverage = sum(p.values())
    if coverage > 0:
        p = {k: v / coverage for k, v in p.items()}
    logger.log({"purpose": purpose, "model": "gpt-4o-mini",
                "user": user, "system": SYSTEM,
                "top_logprobs": [(t.token, t.logprob) for t in tops],
                "coverage": coverage, "ok": True})
    return {"p_labels": p, "coverage": coverage}


def goal_posterior(client, prefix_text: str, goals: list, true_goal: str,
                   logger: CallLogger, purpose: str,
                   chars_block: str = "") -> dict:
    """Average over 4 cyclic goal->label permutations; exact posterior
    per goal + per-permutation spread as the error bar."""
    per_goal = {g: [] for g in goals}
    coverages, perm_argmaxes = [], []
    for shift in range(4):
        order = goals[shift:] + goals[:shift]
        d = one_distribution(client, prefix_text, order, logger,
                             f"{purpose}_perm{shift}", chars_block=chars_block)
        coverages.append(d["coverage"])
        perm_p = {g: d["p_labels"][LABELS[i]] for i, g in enumerate(order)}
        perm_argmaxes.append(max(perm_p, key=perm_p.get))
        for i, g in enumerate(order):
            per_goal[g].append(d["p_labels"][LABELS[i]])
    p = {g: sum(v) / len(v) for g, v in per_goal.items()}
    spread = {g: max(v) - min(v) for g, v in per_goal.items()}
    q = p[true_goal]
    wrong = {g: v for g, v in p.items() if g != true_goal}
    dom = max(p, key=p.get)
    from collections import Counter as _C
    counts = _C(perm_argmaxes)
    H = 0.0
    for v in p.values():
        if v > 0:
            H -= v * math.log2(v)
    return {"p": p, "q": q, "H": round(H / 2.0, 4),  # log2(4)=2
            "W": round(max(wrong.values()), 4),
            "dominant_goal": dom,
            "dominant_is_true": dom == true_goal,
            # perms agreeing with the AVERAGED posterior's dominant —
            # not the modal pick's own frequency
            "argmax_agreement": counts.get(dom, 0),
            "modal_argmax_goal": counts.most_common(1)[0][0],
            "modal_argmax_count": counts.most_common(1)[0][1],
            "max_perm_spread": round(max(spread.values()), 4),
            "min_coverage": round(min(coverages), 4)}


def jsd(p1: dict, p2: dict) -> float:
    goals = p1.keys()
    m = {g: 0.5 * (p1[g] + p2[g]) for g in goals}
    def kl(a, b):
        s = 0.0
        for g in goals:
            if a[g] > 0 and b[g] > 0:
                s += a[g] * math.log2(a[g] / b[g])
        return s
    return round(0.5 * kl(p1, m) + 0.5 * kl(p2, m), 4)


def decompose(points: list) -> dict:
    """Separated decomposition over {t, q, H, W, p, dominant_is_true}.
    Three different narrative events get three different points:
    creating uncertainty, creating a wrong attractor, and resolving —
    a single max-JSD 'branch' conflates them (e.g., an episode whose
    biggest shift is the final CORRECTION, while the misdirection was
    manufactured steps earlier)."""
    for i, pt in enumerate(points):
        if i == 0:
            pt.update(jsd=0.0, dH=0.0, dq=0.0, dW=0.0,
                      misdirection_score=0.0, shift_type="start")
            continue
        prev = points[i - 1]
        pt["jsd"] = jsd(prev["p"], pt["p"])
        pt["dH"] = round(pt["H"] - prev["H"], 4)
        pt["dq"] = round(pt["q"] - prev["q"], 4)
        pt["dW"] = round(pt["W"] - prev["W"], 4)
        pt["misdirection_score"] = round(pt["dW"] - pt["dq"] - pt["dH"], 4)
        if (pt["dW"] > 0 and pt["dq"] < 0 and pt["dH"] < 0
                and not pt["dominant_is_true"]):
            pt["shift_type"] = "wrong_collapse"
        elif pt["dq"] >= 0.30 and pt["dH"] < 0:
            pt["shift_type"] = "resolution"
        elif pt["dH"] >= 0.15:
            pt["shift_type"] = "uncertainty_reopening"
        else:
            pt["shift_type"] = "minor"

    unc_peak = max(points, key=lambda x: x["H"])["t"]
    interp_branch = max(points[1:], key=lambda x: x["jsd"])["t"] \
        if len(points) > 1 else None

    # UC requires a MEANINGFUL entropy increase (>= 0.15), else numerical
    # drift gets labeled as uncertainty creation
    creation_cands = [pt for pt in points[1:] if pt["dH"] >= 0.15]
    uncertainty_creation = max(creation_cands, key=lambda x: x["dH"])["t"] \
        if creation_cands else None

    wrong_onset = next((pt["t"] for pt in points
                        if not pt["dominant_is_true"]), None)
    # first crossing from true-dominant to wrong-dominant (needs a true-
    # dominant point before it; with t0 present, distinguishes prior bias
    # from story-created error)
    def _ag(pt):  # default 4 keeps synthetic/test points ungated
        return pt.get("argmax_agreement", 4)

    wrong_entry = next((points[i]["t"] for i in range(1, len(points))
                        if points[i - 1]["dominant_is_true"]
                        and _ag(points[i - 1]) >= 3
                        and not points[i]["dominant_is_true"]
                        and _ag(points[i]) >= 3), None)

    collapse_cands = [pt for pt in points[1:]
                      if pt["shift_type"] == "wrong_collapse"
                      and _ag(pt) >= 3]
    wrong_collapse = max(collapse_cands, key=lambda x: x["dW"])["t"] \
        if collapse_cands else None

    res, best_gain = None, 0.0
    for i in range(1, len(points)):
        gain = points[i]["q"] - points[i - 1]["q"]
        if (gain >= 0.30 and points[i]["H"] < points[i - 1]["H"]
                and points[i]["dominant_is_true"]
                and _ag(points[i]) >= 3
                and gain > best_gain):
            res, best_gain = points[i]["t"], gain
    committed = [pt["t"] for pt in points
                 if pt["q"] <= 0.20 and pt["W"] >= 0.70 and pt["H"] <= 0.50]
    runs, start = [], None
    for i, t in enumerate(committed):
        if start is None:
            start = prev_t = t
        elif t == prev_t + 1:
            prev_t = t
        else:
            runs.append([start, prev_t])
            start = prev_t = t
    if start is not None:
        runs.append([start, prev_t])
    return {"uncertainty_peak": unc_peak,
            "uncertainty_creation_step": uncertainty_creation,
            "interpretation_branch": interp_branch,
            "wrong_onset": wrong_onset,
            "wrong_entry_step": wrong_entry,
            "wrong_collapse_step": wrong_collapse,
            "resolution_branch": res,
            "committed_wrong_steps": committed,
            "committed_wrong_runs": runs}


def split_clauses(step_text: str) -> list:
    """Sentences, then long sentences split at ', ' before conjunctions.
    Simple and recorded; a clause is the localization unit, not tokens."""
    sents = re.split(r"(?<=[.!?])\s+", step_text.strip())
    out = []
    for s in sents:
        if len(s) > 140:
            parts = re.split(r",\s+(?=and |but |while |as |so )", s)
            out.extend(x.strip() for x in parts if x.strip())
        elif s.strip():
            out.append(s.strip())
    return out


def prefix_text(ep, n_steps: int) -> str:
    return "\n".join(f"[step {i+1}] {s}"
                     for i, s in enumerate(ep.steps[:n_steps]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes")
    ap.add_argument("--report", required=True,
                    help="checks report (usable filter)")
    ap.add_argument("--mode", choices=["steps", "clauses"], default="steps")
    ap.add_argument("--steps-report", default=None,
                    help="steps-mode output (required for --mode clauses)")
    ap.add_argument("--observer", choices=["naive", "agent_b"],
                    default="naive",
                    help="naive: story+goals only (environment instrument). "
                         "agent_b: B's Level-2 information state (visible/"
                         "hidden cores per --core-report), direct answering. "
                         "NOTE: this captures B's INFORMATION state, not its "
                         "reasoning conditions — single-token readout cannot "
                         "include CoT/LF reasoning; those stay with the sweep.")
    ap.add_argument("--core-report", default=None,
                    help="core inferability report (required for agent_b)")
    ap.add_argument("--targets", nargs="+",
                    default=["wrong_entry_step", "wrong_collapse_step", "resolution_branch"],
                    help="clause mode: which decomposition points to "
                         "localize (also: uncertainty_creation_step, "
                         "interpretation_branch)")
    ap.add_argument("--out", default="goal_dist.json")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    logger = CallLogger(args.log or args.out.rsplit(".", 1)[0] + "_calls.jsonl")

    episodes = {e.episode_id: e for e in load_episodes(args.episodes)}
    with open(args.report, "r", encoding="utf-8") as f:
        keep = [r["episode_id"] for r in json.load(f)["results"] if r["usable"]]

    chars_blocks = {eid: "" for eid in keep}
    if args.observer == "agent_b":
        if not args.core_report:
            raise SystemExit("--observer agent_b requires --core-report")
        with open(args.core_report, "r", encoding="utf-8") as f:
            core_acc = {r["episode_id"]: {c["character"]: c["accuracy"]
                                          for c in r["characters"]}
                        for r in json.load(f)["results"]}
        for eid in keep:
            ep = episodes[eid]
            names = [c.name for c in ep.characters]
            visible = min(names, key=lambda n: core_acc[eid][n])
            chars_blocks[eid] = "\n".join(
                f"- {c.name}: moral core = "
                + (c.moral_core if c.name == visible
                   else "(unknown — infer from behavior)")
                for c in ep.characters)

    results = []
    if args.mode == "steps":
        for eid in keep:
            ep = episodes[eid]
            goals = [ep.hidden_goal] + list(ep.meta.get("distractor_goals", []))
            pts = []
            # t0 baseline: candidates (+ any static info) with NO story —
            # distinguishes prior bias (wrong at t0) from story-created
            # error (true at t0, wrong later). Same elicitation protocol
            # as every other point.
            d0 = goal_posterior(client, T0_TEXT,
                                goals, ep.hidden_goal, logger,
                                f"steps_{eid}_t0",
                                chars_block=chars_blocks[eid])
            d0["t"] = 0
            pts.append(d0)
            for t in range(1, len(ep.steps) + 1):
                d = goal_posterior(client, prefix_text(ep, t), goals,
                                   ep.hidden_goal, logger,
                                   f"steps_{eid}_t{t}",
                                   chars_block=chars_blocks[eid])
                d["t"] = t
                pts.append(d)
            dec = decompose(pts)
            results.append({"episode_id": eid, "points": pts, **dec})
            prof = " ".join(f"t{p['t']}:q={p['q']:.2f}/H={p['H']:.2f}"
                            f"{'*' if p['argmax_agreement'] < 3 else ''}"
                            for p in pts)
            print(f"{eid}: UC@{dec['uncertainty_creation_step']} "
                  f"WE@{dec['wrong_entry_step']} "
                  f"WC@{dec['wrong_collapse_step']} "
                  f"R@{dec['resolution_branch']} "
                  f"CWruns{dec['committed_wrong_runs']} | {prof}")

    else:  # clauses — localize each requested decomposition point separately
        if not args.steps_report:
            raise SystemExit("--mode clauses requires --steps-report")
        with open(args.steps_report, "r", encoding="utf-8") as f:
            steps_out = {r["episode_id"]: r for r in json.load(f)["results"]}
        for eid in keep:
            if eid not in steps_out:
                continue
            ep = episodes[eid]
            goals = [ep.hidden_goal] + list(ep.meta.get("distractor_goals", []))
            # distinct target steps, labeled; None targets dropped;
            # duplicates collapse to one localization with merged labels
            step_labels: dict = {}
            for label in args.targets:
                b = steps_out[eid].get(label)
                if b:
                    step_labels.setdefault(b, []).append(label)
            targets_out = []
            step_points = {pt["t"]: pt for pt in steps_out[eid]["points"]}
            for b, labels in sorted(step_labels.items()):
                before = prefix_text(ep, b - 1) if b > 1 else T0_TEXT
                clauses = split_clauses(ep.steps[b - 1])
                # the SPECIFIC wrong attractor at this step (identity, not
                # just max-wrong magnitude, which can change mid-sweep)
                sp = step_points.get(b, {})
                target_wrong = sp.get("dominant_goal") \
                    if sp and not sp.get("dominant_is_true") else None
                trigger_type = ("wrong" if ("wrong_entry_step" in labels
                                            or "wrong_collapse_step" in labels)
                                else "truth" if "resolution_branch" in labels
                                else "distribution")
                rows = []
                prev = goal_posterior(client, before, goals, ep.hidden_goal,
                                      logger, f"clause_{eid}_s{b}_base",
                                      chars_block=chars_blocks[eid])
                base = dict(prev)
                for j in range(1, len(clauses) + 1):
                    text = before + f"\n[step {b}] " + " ".join(clauses[:j])
                    cur = goal_posterior(client, text, goals, ep.hidden_goal,
                                         logger, f"clause_{eid}_s{b}_inc{j}",
                                         chars_block=chars_blocks[eid])
                    row = {"kind": "incremental", "n_clauses": j,
                           "clause_added": clauses[j - 1],
                           "jsd_from_prev": jsd(prev["p"], cur["p"]),
                           "dq": round(cur["q"] - prev["q"], 4),
                           "dW": round(cur["W"] - prev["W"], 4),
                           **{k: cur[k] for k in ("q", "H", "W")}}
                    if target_wrong:
                        row["p_target_wrong"] = round(cur["p"][target_wrong], 4)
                        row["dp_target_wrong"] = round(
                            cur["p"][target_wrong] - prev["p"][target_wrong], 4)
                    rows.append(row)
                    prev = cur
                full = prev
                for j in range(len(clauses)):
                    kept = [c for i, c in enumerate(clauses) if i != j]
                    text = before + f"\n[step {b}] " + " ".join(kept)
                    cur = goal_posterior(client, text, goals, ep.hidden_goal,
                                         logger, f"clause_{eid}_s{b}_del{j}",
                                         chars_block=chars_blocks[eid])
                    row = {"kind": "deletion",
                           "clause_removed": clauses[j],
                           # for evidence clauses (resolution): deletion
                           # DROPS q -> maximize this
                           "q_drop_vs_full": round(full["q"] - cur["q"], 4),
                           # for misleading clauses: deletion RESTORES q ->
                           # maximize this (q_drop would point the wrong way)
                           "q_restore_on_delete": round(cur["q"] - full["q"], 4),
                           "jsd_vs_full": jsd(full["p"], cur["p"]),
                           **{k: cur[k] for k in ("q", "H", "W")}}
                    if target_wrong:
                        row["target_wrong_drop_on_delete"] = round(
                            full["p"][target_wrong] - cur["p"][target_wrong], 4)
                    rows.append(row)
                incs = [r for r in rows if r["kind"] == "incremental"]
                dels = [r for r in rows if r["kind"] == "deletion"]
                if trigger_type == "wrong":
                    # the clause that built the SPECIFIC wrong attractor
                    if target_wrong:
                        trig_inc = max(incs, key=lambda r:
                                       r["dp_target_wrong"])["clause_added"]
                        trig_del = max(dels, key=lambda r:
                                       r["target_wrong_drop_on_delete"]
                                       )["clause_removed"]
                    else:  # fallback: the clause that most hurt the truth
                        trig_inc = min(incs, key=lambda r: r["dq"]
                                       )["clause_added"]
                        trig_del = max(dels, key=lambda r:
                                       r["q_restore_on_delete"]
                                       )["clause_removed"]
                elif trigger_type == "truth":
                    trig_inc = max(incs, key=lambda r: r["dq"])["clause_added"]
                    trig_del = max(dels, key=lambda r: r["q_drop_vs_full"]
                                   )["clause_removed"]
                else:
                    trig_inc = max(incs, key=lambda r: r["jsd_from_prev"]
                                   )["clause_added"]
                    trig_del = max(dels, key=lambda r: r["jsd_vs_full"]
                                   )["clause_removed"]
                targets_out.append({
                    "labels": labels, "step": b,
                    "trigger_type": trigger_type,
                    "target_wrong_goal": target_wrong,
                    "base_q": base["q"], "full_q": full["q"],
                    "clauses": clauses, "rows": rows,
                    "trigger_by_addition": trig_inc,
                    "trigger_by_deletion": trig_del,
                    "triggers_agree": trig_inc == trig_del})
                mark = "AGREE" if trig_inc == trig_del else "DISAGREE"
                print(f"{eid} {'/'.join(labels)}@{b} ({trigger_type}) "
                      f"[{mark}] +:'{trig_inc[:50]}' -:'{trig_del[:50]}'")
            if targets_out:
                results.append({"episode_id": eid, "targets": targets_out})

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"run_config": {
                       "created_at": utc_now(),
                       "mode": args.mode, "observer": args.observer,
                       "clause_targets": args.targets if args.mode == "clauses" else None,
                       "episodes_sha256": sha256_file(args.episodes),
                       "method": "exact posteriors from top_logprobs, "
                                 "4 cyclic label permutations averaged",
                       "model": "gpt-4o-mini",
                   },
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
