"""
run_predictors_v4.py — Level-2 (partial observability) predictor experiment.

Pre-registered question (DECISIONS.md, dated before this run):
  Pilot 1 (all cores visible) showed NO latent-first advantage and Direct
  beating all reasoning conditions — consistent with an information-
  saturated task. v4 hides 2 of 3 moral cores from Agent B, on the SAME
  23 validated episodes, same prefix t*, same conditions. The Level-2
  effect is the paired latent_first-vs-cot_matched difference here,
  compared against its Level-1 value: does latent scaffolding help once
  there is genuinely something latent to infer?

Design:
  - Visibility rule (deterministic, from Check 3): the LEAST-inferable
    character's core is VISIBLE; the other two are hidden. Hidden cores
    below the inferability gate are hidden but NOT scored (irreducible).
  - Structured output, all exact-graded (no free-text matcher needed for
    primary/secondary metrics):
      goal_choice_index      -> primary (k-way, exact)
      target_character       -> secondary (exact vs character named in goal)
      core choices per hidden character -> secondary (4-way MCQ from the
        same core bank as Check 3; options fixed per episode+character
        across conditions and models, so everything stays paired)
      next_beat (free text)  -> stored for the calibrated matcher, later
      confidence (0..1)      -> exploratory calibration analysis only
  - Conditions: direct / cot_matched / latent_first (author-level wording,
    = LF-v2 of pilot 1; recorded in run_config).
  - reasoning text IS stored (pilot-1 audit bug fixed).

Usage:
    python run_predictors_v4.py episodes_v3.json checks_report_v3.json \
        checks_report_v3_analysis.json core_inferability_report.json \
        --out predictions_v4.json
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Optional

from judges import sha256_file, utc_now, load_episodes
from gemini_client import make_client
from check_core_inferability import CORE_BANK


CONDITIONS = ("direct", "cot_matched", "latent_first")

BASE_SYSTEM = """You are an observer analyzing an unfolding story. You see \
only the opening. Some characters' moral cores (behavioral rules) are \
unknown to you and must be inferred from behavior. You will predict the \
author's hidden goal, who is being maneuvered, the unknown moral cores, \
and what happens next."""

REASONING_LINE = {
    "direct":
        "Answer immediately, without writing out reasoning.",
    "cot_matched":
        "First think step by step about the story's events, characters, and "
        "where it is heading. Reason carefully and at length before answering.",
    "latent_first":
        "The story has a single hidden AUTHORIAL goal: one character is being "
        "covertly maneuvered toward some outcome. First infer, from the events "
        "so far, WHO is being maneuvered and TOWARD WHAT outcome, and what "
        "moral cores best explain the behavior of the characters whose cores "
        "are unknown. Reason carefully and at length, state your best "
        "hypothesis about the author's hidden goal, then answer.",
}


def build_output_spec(hidden_chars: list) -> str:
    core_fields = ", ".join(
        f'"{n}": <0-based index into {n}\'s CANDIDATE MORAL CORES>'
        for n in hidden_chars)
    return (
        'Then respond with a JSON object:\n'
        '{"reasoning": "<your reasoning; empty string if answering directly>", '
        '"goal_choice_index": <0-based index into CANDIDATE GOALS>, '
        '"target_character": "<name of the character being maneuvered>", '
        '"inferred_cores": {' + core_fields + '}, '
        '"next_beat": "<one to three sentences predicting the next story step>", '
        '"confidence": <0.0-1.0, your confidence in the goal choice>}'
    )


def gt_target(goal: str, char_names: list) -> Optional[str]:
    """Ground-truth maneuver target = FIRST character name occurring in the
    hidden goal. Matches the family grammar 'covertly cause PERSON to ...';
    later names are beneficiaries/constraints, not targets. Heuristic —
    recorded as such; target accuracy is a secondary metric. None if no
    name occurs."""
    positions = [(goal.lower().find(n.lower()), n) for n in char_names
                 if n.lower() in goal.lower()]
    return min(positions)[1] if positions else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("episodes")
    p.add_argument("report")
    p.add_argument("analysis")
    p.add_argument("core_report")
    p.add_argument("--models", nargs="+",
                   default=["gpt-4o-mini", "gemini-2.5-flash"])
    p.add_argument("--out", default="predictions_v4.json")
    p.add_argument("--log", default=None)
    p.add_argument("--seed", type=int, default=61)
    args = p.parse_args()

    episodes = {e.episode_id: e for e in load_episodes(args.episodes)}
    with open(args.report, "r", encoding="utf-8") as f:
        keep = [r["episode_id"] for r in json.load(f)["results"] if r["usable"]]
    with open(args.analysis, "r", encoding="utf-8") as f:
        analysis = json.load(f)
    t_star = analysis["t_star"]
    with open(args.core_report, "r", encoding="utf-8") as f:
        core_rep = json.load(f)
    core_acc = {r["episode_id"]: {c["character"]: c for c in r["characters"]}
                for r in core_rep["results"]}
    core_min = core_rep["run_config"]["core_min"]

    log_path = args.log or args.out.rsplit(".", 1)[0] + "_calls.jsonl"
    clients = {m: make_client(m, log_path=log_path) for m in args.models}
    rng = random.Random(args.seed)

    records = []
    for eid in keep:
        ep = episodes[eid]
        if eid not in core_acc:
            print(f"  WARNING {eid}: missing from core report — skipped")
            continue
        chars = ep.characters
        names = [c.name for c in chars]
        accs = core_acc[eid]
        # visibility rule: least-inferable core is the visible one
        visible = min(names, key=lambda n: accs[n]["accuracy"])
        hidden = [n for n in names if n != visible]
        scored_hidden = [n for n in hidden if accs[n]["accuracy"] >= core_min]

        prefix_text = "\n".join(f"[step {i+1}] {s}"
                                for i, s in enumerate(ep.steps[:t_star]))
        actual_next = ep.steps[t_star] if t_star < len(ep.steps) else ""

        goal_cands = [ep.hidden_goal] + list(ep.meta.get("distractor_goals", []))
        rng.shuffle(goal_cands)
        true_goal_index = goal_cands.index(ep.hidden_goal)

        # per hidden character: fixed 4-way core options (paired across cells)
        core_opts, core_true_idx = {}, {}
        episode_cores = {c.moral_core for c in chars}
        for n in hidden:
            true_core = next(c.moral_core for c in chars if c.name == n)
            bank = [b for b in CORE_BANK if b not in episode_cores]
            opts = [true_core] + rng.sample(bank, 3)
            rng.shuffle(opts)
            core_opts[n] = opts
            core_true_idx[n] = opts.index(true_core)

        char_lines = []
        for c in chars:
            if c.name == visible:
                char_lines.append(f"- {c.name}: moral core = {c.moral_core}")
            else:
                char_lines.append(f"- {c.name}: moral core = (unknown — infer "
                                  f"from behavior)")
        core_mcq = "\n\n".join(
            f"{n}'s CANDIDATE MORAL CORES:\n" +
            "\n".join(f"[{i}] {o}" for i, o in enumerate(core_opts[n]))
            for n in hidden)
        goal_listing = "\n".join(f"[{i}] {g}" for i, g in enumerate(goal_cands))
        target_truth = gt_target(ep.hidden_goal, names)

        for model in args.models:
            client = clients[model]
            for cond in CONDITIONS:
                user = (f"STORY OPENING:\n{prefix_text}\n\n"
                        f"CHARACTERS:\n" + "\n".join(char_lines) + "\n\n"
                        f"CANDIDATE GOALS:\n{goal_listing}\n\n"
                        f"{core_mcq}\n\n"
                        f"{REASONING_LINE[cond]}\n\n"
                        f"{build_output_spec(hidden)}")
                out = client.json_call(BASE_SYSTEM, user, temperature=0.0,
                                       purpose=f"predict_v4_{cond}")
                try:
                    gchoice = int(out.get("goal_choice_index", -1))
                except (TypeError, ValueError):
                    gchoice = -1
                inferred = out.get("inferred_cores", {}) or {}
                core_results = {}
                for n in hidden:
                    try:
                        ci = int(inferred.get(n, -1))
                    except (TypeError, ValueError):
                        ci = -1
                    core_results[n] = {
                        "choice": ci,
                        "correct": ci == core_true_idx[n],
                        "scored": n in scored_hidden,
                    }
                reasoning = out.get("reasoning", "") or ""
                tgt = str(out.get("target_character", "")).strip()
                records.append({
                    "episode_id": eid,
                    "family_id": ep.meta.get("family_id"),
                    "model": model,
                    "same_model": not model.startswith("gemini"),
                    "condition": cond,
                    "visible_core_char": visible,
                    "hidden_core_chars": hidden,
                    "goal_candidates": goal_cands,
                    "goal_true_index": true_goal_index,
                    "goal_choice_index": gchoice,
                    "goal_correct": gchoice == true_goal_index,
                    "target_truth": target_truth,
                    "target_pred": tgt,
                    "target_correct": (tgt == target_truth)
                                      if target_truth else None,
                    "core_options": core_opts,
                    "core_results": core_results,
                    "confidence": out.get("confidence"),
                    "reasoning": reasoning,
                    "reasoning_len_chars": len(reasoning),
                    "next_beat": out.get("next_beat", ""),
                    "actual_next_step": actual_next,
                })
                mark = "+" if gchoice == true_goal_index else "."
                ncore = sum(1 for n in scored_hidden
                            if core_results[n]["correct"])
                print(f"{mark} {eid} | {model} | {cond} "
                      f"(cores {ncore}/{len(scored_hidden)}, "
                      f"rlen={len(reasoning)})")

    result = {
        "run_config": {
            "created_at": utc_now(),
            "level": "L2_partial_observability",
            "episodes_path": args.episodes,
            "episodes_sha256": sha256_file(args.episodes),
            "core_report_path": args.core_report,
            "core_report_sha256": sha256_file(args.core_report),
            "t_star": t_star,
            "models": args.models,
            "conditions": list(CONDITIONS),
            "base_system": BASE_SYSTEM,
            "reasoning_lines": REASONING_LINE,
            "visibility_rule": "least-inferable core visible; hidden cores "
                               "below gate hidden but not scored",
            "call_log": log_path,
            "rng_seed": args.seed,
        },
        "records": records,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # preview
    from collections import defaultdict
    acc = defaultdict(lambda: [0, 0]); core_a = defaultdict(lambda: [0, 0])
    for r in records:
        k = (r["model"], r["condition"])
        acc[k][1] += 1; acc[k][0] += int(r["goal_correct"])
        for n, cr in r["core_results"].items():
            if cr["scored"]:
                core_a[k][1] += 1; core_a[k][0] += int(cr["correct"])
    print("\nGOAL ACCURACY / CORE-INFERENCE ACCURACY (preview):")
    for k in sorted(acc):
        g, gn = acc[k]; c, cn = core_a[k]
        cstr = f"{c}/{cn}={c/cn:.2f}" if cn else "n/a"
        print(f"  {k[0]:18s} {k[1]:13s}: goal {g}/{gn}={g/gn:.2f} | cores {cstr}")
    print(f"\nPredictions -> {args.out}\nCalls -> {log_path}")
    print("Analyze with: python analyze_predictors.py " + args.out)


if __name__ == "__main__":
    main()
