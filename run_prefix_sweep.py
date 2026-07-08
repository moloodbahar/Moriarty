"""
run_prefix_sweep.py — EXPLORATORY prefix sweep (labeled as such; see
DECISIONS.md entry dated before this run).

Question: does the latent-first deficit have a premature-commitment
signature? Pre-stated metrics (computed by analyze_sweep.py):
  recovery  = P(correct at first post-branch prefix | wrong at last
              pre-branch prefix)
  lock-in   = P(same wrong choice repeated post-branch | wrong pre-branch)
Prediction on record: latent_first shows lower recovery and higher
lock-in than cot_matched.

Design:
  - Same 23 usable episodes; Level-2 masking (visibility rule from the
    core report — the regime where the LF deficit was strongest).
  - Prefixes 2..6 in ONE run, so all cells are internally paired
    (never compare across runs: both APIs are nondeterministic at T=0).
  - Lean output (goal choice + confidence + reasoning); no core MCQs —
    this sweep measures the goal-belief trajectory only.
  - Candidate order fixed per episode across all prefixes/conditions/
    models, so choice indices are comparable along the trajectory.

Cost: 23 episodes x 5 prefixes x 3 conditions x 2 models = 690 calls.

Usage:
    python run_prefix_sweep.py episodes_v3.json checks_report_v3.json \
        core_inferability_report.json --out sweep_v1.json
"""

from __future__ import annotations

import argparse
import json
import random

from judges import sha256_file, utc_now, load_episodes
from gemini_client import make_client

CONDITIONS = ("direct", "cot_matched", "latent_first")

BASE_SYSTEM = """You are an observer analyzing an unfolding story. You see \
only the opening. Some characters' moral cores (behavioral rules) are \
unknown to you. You will predict the author's hidden goal."""

REASONING_LINE = {
    "direct":
        "Answer immediately, without writing out reasoning.",
    "cot_matched":
        "First think step by step about the story's events, characters, and "
        "where it is heading. Reason carefully and at length before answering.",
    "latent_first":
        "The story has a single hidden AUTHORIAL goal: one character is being "
        "covertly maneuvered toward some outcome. First infer, from the events "
        "so far, WHO is being maneuvered and TOWARD WHAT outcome. Reason "
        "carefully and at length, state your best hypothesis about the "
        "author's hidden goal, then answer.",
}

OUTPUT_SPEC = """Then respond with a JSON object:
{"reasoning": "<your reasoning; empty string if answering directly>", \
"goal_choice_index": <0-based index into CANDIDATE GOALS>, \
"confidence": <0.0-1.0, your confidence in the goal choice>}"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("episodes")
    p.add_argument("report")
    p.add_argument("core_report")
    p.add_argument("--prefixes", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    p.add_argument("--models", nargs="+",
                   default=["gpt-4o-mini", "gemini-2.5-flash"])
    p.add_argument("--out", default="sweep_v1.json")
    p.add_argument("--log", default=None)
    p.add_argument("--seed", type=int, default=71)
    args = p.parse_args()

    episodes = {e.episode_id: e for e in load_episodes(args.episodes)}
    with open(args.report, "r", encoding="utf-8") as f:
        keep = [r["episode_id"] for r in json.load(f)["results"] if r["usable"]]
    with open(args.core_report, "r", encoding="utf-8") as f:
        core_rep = json.load(f)
    core_acc = {r["episode_id"]: {c["character"]: c["accuracy"]
                                  for c in r["characters"]}
                for r in core_rep["results"]}

    log_path = args.log or args.out.rsplit(".", 1)[0] + "_calls.jsonl"
    clients = {m: make_client(m, log_path=log_path) for m in args.models}
    rng = random.Random(args.seed)

    records = []
    for eid in keep:
        ep = episodes[eid]
        names = [c.name for c in ep.characters]
        visible = min(names, key=lambda n: core_acc[eid][n])

        goal_cands = [ep.hidden_goal] + list(ep.meta.get("distractor_goals", []))
        rng.shuffle(goal_cands)
        true_index = goal_cands.index(ep.hidden_goal)
        goal_listing = "\n".join(f"[{i}] {g}" for i, g in enumerate(goal_cands))

        char_lines = "\n".join(
            f"- {c.name}: moral core = "
            + (c.moral_core if c.name == visible
               else "(unknown — infer from behavior)")
            for c in ep.characters)

        for prefix_len in args.prefixes:
            if prefix_len > len(ep.steps):
                continue
            prefix_text = "\n".join(f"[step {i+1}] {s}"
                                    for i, s in enumerate(ep.steps[:prefix_len]))
            for model in args.models:
                client = clients[model]
                for cond in CONDITIONS:
                    user = (f"STORY OPENING:\n{prefix_text}\n\n"
                            f"CHARACTERS:\n{char_lines}\n\n"
                            f"CANDIDATE GOALS:\n{goal_listing}\n\n"
                            f"{REASONING_LINE[cond]}\n\n{OUTPUT_SPEC}")
                    out = client.json_call(BASE_SYSTEM, user, temperature=0.0,
                                           purpose=f"sweep_t{prefix_len}_{cond}")
                    try:
                        choice = int(out.get("goal_choice_index", -1))
                    except (TypeError, ValueError):
                        choice = -1
                    reasoning = out.get("reasoning", "") or ""
                    records.append({
                        "episode_id": eid,
                        "model": model,
                        "condition": cond,
                        "prefix_len": prefix_len,
                        "candidates": goal_cands,
                        "true_index": true_index,
                        "choice_index": choice,
                        "goal_correct": choice == true_index,
                        "confidence": out.get("confidence"),
                        "reasoning": reasoning,
                        "reasoning_len_chars": len(reasoning),
                    })
                    mark = "+" if choice == true_index else "."
                    print(f"{mark} {eid} t={prefix_len} | {model} | {cond}")

    result = {
        "run_config": {
            "created_at": utc_now(),
            "label": "EXPLORATORY prefix sweep (see DECISIONS.md)",
            "episodes_path": args.episodes,
            "episodes_sha256": sha256_file(args.episodes),
            "core_report_sha256": sha256_file(args.core_report),
            "level": "L2_partial_observability",
            "prefixes": args.prefixes,
            "models": args.models,
            "conditions": list(CONDITIONS),
            "base_system": BASE_SYSTEM,
            "reasoning_lines": REASONING_LINE,
            "rng_seed": args.seed,
            "call_log": log_path,
        },
        "records": records,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n{len(records)} records -> {args.out}\nCalls -> {log_path}")
    print("Analyze with: python analyze_sweep.py "
          f"{args.out} branch_points_v3.json")


if __name__ == "__main__":
    main()
