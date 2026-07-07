"""
run_predictors.py — Stage 1 predictor experiment for Moriarty.

Tests H1: does Latent-First prediction beat CoT-Matched (reasoning-length
matched) on predicting Agent A's behavior over the horizon, paired per
episode, cross-model?

Design:
  - Only USABLE episodes (from checks_report) are used.
  - Prefix = steps 1..t_star (from the analysis file). B predicts the horizon.
  - PRIMARY output per condition: k-way goal choice over the episode's
    candidate set (exact-graded, McNemar-ready).
  - SECONDARY: free-text "next beat", stored for the calibrated matcher.
  - CONDITIONS: direct / cot_matched (PRIMARY CONTROL) / latent_first /
    latent_first_v2 / latent_struct.
  - latent_struct is the current focus: instead of free-form reasoning, B
    must commit to a STRUCTURED latent state (manipulator, target,
    intended_belief_shift, constraint_used, next action). Free-form
    latent prompts produce fluent reasoning that is not forced to
    represent the manipulation theory; the structured schema is, and each
    field is graded separately by grade_latents.py.
  - MODELS: gpt-4o-mini (same-model as generator), gemini-2.5-flash (cross).
  - Candidate ordering fixed per episode -> all cells strictly paired.

Usage (structured-latent stage; CoT comparison deferred):
    python run_predictors.py episodes_v3.json checks_report_v3.json \
        checks_report_v3_analysis.json --out predictions_v3.json
    python grade_latents.py episodes_v3.json predictions_v3.json

Full condition sweep (later, for the paired H1 test):
    python run_predictors.py ... --conditions direct cot_matched \
        latent_first latent_first_v2 latent_struct
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Optional

from judges import sha256_file, utc_now, load_episodes
from gemini_client import make_client


CONDITIONS = ("direct", "cot_matched", "latent_first", "latent_first_v2",
              "latent_struct")

# Fields of the structured latent state (latent_struct condition).
LATENT_FIELDS = ("manipulator", "target", "intended_belief_shift",
                 "constraint_used", "next_action_that_preserves_constraint")

BASE_SYSTEM = """You are an observer analyzing an unfolding story. You see \
only the opening. You will be given the list of candidate hidden goals the \
author might be pursuing, and you must predict which one it is and what \
happens next. You cannot see the author's plan; infer from the text."""

REASONING_LINE = {
    "direct":
        "Answer immediately, without writing out reasoning.",
    "cot_matched":
        "First think step by step about the story's events, characters, and "
        "where it is heading. Reason carefully and at length before answering.",
    "latent_first":
        "First explicitly infer each character's hidden goal and moral "
        "constraints from their behavior, stating what latent state best "
        "explains the events. Reason carefully and at length. Then answer.",
    "latent_first_v2":
        "The story has a single hidden AUTHORIAL goal: one character is being "
        "covertly maneuvered toward some outcome. First infer, from the events "
        "so far, WHO is being maneuvered and TOWARD WHAT outcome. Treat the "
        "stated moral cores as fixed constraints on how each character can be "
        "moved — do not re-derive them. Reason carefully and at length, state "
        "your best hypothesis about the author's hidden goal, then answer.",
    "latent_struct":
        "The story has a single hidden AUTHORIAL goal: one character is being "
        "covertly maneuvered toward some outcome. Commit to a structured "
        "hypothesis about that manipulation. Every field is mandatory and "
        "must name concrete characters/beliefs from THIS story — no hedging, "
        "no 'unclear'.",
}

OUTPUT_SPEC = """Then respond with a JSON object:
{{"reasoning": "<your reasoning; empty string if answering directly>", \
"goal_choice_index": <0-based index into the CANDIDATE GOALS list>, \
"next_beat": "<one to three sentences predicting the very next story step>"}}"""

# latent_struct: the latent state is not free-form prose but a committed,
# field-by-field theory of the manipulation. Each field is separately
# gradable against ground truth (hidden goal + private rationales).
LATENT_STRUCT_OUTPUT_SPEC = """Respond with a JSON object:
{{"manipulator": "<character name driving the covert maneuver, or AUTHOR \
if the maneuvering is done by the story itself rather than one character>", \
"target": "<character name being maneuvered>", \
"intended_belief_shift": "<one sentence: what the target currently \
believes/intends, and what the manipulator needs them to believe/intend \
instead>", \
"constraint_used": "<which character's moral core the maneuver exploits or \
routes around, quoted or closely paraphrased>", \
"next_action_that_preserves_constraint": "<one to three sentences: the very \
next story step, chosen so that it advances the belief shift WITHOUT any \
character violating their moral core>", \
"goal_choice_index": <0-based index into the CANDIDATE GOALS list>}}"""


def build_user(prefix_text: str, characters: list, candidates: list,
               cond: str) -> str:
    def cname(c): return c["name"] if isinstance(c, dict) else c.name
    def ccore(c): return c["moral_core"] if isinstance(c, dict) else c.moral_core
    chars = "\n".join(f"- {cname(c)}: moral core = {ccore(c)}"
                      for c in characters)
    listing = "\n".join(f"[{i}] {g}" for i, g in enumerate(candidates))
    spec = (LATENT_STRUCT_OUTPUT_SPEC if cond == "latent_struct"
            else OUTPUT_SPEC)
    return (
        f"STORY OPENING:\n{prefix_text}\n\n"
        f"CHARACTERS:\n{chars}\n\n"
        f"CANDIDATE HIDDEN GOALS:\n{listing}\n\n"
        f"{REASONING_LINE[cond]}\n\n"
        f"{spec.format()}"
    )


def usable_ids(report_path: str) -> list:
    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    return [r["episode_id"] for r in rep["results"] if r["usable"]]


def run(episodes_path: str, report_path: str, analysis_path: str,
        models: list, out_path: str, log_path: Optional[str],
        conditions: Optional[list] = None, rng_seed: int = 41) -> dict:
    conditions = list(conditions or CONDITIONS)
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown conditions: {unknown}; valid: {CONDITIONS}")
    episodes = {e.episode_id: e for e in load_episodes(episodes_path)}
    keep = usable_ids(report_path)
    with open(analysis_path, "r", encoding="utf-8") as f:
        analysis = json.load(f)
    t_star = analysis["t_star"]
    if t_star is None:
        raise SystemExit("analysis has no t_star; cannot set prefix/horizon.")

    log_path = log_path or out_path.rsplit(".", 1)[0] + "_calls.jsonl"
    clients = {m: make_client(m, log_path=log_path) for m in models}
    rng = random.Random(rng_seed)

    records = []
    for eid in keep:
        ep = episodes[eid]
        prefix_text = "\n".join(f"[step {i+1}] {s}"
                                for i, s in enumerate(ep.steps[:t_star]))
        actual_next = ep.steps[t_star] if t_star < len(ep.steps) else ""
        candidates = [ep.hidden_goal] + list(ep.meta.get("distractor_goals", []))
        rng.shuffle(candidates)
        true_index = candidates.index(ep.hidden_goal)

        for model in models:
            client = clients[model]
            for cond in conditions:
                user = build_user(prefix_text, ep.characters, candidates, cond)
                out = client.json_call(BASE_SYSTEM, user, temperature=0.0,
                                       purpose=f"predict_{cond}")
                try:
                    choice = int(out.get("goal_choice_index", -1))
                except (TypeError, ValueError):
                    choice = -1
                reasoning = out.get("reasoning", "") or ""
                rec = {
                    "episode_id": eid,
                    "family_id": ep.meta.get("family_id"),
                    "goal_index": ep.meta.get("goal_index"),
                    "model": model,
                    "same_model": not model.startswith("gemini"),
                    "condition": cond,
                    "candidates": candidates,
                    "true_index": true_index,
                    "choice_index": choice,
                    "goal_correct": choice == true_index,
                    "reasoning": reasoning,
                    "reasoning_len_chars": len(reasoning),
                    "next_beat": out.get("next_beat", ""),
                    "actual_next_step": actual_next,
                }
                if cond == "latent_struct":
                    # The committed latent state, field by field. The next
                    # action doubles as the next-beat prediction.
                    rec["latent"] = {f: (out.get(f, "") or "").strip()
                                     for f in LATENT_FIELDS}
                    rec["next_beat"] = rec["latent"][
                        "next_action_that_preserves_constraint"]
                records.append(rec)
                mark = "+" if choice == true_index else "."
                print(f"{mark} {eid} | {model} | {cond} "
                      f"(rlen={len(reasoning)})")

    result = {
        "run_config": {
            "created_at": utc_now(),
            "episodes_path": episodes_path,
            "episodes_sha256": sha256_file(episodes_path),
            "report_path": report_path,
            "analysis_path": analysis_path,
            "t_star": t_star, "horizon": f"{t_star+1}..{analysis['T']}",
            "models": models,
            "conditions": conditions,
            "latent_struct_output_spec": LATENT_STRUCT_OUTPUT_SPEC,
            "n_usable_episodes": len(keep),
            "base_system": BASE_SYSTEM,
            "reasoning_lines": REASONING_LINE,
            "call_log": log_path,
            "rng_seed": rng_seed,
            "note": "goal_correct is exact vs ground truth (no matcher). "
                    "next_beat requires the calibrated matcher before use.",
        },
        "records": records,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    preview_paired(records)
    print(f"\nPredictions -> {out_path}\nCalls -> {log_path}")
    if "latent_struct" in conditions:
        print("NEXT: python grade_latents.py "
              f"{episodes_path} {out_path}  (per-field latent grading)")
    else:
        print("NEXT: calibrate the continuation matcher, then compute McNemar "
              "on latent_first vs cot_matched (cross-model).")
    return result


def preview_paired(records: list) -> None:
    """Console preview only — the real test is McNemar in the analyzer."""
    from collections import defaultdict
    acc = defaultdict(lambda: [0, 0])
    for r in records:
        key = (r["model"], r["condition"])
        acc[key][1] += 1
        acc[key][0] += int(r["goal_correct"])
    print("\nGOAL-CHOICE ACCURACY (preview, not the final metric):")
    for key in sorted(acc):
        c, n = acc[key]
        print(f"  {key[0]:20s} {key[1]:13s}: {c}/{n} = {c/n:.2f}")

    idx = defaultdict(dict)
    for r in records:
        idx[(r["model"], r["episode_id"])][r["condition"]] = r["goal_correct"]
    print("\nPAIRED latent_first vs cot_matched (goal choice):")
    by_model = defaultdict(lambda: [0, 0, 0, 0])
    for (model, _eid), d in idx.items():
        if "latent_first" in d and "cot_matched" in d:
            lf, cot = d["latent_first"], d["cot_matched"]
            cell = by_model[model]
            if lf and not cot: cell[0] += 1
            elif cot and not lf: cell[1] += 1
            elif lf and cot: cell[2] += 1
            else: cell[3] += 1
    for model, (lf_win, cot_win, both, neither) in by_model.items():
        disc = lf_win + cot_win
        print(f"  {model}: LF-only={lf_win} CoT-only={cot_win} "
              f"both={both} neither={neither} (discordant={disc})")
        if disc < 10:
            print("    (too few discordant pairs for a meaningful McNemar at "
                  "pilot scale — direction check only.)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("episodes")
    p.add_argument("report")
    p.add_argument("analysis")
    p.add_argument("--models", nargs="+", default=["gpt-4o-mini", "gemini-2.5-flash"])
    p.add_argument("--conditions", nargs="+", default=["latent_struct"],
                   help=f"subset of {CONDITIONS}. Default: latent_struct only "
                        "— get the structured latent representation right "
                        "before spending calls on the CoT comparison.")
    p.add_argument("--out", default="predictions_v3.json")
    p.add_argument("--log", default=None)
    args = p.parse_args()
    run(args.episodes, args.report, args.analysis, args.models,
        args.out, args.log, conditions=args.conditions)