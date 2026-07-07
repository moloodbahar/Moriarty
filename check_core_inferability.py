"""
check_core_inferability.py — Check 3: are moral cores inferable from behavior?

v4 (Level-2 partial observability) hides 2 of 3 characters' moral cores
from Agent B and SCORES B on inferring them. The masking rule from
DECISIONS.md applies: a masked-and-scored variable must be verified
inferable, else the benchmark scores an unrecoverable quantity.

For every character in every episode, a naive judge sees the FULL story
and picks that character's moral core from a 4-way MCQ (true core + 3
distractors from a generic core bank). Gate per character: accuracy >=
core_min (default 0.6; chance 0.25).

The report feeds run_predictors_v4.py, which uses a deterministic rule:
the LEAST-inferable character's core is the visible one (so the hidden,
scored cores are the inferable ones); any hidden core below the gate is
marked not_scored (irreducible context, never graded).

Usage:
    python check_core_inferability.py episodes_v3.json \
        --report checks_report_v3.json --out core_inferability_report.json
"""

from __future__ import annotations

import argparse
import json
import random

from judges import LLMClient, load_episodes, sha256_file, sha256_text, utc_now

# Generic, world-agnostic behavioral maxims used as MCQ distractors.
CORE_BANK = [
    "Always repay kindness twice over.",
    "Never make a promise you might have to break.",
    "Keep every conversation light; avoid all conflict.",
    "Take charge of any group you find yourself in.",
    "Never reveal what you truly want.",
    "Put efficiency above everyone's feelings.",
    "Always side with the weakest person in the room.",
    "Never trust information you did not verify yourself.",
    "Avoid being the center of attention at any cost.",
    "Never forgive a public slight.",
    "Say whatever keeps everyone calm, true or not.",
    "Follow the rules exactly, whoever gets hurt.",
    "Never ask for help, whatever it costs.",
    "Treat every stranger as a future ally.",
]

CORE_SYSTEM = """You are given a complete short story and the name of one \
character. From the character's BEHAVIOR in the story, pick the moral core \
(behavioral rule) that best explains how they act.

Respond with a JSON object:
{"choice_index": <0-based integer index into the candidate list>, \
"rationale": "<one sentence>"}"""


def core_trials(client, ep, character, n_trials, rng):
    story = "\n".join(f"[step {i+1}] {s}" for i, s in enumerate(ep.steps))
    true_core = character.moral_core
    other_cores = {c.moral_core for c in ep.characters}
    bank = [b for b in CORE_BANK if b not in other_cores]
    trials = []
    for _ in range(n_trials):
        candidates = [true_core] + rng.sample(bank, 3)
        rng.shuffle(candidates)
        true_index = candidates.index(true_core)
        listing = "\n".join(f"[{i}] {c}" for i, c in enumerate(candidates))
        user = (f"STORY:\n{story}\n\nCHARACTER: {character.name}\n\n"
                f"CANDIDATE MORAL CORES:\n{listing}\n\nWhich fits best?")
        out = client.json_call(CORE_SYSTEM, user, purpose="core_inferability_mcq")
        try:
            chosen = int(out.get("choice_index", -1))
        except (TypeError, ValueError):
            chosen = -1
        trials.append({"candidates": candidates, "true_index": true_index,
                       "chosen_index": chosen, "correct": chosen == true_index,
                       "rationale": out.get("rationale", "")})
    acc = sum(t["correct"] for t in trials) / n_trials
    return {"character": character.name, "accuracy": acc, "trials": trials}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("episodes")
    p.add_argument("--report", default=None,
                   help="checks report; if given, only usable episodes are checked")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--n-trials", type=int, default=5)
    p.add_argument("--core-min", type=float, default=0.6)
    p.add_argument("--out", default="core_inferability_report.json")
    p.add_argument("--log", default=None)
    p.add_argument("--seed", type=int, default=53)
    args = p.parse_args()

    episodes = load_episodes(args.episodes)
    if args.report:
        with open(args.report, "r", encoding="utf-8") as f:
            rep = json.load(f)
        keep = {r["episode_id"] for r in rep["results"] if r["usable"]}
        episodes = [e for e in episodes if e.episode_id in keep]

    log_path = args.log or args.out.rsplit(".", 1)[0] + "_calls.jsonl"
    client = LLMClient(model=args.model, log_path=log_path)
    rng = random.Random(args.seed)

    results = []
    for ep in episodes:
        chars = [core_trials(client, ep, c, args.n_trials, rng)
                 for c in ep.characters]
        for c in chars:
            c["inferable"] = c["accuracy"] >= args.core_min
        results.append({"episode_id": ep.episode_id, "characters": chars})
        line = " ".join(f"{c['character']}={c['accuracy']:.2f}"
                        f"{'' if c['inferable'] else '(LOW)'}" for c in chars)
        print(f"{ep.episode_id}: {line}")

    n_low = sum(1 for r in results for c in r["characters"] if not c["inferable"])
    report = {
        "run_config": {
            "created_at": utc_now(),
            "episodes_path": args.episodes,
            "episodes_sha256": sha256_file(args.episodes),
            "judge_model": args.model,
            "n_trials": args.n_trials,
            "core_min": args.core_min,
            "rng_seed": args.seed,
            "call_log": log_path,
            "core_prompt_sha256": sha256_text(CORE_SYSTEM),
            "core_bank": CORE_BANK,
        },
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nCharacters below inferability gate: {n_low}")
    print(f"Report -> {args.out}\nCalls -> {log_path}")


if __name__ == "__main__":
    main()
