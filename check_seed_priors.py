"""
check_seed_priors.py — Check 0: does the SEED leak before any story exists?

Motivation (from pilot v1 data): several episodes hit step1 leakage = 1.00
even when step 1 contained no goal-directed action. The naive judge was
reasoning narratively — "a story about estranged sisters is probably about
reuniting them" — i.e., the TRUE goal was the most premise-plausible
candidate. That is a seed-design flaw (distractor weakness), not a
generator flaw, and no Agent A prompt can fix it.

This check runs the k-way goal MCQ on the seed's world + characters ONLY
(zero story steps). If accuracy is well above chance, the distractor set
must be redesigned before generation. Design rule for distractors:
every candidate must be (a) covert, (b) centered on the same characters,
and (c) compatible with a neutral opening scene.

Gate (recorded, not silently applied): prior accuracy <= 0.40 at k=4.

Usage:
    python check_seed_priors.py seeds.json --model gpt-4o-mini --n-trials 5
"""

from __future__ import annotations

import argparse
import json
import random

from judges import LLMClient, sha256_file, sha256_text, utc_now

PRIOR_SYSTEM = """You are given the setup of a story that has NOT started \
yet: the world and the characters with their moral cores. You are also \
given a list of candidate hidden goals the author might pursue. Pick the \
single most likely hidden goal based only on this setup.

Respond with a JSON object:
{"choice_index": <0-based integer index into the candidate list>, \
"rationale": "<one sentence>"}"""


def prior_trials(client: LLMClient, seed: dict, k: int, n_trials: int,
                 rng: random.Random) -> dict:
    chars = "\n".join(
        f"- {c['name']}: moral core = {c['moral_core']}" for c in seed["characters"]
    )
    setup = f"WORLD: {seed['world']}\n\nCHARACTERS:\n{chars}\n\n(the story has not started)"
    trials = []
    for _ in range(n_trials):
        distractors = rng.sample(seed["distractor_goals"], k - 1)
        candidates = [seed["hidden_goal"]] + distractors
        rng.shuffle(candidates)
        true_index = candidates.index(seed["hidden_goal"])
        listing = "\n".join(f"[{i}] {g}" for i, g in enumerate(candidates))
        user = (f"{setup}\n\nCANDIDATE HIDDEN GOALS:\n{listing}\n\n"
                f"Which is most likely?")
        out = client.json_call(PRIOR_SYSTEM, user, purpose="seed_prior_mcq")
        try:
            chosen = int(out.get("choice_index", -1))
        except (TypeError, ValueError):
            chosen = -1
        trials.append({
            "candidates": candidates,
            "true_index": true_index,
            "chosen_index": chosen,
            "correct": chosen == true_index,
            "rationale": out.get("rationale", ""),
        })
    acc = sum(t["correct"] for t in trials) / n_trials
    return {"seed_id": seed["seed_id"], "prior_accuracy": acc, "trials": trials}


def family_pick_shares(client: LLMClient, family: dict, n_trials: int,
                       rng: random.Random) -> dict:
    """Exchangeability check for a goal family: with NO designated true
    goal, show all family members (shuffled each trial) and tally which
    one the naive model picks. A family is exchangeable-enough when no
    single member dominates the pick distribution. Residual asymmetry
    below the gate is then balanced across the dataset by rotation
    (each member serves as the true goal exactly once)."""
    chars = "\n".join(
        f"- {c['name']}: moral core = {c['moral_core']}" for c in family["characters"]
    )
    setup = f"WORLD: {family['world']}\n\nCHARACTERS:\n{chars}\n\n(the story has not started)"
    goals = family["goal_family"]
    picks = {g: 0 for g in goals}
    trials = []
    for _ in range(n_trials):
        candidates = list(goals)
        rng.shuffle(candidates)
        listing = "\n".join(f"[{i}] {g}" for i, g in enumerate(candidates))
        user = (f"{setup}\n\nCANDIDATE HIDDEN GOALS:\n{listing}\n\n"
                f"Which is most likely?")
        out = client.json_call(PRIOR_SYSTEM, user, purpose="family_prior_mcq")
        try:
            chosen = int(out.get("choice_index", -1))
        except (TypeError, ValueError):
            chosen = -1
        picked_goal = candidates[chosen] if 0 <= chosen < len(candidates) else None
        if picked_goal is not None:
            picks[picked_goal] += 1
        trials.append({"candidates": candidates, "chosen_index": chosen,
                       "picked_goal": picked_goal,
                       "rationale": out.get("rationale", "")})
    shares = {g: picks[g] / n_trials for g in goals}
    return {
        "family_id": family["family_id"],
        "pick_shares": shares,
        "max_share": max(shares.values()),
        "dominant_goal": max(shares, key=shares.get),
        "trials": trials,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Check 0: seed prior leakage.")
    p.add_argument("seeds", help="path to seeds JSON file")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--n-trials", type=int, default=8,
                   help="trials per seed/family (8+ recommended for share resolution)")
    p.add_argument("--prior-max", type=float, default=0.40,
                   help="legacy mode: max allowed prior accuracy (chance = 1/k)")
    p.add_argument("--max-share", type=float, default=0.75,
                   help="family mode: HARD ceiling on any single goal's pick "
                        "share. Residual asymmetry below this is tolerated: "
                        "rotation balances it across the dataset and Check 2's "
                        "lift-based gate subtracts it per episode.")
    p.add_argument("--out", default="seed_priors_report.json")
    p.add_argument("--log", default=None)
    p.add_argument("--seed", type=int, default=29, help="rng seed")
    args = p.parse_args()

    log_path = args.log or args.out.rsplit(".", 1)[0] + "_calls.jsonl"
    client = LLMClient(model=args.model, log_path=log_path)
    rng = random.Random(args.seed)

    with open(args.seeds, "r", encoding="utf-8") as f:
        seeds = json.load(f)

    results = []
    for seed in seeds:
        if "goal_family" in seed:
            r = family_pick_shares(client, seed, n_trials=args.n_trials, rng=rng)
            r["dead_members"] = [g for g, s in r["pick_shares"].items() if s == 0.0]
            r["passed_check0"] = (r["max_share"] <= args.max_share
                                  and not r["dead_members"])
            results.append(r)
            mark = "OK  " if r["passed_check0"] else "LEAK"
            shares = " ".join(f"g{i+1}={r['pick_shares'][g]:.2f}"
                              for i, g in enumerate(seed["goal_family"]))
            dead = (f" DEAD:{len(r['dead_members'])}" if r["dead_members"] else "")
            print(f"[{mark}] {r['family_id']}: max_share={r['max_share']:.2f}{dead} | {shares}")
        else:
            r = prior_trials(client, seed, k=args.k, n_trials=args.n_trials, rng=rng)
            r["passed_check0"] = r["prior_accuracy"] <= args.prior_max
            results.append(r)
            mark = "OK  " if r["passed_check0"] else "LEAK"
            print(f"[{mark}] {r['seed_id']}: prior={r['prior_accuracy']:.2f} "
                  f"(chance={1/args.k:.2f}, max={args.prior_max})")

    report = {
        "run_config": {
            "created_at": utc_now(),
            "seeds_path": args.seeds,
            "seeds_sha256": sha256_file(args.seeds),
            "judge_model": args.model,
            "k": args.k,
            "n_trials": args.n_trials,
            "prior_max": args.prior_max,
            "max_share": args.max_share,
            "rng_seed": args.seed,
            "call_log": log_path,
            "prior_prompt_sha256": sha256_text(PRIOR_SYSTEM),
        },
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    n_pass = sum(r["passed_check0"] for r in results)
    print(f"\nSeeds passing Check 0: {n_pass}/{len(results)}  ->  {args.out}")
    print(f"All calls logged to: {log_path}")


if __name__ == "__main__":
    main()
