"""
generate_episodes.py — Agent A produces episodes from hand-designed seeds.

Why the episodes are generated, not hand-written:
  The hypothesis is about predicting another AGENT's decision process.
  Hand-written stories would make B model the human author, not Agent A.
  Humans design only the ground-truth latent state z (seed spec); Agent A
  turns z into behavior.

Why generation is step-by-step, one API call per step:
  If A wrote all steps in a single call, there would be no sequential
  decision process for B to model — the whole trajectory would be one
  decision. Per-step generation gives A a genuine state at each step,
  conditioned on its own previous output.

Each step call returns:
  {"private_rationale": ..., "step": ...}
  - private_rationale is A's stated reason for the move. It is stored in
    meta["private_rationales"] and MUST NEVER be shown to Agent B or to
    the leakage judge. It is the (imperfect, faithfulness-caveated)
    reference for the secondary reasoning-match metric only.
  - step is the visible story text; A is instructed never to state the
    hidden goal explicitly. Episodes that leak anyway are caught by
    Check 2, not trusted to prompting.

Usage:
    export OPENAI_API_KEY=...
    python generate_episodes.py seeds.json --model gpt-4o-mini \
        --temperature 0.8 --out episodes.json
Then validate:
    python judges.py episodes.json --out checks_report.json
"""

from __future__ import annotations

import argparse
import json

from judges import LLMClient, sha256_file, sha256_text, utc_now

# Bump this whenever A_SYSTEM_TEMPLATE changes. Recorded in every
# episode's meta so leakage/consistency stats are traceable per prompt.
# v1 -> v2: pilot v1 showed step-1 leakage in 9/10 episodes because A's
# opening moves were goal-directed. v2 shows A the alternative goals and
# imposes an explicit ambiguity pacing schedule.
GENERATOR_PROMPT_VERSION = "A_v2_distractor_aware_paced"


A_SYSTEM_TEMPLATE = """You are Agent A, the author of an unfolding story. \
You have a HIDDEN GOAL that the entire story must serve.

WORLD: {world}

CHARACTERS (each must always act consistently with their moral core):
{characters}

YOUR HIDDEN GOAL: {hidden_goal}

An observer will try to guess your hidden goal. These are the ALTERNATIVE \
GOALS the observer will be choosing between, alongside your real one:
{distractors}

Rules:
- Advance the story by exactly ONE step (2-4 sentences) per turn.
- NEVER state or paraphrase the hidden goal in the story text itself.
- Characters must never violate their moral cores without narrative \
justification.

Ambiguity pacing (strict):
- Steps 1-2: pure scene-setting and character establishment. Every event \
must be EQUALLY consistent with the hidden goal AND with every alternative \
goal above. Take no action that favors one candidate over the others.
- Middle steps: narrow gradually. Misdirection that points toward the \
alternative goals is encouraged.
- Final step: the hidden goal must be reached or clearly within reach, and \
the events must now clearly distinguish it from EVERY alternative above.

Respond with a JSON object:
{{"private_rationale": "<1-2 sentences: why this move serves the hidden goal>", \
"step": "<the story step text>"}}"""


def expand_seeds(seeds: list[dict], rotations: str = "all") -> list[dict]:
    """Expand goal-family seeds into per-goal episode specs (rotation).

    Family schema: {family_id, world, characters, goal_family: [g1..gk], n_steps}
    Each rotation instantiates goal i as hidden_goal with the remaining
    family members as distractors, so no candidate is privileged by
    construction and residual premise-affinity balances out across the
    dataset. Legacy specs (with hidden_goal/distractor_goals) pass through.
    rotations: "all" or an integer count of randomly chosen rotations.
    """
    import random as _random
    rng = _random.Random(7)
    out = []
    for seed in seeds:
        if "goal_family" not in seed:
            out.append(seed)
            continue
        goals = seed["goal_family"]
        idxs = list(range(len(goals)))
        if rotations != "all":
            idxs = sorted(rng.sample(idxs, min(int(rotations), len(idxs))))
        for i in idxs:
            out.append({
                "seed_id": f"{seed['family_id']}_g{i + 1}",
                "family_id": seed["family_id"],
                "goal_index": i,
                "world": seed["world"],
                "characters": seed["characters"],
                "hidden_goal": goals[i],
                "distractor_goals": [g for j, g in enumerate(goals) if j != i],
                "n_steps": seed["n_steps"],
            })
    return out


def generate_episode(client: LLMClient, seed: dict, temperature: float) -> dict:
    chars = "\n".join(
        f"- {c['name']}: moral core = {c['moral_core']}" for c in seed["characters"]
    )
    distractors = "\n".join(f"- {g}" for g in seed["distractor_goals"])
    system = A_SYSTEM_TEMPLATE.format(
        world=seed["world"], characters=chars, hidden_goal=seed["hidden_goal"],
        distractors=distractors,
    )
    steps: list[str] = []
    rationales: list[str] = []
    for i in range(seed["n_steps"]):
        story_so_far = "\n".join(
            f"[step {j + 1}] {s}" for j, s in enumerate(steps)
        ) or "(the story has not started yet)"
        remaining = seed["n_steps"] - i
        user = (
            f"STORY SO FAR:\n{story_so_far}\n\n"
            f"Steps remaining including this one: {remaining}. "
            + ("This is the FINAL step: the hidden goal should now be "
               "reached or clearly within reach. " if remaining == 1 else "")
            + "Write the next step."
        )
        out = client.json_call(system, user, temperature=temperature,
                               purpose="agent_a_step")
        step_text = str(out.get("step", "")).strip()
        if not step_text:
            raise RuntimeError(f"{seed['seed_id']}: empty step at index {i}")
        steps.append(step_text)
        rationales.append(str(out.get("private_rationale", "")).strip())

    return {
        "episode_id": seed["seed_id"],
        "characters": seed["characters"],
        "hidden_goal": seed["hidden_goal"],
        "steps": steps,
        "meta": {
            "world": seed["world"],
            "distractor_goals": seed["distractor_goals"],
            "family_id": seed.get("family_id"),
            "goal_index": seed.get("goal_index"),
            "generator_model": client.model,
            "generator_prompt_version": GENERATOR_PROMPT_VERSION,
            "generator_template_sha256": sha256_text(A_SYSTEM_TEMPLATE),
            "generated_at": utc_now(),
            "temperature": temperature,
            # NEVER expose to Agent B or to the leakage judge:
            "private_rationales": rationales,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Generate episodes from seed specs via Agent A.")
    p.add_argument("seeds", help="path to seeds.json")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--out", default="episodes.json")
    p.add_argument("--log", default=None,
                   help="JSONL call log path (default: <out>_calls.jsonl)")
    p.add_argument("--per-seed", type=int, default=1,
                   help="episodes to generate per seed (use >1 later for the real run)")
    p.add_argument("--rotations", default="all",
                   help="for goal-family seeds: 'all' or an integer count")
    args = p.parse_args()

    with open(args.seeds, "r", encoding="utf-8") as f:
        raw_seeds = json.load(f)
    seeds = expand_seeds(raw_seeds, rotations=args.rotations)

    log_path = args.log or args.out.rsplit(".", 1)[0] + "_calls.jsonl"
    client = LLMClient(model=args.model, log_path=log_path)
    episodes = []
    for seed in seeds:
        for rep in range(args.per_seed):
            ep = generate_episode(client, seed, temperature=args.temperature)
            if args.per_seed > 1:
                ep["episode_id"] = f"{seed['seed_id']}_r{rep + 1}"
            episodes.append(ep)
            print(f"generated {ep['episode_id']} ({len(ep['steps'])} steps)")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(episodes, f, ensure_ascii=False, indent=2)

    # Run manifest: everything needed to trace this dataset later,
    # including the full prompt template text (not just its version tag).
    manifest = {
        "created_at": utc_now(),
        "seeds_path": args.seeds,
        "seeds_sha256": sha256_file(args.seeds),
        "out_path": args.out,
        "episodes_sha256": sha256_file(args.out),
        "generator_model": args.model,
        "temperature": args.temperature,
        "per_seed": args.per_seed,
        "rotations": args.rotations,
        "generator_prompt_version": GENERATOR_PROMPT_VERSION,
        "generator_template_sha256": sha256_text(A_SYSTEM_TEMPLATE),
        "generator_template_text": A_SYSTEM_TEMPLATE,
        "call_log": log_path,
        "n_episodes": len(episodes),
    }
    manifest_path = args.out.rsplit(".", 1)[0] + ".manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n{len(episodes)} episodes → {args.out}")
    print(f"Manifest → {manifest_path}")
    print(f"All calls logged to: {log_path}")
    print("Next: python judges.py " + args.out)


if __name__ == "__main__":
    main()
