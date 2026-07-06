"""
judges.py — Step 1 of the Latent-First MVP.

Two manipulation-check judges that must exist and be CALIBRATED before
any generator or predictor is built:

  1. ConsistencyJudge  (Check 1): does each story step actually advance /
     stay compatible with Agent A's hidden goal?
  2. LeakageJudge      (Check 2): can a naive model trivially infer the
     hidden goal from the first step alone?

Deliberate design decisions:
  - Consistency is 3-way (ADVANCES / NEUTRAL / CONTRADICTS), not binary.
    A binary judge conflates "irrelevant step" with "goal-violating step",
    and episodes with many neutral steps are legitimate (misdirection).
    Episode passes Check 1 iff: zero CONTRADICTS and advance_fraction >= tau.
  - Leakage is operationalized as k-way multiple choice: the true goal +
    (k-1) distractor goals, shuffled. Distractors come from the episode's
    own meta["distractor_goals"] — plausible SAME-WORLD alternative goals
    designed in the seed spec. Cross-episode goals are only a fallback:
    goals from different domains would make the MCQ solvable by keyword
    matching and fail Check 2 spuriously. Leakage = accuracy above chance.
  - The leakage judge is run on every prefix length, producing an
    *inferability curve*. A healthy episode is low at step 1 and rises
    later: the goal must be inferable EVENTUALLY (else B has no signal),
    just not trivially at the start.
  - temperature=0, JSON mode, raw responses kept for audit.
  - Judges never see Agent B's outputs. They only validate episodes.

Requires: pip install openai
Env: OPENAI_API_KEY (and optionally OPENAI_BASE_URL for other providers).
A `.env` file in the project root or cwd is loaded automatically if present.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

from openai import OpenAI, OpenAIError

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_DOTENV_LOADED = False


def _load_dotenv() -> None:
    """Load OPENAI_* vars from a local .env file if not already set."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    for directory in (_PROJECT_DIR, os.getcwd()):
        path = os.path.join(directory, ".env")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip("\"'")
                if key and key not in os.environ:
                    os.environ[key] = value


def _resolve_openai_key(api_key: Optional[str]) -> str:
    _load_dotenv()
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        env_path = os.path.join(_PROJECT_DIR, ".env")
        raise OpenAIError(
            "Missing OpenAI API key. Set OPENAI_API_KEY in the environment "
            f"or create {env_path} with:\n"
            "  OPENAI_API_KEY=sk-..."
        )
    return key


_load_dotenv()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------
# Episode schema
# ----------------------------------------------------------------------

@dataclass
class Character:
    name: str
    moral_core: str


@dataclass
class Episode:
    episode_id: str
    characters: list[Character]
    hidden_goal: str
    steps: list[str]                     # story steps produced by Agent A
    meta: dict = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict) -> "Episode":
        return Episode(
            episode_id=d["episode_id"],
            characters=[Character(**c) for c in d["characters"]],
            hidden_goal=d["hidden_goal"],
            steps=list(d["steps"]),
            meta=d.get("meta", {}),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def load_episodes(path: str) -> list[Episode]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Episode.from_dict(d) for d in data]


# ----------------------------------------------------------------------
# Minimal LLM client (OpenAI-compatible), JSON mode, temperature 0
# ----------------------------------------------------------------------

class CallLogger:
    """Append-only JSONL log of every model call (request + response).

    This is the primary provenance record: any judge or analysis can be
    re-run against cached responses without re-spending API calls, and
    every reported number traces back to raw calls. One JSON object per
    line; failures are logged too.
    """

    def __init__(self, path: Optional[str]):
        self.path = path

    def log(self, record: dict) -> None:
        if not self.path:
            return
        record = dict(record)
        record["logged_at"] = utc_now()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class LLMClient:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        retry_sleep: float = 2.0,
        log_path: Optional[str] = None,
    ):
        self.model = model
        _load_dotenv()
        self.client = OpenAI(
            api_key=_resolve_openai_key(api_key),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
        )
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.logger = CallLogger(log_path)

    def json_call(self, system: str, user: str, temperature: float = 0.0,
                  purpose: str = "unspecified") -> dict:
        """Call the model expecting a strict JSON object back."""
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                raw = resp.choices[0].message.content
                out = json.loads(raw)
                out["_raw"] = raw  # keep for audit
                self.logger.log({
                    "purpose": purpose,
                    "model": self.model,
                    "temperature": temperature,
                    "attempt": attempt,
                    "system": system,
                    "user": user,
                    "response_raw": raw,
                    "ok": True,
                })
                return out
            except Exception as e:  # noqa: BLE001 — retry on any transient failure
                last_err = e
                time.sleep(self.retry_sleep * (attempt + 1))
        self.logger.log({
            "purpose": purpose,
            "model": self.model,
            "temperature": temperature,
            "system": system,
            "user": user,
            "ok": False,
            "error": str(last_err),
        })
        raise RuntimeError(f"json_call failed after {self.max_retries} tries: {last_err}")


# ----------------------------------------------------------------------
# Check 1: Consistency judge
# ----------------------------------------------------------------------

CONSISTENCY_SYSTEM = """You are a strict evaluation judge for a research \
experiment. You will be shown the HIDDEN GOAL of a story author, the \
characters with their moral cores, the story so far, and ONE new story step.

Classify the new step with exactly one label:

- "ADVANCES": the step plausibly moves the author closer to the hidden goal \
(including indirect moves: misdirection, setup, information control).
- "NEUTRAL": the step neither helps nor harms the hidden goal.
- "CONTRADICTS": the step actively undermines the hidden goal, OR a \
character clearly violates their stated moral core without narrative \
justification.

Be conservative: only use CONTRADICTS when the conflict is clear.
Respond with a JSON object:
{"label": "ADVANCES" | "NEUTRAL" | "CONTRADICTS", "rationale": "<one or two sentences>"}"""


class ConsistencyJudge:
    def __init__(self, client: LLMClient):
        self.client = client

    def judge_step(self, ep: Episode, step_index: int) -> dict:
        chars = "\n".join(f"- {c.name}: moral core = {c.moral_core}" for c in ep.characters)
        story_so_far = "\n".join(
            f"[step {i + 1}] {s}" for i, s in enumerate(ep.steps[:step_index])
        ) or "(story has not started yet)"
        new_step = ep.steps[step_index]
        user = (
            f"HIDDEN GOAL of the author:\n{ep.hidden_goal}\n\n"
            f"CHARACTERS:\n{chars}\n\n"
            f"STORY SO FAR:\n{story_so_far}\n\n"
            f"NEW STEP (step {step_index + 1}):\n{new_step}\n\n"
            f"Classify the NEW STEP."
        )
        out = self.client.json_call(CONSISTENCY_SYSTEM, user,
                                    purpose="consistency_judge")
        raw_label = str(out.get("label", "")).strip().upper()
        invalid_label = raw_label not in {"ADVANCES", "NEUTRAL", "CONTRADICTS"}
        label = "NEUTRAL" if invalid_label else raw_label
        return {
            "episode_id": ep.episode_id,
            "step_index": step_index,
            "label": label,
            "invalid_label": invalid_label,  # judge failure must never be silent
            "raw_label": raw_label,
            "rationale": out.get("rationale", ""),
            "_raw": out.get("_raw", ""),
        }

    def judge_episode(self, ep: Episode, min_advance_fraction: float = 0.4) -> dict:
        results = [self.judge_step(ep, i) for i in range(len(ep.steps))]
        labels = [r["label"] for r in results]
        n = len(labels)
        n_invalid = sum(r["invalid_label"] for r in results)
        advance_frac = labels.count("ADVANCES") / n if n else 0.0
        contradicts = labels.count("CONTRADICTS")
        # An episode judged with any invalid label is not trusted:
        passed = (contradicts == 0) and (advance_frac >= min_advance_fraction) \
            and (n_invalid == 0)
        return {
            "episode_id": ep.episode_id,
            "per_step": results,
            "advance_fraction": advance_frac,
            "n_contradicts": contradicts,
            "n_invalid_labels": n_invalid,
            "passed_check1": passed,
        }


# ----------------------------------------------------------------------
# Check 2: Leakage judge (k-way multiple choice on story prefixes)
# ----------------------------------------------------------------------

LEAKAGE_SYSTEM = """You are given the beginning of a story and a list of \
candidate hidden goals the author might be pursuing. Pick the single most \
likely hidden goal. If the prefix gives no real evidence, still pick the \
one you consider most plausible.

Respond with a JSON object:
{"choice_index": <0-based integer index into the candidate list>, \
"rationale": "<one sentence>"}"""


class LeakageJudge:
    def __init__(self, client: LLMClient, rng_seed: int = 13):
        self.client = client
        self.rng_seed = rng_seed
        self.rng = random.Random(rng_seed)

    def _mcq_trial(self, prefix_text: str, true_goal: str,
                   distractors: list[str]) -> dict:
        candidates = [true_goal] + list(distractors)
        self.rng.shuffle(candidates)
        true_index = candidates.index(true_goal)
        listing = "\n".join(f"[{i}] {g}" for i, g in enumerate(candidates))
        user = (
            f"STORY PREFIX:\n{prefix_text}\n\n"
            f"CANDIDATE HIDDEN GOALS:\n{listing}\n\n"
            f"Which is most likely?"
        )
        out = self.client.json_call(LEAKAGE_SYSTEM, user, purpose="leakage_mcq")
        try:
            chosen = int(out.get("choice_index", -1))
        except (TypeError, ValueError):
            chosen = -1
        return {
            "candidates": candidates,
            "true_index": true_index,
            "chosen_index": chosen,
            "correct": chosen == true_index,
            "rationale": out.get("rationale", ""),
        }

    def inferability_curve(
        self,
        ep: Episode,
        distractor_goals: list[str],
        k: int = 4,
        n_trials: int = 5,
    ) -> dict:
        """
        Accuracy of goal identification at every prefix length.
        distractor_goals: same-world alternative goals from the episode's
        seed spec (>= k-1 of them); cross-episode goals only as fallback.
        Each trial re-samples distractors and re-shuffles options, so
        n_trials > 1 is meaningful even at temperature 0.
        """
        assert len(distractor_goals) >= k - 1, "need at least k-1 distractors"
        curve = []
        for prefix_len in range(1, len(ep.steps) + 1):
            prefix_text = "\n".join(
                f"[step {i + 1}] {s}" for i, s in enumerate(ep.steps[:prefix_len])
            )
            trials = []
            for _ in range(n_trials):
                distractors = self.rng.sample(distractor_goals, k - 1)
                trials.append(self._mcq_trial(prefix_text, ep.hidden_goal, distractors))
            acc = sum(t["correct"] for t in trials) / n_trials
            curve.append({"prefix_len": prefix_len, "accuracy": acc,
                          "trials": trials})
        return {
            "episode_id": ep.episode_id,
            "k": k,
            "n_trials": n_trials,
            "rng_seed": self.rng_seed,
            "chance": 1.0 / k,
            "n_trials_semantics": "MCQ stability over option shuffles and "
                                  "distractor sampling (temperature=0), NOT "
                                  "model uncertainty",
            "curve": curve,
        }

    @staticmethod
    def passed_check2(
        curve_result: dict,
        step1_max: float = 0.5,
        final_min: float = 0.8,
        prior: Optional[float] = None,
        lift_max: float = 0.20,
    ) -> dict:
        """
        Healthy episode:
          - NOT leaky at step 1. If `prior` (the naive pick-share of this
            episode's true goal from Check 0) is supplied, the gate is
            LIFT-based: step1_accuracy - prior <= lift_max, i.e. the STORY
            must not add early information beyond the environment prior
            (the prior itself is Check 0's jurisdiction, and rotation
            balances residual prior asymmetry across the dataset).
            Without a prior, falls back to the raw gate step1 <= step1_max.
          - goal IS inferable by the final prefix (accuracy >= final_min),
            otherwise Agent B has no signal to work with and the episode
            measures nothing.
        """
        curve = curve_result["curve"]
        step1_acc = curve[0]["accuracy"]
        final_acc = curve[-1]["accuracy"]
        if prior is not None:
            step1_lift = step1_acc - prior
            not_leaky = step1_lift <= lift_max
        else:
            step1_lift = None
            not_leaky = step1_acc <= step1_max
        return {
            "episode_id": curve_result["episode_id"],
            "step1_accuracy": step1_acc,
            "prior": prior,
            "step1_lift": step1_lift,
            "gating_mode": "lift" if prior is not None else "raw",
            "final_accuracy": final_acc,
            "not_leaky": not_leaky,
            "eventually_inferable": final_acc >= final_min,
            "passed_check2": not_leaky and (final_acc >= final_min),
        }


# ----------------------------------------------------------------------
# Convenience: run both checks on an episode file
# ----------------------------------------------------------------------

def run_checks(
    episodes_path: str,
    model: str = "gpt-4o-mini",
    k: int = 4,
    n_trials: int = 5,
    step1_max: float = 0.5,
    final_min: float = 0.8,
    out_path: str = "checks_report.json",
    log_path: Optional[str] = None,
    priors_path: Optional[str] = None,
    lift_max: float = 0.20,
) -> dict:
    """
    step1_max: raw-mode gate (no priors file). Pilot 0.5; full run 0.4.
    priors_path: seed_priors_report.json from check_seed_priors.py. When
      given, episodes whose meta carries family_id are gated on LIFT
      (step1 - prior <= lift_max) instead of raw step1.
    final_min: min required accuracy at the full prefix.
    """
    episodes = load_episodes(episodes_path)
    if log_path is None:
        log_path = out_path.rsplit(".", 1)[0] + "_calls.jsonl"
    client = LLMClient(model=model, log_path=log_path)
    cj = ConsistencyJudge(client)
    lj = LeakageJudge(client)

    # family_id -> {goal_text: naive pick share} from Check 0
    # (goal keys stripped: whitespace drift must not silently break lookups)
    priors_map: dict = {}
    if priors_path:
        with open(priors_path, "r", encoding="utf-8") as f:
            pr = json.load(f)
        for res in pr.get("results", []):
            if "pick_shares" in res:
                priors_map[res["family_id"]] = {
                    g.strip(): s for g, s in res["pick_shares"].items()
                }

    run_config = {
        "created_at": utc_now(),
        "episodes_path": episodes_path,
        "episodes_sha256": sha256_file(episodes_path),
        "judge_model": model,
        "k": k,
        "n_trials": n_trials,
        "step1_max": step1_max,
        "final_min": final_min,
        "lift_max": lift_max,
        "priors_path": priors_path,
        "priors_sha256": sha256_file(priors_path) if priors_path else None,
        "min_advance_fraction": 0.4,
        "leakage_rng_seed": lj.rng_seed,
        "call_log": log_path,
        # Full prompt texts archived so the run is reproducible even
        # without version control on the code:
        "consistency_prompt": CONSISTENCY_SYSTEM,
        "leakage_prompt": LEAKAGE_SYSTEM,
        "consistency_prompt_sha256": sha256_text(CONSISTENCY_SYSTEM),
        "leakage_prompt_sha256": sha256_text(LEAKAGE_SYSTEM),
    }

    all_goals = [e.hidden_goal for e in episodes]
    report = []
    for ep in episodes:
        # Prefer same-world distractors (episode meta); cross-episode goals
        # from different domains make the MCQ solvable by keyword matching
        # and would fail Check 2 spuriously.
        distractors = ep.meta.get("distractor_goals") or \
            [g for g in all_goals if g != ep.hidden_goal]
        c1 = cj.judge_episode(ep)
        curve = lj.inferability_curve(ep, distractors, k=k, n_trials=n_trials)
        prior = None
        missing_prior_reason = None
        fam = ep.meta.get("family_id")
        if priors_map and fam:
            if fam not in priors_map:
                missing_prior_reason = "family_not_in_priors"
            elif ep.hidden_goal.strip() not in priors_map[fam]:
                missing_prior_reason = "goal_not_in_family_priors"
            else:
                prior = priors_map[fam][ep.hidden_goal.strip()]
            if missing_prior_reason:
                print(f"  WARNING {ep.episode_id}: no prior "
                      f"({missing_prior_reason}, family={fam}) — "
                      f"falling back to RAW step-1 gating. Check that the "
                      f"priors file matches this seeds/episodes version.")
        c2 = LeakageJudge.passed_check2(curve, step1_max=step1_max,
                                        final_min=final_min,
                                        prior=prior, lift_max=lift_max)
        report.append({
            "episode_id": ep.episode_id,
            "check1": c1,
            "check2": {**c2, "curve": curve["curve"]},
            "missing_prior": missing_prior_reason is not None,
            "missing_prior_reason": missing_prior_reason,
            "usable": c1["passed_check1"] and c2["passed_check2"],
        })
        lift_str = (f" lift={c2['step1_lift']:+.2f} (prior={c2['prior']:.2f})"
                    if c2["gating_mode"] == "lift" else "")
        print(f"{ep.episode_id}: check1={c1['passed_check1']} "
              f"(advance={c1['advance_fraction']:.2f}, contradicts={c1['n_contradicts']}) | "
              f"check2={c2['passed_check2']} "
              f"(step1={c2['step1_accuracy']:.2f}{lift_str}, final={c2['final_accuracy']:.2f})")

    missing = [r["episode_id"] for r in report if r["missing_prior"]]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"run_config": run_config,
                   "missing_prior_episodes": missing,
                   "results": report},
                  f, ensure_ascii=False, indent=2)
    usable = sum(r["usable"] for r in report)
    print(f"\nUsable episodes: {usable}/{len(report)}  →  {out_path}")
    if priors_path and missing:
        print(f"WARNING: {len(missing)} episode(s) fell back to RAW gating "
              f"(missing prior): {missing}")
        print("         If unexpected, the priors file is out of sync with "
              "this episodes version (compare seeds_sha256 in both manifests).")
    print(f"All calls logged to: {log_path}")
    return {"run_config": run_config, "report": report,
            "missing_prior_episodes": missing,
            "usable": usable, "total": len(report)}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run manipulation checks on episodes.")
    p.add_argument("episodes", help="path to episodes JSON file")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--n-trials", type=int, default=5)
    p.add_argument("--step1-max", type=float, default=0.5,
                   help="pilot: 0.5; full run: 0.4")
    p.add_argument("--final-min", type=float, default=0.8)
    p.add_argument("--out", default="checks_report.json")
    p.add_argument("--log", default=None,
                   help="JSONL call log path (default: <out>_calls.jsonl)")
    p.add_argument("--priors", default=None,
                   help="seed_priors_report.json for lift-based step-1 gating")
    p.add_argument("--lift-max", type=float, default=0.20,
                   help="max allowed step1 accuracy lift over the seed prior")
    args = p.parse_args()
    run_checks(args.episodes, model=args.model, k=args.k,
               n_trials=args.n_trials, step1_max=args.step1_max,
               final_min=args.final_min, out_path=args.out,
               log_path=args.log, priors_path=args.priors,
               lift_max=args.lift_max)