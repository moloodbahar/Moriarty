"""probe_openweights.py — Exact interpretation-distribution curves from an
open-weights model (Step 3 of the confirmatory plan).

Same object as goal_distribution.py, three upgrades:

  1. DETERMINISTIC. Local forward pass, no sampling, no API run-to-run
     variance. Removes the "API nondeterminism" limitation entirely.
  2. EXACT over the FULL vocabulary. Instead of top_logprobs=20, we
     softmax the entire next-token distribution and sum the mass of
     every single-character token that decodes to A/B/C/D. Coverage is
     reported the same way and is typically > 0.99.
  3. CROSS-MODEL. Running the identical protocol on a second model
     family answers the trigger-transfer question directly.

Output schema is IDENTICAL to goal_distribution.py (steps and clauses
modes), so analyze/plot scripts and confirmatory_triggers.py consume
either backend's output interchangeably.

New third mode, --mode replace: the neutral-replacement control.
For each trigger clause (from a triggers file), measure the target
wrong-goal probability under three conditions:
    full      — step as written
    deleted   — step with the trigger clause removed
    replaced  — trigger clause swapped for a length-matched NEUTRAL
                clause from the pre-registered bank below
Pre-registered reading (see PREREGISTRATION.md):
    replaced ~= deleted  << full  -> effect is informational (capture)
    replaced ~= full             -> discourse-coherence artifact

Usage:
    python probe_openweights.py episodes.json --report checks_report.json \
        --mode steps --out ow_steps_naive.json
    python probe_openweights.py episodes.json --report checks_report.json \
        --mode steps --observer agent_b --core-report core_report.json \
        --out ow_steps_agentb.json
    python probe_openweights.py episodes.json --report checks_report.json \
        --mode clauses --steps-report ow_steps_naive.json \
        --out ow_clauses_naive.json
    python probe_openweights.py episodes.json --report checks_report.json \
        --mode replace --triggers confirmatory_triggers.json \
        --out ow_replace_naive.json

Env: MORIARTY_OW_MODEL (default Qwen/Qwen2.5-7B-Instruct),
     MORIARTY_OW_DTYPE, MORIARTY_OW_DEVICE. Gated models (Llama) need
     `huggingface-cli login` first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math

from judges import CallLogger, load_episodes, sha256_file, utc_now
from goal_distribution import (LABELS, SYSTEM, T0_TEXT, decompose, jsd,
                               prefix_text, split_clauses)
import config

# ----------------------------------------------------------------------
# Pre-registered neutral clause bank for --mode replace. Deterministic
# assignment: the neutral clause closest in character length to the
# trigger clause is used (ties -> lower index). Do not edit after the
# PREREGISTRATION.md hash is recorded.
# ----------------------------------------------------------------------
NEUTRAL_BANK = [
    "The clock on the wall showed it was still early.",
    "A phone buzzed somewhere and was quickly silenced.",
    "Outside, the afternoon light shifted slowly across the windows.",
    "Someone refilled a glass of water at the side table.",
    "The hum of the ventilation was the only sound for a moment.",
    "A few papers were straightened and set back down on the table.",
    "The room settled again after a brief shuffle of chairs and footsteps.",
    "For a moment nobody spoke, and the ordinary noises of the building "
    "carried in from the corridor.",
]


def pick_neutral(trigger_clause: str) -> str:
    tl = len(trigger_clause)
    return min(NEUTRAL_BANK, key=lambda c: (abs(len(c) - tl),
                                            NEUTRAL_BANK.index(c)))


class OpenWeightsProbe:
    """Deterministic single-token A/B/C/D posterior from a local model."""

    def __init__(self, model_name: str = None, dtype: str = None,
                 device: str = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.model_name = model_name or config.OPENWEIGHTS_MODEL
        dtype = dtype or config.OPENWEIGHTS_DTYPE
        device = device or config.OPENWEIGHTS_DEVICE
        torch_dtype = {"bfloat16": torch.bfloat16,
                       "float16": torch.float16,
                       "float32": torch.float32}[dtype]
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch_dtype, device_map=device)
        self.model.eval()
        torch.manual_seed(0)
        # One-time vocab scan: every token that decodes (stripped, upper)
        # to exactly one of the four labels. Exact analogue of summing
        # top_logprobs variants ("A", " A", "a", ...), but over the FULL
        # vocab, computed once.
        self.label_token_ids = {lab: [] for lab in LABELS}
        for tid in range(len(self.tok)):
            s = self.tok.decode([tid]).strip().upper()
            if s in self.label_token_ids:
                self.label_token_ids[s].append(tid)
        n = {k: len(v) for k, v in self.label_token_ids.items()}
        print(f"[probe] {self.model_name} | label-token variants: {n}")

    def _prompt_ids(self, user: str):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}]
        ids = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt")
        if ids.shape[1] > config.OPENWEIGHTS_MAX_TOKENS_CTX:
            raise ValueError(f"prompt too long: {ids.shape[1]} tokens")
        return ids.to(self.model.device)

    def one_distribution(self, prefix_text_: str,
                         goals_in_label_order: list,
                         logger: CallLogger, purpose: str,
                         chars_block: str = "") -> dict:
        listing = "\n".join(f"{LABELS[i]}. {g}"
                            for i, g in enumerate(goals_in_label_order))
        user = (f"STORY OPENING:\n{prefix_text_}\n\n"
                + (f"CHARACTERS:\n{chars_block}\n\n" if chars_block else "")
                + f"CANDIDATE HIDDEN GOALS:\n{listing}\n\n"
                f"Which is most likely? Answer with a single letter.")
        ids = self._prompt_ids(user)
        with self.torch.no_grad():
            logits = self.model(ids).logits[0, -1, :].float()
        probs = self.torch.softmax(logits, dim=-1)
        p = {lab: float(sum(probs[t] for t in tids))
             for lab, tids in self.label_token_ids.items()}
        coverage = sum(p.values())
        if coverage > 0:
            p = {k: v / coverage for k, v in p.items()}
        logger.log({"purpose": purpose, "model": self.model_name,
                    "backend": "openweights_exact_full_vocab",
                    "user_sha256": hashlib.sha256(
                        user.encode()).hexdigest(),
                    "p_labels": {k: round(v, 6) for k, v in p.items()},
                    "coverage": round(coverage, 6), "ok": True})
        return {"p_labels": p, "coverage": coverage}

    def goal_posterior(self, prefix_text_: str, goals: list,
                       true_goal: str, logger: CallLogger, purpose: str,
                       chars_block: str = "") -> dict:
        """Identical protocol and output keys to
        goal_distribution.goal_posterior: 4 cyclic permutations averaged,
        per-permutation spread recorded."""
        from collections import Counter
        per_goal = {g: [] for g in goals}
        coverages, perm_argmaxes = [], []
        for shift in range(4):
            order = goals[shift:] + goals[:shift]
            d = self.one_distribution(prefix_text_, order, logger,
                                      f"{purpose}_perm{shift}",
                                      chars_block=chars_block)
            coverages.append(d["coverage"])
            perm_p = {g: d["p_labels"][LABELS[i]]
                      for i, g in enumerate(order)}
            perm_argmaxes.append(max(perm_p, key=perm_p.get))
            for i, g in enumerate(order):
                per_goal[g].append(d["p_labels"][LABELS[i]])
        p = {g: sum(v) / len(v) for g, v in per_goal.items()}
        spread = {g: max(v) - min(v) for g, v in per_goal.items()}
        q = p[true_goal]
        wrong = {g: v for g, v in p.items() if g != true_goal}
        dom = max(p, key=p.get)
        counts = Counter(perm_argmaxes)
        H = -sum(v * math.log2(v) for v in p.values() if v > 0)
        return {"p": p, "q": q, "H": round(H / 2.0, 4),
                "W": round(max(wrong.values()), 4),
                "dominant_goal": dom,
                "dominant_is_true": dom == true_goal,
                "argmax_agreement": counts.get(dom, 0),
                "modal_argmax_goal": counts.most_common(1)[0][0],
                "modal_argmax_count": counts.most_common(1)[0][1],
                "max_perm_spread": round(max(spread.values()), 4),
                "min_coverage": round(min(coverages), 4)}


# ----------------------------------------------------------------------
# Shared setup (mirrors goal_distribution.main)
# ----------------------------------------------------------------------

def build_chars_blocks(episodes, keep, observer, core_report):
    chars_blocks = {eid: "" for eid in keep}
    if observer == "agent_b":
        if not core_report:
            raise SystemExit("--observer agent_b requires --core-report")
        with open(core_report, "r", encoding="utf-8") as f:
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
    return chars_blocks


def run_steps(probe, episodes, keep, chars_blocks, logger):
    results = []
    for eid in keep:
        ep = episodes[eid]
        goals = [ep.hidden_goal] + list(ep.meta.get("distractor_goals", []))
        pts = []
        d0 = probe.goal_posterior(T0_TEXT, goals, ep.hidden_goal, logger,
                                  f"steps_{eid}_t0",
                                  chars_block=chars_blocks[eid])
        d0["t"] = 0
        pts.append(d0)
        for t in range(1, len(ep.steps) + 1):
            d = probe.goal_posterior(prefix_text(ep, t), goals,
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
    return results


def run_clauses(probe, episodes, keep, chars_blocks, logger,
                steps_report, targets):
    with open(steps_report, "r", encoding="utf-8") as f:
        steps_out = {r["episode_id"]: r for r in json.load(f)["results"]}
    results = []
    for eid in keep:
        if eid not in steps_out:
            continue
        ep = episodes[eid]
        goals = [ep.hidden_goal] + list(ep.meta.get("distractor_goals", []))
        step_labels: dict = {}
        for label in targets:
            b = steps_out[eid].get(label)
            if b:
                step_labels.setdefault(b, []).append(label)
        targets_out = []
        step_points = {pt["t"]: pt for pt in steps_out[eid]["points"]}
        for b, labels in sorted(step_labels.items()):
            before = prefix_text(ep, b - 1) if b > 1 else T0_TEXT
            clauses = split_clauses(ep.steps[b - 1])
            sp = step_points.get(b, {})
            target_wrong = sp.get("dominant_goal") \
                if sp and not sp.get("dominant_is_true") else None
            trigger_type = ("wrong" if ("wrong_entry_step" in labels
                                        or "wrong_collapse_step" in labels)
                            else "truth" if "resolution_branch" in labels
                            else "distribution")
            rows = []
            prev = probe.goal_posterior(before, goals, ep.hidden_goal,
                                        logger, f"clause_{eid}_s{b}_base",
                                        chars_block=chars_blocks[eid])
            base = dict(prev)
            for j in range(1, len(clauses) + 1):
                text = before + f"\n[step {b}] " + " ".join(clauses[:j])
                cur = probe.goal_posterior(text, goals, ep.hidden_goal,
                                           logger,
                                           f"clause_{eid}_s{b}_inc{j}",
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
                cur = probe.goal_posterior(text, goals, ep.hidden_goal,
                                           logger,
                                           f"clause_{eid}_s{b}_del{j}",
                                           chars_block=chars_blocks[eid])
                row = {"kind": "deletion",
                       "clause_removed": clauses[j],
                       "q_drop_vs_full": round(full["q"] - cur["q"], 4),
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
                if target_wrong:
                    trig_inc = max(incs, key=lambda r:
                                   r["dp_target_wrong"])["clause_added"]
                    trig_del = max(dels, key=lambda r:
                                   r["target_wrong_drop_on_delete"]
                                   )["clause_removed"]
                else:
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
    return results


def run_replace(probe, episodes, keep, chars_blocks, logger, triggers_path):
    """Neutral-replacement control at each trigger clause."""
    with open(triggers_path, "r", encoding="utf-8") as f:
        trig = json.load(f)
    entries = trig.get("validated", []) + trig.get("candidates", [])
    results = []
    for e in entries:
        eid = e["episode_id"]
        if eid not in keep or eid not in episodes:
            continue
        ep = episodes[eid]
        goals = [ep.hidden_goal] + list(ep.meta.get("distractor_goals", []))
        b = e["step"]
        clause = e["trigger_clause"]
        # match the wrong goal by prefix (triggers file may truncate)
        target_wrong = next(
            (g for g in goals if g.startswith(e["wrong_goal"][:40])), None)
        if target_wrong is None or target_wrong == ep.hidden_goal:
            print(f"[replace] {eid}: cannot match wrong goal — skipped")
            continue
        before = prefix_text(ep, b - 1) if b > 1 else T0_TEXT
        clauses = split_clauses(ep.steps[b - 1])
        try:
            j = clauses.index(clause)
        except ValueError:
            j = next((i for i, c in enumerate(clauses)
                      if clause[:40] in c or c[:40] in clause), None)
            if j is None:
                print(f"[replace] {eid}: trigger clause not found — skipped")
                continue
        neutral = pick_neutral(clauses[j])
        conds = {
            "full": clauses,
            "deleted": [c for i, c in enumerate(clauses) if i != j],
            "replaced": [neutral if i == j else c
                         for i, c in enumerate(clauses)],
        }
        row = {"episode_id": eid, "step": b, "trigger_clause": clauses[j],
               "neutral_clause": neutral, "target_wrong_goal": target_wrong}
        for name, cl in conds.items():
            text = before + f"\n[step {b}] " + " ".join(cl)
            d = probe.goal_posterior(text, goals, ep.hidden_goal, logger,
                                     f"replace_{eid}_s{b}_{name}",
                                     chars_block=chars_blocks[eid])
            row[name] = {"p_target_wrong": round(d["p"][target_wrong], 4),
                         "q": d["q"], "H": d["H"],
                         "argmax_agreement": d["argmax_agreement"]}
        pf, pd_, pr = (row["full"]["p_target_wrong"],
                       row["deleted"]["p_target_wrong"],
                       row["replaced"]["p_target_wrong"])
        # Pre-registered verdict rule (PREREGISTRATION.md §4):
        # informational if the replaced condition recovers >= 50% of the
        # deletion effect; artifact if it recovers < 20%.
        effect = pf - pd_
        recovered = (pf - pr) / effect if abs(effect) > 1e-6 else 0.0
        row["deletion_effect"] = round(effect, 4)
        row["replacement_recovery_fraction"] = round(recovered, 4)
        row["verdict"] = ("informational" if recovered >= 0.5
                          else "artifact" if recovered < 0.2
                          else "mixed")
        results.append(row)
        print(f"{eid}@s{b}: full={pf:.3f} del={pd_:.3f} repl={pr:.3f} "
              f"-> {row['verdict']}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes")
    ap.add_argument("--report", required=True)
    ap.add_argument("--mode", choices=["steps", "clauses", "replace"],
                    default="steps")
    ap.add_argument("--steps-report", default=None)
    ap.add_argument("--triggers", default=None,
                    help="triggers json (required for --mode replace)")
    ap.add_argument("--observer", choices=["naive", "agent_b"],
                    default="naive")
    ap.add_argument("--core-report", default=None)
    ap.add_argument("--targets", nargs="+",
                    default=["wrong_entry_step", "wrong_collapse_step",
                             "resolution_branch"])
    ap.add_argument("--model", default=None,
                    help="HF model id (default: config.OPENWEIGHTS_MODEL)")
    ap.add_argument("--out", default="ow_goal_dist.json")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    probe = OpenWeightsProbe(model_name=args.model)
    logger = CallLogger(args.log or args.out.rsplit(".", 1)[0]
                        + "_calls.jsonl")
    episodes = {e.episode_id: e for e in load_episodes(args.episodes)}
    with open(args.report, "r", encoding="utf-8") as f:
        keep = [r["episode_id"] for r in json.load(f)["results"]
                if r["usable"]]
    chars_blocks = build_chars_blocks(episodes, keep, args.observer,
                                      args.core_report)

    if args.mode == "steps":
        results = run_steps(probe, episodes, keep, chars_blocks, logger)
    elif args.mode == "clauses":
        if not args.steps_report:
            raise SystemExit("--mode clauses requires --steps-report")
        results = run_clauses(probe, episodes, keep, chars_blocks, logger,
                              args.steps_report, args.targets)
    else:
        if not args.triggers:
            raise SystemExit("--mode replace requires --triggers")
        results = run_replace(probe, episodes, keep, chars_blocks, logger,
                              args.triggers)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"run_config": {
                       "created_at": utc_now(),
                       "mode": args.mode, "observer": args.observer,
                       "clause_targets": (args.targets
                                          if args.mode == "clauses"
                                          else None),
                       "episodes_sha256": sha256_file(args.episodes),
                       "method": "exact posteriors from full-vocab "
                                 "softmax, 4 cyclic label permutations "
                                 "averaged, deterministic local forward",
                       "model": probe.model_name,
                       "backend": "openweights",
                   },
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
