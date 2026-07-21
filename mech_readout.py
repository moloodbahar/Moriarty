"""mech_readout.py — layer-level readout of goal-concept availability at
validated trigger clauses (Step 4: turning §7.4's prediction into an
experiment).

Pre-registered prediction (PREREGISTRATION.md §5): at a validated
trigger clause, the specific wrong goal should become AVAILABLE to the
model earlier/stronger across layers than when the clause is absent —
and this difference should exceed the same contrast at resolution
steps for the true goal in the opposite direction.

Method: logit lens over the same one-token probe prompt used
behaviorally. For each trigger and each of the 4 label permutations,
two conditions are run:
    with_clause    — prefix through the trigger step as written
    without_clause — same prefix with the trigger clause deleted
For every layer L, the hidden state at the final position is passed
through the model's final norm and unembedding, softmaxed, restricted
to the A–D label-token sets, renormalized, and mapped back to goals.
This yields per-layer curves p_L(wrong goal) and p_L(true goal),
averaged over permutations.

Reported per trigger:
    auc_wrong_with / auc_wrong_without   — mean over layers in the top
                                           half of the network
    delta_auc_wrong                      — the pre-registered statistic
    first_layer_wrong_dominant           — earliest layer where the
                                           wrong goal is argmax
Group-level test: one-sided Wilcoxon signed-rank on delta_auc_wrong
across triggers (scipy optional; falls back to sign counts).

This is a logit-lens readout, not a causal claim: it tests whether the
wrong-goal concept is linearly readable from the residual stream at the
answer position, which is the weakest version of the workspace
prediction and therefore the right first test.

Usage:
    python mech_readout.py episodes.json \
        --triggers confirmatory_triggers.json \
        --out mech_readout.json
Optional: --observer agent_b --core-report core_report.json
          --model <hf id>   (default: config.OPENWEIGHTS_MODEL)
"""

from __future__ import annotations

import argparse
import json

from judges import load_episodes, sha256_file, utc_now
from goal_distribution import (LABELS, SYSTEM, T0_TEXT, prefix_text,
                               split_clauses)
from probe_openweights import build_chars_blocks
import config


class LogitLens:
    def __init__(self, model_name: str = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.model_name = model_name or config.OPENWEIGHTS_MODEL
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[config.OPENWEIGHTS_DTYPE]
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype,
            device_map=config.OPENWEIGHTS_DEVICE)
        self.model.eval()
        self.label_token_ids = {lab: [] for lab in LABELS}
        for tid in range(len(self.tok)):
            s = self.tok.decode([tid]).strip().upper()
            if s in self.label_token_ids:
                self.label_token_ids[s].append(tid)
        # final norm + unembedding, robust across llama/qwen-style stacks
        base = getattr(self.model, "model", self.model)
        self.final_norm = getattr(base, "norm", None) \
            or getattr(base, "final_layernorm", None)
        self.lm_head = self.model.get_output_embeddings()
        if self.final_norm is None or self.lm_head is None:
            raise RuntimeError("could not locate final norm / lm_head "
                               f"for {self.model_name}")

    def layer_label_dists(self, user: str) -> list:
        """Per-layer renormalized distribution over the 4 labels at the
        final position. Index 0 = embeddings, last = final layer."""
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}]
        ids = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True,
            return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model(ids, output_hidden_states=True)
        dists = []
        for h in out.hidden_states:
            v = self.final_norm(h[0, -1, :]).to(self.lm_head.weight.dtype)
            logits = self.lm_head(v).float()
            probs = self.torch.softmax(logits, dim=-1)
            p = {lab: float(sum(probs[t] for t in tids))
                 for lab, tids in self.label_token_ids.items()}
            s = sum(p.values())
            dists.append({k: (v_ / s if s > 0 else 0.25)
                          for k, v_ in p.items()})
        return dists

    def goal_layer_curves(self, prefix: str, goals: list,
                          chars_block: str = "") -> dict:
        """Per-layer per-goal probabilities, averaged over 4 cyclic
        permutations (same protocol as the behavioral probe)."""
        acc = None
        for shift in range(4):
            order = goals[shift:] + goals[:shift]
            listing = "\n".join(f"{LABELS[i]}. {g}"
                                for i, g in enumerate(order))
            user = (f"STORY OPENING:\n{prefix}\n\n"
                    + (f"CHARACTERS:\n{chars_block}\n\n"
                       if chars_block else "")
                    + f"CANDIDATE HIDDEN GOALS:\n{listing}\n\n"
                    f"Which is most likely? Answer with a single letter.")
            dists = self.layer_label_dists(user)
            if acc is None:
                acc = [{g: 0.0 for g in goals} for _ in dists]
            for li, d in enumerate(dists):
                for i, g in enumerate(order):
                    acc[li][g] += d[LABELS[i]] / 4.0
        return acc  # list over layers of {goal: p}


def top_half_mean(vals):
    h = vals[len(vals) // 2:]
    return sum(h) / len(h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes")
    ap.add_argument("--triggers", required=True)
    ap.add_argument("--observer", choices=["naive", "agent_b"],
                    default="naive")
    ap.add_argument("--core-report", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="mech_readout.json")
    args = ap.parse_args()

    lens = LogitLens(model_name=args.model)
    episodes = {e.episode_id: e for e in load_episodes(args.episodes)}
    with open(args.triggers, "r", encoding="utf-8") as f:
        trig = json.load(f)
    entries = [t for t in trig.get("validated", [])
               if "trigger_clause" in t]
    keep = [t["episode_id"] for t in entries if t["episode_id"] in episodes]
    chars_blocks = build_chars_blocks(episodes, keep, args.observer,
                                      args.core_report)

    results, deltas = [], []
    for e in entries:
        eid = e["episode_id"]
        if eid not in episodes:
            continue
        ep = episodes[eid]
        goals = [ep.hidden_goal] + list(ep.meta.get("distractor_goals", []))
        wrong = next((g for g in goals
                      if g.startswith(e["wrong_goal"][:40])), None)
        if wrong is None:
            print(f"[mech] {eid}: wrong goal not matched — skipped")
            continue
        b, clause = e["step"], e["trigger_clause"]
        before = prefix_text(ep, b - 1) if b > 1 else T0_TEXT
        clauses = split_clauses(ep.steps[b - 1])
        try:
            j = clauses.index(clause)
        except ValueError:
            j = next((i for i, c in enumerate(clauses)
                      if clause[:40] in c or c[:40] in clause), None)
            if j is None:
                print(f"[mech] {eid}: clause not found — skipped")
                continue
        cond_prefix = {
            "with_clause": before + f"\n[step {b}] " + " ".join(clauses),
            "without_clause": before + f"\n[step {b}] "
            + " ".join(c for i, c in enumerate(clauses) if i != j),
        }
        row = {"episode_id": eid, "step": b, "trigger_clause": clauses[j],
               "wrong_goal": wrong[:60], "n_layers": None}
        for name, pref in cond_prefix.items():
            curves = lens.goal_layer_curves(pref, goals,
                                            chars_blocks.get(eid, ""))
            row["n_layers"] = len(curves)
            pw = [c[wrong] for c in curves]
            pt_ = [c[ep.hidden_goal] for c in curves]
            row[name] = {
                "p_wrong_by_layer": [round(x, 4) for x in pw],
                "p_true_by_layer": [round(x, 4) for x in pt_],
                "auc_wrong_top_half": round(top_half_mean(pw), 4),
                "first_layer_wrong_dominant": next(
                    (li for li, c in enumerate(curves)
                     if max(c, key=c.get) == wrong), None)}
        d = (row["with_clause"]["auc_wrong_top_half"]
             - row["without_clause"]["auc_wrong_top_half"])
        row["delta_auc_wrong"] = round(d, 4)
        deltas.append(d)
        results.append(row)
        print(f"{eid}@s{b}: delta_auc_wrong={d:+.4f} "
              f"(first wrong-dominant layer with/without: "
              f"{row['with_clause']['first_layer_wrong_dominant']}/"
              f"{row['without_clause']['first_layer_wrong_dominant']})")

    group = {"n": len(deltas),
             "n_positive": sum(1 for d in deltas if d > 0),
             "mean_delta": round(sum(deltas) / len(deltas), 4)
             if deltas else None}
    try:
        from scipy.stats import wilcoxon
        if len(deltas) >= 5:
            stat, p = wilcoxon(deltas, alternative="greater")
            group["wilcoxon_p_one_sided"] = round(float(p), 5)
    except ImportError:
        group["wilcoxon_p_one_sided"] = "scipy not installed"

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"run_config": {"created_at": utc_now(),
                                  "model": lens.model_name,
                                  "observer": args.observer,
                                  "episodes_sha256":
                                      sha256_file(args.episodes),
                                  "method": "logit lens at answer "
                                            "position, 4 permutations "
                                            "averaged"},
                   "group": group, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\ngroup: {group}\n-> {args.out}")


if __name__ == "__main__":
    main()
