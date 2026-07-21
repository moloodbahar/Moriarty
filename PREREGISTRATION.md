# Preregistration — Moriarty Confirmatory Study

Frozen on commit. `confirmatory_triggers.py` records this file's SHA-256
in its output; any edit after the confirmatory data exist invalidates
the confirmatory label and demotes all results to exploratory.

## 1. Data

- New goal families only: `seeds_v4_confirmatory.json` (f11–f26,
  16 families × 4 rotations = 64 generated episodes).
- Pilot (`episodes_v3.json`) and replication episode IDs are excluded;
  `confirmatory_triggers.py --exclude` enforces this and aborts on
  overlap.
- All validation gates (Checks 0–3) unchanged from the paper: family
  prior gate (no goal > 0.75, none dead, n = 24), zero contradictions,
  ≥ 40% advancing, leakage lift ≤ 0.20, reachability ≥ 0.80, core
  inferability ≥ 0.60. Episodes are gated BEFORE any probe runs.
- Stopping rule: generate all 64; no regeneration of failed episodes;
  usable count is whatever survives the gates.

## 2. Instruments

- Primary probe: open-weights model (`config.OPENWEIGHTS_MODEL`,
  default Qwen/Qwen2.5-7B-Instruct), exact full-vocab posterior,
  4 cyclic label permutations averaged, deterministic
  (`probe_openweights.py`). Secondary probe: gpt-4o-mini via the
  original `goal_distribution.py`, for cross-model trigger transfer.
- Both observers (naive, agent_b) run for steps and clauses modes.

## 3. Frozen interpretive-capture filter (H1)

Constants in `confirmatory_triggers.py`, identical to the pilot filter:
same wrong-collapse step and same wrong goal in both observers;
addition/deletion agreement within each observer; same clause across
observers; argmax agreement ≥ 3/4 in both; incremental wrong-goal gain
≥ 0.15 in both; deletion drop ≥ 0.10 in both.

- **H1 (replication):** ≥ 3 validated triggers among the usable
  confirmatory episodes on the primary probe.
- **H1b (cross-model transfer):** among clauses validated on the
  primary probe, ≥ 50% are also selected by addition on the secondary
  probe (gpt-4o-mini) at the same step for the same wrong goal.
- Per-gate attrition is reported in full regardless of outcome.

## 4. Neutral-replacement control (H2)

`probe_openweights.py --mode replace`, neutral bank frozen in that
file; assignment is deterministic (closest character length, ties to
lower index). Recovery fraction = (p_full − p_replaced) /
(p_full − p_deleted).

- **H2:** median recovery fraction across validated triggers ≥ 0.5
  (capture is informational, not a discourse-coherence artifact).
  Verdict thresholds per trigger: informational ≥ 0.5; artifact < 0.2;
  otherwise mixed.

## 5. Mechanistic readout (H3)

`mech_readout.py`: logit lens at the answer position, 4 permutations
averaged, conditions with_clause vs without_clause per validated
trigger. Statistic: delta_auc_wrong = mean p(wrong goal) over the top
half of layers, with minus without.

- **H3:** delta_auc_wrong > 0 for a majority of validated triggers;
  one-sided Wilcoxon signed-rank p < 0.05 if n ≥ 5.
- A null here is reported as a null; it bounds the workspace prediction
  of §7.4, it does not undermine H1/H2.

## 6. Prediction horizon

The pooled-horizon rule is retired. The registered confirmatory horizon
is branch-relative: for each episode, the first prefix at or after the
measured wrong-collapse step (or, absent one, the interpretation-branch
step). Any prompting-condition comparison on confirmatory data uses
exact McNemar at this per-episode horizon.

## 7. Exclusions and deviations

Any deviation from the above is recorded in DECISIONS.md with a date
and demotes the affected analysis to exploratory. Analyses not listed
here are exploratory by default.
