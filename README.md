# The Moriarty Framework

A validated environment for measuring **hidden-goal interpretation as a
trajectory** — not only whether an observer picks the right goal at the end,
but how probability mass moves among four candidate goals as a six-step story
unfolds, when a wrong interpretation captures the trajectory, and which
textual clauses are interventionally implicated in that capture.

Paper: *The Moriarty Framework: A Validated Environment for Measuring
Hidden-Goal Trajectories and Interpretive Capture*  
Repository: https://github.com/moloodbahar/Moriarty  
Design log (dated, pre-registered amendments): `DECISIONS.md`

---

## What the framework does

An episode is built **by construction**:

- **Agent A** receives a world, cast, moral cores, a true hidden goal, and
  three same-world distractors; it writes one story step per API call.
- **Agent B** sees only the observable prefix and must infer the hidden goal
  (Level 1: all cores visible; Level 2: two cores hidden, one visible).

Before any observer experiment, every episode must pass **five validation
gates**: judge calibration, seed priors (Check 0), goal consistency
(Check 1), leakage + reachability (Check 2), and constraint inferability
(Check 3).

The **central measurement object** is an interpretive trajectory. An
exact-posterior probe reads answer-token log-probabilities for `A/B/C/D`,
cycles goals through all four label positions, averages back to semantic
goals, and tracks:

| Symbol | Meaning |
|---|---|
| `q` | true-goal probability |
| `H` | normalized entropy |
| `W` | max wrong-goal probability |
| `Δq`, `ΔH`, `ΔW`, JSD | step-to-step shifts |

From these curves the instrument separates **uncertainty creation**, **wrong
entry**, **wrong collapse**, **committed-wrong runs**, and **resolution**.
Clause-level **incremental addition** and **deletion ablation** localize
*interpretive capture*: textual cues that concentrate belief on one
specific wrong goal. Five clauses pass a strict cross-observer filter
(`validated_triggers.json`).

A **secondary** behavioral ablation compares Direct, CoT-Matched, and
Latent-First prompting at Level 1 and Level 2. That analysis is reported in
the paper but is not the framework's primary contribution.

---

## Prerequisites

```bash
pip install openai google-generativeai matplotlib
```

API keys (`.env` in the repo root or shell environment):

| Variable | Used for |
|---|---|
| `OPENAI_API_KEY` | Agent A, all judges, exact-posterior probe, gpt-4o-mini predictors |
| `GEMINI_API_KEY` | gemini-2.5-flash predictors (Level 1 / Level 2) |

All run artifacts go under `results/<run_name>/` (gitignored). Each run
writes `RUN.json` (manifest), reports with input SHA-256 hashes, and
`*_calls.jsonl` provenance logs.

**Windows (PowerShell):** use single-line commands, or continue lines with
backtick `` ` `` — not `\` (bash-style `\` is passed as a literal argument).

---

## Quick start — replicate the environment

```bash
python run_pipeline.py --name full_run_jul9
```

Reuse an existing folder (overwrites artifacts step by step):

```bash
python run_pipeline.py --name full_run_jul9 --force
```

Environment + checks only (skip expensive predictors):

```bash
python run_pipeline.py --name full_run_jul9 --skip-l1 --skip-l2
```

Preview without executing:

```bash
python run_pipeline.py --name full_run_jul9 --dry-run
```

This runs, in order: judge calibration → Check 0 → episode generation →
Checks 1+2 → analysis + diagnostic plots → Check 3 → Level-1 predictors →
Level-2 predictors.

---

## What to run for each paper result

Replace `full_run_jul9` with your run folder. `$RUN` below means
`results/full_run_jul9`.

| Paper section | What you need | Command / artifact |
|---|---|---|
| §4 Validation protocol | Gates 0–3 + calibration | Phase A (`run_pipeline.py`) |
| §5.1 Environment replicates | Usable count, inferability curve, t\* | `$RUN/checks_report_analysis.json`, `$RUN/checks_plots.png` |
| §5.2 Interpretive trajectories | Step-level `q`, `H`, `W` curves | Phase C steps 1–2 below |
| §5.3 Cross-observer alignment | Naive vs Agent-B-information probes | Phase C (both observers) |
| §5.4 Interpretive capture (Table 1) | Clause addition + deletion | Phase C steps 3–4 |
| §5.5 Exemplar + Figure 1 (`f03`) | Trajectory figure | Phase C step 5 |
| §5.6 Latent-first ablation (Tables 2–3) | L1 / L2 predictors | Phase A → `$RUN/predictions_L1.json`, `predictions_L2.json` |
| §5.7 Branch-aligned sweep (exploratory) | Recovery / LockIn | Phase B |
| Appendix reproducibility | Full artifact bundle | `$RUN/RUN.json` + all reports |

**Replication numbers in the paper:** pilot usable **23/40**; independent
full-pipeline replication (`full_run_jul9`) usable **21/40**. Run your own
folder to reproduce on fresh generations.

---

## Phase A — environment validation and prompting ablation

Orchestrated by `run_pipeline.py`. Manual equivalent:

```bash
python calibrate_judge.py calibration_cases.json --repeats 3 --out $RUN/calibration_report.json --log $RUN/calibration_calls.jsonl

python check_seed_priors.py seeds_v3_1.json --n-trials 24 --out $RUN/seed_priors_report.json --log $RUN/seed_priors_report_calls.jsonl

python generate_episodes.py seeds_v3_1.json --temperature 0.8 --out $RUN/episodes.json --log $RUN/episodes_calls.jsonl

python judges.py $RUN/episodes.json --priors $RUN/seed_priors_report.json --out $RUN/checks_report.json --log $RUN/checks_report_calls.jsonl

python analyze_checks.py $RUN/checks_report.json --priors $RUN/seed_priors_report.json
python plot_checks.py $RUN/checks_report.json --out $RUN/checks_plots.png

python check_core_inferability.py $RUN/episodes.json --report $RUN/checks_report.json --out $RUN/core_inferability_report.json --log $RUN/core_inferability_report_calls.jsonl

python run_predictors.py $RUN/episodes.json $RUN/checks_report.json $RUN/checks_report_analysis.json --conditions direct cot_matched latent_first latent_first_v2 --out $RUN/predictions_L1.json --log $RUN/predictions_L1_calls.jsonl
python analyze_predictors.py $RUN/predictions_L1.json --report $RUN/checks_report.json > $RUN/predictions_L1_analysis.txt

python run_predictors_v4.py $RUN/episodes.json $RUN/checks_report.json $RUN/checks_report_analysis.json $RUN/core_inferability_report.json --out $RUN/predictions_L2.json --log $RUN/predictions_L2_calls.jsonl
python analyze_predictors.py $RUN/predictions_L2.json --report $RUN/checks_report.json > $RUN/predictions_L2_analysis.txt
```

### Phase A outputs

| File | Role |
|---|---|
| `calibration_report.json` | Judge must pass before any labels are used |
| `seed_priors_report.json` | Check 0: premise prior π̂(g) per family |
| `episodes.json` | 40 generated six-step stories (Agent A) |
| `checks_report.json` | Checks 1+2 per episode + prefix inferability curves |
| `checks_report_analysis.json` | Failure taxonomy, rotation balance, registered t\* |
| `checks_plots.png` | Four-panel environment diagnostic |
| `core_inferability_report.json` | Check 3: which cores are inferable (Level-2 masking) |
| `predictions_L1.json` | Level-1 Direct / CoT-Matched / Latent-First (v1, v2) |
| `predictions_L2.json` | Level-2 under partial observability |
| `*_calls.jsonl` | Full API provenance per step |

### Validation gates (episode usable only if ALL pass)

| Gate | Script | Criterion |
|---|---|---|
| Judge calibration | `calibrate_judge.py` | ≥ 7/8 accuracy; CONTRADICTS recall = 1.0 |
| Check 0 (seed priors) | `check_seed_priors.py` | max family share ≤ 0.75; no dead goal (n=24) |
| Check 1 (consistency) | `judges.py` | 0 CONTRADICTS; ≥ 40% ADVANCES; 0 invalid labels |
| Check 2 (leakage) | `judges.py` | step-1 lift over prior ≤ 0.20; final accuracy ≥ 0.80 |
| Check 3 (cores) | `check_core_inferability.py` | inferable core acc ≥ 0.60; unscorable cores never scored |

**Masking rule:** a hidden variable is either demonstrably recoverable (and
scored) or irreducible context (not scored). Hidden-but-unscorable-but-scored
is forbidden.

---

## Phase C — interpretive trajectories and interpretive capture

**Requires Phase A.** This is the paper's central analysis path. Uses
`goal_distribution.py` (exact posteriors via `top_logprobs`, 4-permutation
averaging, `t=0` no-story baseline). Model: `gpt-4o-mini`.

**Cost estimate (21 usable episodes):** ~504 calls per observer for steps
mode; clause mode adds ~(2m+1)×4 calls per localized step.

### C1. Step-level trajectories — naive observer

```bash
python goal_distribution.py $RUN/episodes.json --report $RUN/checks_report.json --mode steps --observer naive --out $RUN/gd_steps_naive.json
```

### C2. Step-level trajectories — Agent-B-information observer

Receives Level-2 visible/hidden core structure (information state only —
not CoT or Latent-First reasoning traces):

```bash
python goal_distribution.py $RUN/episodes.json --report $RUN/checks_report.json --mode steps --observer agent_b --core-report $RUN/core_inferability_report.json --out $RUN/gd_steps_agentB.json
```

**Cross-observer alignment** (paper §5.3): compare `wrong_collapse_step`,
dominant wrong goal, uncertainty-creation step, and resolution step across
both JSON files. Summary recorded in `DECISIONS.md` for `full_run_jul9`.

### C3. Clause-level localization — naive

Localizes wrong-entry, wrong-collapse, and resolution steps via incremental
addition and per-clause deletion:

```bash
python goal_distribution.py $RUN/episodes.json --report $RUN/checks_report.json --mode clauses --steps-report $RUN/gd_steps_naive.json --observer naive --out $RUN/gd_clauses_naive.json
```

### C4. Clause-level localization — Agent-B-information

```bash
python goal_distribution.py $RUN/episodes.json --report $RUN/checks_report.json --mode clauses --steps-report $RUN/gd_steps_agentB.json --observer agent_b --core-report $RUN/core_inferability_report.json --out $RUN/gd_clauses_agentB.json
```

### C5. Validated triggers and Figure 1

The strict cross-observer filter (same collapse step + wrong goal,
addition/deletion agreement within each observer, same clause across
observers, argmax agreement ≥ 3/4, incremental gain ≥ 0.15, deletion drop
≥ 0.10 in both) yields **five validated triggers**. The curated result for
the replication run is in `validated_triggers.json` at the repo root (see
`DECISIONS.md` for verification notes).

**Figure 1** — exemplar trajectory for `f03_product_team_g3`:

```bash
python plot_trajectory.py $RUN/gd_steps_naive.json $RUN/gd_steps_agentB.json --episode f03_product_team_g3 --validated validated_triggers.json --out $RUN/fig_f03_trajectory.png
```

Plots `q`, validated wrong-goal probability, and `H` across steps `t=0…6`
for both probes; marks wrong collapse at step 2.

### Phase C outputs

| File | Role |
|---|---|
| `gd_steps_naive.json` | Per-episode trajectory + decomposition (naive) |
| `gd_steps_agentB.json` | Same, Agent-B information state |
| `gd_clauses_naive.json` | Clause addition/deletion per branch point |
| `gd_clauses_agentB.json` | Same, Agent-B observer |
| `fig_f03_trajectory.png` | Paper Figure 1 |

---

## Phase B — exploratory branch-aligned sweep

**Requires Phase A.** Pre-registered exploratory analysis for the Level-2
latent-first deficit (Recovery / LockIn). See `DECISIONS.md` for metrics and
outcomes. Step 3 is ~690 API calls.

```bash
python branch_points.py $RUN/checks_report.json --usable-only --out $RUN/branch_points.json

python plot_branch_points.py $RUN/branch_points.json --out $RUN/branch_points_fig.png

python run_prefix_sweep.py $RUN/episodes.json $RUN/checks_report.json $RUN/core_inferability_report.json --out $RUN/sweep.json

python analyze_sweep.py $RUN/sweep.json $RUN/branch_points.json
```

---

## Optional follow-ups

**Re-gate Check 2** with new priors (no API calls; reuses stored curves):

```bash
python check_seed_priors.py seeds_v3_1.json --n-trials 24 --out $RUN/seed_priors_n24.json
python regate.py $RUN/checks_report.json $RUN/seed_priors_n24.json --episodes $RUN/episodes.json --out $RUN/checks_report_regated.json
```

**Structured latent grading** (separate from main L1/L2 sweep):

```bash
python run_predictors.py $RUN/episodes.json $RUN/checks_report.json $RUN/checks_report_analysis.json --conditions latent_struct --out $RUN/predictions_latent_struct.json
python grade_latents.py $RUN/episodes.json $RUN/predictions_latent_struct.json --out $RUN/latent_grades.json
```

---

## End-to-end checklist (paper replication)

Minimal path to reproduce the paper's main claims on a fresh run:

1. **Phase A** — `python run_pipeline.py --name <run>`
2. Inspect `$RUN/checks_report_analysis.json` (usable count, curve, t\*)
3. **Phase C1–C2** — `goal_distribution.py` steps (naive + agent_b)
4. **Phase C3–C4** — `goal_distribution.py` clauses (naive + agent_b)
5. **Phase C5** — `plot_trajectory.py` for Figure 1
6. Compare clause outputs against `validated_triggers.json` criteria (or
   re-apply the strict filter from fresh clause JSON)
7. **Phase B** (optional) — branch sweep for exploratory Recovery/LockIn
8. Read `DECISIONS.md` for registered vs exploratory vs amended analyses

---

## File map

| Area | Files |
|---|---|
| Seeds | `seeds_v3_1.json` (10 families × 4 goal rotations → 40 episodes) |
| Generation | `generate_episodes.py` (Agent A, sequential steps) |
| Validation | `calibrate_judge.py`, `check_seed_priors.py`, `judges.py`, `check_core_inferability.py` |
| Orchestration | `run_pipeline.py` |
| Analysis | `analyze_checks.py`, `plot_checks.py`, `analyze_predictors.py`, `regate.py` |
| Predictors | `run_predictors.py` (L1), `run_predictors_v4.py` (L2), `gemini_client.py` |
| Trajectories | `goal_distribution.py`, `plot_trajectory.py` |
| Exploratory | `branch_points.py`, `plot_branch_points.py`, `run_prefix_sweep.py`, `analyze_sweep.py` |
| Validated triggers | `validated_triggers.json` (five cross-observer clauses) |
| Record | `DECISIONS.md`, `pilot_report.md` |

Historical seeds (`seeds.json`, `seeds_v2.json`, `seeds_v3.json`) are kept
for the methods narrative; current runs use `seeds_v3_1.json`.

---

## Provenance and claim boundaries

Every model call is logged to JSONL with request, response, purpose tag,
model, and temperature. Reports embed `run_config` with input SHA-256 hashes
and prompt templates. Thresholds are not loosened post hoc; instrument
changes are dated in `DECISIONS.md` before the runs they govern.

**Supported claim:** clause-level interpretive capture — a local textual
cue increases probability on the same wrong hidden goal under both a naive
probe and an Agent-B-information probe; addition and deletion analyses
converge; deleting the clause weakens that wrong goal.

**Not claimed:** token-level mechanisms, internal reasoning states, unique
causality, or that the probe equals Agent B's CoT/Latent-First process.
The object is an output distribution over four predefined goal candidates.

---

## Citation

If you use this framework, cite the paper (preprint link TBD) and point to
this repository. Key methodological references are in the paper bibliography
(Shlegeris 2025 on evaluation settings; Gurnee et al. 2026 on workspace
readouts as a suggested mechanistic continuation).
