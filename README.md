# Moriarty — a validated environment for latent-goal inference

Adversarial narrative games in which an author agent (A) pursues an
experimenter-specified hidden goal under per-character moral constraints,
and an observer agent (B) must infer the latent state from the unfolding
story. Latent variables are specified **by construction**, so inference is
scored against ground truth; every episode passes manipulation checks
before any predictor sees it. Full decision log with dates: `DECISIONS.md`.

## Status (2026-07-07)

- Environment validated: 23/40 usable episodes; pooled inferability curve
  0.19 → 0.99 (chance 0.25); horizon t*=2 by pre-registered rule.
- Pilot 1 (Level 1, all cores visible): **no evidence for H1** — neither
  character-level nor author-level latent-first prompting beats
  budget-matched CoT; Direct ties or beats all reasoning conditions.
  Pre-committed one-revision rule honored; null recorded in DECISIONS.md.
- Current step: Level 2 (partial observability) — same 23 episodes, 2 of 3
  moral cores hidden from B; pre-registered question is whether the
  latent-first advantage appears once there is genuinely something latent
  to infer.

## Pipeline (run in this order)

Set a run name once; every artifact lands under `results/<run_name>/`.
Requires `OPENAI_API_KEY` (and `GEMINI_API_KEY` for predictor steps).
Put keys in `.env` or the shell environment.

### Phase A — main pipeline (one command)

```bash
python run_pipeline.py --name full_run_jul9
```

If the folder already exists from a partial run:

```bash
python run_pipeline.py --name full_run_jul9 --force
```

Skip expensive predictor steps (environment + checks only):

```bash
python run_pipeline.py --name full_run_jul9 --skip-l1 --skip-l2
```

Preview commands without executing:

```bash
python run_pipeline.py --name full_run_jul9 --dry-run
```

**Phase A outputs** (all under `results/full_run_jul9/`):

| File | Step |
|---|---|
| `RUN.json` | manifest of steps + artifact paths |
| `calibration_report.json` | judge calibration gate |
| `seed_priors_report.json` | Check 0 (seed priors) |
| `episodes.json` + `episodes.manifest.json` | generated stories |
| `checks_report.json` | Checks 1+2 |
| `checks_report_analysis.json` | failure taxonomy, rotation balance, **t\*** |
| `checks_plots.png` | 4-panel diagnostic figure |
| `core_inferability_report.json` | Check 3 (core inferability) |
| `predictions_L1.json` + `predictions_L1_analysis.txt` | Level-1 predictors |
| `predictions_L2.json` + `predictions_L2_analysis.txt` | Level-2 predictors |
| `*_calls.jsonl` | full API call logs per step |

### Phase B — exploratory branch / prefix sweep (after Phase A)

Uses only Phase A outputs as inputs. Step 1–2 are free (no API calls);
step 3 is ~690 calls. See `DECISIONS.md` for pre-registered metrics.

Replace `full_run_jul9` with your run folder name:

```bash
# B1. Branch points from stored inferability curves
python branch_points.py results/full_run_jul9/checks_report.json \
    --usable-only --out results/full_run_jul9/branch_points.json

# B2. Branch-step figure
python plot_branch_points.py results/full_run_jul9/branch_points.json \
    --out results/full_run_jul9/branch_points_fig.png

# B3. Prefix sweep (Level-2 masking, prefixes 2..6, one paired run)
python run_prefix_sweep.py \
    results/full_run_jul9/episodes.json \
    results/full_run_jul9/checks_report.json \
    results/full_run_jul9/core_inferability_report.json \
    --out results/full_run_jul9/sweep.json

# B4. Recovery / lock-in analysis (prints to terminal)
python analyze_sweep.py \
    results/full_run_jul9/sweep.json \
    results/full_run_jul9/branch_points.json
```

### Optional follow-ups

**Re-gate Check 2** with new priors (zero API calls; reuses stored curves):

```bash
python check_seed_priors.py seeds_v3_1.json --n-trials 24 \
    --out results/full_run_jul9/seed_priors_n24.json
python regate.py results/full_run_jul9/checks_report.json \
    results/full_run_jul9/seed_priors_n24.json \
    --episodes results/full_run_jul9/episodes.json \
    --out results/full_run_jul9/checks_report_regated.json
```

**Structured latent** (separate from the main L1/L2 sweep):

```bash
python run_predictors.py \
    results/full_run_jul9/episodes.json \
    results/full_run_jul9/checks_report.json \
    results/full_run_jul9/checks_report_analysis.json \
    --conditions latent_struct \
    --out results/full_run_jul9/predictions_latent_struct.json
python grade_latents.py \
    results/full_run_jul9/episodes.json \
    results/full_run_jul9/predictions_latent_struct.json \
    --out results/full_run_jul9/latent_grades.json
```

### Manual step-by-step (alternative to `run_pipeline.py`)

Same order as Phase A, but pass `--out` paths explicitly. Example with
`results/my_run/` as the output folder:

```bash
python calibrate_judge.py calibration_cases.json --repeats 3 \
    --out results/my_run/calibration_report.json \
    --log results/my_run/calibration_calls.jsonl

python check_seed_priors.py seeds_v3_1.json --n-trials 24 \
    --out results/my_run/seed_priors_report.json \
    --log results/my_run/seed_priors_report_calls.jsonl

python generate_episodes.py seeds_v3_1.json --temperature 0.8 \
    --out results/my_run/episodes.json \
    --log results/my_run/episodes_calls.jsonl

python judges.py results/my_run/episodes.json \
    --priors results/my_run/seed_priors_report.json \
    --out results/my_run/checks_report.json \
    --log results/my_run/checks_report_calls.jsonl

python analyze_checks.py results/my_run/checks_report.json \
    --priors results/my_run/seed_priors_report.json
python plot_checks.py results/my_run/checks_report.json \
    --out results/my_run/checks_plots.png

python check_core_inferability.py results/my_run/episodes.json \
    --report results/my_run/checks_report.json \
    --out results/my_run/core_inferability_report.json \
    --log results/my_run/core_inferability_report_calls.jsonl

python run_predictors.py \
    results/my_run/episodes.json \
    results/my_run/checks_report.json \
    results/my_run/checks_report_analysis.json \
    --conditions direct cot_matched latent_first latent_first_v2 \
    --out results/my_run/predictions_L1.json \
    --log results/my_run/predictions_L1_calls.jsonl
python analyze_predictors.py results/my_run/predictions_L1.json \
    --report results/my_run/checks_report.json \
    > results/my_run/predictions_L1_analysis.txt

python run_predictors_v4.py \
    results/my_run/episodes.json \
    results/my_run/checks_report.json \
    results/my_run/checks_report_analysis.json \
    results/my_run/core_inferability_report.json \
    --out results/my_run/predictions_L2.json \
    --log results/my_run/predictions_L2_calls.jsonl
python analyze_predictors.py results/my_run/predictions_L2.json \
    --report results/my_run/checks_report.json \
    > results/my_run/predictions_L2_analysis.txt
```

## The gates (an episode is used only if ALL pass)

| Check | Instrument | Gate |
|---|---|---|
| Judge calibration | hand-labeled cases | acc >= 7/8 AND CONTRADICTS recall = 1.0 |
| Check 0 (seed) | family pick-share MCQ, no story | max share <= 0.75; no dead members (n=24) |
| Check 1 (consistency) | 3-way judge per step | 0 CONTRADICTS; >= 40% ADVANCES; 0 invalid labels |
| Check 2 (leakage) | prefix MCQ inferability curve | step1 lift over prior <= 0.20; final >= 0.8 |
| Check 3 (cores) | full-story core MCQ | hidden+scored core: acc >= 0.6, else hidden-not-scored |

Masking rule: every hidden variable is either verified-inferable (and
scored) or irreducible context (and never scored). Hidden-but-unscorable-
but-scored is the forbidden combination.

## File map

Environment: `seeds_v3_1.json` (goal families) · `generate_episodes.py` ·
`judges.py` (Checks 1+2, LLMClient, CallLogger, provenance helpers) ·
`check_seed_priors.py` (Check 0) · `check_core_inferability.py` (Check 3) ·
`calibrate_judge.py` + `calibration_cases.json`
Orchestration: `run_pipeline.py` (Phase A) · `results/` (per-run output folders)
Analysis: `analyze_checks.py` · `plot_checks.py` · `analyze_predictors.py` ·
`regate.py` (re-gate stored curves)
Prediction: `run_predictors.py` (L1) · `run_predictors_v4.py` (L2) ·
`grade_latents.py` (structured latent grading) · `gemini_client.py`
Exploratory: `branch_points.py` · `plot_branch_points.py` ·
`run_prefix_sweep.py` · `analyze_sweep.py` (Phase B)
Paper/program: `pilot_report.md` · `DECISIONS.md`
Provenance: per-run `RUN.json` · `*_calls.jsonl` · episode manifests
Historical (kept for the methods narrative): `seeds.json`, `seeds_v2.json`,
`seeds_v3.json`

## Provenance conventions

Every model call (request + response + purpose tag) is appended to a JSONL
call log. Every report embeds a `run_config` with input-file SHA-256s,
thresholds, rng seeds, and full judge-prompt texts. Generators write a
manifest with the full prompt template. Thresholds are never loosened
post hoc; instrument changes are dated in `DECISIONS.md` before the runs
they govern. Keys: `OPENAI_API_KEY`, `GEMINI_API_KEY` (in `.env` or shell).
Run artifacts live under `results/` (gitignored).
