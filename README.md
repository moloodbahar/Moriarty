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

**Recommended — one self-contained results folder:**

```bash
python run_pipeline.py --name my_fresh_run
# -> results/my_fresh_run/  (all reports, logs, plots; see RUN.json manifest)
```

Or step-by-step (outputs land in the repo root unless you pass `--out`):

```bash
# 0. Judge calibration gate (accuracy >= 7/8 AND perfect CONTRADICTS recall)
python calibrate_judge.py calibration_cases.json --repeats 3

# 1. Check 0 — seed priors / family exchangeability (n=24 for gating runs)
python check_seed_priors.py seeds_v3_1.json --n-trials 24 --out seed_priors.json

# 2. Generate episodes (goal families, rotation; step-by-step realization)
python generate_episodes.py seeds_v3_1.json --temperature 0.8 --out episodes.json

# 3. Checks 1+2 — consistency + leakage (lift-gated against Check 0 priors)
python judges.py episodes.json --priors seed_priors.json --out checks_report.json

# 4. Summary, failure taxonomy, rotation balance, horizon t*
python analyze_checks.py checks_report.json --priors seed_priors.json
python plot_checks.py checks_report.json          # 4-panel diagnostic figure

# 5. Check 3 — core inferability (required before Level-2 prediction)
python check_core_inferability.py episodes.json --report checks_report.json \
    --out core_inferability_report.json

# 6. Predictor experiments (paired: direct / cot_matched / latent_first,
#    same-model gpt-4o-mini + cross-model gemini)
python run_predictors.py episodes.json checks_report.json \
    checks_report_analysis.json --out predictions_L1.json        # Level 1
python run_predictors_v4.py episodes.json checks_report.json \
    checks_report_analysis.json core_inferability_report.json \
    --out predictions_L2.json                                    # Level 2

# 7. Paired analysis (exact McNemar vs cot_matched; ceiling sensitivity)
python analyze_predictors.py predictions_L2.json --report checks_report.json
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
Analysis: `analyze_checks.py` · `plot_checks.py` · `analyze_predictors.py`
Prediction: `run_predictors.py` (L1) · `run_predictors_v4.py` (L2) ·
`gemini_client.py`
Paper/program: `moriarty_draft.tex` · `language_between_proposal.md`
Provenance: `DECISIONS.md` · per-run `*_calls.jsonl` + manifests
Historical (kept for the methods narrative): `seeds.json`, `seeds_v2.json`,
`seeds_v3.json`, `dryrun_harness.py`

## Provenance conventions

Every model call (request + response + purpose tag) is appended to a JSONL
call log. Every report embeds a `run_config` with input-file SHA-256s,
thresholds, rng seeds, and full judge-prompt texts. Generators write a
manifest with the full prompt template. Thresholds are never loosened
post hoc; instrument changes are dated in `DECISIONS.md` before the runs
they govern. Keys: OPENAI_API_KEY, GEMINI_API_KEY.
