# Moriarty

**A validated environment for measuring hidden-goal interpretation as a trajectory.**

Moriarty is not a trained model. It is a research environment for studying how an observer moves among competing interpretations of another agent's hidden goal as evidence unfolds.

Instead of asking only whether the observer is correct at the end, Moriarty asks:

> When does uncertainty increase? When does one coherent wrong goal take over? How long does that interpretation persist? What textual cue creates the shift? And does the path itself affect how later evidence is integrated?

- **Paper:** *The Moriarty Framework: A Validated Environment for Measuring Hidden-Goal Trajectories and Interpretive Capture* — manuscript in preparation
- **Design and decision log:** [`DECISIONS.md`](DECISIONS.md)
- **Current seed families:** [`seeds_v3_1.json`](seeds_v3_1.json)
- **Validated clause-level triggers:** [`validated_triggers.json`](validated_triggers.json)

---

## Why this environment exists

Hidden-goal inference is normally ill-posed. Observable behavior can fit several intentions, and a model's written reasoning is not reliable ground truth about its internal computation.

Moriarty addresses the measurement problem by fixing the latent variables **before generation**:

1. The experimenter specifies a hidden goal and persistent character constraints.
2. **Agent A** generates a six-step story while pursuing that goal.
3. The episode is validated for premise bias, goal consistency, early leakage, eventual recoverability, and constraint visibility.
4. **Agent B** receives only the observable trajectory and must infer the hidden goal.
5. A log-probability-based probe tracks the observer's goal-hypothesis distribution at every story prefix.
6. Clause-level addition and deletion interventions localize the textual cues that produce a particular interpretive shift.

The environment is therefore designed to make both the hidden state and the observer's changing interpretation measurable.

```text
DESIGNED LATENT STATE
        ↓
SEQUENTIAL GENERATION BY AGENT A
        ↓
VALIDATION: PRIOR · CONSISTENCY · LEAKAGE · REACHABILITY · CORE VISIBILITY
        ↓
GOAL-HYPOTHESIS TRAJECTORY ACROSS STORY PREFIXES
        ↓
CLAUSE-LEVEL ADDITION AND DELETION INTERVENTIONS
```

---

## The central object: an interpretation trajectory

For each story prefix, the observer assigns probability mass to four candidate hidden goals. Goals are cycled through all four answer-label positions and mapped back to semantic goals before averaging.

This produces an operational goal-hypothesis distribution over time:

```text
prior bias
   → uncertainty creation
   → wrong entry
   → wrong-goal collapse
   → committed-wrong persistence
   → uncertainty reopening
   → resolution
```

| Symbol | Meaning |
|---|---|
| `q_t` | probability assigned to the true goal at step `t` |
| `H_t` | normalized entropy over the four goal hypotheses |
| `W_t` | probability of the dominant wrong goal |
| `Δq`, `ΔH`, `ΔW` | step-to-step changes |
| `JSD` | Jensen-Shannon divergence between consecutive distributions |

A **wrong collapse** is not merely uncertainty or a random mistake. It is a reduction in uncertainty caused by one specific false goal becoming dominant.

I use **interpretive capture** for the stronger clause-level phenomenon in which a local textual cue concentrates belief on one coherent wrong goal, and removing that cue weakens the same hypothesis.

---

## Current status

As of July 2026, the environment and main analysis have been run twice.

| Result | Current evidence |
|---|---|
| Environment replication | 23/40 usable episodes in the pilot; 21/40 in an independent full-pipeline replication |
| Inferability | The true goal is not required to be obvious early, but must become recoverable by the end |
| Cross-observer wrong collapse | Both observer-information conditions detect wrong collapse in 12 episodes |
| Step agreement | 11/12 jointly detected collapses occur at the same story step |
| Wrong-goal agreement | All 11 matched-step cases collapse onto the same specific wrong goal |
| Clause localization | Five clauses pass the strict cross-observer addition/deletion filter |

The five validated clauses are recorded in [`validated_triggers.json`](validated_triggers.json).

A separate behavioral ablation compares **Direct**, **CoT-Matched**, and **Latent-First** prompting. It is included as an additional use of the environment, not as Moriarty's primary contribution.

---

## The next research question: does the path matter?

Two observers can reach the same final answer through different interpretive paths. One may remain uncertain and update smoothly. Another may first become strongly committed to a coherent wrong goal and recover only after strong corrective evidence.

The next planned study asks whether this history is computationally important:

- Does the strength and duration of wrong-goal dominance predict slower integration of later evidence?
- Are correct final answers reached after wrong collapse less robust to renewed misleading cues?
- Do behavioral interpretation trajectories correspond to evolving true- and wrong-goal representations inside the model?
- Can suppressing or strengthening those representations alter persistence and recovery?

A proposed mechanistic extension uses the **J-lens / J-space** approach from Anthropic's global-workspace work to track the competition between the true and validated wrong goal across layers and story steps. This experiment is planned but has **not yet been run**.

---

## Validation gates

An episode is used only if every applicable gate passes.

| Gate | Script | Criterion |
|---|---|---|
| Judge calibration | `calibrate_judge.py` | at least 7/8 accuracy and perfect `CONTRADICTS` recall |
| Check 0: seed priors | `check_seed_priors.py` | maximum family pick share at most 0.75; no dead goal, `n=24` |
| Check 1: goal consistency | `judges.py` | zero `CONTRADICTS`, at least 40% `ADVANCES`, no invalid labels |
| Check 2: leakage and reachability | `judges.py` | step-1 lift over prior at most 0.20; final accuracy at least 0.80 |
| Check 3: core inferability | `check_core_inferability.py` | hidden-and-scored core accuracy at least 0.60 |

**Masking rule:** a hidden variable is either demonstrably recoverable and scored, or treated as irreducible context and not scored. Hidden-but-unscorable-but-scored is forbidden.

---

## Installation

```bash
pip install openai google-generativeai matplotlib
```

Create a local `.env` file or export the keys in your shell:

```text
OPENAI_API_KEY=...
GEMINI_API_KEY=...
```

| Variable | Used for |
|---|---|
| `OPENAI_API_KEY` | Agent A, validation judges, trajectory probe, and `gpt-4o-mini` observers |
| `GEMINI_API_KEY` | `gemini-2.5-flash` observer experiments |

Do not commit API keys or private logs. Run artifacts are written under `results/<run_name>/`, which is gitignored.

---

## Quick start

Run the full environment and observer pipeline:

```bash
python run_pipeline.py --name my_run
```

Run only the environment and validation checks:

```bash
python run_pipeline.py --name my_run --skip-l1 --skip-l2
```

Preview commands without executing them:

```bash
python run_pipeline.py --name my_run --dry-run
```

Reuse an existing run folder:

```bash
python run_pipeline.py --name my_run --force
```

The main pipeline runs:

```text
judge calibration
→ seed-prior measurement
→ sequential episode generation
→ consistency and inferability checks
→ environment analysis
→ core inferability
→ Level-1 observer ablation
→ Level-2 partial-observability ablation
```

**Windows PowerShell:** use single-line commands, or use the PowerShell continuation character `` ` ``. Bash-style `\` is passed as a literal argument.

---

## Reproduce the central trajectory analysis

Let:

```bash
RUN=results/my_run
```

### 1. Naive-observer trajectories

```bash
python goal_distribution.py $RUN/episodes.json \
  --report $RUN/checks_report.json \
  --mode steps \
  --observer naive \
  --out $RUN/gd_steps_naive.json
```

### 2. Agent-B-information trajectories

This probe receives the Level-2 visible/hidden core structure. It does **not** reproduce Agent B's chain-of-thought or Latent-First reasoning trace.

```bash
python goal_distribution.py $RUN/episodes.json \
  --report $RUN/checks_report.json \
  --mode steps \
  --observer agent_b \
  --core-report $RUN/core_inferability_report.json \
  --out $RUN/gd_steps_agentB.json
```

### 3. Clause-level localization

```bash
python goal_distribution.py $RUN/episodes.json \
  --report $RUN/checks_report.json \
  --mode clauses \
  --steps-report $RUN/gd_steps_naive.json \
  --observer naive \
  --out $RUN/gd_clauses_naive.json
```

```bash
python goal_distribution.py $RUN/episodes.json \
  --report $RUN/checks_report.json \
  --mode clauses \
  --steps-report $RUN/gd_steps_agentB.json \
  --observer agent_b \
  --core-report $RUN/core_inferability_report.json \
  --out $RUN/gd_clauses_agentB.json
```

### 4. Plot the `f03_product_team_g3` exemplar

```bash
python plot_trajectory.py \
  $RUN/gd_steps_naive.json \
  $RUN/gd_steps_agentB.json \
  --episode f03_product_team_g3 \
  --validated validated_triggers.json \
  --out $RUN/fig_f03_trajectory.png
```

The strict trigger filter requires:

- the same wrong-collapse step and wrong goal in both observer-information conditions;
- addition/deletion agreement within each condition;
- the same clause across conditions;
- semantic argmax agreement of at least 3/4 across label permutations;
- incremental wrong-goal gain of at least 0.15; and
- deletion drop of at least 0.10 in both conditions.

---

<details>
<summary><strong>Secondary observer ablation and branch-aligned sweep</strong></summary>

### Level 1

```bash
python run_predictors.py \
  $RUN/episodes.json \
  $RUN/checks_report.json \
  $RUN/checks_report_analysis.json \
  --conditions direct cot_matched latent_first latent_first_v2 \
  --out $RUN/predictions_L1.json \
  --log $RUN/predictions_L1_calls.jsonl
```

### Level 2

```bash
python run_predictors_v4.py \
  $RUN/episodes.json \
  $RUN/checks_report.json \
  $RUN/checks_report_analysis.json \
  $RUN/core_inferability_report.json \
  --out $RUN/predictions_L2.json \
  --log $RUN/predictions_L2_calls.jsonl
```

### Exploratory branch-aligned sweep

```bash
python branch_points.py $RUN/checks_report.json \
  --usable-only \
  --out $RUN/branch_points.json

python run_prefix_sweep.py \
  $RUN/episodes.json \
  $RUN/checks_report.json \
  $RUN/core_inferability_report.json \
  --out $RUN/sweep.json

python analyze_sweep.py \
  $RUN/sweep.json \
  $RUN/branch_points.json
```

</details>

---

## Main outputs

| File | Role |
|---|---|
| `RUN.json` | run manifest and artifact paths |
| `calibration_report.json` | judge calibration result |
| `seed_priors_report.json` | no-story premise priors |
| `episodes.json` | generated six-step stories |
| `checks_report.json` | consistency and prefix inferability |
| `checks_report_analysis.json` | usable count, failure taxonomy, and registered horizon |
| `core_inferability_report.json` | character-core visibility and scoring mask |
| `gd_steps_naive.json` | naive-observer trajectory decomposition |
| `gd_steps_agentB.json` | Agent-B-information trajectory decomposition |
| `gd_clauses_naive.json` | naive clause interventions |
| `gd_clauses_agentB.json` | Agent-B-information clause interventions |
| `predictions_L1.json` | Level-1 prompting ablation |
| `predictions_L2.json` | Level-2 partial-observability ablation |
| `*_calls.jsonl` | raw request/response provenance logs |

---

## Repository map

| Area | Files |
|---|---|
| Seed construction | `seeds_v3_1.json` |
| Sequential generation | `generate_episodes.py` |
| Validation | `calibrate_judge.py`, `check_seed_priors.py`, `judges.py`, `check_core_inferability.py` |
| Orchestration | `run_pipeline.py` |
| Environment analysis | `analyze_checks.py`, `plot_checks.py`, `regate.py` |
| Observer ablations | `run_predictors.py`, `run_predictors_v4.py`, `analyze_predictors.py`, `gemini_client.py` |
| Trajectory analysis | `goal_distribution.py`, `plot_trajectory.py` |
| Branch-aligned analysis | `branch_points.py`, `plot_branch_points.py`, `run_prefix_sweep.py`, `analyze_sweep.py` |
| Trigger record | `validated_triggers.json` |
| Research record | `DECISIONS.md`, `pilot_report.md` |

Historical seed files are retained to preserve the design history. Current runs use `seeds_v3_1.json`.

---

## Reproducibility and provenance

Every model call is written to an append-only JSONL log containing the request, response, purpose tag, model, and temperature. Reports embed:

- SHA-256 hashes of their inputs;
- prompt templates and prompt hashes;
- random seeds;
- thresholds;
- model configuration; and
- artifact paths.

Thresholds are not silently loosened after results are observed. Instrument changes, failed designs, amendments, and analysis-status labels are recorded in [`DECISIONS.md`](DECISIONS.md).

Because raw result folders may contain large logs, they are gitignored. A cleaned, versioned dataset release is planned.

---

## Claim boundaries

### Supported

Moriarty measures clause-level **interpretive capture** under a predefined four-goal hypothesis space. In five strictly filtered episodes, the same clause increases probability on the same wrong hidden goal under both a naive probe and an Agent-B-information probe; independent addition and deletion analyses converge, and removing the clause weakens that wrong-goal hypothesis.

### Not claimed

This repository does not currently establish:

- direct access to a model's internal beliefs;
- token-level or layer-level mechanisms;
- unique causality of a single clause;
- that the one-token probe reproduces Agent B's original reasoning process;
- transfer to naturally occurring hidden goals; or
- that interpretation history is causally important.

The last point is the target of the planned trajectory-mechanism study.

---

## Citation

A preprint citation will be added when the manuscript is released. For now, please cite the repository:

```bibtex
@misc{arman2026moriarty,
  author       = {Molood (Melody) Arman},
  title        = {The Moriarty Framework: A Validated Environment for Measuring Hidden-Goal Trajectories and Interpretive Capture},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/moloodbahar/Moriarty}
}
```

---

## Contact and collaboration

I am interested in collaboration on:

- hidden-goal and belief-trajectory evaluation;
- mechanistic interpretability of true-versus-wrong goal competition;
- J-space / J-lens trajectory analysis;
- path dependence in model reasoning;
- multi-agent misdirection and recovery; and
- benchmark and dataset development.

Open an issue or contact **Molood (Melody) Arman** through the repository profile.