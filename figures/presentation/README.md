# MORIARTY — presentation figures

Three coordinated, editable 1920 × 1080 SVG figures and a self-contained presentation/export file.

## Files and use

- **moriarty_pipeline.svg** — the main audience-facing pipeline figure. A four-stage narrative, a labeled illustrative trajectory, and a recorded clause intervention example.
- **moriarty_measurement.svg** — the probability probe, information conditions, event definitions, and strict clause filter.
- **moriarty_validation.svg** — validation, recorded evidence, secondary prompting experiments, and implemented confirmatory extensions.
- **moriarty_figures.html** — download and open locally in a modern browser. Switch between figures, present fullscreen, save each figure as SVG or 3840 × 2160 PNG, or print all figures to PDF. No network calls or external fonts/assets are required.

For PowerPoint, insert the SVG as a picture to retain sharp vector rendering. PowerPoint versions supporting “Convert to Shape” can expose the vector elements for editing; complex text or gradient handling depends on the application. The original SVG text and shapes also remain editable in a vector editor. This package is not a native PPTX file.

## Suggested 45-second narration

“MORIARTY starts with something hidden-goal inference usually lacks: a goal we know in advance. I specify that goal and the characters’ persistent constraints. Agent A then produces an unfolding story, one step at a time. After validating that the goal is neither leaked too early nor impossible to recover, I probe an observer at every prefix. This shows when uncertainty becomes commitment to a coherent wrong goal. Finally, I add and delete clauses to identify the cue that strengthens that specific interpretation. So the object of study is the trajectory into error and recovery, not just whether the final answer is right.”

## Source and status

Figures are grounded in repository source at commit **f9c34fbfbc02da2abbd1f1f915b48660356f2f9f**. They were prepared through source inspection; no model experiments were rerun. The attached manuscript and arena HTML could not be read using the file tools available in the creation session, so these figures do not claim to reproduce details unique to those attachments.

Primary sources:

- [Framework scope, recorded aggregate results, and claim boundaries](https://github.com/moloodbahar/Moriarty/blob/f9c34fbfbc02da2abbd1f1f915b48660356f2f9f/README.md)
- [Latent seed families and the product-team example](https://github.com/moloodbahar/Moriarty/blob/f9c34fbfbc02da2abbd1f1f915b48660356f2f9f/seeds_v3_1.json)
- [Sequential Agent A generation](https://github.com/moloodbahar/Moriarty/blob/f9c34fbfbc02da2abbd1f1f915b48660356f2f9f/generate_episodes.py)
- [Probability probe, event definitions, and clause interventions](https://github.com/moloodbahar/Moriarty/blob/f9c34fbfbc02da2abbd1f1f915b48660356f2f9f/goal_distribution.py)
- [Recorded validated triggers and effects](https://github.com/moloodbahar/Moriarty/blob/f9c34fbfbc02da2abbd1f1f915b48660356f2f9f/validated_triggers.json)
- [Consistency, leakage, and recoverability checks](https://github.com/moloodbahar/Moriarty/blob/f9c34fbfbc02da2abbd1f1f915b48660356f2f9f/judges.py)
- [Premise-prior checks](https://github.com/moloodbahar/Moriarty/blob/f9c34fbfbc02da2abbd1f1f915b48660356f2f9f/check_seed_priors.py)
- [Core inferability and scoring mask](https://github.com/moloodbahar/Moriarty/blob/f9c34fbfbc02da2abbd1f1f915b48660356f2f9f/check_core_inferability.py)
- [Current orchestrator](https://github.com/moloodbahar/Moriarty/blob/f9c34fbfbc02da2abbd1f1f915b48660356f2f9f/run_pipeline.py)
- [Confirmatory preregistration](https://github.com/moloodbahar/Moriarty/blob/f9c34fbfbc02da2abbd1f1f915b48660356f2f9f/PREREGISTRATION.md)
- [Confirmatory runbook](https://github.com/moloodbahar/Moriarty/blob/f9c34fbfbc02da2abbd1f1f915b48660356f2f9f/RUNBOOK_confirmatory.md)
- [Model configuration](https://github.com/moloodbahar/Moriarty/blob/f9c34fbfbc02da2abbd1f1f915b48660356f2f9f/config.py)
- [Open-weights instrument](https://github.com/moloodbahar/Moriarty/blob/f9c34fbfbc02da2abbd1f1f915b48660356f2f9f/probe_openweights.py)
- [Layer readout](https://github.com/moloodbahar/Moriarty/blob/f9c34fbfbc02da2abbd1f1f915b48660356f2f9f/mech_readout.py)

The recorded aggregate evidence and the extensions are deliberately distinguished. The README reports 23/40 usable pilot episodes, 21/40 usable replication episodes, 12 episodes with wrong collapse in both information conditions, matching collapse steps in 11/12, the same specific wrong goal in all 11 matched cases, and five strictly filtered clauses. Presence of confirmatory code and a preregistration is not evidence that the confirmatory hypotheses were established.

## Exact example in Figure 1

Episode: `f03_product_team_g3`; collapse step: **2**.

- True goal: “Get Noel to commit to a position before knowing whom the confident majority favors, believing the initiative was his own.”
- Dominant wrong goal: “Get Dara to voluntarily withdraw his redesign proposal without ever being publicly contradicted.”
- Character constraints: Dara never backs down in front of an audience; Yara protects the team from blame, even by taking it herself; Noel defers to whoever seems most confident.
- The displayed phrase “eager to protect her teammate” is an excerpt of the recorded trigger clause, not a replacement sentence or the complete clause.

| Information condition | Addition gain on the target wrong goal | Drop when deleted |
|---|---:|---:|
| Naive | 0.7287 | 0.1461 |
| Agent-B-information | 0.3778 | 0.1880 |

The figure rounds to three decimals. Deletion is displayed as a signed change (−0.146, −0.188); the source stores positive drop magnitudes.

The four-line trajectory in Figure 1 is **illustrative, not an experimental run**. It must not be attributed to the product-team episode or used as empirical evidence. Its probabilities are explicitly constructed to show the distinction between uncertainty, wrong-goal concentration, persistence, and recovery:

| Prefix | G1: wrong | G2 | G3: true | G4 |
|---|---:|---:|---:|---:|
| 0 | 0.300 | 0.120 | 0.480 | 0.100 |
| 1 | 0.320 | 0.220 | 0.280 | 0.180 |
| 2 | 0.870 | 0.050 | 0.050 | 0.030 |
| 3 | 0.900 | 0.040 | 0.040 | 0.020 |
| 4 | 0.580 | 0.160 | 0.160 | 0.100 |
| 5 | 0.250 | 0.100 | 0.600 | 0.050 |
| 6 | 0.040 | 0.025 | 0.920 | 0.015 |

## Technical interpretation

### 1. Fixed latent design; sequential generation

Ten current seed families each supply a world, three character cores, four candidate goals, and six steps. Goal rotations make each candidate the true goal in a separate episode. Agent A receives the true goal and distractors; the generation prompt encourages early ambiguity, gradual narrowing, and final distinguishability. Each generation call sees the existing story and the number of remaining steps. The default generator is gpt-4o-mini at temperature 0.8.

Agent A returns a visible step and a private rationale. The private rationale is logged, is not guaranteed faithful, and is withheld from Agent B and the leakage judge. The framework here is a sequential generation-and-observation environment. The diagram does not imply a two-way strategic game, online reinforcement learning, or B-to-A feedback.

### 2. Two distinct notions of a prior

Check 0 measures no-story premise preferences using the setup, characters, and candidate goals. The trajectory probe’s t=0 uses candidate goals with no story, plus static character information when that condition exposes it. These are different prompts serving different purposes; the diagram does not identify the two distributions.

### 3. The one-token probe is operational

For each prefix and each of four cyclic goal-to-label assignments:

1. Ask for one letter A–D.
2. Request one output token at temperature 0 with log-probabilities and top_logprobs=20.
3. Normalize returned token strings by stripping whitespace and uppercasing; sum exp(logprob) for tokens mapped to the same label.
4. Record total retrieved A–D probability mass (“coverage”), then normalize within those labels.
5. Map labels back to semantic goals.
6. Average the four semantic distributions.

Four cyclic rotations put every goal in every answer position; they are not all 24 possible permutations. Track maximum per-goal permutation spread and the number of permutation argmaxes agreeing with the averaged distribution’s dominant semantic goal.

The API instrument is conditioned on retrieved answer-label mass. It is not guaranteed to recover absent tokens outside the returned top-20 list. Consequently it should not be described as access to a full internal belief state, a calibrated Bayesian posterior, or a complete semantic probability distribution. The open-weights extension uses full-vocabulary logits to avoid the API top-k retrieval limitation.

Every prefix is a fresh direct call conditioned on that prefix. Prior probe answers are not fed back as the observer’s memory. This measures response trajectories as evidence changes, without proving stateful belief persistence inside one continuing inference process.

### 4. Information conditions

Naive: visible story prefix and candidate goals; no cores.

Agent-B-information: the same inputs plus the least-inferable character core shown explicitly; other cores marked unknown. This reproduces a visible/hidden information arrangement, not an original CoT or Latent-First reasoning trace. Both trajectory conditions use the same model in the reported API instrument; this is not cross-model replication.

### 5. Metrics and event selection

Let g* be the true goal. q_t=p_t(g*); W_t is the maximum probability among wrong goals. H_t is base-2 entropy divided by log2(4)=2. JSD compares consecutive four-goal distributions using base-2 KL.

- Wrong-collapse candidate: ΔW>0, Δq<0, ΔH<0 and wrong semantic argmax. Among candidates with argmax agreement ≥3/4, select the largest ΔW.
- Committed wrong: q≤0.20, W≥0.70, H≤0.50; consecutive qualifying steps form persistence runs.
- Resolution: Δq≥0.30, decreasing H, true semantic argmax and agreement ≥3/4; select the strongest qualifying gain.
- Wrong entry: first true-dominant to wrong-dominant transition, with agreement ≥3/4 on both adjacent points.
- Uncertainty creation: strongest entropy rise ≥0.15.
- Uncertainty peak: maximum H.
- Interpretation branch: maximum consecutive JSD.
- Entropy reopening is distinguished from goal resolution; one largest-JSD event would conflate different types of transition.

### 6. Clause localization

The default localization targets are wrong entry, wrong collapse, and resolution; duplicate steps are merged. Clause segmentation is a recorded heuristic: split sentences, then split long sentences before specified conjunctions.

Addition appends clauses in their original order to the previous prefix. Deletion removes each individual clause from the complete target step. At a wrong-collapse step, track the **same named wrong goal** through every edit, not whichever wrong goal happens to be largest after the edit.

The strict cross-condition filter requires:

- Same wrong-collapse step and same wrong goal in both information conditions.
- Addition and deletion select the same clause within each condition.
- Same clause across conditions.
- Semantic argmax agreement ≥3/4 in both.
- Incremental gain on the target wrong goal ≥0.15 in both.
- Deletion drop on that target ≥0.10 in both.

These are local input interventions supporting interpretive capture in the defined task. They do not establish unique clause causality, token-level mechanisms, or causal path dependence.

### 7. Validation and scoring

The figures show documented thresholds. Calibration is ≥7/8 correct with perfect CONTRADICTS recall. Family prior gate is max pick share ≤0.75 and no dead goal. Consistency requires zero contradictions, ≥40% advancing steps, and valid labels. Leakage uses first-step accuracy minus the appropriate premise prior ≤0.20. Final recoverability requires accuracy ≥0.80.

The current code uses Check 3 to assess individual cores and construct a fair scoring mask. A hidden core below 0.60 is not scored. It is not an additional blanket exclusion of an otherwise usable episode. The least-inferable core is made visible in the partial-information condition.

There are version differences in trial counts: the README/preregistration discuss n=24 for family priors, the current orchestrator defaults to n=100, and the standalone seed-prior script has its own default. Exact counts must therefore come from the chosen run manifest rather than be inferred from a slide. The current orchestrator also references `filter_seeds.py`, which was not present in the inspected repository root; no claim of an end-to-end successful rerun is made.

### 8. Secondary experiments and extension status

Prompting ablations compare Direct, CoT-Matched, and Latent-First (including versioned variants), at Level 1 with cores visible and Level 2 with partial core visibility. These use separate predictor scripts and are distinct from the direct single-token instrument.

The confirmatory specification uses 16 new families ×4 goal rotations =64 generated episodes, with no regeneration of failures, exclusion of pilot/replication IDs, Qwen/Qwen2.5-7B-Instruct as the default primary full-vocabulary probe, and gpt-4o-mini for transfer checks.

Preregistered targets: H1 ≥3 validated triggers; H1b ≥50% cross-model addition selection among primary validated clauses; H2 median neutral-replacement recovery fraction ≥0.5; H3 positive mean upper-layer wrong-goal mass difference for a majority of triggers, with one-sided Wilcoxon p<0.05 if n≥5. The neutral replacement bank uses deterministic length matching. The mechanistic script applies final normalization and unembedding at the answer position across layers, averaged over four label rotations, with versus without the trigger clause. A logit-lens readout is not proof of a causal workspace mechanism.

The confirmatory prediction horizon is branch-relative. Older exploratory analyses used different horizon rules. Planned claims require actual result artifacts.

## Verification of the figures

- Source-level definitions and example values were checked against the repository.
- SVG element nesting and canvas/card text bounds were checked programmatically using conservative approximate text widths.
- Illustrative probability rows sum to one.
- No browser rendering or exported PNG/PDF was executed in the creation session; the HTML supplies browser-based export.
- All text and shapes are vector elements. No generated image was used for technical labels, equations, or data.
