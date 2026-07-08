# Moriarty Pilot Report

## Latent-First Prompting Fails Where There Is Something Latent to Infer: A Validated Narrative Environment and Two Pre-Registered Null-to-Negative Results

*Molood (Melody) Arman — draft v1, 2026-07-07. All thresholds, instrument
changes, and pre-registrations dated in DECISIONS.md before the runs they
govern; every model call archived in JSONL logs; every report carries
input hashes.*

### Summary

We built a validated environment for latent-goal inference — adversarial
narrative episodes in which an author model pursues an
experimenter-specified hidden goal under per-character moral constraints —
and used it to test, twice, whether instructing an observer model to
explicitly infer latent state before predicting ("latent-first") improves
hidden-goal identification over reasoning-length-matched chain-of-thought.
It does not. At full information (Level 1) latent-first is null-to-negative
in both framings tested; under partial observability (Level 2, two of three
moral cores hidden), where the information-saturation explanation of the
Level-1 null predicted latent-first should gain, its deficit instead
widened (cross-model paired: 0-vs-5 discordant, exact McNemar p=0.062).
Meanwhile hidden-trait inference is easy (0.76–0.80 from two story steps,
chance 0.25) and completely condition-invariant — dissociating trait
inference from goal-trajectory inference. Direct answering, with no
deliberation at all, ties or beats every reasoning condition throughout.
The pilot also yields an environment-construction methodology whose failed
intermediate versions are themselves findings: hidden-goal settings fail
on two measurable walls (early leakage vs. eventual unreachability), seed
priors leak through worlds and character constraints rather than
distractors, and latent constraints are inferable only in trajectories
that exercise them.

### 1. Environment and validation

Episodes are generated from seed specifications: a world, three
characters with one-sentence moral cores, and a **goal family** of four
schema-parallel covert goals, one of which is instantiated as the hidden
goal per episode (rotation), with the other three as same-world
distractors. The author (gpt-4o-mini, versioned prompt) realizes the goal
step-by-step over six steps; every call is logged.

An episode enters the evaluation set only after passing:

- **Judge calibration gate** — the goal-consistency judge scores 8/8 on
  hand-labeled cases (three-way ADVANCES/NEUTRAL/CONTRADICTS; gate
  requires perfect recall on CONTRADICTS), stable over 3 repeats.
- **Check 0 (seed priors)** — a naive model's pick-share over the goal
  family given only world + characters. Measured, not merely gated: the
  shares feed Check 2.
- **Check 1 (consistency)** — zero CONTRADICTS, ≥40% ADVANCES, zero
  invalid judge outputs.
- **Check 2 (leakage, lift-gated)** — a prefix-MCQ inferability curve;
  step-1 accuracy may exceed the episode's Check-0 prior by at most 0.20
  (the story must not add early information; the environment prior is
  Check 0's jurisdiction and rotation balances it), and full-prefix
  accuracy must reach 0.8 (else the goal is unrecoverable and the episode
  measures nothing).
- **Check 3 (core inferability)** — required for partial-observability
  scoring; see §4.

Result: 23/40 episodes usable. Pooled inferability over usable episodes:
0.19 → 0.43 → 0.68 → 0.73 → 0.90 → 0.99 (chance 0.25), fixing the
prediction prefix at t\*=2 by a rule registered before any predictor ran.
Notably, step-1 accuracy pools *below chance*: under rotation, the
premise-modal goal is usually false, so the environment prior actively
misleads — 18/23 usable episodes sit below chance at step 1, and 11 have
step-1 accuracy below their own prior (the author's opening actively
suppresses the prior).

**Methodology findings from the failed versions** (all runs preserved):
(i) v1 seeds leaked at step 1 in 9/10 episodes; (ii) redesigned
distractors did not fix priors because worlds and moral cores — designed
downstream of the goal — encode it (naive-judge rationales explicitly
cite core tensions); (iii) hand-designed family symmetry failed three
times; the working design combines rotation, prior *measurement*, and
lift-gating rather than wording perfection; (iv) the dominant failure
mode then flipped from leakage to unreachability (11 episodes whose goals
never became inferable), exposing a concealment–reachability tension that
the two gates now measure from both sides; (v) attrition is skewed across
rotation positions (7/8/3/5 of 10), a documented confound for the
confirmatory design.

### 2. Pre-registered experiment 1 (Level 1: full information)

Observer conditions on identical prefixes and identical candidate
orderings: **Direct**, **CoT-Matched** (budget-matched step-by-step
reasoning; primary control), **Latent-First** (infer latent state, then
answer). Models: gpt-4o-mini (same-model as generator; self-simulation
risk) and gemini-2.5-flash (cross-model; designated primary). Primary
metric: k-way goal choice, exact-graded, paired, exact McNemar.

The first latent-first framing was diagnosed from reasoning traces as
mis-aimed (character-level psychology rather than authorial intent) —
its errors were sibling goals targeting the wrong character. One revision
was pre-committed and run (author-level framing, LF-v2). Within-run
results (23 episodes):

| model | Direct | CoT | LF-v1 | LF-v2 | LF-v2 vs CoT paired |
|---|---|---|---|---|---|
| gpt-4o-mini | 0.48 | 0.48 | 0.43 | 0.35 | 1-vs-4, p=0.375 |
| gemini-2.5-flash | 0.43 | 0.39 | 0.35 | 0.35 | 2-vs-3, p=1.0 |

Conclusion (registered): no evidence for latent-first at Level 1; both
framings null-to-negative; Direct competitive with everything.

### 3. Pre-registered experiment 2 (Level 2: partial observability)

Hypothesis registered before the run: Level 1 was
information-saturated (all cores and candidates given), leaving nothing
latent to infer; hiding two of three moral cores should create the regime
where latent-first can differentiate. Same 23 episodes, same t\*, same
conditions; visibility rule: the least-inferable core (per Check 3) is
visible; hidden cores below the inferability gate are hidden but never
scored. The observer outputs a structured belief state (goal choice,
maneuver target, per-character core MCQs, next-beat, confidence), all
primary/secondary fields exact-graded.

| model | Direct | CoT | LF | LF vs CoT paired | cores (all conds) |
|---|---|---|---|---|---|
| gpt-4o-mini | 0.52 | 0.48 | 0.35 | 2-vs-5, p=0.453 | 0.76–0.80 |
| gemini-2.5-flash | 0.43 | 0.48 | 0.26 | **0-vs-5, p=0.062** | 0.76 |

The prediction was directional and it failed in reverse: the latent-first
deficit widened under partial observability. Sensitivity excluding four
prior-driven ceiling episodes: unchanged direction (gemini 0-vs-4).

### 4. Findings

**F1 — Latent-first prompting does not help and likely hurts.** Across
two framings, two information levels, and two model families, explicit
latent-state scaffolding never beats budget-matched CoT and is
directionally worse in every cell; the designated-primary cross-model
comparison reaches 0-vs-5 discordant pairs.

**F2 — No deliberation dividend at short prefixes.** Direct (zero
reasoning tokens) ties or beats all reasoning conditions at both levels.
Whatever goal-relevant signal exists at t\*=2 is extracted immediately;
added deliberation is at best neutral.

**F3 — Trait–trajectory dissociation.** Hidden moral cores are inferred
at 0.76–0.80 from two steps, identically across conditions, while goal
accuracy spans 0.26–0.52 and is condition-sensitive (negatively).
Models read *who characters are* easily and un-promptably; reading
*where the plot is going* is hard, and instructed theorizing about it is
counterproductive. Mechanism hypothesis (from recovered reasoning traces;
untested): latent-first induces premature commitment to a single latent
theory that anchors the goal choice, whereas trait evidence is local and
abundant enough to be immune.

**F4 — Constraint exercise.** 17/69 character-cores failed full-story
inferability, concentrated on *negative* constraints ("never be the first
to apologize", "never let a student see you lie"): a constraint is
behaviorally visible only in trajectories that test it, and the same
character's core ranges from 1.00 to 0.40 across episodes of one family.
Constraint-side twin of the goal-side reachability wall.

**F5 — Environment methodology.** Specified-by-construction latent
state; calibrated-before-trusted judges; prior measurement + rotation +
lift-gating in place of unattainable hand-designed symmetry; masking rule
(every hidden variable is verified-inferable-and-scored or
irreducible-and-unscored); full call-level provenance.

### 5. Limitations

n=23 episodes, one narrative domain, two small models; k-way MCQ format
(not free-form prediction); a single pre-registered prefix (t\*=2) placed
by a rule that F2 suggests lands in a deliberation-free regime; LF-v2's
mean reasoning length ran below CoT (residual budget confound at Level 1);
both APIs are nondeterministic at temperature 0 (run-to-run cell variance
observed); ceiling episodes and dead distractor options mildly inflate
some cells (sensitivity analyses included); rotation-position attrition
imbalance documented in §1.

### 6. What follows

*Exploratory (labeled, not yet run):* the premature-commitment hypothesis
predicts the LF deficit should shrink as evidence accumulates — a prefix
sweep (t=3,4,5) over the same episodes tests this cheaply.

*Confirmatory design changes (new pre-registration required):* revised
horizon rule (F2 shows the current rule selects an uninformative regime);
position-stratified acceptance to repair rotation balance; 300+ episodes;
stronger models.

*Training route:* not automatically licensed. Its stated go/no-go —
prompted latent inference showing traction to amplify — was not met. A
training proposal must be re-justified as a new hypothesis (e.g., that
belief-state supervision differs mechanistically from belief-state
instruction), against this pilot's baseline.

### Reproducibility

DECISIONS.md (dated instrument changes and pre-registrations), per-run
JSONL call logs (every request and response), manifests with input
SHA-256s and full prompt texts, and the analysis scripts reproduce every
number above from the archived reports.
