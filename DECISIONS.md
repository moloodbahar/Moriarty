# DECISIONS.md — instrument and design decisions, dated

## 2026-07-09 — OUTCOME: interpretive capture localized; five validated triggers (verified)

Logprob instrument (goal_distribution.py, exact posteriors via
top_logprobs, 4-permutation averaging, t0 baseline, reliability gates)
run at steps + clause level, naive and Agent-B-information observers, on
full_run_jul9. All review-claimed numbers independently verified from
raw outputs. Cross-observer: wrong-collapse detected by both in 12
episodes, same step 11/12, SAME SPECIFIC WRONG GOAL 11/11; uncertainty
creation same step 8/8; resolution same step 10/12. Clause level:
within-observer add=delete agreement (wrong triggers) 8/15 naive, 13/16
agent_b. Strict two-tier filter -> validated_triggers.json: 5 VALIDATED
wrong-attractor trigger clauses (f01_charity_g2@1, f02_library_g2@1,
f03_product_team_g3@2, f04_chess_club_g3@1, f08_wedding_g3@1; dp_target
up to +0.87 naive / +0.74 agent_b, deletion reverses), 6 candidates with
named failed checks. On-record prediction confirmed: B's information
state enters the same wrong trajectory the story digs, at the same step,
onto the same goal, via the same clause in the validated subset. New
observation: measurable resolution lag (f03: naive resolves t4, agent_b
t5). Claim boundaries recorded: clause-level output-distribution probe
(not token-level, not internals, not B's reasoning trace); no unique-
causality claim from AGREE; object is goal-hypothesis distribution
("interpretive branching"), not free-form continuation diversity.
Protocol-relativity caveats: f05_estate_g2 not committed-wrong and
f04_chess_club_g4 unresolved (q=0.19 at t6) under this protocol despite
old-gate passage — exemplar figures must be labeled per instrument;
confirmatory gating should use the logprob instrument.

## 2026-07-09 — OUTCOME: exploratory sweep — partial support, one reframe

Against the pre-stated criteria (LF lower recovery AND higher lock-in
than CoT, both models): PARTIAL. gpt-4o-mini full signature (recovery
0.38 vs 0.71; lock-in 0.62 vs 0.29); gemini recovery-only (0.60 vs 0.83;
lock-in 2/10 vs 2/12 = null). Not confirmation; directional in one model.
REFRAME (recorded): sweep calls are stateless across prefixes, so lock-in
measures per-call re-derivation of the same wrong goal from post-branch
text — an instruction-induced integration failure, not cross-time
anchoring; the multi-turn (belief-persistence) version is a distinct v5
design. STRONGEST PATTERN (exploratory, consistent both models):
branch-aligned curves show LF lagging at +1..+3 after information release
(e.g., +1: 0.75 vs 0.88/0.88 and 0.75 vs 0.81/0.81) — the deficit is
slower post-branch integration, not worse reading generally. Secondary:
CoT is best recoverer in both models (deliberation dividend is
branch-localized — refines F2); self-reported confidence uninformative
(0.85±0.05 across accuracy 0.14–1.00, identical across conditions).

## 2026-07-09 — AMENDMENT to sweep pre-registration (before results known)

The exploratory sweep runs on the NEW full_run_jul9 sample (21 usable
episodes, fresh generation, n=24 priors) with branch points computed from
that run's checks report — not the original 23 episodes named in the
2026-07-08 entry. Deviation reason: the replication run superseded the old
sample as the current dataset; testing the mechanism on episodes the
predictors have never touched is strictly cleaner. Metrics, prediction,
and constraints from the 2026-07-08 entry unchanged. Sweep-side note: 7
of 21 episodes branch at step 2 (= min prefix) and enter no recovery
denominator.

## 2026-07-09 — OUTCOME: independent replication (full_run_jul9)

Full pipeline rerun end-to-end on freshly generated episodes (same
seeds_v3_1, same prompts, n=24 priors, versioned results/ folder).
ENVIRONMENT REPLICATED: usable 21/40 (was 23/40); pooled curve
0.17→0.55→0.63→0.66→0.79→0.99, t*=2 by the same rule; 16/21 usable
episodes below chance at step 1 (was 18/23); failure walls again
two-sided (9 lift, 9 unreachable); calibration 8/8 stable; constraint
exercise replicated (18 low cores, again concentrated on negative
constraints; f02_library_g1 hit the zero-scorable-hidden-cores edge case,
handled as designed); branch structure replicated (bimodal: 7 at step 2,
9 at steps 5–6; committed-wrong 9/21).
L2 DEFICIT REPLICATED: latent_first vs cot_matched paired — gemini
1-vs-5 (was 0-vs-5), gpt-4o-mini 2-vs-4 (was 2-vs-5). Four cells, two
independent samples, one direction. Exploratory post-hoc pool (gemini
across samples): 1-vs-10, exact p≈0.012.
L1 DID NOT REPLICATE IN EITHER DIRECTION: new-sample gemini L1 shows
LF-v1 5-vs-1 OVER CoT (CoT collapsed to 0.33) — the same comparison has
now produced 0-vs-4, 2-vs-3, and 5-vs-1 across three runs. Conclusion:
at Level 1 there is no stable ordering among reasoning conditions at
this n with these models; single-run L1 directions (including ones
favoring our hypotheses) must not be cited. Stable facts only: Direct
never loses (both levels, all runs); LF loses at L2 (both samples).
DOWNGRADED: rotation-position attrition skew (7/8/3/5 old vs 4/7/5/5
new) does not track positions across samples — sampling noise, not a
systematic confound; report claim softened accordingly.

## 2026-07-08 — EXPLORATORY prefix sweep: metrics stated before the run

Labeled exploratory (the pilot's pre-registered questions are closed; this
tests the premature-commitment MECHANISM hypothesis from the recovered
traces). Same 23 episodes, Level-2 masking, prefixes 2–6 in one internally
paired run, conditions direct/cot_matched/latent_first, 690 calls.
Pre-stated metrics against branch steps from branch_points.py:
RECOVERY = P(correct at first post-branch prefix | wrong at last
pre-branch prefix); LOCK-IN = P(same wrong choice repeated post-branch |
wrong pre-branch). Prediction on record: latent_first shows lower recovery
and higher lock-in than cot_matched. Noted constraints: 9 episodes branch
at step 2 (= min prefix) and have no pre-branch observation, so recovery
denominators are small (~≤14 per model-condition) — directional readout
only; and the branch-point critique does NOT extend to the LF-vs-CoT
deficit, which was measured paired at identical prefixes (horizon choice
explains the absent deliberation dividend, not the LF disadvantage).

## 2026-07-07 — OUTCOME: prior-robustness re-gate; conclusions unchanged

User-initiated audit verified the full hash chain (seeds_v3_1 → episodes_v3
→ n=12 priors → checks_report_v3 → predictions) with zero mismatches and
zero used episodes failing episode-level gates. Known weakness addressed:
priors re-measured at n=24 (seed_priors_v3_1_n24.json) and the stored
curves re-gated (regate.py, zero new leakage calls): usable 23→22, flips
= f03_product_team_g4 (in; conservative exclusion under n=12 dead-member
prior; no predictions exist for it and none are added post hoc),
f06_classroom_g1 (out; lift 0.18–0.22 straddles the 0.20 gate),
f08_wedding_g3 (out; prior 0.83→0.33 — run-level prior instability,
recorded as an instrument finding). Sensitivity on the 21-episode robust
core: all paired directions unchanged; designated primary identical
(L2 gemini 0-vs-5, p=0.062); L2 4o-mini 2-vs-4 (p=0.688); L1 LF-v2
1-vs-4 (p=0.375). Pilot conclusions stand. Confirmatory standards
adopted: n=24+ priors averaged over ≥2 independent runs, versioned
output filenames, git for the working directory.

## 2026-07-07 — OUTCOME (Level 2): pre-registered hypothesis falsified; prompting route closed

Paired latent_first vs cot_matched at Level 2, same 23 episodes, t*=2:
gemini-2.5-flash (cross-model, designated primary) 0-vs-5 discordant,
p=0.062; gpt-4o-mini 2-vs-5, p=0.453. Accuracies: LF 0.26/0.35 vs CoT
0.48/0.48; Direct 0.43/0.52. The information-saturation explanation of the
Level-1 null predicted the gap would move toward LF under partial
observability; it moved away. Secondary: hidden-core inference is 0.76-0.80
in ALL conditions (n=41 scored cells per condition-model) — trait
inference is easy, condition-invariant, and dissociated from goal
inference. Per the pre-registration above: the prompting-route conclusion
is negative at both levels; the pilot write-up covers both; the training
route (SFT/RL) is NOT automatically licensed — its go/no-go criterion
(prompted latent inference showing traction) was not met and any training
proposal must be re-justified as a new hypothesis.

## 2026-07-07 — PRE-REGISTRATION: Level-2 (partial observability) pilot

Motivated BY the Level-1 null, stated before any Level-2 call is made:
Level-1's information saturation (all cores + candidates given; Direct
beat all reasoning) is hypothesized to have left nothing latent to infer.
Level 2 hides 2 of 3 moral cores from B on the SAME 23 validated episodes,
same t*=2, same conditions (direct / cot_matched / latent_first
author-level). Visibility rule: least-inferable core (per Check 3) is
visible; hidden cores below the 0.6 inferability gate are hidden but not
scored. Primary pre-registered comparison: paired latent_first vs
cot_matched on goal choice at Level 2 (exact McNemar, cross-model first);
the Level-2 EFFECT is this difference contrasted with its Level-1 value.
Secondary: core-inference accuracy by condition; target accuracy
(first-name heuristic, recorded as heuristic). Exploratory: confidence
calibration. One run; no prompt revisions on this data; if latent_first
again fails to beat cot_matched, the program-level conclusion for the
prompting route is negative and the write-up covers both levels.

## 2026-07-07 — OUTCOME: LF-v2 did not reverse the direction; the null stands

Pre-commitment honored. On the same 23 usable episodes, within-run paired:
latent_first_v2 vs cot_matched = 1-vs-4 discordant (gpt-4o-mini, p=0.375,
exact McNemar) and 2-vs-3 (gemini-2.5-flash, p=1.0); accuracies 0.35 vs
0.48 / 0.35 vs 0.39. LF-v1 rerun: 2-vs-3 in both models (first run's
0-for-7 was partly API run-noise; both providers are nondeterministic at
temperature 0). Direct (no reasoning) ties or beats all reasoning
conditions in both models. Pilot conclusion: NO EVIDENCE that explicit
latent-state scaffolding (character-level or author-level framing)
improves goal identification over budget-matched CoT at the pre-registered
information-poor prefix (t*=2). No further LF revisions on this data.
Caveats recorded: n=23, single prefix, two small models, LF-v2 mean
reasoning length below CoT (residual budget confound), run-to-run API
variance. Design lesson: the horizon rule placed the primary comparison
at a prefix where Direct's performance shows no extractable
deliberation signal exists; the confirmatory study needs a revised,
newly pre-registered horizon rule (this pilot is closed).

## 2026-07-06 — Order deviation logged: generation ran despite Check 0 = 1/10

seeds_v3_1 generation and judging proceeded although the Check 0 gate
failed. Consequence analysis: lift-based Check 2 was designed to absorb
prior asymmetry (measured π̂ is subtracted per episode; rotation balances
residuals), so asymmetric families are tolerable. Residual exposure:
(a) true-goal-dead episodes face a stricter effective gate (lift with
prior 0 = raw step1 ≤ lift_max) — conservative; (b) dead distractors
mildly inflate final-prefix accuracy (effective k < 4) — documented pilot
limitation, reported by analyze_checks.py, not gated.

## 2026-07-06 — Check 0 precision: n_trials = 24 for any priors run that feeds the lift gate

At n = 12, π̂ has binomial σ up to ≈ 0.14 (evidence: unchanged families
f02/f10 flipped OK↔LEAK between runs), which consumes most of the 0.20
lift budget. Priors runs used for gating now use 24 trials/family.
Measurement-precision change, not a threshold change. Dead-member flags at
low n are treated as warnings (a true share of 0.10 shows 0/12 with ~28%
probability); hard dead-member conclusions require the 24-trial run.

## 2026-07-06 — Position update: hand-designed family symmetry abandoned

Three consecutive hand-design attempts (v2 covert rule, v3 neutralized
worlds, v3.1 core-engagement symmetry) failed to produce balanced pick
shares. Conclusion: Check 0 is a measurement instrument (π̂ estimation)
rather than a symmetry gate; exchangeability is pursued via rotation +
lift, not via wording perfection. Family-level hard requirements that
remain: max_share ≤ 0.75 and viable members (confirmed at n = 24).

## 2026-07-06 — Check 2 v2: lift-based step-1 gating (defined BEFORE any v3 episode exists)

**Finding.** Check 0 on seeds_v3 (pick-share mode, n=8 trials/family) showed
7-8/10 families with a dominant member (max_share 0.62–1.00) and several
dead members (share = 0.00). Judge rationales show the mechanism: the naive
model picks the goal that most engages the cast's moral-core tensions.

**Instrument problem identified.** Check 2's raw step-1 accuracy conflates
two quantities: the environment prior (how guessable the true goal is from
world+cast alone — Check 0's jurisdiction) and the information the STORY
adds at step 1 (the generator leakage Check 2 is supposed to measure).

**Decision.** Check 2 step-1 gate becomes lift-based when a priors report
is available: step1_accuracy − prior(true goal) ≤ lift_max (0.20).
Eventual-inferability gate unchanged (final ≥ 0.8). Raw step1 remains
recorded in every report. Check 0 family gate becomes: max_share ≤ 0.75
(hard ceiling) AND no dead members; residual asymmetry below the ceiling is
balanced across the dataset by goal rotation.

**Pre/post-hoc status.** This is a redefinition of a manipulation-check
instrument, made after Check 0 results on seeds but BEFORE any v3 episode
was generated and before any predictor condition has ever been run. No
episode has been rescued by this change. H1, its conditions, and its primary
metric are untouched.

## 2026-07-06 — Seed design principle v3.1: core-engagement symmetry

Family members must share one grammatical schema ("covertly cause PERSON to
do ACTION while believing it was their own") and each member must target the
core tension of a DIFFERENT cast member, so that "the goal engaging the most
core tension" is not unique. Discovered from Check 0 rationales; replaces
the failed v2 rule ("covert + same characters + neutral opening"), which
addressed distractor covertness but not core-affinity.

## 2026-07-06 — Prior history (for the paper's methods narrative)

- v1 seeds + A_v1: usable 1/10; step-1 leakage 9/10 (raw gate).
- v2 seeds (covert distractors) + Check 0 legacy: prior leak 7/10 — proved
  the world/cores themselves encode the goal (they were designed downstream
  of it).
- v3 families + Check 0 pick-share: dominant members + dead members →
  core-engagement mechanism identified from rationales.
All numbers, prompts, and raw calls preserved in versioned reports/logs.
