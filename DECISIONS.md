# DECISIONS.md — instrument and design decisions, dated

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
