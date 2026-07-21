# Confirmatory runbook (steps 1–4)

Order matters; each stage's output feeds the next. Freeze
PREREGISTRATION.md and DECISIONS.md in the same commit as the code,
BEFORE running stage B.

## A. Environment
    pip install -r requirements.txt
    export OPENAI_API_KEY=...            # generator + judges + secondary probe
    export GEMINI_API_KEY=...            # only if rerunning cross-model sweep
    # optional: export MORIARTY_OW_MODEL=meta-llama/Llama-3.1-8B-Instruct
    #           (gated: huggingface-cli login first). Default is
    #           Qwen/Qwen2.5-7B-Instruct (ungated). ~16 GB VRAM in bf16;
    #           on CPU set MORIARTY_OW_DTYPE=float32 (slow but exact).

## B. Generate + gate the confirmatory episodes (Step 1)
    python run_pipeline.py --name confirmatory_v4 \
        --seeds seeds_v4_confirmatory.json --skip-l1 --skip-l2
    # -> episodes json + checks report + core report for the new set.
    # Do NOT regenerate failed episodes (preregistration §1).

## C. Open-weights trajectory probe, both observers (Step 3)
    python probe_openweights.py EP.json --report CHECKS.json \
        --mode steps --out ow_steps_naive.json
    python probe_openweights.py EP.json --report CHECKS.json \
        --mode steps --observer agent_b --core-report CORES.json \
        --out ow_steps_agentb.json
    python probe_openweights.py EP.json --report CHECKS.json \
        --mode clauses --steps-report ow_steps_naive.json \
        --out ow_clauses_naive.json
    python probe_openweights.py EP.json --report CHECKS.json \
        --mode clauses --observer agent_b --core-report CORES.json \
        --steps-report ow_steps_agentb.json --out ow_clauses_agentb.json

## D. Frozen filter on new data only (Step 2)
    python confirmatory_triggers.py \
        --steps-naive ow_steps_naive.json --steps-agentb ow_steps_agentb.json \
        --clauses-naive ow_clauses_naive.json --clauses-agentb ow_clauses_agentb.json \
        --exclude episodes_v3.json <replication_episodes>.json \
        --prereg PREREGISTRATION.md --out confirmatory_triggers.json

## E. Cross-model transfer (H1b): repeat C with the API probe
    python goal_distribution.py EP.json --report CHECKS.json \
        --mode steps --out api_steps_naive.json
    # ... (same four calls as C with goal_distribution.py), then D again
    # with the api_* files -> api_confirmatory_triggers.json; compare
    # validated clause sets.

## F. Neutral-replacement control (H2)
    python probe_openweights.py EP.json --report CHECKS.json \
        --mode replace --triggers confirmatory_triggers.json \
        --out ow_replace_naive.json

## G. Mechanistic readout (H3)
    python mech_readout.py EP.json --triggers confirmatory_triggers.json \
        --out mech_readout.json

Costs: B ≈ a few dollars of gpt-4o-mini; C/F/G are local compute only
(~64 eps × 7 prefixes × 4 perms ≈ 1.8k forward passes for steps mode —
minutes on one GPU); E ≈ the pilot probe cost again.
