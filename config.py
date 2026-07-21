"""config.py — central model configuration for the confirmatory pipeline.

Existing pilot/replication scripts keep their hardcoded models (frozen
provenance). All NEW scripts read from here, overridable via env vars,
so multi-model runs never require find-and-replace.
"""

import os

# --- API models (pilot instruments; unchanged) -------------------------
API_PROBE_MODEL = os.environ.get("MORIARTY_API_PROBE_MODEL", "gpt-4o-mini")
GENERATOR_MODEL = os.environ.get("MORIARTY_GENERATOR_MODEL", "gpt-4o-mini")
CROSS_MODEL = os.environ.get("MORIARTY_CROSS_MODEL", "gemini-2.5-flash")

# --- Open-weights probe (confirmatory instrument) ----------------------
# Any HF causal LM with a chat template works. Tested targets:
#   meta-llama/Llama-3.1-8B-Instruct   (gated; needs HF token)
#   Qwen/Qwen2.5-7B-Instruct           (ungated)
OPENWEIGHTS_MODEL = os.environ.get(
    "MORIARTY_OW_MODEL", "Qwen/Qwen2.5-7B-Instruct")
OPENWEIGHTS_DTYPE = os.environ.get("MORIARTY_OW_DTYPE", "bfloat16")
OPENWEIGHTS_DEVICE = os.environ.get("MORIARTY_OW_DEVICE", "auto")
# max context guard; episodes are short so this is generous
OPENWEIGHTS_MAX_TOKENS_CTX = int(
    os.environ.get("MORIARTY_OW_MAX_CTX", "4096"))
