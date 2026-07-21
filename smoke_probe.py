"""smoke_probe.py — pre-flight check of the open-weights instruments."""
import json, sys
from judges import CallLogger, load_episodes
from probe_openweights import OpenWeightsProbe, run_steps
from goal_distribution import prefix_text

episodes = {e.episode_id: e for e in load_episodes(sys.argv[1])}
keep = [r["episode_id"] for r in json.load(open(sys.argv[2]))["results"]
        if r["usable"]][:2]
probe = OpenWeightsProbe()
ok1 = all(len(v) > 0 for v in probe.label_token_ids.values())
chars = {eid: "" for eid in keep}
r1 = run_steps(probe, episodes, keep, chars, CallLogger(None))
r2 = run_steps(probe, episodes, keep, chars, CallLogger(None))
covs = [p["min_coverage"] for r in r1 for p in r["points"]]
ok2 = min(covs) > 0.95
ok3 = json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
from mech_readout import LogitLens
lens = LogitLens(model_name=probe.model_name)
ep = episodes[keep[0]]
goals = [ep.hidden_goal] + list(ep.meta.get("distractor_goals", []))
curves = lens.goal_layer_curves(prefix_text(ep, 2), goals)
nl = probe.model.config.num_hidden_layers
ok4 = len(curves) == nl + 1 and all(abs(sum(c.values())-1) < 1e-3 for c in curves)
for i, (ok, msg) in enumerate([(ok1, "label tokens A-D found"),
                               (ok2, f"min_coverage>0.95 (min={min(covs):.3f})"),
                               (ok3, "deterministic across runs"),
                               (ok4, f"logit-lens {len(curves)} curves, normalized")]):
    print(f"[{i+1}] {msg}: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if all([ok1, ok2, ok3, ok4]) else 1)