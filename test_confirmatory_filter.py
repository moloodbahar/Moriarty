"""test_confirmatory_filter.py — fixture test of the frozen filter.
No API, no GPU.   Run:  python test_confirmatory_filter.py"""
import json, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))

def pt(t, dom_true=True):
    return {"t": t, "q": 0.1, "H": 0.3, "W": 0.8, "p": {},
            "dominant_is_true": dom_true, "argmax_agreement": 4}

def steps(eid, wc=2):
    return {"episode_id": eid, "wrong_collapse_step": wc,
            "points": [pt(t, dom_true=(t != wc)) for t in range(7)]}

def clauses(eid, clause, dp=0.4, drop=0.3, agree=True):
    rows = [{"kind": "incremental", "n_clauses": 1, "clause_added": clause,
             "dp_target_wrong": dp, "q": .1, "H": .2, "W": .9,
             "jsd_from_prev": .5, "dq": -.3, "dW": .4},
            {"kind": "deletion", "clause_removed": clause,
             "target_wrong_drop_on_delete": drop, "q": .3, "H": .5, "W": .5,
             "q_drop_vs_full": -.2, "q_restore_on_delete": .2,
             "jsd_vs_full": .3}]
    return {"episode_id": eid, "targets": [{
        "labels": ["wrong_collapse_step"], "step": 2, "trigger_type": "wrong",
        "target_wrong_goal": "wrong goal G", "base_q": .5, "full_q": .05,
        "clauses": [clause], "rows": rows, "trigger_by_addition": clause,
        "trigger_by_deletion": clause if agree else "other",
        "triggers_agree": agree}]}

mk = lambda r: {"run_config": {"model": "fixture", "backend": "test"},
                "results": r}
run = lambda a: subprocess.run(
    [sys.executable, os.path.join(HERE, "confirmatory_triggers.py")] + a,
    capture_output=True, text=True)

with tempfile.TemporaryDirectory() as td:
    p = lambda n: os.path.join(td, n)
    json.dump(mk([steps("e1"), steps("e2"), steps("e3")]), open(p("sn"), "w"))
    json.dump(mk([steps("e1"), steps("e2"), steps("e3")]), open(p("sb"), "w"))
    json.dump(mk([clauses("e1", "trigger one"), clauses("e2", "c2", dp=0.05),
                  clauses("e3", "cA")]), open(p("cn"), "w"))
    json.dump(mk([clauses("e1", "trigger one"), clauses("e2", "c2", dp=0.05),
                  clauses("e3", "cB")]), open(p("cb"), "w"))
    json.dump([{"episode_id": "pilot_ep"}], open(p("ex"), "w"))
    json.dump([{"episode_id": "e1"}], open(p("ov"), "w"))
    base = ["--steps-naive", p("sn"), "--steps-agentb", p("sb"),
            "--clauses-naive", p("cn"), "--clauses-agentb", p("cb"),
            "--prereg", os.path.join(HERE, "PREREGISTRATION.md")]
    r = run(base + ["--exclude", p("ex"), "--out", p("out")])
    assert r.returncode == 0, r.stderr
    d = json.load(open(p("out")))
    assert [v["episode_id"] for v in d["validated"]] == ["e1"]
    fails = {c["episode_id"]: c.get("fail", "") for c in d["candidates"]}
    assert "dp_target_wrong" in fails["e2"] and "different clause" in fails["e3"]
    print("PASS  filter gates")
    r = run(base + ["--exclude", p("ov"), "--out", p("bad")])
    assert r.returncode != 0 and "CONFIRMATORY VIOLATION" in r.stderr
    print("PASS  overlap abort\nALL TESTS PASS")