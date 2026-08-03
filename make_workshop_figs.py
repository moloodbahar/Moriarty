"""Generate workshop figures from confirmatory_v4_3 goal_distribution outputs.

Usage: python make_workshop_figs.py <run_dir>
Expects gd_steps_naive.json, gd_steps_agentB.json, gd_clauses_naive.json,
gd_clauses_agentB.json in <run_dir>. Writes fig_*.pdf/png next to the script.
"""
import json, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else "."
def load(name):
    with open(os.path.join(RUN, name)) as f:
        return {r["episode_id"]: r for r in json.load(f)["results"]}

gsn, gsb = load("gd_steps_naive.json"), load("gd_steps_agentB.json")
gcn, gcb = load("gd_clauses_naive.json"), load("gd_clauses_agentB.json")

C_TRUE, C_WRONG, C_B = "#1a7a3a", "#b02318", "#4477aa"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})

# ---------------------------------------------------------------- Figure 1
# f14_coop_g1: dual-observer capture-and-release trajectory
fig, ax = plt.subplots(figsize=(5.4, 3.0))
for d, ls, tag in [(gsn, "-", "naive"), (gsb, "--", "Agent-B info")]:
    pts = d["f14_coop_g1"]["points"]
    t = [p["t"] for p in pts]
    ax.plot(t, [p["q"] for p in pts], ls, color=C_TRUE, lw=1.8,
            label=f"$q_t$ true goal ({tag})")
    ax.plot(t, [p["W"] for p in pts], ls, color=C_WRONG, lw=1.8,
            label=f"$W_t$ max wrong goal ({tag})")
ax.axvspan(2, 5, color=C_WRONG, alpha=0.08)
ax.annotate("committed-wrong run\n(both observers)", xy=(3.5, 0.55),
            ha="center", fontsize=8, color=C_WRONG)
ax.annotate("capture\n(step 2)", xy=(2, 0.97), xytext=(1.1, 0.78),
            fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
ax.annotate("release clause\n(step 6)", xy=(6, 1.0), xytext=(4.7, 0.8),
            fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_xlabel("story step (0 = no-story prior)")
ax.set_ylabel("probability")
ax.set_xticks(range(7)); ax.set_ylim(-0.03, 1.06)
ax.legend(fontsize=7, loc="center left", frameon=False)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"fig_coop_trajectory.{ext}", bbox_inches="tight")

# ---------------------------------------------------------------- Figure 2
# f18_rescue_g2: clause anatomy of the validated capture trigger
def wrong_target(d):
    return [x for x in d["f18_rescue_g2"]["targets"] if x["trigger_type"] == "wrong"][0]

fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
# Short ticks only — long glosses go in a shared caption (avoids overlap on panel b).
tick_labels = ["c1", "c2", "c3"]
clause_gloss = (
    "c1: sunrise, team gathers    "
    "c2: Emil checks gear, Shirin organizes    "
    "c3: Dov paces, eager for first call-out"
)
for ax, (tgt, tag, col) in zip(axes, [(wrong_target(gcn), "naive", C_WRONG),
                                      (wrong_target(gcb), "Agent-B info", C_B)]):
    inc = [r for r in tgt["rows"] if r["kind"] == "incremental"]
    dele = [r for r in tgt["rows"] if r["kind"] == "deletion"]
    x = np.arange(3)
    p_after = [r["p_target_wrong"] for r in inc]
    drops = [r["target_wrong_drop_on_delete"] for r in dele]
    ax.plot([0] + list(x + 1), [p_after[0] - inc[0]["dp_target_wrong"]] + p_after,
            "-o", color=col, lw=1.6, ms=4, label="p(wrong) after adding clause")
    ax.bar(x + 1, drops, width=0.45, color=col, alpha=0.35,
           label="drop on deleting clause")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x + 1)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_xlim(-0.2, 3.6)
    ax.set_ylim(-0.5, 1.0)
    ax.set_title(f"{tag} observer", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
axes[0].set_ylabel("p(target wrong goal)")
fig.suptitle("Anatomy of the validated capture trigger (f18_rescue_g2, step 1)",
             fontsize=9, y=1.02)
fig.text(0.5, -0.02, clause_gloss, ha="center", va="top", fontsize=7.5,
         color="#333")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"fig_trigger_anatomy.{ext}", bbox_inches="tight")

# ---------------------------------------------------------------- Figure 3
# Failure-wall migration across the three datasets
fig, ax = plt.subplots(figsize=(4.2, 2.4))
datasets = ["Pilot", "Replication", "Confirmatory"]
leak, unreach = [6, 9, 12], [11, 9, 0]
x = np.arange(3); w = 0.36
ax.bar(x - w/2, leak, w, color="#c47a1d", label="leakage (lift gate)")
ax.bar(x + w/2, unreach, w, color="#555", label="unreachable (final gate)")
for i, v in enumerate(unreach):
    if v == 0:
        ax.annotate("0", xy=(x[i] + w/2, 0.15), ha="center", fontsize=9,
                    fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(datasets)
ax.set_ylabel("episodes failing")
ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"fig_failure_migration.{ext}", bbox_inches="tight")

# ---------------------------------------------------------------- Figure 4
# Appendix small multiples: all 14 usable trajectories, both observers
eps = sorted(gsn.keys())
fig, axes = plt.subplots(2, 7, figsize=(12, 3.4), sharex=True, sharey=True)
for ax, e in zip(axes.flat, eps):
    for d, ls in [(gsn, "-"), (gsb, "--")]:
        pts = d[e]["points"]
        ax.plot([p["t"] for p in pts], [p["q"] for p in pts], ls,
                color=C_TRUE, lw=1.1)
        ax.plot([p["t"] for p in pts], [p["W"] for p in pts], ls,
                color=C_WRONG, lw=1.1)
    wc = gsn[e]["wrong_collapse_step"]
    if wc is not None:
        ax.axvline(wc, color=C_WRONG, lw=0.7, alpha=0.5)
    ax.set_title(e.replace("_", "\\_") if False else e, fontsize=6.5)
    ax.set_ylim(-0.05, 1.05); ax.set_xticks([0, 3, 6])
fig.text(0.5, 0.0, "story step", ha="center", fontsize=8)
fig.text(0.0, 0.5, "probability", va="center", rotation="vertical", fontsize=8)
fig.suptitle("All 14 usable trajectories: $q_t$ (green) vs $W_t$ (red); "
             "solid = naive, dashed = Agent-B info; vertical line = naive wrong collapse",
             fontsize=8, y=1.03)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"fig_all_trajectories.{ext}", bbox_inches="tight")

print("wrote fig_coop_trajectory, fig_trigger_anatomy, fig_failure_migration, fig_all_trajectories")
