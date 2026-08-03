"""Generate the pilot exemplar trajectory figure (fig_f03_trajectory) for
the Moriarty preprint, in the same visual style as fig_coop_trajectory.

Usage:
    python make_f03_figure.py <pilot_run_dir>          # real data (preferred)
    python make_f03_figure.py --fallback               # paper-table values only

<pilot_run_dir> must contain gd_steps_naive.json and gd_steps_agentB.json
with the same schema as the confirmatory run. Outputs
fig_f03_trajectory.png and .pdf in the current directory.
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPISODE = "f03_product_team_g3"
C_TRUE, C_WRONG = "#1a7a3a", "#b02318"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})


def load_series(run_dir):
    """Read q_t and W_t for both observers from the pilot run JSONs."""
    out = {}
    for name, key in [("gd_steps_naive.json", "naive"),
                      ("gd_steps_agentB.json", "agentB")]:
        with open(os.path.join(run_dir, name)) as f:
            rec = {r["episode_id"]: r for r in json.load(f)["results"]}[EPISODE]
        pts = rec["points"]
        out[key] = {
            "t": [p["t"] for p in pts],
            "q": [p["q"] for p in pts],
            "W": [p["W"] for p in pts],
            "committed": rec.get("committed_wrong_steps") or [],
            "collapse": rec.get("wrong_collapse_step"),
        }
    return out


def fallback_series():
    """Values reported in the preprint (appendix trajectory table + text).

    W_t is only published at the collapse step (0.925 / 0.933); the other
    W values here are visually plausible placeholders bounded by 1 - q_t.
    Replace with real data before using this figure in the paper.
    """
    t = list(range(7))
    naive_q = [0.440, 0.527, 0.004, 0.195, 0.767, 0.999, 1.000]
    agent_q = [0.137, 0.231, 0.007, 0.028, 0.715, 0.988, 1.000]
    naive_W = [0.30, 0.30, 0.925, 0.55, 0.15, 0.001, 0.000]
    agent_W = [0.45, 0.50, 0.933, 0.70, 0.20, 0.010, 0.000]
    return {
        "naive": {"t": t, "q": naive_q, "W": naive_W,
                  "committed": [2], "collapse": 2},
        "agentB": {"t": t, "q": agent_q, "W": agent_W,
                   "committed": [2], "collapse": 2},
    }


def main():
    if len(sys.argv) > 1 and sys.argv[1] != "--fallback":
        data, provisional = load_series(sys.argv[1]), False
    else:
        data, provisional = fallback_series(), True
        print("WARNING: using paper-table fallback values; W_t is only "
              "exact at the collapse step. Regenerate from the pilot run "
              "before publication.")

    fig, ax = plt.subplots(figsize=(7.2, 3.9))

    # shaded committed-wrong region (naive), if present
    committed = data["naive"]["committed"]
    if committed:
        lo, hi = min(committed) - 0.5, max(committed) + 0.5
        ax.axvspan(lo, hi, color=C_WRONG, alpha=0.07, zorder=0)

    for key, ls, lw in [("naive", "-", 2.2), ("agentB", "--", 1.8)]:
        d = data[key]
        ax.plot(d["t"], d["q"], ls, color=C_TRUE, lw=lw,
                marker="o", ms=4, zorder=3)
        ax.plot(d["t"], d["W"], ls, color=C_WRONG, lw=lw,
                marker="s", ms=4, zorder=3)

    # annotations
    col = data["naive"]["collapse"]
    if col is not None:
        ax.annotate("shared wrong-goal collapse\n(same wrong goal, both observers)",
                    xy=(col, data["naive"]["W"][col]),
                    xytext=(col + 0.55, 0.86), fontsize=8.5,
                    arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("uncertainty reopens", xy=(3, data["naive"]["q"][3]),
                xytext=(2.85, 0.40), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("resolution", xy=(5, data["naive"]["q"][5]),
                xytext=(4.35, 0.72), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", lw=0.8))

    # legend built manually so line style and color read independently
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=C_TRUE, lw=2.2, marker="o", ms=4,
               label=r"true goal $q_t$"),
        Line2D([], [], color=C_WRONG, lw=2.2, marker="s", ms=4,
               label=r"max wrong goal $W_t$"),
        Line2D([], [], color="k", lw=2.0, ls="-", label="naive observer"),
        Line2D([], [], color="k", lw=1.6, ls="--",
               label="Agent-B information"),
    ]
    ax.legend(handles=handles, fontsize=8.5, frameon=False,
              loc="center left", bbox_to_anchor=(1.01, 0.5))

    ax.set_xlabel("story step (0 = no-story prior)")
    ax.set_ylabel("probability")
    ax.set_xticks(data["naive"]["t"])
    ax.set_ylim(-0.04, 1.06)
    title = f"Interpretive capture and recovery: {EPISODE}"
    if provisional:
        title += "  [PROVISIONAL W]"
    ax.set_title(title, fontsize=10, loc="left")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"fig_f03_trajectory.{ext}", bbox_inches="tight")
    print("wrote fig_f03_trajectory.png / .pdf")


if __name__ == "__main__":
    main()
