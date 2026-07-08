"""
plot_branch_points.py — Figure: branch-step histogram + committed-wrong exemplar.

Usage:
    python branch_points.py checks_report_v3.json --usable-only --out branch_points_v3.json
    python plot_branch_points.py branch_points_v3.json --out branch_points_fig.png
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("branch_report")
    p.add_argument("--exemplar", default=None,
                   help="episode_id for panel B (default: auto-pick the "
                        "committed-wrong episode with the largest branch jump)")
    p.add_argument("--out", default="branch_points_fig.png")
    args = p.parse_args()

    with open(args.branch_report, "r", encoding="utf-8") as f:
        results = json.load(f)["results"]
    used = [r for r in results if r["usable"]]

    ex = None
    if args.exemplar:
        ex = next(r for r in results if r["episode_id"] == args.exemplar)
    else:
        cw = [r for r in used if r["committed_wrong_steps"]]
        ex = max(cw, key=lambda r: r["branch_jump"]) if cw else used[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    dist = Counter(r["branch_step"] for r in used if r["branch_step"])
    steps = sorted(dist)
    ax1.bar(steps, [dist[s] for s in steps], color="#2a7", width=0.6)
    ax1.set_xlabel("branch step (argmax accuracy jump)")
    ax1.set_ylabel("usable episodes")
    ax1.set_title(f"A. Branch-step distribution (n={len(used)})")
    ax1.set_xticks(range(2, 7))

    xs = [row["prefix_len"] for row in ex["curve"]]
    acc = [row["accuracy"] for row in ex["curve"]]
    ent = [row["entropy"] for row in ex["curve"]]
    ax2.plot(xs, acc, "o-", color="#2a7", label="goal-ID accuracy")
    ax2.plot(xs, ent, "s--", color="#a52", label="pick entropy (norm.)")
    ax2.axhline(0.25, color="gray", ls=":", lw=1, label="chance")
    if ex["branch_step"]:
        ax2.axvline(ex["branch_step"], color="k", lw=1, alpha=0.6)
        ax2.text(ex["branch_step"] + 0.05, 0.02, "branch", fontsize=9)
    for s in ex["committed_wrong_steps"]:
        ax2.axvspan(s - 0.5, s + 0.5, color="#c33", alpha=0.12)
    ax2.set_xlabel("prefix length")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title(f"B. Committed-wrong exemplar: {ex['episode_id']}\n"
                  f"(shaded = confident + wrong)")
    ax2.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(args.out, dpi=180)
    print(f"figure -> {args.out}  (exemplar: {ex['episode_id']})")


if __name__ == "__main__":
    main()
