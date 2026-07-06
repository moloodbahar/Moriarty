"""
plot_checks.py — Four diagnostic plots from a checks report.

Usage:
    python plot_checks.py checks_report_v3.json --out checks_v3_plots.png

Panels:
  A. Inferability curves: every episode (usable solid, excluded dashed
     gray), pooled usable curve bold, chance and final_min lines, t*.
  B. The lift gate at work: prior (x) vs raw step-1 accuracy (y).
     Diagonal = lift 0; dashed = lift gate (+lift_max). Points BELOW the
     diagonal are stories that actively suppress the environment prior.
  C. Attrition by rotation position: exclusive failure taxonomy per
     goal position (check1 > lift > final priority).
  D. Reachability: prior (x) vs final-prefix accuracy (y). A cluster of
     low-prior/low-final points = "A cannot make unlikely goals
     inferable" — the unreachability failure mode.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def category(r: dict) -> str:
    c1, c2 = r["check1"], r["check2"]
    if not c1["passed_check1"]:
        return "check1"
    if not c2["not_leaky"]:
        return "lift"
    if not c2["eventually_inferable"]:
        return "final"
    return "usable"


COLORS = {"usable": "#2a9d3a", "check1": "#8a8a8a",
          "lift": "#d62728", "final": "#e8871a"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("report")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    with open(args.report, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = data["run_config"]
    results = data["results"]
    chance = 1.0 / cfg.get("k", 4)
    final_min = cfg.get("final_min", 0.8)
    lift_max = cfg.get("lift_max", 0.20)

    cats = {r["episode_id"]: category(r) for r in results}

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    (axA, axB), (axC, axD) = axes

    # ---- A: spaghetti + pooled ----
    pooled = defaultdict(list)
    for r in results:
        xs = [pt["prefix_len"] for pt in r["check2"]["curve"]]
        ys = [pt["accuracy"] for pt in r["check2"]["curve"]]
        if cats[r["episode_id"]] == "usable":
            axA.plot(xs, ys, color=COLORS["usable"], alpha=0.35, lw=1)
            for x, y in zip(xs, ys):
                pooled[x].append(y)
        else:
            axA.plot(xs, ys, color="#bbbbbb", alpha=0.5, lw=1, ls="--")
    px = sorted(pooled)
    axA.plot(px, [sum(pooled[x]) / len(pooled[x]) for x in px],
             color="black", lw=3, label="pooled (usable)")
    axA.axhline(chance, color="gray", ls=":", label=f"chance={chance:.2f}")
    axA.axhline(final_min, color=COLORS["final"], ls=":",
                label=f"final gate={final_min}")
    axA.set_title("A. Inferability curves (usable solid, excluded dashed)")
    axA.set_xlabel("prefix length"); axA.set_ylabel("goal-ID accuracy")
    axA.set_ylim(-0.03, 1.03); axA.legend(fontsize=8)

    # ---- B: lift gate scatter ----
    for r in results:
        c2 = r["check2"]
        if c2.get("prior") is None:
            continue
        axB.scatter(c2["prior"], c2["step1_accuracy"],
                    color=COLORS[cats[r["episode_id"]]], s=45,
                    edgecolors="black", linewidths=0.4, zorder=3)
    lim = [-0.03, 1.03]
    axB.plot(lim, lim, color="black", lw=1, label="lift = 0")
    axB.plot(lim, [v + lift_max for v in lim], color=COLORS["lift"],
             ls="--", lw=1, label=f"lift gate (+{lift_max})")
    axB.fill_between(lim, [v + lift_max for v in lim], 1.03,
                     color=COLORS["lift"], alpha=0.06)
    axB.set_xlim(lim); axB.set_ylim(lim)
    axB.set_title("B. Lift gate: prior vs raw step-1 accuracy")
    axB.set_xlabel("environment prior (Check 0 pick-share of true goal)")
    axB.set_ylabel("step-1 accuracy (with story)")
    axB.legend(fontsize=8)

    # ---- C: attrition by rotation position ----
    by_pos = defaultdict(Counter)
    for r in results:
        eid = r["episode_id"]
        pos = "g" + eid.rsplit("_g", 1)[-1] if "_g" in eid else "?"
        by_pos[pos][cats[eid]] += 1
    positions = sorted(by_pos)
    bottoms = [0.0] * len(positions)
    for cat in ["usable", "final", "lift", "check1"]:
        vals = [by_pos[p][cat] for p in positions]
        axC.bar(positions, vals, bottom=bottoms, color=COLORS[cat], label=cat)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    axC.set_title("C. Attrition by rotation position (exclusive categories)")
    axC.set_ylabel("episodes"); axC.legend(fontsize=8)

    # ---- D: reachability ----
    for r in results:
        c2 = r["check2"]
        if c2.get("prior") is None:
            continue
        axD.scatter(c2["prior"], c2["final_accuracy"],
                    color=COLORS[cats[r["episode_id"]]], s=45,
                    edgecolors="black", linewidths=0.4, zorder=3)
    axD.axhline(final_min, color=COLORS["final"], ls=":",
                label=f"final gate={final_min}")
    axD.set_xlim(lim); axD.set_ylim(lim)
    axD.set_title("D. Reachability: prior vs final-prefix accuracy")
    axD.set_xlabel("environment prior of true goal")
    axD.set_ylabel("final-prefix accuracy")
    axD.legend(fontsize=8)

    fig.suptitle(f"Moriarty manipulation checks — {args.report}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = args.out or args.report.rsplit(".", 1)[0] + "_plots.png"
    fig.savefig(out, dpi=160)
    print(f"saved -> {out}")

    # console companion stats
    n_ceiling = sum(1 for r in results
                    if cats[r["episode_id"]] == "usable"
                    and r["check2"]["step1_accuracy"] >= 0.6)
    below = sum(1 for r in results if cats[r["episode_id"]] == "usable"
                and r["check2"]["step1_accuracy"] < chance)
    print(f"usable episodes with step1 >= 0.6 (prior-driven 'ceiling' items): {n_ceiling}")
    print(f"usable episodes with step1 BELOW chance (prior actively misleads): {below}")


if __name__ == "__main__":
    main()
