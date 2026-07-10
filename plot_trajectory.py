"""
plot_trajectory.py — Figure 1: interpretive-capture trajectory for one episode.

Plots true-goal probability (q), validated wrong-goal probability, and entropy (H)
across prefix steps for naive and Agent-B-information probes.

Usage:
    python goal_distribution.py results/<run>/episodes.json \\
        --report results/<run>/checks_report.json --mode steps \\
        --observer naive --out results/<run>/gd_steps_naive.json
    python goal_distribution.py results/<run>/episodes.json \\
        --report results/<run>/checks_report.json --mode steps \\
        --observer agent_b --core-report results/<run>/core_inferability_report.json \\
        --out results/<run>/gd_steps_agentB.json
    python plot_trajectory.py results/<run>/gd_steps_naive.json \\
        results/<run>/gd_steps_agentB.json \\
        --episode f03_product_team_g3 \\
        --validated validated_triggers.json \\
        --out results/<run>/fig_f03_trajectory.png
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_episode(path: str, episode_id: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for row in data["results"]:
        if row["episode_id"] == episode_id:
            return row
    raise KeyError(f"{episode_id} not found in {path}")


def _match_goal(goal_keys: list[str], prefix: str) -> str:
    for g in goal_keys:
        if g.startswith(prefix) or prefix.startswith(g[:60]):
            return g
    for g in goal_keys:
        if prefix[:40] in g:
            return g
    raise KeyError(f"no goal matching prefix {prefix!r} among {goal_keys}")


def _series(points: list[dict], wrong_goal: str | None) -> dict[str, list]:
    ts = [pt["t"] for pt in points]
    q = [pt["q"] for pt in points]
    h = [pt["H"] for pt in points]
    if wrong_goal:
        p_wrong = [pt["p"][wrong_goal] for pt in points]
    else:
        p_wrong = [pt["W"] for pt in points]
    return {"t": ts, "q": q, "p_wrong": p_wrong, "H": h}


def main() -> None:
    p = argparse.ArgumentParser(description="Plot interpretive-capture trajectory.")
    p.add_argument("naive_steps", help="gd_steps JSON from --observer naive")
    p.add_argument("agentb_steps", help="gd_steps JSON from --observer agent_b")
    p.add_argument("--episode", default="f03_product_team_g3")
    p.add_argument("--validated", default=None,
                   help="validated_triggers.json (for wrong-goal id + branch step)")
    p.add_argument("--out", default="fig_trajectory.png")
    args = p.parse_args()

    naive = _load_episode(args.naive_steps, args.episode)
    agentb = _load_episode(args.agentb_steps, args.episode)

    branch_step = naive.get("wrong_collapse_step") or naive.get("interpretation_branch")
    wrong_prefix: str | None = None
    trigger_clause: str | None = None
    if args.validated:
        with open(args.validated, "r", encoding="utf-8") as f:
            triggers = json.load(f)["validated"]
        trig = next(t for t in triggers if t["episode_id"] == args.episode)
        branch_step = trig["step"]
        wrong_prefix = trig["wrong_goal"]
        trigger_clause = trig.get("trigger_clause")

    sample_keys = list(naive["points"][0]["p"].keys())
    wrong_goal = _match_goal(sample_keys, wrong_prefix) if wrong_prefix else None

    s_naive = _series(naive["points"], wrong_goal)
    s_agent = _series(agentb["points"], wrong_goal)

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    fig.suptitle(
        f"Figure 1: Illustrative interpretive-capture trajectory for {args.episode}",
        fontsize=11,
        y=0.98,
    )

    panels = [
        ("q", "True-goal probability", "#2a7"),
        ("p_wrong", "Validated wrong-goal probability", "#c33"),
        ("H", "Entropy (nats)", "#36c"),
    ]

    for ax, (key, ylabel, color) in zip(axes, panels):
        ax.plot(s_naive["t"], s_naive[key], "o-", color=color, lw=2,
                label="naive probe", ms=5)
        ax.plot(s_agent["t"], s_agent[key], "s--", color=color, lw=2,
                alpha=0.85, label="Agent-B probe", ms=5)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        if branch_step is not None:
            ax.axvline(branch_step, color="k", ls=":", lw=1.2, alpha=0.55)
        for s in naive.get("committed_wrong_steps", []):
            ax.axvspan(s - 0.45, s + 0.45, color="#c33", alpha=0.08)

    axes[-1].set_xlabel("prefix step t")
    axes[-1].set_xticks(range(0, 7))
    axes[0].legend(loc="center right", fontsize=9)

    if branch_step is not None:
        axes[0].text(
            branch_step + 0.08, 0.55,
            f"wrong collapse @ {branch_step}",
            fontsize=8, color="0.3",
        )
    if wrong_goal:
        short = wrong_goal[:72] + ("..." if len(wrong_goal) > 72 else "")
        fig.text(0.5, 0.01, f"Validated wrong goal: {short}", ha="center",
                 fontsize=8, color="0.35", wrap=True)
    if trigger_clause:
        fig.text(0.5, 0.035, f"Trigger clause: {trigger_clause[:90]}...",
                 ha="center", fontsize=7.5, color="0.4", style="italic")

    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(args.out, dpi=180)
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
