"""
run_pipeline.py — Run the full Moriarty pipeline into one results folder.

Every artifact (reports, call logs, plots, analysis text) is written under
results/<run_name>/ so a fresh end-to-end run is self-contained and easy to
find later.

Usage:
    python run_pipeline.py
    python run_pipeline.py --name pilot_v3_fresh
    python run_pipeline.py --name pilot_v3_fresh --force   # reuse existing folder
    python run_pipeline.py --skip-l1 --skip-l2   # env + checks only
    python run_pipeline.py --dry-run

Requires: OPENAI_API_KEY (and GEMINI_API_KEY for predictor steps).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results"

DEFAULT_SEEDS = "seeds_v3_1.json"
DEFAULT_CALIBRATION = "calibration_cases.json"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")


def run_cmd(cmd: list[str], *, dry_run: bool = False) -> None:
    printable = " ".join(cmd)
    print(f"\n>>> {printable}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def capture_cmd(cmd: list[str], out_path: Path, *, dry_run: bool = False) -> None:
    printable = " ".join(cmd)
    print(f"\n>>> {printable}  >  {out_path.name}")
    if dry_run:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=f, stderr=subprocess.STDOUT)
    print(f"    saved {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Run Moriarty pipeline into results/<name>/")
    p.add_argument("--name", default=None,
                   help="run folder name under results/ (default: UTC timestamp)")
    p.add_argument("--seeds", default=DEFAULT_SEEDS)
    p.add_argument("--n-trials", type=int, default=24,
                   help="Check 0 trials per family (24 for gating runs)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--skip-calibration", action="store_true")
    p.add_argument("--skip-l1", action="store_true",
                   help="skip Level-1 predictors (run_predictors.py)")
    p.add_argument("--skip-l2", action="store_true",
                   help="skip Level-2 predictors (run_predictors_v4.py)")
    p.add_argument("--dry-run", action="store_true",
                   help="print commands only, do not execute")
    p.add_argument("--force", action="store_true",
                   help="reuse an existing results/<name>/ folder (overwrites "
                        "artifacts as each step runs)")
    args = p.parse_args()

    run_name = args.name or utc_stamp()
    run_dir = RESULTS_ROOT / run_name
    py = sys.executable

    if not args.dry_run:
        if run_dir.exists() and not args.force:
            raise SystemExit(
                f"Run folder already exists: {run_dir}\n"
                "Pick a new --name, delete that folder, or pass --force to "
                "reuse it (steps will overwrite files as they complete)."
            )
        run_dir.mkdir(parents=True, exist_ok=True)

    # --- paths (all under run_dir) ---
    paths = {
        "calibration_report": run_dir / "calibration_report.json",
        "calibration_calls": run_dir / "calibration_calls.jsonl",
        "seed_priors": run_dir / "seed_priors_report.json",
        "seed_priors_calls": run_dir / "seed_priors_report_calls.jsonl",
        "episodes": run_dir / "episodes.json",
        "episodes_calls": run_dir / "episodes_calls.jsonl",
        "checks_report": run_dir / "checks_report.json",
        "checks_calls": run_dir / "checks_report_calls.jsonl",
        "checks_analysis": run_dir / "checks_report_analysis.json",
        "checks_plots": run_dir / "checks_plots.png",
        "core_inferability": run_dir / "core_inferability_report.json",
        "core_inferability_calls": run_dir / "core_inferability_report_calls.jsonl",
        "predictions_l1": run_dir / "predictions_L1.json",
        "predictions_l1_calls": run_dir / "predictions_L1_calls.jsonl",
        "predictions_l1_analysis": run_dir / "predictions_L1_analysis.txt",
        "predictions_l2": run_dir / "predictions_L2.json",
        "predictions_l2_calls": run_dir / "predictions_L2_calls.jsonl",
        "predictions_l2_analysis": run_dir / "predictions_L2_analysis.txt",
    }

    manifest = {
        "run_name": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.relative_to(ROOT)),
        "seeds": args.seeds,
        "n_trials_check0": args.n_trials,
        "temperature": args.temperature,
        "steps": [],
        "artifacts": {k: str(v.relative_to(ROOT)) for k, v in paths.items()},
    }

    def record(step: str, status: str = "ok", note: str = "") -> None:
        manifest["steps"].append({"step": step, "status": status, "note": note})
        if not args.dry_run:
            with open(run_dir / "RUN.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Run folder: {run_dir}")
    if args.dry_run:
        print("(dry run — no files created, no commands executed)")

    # 0. Judge calibration
    if not args.skip_calibration:
        run_cmd([
            py, "calibrate_judge.py", DEFAULT_CALIBRATION,
            "--repeats", "3",
            "--out", str(paths["calibration_report"]),
            "--log", str(paths["calibration_calls"]),
        ], dry_run=args.dry_run)
        record("calibrate_judge")
    else:
        record("calibrate_judge", "skipped")

    # 1. Check 0 — seed priors
    run_cmd([
        py, "check_seed_priors.py", args.seeds,
        "--n-trials", str(args.n_trials),
        "--out", str(paths["seed_priors"]),
        "--log", str(paths["seed_priors_calls"]),
    ], dry_run=args.dry_run)
    record("check_seed_priors")

    # 2. Generate episodes
    run_cmd([
        py, "generate_episodes.py", args.seeds,
        "--temperature", str(args.temperature),
        "--out", str(paths["episodes"]),
        "--log", str(paths["episodes_calls"]),
    ], dry_run=args.dry_run)
    record("generate_episodes")

    # 3. Checks 1+2
    run_cmd([
        py, "judges.py", str(paths["episodes"]),
        "--priors", str(paths["seed_priors"]),
        "--out", str(paths["checks_report"]),
        "--log", str(paths["checks_calls"]),
    ], dry_run=args.dry_run)
    record("judges")

    # 4. Analyze checks + plot
    run_cmd([
        py, "analyze_checks.py", str(paths["checks_report"]),
        "--priors", str(paths["seed_priors"]),
    ], dry_run=args.dry_run)
    record("analyze_checks")

    run_cmd([
        py, "plot_checks.py", str(paths["checks_report"]),
        "--out", str(paths["checks_plots"]),
    ], dry_run=args.dry_run)
    record("plot_checks")

    # 5. Check 3 — core inferability
    run_cmd([
        py, "check_core_inferability.py", str(paths["episodes"]),
        "--report", str(paths["checks_report"]),
        "--out", str(paths["core_inferability"]),
        "--log", str(paths["core_inferability_calls"]),
    ], dry_run=args.dry_run)
    record("check_core_inferability")

    # 6. Level-1 predictors
    if not args.skip_l1:
        run_cmd([
            py, "run_predictors.py",
            str(paths["episodes"]),
            str(paths["checks_report"]),
            str(paths["checks_analysis"]),
            "--conditions", "direct", "cot_matched", "latent_first",
            "latent_first_v2",
            "--out", str(paths["predictions_l1"]),
            "--log", str(paths["predictions_l1_calls"]),
        ], dry_run=args.dry_run)
        capture_cmd([
            py, "analyze_predictors.py",
            str(paths["predictions_l1"]),
            "--report", str(paths["checks_report"]),
        ], paths["predictions_l1_analysis"], dry_run=args.dry_run)
        record("run_predictors_L1")
    else:
        record("run_predictors_L1", "skipped")

    # 7. Level-2 predictors
    if not args.skip_l2:
        run_cmd([
            py, "run_predictors_v4.py",
            str(paths["episodes"]),
            str(paths["checks_report"]),
            str(paths["checks_analysis"]),
            str(paths["core_inferability"]),
            "--out", str(paths["predictions_l2"]),
            "--log", str(paths["predictions_l2_calls"]),
        ], dry_run=args.dry_run)
        capture_cmd([
            py, "analyze_predictors.py",
            str(paths["predictions_l2"]),
            "--report", str(paths["checks_report"]),
        ], paths["predictions_l2_analysis"], dry_run=args.dry_run)
        record("run_predictors_L2")
    else:
        record("run_predictors_L2", "skipped")

    record("complete")
    print(f"\nDone. Final results in: {run_dir}")
    print(f"Manifest: {run_dir / 'RUN.json'}")
    print("\nKey outputs:")
    show_keys = [
        ("checks report", "checks_report"),
        ("checks analysis (t*)", "checks_analysis"),
        ("checks plots", "checks_plots"),
        ("core inferability", "core_inferability"),
    ]
    if not args.skip_l1:
        show_keys += [
            ("predictions L1", "predictions_l1"),
            ("predictions L1 analysis", "predictions_l1_analysis"),
        ]
    if not args.skip_l2:
        show_keys += [
            ("predictions L2", "predictions_l2"),
            ("predictions L2 analysis", "predictions_l2_analysis"),
        ]
    for label, key in show_keys:
        rel = paths[key].relative_to(ROOT)
        if args.dry_run or paths[key].exists():
            print(f"  {label:24s}  {rel}")


if __name__ == "__main__":
    main()
