#!/usr/bin/env python3
"""Run multiple training experiments sequentially.

Usage:
    # Run named experiments (uses configs/experiment/*.yaml)
    python sweep.py freeze strong_aug gradual_aug

    # Run from a YAML sweep file
    python sweep.py --sweep sweep.yaml

    # Dry-run: print commands without executing
    python sweep.py --dry-run freeze strong_aug

    # Continue even if a run fails
    python sweep.py --skip-failed freeze strong_aug gradual_aug
"""
import argparse
import subprocess
import sys
from datetime import datetime

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def run_experiment(name: str, overrides: list, dry_run: bool = False) -> int:
    cmd = [sys.executable, "-m", "protogcn.train"] + overrides
    print(f"\n{'='*60}")
    print(f"Experiment : {name}")
    print(f"Command    : {' '.join(cmd)}")
    print(f"{'='*60}")
    if dry_run:
        return 0
    result = subprocess.run(cmd)
    return result.returncode


def build_runs_from_args(experiments: list) -> list:
    """Build run list from experiment names."""
    runs = []
    for exp in experiments:
        # Experiment configs already set name/work_dir via @package _global_
        runs.append({"name": exp, "overrides": [f"experiment={exp}"]})
    return runs


def build_runs_from_yaml(path: str) -> list:
    if not HAS_YAML:
        print("ERROR: pyyaml is not installed. Run: pip install pyyaml")
        sys.exit(1)
    with open(path) as f:
        sweep_cfg = yaml.safe_load(f)

    runs = []
    for run in sweep_cfg.get("runs", []):
        overrides = list(run.get("overrides", []))
        if "experiment" in run:
            overrides = [f"+experiment={run['experiment']}"] + overrides
        name = run.get("name") or run.get("experiment", "run")
        if not any(o.startswith("name=") for o in overrides):
            overrides += [f"name={name}"]
        if not any(o.startswith("work_dir=") for o in overrides):
            overrides += [f"work_dir=work_dirs/{name}"]
        runs.append({"name": name, "overrides": overrides})
    return runs


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple training experiments sequentially",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "experiments", nargs="*",
        help="Experiment config names (e.g., freeze strong_aug)"
    )
    parser.add_argument(
        "--sweep", type=str, metavar="FILE",
        help="Path to sweep YAML config file"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without running"
    )
    parser.add_argument(
        "--skip-failed", action="store_true",
        help="Continue to next run even if a run fails"
    )
    args = parser.parse_args()

    if args.sweep:
        runs = build_runs_from_yaml(args.sweep)
    elif args.experiments:
        runs = build_runs_from_args(args.experiments)
    else:
        parser.print_help()
        return

    total = len(runs)
    print(f"\nTotal experiments: {total}")
    if args.dry_run:
        print("(dry-run mode — commands will not be executed)")

    start_time = datetime.now()
    failed = []

    for i, run in enumerate(runs, 1):
        print(f"\n[{i}/{total}] Starting '{run['name']}'  |  elapsed: {datetime.now() - start_time}")
        rc = run_experiment(run["name"], run["overrides"], args.dry_run)
        if rc != 0:
            failed.append(run["name"])
            print(f"\n[{i}/{total}] FAILED '{run['name']}' (exit code: {rc})")
            if not args.skip_failed:
                print("Stopping sweep. Use --skip-failed to continue on failure.")
                break
        else:
            print(f"\n[{i}/{total}] DONE '{run['name']}'  |  elapsed: {datetime.now() - start_time}")

    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    if failed:
        print(f"Sweep finished with {len(failed)} failure(s) in {elapsed}")
        for name in failed:
            print(f"  FAILED: {name}")
        sys.exit(1)
    else:
        print(f"All {total} experiment(s) completed successfully in {elapsed}")


if __name__ == "__main__":
    main()
