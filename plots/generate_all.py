"""Regenerate all paper figures from scripts in plots/.

Usage:
    uv run python plots/generate_all.py          # Run all scripts
    uv run python plots/generate_all.py --only train_test jaxmaze_rts
    uv run python plots/generate_all.py --refresh  # Re-fetch WandB data
    uv run python plots/generate_all.py --list     # List available scripts
"""

import argparse
import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# All plot scripts, in execution order.
# Notebooks are excluded — they must be run manually.
SCRIPTS = [
  # JaxMaze (human + model DataFrame)
  "jaxmaze_rts.py",
  "jaxmaze_envs_experiment_figure.py",
  "jaxmaze_envs_raw.py",
  "jaxmaze_sf_analysis.py",
  "jaxmaze_overlap_examples.py",
  "jaxmaze_rt_by_action.py",
  "jaxmaze_top_rt_analysis.py",
  "jaxmaze_results_threshold_robustness.py",
  # Craftax
  "craftax_ai_results.py",
  "craftax_ai_ablations.py",
  "craftax_ai_baselines.py",
  "craftax_multi_overlap_examples.py",
  # Cross-environment
  "overlap_distribution.py",
  "optimal_length.py",
  # WandB training curves
  "train_test.py",
  "jaxmaze_preplay_ablation.py",
  "jaxmaze_her_ablation.py",
  "craftax_ai_dyna_multigoal.py",
]

NOTEBOOKS = [
  "jaxmaze_results.ipynb",
  "craftax_cogsci_results.ipynb",
]


def main():
  parser = argparse.ArgumentParser(description="Regenerate all paper figures")
  parser.add_argument(
    "--only",
    nargs="+",
    help="Run only these scripts (without .py extension is fine)",
  )
  parser.add_argument(
    "--refresh",
    action="store_true",
    help="Pass --refresh to WandB scripts to re-fetch data",
  )
  parser.add_argument(
    "--list",
    action="store_true",
    help="List available scripts and exit",
  )
  args = parser.parse_args()

  if args.list:
    print("Available scripts:")
    for s in SCRIPTS:
      print(f"  {s}")
    print("\nNotebooks (run manually):")
    for n in NOTEBOOKS:
      print(f"  {n}")
    return

  scripts = SCRIPTS
  if args.only:
    # Match with or without .py extension
    selected = set()
    for name in args.only:
      if not name.endswith(".py"):
        name = name + ".py"
      if name in SCRIPTS:
        selected.add(name)
      else:
        print(f"Warning: unknown script '{name}', skipping")
    scripts = [s for s in SCRIPTS if s in selected]

  failed = []
  for script in scripts:
    print(f"\n{'=' * 60}")
    print(f"Running {script}")
    print("=" * 60)

    cmd = [sys.executable, os.path.join(SCRIPT_DIR, script)]
    if args.refresh:
      cmd.append("--refresh")

    result = subprocess.run(cmd)
    if result.returncode != 0:
      print(f"FAILED: {script}")
      failed.append(script)

  # Summary
  print(f"\n{'=' * 60}")
  print(f"Done. {len(scripts) - len(failed)}/{len(scripts)} succeeded.")
  if failed:
    print(f"Failed: {', '.join(failed)}")
  if NOTEBOOKS:
    print(f"\nNote: {len(NOTEBOOKS)} notebooks must be run manually:")
    for n in NOTEBOOKS:
      print(f"  {n}")


if __name__ == "__main__":
  main()
