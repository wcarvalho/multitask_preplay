"""
JaxMaze (HouseMaze) train/test bar plots and learning curves.

Usage:
    python plots/jaxmaze_train_test.py
    python plots/jaxmaze_train_test.py --no-bar
    python plots/jaxmaze_train_test.py --refresh
"""

import argparse

from wandb_utils import plot_train_test

PROJECT = "housemaze"
GROUPS = {
  "qlearning": "ql-final-epsilon-1",
  "usfa": "Group: usfa-pnas-revision-2",
  "dyna": "Group: dyna-pnas-revision-2",
  "her": "her-pnas-revision-2",
  "preplay": "preplay-pnas-revision-5",
}
KEYS = {
  "train_keys": [
    ("evaluator_performance/0.1 1.train \n two_paths - AvgReturn", "Two Paths (Train)"),
    ("evaluator_performance/0.1 1.train \n shortcut - AvgReturn", "Shortcut (Train)"),
  ],
  "test_keys": [
    ("evaluator_performance/0.1 0.test \n two_paths - AvgReturn", "Two Paths (Test)"),
    ("evaluator_performance/0.1 0.test \n shortcut - AvgReturn", "Shortcut (Test)"),
  ],
  "train_step_key": "evaluator_performance/num_actor_steps",
  "test_step_key": "evaluator_performance/num_actor_steps",
}
XLIM = [0, 50]


def main():
  parser = argparse.ArgumentParser(description="JaxMaze train/test plots")
  parser.add_argument("--refresh", action="store_true", help="Re-fetch from WandB")
  parser.add_argument(
    "--bar",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Bar plots (default) or --no-bar for learning curves",
  )
  parser.add_argument("--xlim", nargs=2, type=float, default=XLIM)
  parser.add_argument("--output", help="Output filename")
  args = parser.parse_args()

  xlim_actual = [args.xlim[0] * 1e6, args.xlim[1] * 1e6]
  plot_train_test(
    project=PROJECT,
    groups=GROUPS,
    keys_dict=KEYS,
    xlim=xlim_actual,
    refresh=args.refresh,
    output_name=args.output or "jaxmaze_train_test.pdf",
    bar_plot=args.bar,
    legend_ax=(0, 0),
    legend_loc="lower left",
  )


if __name__ == "__main__":
  main()
