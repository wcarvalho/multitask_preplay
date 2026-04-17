"""
Craftax multigoal train/test bar plots and learning curves.

Usage:
    python plots/craftax_train_test.py
    python plots/craftax_train_test.py --no-bar
    python plots/craftax_train_test.py --refresh
"""

import argparse

from wandb_utils import plot_train_test

PROJECT = "craftax-multigoal"
GROUPS = {
  "qlearning": "Group: ql-pnas-revision-3",
  "usfa": "Group: usfa-pnas-revision-3",
  "dyna": "Group: dyna-pnas-revision-2",
  "her": "Group: her-pnas-revision-1",
  "preplay": "Group: preplay-pnas-revision-6",
}
KEYS = {
  "train_key": "actor_performance/0.episode_return",
  "test_key": "evaluator_performance/0.episode_return",
  "train_step_key": "actor_performance/num_actor_steps",
  "test_step_key": "evaluator_performance/num_actor_steps",
}
XLIM = [0, 40]


def main():
  parser = argparse.ArgumentParser(description="Craftax multigoal train/test plots")
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
    output_name=args.output or "craftax_train_test.pdf",
    bar_plot=args.bar,
    legend_ax=0,
    legend_loc="lower left",
  )


if __name__ == "__main__":
  main()
