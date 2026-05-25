"""
Bar plot comparing HER with and without all-goals on HouseMaze.

Usage:
    python figures_supplemental/her_all_goals_ablation.py
    python figures_supplemental/her_all_goals_ablation.py --no-bar   # learning curves
    python figures_supplemental/her_all_goals_ablation.py --refresh   # re-fetch from wandb
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_configs

from train_test import (
  SAVE_DIR,
  _plot_bar_panel,
  _plot_panel,
  get_group_history,
)

PROJECT = "housemaze"

# Filter by run display_name, not group
RUNS = {
  "with": "alg=her,all_=1.0,tota=20000000,exp=her_test_big",
  "without": "alg=her,all_=0.0,tota=20000000,exp=her_test_big",
}

KEYS = {
  "train_key": "actor_performance/0.0 avg_episode_return",
  "test_key": "evaluator_performance/0.1 0.test \n her_test_big - AvgReturn",
  "train_step_key": "actor_performance/num_actor_steps",
  "test_step_key": "evaluator_performance/num_actor_steps",
}

# Register ablation-specific colors and names
plot_configs.model_colors.update(
  {
    "with": plot_configs.model_colors["her"],
    "without": "#999999",
  }
)
plot_configs.model_names.update(
  {
    "with": "HER (All Goals)",
    "without": "HER (No All Goals)",
  }
)


def main():
  parser = argparse.ArgumentParser(description="HER all-goals ablation bar plots")
  parser.add_argument("--refresh", action="store_true", help="Re-fetch from WandB")
  parser.add_argument(
    "--bar",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Bar plots (default) or --no-bar for learning curves",
  )
  parser.add_argument("--xlim", nargs=2, type=float, default=[0, 20])
  args = parser.parse_args()

  # Fetch data using display_name filter
  datasets = {}
  for model_key, run_name in RUNS.items():
    datasets[model_key] = get_group_history(
      group=run_name,
      keys_dict=KEYS,
      project=PROJECT,
      refresh=args.refresh,
      filter_key="display_name",
    )

  train_key = KEYS["train_key"]
  train_step_key = KEYS["train_step_key"]
  test_step_key = KEYS["test_step_key"]
  test_panels = [(KEYS["test_key"], "Test")]

  n_panels = 1 + len(test_panels)
  fig, axs = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6))
  if n_panels == 1:
    axs = [axs]

  xlim_actual = [args.xlim[0] * 1e6, args.xlim[1] * 1e6]
  plot_fn = _plot_bar_panel if args.bar else _plot_panel

  kwargs = {} if args.bar else {"xlim": xlim_actual}
  plot_fn(
    axs[0], datasets, train_key, train_step_key, "Train", show_ylabel=True, **kwargs
  )
  for i, (test_key, test_label) in enumerate(test_panels):
    show_legend = not args.bar and i == 0
    plot_fn(
      axs[1 + i],
      datasets,
      test_key,
      test_step_key,
      test_label,
      show_ylabel=False,
      show_legend=show_legend,
      **kwargs,
    )

  fig.tight_layout()
  os.makedirs(SAVE_DIR, exist_ok=True)
  path = os.path.join(SAVE_DIR, "her_all_goals_ablation.pdf")
  fig.savefig(path, bbox_inches="tight", dpi=300)
  print(f"Saved {path}")
  plt.close(fig)


if __name__ == "__main__":
  main()
