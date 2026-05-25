"""
Bar plot comparing preplay ablation variants (all-goals ablation) on HouseMaze.

Usage:
    python figures_supplemental/preplay_all_goals_ablation.py
    python figures_supplemental/preplay_all_goals_ablation.py --no-bar   # learning curves
    python figures_supplemental/preplay_all_goals_ablation.py --refresh   # re-fetch from wandb
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_configs

from train_test import (
  HOUSEMAZE_KEYS,
  SAVE_DIR,
  _plot_bar_panel,
  _plot_panel,
  get_group_history,
)

PROJECT = "housemaze"

GROUPS = {
  "preplay": "preplay-pnas-revision-5",
  "peng": "preplay-peng-ablation-pnas-revision-2",
  "cql": "preplay-cql-ablation-pnas-revision-2",
  "all_goals": "preplay-all-goals-ablation-pnas-revision-2",
}

KEYS = HOUSEMAZE_KEYS

# Register ablation-specific colors and names
plot_configs.model_colors.update(
  {
    "peng": "#679FE5",  # pretty blue
    "cql": "#9B80E6",  # nice purple
    "all_goals": "#999999",  # light gray
  }
)
plot_configs.model_names.update(
  {
    "peng": "w/o Off-task Q(λ)",
    "cql": "w/o CQL",
    "all_goals": "w/o All Goals",
  }
)


def main():
  parser = argparse.ArgumentParser(description="Preplay all-goals ablation bar plots")
  parser.add_argument("--refresh", action="store_true", help="Re-fetch from WandB")
  parser.add_argument(
    "--bar",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Bar plots (default) or --no-bar for learning curves",
  )
  parser.add_argument("--xlim", nargs=2, type=float, default=[0, 30])
  args = parser.parse_args()

  # Fetch data
  datasets = {}
  for model_key, group_name in GROUPS.items():
    datasets[model_key] = get_group_history(
      group=group_name,
      keys_dict=KEYS,
      project=PROJECT,
      refresh=args.refresh,
    )

  train_key = KEYS["train_key"]
  train_step_key = KEYS["train_step_key"]
  test_step_key = KEYS["test_step_key"]
  test_panels = KEYS["test_keys"]

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
  path = os.path.join(SAVE_DIR, "preplay_all_goals_ablation.pdf")
  fig.savefig(path, bbox_inches="tight", dpi=300)
  print(f"Saved {path}")
  plt.close(fig)


if __name__ == "__main__":
  main()
