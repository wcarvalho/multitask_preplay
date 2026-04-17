"""
Bar plot comparing HER with and without all-goals on HouseMaze.

Usage:
    python plots/jaxmaze_her_ablation.py
    python plots/jaxmaze_her_ablation.py --no-bar   # learning curves
    python plots/jaxmaze_her_ablation.py --refresh   # re-fetch from wandb
"""

import argparse
import os

import matplotlib.pyplot as plt

import plot_configs
from wandb_utils import plot_bar_panel, plot_panel
from wandb_utils import OUTPUT_DIR, get_group_history

PROJECT = "housemaze"

# ── HER big (with vs without all-goals) ──
BIG_RUNS = {
  "with": "alg=her,all_=1.0,tota=20000000,exp=her_test_big",
  "without": "alg=her,all_=0.0,tota=20000000,exp=her_test_big",
}
BIG_GROUP = "her-test-big-pnas-revision-3"
BIG_KEYS = {
  "train_key": "actor_performance/0.0 avg_episode_return",
  "test_key": "evaluator_performance/0.1 0.test \n her_test_big - AvgReturn",
  "train_step_key": "actor_performance/num_actor_steps",
  "test_step_key": "evaluator_performance/num_actor_steps",
}

# ── HER small (single condition, no all-goals) ──
SMALL_GROUP = "her-test-small-pnas-revision-1"
SMALL_KEYS = {
  "train_key": "actor_performance/0.0 avg_episode_return",
  "test_key": "evaluator_performance/0.1 0.test \n her_test - AvgReturn",
  "train_step_key": "actor_performance/num_actor_steps",
  "test_step_key": "evaluator_performance/num_actor_steps",
}

# ── Legend settings (change here) ──
LEGEND_AX = 1  # which panel gets the legend (0=Train, 1=Test, ...)
LEGEND_LOC = "upper right"

# Register ablation-specific colors and names
plot_configs.model_colors.update(
  {
    "with": plot_configs.model_colors["her"],
    "without": "#999999",
    "her_small": "#999999",
  }
)
plot_configs.model_names.update(
  {
    "with": "HER (All Goals)",
    "without": "HER (No All Goals)",
    "her_small": "HER (No All Goals)",
  }
)


def _plot_train_test(datasets, keys, args, output_name, legend_ax=LEGEND_AX):
  """Shared plotting logic for a single train/test figure."""
  train_key = keys["train_key"]
  train_step_key = keys["train_step_key"]
  test_step_key = keys["test_step_key"]
  test_panels = [(keys["test_key"], "Test")]

  n_models = len(datasets)
  panel_width = max(3, 4 * n_models)
  n_panels = 1 + len(test_panels)
  fig, axs = plt.subplots(1, n_panels, figsize=(panel_width * n_panels, 6), sharey=True)
  if n_panels == 1:
    axs = [axs]

  xlim_actual = [args.xlim[0] * 1e6, args.xlim[1] * 1e6]
  plot_fn = plot_bar_panel if args.bar else plot_panel

  all_panels = [("Train", train_key, train_step_key)] + [
    (label, k, test_step_key) for k, label in test_panels
  ]
  for i, (panel_title, k, sk) in enumerate(all_panels):
    kwargs = {} if args.bar else {"xlim": xlim_actual}
    plot_fn(
      axs[i],
      datasets,
      k,
      sk,
      panel_title,
      show_ylabel=(i == 0),
      show_legend=(i == legend_ax),
      legend_loc=LEGEND_LOC,
      **kwargs,
    )

  fig.tight_layout()
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  path = os.path.join(OUTPUT_DIR, output_name)
  fig.savefig(path, bbox_inches="tight", dpi=300)
  print(f"Saved {path}")
  plt.close(fig)


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

  # ── HER big: with vs without all-goals ──
  big_datasets = {}
  for model_key, run_name in BIG_RUNS.items():
    big_datasets[model_key] = get_group_history(
      group=run_name,
      keys_dict=BIG_KEYS,
      project=PROJECT,
      refresh=args.refresh,
      filter_key="display_name",
      extra_filters={"group": BIG_GROUP},
    )
  _plot_train_test(big_datasets, BIG_KEYS, args, "jaxmaze_her_ablation.pdf")

  # ── HER small: single condition ──
  small_datasets = {
    "her_small": get_group_history(
      group=SMALL_GROUP,
      keys_dict=SMALL_KEYS,
      project=PROJECT,
      refresh=args.refresh,
    )
  }
  _plot_train_test(small_datasets, SMALL_KEYS, args, "jaxmaze_her_ablation_small.pdf")


if __name__ == "__main__":
  main()
