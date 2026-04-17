"""
Bar plot comparing preplay ablation variants (all-goals ablation) on HouseMaze.

Usage:
    python plots/jaxmaze_preplay_ablation.py
    python plots/jaxmaze_preplay_ablation.py --no-bar   # learning curves
    python plots/jaxmaze_preplay_ablation.py --refresh   # re-fetch from wandb
"""

import argparse
import os

import matplotlib.pyplot as plt

import plot_configs
from wandb_utils import OUTPUT_DIR, get_group_history, plot_bar_panel, plot_panel

PROJECT = "housemaze"

GROUPS = {
  "preplay": "preplay-pnas-revision-5",
  "peng": "preplay-peng-ablation-pnas-revision-2",
  "cql": "preplay-cql-ablation-pnas-revision-2",
  "all_goals": "preplay-all-goals-ablation-pnas-revision-2",
}

KEYS = {
  "train_key": "actor_performance/0.0 avg_episode_return",
  "test_keys": [
    ("evaluator_performance/0.1 0.test \n two_paths - AvgReturn", "Two Paths"),
    ("evaluator_performance/0.1 0.test \n shortcut - AvgReturn", "Shortcut"),
  ],
  "train_step_key": "actor_performance/num_actor_steps",
  "test_step_key": "evaluator_performance/num_actor_steps",
}

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
    "peng": "Off-task Q(λ)",
    "cql": "CQL",
    "all_goals": "All Goals",
  }
)

# ── Legend settings (change here) ──
LEGEND_AX = 2  # which panel gets the legend (0=Train, 1=Test1, 2=Test2, ...)
LEGEND_LOC = "upper right"


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
  fig, axs = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6), sharey=True)
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
      show_legend=(i == LEGEND_AX),
      legend_loc=LEGEND_LOC,
      **kwargs,
    )

  fig.tight_layout()
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  path = os.path.join(OUTPUT_DIR, "jaxmaze_preplay_ablation.pdf")
  fig.savefig(path, bbox_inches="tight", dpi=300)
  print(f"Saved {path}")
  plt.close(fig)


if __name__ == "__main__":
  main()
