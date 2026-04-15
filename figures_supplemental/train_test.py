"""
Plot train/test learning curves from WandB experiments.

Usage:
    # Craftax multigoal (default)
    python figures_supplemental/train_test.py

    # Craftax multigoal with custom xlim
    python figures_supplemental/train_test.py --xlim 0 50

    # HouseMaze with epsilon groups
    python figures_supplemental/train_test.py --config housemaze_epsilon

    # Custom project and groups
    python figures_supplemental/train_test.py --project my-project --groups qlearning=ql-group usfa=usfa-group
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from tqdm.auto import tqdm

# Add parent directory to path for plot_configs import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_configs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "train_test_plots")
CACHE_DIR = os.path.join(SAVE_DIR, "cache")

ENTITY = "wcarvalho92"

# Default metric keys (for craftax-multigoal)
DEFAULT_KEYS = {
  "train_key": "actor_performance/0.episode_return",
  "test_key": "evaluator_performance/0.episode_return",
  "train_step_key": "actor_performance/num_actor_steps",
  "test_step_key": "evaluator_performance/num_actor_steps",
}

# Housemaze metric keys (multiple test keys for different environments)
HOUSEMAZE_KEYS = {
  "train_key": "actor_performance/0.0 avg_episode_return",
  "test_keys": [
    ("evaluator_performance/0.1 0.test \n two_paths - AvgReturn", "Two Paths"),
    ("evaluator_performance/0.1 0.test \n shortcut - AvgReturn", "Shortcut"),
  ],
  "train_step_key": "actor_performance/num_actor_steps",
  "test_step_key": "evaluator_performance/num_actor_steps",
}

# Preset configurations for different experiments
CONFIGS = {
  "craftax_multigoal": {
    "project": "craftax-multigoal",
    "groups": {
      "qlearning": "ql-final-1",
      "usfa": "usfa-final-5",
      "dyna": "dyna-final-2",
      "preplay": "preplay-final-1",
    },
    "xlim": [0, 40],
    "keys": DEFAULT_KEYS,
  },
  "housemaze_epsilon": {
    "project": "housemaze",
    "groups": {
      "qlearning": "ql-final-epsilon-1",
      "usfa": "usfa-final-epsilon-1",
      "dyna": "dyna-final-epsilon-1",
      "her": "her-final-epsilon-1",
    },
    "xlim": [0, 30],
    "keys": HOUSEMAZE_KEYS,
  },
}


def get_group_history(
  group,
  keys_dict,
  entity=ENTITY,
  project="craftax-multigoal",
  cache_dir=CACHE_DIR,
  window_size=20,
  refresh=False,
  filter_key="group",
):
  """Fetch time-series data for a wandb group, with caching and smoothing."""
  os.makedirs(cache_dir, exist_ok=True)
  # Include project in cache filename to avoid collisions
  safe_name = group.replace("/", "_").replace(",", "_").replace("=", "_")
  cache_file = os.path.join(cache_dir, f"{project}_{safe_name}_timeseries.json")

  if refresh and os.path.exists(cache_file):
    os.remove(cache_file)

  if os.path.exists(cache_file):
    print(f"Loaded {cache_file}")
    return pd.read_json(cache_file)

  # Extract all keys to fetch
  keys = [
    keys_dict["train_key"],
    keys_dict["train_step_key"],
    keys_dict["test_step_key"],
  ]
  # Handle single test_key or multiple test_keys
  if "test_key" in keys_dict:
    keys.append(keys_dict["test_key"])
  if "test_keys" in keys_dict:
    keys.extend([k for k, _ in keys_dict["test_keys"]])

  print(f"Fetching runs for group={group}")
  api = wandb.Api()
  # Include finished and crashed runs (crashed runs may still have data)
  filters = {filter_key: group, "state": {"$in": ["finished", "crashed"]}}
  runs = api.runs(f"{entity}/{project}", filters=filters)

  run_data = []
  for run in tqdm(runs, desc=f"Processing {group} runs"):
    try:
      # Use scan_history without keys filter (more reliable)
      history_rows = list(run.scan_history())
      if not history_rows:
        print(f"  No data for run {run.id}")
        continue
      history = pd.DataFrame(history_rows)
      # Filter to only needed columns (keep _step for sorting, keys for data)
      cols_to_keep = ["_step", "run_id"] + [k for k in keys if k in history.columns]
      history["run_id"] = run.id
      history = history[[c for c in cols_to_keep if c in history.columns]]
      run_data.append(history)
    except Exception as e:
      print(f"  Error processing run {run.id}: {e}")

  if not run_data:
    print(f"  No data collected for {group}")
    return pd.DataFrame()

  df = pd.concat(run_data, ignore_index=True)

  # Rolling-window smoothing per run (exclude step keys from smoothing)
  step_keys = {keys_dict["train_step_key"], keys_dict["test_step_key"]}
  smooth_dfs = []
  for run_id in df["run_id"].unique():
    run_slice = df[df["run_id"] == run_id].sort_values("_step")
    if len(run_slice) >= window_size:
      smooth_cols = [
        c
        for c in run_slice.select_dtypes(include=[np.number]).columns
        if c not in step_keys and c != "_step"
      ]
      run_slice[smooth_cols] = (
        run_slice[smooth_cols].rolling(window=window_size, min_periods=1).mean()
      )
    smooth_dfs.append(run_slice)

  df = pd.concat(smooth_dfs, ignore_index=True)
  df.to_json(cache_file)
  print(f"Saved {cache_file}")
  return df


def _plot_panel(
  ax, datasets, key, step_key, title, xlim=None, show_legend=False, show_ylabel=True
):
  """Plot mean +/- SEM for one or more groups on a single axes."""
  # PNAS-ready font sizes
  TITLE_SIZE = 24
  LABEL_SIZE = 20
  TICK_SIZE = 18
  LEGEND_SIZE = 14
  LINE_WIDTH = 3

  for model_key, df in datasets.items():
    if df.empty:
      continue
    # Check if required columns exist
    if key not in df.columns or step_key not in df.columns:
      print(f"  Warning: {model_key} missing columns {key} or {step_key}")
      continue
    # Filter to rows with valid data for this key
    valid_df = df.dropna(subset=[key, step_key])
    if valid_df.empty:
      continue
    grouped = valid_df.groupby(step_key).agg({key: ["mean", "sem"]})
    steps = grouped.index.values
    means = grouped[(key, "mean")].values
    sems = grouped[(key, "sem")].values

    color = plot_configs.model_colors.get(model_key, "gray")
    label = plot_configs.model_names.get(model_key, model_key)
    ax.plot(steps, means, label=label, color=color, linewidth=LINE_WIDTH)
    ax.fill_between(steps, means - sems, means + sems, color=color, alpha=0.25)

  ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
  ax.set_xlabel("Environment Steps", fontsize=LABEL_SIZE)
  if show_ylabel:
    ax.set_ylabel("Episode Return", fontsize=LABEL_SIZE)
  ax.tick_params(axis="both", labelsize=TICK_SIZE)
  ax.grid(True, alpha=0.3, linewidth=0.5)
  if xlim is not None:
    ax.set_xlim(xlim[0], xlim[1])
  ax.set_ylim(0, 1)
  if show_legend:
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9, loc="lower right")
  ax.xaxis.set_major_formatter(lambda x, pos: f"{x / 1e6:.0f}M")


def _plot_bar_panel(
  ax, datasets, key, step_key, title, show_legend=False, show_ylabel=True
):
  """Plot bar chart of best (peak) mean performance for each model."""
  TITLE_SIZE = 24
  LABEL_SIZE = 20
  TICK_SIZE = 18
  LINE_WIDTH = 2

  # Order models consistently using plot_configs.model_order
  ordered_keys = [k for k in plot_configs.model_order if k in datasets]
  # Add any keys not in model_order
  ordered_keys += [k for k in datasets if k not in ordered_keys]

  labels = []
  means = []
  sems = []
  colors = []

  for model_key in ordered_keys:
    df = datasets[model_key]
    if df.empty:
      continue
    if key not in df.columns or step_key not in df.columns:
      print(f"  Warning: {model_key} missing columns {key} or {step_key}")
      continue
    valid_df = df.dropna(subset=[key, step_key])
    if valid_df.empty:
      continue

    # Find the step with peak mean performance
    grouped = valid_df.groupby(step_key).agg({key: ["mean", "sem"]})
    best_idx = grouped[(key, "mean")].idxmax()
    best_mean = grouped.loc[best_idx, (key, "mean")]
    best_sem = grouped.loc[best_idx, (key, "sem")]

    labels.append(plot_configs.model_names.get(model_key, model_key))
    means.append(best_mean)
    sems.append(best_sem)
    colors.append(plot_configs.model_colors.get(model_key, "gray"))

  if not labels:
    return

  x = np.arange(len(labels))
  ax.bar(
    x,
    means,
    yerr=sems,
    capsize=5,
    color=colors,
    edgecolor="black",
    linewidth=LINE_WIDTH,
    width=0.6,
  )

  ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
  ax.set_xticks(x)
  ax.set_xticklabels(labels, fontsize=TICK_SIZE, rotation=45, ha="right")
  if show_ylabel:
    ax.set_ylabel("Best Episode Return", fontsize=LABEL_SIZE)
  ax.tick_params(axis="y", labelsize=TICK_SIZE)
  ax.grid(True, alpha=0.3, linewidth=0.5, axis="y")
  ax.set_ylim(0, 1)


def plot_train_test(
  project,
  groups,
  keys_dict=None,
  save_dir=SAVE_DIR,
  xlim=None,
  refresh=False,
  output_name=None,
  bar_plot=True,
):
  """Plot Train and Test learning curves. Supports multiple test panels."""
  if xlim is None:
    xlim = [0, 40e6]
  if keys_dict is None:
    keys_dict = DEFAULT_KEYS

  # Fetch data for all groups
  datasets = {}
  for model_key, group_name in groups.items():
    df = get_group_history(
      group=group_name,
      keys_dict=keys_dict,
      project=project,
      refresh=refresh,
    )
    datasets[model_key] = df

  train_key = keys_dict["train_key"]
  train_step_key = keys_dict["train_step_key"]
  test_step_key = keys_dict["test_step_key"]

  # Determine test panels: single test_key or multiple test_keys
  if "test_keys" in keys_dict:
    test_panels = keys_dict["test_keys"]  # List of (key, label) tuples
  else:
    test_panels = [(keys_dict["test_key"], "Test")]

  n_panels = 1 + len(test_panels)  # Train + test panels
  fig, axs = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6))
  if n_panels == 1:
    axs = [axs]

  if bar_plot:
    # Bar plots showing best (peak) performance per model
    _plot_bar_panel(
      axs[0],
      datasets,
      train_key,
      train_step_key,
      "Train",
      show_legend=False,
      show_ylabel=True,
    )
    for i, (test_key, test_label) in enumerate(test_panels):
      _plot_bar_panel(
        axs[1 + i],
        datasets,
        test_key,
        test_step_key,
        test_label,
        show_legend=False,
        show_ylabel=False,
      )
  else:
    # Learning curve plots
    _plot_panel(
      axs[0],
      datasets,
      train_key,
      train_step_key,
      "Train",
      xlim=xlim,
      show_legend=True,
      show_ylabel=True,
    )
    for i, (test_key, test_label) in enumerate(test_panels):
      _plot_panel(
        axs[1 + i],
        datasets,
        test_key,
        test_step_key,
        test_label,
        xlim=xlim,
        show_legend=False,
        show_ylabel=False,
      )

  fig.tight_layout()

  os.makedirs(save_dir, exist_ok=True)
  filename = output_name or f"train_test_{project}.pdf"
  path = os.path.join(save_dir, filename)
  fig.savefig(path, bbox_inches="tight", dpi=300)
  print(f"Saved {path}")
  plt.close(fig)


def parse_groups(groups_list):
  """Parse groups from CLI format: ['qlearning=ql-group', 'usfa=usfa-group']"""
  groups = {}
  for item in groups_list:
    key, value = item.split("=")
    groups[key] = value
  return groups


def main():
  parser = argparse.ArgumentParser(
    description="Plot train/test learning curves from WandB"
  )
  parser.add_argument(
    "--config",
    choices=list(CONFIGS.keys()),
    help="Use a preset configuration",
  )
  parser.add_argument(
    "--project",
    help="WandB project name (overrides config)",
  )
  parser.add_argument(
    "--groups",
    nargs="+",
    help="Groups in format: model_key=group_name (e.g., qlearning=ql-final-1)",
  )
  parser.add_argument(
    "--xlim",
    nargs=2,
    type=float,
    help="X-axis limits in millions (e.g., --xlim 0 40)",
  )
  parser.add_argument(
    "--refresh",
    action="store_true",
    help="Refresh cached data from WandB",
  )
  parser.add_argument(
    "--output",
    help="Output filename (default: train_test_<project>.pdf)",
  )
  parser.add_argument(
    "--bar",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Show bar plots of best performance (default: True). Use --no-bar for learning curves.",
  )

  args = parser.parse_args()

  # Determine which configs to plot
  if args.config:
    configs_to_plot = [args.config]
  elif args.project or args.groups:
    # Custom project/groups specified, use defaults
    configs_to_plot = [None]
  else:
    # No config specified, plot all configs
    configs_to_plot = list(CONFIGS.keys())

  for config_name in configs_to_plot:
    if config_name:
      config = CONFIGS[config_name]
      project = config["project"]
      groups = config["groups"]
      xlim = config.get("xlim", [0, 40])
      keys_dict = config.get("keys", DEFAULT_KEYS)
    else:
      # Custom mode with CLI args
      project = args.project
      groups = parse_groups(args.groups) if args.groups else {}
      xlim = args.xlim or [0, 40]
      keys_dict = DEFAULT_KEYS

    # Override with CLI args if specified
    if args.project:
      project = args.project
    if args.groups:
      groups = parse_groups(args.groups)
    if args.xlim:
      xlim = args.xlim

    # Convert xlim to actual values (input is in millions)
    xlim_actual = [xlim[0] * 1e6, xlim[1] * 1e6]

    plot_train_test(
      project=project,
      groups=groups,
      keys_dict=keys_dict,
      xlim=xlim_actual,
      refresh=args.refresh,
      output_name=args.output,
      bar_plot=args.bar,
    )


if __name__ == "__main__":
  main()
