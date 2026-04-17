"""
Craftax AI ablation results.

Migrated from figures/craftax_AI_ablation_results.py.

Usage:
    python plots/craftax_ai_ablations.py
    python plots/craftax_ai_ablations.py --refresh
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import wandb

# Repo-root import for data_configs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_configs import CRAFTAX_AI_DIR

import plot_configs as configs
from plot_configs import default_colors

# Directories
DIRECTORY = os.path.join(CRAFTAX_AI_DIR, "ablations")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Constants for plot styling
DEFAULT_TITLE_SIZE = 16
DEFAULT_XLABEL_SIZE = 12
DEFAULT_YLABEL_SIZE = 14
DEFAULT_LEGEND_SIZE = 12

# Colors and names
model_colors = {
  **configs.model_colors,
  "preplay-loss-ablation": default_colors["google orange"],
  "preplay-q-ablation": default_colors["google blue"],
  "preplay-precondition-ablation": default_colors["reddish purple"],
  "dyna-precondition-ablation": default_colors["dark gray"],
  "preplay-randomization-ablation": default_colors["black"],
}

model_names = {
  **configs.model_names,
  "preplay-loss-ablation": "Loss ablation $\\beta_{g}=0$",
  "preplay-q-ablation": "Sim policy ablation $\\alpha_{g}=0$",
  "preplay-precondition-ablation": "Multitask Preplay + No precondition",
  "dyna-precondition-ablation": "Dyna + No precondition",
  "preplay-randomization-ablation": "Randomized object locations",
}


def get_runs(group, name=None, entity="wcarvalho92", project="craftax"):
  """Get runs from wandb based on group and optional name."""
  api = wandb.Api()
  return api.runs(
    f"{entity}/{project}",
    filters={
      "group": group,
      **({"display_name": name} if name else {}),
    },
  )


def get_run_history(
  model_to_group,
  key="evaluator_performance-512/0.score",
  window_size=5,
  debug=False,
  refresh=False,
):
  """
  Retrieve time-series data from wandb for specified models and metric.

  Args:
      model_to_group: Dictionary mapping model names to wandb group names.
      key: The metric key to extract.
      window_size: Size of rolling window for smoothing.
      debug: If True, limit data retrieval for testing.
      refresh: If True, ignore cached data and re-fetch from wandb.

  Returns:
      Dictionary mapping model names to DataFrames of smoothed time-series data.
  """
  os.makedirs(DIRECTORY, exist_ok=True)
  all_data = {}

  for model, group in tqdm(model_to_group.items(), desc="Models"):
    cache_file = os.path.join(DIRECTORY, f"{model}_{group}_timeseries.json")

    # Try to load cached data
    if os.path.exists(cache_file) and not debug and not refresh:
      with open(cache_file, "r") as f:
        print(f"Loaded {cache_file}")
        all_data[model] = pd.read_json(f)
      continue

    # Remove stale cache if refreshing
    if refresh and os.path.exists(cache_file):
      os.remove(cache_file)

    # Get runs from wandb
    print(f"Fetching runs for {group}")
    runs = get_runs(group, name=None)

    # Collect data from each run
    run_data = []
    for run in tqdm(runs, desc=f"Processing {model} runs"):
      if debug and len(run_data) >= 2:
        break

      try:
        history = run.history(keys=[key])

        if history.empty:
          print(f"No data for key '{key}' in run {run.id}")
          continue

        history["run_id"] = run.id
        run_data.append(history)
      except Exception as e:
        print(f"Error processing run {run.id}: {e}")

    if not run_data:
      print(f"No data collected for {model}")
      all_data[model] = pd.DataFrame()
      continue

    model_df = pd.concat(run_data, ignore_index=True)

    # Apply rolling window smoothing for each run separately
    smooth_dfs = []
    for run_id in model_df["run_id"].unique():
      run_slice = model_df[model_df["run_id"] == run_id].sort_values("_step")
      if len(run_slice) >= window_size:
        numeric_cols = run_slice.select_dtypes(include=[np.number]).columns
        run_slice[numeric_cols] = (
          run_slice[numeric_cols].rolling(window=window_size, min_periods=1).mean()
        )
      smooth_dfs.append(run_slice)

    model_df = pd.concat(smooth_dfs, ignore_index=True)

    # Cache the data
    model_df.to_json(cache_file)
    print(f"Saved {cache_file}")

    all_data[model] = model_df

  return all_data


def plot_performance_curves(
  data,
  ax,
  key="evaluator_performance-512/0.score",
  xlabel="Actor Steps",
  ylabel="Score",
  title=None,
  model_to_line=None,
):
  """
  Plot performance curves for multiple models on a given Axes object.

  Args:
      data: Dictionary mapping model names to DataFrames of time-series data.
      ax: The matplotlib Axes object to plot on.
      key: The metric key being plotted.
      xlabel: Label for x-axis.
      ylabel: Label for y-axis.
      title: Plot title (if None, derives from key).
      model_to_line: Optional dict mapping model names to line style properties.
  """
  if title is None:
    title = key.replace("/", " - ")

  for model, df in data.items():
    if df.empty:
      continue

    grouped = df.groupby("_step").agg({key: ["mean", "sem"]})
    steps_unique = grouped.index.values
    means = grouped[(key, "mean")].values
    sems = grouped[(key, "sem")].values

    line_props = {
      "label": model_names.get(model, model),
      "color": model_colors.get(model, "gray"),
      "linewidth": 2,
    }

    if model_to_line and model in model_to_line:
      line_props.update(model_to_line[model])

    line = ax.plot(steps_unique, means, **line_props)

    ax.fill_between(
      steps_unique,
      means - sems,
      means + sems,
      color=line[0].get_color(),
      alpha=0.3,
    )

  ax.grid(True, alpha=0.3)
  ax.legend(fontsize=DEFAULT_LEGEND_SIZE)

  ax.set_xlabel(xlabel, fontsize=DEFAULT_XLABEL_SIZE)
  ax.set_ylabel(ylabel, fontsize=DEFAULT_YLABEL_SIZE)
  ax.set_title(title, fontsize=DEFAULT_TITLE_SIZE)

  all_dfs = [df for df in data.values() if not df.empty]
  if not all_dfs:
    return ax

  max_step = max([df["_step"].max() for df in all_dfs])

  if max_step > 1e6:
    ax.xaxis.set_major_formatter(
      lambda x, pos: f"{x / 1e6:.0f}M" if x >= 1e6 else f"{x / 1e3:.0f}K"
    )
  elif max_step > 1e3:
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x / 1e3:.0f}K")

  return ax


def plot_performance_max_bars(
  data,
  ax,
  key="evaluator_performance-512/0.score",
  xlabel="Model",
  ylabel="Score",
  title=None,
  model_to_bar=None,
  y_axis=None,
):
  """
  Plot bar chart showing maximum performance values for multiple models.

  Args:
      data: Dictionary mapping model names to DataFrames of time-series data.
      ax: The matplotlib Axes object to plot on.
      key: The metric key being plotted.
      xlabel: Label for x-axis.
      ylabel: Label for y-axis.
      title: Plot title (if None, derives from key).
      model_to_bar: Optional dict mapping model names to bar style properties.
      y_axis: Optional tuple for y-axis limits.
  """
  if title is None:
    title = key.replace("/", " - ")

  model_names_list = []
  max_means = []
  max_sems = []
  colors = []

  for model, df in data.items():
    if df.empty:
      continue

    run_maxes = []
    for run_id in df["run_id"].unique():
      run_data = df[df["run_id"] == run_id]
      if len(run_data) > 0:
        run_maxes.append(run_data[key].max())

    if len(run_maxes) > 0:
      max_mean = np.mean(run_maxes)
      max_sem = np.std(run_maxes) / np.sqrt(len(run_maxes))

      if model_to_bar and model in model_to_bar and "label" in model_to_bar[model]:
        label = model_to_bar[model]["label"]
      else:
        label = model_names.get(model, model)
      model_names_list.append(label)
      max_means.append(max_mean)
      max_sems.append(max_sem)
      colors.append(model_colors.get(model, "gray"))

  x_pos = np.arange(len(model_names_list))
  bar_props = {"alpha": 0.8, "capsize": 5}

  if model_to_bar:
    for i, model in enumerate(data.keys()):
      if model in model_to_bar:
        bar_style = {k: v for k, v in model_to_bar[model].items() if k != "label"}
        bar_props.update(bar_style)

  bars = ax.bar(x_pos, max_means, yerr=max_sems, color=colors, **bar_props)

  for i, (bar, mean, sem) in enumerate(zip(bars, max_means, max_sems)):
    height = bar.get_height()
    ax.text(
      bar.get_x() + bar.get_width() / 2.0,
      height + sem + 0.05,
      f"{mean:.2f}",
      ha="center",
      va="bottom",
      fontsize=10,
    )

  ax.set_xticks(x_pos)
  ax.set_xticklabels(model_names_list, rotation=0, ha="center")
  ax.set_xlabel(xlabel, fontsize=DEFAULT_XLABEL_SIZE)
  ax.set_ylabel(ylabel, fontsize=DEFAULT_YLABEL_SIZE)
  ax.set_title(title, fontsize=DEFAULT_TITLE_SIZE)
  ax.grid(True, alpha=0.3, axis="y")
  if y_axis is not None:
    ax.set_ylim(y_axis)
  return ax


def save_figure(fig, filename, directory=OUTPUT_DIR):
  """Save figure as PDF."""
  os.makedirs(directory, exist_ok=True)
  path = os.path.join(directory, f"{filename}.pdf")
  plt.savefig(path, bbox_inches="tight", dpi=300)
  print(f"Saved figure to {path}")
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser(description="Craftax AI ablation plots")
  parser.add_argument(
    "--refresh", action="store_true", help="Clear cached data and re-fetch"
  )
  args = parser.parse_args()

  # --- Plot 1: Loss and Q Ablation ---
  fig, axs = plt.subplots(1, 2, figsize=(18, 6))

  model_to_group_1 = {
    "preplay": "preplay-final-5",
    "preplay-loss-ablation": "preplay-main-loss-coeff-1",
    "preplay-q-ablation": "preplay-main-q-coeff-1",
  }

  data_1 = get_run_history(
    model_to_group=model_to_group_1,
    key="evaluator_performance-512/0.score",
    window_size=10,
    debug=False,
    refresh=args.refresh,
  )

  model_to_line_1 = {
    "preplay": {"linestyle": "-", "linewidth": 3},
    "preplay-loss-ablation": {"linestyle": "--", "linewidth": 3},
    "preplay-q-ablation": {"linestyle": "-.", "linewidth": 2.5},
  }

  plot_performance_curves(
    data_1,
    ax=axs[0],
    key="evaluator_performance-512/0.score",
    xlabel="Actor Steps",
    ylabel="Score",
    title="Preplay Loss & Sim Policy Ablation",
    model_to_line=model_to_line_1,
  )

  # --- Plot 2: Precondition Ablation ---
  model_to_group_2 = {
    "preplay": "preplay-final-5",
    "dyna": "dyna-final-5",
    "preplay-precondition-ablation": "preplay-precondition-1",
    "dyna-precondition-ablation": "dyna-precondition-1",
  }

  data_2 = get_run_history(
    model_to_group=model_to_group_2,
    key="evaluator_performance-512/0.score",
    window_size=10,
    debug=False,
    refresh=args.refresh,
  )

  model_to_line_2 = {
    "preplay": {"linestyle": "-", "linewidth": 3},
    "dyna": {"linestyle": "-", "linewidth": 3},
    "preplay-precondition-ablation": {"linestyle": "--", "linewidth": 2.5},
    "dyna-precondition-ablation": {"linestyle": "--", "linewidth": 2.5},
  }

  plot_performance_curves(
    data_2,
    ax=axs[1],
    key="evaluator_performance-512/0.score",
    xlabel="Actor Steps",
    ylabel="Score",
    title="Preplay & Dyna Precondition Ablation",
    model_to_line=model_to_line_2,
  )

  fig.tight_layout()
  save_figure(fig, "craftax_ai_ablations_loss_sim_policy")

  # --- Plot 3: Randomization Ablation ---
  fig, ax = plt.subplots(1, 1, figsize=(8, 6))
  model_to_group_3 = {
    "preplay": "preplay-final-5",
    "preplay-randomization-ablation": "preplay-randomize-2",
  }

  data_3 = get_run_history(
    model_to_group=model_to_group_3,
    key="evaluator_performance-512/0.score",
    window_size=10,
    debug=False,
    refresh=args.refresh,
  )

  model_to_line_3 = {
    "preplay": {
      "linestyle": "-",
      "linewidth": 3,
      "label": "Structured object locations",
    },
    "preplay-randomization-ablation": {
      "linestyle": "-.",
      "linewidth": 2.5,
      "label": "Randomized object locations",
    },
  }

  plot_performance_max_bars(
    data_3,
    ax=ax,
    key="evaluator_performance-512/0.score",
    xlabel="",
    ylabel="Score",
    title="Performance of Multitask Preplay\nwhen object locations are randomized",
    y_axis=(0, 7),
    model_to_bar=model_to_line_3,
  )

  fig.tight_layout()
  save_figure(fig, "craftax_ai_ablations_randomization")


if __name__ == "__main__":
  main()
