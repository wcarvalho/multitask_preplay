import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm.auto import tqdm

from configs import CRAFTAX_AI_DIR, default_colors
import data_configs

# Define directory for saving data and results
DIRECTORY = os.path.join(CRAFTAX_AI_DIR, "ablations")
os.makedirs(DIRECTORY, exist_ok=True)

# Constants for plot styling
DEFAULT_TITLE_SIZE = 16
DEFAULT_XLABEL_SIZE = 12
DEFAULT_YLABEL_SIZE = 14
DEFAULT_LEGEND_SIZE = 12

# Define colors for the plot

model_colors = {
  **configs.model_colors,
  "preplay-loss-ablation": default_colors["google orange"],
  "preplay-q-ablation": default_colors["google blue"],
  "preplay-precondition-ablation": default_colors["reddish purple"],
  "dyna-precondition-ablation": default_colors["dark gray"],
}

model_names = {
  **configs.model_names,
  "preplay-loss-ablation": "Loss ablation $\\beta_{g}=0$",
  "preplay-q-ablation": "Sim policy ablation $\\alpha_{g}=0$",
  "preplay-precondition-ablation": "Multitask Preplay + No precondition",
  "dyna-precondition-ablation": "Dyna + No precondition",
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
  model_to_group, key="evaluator_performance-512/0.score", window_size=5, debug=False
):
  """
  Retrieves time-series data from wandb for specified models and metric.

  Args:
      model_to_group: Dictionary mapping model names to wandb group names
      key: The metric key to extract
      window_size: Size of rolling window for smoothing
      debug: If True, limit data retrieval for testing

  Returns:
      Dictionary with model names as keys and DataFrames of smoothed time-series data as values
  """
  all_data = {}

  # Collect data for each model
  for model, group in tqdm(model_to_group.items(), desc="Models"):
    cache_file = os.path.join(DIRECTORY, f"{model}_{group}_timeseries.json")

    # Try to load cached data
    if os.path.exists(cache_file) and not debug:
      with open(cache_file, "r") as f:
        print(f"Loaded {cache_file}")
        all_data[model] = pd.read_json(f)
      continue

    # Get runs from wandb
    print(f"Fetching runs for {group}")
    runs = get_runs(group, name=None)

    # Collect data from each run
    run_data = []
    for run in tqdm(runs, desc=f"Processing {model} runs"):
      if debug and len(run_data) >= 2:
        break

      try:
        # Get history for this run
        history = run.history(keys=[key])

        if history.empty:
          print(f"No data for key '{key}' in run {run.id}")
          continue

        # Add run_id to identify data from different runs
        history["run_id"] = run.id
        run_data.append(history)
      except Exception as e:
        print(f"Error processing run {run.id}: {e}")

    if not run_data:
      print(f"No data collected for {model}")
      all_data[model] = pd.DataFrame()
      continue

    # Combine all runs into one dataframe
    model_df = pd.concat(run_data, ignore_index=True)

    # Apply rolling window smoothing for each run separately
    smooth_dfs = []
    for run_id in model_df["run_id"].unique():
      run_slice = model_df[model_df["run_id"] == run_id].sort_values("_step")
      # Apply rolling window (if there's enough data)
      if len(run_slice) >= window_size:
        # Extract numeric columns and apply rolling mean
        numeric_cols = run_slice.select_dtypes(include=[np.number]).columns
        run_slice[numeric_cols] = (
          run_slice[numeric_cols].rolling(window=window_size, min_periods=1).mean()
        )
      smooth_dfs.append(run_slice)

    # Combine all smoothed data
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
      data: Dictionary with model names as keys and DataFrames of time-series data as values
      ax: The matplotlib Axes object to plot on
      key: The metric key being plotted
      xlabel: Label for x-axis
      ylabel: Label for y-axis
      title: Plot title (if None, derives from key)
      model_to_line: Optional dictionary mapping model names to line styles (e.g., {'model': {'linestyle': '--', 'linewidth': 3}})
  """
  # Auto-generate title from key if not provided
  if title is None:
    title = key.replace("/", " - ")

  # Plot each model's data
  for model, df in data.items():
    if df.empty:
      continue

    # Extract steps and values
    steps = df["_step"]
    values = df[key]

    # Calculate mean and standard error across runs at each step
    grouped = df.groupby("_step").agg({key: ["mean", "sem"]})
    steps_unique = grouped.index.values
    means = grouped[(key, "mean")].values
    sems = grouped[(key, "sem")].values

    # Get line style properties
    line_props = {
      "label": model_names.get(model, model),
      "color": model_colors.get(model, "gray"),
      "linewidth": 2,
    }

    # Apply custom line properties if provided
    if model_to_line and model in model_to_line:
      line_props.update(model_to_line[model])

    # Plot the mean line with the specified properties
    line = ax.plot(steps_unique, means, **line_props)

    # Add shaded area for standard error
    ax.fill_between(
      steps_unique, means - sems, means + sems, color=line[0].get_color(), alpha=0.3
    )

  # Add grid and legend
  ax.grid(True, alpha=0.3)
  ax.legend(fontsize=DEFAULT_LEGEND_SIZE)

  # Set labels and title
  ax.set_xlabel(xlabel, fontsize=DEFAULT_XLABEL_SIZE)
  ax.set_ylabel(ylabel, fontsize=DEFAULT_YLABEL_SIZE)
  ax.set_title(title, fontsize=DEFAULT_TITLE_SIZE)

  # Format x-axis to show steps in a more readable way (e.g., "1M" instead of "1000000")
  all_dfs = [df for df in data.values() if not df.empty]
  if not all_dfs:  # Check if there is any data to plot
    return ax  # Return early if no data

  max_step = max([df["_step"].max() for df in all_dfs])

  if max_step > 1e6:
    ax.xaxis.set_major_formatter(
      lambda x, pos: f"{x / 1e6:.0f}M" if x >= 1e6 else f"{x / 1e3:.0f}K"
    )
  elif max_step > 1e3:
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x / 1e3:.0f}K")

  return ax  # Return the modified Axes object


def save_figure(fig, filename, directory=DIRECTORY):
  """Save figure in multiple formats."""
  os.makedirs(directory, exist_ok=True)
  # plt.savefig(os.path.join(directory, f"{filename}.png"), bbox_inches='tight', dpi=300)
  plt.savefig(os.path.join(directory, f"{filename}.pdf"), bbox_inches="tight", dpi=300)
  print(f"Saved figure to {directory}/{filename}.pdf")
  plt.close(fig)


if __name__ == "__main__":
  # Create a figure with two subplots
  fig, axs = plt.subplots(1, 2, figsize=(18, 6))  # Adjust figsize as needed

  # --- Plot 1: Loss and Q Ablation ---
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
  )

  model_to_line_1 = {
    "preplay": {"linestyle": "-", "linewidth": 3},
    "preplay-loss-ablation": {"linestyle": "--", "linewidth": 3},
    "preplay-q-ablation": {"linestyle": "-.", "linewidth": 2.5},
  }

  plot_performance_curves(
    data_1,
    ax=axs[0],  # Pass the first Axes object
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
  )

  model_to_line_2 = {
    "preplay": {"linestyle": "-", "linewidth": 3},
    "dyna": {"linestyle": "-", "linewidth": 3},
    "preplay-precondition-ablation": {"linestyle": "--", "linewidth": 2.5},
    "dyna-precondition-ablation": {"linestyle": "--", "linewidth": 2.5},
  }

  plot_performance_curves(
    data_2,
    ax=axs[1],  # Pass the second Axes object
    key="evaluator_performance-512/0.score",
    xlabel="Actor Steps",
    ylabel="Score",  # Keep ylabel consistent or remove for second plot if desired
    title="Preplay & Dyna Precondition Ablation",
    model_to_line=model_to_line_2,
  )

  # Adjust layout and save the combined figure
  fig.tight_layout()
  save_figure(fig, "preplay_ablation")  # Save the combined figure
