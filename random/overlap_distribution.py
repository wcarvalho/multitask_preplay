"""
Overlap Distribution Histograms

This script generates a 2x2 figure showing the distribution of path overlap values
across different experimental conditions to understand how participants reuse learned paths.

The four panels show:
1. Two Paths (JaxMaze) - manipulation="paths"
2. Shortcut (JaxMaze) - manipulation="shortcut"
3. Craftax - Tell Goal (tell_reuse=1)
4. Craftax - Don't Tell Goal (tell_reuse=0)

Usage:
    uv run python random/overlap_distribution.py
"""

import sys
import os

sys.path.insert(0, ".")
sys.path.append("simulations")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from analysis import vis_utils
from analysis import jaxmaze_analysis
from figures.figure_utils import save_figure
import data_configs
import plot_configs

# Overlap thresholds from data_configs
THRESHOLDS = {
  "paths": data_configs.TWO_PATHS_OVERLAP_THRESHOLD,  # 0.5
  "shortcut": data_configs.SHORTCUT_OVERLAP_THRESHOLD,  # 0.7
  "craftax": data_configs.CRAFTAX_OVERLAP_THRESHOLD,  # 0.25
}


def show_NaN_overlap_values(eval_df, full_df, save_dir):
  """Visualize train/test pairs where overlap is NaN to diagnose why.

  For each NaN-overlap eval row, finds ALL training episodes for the same
  user/world/block and creates an 8x2 figure (left: train episodes,
  right: the eval episode repeated). Saves up to 10 examples.
  """

  nan_df = eval_df.filter(pl.col("overlap").is_nan())
  if len(nan_df) == 0:
    print(f"  No NaN overlap values found.")
    return

  os.makedirs(save_dir, exist_ok=True)

  n_examples = min(10, len(nan_df))
  print(f"  Found {len(nan_df)} NaN overlap rows, visualizing {n_examples}...")

  for i in range(n_examples):
    test_row = nan_df.row(i, named=True)

    # Find ALL training episodes for this user/world/block
    train_episodes = full_df.filter(
      user_id=test_row["user_id"],
      world=test_row["world"],
      block_name=test_row["block_name"],
      task_set=0,
      eval=False,
    )

    n_train = len(train_episodes)
    n_rows = max(n_train, 1)  # At least 1 row even if no train episodes

    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 3 * n_rows))
    if n_rows == 1:
      axes = axes.reshape(1, 2)

    for row_idx in range(n_rows):
      # Left column: training episode
      if row_idx < n_train:
        train_row = train_episodes.row(row_idx, named=True)
        vis_utils.visualize_jaxmaze_row(train_row, ax_image=axes[row_idx, 0])
        success_str = "✓" if train_row.get("success") else "✗"
        axes[row_idx, 0].set_title(
          f"Train {row_idx} ({success_str}) | start_pos={train_row.get('start_pos', '?')}"
        )
      else:
        axes[row_idx, 0].text(
          0.5,
          0.5,
          "No train episode",
          ha="center",
          va="center",
          transform=axes[row_idx, 0].transAxes,
        )
        axes[row_idx, 0].set_title("Train (missing)")

      # Right column: eval episode (repeated)
      vis_utils.visualize_jaxmaze_row(test_row, ax_image=axes[row_idx, 1])
      if row_idx == 0:
        axes[row_idx, 1].set_title(f"Eval | start_pos={test_row.get('start_pos', '?')}")
      else:
        axes[row_idx, 1].set_title("")

    fig.suptitle(
      f"NaN overlap example {i}\n"
      f"user={test_row.get('user_id', '?')} | n_train={n_train} | world={test_row.get('world', '?')}",
      fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, f"example_{i}", directory=save_dir)
    plt.close(fig)

  print(f"  Saved {n_examples} NaN diagnostic figures to {save_dir}")


def plot_histogram(ax, overlap_values, title, threshold, color):
  """Plot a histogram of overlap values with threshold line and statistics."""
  # Filter out NaN values
  overlap_values = overlap_values.drop_nulls().to_numpy()

  if len(overlap_values) == 0:
    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    return

  # Create histogram
  bins = np.linspace(0, 1, 31)  # 30 bins from 0 to 1
  ax.hist(overlap_values, bins=bins, color=color, edgecolor="black", alpha=0.7)

  # Add threshold line
  ax.axvline(
    x=threshold,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Threshold = {threshold}",
  )

  # Calculate statistics
  mean_val = np.mean(overlap_values)
  median_val = np.median(overlap_values)
  n_total = len(overlap_values)
  n_above = np.sum(overlap_values > threshold)
  pct_above = 100 * n_above / n_total

  # Add statistics text
  stats_text = (
    f"N = {n_total}\n"
    f"Mean = {mean_val:.3f}\n"
    f"Median = {median_val:.3f}\n"
    f"Above threshold: {pct_above:.1f}%"
  )
  ax.text(
    0.97,
    0.97,
    stats_text,
    transform=ax.transAxes,
    verticalalignment="top",
    horizontalalignment="right",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
  )

  # Labels and title
  ax.set_xlabel("Overlap", fontsize=12)
  ax.set_ylabel("Count", fontsize=12)
  ax.set_title(title, fontsize=14)
  ax.legend(loc="upper left", fontsize=10)
  ax.set_xlim(0, 1)


def main():
  # Load full JaxMaze dataframe (needed for train episode lookups)
  print("Loading JaxMaze human data...")
  jaxmaze_df = pl.read_parquet(data_configs.get_dataframe_path("jaxmaze", "human"))

  print("Loading Craftax human data...")
  craftax_df = pl.read_parquet(data_configs.get_dataframe_path("craftax", "human"))

  # Use the same filtered cohorts as the main analysis (including train episodes for NaN diagnosis)
  print("\nFiltering JaxMaze Two Paths data (canonical cohort)...")
  paths_all_df = jaxmaze_analysis.get_path_reuse_eval_data(jaxmaze_df, eval_only=False)
  paths_df = paths_all_df.filter(eval=True)
  print(f"  Two Paths eval episodes: {len(paths_df)}")

  print("\nFiltering JaxMaze Shortcut data (canonical cohort)...")
  shortcut_all_df = jaxmaze_analysis.get_shortcut_eval_data(jaxmaze_df, eval_only=False)
  shortcut_df = shortcut_all_df.filter(eval=True)
  print(f"  Shortcut eval episodes: {len(shortcut_df)}")

  # Diagnose NaN overlap values
  show_NaN_overlap_values(
    paths_df, paths_all_df, "random/plots/overlap_distribution/paths"
  )
  show_NaN_overlap_values(
    shortcut_df, shortcut_all_df, "random/plots/overlap_distribution/shortcut"
  )

  # Craftax filtering (first 100 users per tell_reuse, with training success filter)
  print("\nFiltering Craftax data...")
  craftax_tell_df = craftax_df.filter(eval=True, tell_reuse=1, min_train_success=True)
  craftax_no_tell_df = craftax_df.filter(
    eval=True, tell_reuse=0, min_train_success=True
  )
  print(f"  Craftax Tell Goal eval episodes: {len(craftax_tell_df)}")
  print(f"  Craftax Don't Tell Goal eval episodes: {len(craftax_no_tell_df)}")

  # Create 2x2 figure
  fig, axes = plt.subplots(2, 2, figsize=(12, 10))
  fig.suptitle("Path Overlap Distributions Across Experiments", fontsize=16, y=0.98)

  colors = [
    plot_configs.default_colors["sky blue"],
    plot_configs.default_colors["bluish green"],
    plot_configs.default_colors["orange"],
    plot_configs.default_colors["reddish purple"],
  ]

  plot_histogram(
    axes[0, 0],
    paths_df["overlap"],
    "JaxMaze: Two Paths",
    THRESHOLDS["paths"],
    colors[0],
  )
  plot_histogram(
    axes[0, 1],
    shortcut_df["overlap"],
    "JaxMaze: Shortcut",
    THRESHOLDS["shortcut"],
    colors[1],
  )
  plot_histogram(
    axes[1, 0],
    craftax_tell_df["overlap"],
    "Craftax: Tell Goal (Known)",
    THRESHOLDS["craftax"],
    colors[2],
  )
  plot_histogram(
    axes[1, 1],
    craftax_no_tell_df["overlap"],
    "Craftax: Don't Tell Goal (Unknown)",
    THRESHOLDS["craftax"],
    colors[3],
  )

  plt.tight_layout()

  # Save figure
  save_dir = "random/plots"
  os.makedirs(save_dir, exist_ok=True)
  save_figure(fig, "overlap_distribution", directory=save_dir)

  print("\nDone!")


if __name__ == "__main__":
  main()
