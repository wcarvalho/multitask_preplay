"""Robustness of path_reuse and shortcut results across overlap thresholds.

Generates separate files for scatter and RT plots:
  - path_reuse: scatter (1x3) + RT (3x2, rows=thresholds, cols=first/max RT)
  - shortcut: scatter (1x3) only

Usage:
    python plots/jaxmaze_results_threshold_robustness.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt

import polars as pl

import data_configs
from analysis import analysis_utils, jaxmaze_analysis

THRESHOLDS = [0.3, 0.5, 0.7]
OUTPUT_DIR = os.path.join(
  os.path.dirname(os.path.abspath(__file__)),
  "output",
  "jaxmaze_results_threshold_robustness",
)


def save_figure(fig, filename):
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  filepath = os.path.join(OUTPUT_DIR, f"{filename}.pdf")
  fig.savefig(filepath, bbox_inches="tight", dpi=300)
  print(f"Saved {filepath}")
  plt.close(fig)


def plot_scatter_robustness(manipulation, sub_df, mdf):
  """1x3 grid: one success-reuse scatter per threshold."""
  fig, axes = plt.subplots(1, len(THRESHOLDS), figsize=(6 * len(THRESHOLDS), 4))

  for i, threshold in enumerate(THRESHOLDS):
    analysis_utils.plot_success_rate_path_reuse_metrics(
      df=sub_df,
      model_df=mdf,
      ax=axes[i],
      experiment_name=f"robustness_{manipulation}_t{threshold}",
      title=f"Threshold = {threshold}",
      figsize=(6, 4),
      legend_loc="center left",
      legend_ncol=1,
      overlap_threshold=threshold,
      center="mean",
      mu=0.5,
    )

  fig.tight_layout()
  return fig


def plot_rt_robustness(manipulation, sub_df):
  """3x2 grid: rows=thresholds, cols=first_log_rt / max_log_rt."""
  fig, axes = plt.subplots(len(THRESHOLDS), 2, figsize=(10, 5 * len(THRESHOLDS)))

  for row, threshold in enumerate(THRESHOLDS):
    experiment_name = f"robustness_{manipulation}_t{threshold}"

    for col, measure in enumerate(["first_log_rt", "max_log_rt"]):
      ax = axes[row, col]
      analysis_utils.plot_bar_rt_comparison(
        sub_df,
        measure,
        ax=ax,
        n_simulations=1000,
        experiment_name=experiment_name,
        rereun_analysis=False,
        ylim=(7, 8),
        use_box_plot=False,
        overlap_threshold=threshold,
        ylabel="Log RT",
        compute_power=False,
      )

      if row == 0:
        col_title = "First Response Time" if col == 0 else "Max Response Time"
        ax.set_title(col_title)

    axes[row, 0].set_ylabel(f"Threshold = {threshold}\nLog RT")

  fig.tight_layout()
  return fig


def main():
  os.makedirs(OUTPUT_DIR, exist_ok=True)

  user_df = pl.read_parquet(data_configs.get_dataframe_path("jaxmaze", "human"))
  model_df = data_configs.load_dataframes("jaxmaze")

  # --- Path Reuse ---
  print("\n=== Path Reuse Threshold Robustness ===")
  path_reuse_df = jaxmaze_analysis.get_path_reuse_eval_data(user_df, tell_reuse=1)
  path_reuse_mdf = model_df.filter(
    manipulation="paths", world="big_m3_maze1", eval=True
  )

  fig = plot_scatter_robustness("path_reuse", path_reuse_df, path_reuse_mdf)
  save_figure(fig, "jaxmaze_results_threshold_robustness_path_reuse_scatter")

  fig = plot_rt_robustness("path_reuse", path_reuse_df)
  save_figure(fig, "jaxmaze_results_threshold_robustness_path_reuse_rt")

  # --- Shortcut ---
  print("\n=== Shortcut Threshold Robustness ===")
  shortcut_df = jaxmaze_analysis.get_shortcut_eval_data(user_df, tell_reuse=1)
  shortcut_mdf = model_df.filter(world="big_m1_maze3_shortcut", eval=True)

  fig = plot_scatter_robustness("shortcut", shortcut_df, shortcut_mdf)
  save_figure(fig, "jaxmaze_results_threshold_robustness_shortcut_scatter")


if __name__ == "__main__":
  main()
