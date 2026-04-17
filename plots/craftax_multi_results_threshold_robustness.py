"""Robustness of Craftax path reuse results across overlap and cosine thresholds.

Generates an NxM grid of scatter plots (success rate vs path reuse) for each
reuse condition:
  - condition1 (overlap_and_cosine): overlap > threshold AND cosine > cosine_threshold
  - condition2 (double_overlap): overlap > 2 * threshold
  - combined: condition1 OR condition2

Each condition produces a separate PDF with:
  - Rows = overlap_threshold values
  - Columns = cosine_threshold values

Usage:
    python plots/craftax_multi_results_threshold_robustness.py
    python plots/craftax_multi_results_threshold_robustness.py --overlap_thresholds 0.1 0.25 0.5 --cosine_thresholds 0.3 0.5 0.7
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
  os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulations"
  )
)

import matplotlib.pyplot as plt

import plot_configs  # noqa: F401

OUTPUT_DIR = os.path.join(
  os.path.dirname(os.path.abspath(__file__)),
  "output",
  "craftax_multi_results_threshold_robustness",
)

REUSE_CONDITIONS = [
  ("overlap_and_cosine", "Condition 1: overlap > t AND cosine > c"),
  ("double_overlap", "Condition 2: overlap > 2t"),
  ("combined", "Combined: condition 1 OR condition 2"),
]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--overlap_thresholds",
    nargs="+",
    type=float,
    default=[0.25, 0.5, 0.75],
  )
  parser.add_argument(
    "--cosine_thresholds",
    nargs="+",
    type=float,
    default=[0.5, 0.7, 0.9],
  )
  args = parser.parse_args()

  import polars as pl

  import data_configs
  from analysis import craftax_analysis
  from analysis.download_dataframes import download_craftax_data

  download_craftax_data()

  user_df = pl.read_parquet(data_configs.get_dataframe_path("craftax", "human"))
  model_df = data_configs.load_dataframes("craftax")

  sub_df = craftax_analysis.get_path_reuse_eval_data(user_df)

  n_rows = len(args.overlap_thresholds)
  n_cols = len(args.cosine_thresholds)

  os.makedirs(OUTPUT_DIR, exist_ok=True)

  for condition_key, condition_label in REUSE_CONDITIONS:
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    # Handle single-row or single-col case
    if n_rows == 1 and n_cols == 1:
      axes = [[axes]]
    elif n_rows == 1:
      axes = [axes]
    elif n_cols == 1:
      axes = [[ax] for ax in axes]

    for row, ot in enumerate(args.overlap_thresholds):
      for col, ct in enumerate(args.cosine_thresholds):
        ax = axes[row][col]
        craftax_analysis.plot_success_rate_path_reuse_metrics(
          df=sub_df,
          model_df=model_df,
          ax=ax,
          overlap_threshold=ot,
          cosine_threshold=ct,
          center="mean",
          figsize=(6, 4),
          reuse_condition=condition_key,
        )
        legend = ax.get_legend()
        if legend is not None:
          legend.remove()

        if row == 0:
          ax.set_title(f"cosine={ct}")
        else:
          ax.set_title("")

        if col == 0:
          ax.set_ylabel(f"overlap={ot}\nSuccess Rate (%)")

    fig.suptitle(condition_label, fontsize=14, y=1.02)
    fig.tight_layout()

    filepath = os.path.join(
      OUTPUT_DIR,
      f"craftax_multi_results_threshold_robustness_{condition_key}.pdf",
    )
    fig.savefig(filepath, bbox_inches="tight", dpi=300)
    print(f"Saved {filepath}")
    plt.close(fig)


if __name__ == "__main__":
  main()
