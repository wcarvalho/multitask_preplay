"""JaxMaze train/test bar plots from local parquet dataframes (no WandB).

Usage:
    python plots/jaxmaze_train_test_df.py
    python plots/jaxmaze_train_test_df.py --metric total_reward
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plots.df_train_test import plot_train_test_from_df

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

ALGOS = ["qlearning", "usfa", "dyna", "her", "preplay"]
# Per-split overlap_threshold matches the main paper:
#   path_reuse_results uses 0.5, shortcut_results uses 0.6.
SPLITS = [("paths", "Two Paths", 0.5), ("shortcut", "Shortcut", 0.6)]


def main():
  parser = argparse.ArgumentParser(description="JaxMaze train/test bar plot (df)")
  parser.add_argument(
    "--metric", default="success", choices=["success", "total_reward"]
  )
  parser.add_argument("--output", default="jaxmaze_train_test_df.pdf")
  args = parser.parse_args()

  ylabel = "Success Rate" if args.metric == "success" else "Return"
  ylim = (0, 1.1) if args.metric == "success" else None

  plot_train_test_from_df(
    env="jaxmaze",
    save_dir=OUTPUT_DIR,
    output_name=args.output,
    metric=args.metric,
    ylabel=ylabel,
    ylim=ylim,
    splits=SPLITS,
    algos=ALGOS,
    legend_ax=0,
    legend_loc="lower left",
  )


if __name__ == "__main__":
  main()
