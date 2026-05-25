"""Craftax multigoal train/test bar plots from local parquet dataframes (no WandB).

Usage:
    python plots/craftax_train_test_df.py
    python plots/craftax_train_test_df.py --metric success
    python plots/craftax_train_test_df.py --no-humans
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl

import data_configs
from analysis import craftax_analysis
from plots.df_train_test import plot_train_test_from_df

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

ALGOS = ["qlearning", "usfa", "dyna", "her", "preplay"]


def _load_human_dfs():
  """Return (human_train_df, human_test_df) using the canonical first-100 cohort."""
  user_df = pl.read_parquet(data_configs.get_dataframe_path("craftax", "human"))
  test_df = craftax_analysis.get_path_reuse_eval_data(user_df)
  user_ids = test_df["user_id"].unique().to_list()
  train_df = user_df.filter(
    pl.col("user_id").is_in(user_ids),
    pl.col("eval") == False,  # noqa: E712
  )
  return train_df, test_df


def main():
  parser = argparse.ArgumentParser(description="Craftax train/test bar plot (df)")
  parser.add_argument(
    "--metric", default="success", choices=["success", "total_reward"]
  )
  parser.add_argument("--output", default="craftax_train_test_df.pdf")
  parser.add_argument(
    "--no-humans", action="store_true", help="Skip the human bars (model-only plot)."
  )
  parser.add_argument(
    "--human-split",
    default="by_tell_reuse",
    choices=["by_tell_reuse", "combined"],
    help="Show separate bars for known/unknown eval goal, or one combined bar.",
  )
  args = parser.parse_args()

  ylabel = "Success Rate"
  ylim = (0, 1.1)

  if args.no_humans:
    human_train_df, human_test_df = None, None
  else:
    human_train_df, human_test_df = _load_human_dfs()

  plot_train_test_from_df(
    env="craftax",
    save_dir=OUTPUT_DIR,
    output_name=args.output,
    metric=args.metric,
    ylabel=ylabel,
    ylim=ylim,
    splits=None,
    algos=ALGOS,
    legend_ax=0,
    legend_loc="lower left",
    # Match craftax_analysis.path_reuse_manipulation_analysis defaults
    overlap_threshold=0.25,
    cosine_threshold=0.5,
    human_train_df=human_train_df,
    human_test_df=human_test_df,
    human_split=args.human_split,
  )


if __name__ == "__main__":
  main()
