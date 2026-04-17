"""JaxMaze main human+model results.

Runs all four JaxMaze manipulations: path reuse, shortcut, start, juncture.

Migrated from plots/jaxmaze_results.ipynb.

Usage:
    python plots/jaxmaze_results.py
"""

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

import plot_configs  # noqa: F401

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def main():
  import polars as pl

  import data_configs
  from analysis import jaxmaze_analysis
  from analysis.download_dataframes import download_jaxmaze_data

  download_jaxmaze_data()

  # Load data
  user_df = pl.read_parquet(data_configs.get_dataframe_path("jaxmaze", "human"))
  model_df = data_configs.load_dataframes("jaxmaze")

  # Two Paths Manipulation
  jaxmaze_analysis.path_reuse_results(
    user_df,
    model_df,
    save_dir=OUTPUT_DIR,
    tell_reuse=1,
    display_figs=True,
    save_figs=True,
    n_simulations=1000,
    rereun_analysis=False,
    overlap_threshold=0.5,
  )

  # Shortcut Manipulation
  jaxmaze_analysis.shortcut_results(
    user_df,
    model_df,
    save_dir=OUTPUT_DIR,
    overlap_threshold=0.6,
    save_figs=True,
    mu=0.5,
  )

  # Start Position Manipulation
  jaxmaze_analysis.start_results(
    user_df,
    save_dir=OUTPUT_DIR,
    ylim=[-0.05, 0.15],
  )

  # Juncture Manipulation
  jaxmaze_analysis.juncture_results(
    user_df,
    save_dir=OUTPUT_DIR,
    ylim=[0, 0.25],
  )


if __name__ == "__main__":
  main()
