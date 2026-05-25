"""Craftax multigoal human+model results.

Migrated from plots/craftax_cogsci_results.ipynb.

Usage:
    python plots/craftax_multi_results.py
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

OUTPUT_DIR = os.path.join(
  os.path.dirname(os.path.abspath(__file__)), "output", "craftax_multi_results"
)


def main():
  import polars as pl

  import data_configs
  from analysis import craftax_analysis
  from analysis.download_dataframes import download_craftax_data

  download_craftax_data()

  # Load data
  user_df = pl.read_parquet(data_configs.get_dataframe_path("craftax", "human"))
  model_df = data_configs.load_dataframes("craftax")

  # Path reuse analysis (produces both median and mean plots)
  craftax_analysis.path_reuse_manipulation_analysis(
    user_df=user_df,
    model_df=model_df,
    save_dir=OUTPUT_DIR,
    overlap_threshold=0.25,
    cosine_threshold=0.5,
    legend_loc="center left",
  )


if __name__ == "__main__":
  main()
