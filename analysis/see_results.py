"""Quick summary of model success rates and path reuse across experiments."""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl

from analysis import analysis_utils
from analysis import jaxmaze_analysis
from analysis import craftax_analysis
import data_configs
import plot_configs


def compute_model_metrics_by_seed(model_df):
  """Compute success rate and mean overlap per model per seed.

  Returns a polars DataFrame with columns: algo, name, seed, success, overlap.
  """
  mdf = analysis_utils.get_polars_df(model_df)
  if "algo" not in mdf.columns:
    return pl.DataFrame()

  agg_cols = [pl.col("success").mean()]
  if "overlap" in mdf.columns:
    agg_cols.append(pl.col("overlap").mean())

  result = (
    mdf.group_by(["algo", "user_id"])
    .agg(agg_cols)
    .rename({"user_id": "seed"})
    .with_columns(
      pl.col("algo").replace(plot_configs.model_names).alias("name"),
      (pl.col("success") * 100).alias("success"),
    )
    .sort(["algo", "seed"])
  )
  if "overlap" in result.columns:
    result = result.with_columns((pl.col("overlap") * 100).alias("overlap"))

  return result


def main():
  print("Loading JaxMaze data...")
  jm_user_df = pl.read_parquet(data_configs.get_dataframe_path("jaxmaze", "human"))
  jm_model_df = data_configs.load_dataframes("jaxmaze")

  print("Loading Craftax data...")
  cx_user_df = pl.read_parquet(data_configs.get_dataframe_path("craftax", "human"))
  cx_model_df = data_configs.load_dataframes("craftax")

  # Get filtered eval data for each experiment
  print("\n--- Filtering data ---")
  _jm_pr_user, jm_pr_model = jaxmaze_analysis.get_path_reuse_eval_data(
    jm_user_df, jm_model_df
  )
  _jm_sc_user, jm_sc_model = jaxmaze_analysis.get_shortcut_eval_data(
    jm_user_df, jm_model_df
  )
  _cx_pr_user, cx_pr_model = craftax_analysis.get_path_reuse_eval_data(
    cx_user_df, cx_model_df
  )

  experiments = {
    "JaxMaze Path Reuse": jm_pr_model,
    "JaxMaze Shortcut": jm_sc_model,
    "Craftax": cx_pr_model,
  }

  for exp_name, mdf in experiments.items():
    print("\n" + "=" * 70)
    print(f"  {exp_name} — Model Success Rate (%) & Path Reuse (%)")
    print("=" * 70)

    metrics = compute_model_metrics_by_seed(mdf)
    if metrics.is_empty():
      print("  No model data.")
      continue

    # Per-seed table
    display_cols = ["name", "seed", "success"]
    if "overlap" in metrics.columns:
      display_cols.append("overlap")
    with pl.Config(tbl_rows=100, float_precision=1):
      print(metrics.select(display_cols))

    # Summary: mean across seeds per model
    summary_aggs = [
      pl.col("success").mean().alias("mean_success"),
      pl.col("success").std().alias("std_success"),
      pl.len().alias("n_seeds"),
    ]
    if "overlap" in metrics.columns:
      summary_aggs.extend(
        [
          pl.col("overlap").mean().alias("mean_overlap"),
          pl.col("overlap").std().alias("std_overlap"),
        ]
      )

    summary = metrics.group_by("name").agg(summary_aggs).sort("name")

    print(f"\n  Summary (mean +/- std across seeds):")
    with pl.Config(tbl_rows=100, float_precision=1):
      print(summary)


if __name__ == "__main__":
  main()
