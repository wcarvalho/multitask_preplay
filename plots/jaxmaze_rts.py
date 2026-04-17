"""Plot individual reaction times for JaxMaze two-paths and juncture manipulations.

Generates min/median/max RT difference plots for:
  - Two paths manipulation (reuse vs new path)
  - Juncture manipulation (eval task 1 vs eval task 2)

Source: figures_supplemental/jaxmaze_individual_rts.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import plot_configs
from analysis import analysis_utils
from analysis.jaxmaze_analysis import filter_users_by_success

import data_configs

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def save_figure(fig, filename):
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}.pdf"), bbox_inches="tight", dpi=300)
  print(f"Saved figure to {OUTPUT_DIR}/{filename}.pdf")
  plt.close()


def parse_reaction_times(rt_str: str) -> np.ndarray:
  """Parse stringified numpy array from the reaction_times column."""
  return np.fromstring(rt_str.strip("[]"), sep=" ")


def _pick_min_max_rt_row(sub_df: pl.DataFrame) -> dict:
  """Pick the row with the minimum max_rt and return its RT columns."""
  row = sub_df.sort("max_rt").row(0, named=True)
  return row


def create_two_paths_diff_rt_df(user_df: pl.DataFrame) -> pl.DataFrame:
  """Create a DataFrame with episodes having minimum max reaction time
  for each user in reuse=0 and reuse=1 conditions."""

  path_reuse_df, _ = filter_users_by_success(
    user_df.filter(
      tell_reuse=1,
      eval_shares_start_pos=True,
      manipulation="paths",
      world="big_m3_maze1",
      eval=True,
    ),
    analysis_name="path_reuse_results",
  )
  path_reuse_df = analysis_utils.add_reuse_column(
    path_reuse_df,
    reuse_column="reuse",
    overlap_threshold=data_configs.TWO_PATHS_OVERLAP_THRESHOLD,
    cosine_threshold=data_configs.COSINE_THRESHOLD,
  )

  df = path_reuse_df
  user_ids = df["user_id"].unique().to_list()

  rows = []

  for user_id in user_ids:
    reuse0_df = df.filter(user_id=user_id, reuse=0)
    reuse1_df = df.filter(user_id=user_id, reuse=1)

    if len(reuse0_df) == 0 or len(reuse1_df) == 0:
      continue

    r0 = _pick_min_max_rt_row(reuse0_df)
    r1 = _pick_min_max_rt_row(reuse1_df)

    # OUTLIERS not useful for visualization
    if max(r1["max_rt"], r0["max_rt"]) > 5:
      continue

    rows.append(
      {
        "user_id": user_id,
        "reuse0_first_rt": r0["first_rt"],
        "reuse0_max_rt": r0["max_rt"],
        "reuse0_first_log_rt": r0["first_log_rt"],
        "reuse0_max_log_rt": r0["max_log_rt"],
        "reuse0_rts": r0["reaction_times"],
        "reuse1_first_rt": r1["first_rt"],
        "reuse1_max_rt": r1["max_rt"],
        "reuse1_first_log_rt": r1["first_log_rt"],
        "reuse1_max_log_rt": r1["max_log_rt"],
        "reuse1_rts": r1["reaction_times"],
        "diff_first_rt": r0["first_rt"] - r1["first_rt"],
        "diff_max_rt": r0["max_rt"] - r1["max_rt"],
        "diff_first_log_rt": r0["first_log_rt"] - r1["first_log_rt"],
        "diff_max_log_rt": r0["max_log_rt"] - r1["max_log_rt"],
      }
    )

  return pl.DataFrame(rows)


def create_juncture_diff_rt_df(user_df: pl.DataFrame) -> pl.DataFrame:
  """Create a DataFrame with episodes having maximum difference in first RT
  between condition=1 and condition=2 for each user's maze."""

  df, _ = filter_users_by_success(
    user_df.filter(manipulation="juncture"), analysis_name="juncture_results"
  )

  def get_maze_setting(maze_str: str) -> str:
    if "short" in maze_str.lower():
      return "short"
    elif "long" in maze_str.lower():
      return "long"
    raise ValueError(f"Could not determine setting from maze string: {maze_str}")

  df = df.with_columns(
    setting=pl.col("world").map_elements(get_maze_setting, return_dtype=pl.String),
  )
  df = df.filter(setting="short")

  user_ids = df["user_id"].unique().to_list()

  rows = []

  cond1_mazes = sorted(df.filter(condition=1)["world"].unique())
  cond2_mazes = sorted(df.filter(condition=2)["world"].unique())
  for user_id in user_ids:
    user_df_filtered = df.filter(user_id=user_id)

    max_diff = -float("inf")
    max_metrics = None

    for cond1_maze, cond2_maze in zip(cond1_mazes, cond2_mazes):
      reuse1_df = user_df_filtered.filter(world=cond1_maze)
      reuse0_df = user_df_filtered.filter(world=cond2_maze)
      if len(reuse0_df) == 0 or len(reuse1_df) == 0:
        print(f"len(cond1_df) = {len(reuse0_df)} or len(cond2_df) = {len(reuse1_df)}")
        continue

      r1 = reuse1_df.row(0, named=True)
      r0 = reuse0_df.row(0, named=True)

      # OUTLIERS not useful for visualization
      if max(r1["max_rt"], r0["max_rt"]) > 5:
        continue

      diff_first_rt = r0["first_rt"] - r1["first_rt"]

      if diff_first_rt > max_diff:
        max_diff = diff_first_rt
        maze_name = cond1_maze.split("_eval")[0]
        max_metrics = {
          "maze_base": maze_name,
          "reuse1_first_rt": r1["first_rt"],
          "reuse1_max_rt": r1["max_rt"],
          "reuse1_first_log_rt": r1["first_log_rt"],
          "reuse1_max_log_rt": r1["max_log_rt"],
          "reuse1_rts": r1["reaction_times"],
          "reuse0_first_rt": r0["first_rt"],
          "reuse0_max_rt": r0["max_rt"],
          "reuse0_first_log_rt": r0["first_log_rt"],
          "reuse0_max_log_rt": r0["max_log_rt"],
          "reuse0_rts": r0["reaction_times"],
          "diff_first_rt": diff_first_rt,
          "diff_max_rt": r0["max_rt"] - r1["max_rt"],
          "diff_first_log_rt": r0["first_log_rt"] - r1["first_log_rt"],
          "diff_max_log_rt": r0["max_log_rt"] - r1["max_log_rt"],
        }

    if max_metrics is not None:
      row = {"user_id": user_id, **max_metrics}
      rows.append(row)

  return pl.DataFrame(rows)


def plot_min_median_max_differences(
  result_df,
  metric="first_rt",
  left_title_fn=lambda s: s,
  right_title_fn=lambda s: s,
  figsize=(15, 12),
  num_users=None,
):
  """Plot reaction times for users with minimum, median, and maximum differences."""

  sort_metric = f"reuse1_{metric}"
  sorted_df = result_df.sort(sort_metric)

  min_idx = 0
  max_idx = len(sorted_df) - 1
  median_idx = len(sorted_df) // 2

  fig, axes = plt.subplots(3, 2, figsize=figsize)

  index_names = ["min", "median", "max"]
  for i, idx in enumerate([min_idx, median_idx, max_idx]):
    rt0 = parse_reaction_times(sorted_df["reuse0_rts"][idx])
    rt1 = parse_reaction_times(sorted_df["reuse1_rts"][idx])

    analysis_utils.plot_reaction_times(
      rt0,
      ax=axes[i, 0],
      color=plot_configs.default_colors["nice purple"],
      title=left_title_fn(index_names[i].capitalize()),
      show_xlabel=False,
      remove_last=False,
    )

    analysis_utils.plot_reaction_times(
      rt1,
      ax=axes[i, 1],
      color=plot_configs.default_colors["bluish green"],
      title=right_title_fn(index_names[i].capitalize()),
      ylabel=None,
      show_xlabel=False,
      remove_last=False,
    )

    y_min = min(axes[i, 0].get_ylim()[0], axes[i, 1].get_ylim()[0])
    y_max = max(axes[i, 0].get_ylim()[1], axes[i, 1].get_ylim()[1])
    axes[i, 0].set_ylim(y_min, y_max)
    axes[i, 1].set_ylim(y_min, y_max)

  plt.tight_layout(rect=[0, 0, 1, 0.97])
  return fig


def main():
  from analysis.download_dataframes import download_jaxmaze_data

  download_jaxmaze_data()
  user_df = pl.read_parquet(data_configs.get_dataframe_path("jaxmaze", "human"))

  #########################################################
  # Two paths
  #########################################################
  paths_diff_df = create_two_paths_diff_rt_df(user_df)

  fig = plot_min_median_max_differences(
    paths_diff_df,
    "first_rt",
    left_title_fn=lambda s: f"Took new path ({s} first RT)",
    right_title_fn=lambda s: f"Reused old path ({s} first RT)",
    figsize=(15, 12),
  )
  save_figure(fig, "jaxmaze_rts_two_paths")

  #########################################################
  # Juncture
  #########################################################
  juncture_df = create_juncture_diff_rt_df(user_df)
  fig = plot_min_median_max_differences(
    juncture_df,
    "first_rt",
    left_title_fn=lambda s: f"Eval task 2 ({s} first RT)",
    right_title_fn=lambda s: f"Eval task 1 ({s} first RT)",
    figsize=(15, 12),
  )
  save_figure(fig, "jaxmaze_rts_juncture")


if __name__ == "__main__":
  main()
