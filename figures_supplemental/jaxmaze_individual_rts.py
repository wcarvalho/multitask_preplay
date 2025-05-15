""" """

import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent_dir, "simulations"))


import polars as pl
import jax
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from analysis import housemaze_model_data
from analysis import housemaze_user_data

from analysis import housemaze_utils
from analysis import experiment_analysis
from analysis import housemaze_analysis
from nicewebrl import dataframe

import data_configs


def save_figure(fig, filename, directory=None):
  directory = directory or f"{data_configs.DIRECTORY}/jaxmaze_individual_rts"
  os.makedirs(directory, exist_ok=True)
  # plt.savefig(os.path.join(directory, f"{filename}.png"), bbox_inches='tight', dpi=300)
  plt.savefig(os.path.join(directory, f"{filename}.pdf"), bbox_inches="tight", dpi=300)
  print(f"Saved figure to {directory}/{filename}.pdf")
  plt.close()


def create_two_paths_diff_rt_df(df: dataframe.DataFrame):
  """
  Create a DataFrame with episodes having minimum max reaction time for each user in reuse=0 and reuse=1 conditions.

  Args:
      df: DataFrame containing episodes with user_id and reuse fields

  Returns:
      DataFrame with one row per user, containing min max RT episodes and RT metrics
  """
  df = df.filter_by_group(
    input_episode_filter=experiment_analysis.filter_train_by_min_success,
    input_settings=dict(eval=False),
    output_settings=dict(manipulation=3, tell_reuse=1),
    group_key="user_id",
  ).filter(eval=True)

  # Get unique user IDs
  user_ids = df["user_id"].unique().to_list()

  rows = []
  episodes_list = []

  def get_rts(e):
    return e.reaction_times[:-1]

  for user_id in user_ids:
    # Filter episodes for this user with reuse=0
    reuse0_df = df.filter(user_id=user_id, reuse="0")
    # Filter episodes for this user with reuse=1
    reuse1_df = df.filter(user_id=user_id, reuse="1")

    # Skip if either condition is missing
    if len(reuse0_df) == 0 or len(reuse1_df) == 0:
      continue

    # Find episode with minimum max reaction time for reuse=0
    reuse0_max_rts = [np.max(get_rts(e)) for e in reuse0_df.episodes]
    reuse0_min_idx = np.argmin(reuse0_max_rts)
    reuse0_episode = reuse0_df.episodes[reuse0_min_idx]
    reuse0_idx = reuse0_min_idx

    # Find episode with minimum max reaction time for reuse=1
    reuse1_max_rts = [np.max(get_rts(e)) for e in reuse1_df.episodes]
    reuse1_min_idx = np.argmin(reuse1_max_rts)
    reuse1_episode = reuse1_df.episodes[reuse1_min_idx]
    reuse1_idx = reuse1_min_idx

    # Calculate metrics for reuse=0 episode
    reuse0_first_rt = reuse0_episode.reaction_times[0]
    reuse0_max_rt = np.max(get_rts(reuse0_episode))
    reuse0_first_log_rt = np.log(1000 * reuse0_first_rt)
    reuse0_max_log_rt = np.log(1000 * reuse0_max_rt)

    # Calculate metrics for reuse=1 episode
    reuse1_first_rt = reuse1_episode.reaction_times[0]
    reuse1_max_rt = np.max(get_rts(reuse1_episode))
    reuse1_first_log_rt = np.log(1000 * reuse1_first_rt)
    reuse1_max_log_rt = np.log(1000 * reuse1_max_rt)

    # Calculate differences
    diff_first_rt = reuse0_first_rt - reuse1_first_rt
    diff_max_rt = reuse0_max_rt - reuse1_max_rt
    diff_first_log_rt = reuse0_first_log_rt - reuse1_first_log_rt
    diff_max_log_rt = reuse0_max_log_rt - reuse1_max_log_rt

    if max(reuse1_max_rt, reuse0_max_rt) > 5:
      continue
    # Create row data
    row = {
      "user_id": user_id,
      "reuse0_idx": reuse0_idx,
      "reuse1_idx": reuse1_idx,
      "reuse0_first_rt": reuse0_first_rt,
      "reuse0_max_rt": reuse0_max_rt,
      "reuse0_first_log_rt": reuse0_first_log_rt,
      "reuse0_max_log_rt": reuse0_max_log_rt,
      "reuse1_first_rt": reuse1_first_rt,
      "reuse1_max_rt": reuse1_max_rt,
      "reuse1_first_log_rt": reuse1_first_log_rt,
      "reuse1_max_log_rt": reuse1_max_log_rt,
      "diff_first_rt": diff_first_rt,
      "diff_max_rt": diff_max_rt,
      "diff_first_log_rt": diff_first_log_rt,
      "diff_max_log_rt": diff_max_log_rt,
    }

    rows.append(row)
    episodes_list.append((reuse0_episode, reuse1_episode))

  # Create polars DataFrame
  pl_df = pl.DataFrame(rows)

  # Create DataFrame wrapper with episodes
  return dataframe.DataFrame(pl_df, episodes_list)


def create_juncture_diff_rt_df(df: dataframe.DataFrame, **kwargs):
  """
  Create a DataFrame with episodes having maximum difference in first reaction time
  between condition=1 and condition=2 for each user's maze.

  Args:
      df: DataFrame containing episodes with user_id, maze, and condition fields

  Returns:
      DataFrame with one row per user, containing the episodes and RT metrics for the
      maze with the highest difference in first reaction time between conditions
  """
  # Get unique user IDs
  kwargs = kwargs or dict(manipulation=4, tell_reuse=0)
  df = df.filter(**kwargs)

  def get_maze_setting(maze_str: str) -> str:
    if "short" in maze_str.lower():
      return "short"
    elif "long" in maze_str.lower():
      return "long"
    raise ValueError(f"Could not determine setting from maze string: {maze_str}")

  # Add setting column based on maze name and clean maze names
  df = df.with_columns(
    setting=pl.col("maze").map_elements(get_maze_setting, return_dtype=pl.String),
  )
  df = df.filter(setting="short")

  # maze_name=pl.col("maze").map_elements(lambda i: i.split("_eval")[0], return_dtype=pl.String)
  user_ids = df["user_id"].unique().to_list()

  rows = []
  episodes_list = []

  def get_rts(e):
    return e.reaction_times[:-1] if hasattr(e, "reaction_times") else e.reaction_times

  # Get unique maze names
  cond1_mazes = sorted(df.filter(condition=1)["maze"].unique())
  cond2_mazes = sorted(df.filter(condition=2)["maze"].unique())
  for user_id in user_ids:
    # Filter episodes for this user
    user_df = df.filter(user_id=user_id)

    min_diff = float("inf")
    max_diff = -float("inf")
    max_maze = None
    max_metrics = None

    for cond1_maze, cond2_maze in zip(cond1_mazes, cond2_mazes):
      reuse1_df = user_df.filter(maze=cond1_maze)
      reuse0_df = user_df.filter(maze=cond2_maze)
      if len(reuse0_df) == 0 or len(reuse1_df) == 0:
        print(f"len(cond1_df) = {len(reuse0_df)} or len(cond2_df) = {len(reuse1_df)}")
        continue

      # only 1 per condition
      reuse1_episode = reuse1_df.episodes[0]
      reuse0_episode = reuse0_df.episodes[0]

      # Calculate metrics for condition=1 episode
      reuse1_first_rt = reuse1_episode.reaction_times[0]
      reuse1_max_rt = np.max(get_rts(reuse1_episode))
      reuse1_first_log_rt = np.log(1000 * reuse1_first_rt)
      reuse1_max_log_rt = np.log(1000 * reuse1_max_rt)

      # Calculate metrics for condition=2 episode
      reuse0_first_rt = reuse0_episode.reaction_times[0]
      reuse0_max_rt = np.max(get_rts(reuse0_episode))
      reuse0_first_log_rt = np.log(1000 * reuse0_first_rt)
      reuse0_max_log_rt = np.log(1000 * reuse0_max_rt)

      # Calculate differences
      diff_first_rt = reuse0_first_rt - reuse1_first_rt
      diff_max_rt = reuse0_max_rt - reuse1_max_rt
      diff_first_log_rt = reuse0_first_log_rt - reuse1_first_log_rt
      diff_max_log_rt = reuse0_max_log_rt - reuse1_max_log_rt

      if max(reuse1_max_rt, reuse0_max_rt) > 5:
        continue
      # Check if this maze has highest difference in first RT
      if diff_first_rt > max_diff:
        # if diff_first_rt < min_diff:
        max_diff = diff_first_rt
        min_diff = diff_first_rt
        maze_name = cond1_maze.split("_eval")[0]
        max_maze = maze_name
        max_reuse1_episode = reuse1_episode
        max_reuse0_episode = reuse0_episode
        max_metrics = {
          "maze_base": maze_name,
          "reuse1_first_rt": reuse1_first_rt,
          "reuse1_max_rt": reuse1_max_rt,
          "reuse1_first_log_rt": reuse1_first_log_rt,
          "reuse1_max_log_rt": reuse1_max_log_rt,
          "reuse0_first_rt": reuse0_first_rt,
          "reuse0_max_rt": reuse0_max_rt,
          "reuse0_first_log_rt": reuse0_first_log_rt,
          "reuse0_max_log_rt": reuse0_max_log_rt,
          "diff_first_rt": diff_first_rt,
          "diff_max_rt": diff_max_rt,
          "diff_first_log_rt": diff_first_log_rt,
          "diff_max_log_rt": diff_max_log_rt,
        }

    # Add the user's max difference maze to the results
    if max_maze is not None:
      row = {"user_id": user_id, **max_metrics}
      rows.append(row)
      episodes_list.append((max_reuse0_episode, max_reuse1_episode))

  # Create polars DataFrame
  pl_df = pl.DataFrame(rows)

  # Create DataFrame wrapper with episodes
  return dataframe.DataFrame(pl_df, episodes_list)


def plot_min_median_max_differences(
  result_df,
  metric="first_rt",
  left_title_fn=lambda s: s,
  right_title_fn=lambda s: s,
  figsize=(15, 12),
  num_users=None,
):
  """
  Plot reaction times for users with minimum, median, and maximum differences
  for the specified metric.

  Args:
      result_df: DataFrame with minimum max RT episodes
      metric: Which difference metric to use (default: "first_rt")
      figsize: Figure size (default: (15, 12))
      num_users: Number of users to include (optional)

  Returns:
      fig: Figure object
  """
  # Sort the dataframe by the specified metric

  sort_metric = f"reuse1_{metric}"
  # sort_metric = f"diff_{metric}"
  sorted_df = result_df.sort(sort_metric)

  # Get indices for min, median, and max differences
  min_idx = 0
  max_idx = len(sorted_df) - 1
  median_idx = len(sorted_df) // 2

  # Create figure with 3 rows (min, median, max) and 2 columns (reuse=0, reuse=1)
  fig, axes = plt.subplots(3, 2, figsize=figsize)

  index_names = ["min", "median", "max"]
  for i, idx in enumerate([min_idx, median_idx, max_idx]):
    min_reuse0_episode = sorted_df.episodes[idx][0]
    min_reuse1_episode = sorted_df.episodes[idx][1]
    # Plot reaction times for min difference user
    rt0 = min_reuse0_episode.reaction_times[:-1]
    rt1 = min_reuse1_episode.reaction_times[:-1]

    # display_metric = f"diff_{metric}"
    # metric_value = sorted_df[display_metric][idx]
    experiment_analysis.plot_reaction_times(
      rt0,
      ax=axes[i, 0],
      color=experiment_analysis.default_colors["nice purple"],
      # title=f"{index_names[i].capitalize()} {display_metric.replace('_', ' ').capitalize()}: {metric_value:.3f}s",
      title=left_title_fn(index_names[i].capitalize()),
      show_xlabel=False,
    )

    experiment_analysis.plot_reaction_times(
      rt1,
      ax=axes[i, 1],
      color=experiment_analysis.default_colors["bluish green"],
      title=right_title_fn(index_names[i].capitalize()),
      # title=f" {right_title} ({index_names[i].capitalize()} RT)",
      ylabel=None,
      show_xlabel=False,
    )

    # Set the same y-axis limits for both plots in this row
    y_min = min(axes[i, 0].get_ylim()[0], axes[i, 1].get_ylim()[0])
    y_max = max(axes[i, 0].get_ylim()[1], axes[i, 1].get_ylim()[1])
    axes[i, 0].set_ylim(y_min, y_max)
    axes[i, 1].set_ylim(y_min, y_max)

  plt.tight_layout(rect=[0, 0, 1, 0.97])
  return fig


if __name__ == "__main__":
  from glob import glob
  import data_configs

  # Path to user data
  USER_RESULTS_DIR = os.path.join(data_configs.JAXMAZE_USER_DIR, "user_data/exps")

  # Load human data
  human_data_pattern = "final*v2*"
  files = f"{USER_RESULTS_DIR}/*{human_data_pattern}*.json"
  valid_files = list(set(glob(files)))

  user_df = housemaze_user_data.get_human_data(
    valid_files,
    overwrite_episode_data=False,
    overwrite_episode_info=False,
    require_finished=False,
    load_df_only=False,
    debug=False,
  )
  #########################################################
  # Two paths
  #########################################################
  # Filter to get evaluation episodes with path reuse manipulation
  path_reuse_df = user_df.filter_by_group(
    input_episode_filter=experiment_analysis.filter_train_by_min_success,
    input_settings=dict(eval=False),
    output_settings=dict(manipulation=3, tell_reuse=1),
    group_key="user_id",
  ).filter(eval=True)
  paths_diff_df = create_two_paths_diff_rt_df(user_df)

  fig = plot_min_median_max_differences(
    paths_diff_df,
    "first_rt",
    left_title_fn=lambda s: f"Took new path ({s} first RT)",
    right_title_fn=lambda s: f"Reused old path ({s} first RT)",
    figsize=(15, 12),
  )
  save_figure(fig, "two_paths_first")

  # fig = plot_min_median_max_differences(paths_diff_df, 'max_rt', left_title="Reuse 0", right_title="Reuse 1")
  # save_figure(fig, 'two_paths_max')

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
  save_figure(fig, "juncture_first")

  # fig = plot_min_median_max_differences(juncture_df, 'max_rt')
  # save_figure(fig, 'juncture_max')
