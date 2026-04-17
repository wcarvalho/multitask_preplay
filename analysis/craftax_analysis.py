"""
Functions for
(1) getting dataframes related to different experiments
(2) plotting related metrics
"""

import sys
import os
import pickle
import inspect

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_project_root)
sys.path.append(os.path.join(_project_root, "simulations"))

import data_configs
from typing import Tuple

import jax
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import os.path
from analysis import analysis_utils

from nicewebrl import DataFrame
from IPython.display import HTML, display

import plot_configs
import data_configs
from plot_configs import model_colors


def __getattr__(name):
  if name in ("OPTIMAL_TEST_PATHS", "OPTIMAL_TEST_LENGTHS"):
    from data_processing import utils_craftax

    return getattr(utils_craftax, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def filter_to_str(filter: dict):
  return "".join([f"{k}={v}" for k, v in filter.items()])


# def get_path_reuse_df(
#  user_df: DataFrame,
#  tell_reuse: int = 1,
#  eval_map: bool = False,
# ):
#  sub_df = user_df.filter_by_group(
#    input_episode_filter=analysis_utils.filter_train_by_min_success,
#    input_settings=dict(eval=False),
#    output_settings=dict(
#      manipulation="paths", tell_reuse=tell_reuse, eval_map=int(eval_map)
#    ),
#    group_key="user_id",
#  ).filter(eval=True)
#  sub_df = analysis_utils.fix_reuse_column(sub_df)
#  sub_df = sub_df.filter(
#    pl.col("success").is_not_null() & pl.col("reuse").is_not_null()
#  )
#  return sub_df


# def path_similarity(path1, path2):
#  path1 = path1[-10:]
#  path2 = path2[-10:]

#  # Convert to direction vectors
#  def to_directions(path):
#    dirs = []
#    for i in range(len(path) - 1):
#      dx = path[i + 1][0] - path[i][0]
#      dy = path[i + 1][1] - path[i][1]
#      mag = (dx * dx + dy * dy) ** 0.5
#      if mag > 0:
#        dirs.append((dx / mag, dy / mag))
#    return dirs

#  dirs1 = to_directions(path1)
#  dirs2 = to_directions(path2)

#  # Resample longer to match shorter
#  min_len = min(len(dirs1), len(dirs2))
#  if len(dirs1) > min_len:
#    step = len(dirs1) / min_len
#    dirs1 = [dirs1[int(i * step)] for i in range(min_len)]
#  elif len(dirs2) > min_len:
#    step = len(dirs2) / min_len
#    dirs2 = [dirs2[int(i * step)] for i in range(min_len)]

#  # Average dot product
#  dots = sum(d1[0] * d2[0] + d1[1] * d2[1] for d1, d2 in zip(dirs1, dirs2))
#  similarity = (dots / min_len + 1) / 2  # Map from [-1,1] to [0,1]

#  return similarity


def visualize_user_path_reuse(df: DataFrame, user_id: int, idx=None, **kwargs):
  from simulations import craftax_utils

  user_df = df.filter(user_id=user_id)
  test_mazes = user_df["maze"].unique()
  test_mazes = [t for t in test_mazes if "eval" in t]
  test_mazes = sorted(test_mazes)

  for i in range(len(test_mazes)):
    if idx is not None:
      if i != idx:
        continue
    # get random test maze
    i = int(i)
    test_maze = test_mazes[i]
    test_df = user_df.filter(eval=True, maze=test_maze, **kwargs)
    if len(test_df.episodes) == 0:
      print(f"No test episodes for maze: {test_maze}")
      continue

    # get corresponding train maze
    train_maze = test_maze.replace("eval1", "training")
    start_pos = test_df["start_pos"].to_list()[0]
    train_df = user_df.filter(
      maze=train_maze, room=0, eval=False, success=1, start_pos=start_pos
    )

    if len(train_df.episodes) == 0:
      print(f"No train episodes for maze: {train_maze}")
      continue

    ############################################################
    # plot {train, test} paths for test reactio times
    ############################################################
    width = 5
    fig, axs = plt.subplots(2, 2, figsize=(2 * width, 2 * width))
    axs = axs.flatten()

    def make_image_path_panel(episode, ax):
      first_state = first(episode.timesteps.state)
      image = craftax_utils.render_fn(first_state, show_agent=False)
      path = episode.positions
      actions = craftax_utils.actions_from_path(path)
      craftax_utils.place_arrows_on_image(
        image=image,
        positions=path,
        actions=actions,
        maze_height=first_state.map.shape[1],
        maze_width=first_state.map.shape[2],
        ax=ax,
        display_image=True,
        arrow_color="red",
        show_path_length=False,
        start_color="red",
      )

    # plot {train, test} images
    first = lambda t: jax.tree_util.tree_map(lambda x: x[0], t)
    train_episode = train_df.episodes[0]
    test_episode = test_df.episodes[0]
    with jax.disable_jit():
      # display(HTML(test_df.to_pandas().to_html()))
      make_image_path_panel(train_episode, axs[0])
      title = f"User: {user_id}. {train_maze}"
      path_length = len(train_episode.positions)
      title += f"\nSuccess: {train_df['success'][0]}. Path Length: {path_length}"
      axs[0].set_title(title, fontsize=13)
      if len(train_df.episodes) > 1:
        make_image_path_panel(train_df.episodes[1], axs[1])
      else:
        axs[1].remove()

      make_image_path_panel(test_episode, axs[2])
      train_path = train_episode.positions
      test_path = test_episode.positions
      # similarity = path_similarity(train_path, test_path)
      world_seed = test_df["world_seed"].to_list()[0]
      path_length = len(test_episode.positions) - 1
      title = f"{test_maze}. Path Length: {path_length}"
      title += (
        f"\nOverlap: {test_df['overlap'][0]:.2f}. Reuse: {bool(test_df['reuse'][0])}"
      )
      from data_processing.utils_craftax import OPTIMAL_TEST_LENGTHS

      title += f"\nOptimal Path Length: {OPTIMAL_TEST_LENGTHS[world_seed]}"
      axs[2].set_title(title, fontsize=13)

      # plot reaction times
      reaction_times = test_episode.reaction_times
      axs[3].bar(range(len(reaction_times)), reaction_times, color="lightblue")
      axs[3].set_xlabel("Time")
      axs[3].set_title("Reaction Times")
      axs[3].set_ylim(0, max(reaction_times) * 1.1)

    plt.show()


def plot_success_rate_path_reuse_metrics(
  df: DataFrame,
  model_df: DataFrame = None,
  experiment_name: str = None,
  ax=None,
  reuse_column: str = "reuse",
  title="Success Rate and Path Reuse",
  figsize=(8, 8),
  legend_loc: str = "lower right",
  legend_ncol: int = 1,
  overlap_threshold: float = 0.15,
  cosine_threshold: float = None,
  center: str = "mean",
  reuse_condition: str = "combined",
) -> Tuple[plt.Figure, plt.Axes, dict]:
  """Plot success rate vs path reuse as a 2D scatter plot with error bars.

  Args:
      df (DataFrame): DataFrame containing human data
      model_df (DataFrame, optional): DataFrame containing model data. If None, only plots human data.
      experiment_name (str, optional): Name for stats tracking in paper_stats.yaml
      ax (plt.Axes, optional): Matplotlib axes to plot on. If None, creates new figure
      reuse_column (str, optional): Column name for path reuse metric
      title (str, optional): Plot title
      figsize (tuple, optional): Figure size if creating new figure
      legend_loc (str, optional): Location of the legend
      legend_ncol (int, optional): Number of columns in the legend
      center (str, optional): Center measure ("mean" or "median")

  Returns:
      tuple: (fig, ax, all_data) containing the figure, axes, and data dict
  """

  # Create figure if needed
  if ax is None:
    fig, ax = plt.subplots(figsize=figsize)
  else:
    fig = ax.figure

  # Prepare data for plotting
  all_data = {}
  # Process human data for tell_reuse=1 and tell_reuse=0
  tell_reuse_values = [1, 0]
  tell_reuse_labels = [
    "Human (known eval goal)",
    "Human (unknown eval goal)",
  ]
  tell_reuse_markers = ["o", "x"]  # Plus for True, X for False

  for i, (tell_value, label, marker) in enumerate(
    zip(tell_reuse_values, tell_reuse_labels, tell_reuse_markers)
  ):
    # Filter data for this tell_reuse value
    filtered_df = df.filter(tell_reuse=tell_value)
    human_data = analysis_utils.get_human_success_rate_path_reuse_data(
      df=filtered_df,
      overlap_threshold=overlap_threshold,
      reuse_column=reuse_column,
      experiment_name=experiment_name,
      label=label,
      cosine_threshold=cosine_threshold,
      center=center,
      reuse_condition=reuse_condition,
    )
    all_data[label] = human_data["human"]

  if model_df is not None:
    model_data = analysis_utils.get_model_success_rate_path_reuse_data(
      model_df=model_df.filter(eval=True),
      overlap_threshold=overlap_threshold,
      reuse_column=reuse_column,
      experiment_name=experiment_name,
      cosine_threshold=cosine_threshold,
      center=center,
      reuse_condition=reuse_condition,
    )
    all_data.update(model_data)

  # Define colors for human data points
  human_colors = {
    tell_reuse_labels[0]: plot_configs.default_colors["orange"],  # Green
    tell_reuse_labels[1]: plot_configs.default_colors["google blue"],  # Red
  }

  # Plot data points with error bars
  marker_size = 100  # Default size for all scatter points

  # Then plot model data points
  if model_df is not None:
    ordered_keys = [k for k in analysis_utils.model_order if k in all_data]
    for key in ordered_keys:
      if key in tell_reuse_labels:  # Skip if it's human data (already plotted)
        continue

      data = all_data[key]
      color = model_colors[key]
      xerr = analysis_utils.get_err(data, "reuse")
      yerr = analysis_utils.get_err(data, "success")

      ax.errorbar(
        data["reuse"],
        data["success"],
        xerr=xerr,
        yerr=yerr,
        fmt="none",
        color=color,
        capsize=5,
        capthick=2,
        elinewidth=2,
        zorder=2,
      )

      ax.scatter(
        data["reuse"],
        data["success"],
        color=color,
        s=marker_size,
        # marker=data["marker"],
        label=analysis_utils.model_names[key],
        zorder=3,
      )

  # First plot human data points
  for idx, key in enumerate(tell_reuse_labels):
    data = all_data[key]
    color = human_colors.get(key, "#333333")
    marker = tell_reuse_markers[idx]
    xerr = analysis_utils.get_err(data, "reuse")
    yerr = analysis_utils.get_err(data, "success")
    ax.errorbar(
      data["reuse"],
      data["success"],
      xerr=xerr,
      yerr=yerr,
      fmt="none",
      color=color,
      capsize=5,
      capthick=2,
      elinewidth=2,
      zorder=2,
    )

    ax.scatter(
      data["reuse"],
      data["success"],
      color=color,
      s=marker_size,
      marker=marker,
      label=key,
      zorder=3,
      linewidths=2,  # Make markers thicker for better visibility
    )

  # Customize axes
  ax.set_xlabel("Path Reuse (%)", fontsize=analysis_utils.DEFAULT_LABEL_SIZE)
  ax.set_ylabel("Success Rate (%)", fontsize=analysis_utils.DEFAULT_LABEL_SIZE)
  ax.set_title(title, fontsize=analysis_utils.DEFAULT_TITLE_SIZE)

  # Add chance level line for success rate
  ax.axhline(y=50, color="r", linestyle="--", alpha=0.5, label="Chance level")
  ax.axvline(x=50, color="r", linestyle="--", alpha=0.5)

  # Set axis limits with some padding
  ax.set_xlim(-5, 105)
  ax.set_ylim(-5, 105)

  # Add grid
  ax.grid(True, linestyle="--", alpha=0.7)

  # Add legend
  ax.legend(
    loc=legend_loc,
    ncol=legend_ncol,
    columnspacing=1,
    handletextpad=0.5,
    fontsize=analysis_utils.DEFAULT_LEGEND_SIZE,
  )

  return fig, ax, all_data


def plot_success_rate_path_reuse_metrics_efficiency(
  df: DataFrame,
  model_df: DataFrame = None,
  experiment_name: str = None,
  ax=None,
  reuse_column: str = "reuse",
  title="Success Rate and Path Reuse",
  figsize=(8, 8),
  legend_loc: str = "lower right",
  legend_ncol: int = 1,
  overlap_threshold: float = 0.15,
  cosine_threshold: float = None,
) -> Tuple[plt.Figure, plt.Axes]:
  """Plot success rate vs path reuse as a 2D scatter plot with error bars,
  including suboptimal path splits.

  Args:
      df (DataFrame): DataFrame containing human data
      model_df (DataFrame, optional): DataFrame containing model data. If None, only plots human data.
      experiment_name (str, optional): Name for stats tracking in paper_stats.yaml
      ax (plt.Axes, optional): Matplotlib axes to plot on. If None, creates new figure
      reuse_column (str, optional): Column name for path reuse metric
      title (str, optional): Plot title
      figsize (tuple, optional): Figure size if creating new figure
      legend_loc (str, optional): Location of the legend
      legend_ncol (int, optional): Number of columns in the legend

  Returns:
      tuple: (fig, ax) containing the figure and axes object
  """

  # Create figure if needed
  if ax is None:
    fig, ax = plt.subplots(figsize=figsize)
  else:
    fig = ax.figure

  # Prepare data for plotting
  all_data = {}

  # Process human data for all combinations of tell_reuse and suboptimal_path
  tell_reuse_values = [1, 0]
  suboptimal_path_values = [False, True]

  # Define labels and markers for all 4 combinations
  human_labels = []
  human_filters = []
  human_markers = []
  human_colors_list = []

  # Define color scheme for human data
  human_base_colors = {
    1: plot_configs.default_colors["orange"],  # tell_reuse=1
    0: plot_configs.default_colors["google blue"],  # tell_reuse=0
  }

  # Define markers for suboptimal_path
  suboptimal_markers = {
    False: "o",  # circle for optimal
    True: "s",  # square for suboptimal
  }

  for tell_value in tell_reuse_values:
    for suboptimal_value in suboptimal_path_values:
      # Create label
      goal_info = "known eval goal" if tell_value == 1 else "unknown eval goal"
      path_info = "inefficient" if suboptimal_value else "efficient"
      label = f"Human ({goal_info}, {path_info})"

      human_labels.append(label)
      human_filters.append(
        {"tell_reuse": tell_value, "suboptimal_path": suboptimal_value}
      )
      # Use "x" marker for efficient + unknown eval goal
      if tell_value == 0 and not suboptimal_value:
        human_markers.append("x")
      else:
        human_markers.append(suboptimal_markers[suboptimal_value])

      # Use lighter version of color for suboptimal paths
      base_color = human_base_colors[tell_value]
      if suboptimal_value:
        # Make color lighter for suboptimal paths
        import matplotlib.colors as mcolors

        rgba = mcolors.to_rgba(base_color)
        # Lighten by mixing with white
        lightened = tuple(c * 0.7 + 0.3 for c in rgba[:3]) + (rgba[3],)
        human_colors_list.append(lightened)
      else:
        human_colors_list.append(base_color)

  # Process human data
  for i, (label, filters, marker, color) in enumerate(
    zip(human_labels, human_filters, human_markers, human_colors_list)
  ):
    # Filter data for this combination
    filtered_df = df
    for key, value in filters.items():
      filtered_df = filtered_df.filter(**{key: value})

    human_data = analysis_utils.get_human_success_rate_path_reuse_data(
      df=filtered_df,
      overlap_threshold=overlap_threshold,
      reuse_column=reuse_column,
      experiment_name=experiment_name,
      label=label,
      cosine_threshold=cosine_threshold,
    )
    all_data[label] = human_data["human"]

  # Process model data if provided
  if model_df is not None:
    model_data = analysis_utils.get_model_success_rate_path_reuse_data(
      model_df=model_df.filter(eval=True),
      overlap_threshold=overlap_threshold,
      reuse_column=reuse_column,
      experiment_name=experiment_name,
      cosine_threshold=cosine_threshold,
    )
    all_data.update(model_data)

  # Plot data points with error bars
  marker_size = 100  # Default size for all scatter points

  # First plot model data points if available
  if model_df is not None:
    ordered_keys = [k for k in analysis_utils.model_order if k in all_data]
    for key in ordered_keys:
      if any(label in key for label in human_labels):  # Skip if it's human data
        continue

      data = all_data[key]
      color = model_colors[key]
      xerr = analysis_utils.get_err(data, "reuse")
      yerr = analysis_utils.get_err(data, "success")

      ax.errorbar(
        data["reuse"],
        data["success"],
        xerr=xerr,
        yerr=yerr,
        fmt="none",
        color=color,
        capsize=5,
        capthick=2,
        elinewidth=2,
        zorder=2,
      )

      ax.scatter(
        data["reuse"],
        data["success"],
        color=color,
        s=marker_size,
        label=analysis_utils.model_names[key],
        zorder=3,
      )

  # Then plot human data points
  for idx, (label, marker, color) in enumerate(
    zip(human_labels, human_markers, human_colors_list)
  ):
    data = all_data[label]
    xerr = analysis_utils.get_err(data, "reuse")
    yerr = analysis_utils.get_err(data, "success")

    ax.errorbar(
      data["reuse"],
      data["success"],
      xerr=xerr,
      yerr=yerr,
      fmt="none",
      color=color,
      capsize=5,
      capthick=2,
      elinewidth=2,
      zorder=2,
    )

    ax.scatter(
      data["reuse"],
      data["success"],
      color=color,
      s=marker_size,
      marker=marker,
      label=label,
      zorder=3,
      linewidths=2,  # Make markers thicker for better visibility
      edgecolors="black"
      if marker == "s"
      else color,  # Add black edge to squares for clarity
      alpha=0.9,
    )

  # Customize axes
  ax.set_xlabel("Path Reuse (%)", fontsize=analysis_utils.DEFAULT_LABEL_SIZE)
  ax.set_ylabel("Success Rate (%)", fontsize=analysis_utils.DEFAULT_LABEL_SIZE)
  ax.set_title(title, fontsize=analysis_utils.DEFAULT_TITLE_SIZE)

  # Add chance level line for success rate
  ax.axhline(y=50, color="r", linestyle="--", alpha=0.5, label="Chance level")
  ax.axvline(x=50, color="r", linestyle="--", alpha=0.5)

  # Set axis limits with some padding
  ax.set_xlim(-5, 105)
  ax.set_ylim(-5, 105)

  # Add grid
  ax.grid(True, linestyle="--", alpha=0.7)

  # Add legend
  ax.legend(
    loc=legend_loc,
    ncol=legend_ncol,
    columnspacing=1,
    handletextpad=0.5,
    fontsize=analysis_utils.DEFAULT_LEGEND_SIZE,
  )

  return fig, ax


def plot_efficiency(
  df: DataFrame,
  ax=None,
  figsize=(10, 5),
  title="Path Efficiency by Goal Knowledge",
  bar_width=0.35,
  ylabel="Proportion (%)",
  show_values=True,
) -> Tuple[plt.Figure, plt.Axes]:
  """Plot bar graphs showing proportion of efficient vs inefficient paths.

  Creates 2 panels:
  - Left: known eval goal (tell_reuse=1)
  - Right: unknown eval goal (tell_reuse=0)

  Each panel shows bars for efficient and inefficient path proportions.

  Args:
      df (DataFrame): DataFrame containing data with 'tell_reuse' and 'suboptimal_path' columns
      ax (array of plt.Axes, optional): Array of 2 matplotlib axes. If None, creates new figure
      figsize (tuple, optional): Figure size if creating new figure
      title (str, optional): Overall figure title
      bar_width (float, optional): Width of bars
      ylabel (str, optional): Y-axis label
      show_values (bool, optional): Whether to show percentage values on bars

  Returns:
      tuple: (fig, axes) containing the figure and axes array
  """

  # Create figure if needed
  if ax is None:
    fig, axes = plt.subplots(1, 2, figsize=figsize)
  else:
    axes = ax
    fig = axes[0].figure

  # Define conditions and labels
  tell_reuse_conditions = [1, 0]
  panel_titles = ["Known Eval Goal", "Unknown Eval Goal"]

  # Process each tell_reuse condition
  for idx, (tell_reuse_val, panel_title) in enumerate(
    zip(tell_reuse_conditions, panel_titles)
  ):
    ax = axes[idx]

    # Filter data for this condition
    condition_df = df.filter(tell_reuse=tell_reuse_val, eval=True)

    # Calculate proportions
    total_count = len(condition_df)
    efficient_count = len(condition_df.filter(suboptimal_path=False))
    inefficient_count = len(condition_df.filter(suboptimal_path=True))

    # Calculate percentages
    if total_count > 0:
      efficient_pct = (efficient_count / total_count) * 100
      inefficient_pct = (inefficient_count / total_count) * 100
    else:
      efficient_pct = 0
      inefficient_pct = 0

    # Create bars
    x_pos = np.arange(2)
    bars = ax.bar(
      x_pos,
      [efficient_pct, inefficient_pct],
      bar_width,
      color=[
        plot_configs.default_colors["bluish green"],  # Efficient
        plot_configs.default_colors["nice purple"],  # Inefficient
      ],
      edgecolor="black",
      linewidth=1.5,
    )

    # Add value labels on bars if requested
    if show_values:
      for bar, value in zip(bars, [efficient_pct, inefficient_pct]):
        height = bar.get_height()
        ax.text(
          bar.get_x() + bar.get_width() / 2.0,
          height,
          f"{value:.1f}%",
          ha="center",
          va="bottom",
          fontsize=10,
        )

    # Customize subplot
    ax.set_ylabel(ylabel, fontsize=analysis_utils.DEFAULT_LABEL_SIZE)
    ax.set_title(panel_title, fontsize=analysis_utils.DEFAULT_TITLE_SIZE)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
      ["Efficient", "Inefficient"], fontsize=analysis_utils.DEFAULT_LABEL_SIZE
    )
    ax.set_ylim(0, 110)  # Add some space for value labels
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Add sample size annotation
    ax.text(
      0.5,
      0.02,
      f"n = {total_count}",
      transform=ax.transAxes,
      ha="center",
      fontsize=10,
      style="italic",
    )

  # Add overall title
  fig.suptitle(title, fontsize=analysis_utils.DEFAULT_TITLE_SIZE + 2)

  # Adjust layout
  fig.tight_layout(rect=[0, 0.03, 1, 0.95])

  return fig, axes


def get_path_reuse_eval_data(user_df, eval_only=True):
  """Return filtered (user_df, model_df) for craftax path reuse experiment."""
  filter_kwargs = {}
  if eval_only:
    filter_kwargs["eval"] = True
    filter_kwargs["eval_shares_start_pos"] = True
  sub_df, _ = analysis_utils.filter_users_by_success_and_tell_reuse(
    user_df.filter(**filter_kwargs)
  )
  return sub_df


def path_reuse_manipulation_analysis(
  user_df: DataFrame,
  model_df: DataFrame,
  save_dir: str = None,
  save_figs: bool = True,
  display_figs: bool = True,
  verbosity: int = 1,
  reuse_column: str = "reuse",
  legend_loc="upper left",
  overlap_threshold: float = 0.15,
  cosine_threshold: float = None,
  center: str = None,
):
  ############################################################
  # Create stats file
  ############################################################
  save_dir = save_dir or data_configs.CRAFTAX_RESULTS_DIR
  save_dir = os.path.join(save_dir, "5.craftax_path_reuse_manipulation")
  os.makedirs(save_dir, exist_ok=True)

  ############################################################
  # Plot success rate and path reuse
  ############################################################
  if reuse_column == "reuse":
    title = "Generalization Success & Path Reuse"
  else:
    title = "Efficient Path Reuse & Generalization Success Rate"

  ############################################################
  # Get 1st 100
  ############################################################

  # Filter dataframe to only include rows with those user IDs
  user_df, first_100_users = analysis_utils.filter_users_by_success_and_tell_reuse(
    user_df.filter(
      eval_shares_start_pos=True,
      eval=True,
    )
  )

  experiment_name = "5.craftax_path_reuse_stats"
  fig, ax, all_data = plot_success_rate_path_reuse_metrics(
    df=user_df,
    model_df=model_df,
    experiment_name=experiment_name,
    title=title,
    reuse_column=reuse_column,
    figsize=(6, 4),
    legend_loc=legend_loc,
    overlap_threshold=overlap_threshold,
    cosine_threshold=cosine_threshold,
    center=center,
  )

  if save_figs:
    fig.savefig(
      os.path.join(save_dir, "success_path_reuse.pdf"), bbox_inches="tight", dpi=300
    )

  # Plot path reuse rate histograms (2-column: known vs unknown eval goal)
  known_label = "Human (known eval goal)"
  unknown_label = "Human (unknown eval goal)"
  fig_hist, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
  for ax_h, label, panel_title in [
    (ax_left, known_label, "Known Eval Goal"),
    (ax_right, unknown_label, "Unknown Eval Goal"),
  ]:
    rates = all_data[label]["reuse_rates"]
    sns.histplot(
      rates * 100,
      kde=False,
      ax=ax_h,
      bins=np.arange(0, 110, 10),
      stat="count",
    )
    ax_h.set_title(panel_title, fontsize=analysis_utils.DEFAULT_TITLE_SIZE)
    ax_h.set_ylabel("Counts", fontsize=analysis_utils.DEFAULT_LABEL_SIZE)
    ax_h.set_xlabel("", fontsize=analysis_utils.DEFAULT_LABEL_SIZE)
    ax_h.set_xlim(-5, 105)
  fig_hist.tight_layout()
  if save_figs:
    fig_hist.savefig(
      os.path.join(save_dir, "path_reuse_histogram.pdf"),
      bbox_inches="tight",
      dpi=300,
    )

  fig, ax = plot_success_rate_path_reuse_metrics_efficiency(
    df=user_df,
    model_df=model_df,
    experiment_name="5.craftax_path_reuse_stats_efficiency",
    title=title,
    reuse_column=reuse_column,
    figsize=(6, 4),
    legend_loc=legend_loc,
    overlap_threshold=overlap_threshold,
    cosine_threshold=cosine_threshold,
  )

  if save_figs:
    fig.savefig(
      os.path.join(save_dir, "success_path_reuse_efficiency.pdf"),
      bbox_inches="tight",
      dpi=300,
    )

    fig, ax = plot_efficiency(
      df=user_df,
      title=title,
      figsize=(6, 4),
    )

    fig.savefig(os.path.join(save_dir, "efficiency.pdf"), bbox_inches="tight", dpi=300)


def plot_non_reuse_frequency_by_world_seed(
  user_df: DataFrame,
  model_df: DataFrame,
  save_dir: str,
  save_figs: bool = True,
  display_figs: bool = True,
  figsize=(10, 6),
  title="Frequency of Non-Reuse (reuse=0) by World Seed",
  tell_reuse=1,
  overlap_threshold: float = 0.2,
):
  """
  Creates a grouped bar plot showing the frequency of non-reuse (reuse == 0)
  for each world_seed, comparing user data and model data.

  Frequency is calculated as: count(reuse=0) / count(total) for each world_seed.

  Args:
      user_df (DataFrame): DataFrame containing human data with 'world_seed' and 'reuse' columns.
      model_df (DataFrame): DataFrame containing model data with 'world_seed' and 'reuse' columns.
      save_dir (str): Directory to save the plot.
      save_figs (bool, optional): Whether to save the figure. Defaults to True.
      display_figs (bool, optional): Whether to display the figure. Defaults to True.
      figsize (tuple, optional): Figure size. Defaults to (10, 6).
      title (str, optional): Plot title. Defaults to "Frequency of Non-Reuse (reuse=0) by World Seed".
  """
  user_df, _ = analysis_utils.filter_users_by_success_and_tell_reuse(user_df)
  user_df = user_df.with_columns(
    (pl.col("overlap") > overlap_threshold).cast(pl.Float64).alias("reuse")
  )
  model_df = model_df.with_columns(
    (pl.col("overlap") > overlap_threshold).cast(pl.Float64).alias("reuse")
  )

  # --- Process User Data ---
  settings = dict(eval=True)
  user_settings = dict(eval=True, tell_reuse=tell_reuse)
  user_counts = (
    user_df.filter(**user_settings)
    .group_by("world")
    .agg(count=pl.count())
    .sort("world")
  )
  user_reuse0 = user_df.filter(reuse=1, **user_settings)
  user_reuse0_counts = user_reuse0.group_by("world").agg(count=pl.count()).sort("world")

  # Merge and calculate probabilities
  merged_user = user_counts.join(
    user_reuse0_counts, on="world", how="left", suffix="_reuse0"
  )
  merged_user = merged_user.with_columns(
    user_frequency=pl.col("count_reuse0") / pl.col("count")
  )

  # --- Process Model Data ---
  algo = "preplay"
  model_counts = (
    model_df.filter(algo=algo, **settings)
    .group_by("world")
    .agg(count=pl.count())
    .sort("world")
  )
  model_reuse0 = model_df.filter(reuse=1, **settings, algo=algo)
  model_reuse0_counts = (
    model_reuse0.group_by("world").agg(count=pl.count()).sort("world")
  )

  # Merge and calculate probabilities
  merged_model = model_counts.join(
    model_reuse0_counts, on="world", how="left", suffix="_reuse0"
  )
  merged_model = merged_model.with_columns(
    model_frequency=pl.col("count_reuse0") / pl.col("count")
  )

  # --- Combine Data for Plotting ---
  plot_data = (
    merged_user.select(["world", "user_frequency"])
    .join(
      merged_model.select(["world", "model_frequency"]),
      on="world",
      how="outer",
    )
    .sort("world")
  )

  world_seeds = plot_data["world"].to_list()
  user_frequencies = plot_data["user_frequency"].to_numpy()
  model_frequencies = plot_data["model_frequency"].to_numpy()

  # --- Create Plot ---
  x = np.arange(len(world_seeds))  # the label locations
  width = 0.35  # the width of the bars

  fig, ax = plt.subplots(figsize=figsize)
  rects1 = ax.bar(
    x - width / 2,
    user_frequencies,
    width,
    label="User",
    color=plot_configs.default_colors["orange"],
  )
  rects2 = ax.bar(
    x + width / 2,
    model_frequencies,
    width,
    label="Model",
    color=plot_configs.default_colors["light gray"],
  )

  # Add some text for labels, title and axes ticks
  ax.set_ylabel("Probability(reuse=1)")
  ax.set_xlabel("World Seed")
  ax.set_title(title, fontsize=analysis_utils.DEFAULT_TITLE_SIZE)
  ax.set_xticks(x)
  ax.set_xticklabels(world_seeds, rotation=45, ha="right")
  ax.legend(fontsize=analysis_utils.DEFAULT_LEGEND_SIZE)
  ax.grid(True, linestyle="--", alpha=0.7, axis="y")

  ax.set_ylim(
    0, max(np.max(user_frequencies), np.max(model_frequencies)) * 1.1
  )  # Add some padding to y-axis

  fig.tight_layout()

  # --- Save and Display ---
  if save_figs:
    plot_filename_base = os.path.join(save_dir, "non_reuse_frequency_by_seed")
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{plot_filename_base}.pdf", bbox_inches="tight", dpi=300)
    # fig.savefig(f"{plot_filename_base}.png", bbox_inches="tight", dpi=300)
    print(f"Saved plot to {plot_filename_base}.pdf")

  return fig, ax
