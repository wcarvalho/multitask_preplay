"""
Functions for
(1) getting dataframes related to different experiments
(2) plotting related metrics
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_configs
from typing import List, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import os.path
from analysis import analysis_utils

from nicewebrl import DataFrame
from simulations import craftax_utils
from simulations import craftax_experiment_configs
import craftax_experiment_structure as experiment
from IPython.display import HTML, display
from functools import partial
from data_configs import default_colors, model_colors, model_names

OPTIMAL_TEST_PATHS = {}
for config in craftax_experiment_configs.PATHS_CONFIGS:
  # Create cache path
  cache_dir = "craftax_cache/optimal_paths"
  os.makedirs(cache_dir, exist_ok=True)
  cache_file = os.path.join(cache_dir, f"path_{hash(str(config))}.npy")

  # Try to load from cache
  if os.path.exists(cache_file):
    path = np.load(cache_file)
  else:
    # Calculate path and save to cache
    env_params = craftax_experiment_configs.make_block_env_params(
      config, experiment.default_params
    ).replace(
      # goal_locations=config.test_object_location,
      # current_goal=jnp.asarray(config.test_objects[0], dtype=jnp.int32),
      start_positions=experiment.make_start_position(config.start_eval_positions),
    )
    timestep = experiment.jax_web_env.reset(jax.random.PRNGKey(0), env_params)
    goal_position = config.test_object_location
    path, _ = craftax_utils.astar(timestep.state, goal_position)
    path = np.array(path)

    np.save(cache_file, path)
  OPTIMAL_TEST_PATHS[config.world_seed] = path

OPTIMAL_TEST_LENGTHS = {k: len(v) - 1 for k, v in OPTIMAL_TEST_PATHS.items()}


def filter_to_str(filter: dict):
  return "".join([f"{k}={v}" for k, v in filter.items()])


def get_path_reuse_df(
  user_df: DataFrame,
  tell_reuse: int = 1,
  eval_map: bool = False,
):
  sub_df = user_df.filter_by_group(
    input_episode_filter=analysis_utils.filter_train_by_min_success,
    input_settings=dict(eval=False),
    output_settings=dict(
      manipulation="paths", tell_reuse=tell_reuse, eval_map=int(eval_map)
    ),
    group_key="user_id",
  ).filter(eval=True)
  sub_df = analysis_utils.fix_reuse_column(sub_df)
  sub_df = sub_df.filter(
    pl.col("success").is_not_null() & pl.col("reuse").is_not_null()
  )
  return sub_df


def path_similarity(path1, path2):
  path1 = path1[-10:]
  path2 = path2[-10:]

  # Convert to direction vectors
  def to_directions(path):
    dirs = []
    for i in range(len(path) - 1):
      dx = path[i + 1][0] - path[i][0]
      dy = path[i + 1][1] - path[i][1]
      mag = (dx * dx + dy * dy) ** 0.5
      if mag > 0:
        dirs.append((dx / mag, dy / mag))
    return dirs

  dirs1 = to_directions(path1)
  dirs2 = to_directions(path2)

  # Resample longer to match shorter
  min_len = min(len(dirs1), len(dirs2))
  if len(dirs1) > min_len:
    step = len(dirs1) / min_len
    dirs1 = [dirs1[int(i * step)] for i in range(min_len)]
  elif len(dirs2) > min_len:
    step = len(dirs2) / min_len
    dirs2 = [dirs2[int(i * step)] for i in range(min_len)]

  # Average dot product
  dots = sum(d1[0] * d2[0] + d1[1] * d2[1] for d1, d2 in zip(dirs1, dirs2))
  similarity = (dots / min_len + 1) / 2  # Map from [-1,1] to [0,1]

  return similarity


def visualize_user_path_reuse(df: DataFrame, user_id: int, idx=None, **kwargs):
  user_df = df.filter(user_id=user_id)
  test_mazes = user_df["name"].unique()
  test_mazes = [t for t in test_mazes if "eval" in t]
  test_mazes = sorted(test_mazes)

  for i in range(len(test_mazes)):
    if idx is not None:
      if i != idx:
        continue
    # get random test maze
    i = int(i)
    test_maze = test_mazes[i]
    test_df = user_df.filter(eval=True, name=test_maze, **kwargs)
    if len(test_df.episodes) == 0:
      print(f"No test episodes for maze: {test_maze}")
      continue

    # get corresponding train maze
    train_maze = test_maze.replace("eval1", "training")
    start_pos = test_df["start_pos"].to_list()[0]
    train_df = user_df.filter(
      name=train_maze, room=0, eval=False, success=1, start_pos=start_pos
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
    first = lambda t: jax.tree_map(lambda x: x[0], t)
    train_episode = train_df.episodes[0]
    test_episode = test_df.episodes[0]
    with jax.disable_jit():
      display(HTML(test_df.to_pandas().to_html()))
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
      title += f"\nOptimal Path Length: {OPTIMAL_TEST_LENGTHS[world_seed]}"
      axs[2].set_title(title, fontsize=13)

      # plot reaction times
      reaction_times = test_episode.reaction_times
      axs[3].bar(range(len(reaction_times)), reaction_times, color="lightblue")
      axs[3].set_xlabel("Time")
      axs[3].set_title("Reaction Times")
      axs[3].set_ylim(0, max(reaction_times) * 1.1)

    plt.show()


# def add_to_file(stats_file, text):
#  with open(data_configs.PAPER_STATS_MODEL_FILE, "a") as f:
#    try:
#      f.write(f"{os.path.basename(stats_file.name)}\n")
#    except:
#      pass
#    f.write(f"{text}\n")
#    f.write("\n")


def plot_success_rate_path_reuse_metrics(
  df: DataFrame,
  model_df: DataFrame = None,
  stats_file=None,
  ax=None,
  reuse_column: str = "reuse",
  title="Success Rate and Path Reuse",
  figsize=(8, 8),
  legend_loc: str = "lower right",
  legend_ncol: int = 1,
) -> Tuple[plt.Figure, plt.Axes]:
  """Plot success rate vs path reuse as a 2D scatter plot with error bars.

  Args:
      df (DataFrame): DataFrame containing human data
      model_df (DataFrame, optional): DataFrame containing model data. If None, only plots human data.
      stats_file (file, optional): File to write statistics to
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

  # Process human data for tell_reuse=1 and tell_reuse=0
  tell_reuse_values = [1, 0]
  tell_reuse_labels = [
    "Human (know eval goal = True)",
    "Human (know eval goal = False)",
  ]
  tell_reuse_markers = ["o", "x"]  # Plus for True, X for False

  for i, (tell_value, label, marker) in enumerate(
    zip(tell_reuse_values, tell_reuse_labels, tell_reuse_markers)
  ):
    # Filter data for this tell_reuse value
    filtered_df = df.filter(tell_reuse=tell_value, eval=True)

    # Calculate statistics with consistent ordering
    user_data = (
      filtered_df.group_by("user_id")
      .agg(
        success_mean=pl.col("success").mean(), reuse_mean=pl.col(reuse_column).mean()
      )
      .sort("user_id")  # Sort by user_id for consistent ordering
    )

    human_successes = user_data["success_mean"].to_numpy()
    human_success_mean = np.mean(human_successes)

    human_success_se = np.sqrt(
      (human_success_mean * (1 - human_success_mean)) / len(human_successes)
    )

    stats_file.write("================================\n")
    stats_file.write(f"Tell Reuse: {tell_value}\n")
    stats_file.write("================================\n")
    results = analysis_utils.power_analysis_path_reuse(
      filtered_df,
      measure=reuse_column,
      mu=0.5,
      alpha=0.05,
      plot=False,
      stats_file=stats_file,
    )
    summary = "Success:\n"
    summary += f"(Mean={human_success_mean:.2f}, SE={human_success_se:.2f}, SD={np.sqrt(human_success_mean * (1 - human_success_mean)):.2f})\n"
    stats_file.write(summary)

    # human_reuse = user_data["reuse_mean"].to_numpy()
    human_reuse_mean = results["mean"]
    human_reuse_se = results["se"]

    analysis_utils.add_to_file(
      stats_file,
      algo="human data",
      label="success",
      text=f"success: {100 * human_success_mean:.2f}, SE={100 * human_success_se:.2f}",
    )
    # experiment_analysis.add_to_file(
    #  stats_file,
    #  algo='human data',
    #  label='reuse',
    #  text=f"reuse: {100*human_reuse_mean:.2f}, SE={100*human_reuse_se:.2f}")
    # add_to_file(stats_file, f"reuse={tell_value}: Human reuse: {100*human_reuse_mean:.2f}, SE={100*human_reuse_se:.2f}")

    # Add to data dictionary
    all_data[label] = {
      "success": 100 * human_success_mean,
      "reuse": 100 * human_reuse_mean,
      "success_se": 100 * human_success_se,
      "reuse_se": 100 * human_reuse_se,
      "marker": marker,  # Store the marker type
    }

  # Add model data if provided
  if model_df is not None:
    # Calculate model statistics
    model_stats = (
      model_df.filter(eval=True)
      .group_by("algo")
      .agg(
        success_mean=(pl.col("success").cast(pl.Float64).mean() * 100),
        success_se=(
          pl.col("success").cast(pl.Float64).mean()
          * (1 - pl.col("success").cast(pl.Float64).mean())
          / pl.count()
        ).sqrt()
        * 100,
        reuse_mean=(pl.col("reuse").cast(pl.Float64).mean() * 100),
        reuse_se=(
          pl.col("reuse").cast(pl.Float64).mean()
          * (1 - pl.col("reuse").cast(pl.Float64).mean())
          / pl.count()
        ).sqrt()
        * 100,
      )
    )

    # Add model data
    algos = model_stats["algo"].unique().to_list()
    for algo in analysis_utils.model_order:
      if algo not in algos:
        continue
      row = model_stats.filter(algo=algo)
      all_data[algo] = {
        "success": row["success_mean"].to_numpy()[0],
        "reuse": row["reuse_mean"].to_numpy()[0],
        "success_se": min(row["success_se"].to_numpy()[0], 5),
        "reuse_se": min(row["reuse_se"].to_numpy()[0], 5),
        "marker": "o",  # Use circle marker for models
      }
      analysis_utils.add_to_file(
        stats_file,
        algo=algo,
        label="success",
        text=f"success: {all_data[algo]['success']:.2f}, SE={all_data[algo]['success_se']:.2f}",
      )
      analysis_utils.add_to_file(
        stats_file,
        algo=algo,
        label="reuse",
        text=f"reuse: {all_data[algo]['reuse']:.2f}, SE={all_data[algo]['reuse_se']:.2f}",
      )

  # Define colors for human data points
  human_colors = {
    tell_reuse_labels[0]: plot_configs.default_colors["orange"],  # Green
    tell_reuse_labels[1]: plot_configs.default_colors["light gray"],  # Red
  }

  # Plot data points with error bars
  marker_size = 50  # Default size for all scatter points

  # First plot human data points
  for key in tell_reuse_labels:
    if key in all_data:
      data = all_data[key]
      color = human_colors.get(key, "#333333")
      marker = data["marker"]

      ax.errorbar(
        data["reuse"],
        data["success"],
        xerr=data["reuse_se"],
        yerr=data["success_se"],
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

  # Then plot model data points
  if model_df is not None:
    ordered_keys = [k for k in analysis_utils.model_order if k in all_data]
    for key in ordered_keys:
      if key in tell_reuse_labels:  # Skip if it's human data (already plotted)
        continue

      data = all_data[key]
      color = model_colors[key]
      print(f"Plotting {key} with color {color}")
      ax.errorbar(
        data["reuse"],
        data["success"],
        xerr=data["reuse_se"],
        yerr=data["success_se"],
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
        marker=data["marker"],
        label=analysis_utils.model_names[key],
        zorder=3,
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


def num_users(df):
  return len(df["user_id"].unique())


def filter_users_by_success_and_tell_reuse(df):
  print("Num initial users: ", num_users(df))

  df = df.filter(min_train_success=True, eval=True)
  print("Num initial users after success filter: ", num_users(df))

  first_100_users = []
  for tell_reuse in [1, 0]:
    unique_user_ids = df.filter(tell_reuse=tell_reuse)["user_id"].unique()
    unique_user_ids = unique_user_ids[: min(100, len(unique_user_ids))]
    print(f"Adding {len(unique_user_ids)} users for tell_reuse={tell_reuse}")
    first_100_users.extend(unique_user_ids)

  # Filter dataframe to only include rows with those user IDs
  df = df.filter(pl.col("user_id").is_in(first_100_users))
  print("Num initial users after first 100 filter: ", num_users(df))
  return df


def path_reuse_manipulation_analysis(
  sub_df: DataFrame,
  model_df: DataFrame,
  save_dir: str = None,
  save_figs: bool = True,
  display_figs: bool = True,
  verbosity: int = 1,
  reuse_column: str = "reuse",
  n_simulations: int = 1000,
):
  ############################################################
  # Create stats file
  ############################################################
  save_dir = save_dir or data_configs.CRAFTAX_RESULTS_DIR
  save_dir = os.path.join(save_dir, "5.craftax_path_reuse_manipulation")
  os.makedirs(save_dir, exist_ok=True)
  stats_file = os.path.join(save_dir, "craftax_path_reuse_stats.txt")
  stats_file = open(stats_file, "w")
  stats_file.write("Statistical Analysis\n\n")

  ############################################################
  # Plot success rate and path reuse
  ############################################################
  if reuse_column == "reuse":
    title = "Path Reuse & Generalization Success Rate"
  else:
    title = "Efficient Path Reuse & Generalization Success Rate"

  ############################################################
  # Get 1st 100
  ############################################################
  # Get unique user IDs
  # first_100_users = []
  # for tell_reuse in [1, 0]:
  #  unique_user_ids = sub_df.filter(tell_reuse=tell_reuse)["user_id"].unique()
  #  unique_user_ids = unique_user_ids[:min(100, len(unique_user_ids))]
  #  print(f"Adding {len(unique_user_ids)} users for tell_reuse={tell_reuse}")
  #  first_100_users.extend(unique_user_ids)

  # Filter dataframe to only include rows with those user IDs
  sub_df = filter_users_by_success_and_tell_reuse(sub_df)

  # sub_df = sub_df.filter_by_group(
  #  input_episode_filter=partial(experiment_analysis.filter_train_by_min_success, min_successes=16*4),
  #  input_settings=dict(eval=False),
  #  output_settings=dict(),
  #  group_key="user_id",
  # ).filter(eval=True)

  # first plot when tell_reuse is 1
  fig, ax = plot_success_rate_path_reuse_metrics(
    df=sub_df,
    model_df=model_df,
    stats_file=stats_file,
    title=title,
    reuse_column=reuse_column,
    figsize=(6, 4),
    legend_loc="upper left",
  )
  if reuse_column == "efficient_reuse":
    ax.set_xlabel(
      "Efficient Path Reuse (%)", fontsize=analysis_utils.DEFAULT_LABEL_SIZE
    )

  if save_figs:
    fig.savefig(
      os.path.join(save_dir, "success_path_reuse.pdf"), bbox_inches="tight", dpi=300
    )
    # fig.savefig(os.path.join(save_dir, "success_path_reuse.png"), bbox_inches="tight", dpi=300)
  # if display_figs:
  #  from IPython.display import display
  #  display(fig)

  #############################################################
  ## Plot path length
  #############################################################
  # fig, ax = plt.subplots(figsize=(4, 4))
  # experiment_analysis.plot_bar_rt_comparison(
  # sub_df.filter(success=1),
  # "path_length",
  ##title="Path Length",
  ##ylabel="Length",
  ##xlabels=["New Path", "Partial Reuse"],
  ##colors=[experiment_analysis.default_colors["nice purple"], experiment_analysis.default_colors["bluish green"]],
  # n_simulations=n_simulations,
  # stats_file=stats_file,
  # ax=ax,
  # )
  # plt.show()

  stats_file.close()
  # if verbosity > 0:
  #  with open(stats_filename, "r") as f:
  #    print(f.read())


def plot_non_reuse_frequency_by_world_seed(
  user_df: DataFrame,
  model_df: DataFrame,
  save_dir: str,
  save_figs: bool = True,
  display_figs: bool = True,
  figsize=(10, 6),
  title="Frequency of Non-Reuse (reuse=0) by World Seed",
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
  user_df = filter_users_by_success_and_tell_reuse(user_df)
  # --- Process User Data ---
  settings = dict(eval=True)
  user_counts = (
    user_df.filter(**settings)
    .group_by("world_seed")
    .agg(count=pl.count())
    .sort("world_seed")
  )
  user_reuse0 = user_df.filter(reuse=1, **settings)
  user_reuse0_counts = (
    user_reuse0.group_by("world_seed").agg(count=pl.count()).sort("world_seed")
  )

  # Merge and calculate probabilities
  merged_user = user_counts.join(
    user_reuse0_counts, on="world_seed", how="left", suffix="_reuse0"
  )
  merged_user = merged_user.with_columns(
    user_frequency=pl.col("count_reuse0") / pl.col("count")
  )

  # --- Process Model Data ---
  algo = "preplay"
  model_counts = (
    model_df.filter(algo=algo, **settings)
    .group_by("world_seed")
    .agg(count=pl.count())
    .sort("world_seed")
  )
  model_reuse0 = model_df.filter(reuse=1, **settings, algo=algo)
  model_reuse0_counts = (
    model_reuse0.group_by("world_seed").agg(count=pl.count()).sort("world_seed")
  )

  # Merge and calculate probabilities
  merged_model = model_counts.join(
    model_reuse0_counts, on="world_seed", how="left", suffix="_reuse0"
  )
  merged_model = merged_model.with_columns(
    model_frequency=pl.col("count_reuse0") / pl.col("count")
  )

  # --- Combine Data for Plotting ---
  plot_data = (
    merged_user.select(["world_seed", "user_frequency"])
    .join(
      merged_model.select(["world_seed", "model_frequency"]),
      on="world_seed",
      how="outer",
    )
    .sort("world_seed")
  )

  world_seeds = plot_data["world_seed"].to_list()
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

  if display_figs:
    from IPython.display import display

    display(fig)
  else:
    plt.close(fig)  # Close the figure if not displaying to save memory

  return fig, ax
