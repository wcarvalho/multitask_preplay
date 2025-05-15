import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List, Tuple
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import os.path

from housemaze.human_dyna import utils

from analysis import analysis_utils
from nicewebrl.dataframe import DataFrame
import matplotlib.patches as mpatches
import data_configs
import plot_configs

DEFAULT_TITLE_SIZE = 15
DEFAULT_LABEL_SIZE = 15
DEFAULT_LEGEND_SIZE = 10.5

image_dict = utils.load_image_dict()


def add_to_file(stats_file, text):
  with open(data_configs.PAPER_STATS_FILE, "a") as f:
    try:
      f.write(f"{os.path.basename(stats_file.name)}\n")
    except:
      pass
    f.write(f"{text}\n")
    f.write("\n")


def num_users(df):
  return len(df["user_id"].unique())


def filter_users_by_success(df, **kwargs):
  print("Num initial users: ", num_users(df))

  df = df.filter(min_train_success=True, eval=True)
  print("Num initial users after success filter: ", num_users(df))

  unique_user_ids = list(df["user_id"].unique())
  unique_user_ids = unique_user_ids[: min(100, len(unique_user_ids))]
  print(f"Adding {len(unique_user_ids)} users")

  df = df.filter(pl.col("user_id").is_in(unique_user_ids))
  print("Num initial users after first 100 filter: ", num_users(df))
  return df


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


######################################
# Model Analysis
######################################


def episode_sf_value(e, idx=None):
  actions = e.actions
  preds = e.transitions.extras["preds"]
  sf_values = preds.sf  # [T, N, A, W]
  actions = e.actions  # [T]

  sf_values = jnp.take_along_axis(sf_values, actions[:, None, None, None], axis=-2)

  sf_values = jnp.squeeze(sf_values, axis=-2)  # [T, N, W]

  in_episode = analysis_utils.get_in_episode(e.timesteps)
  sf_values = sf_values[in_episode]
  # [T', ... ]
  if idx is not None:
    sf_values = sf_values[:, idx]
  return sf_values


def plot_sf_values(
  e,
  idxs=None,
  line_mask=None,
  line_names=None,
  figsize=None,
  colors=None,
  styles=None,
  task_w=None,
  plot_q_values=True,
):
  """Plot successor feature values as lines in multiple panels.

  Args:
      e: Episode data
      idxs: List of indices for SF values to plot in separate panels. If None, plots all indices
      line_mask: Optional boolean mask of length N to filter which lines to plot
      line_names: Optional list of names for each line
      figsize: Optional figure size tuple (width, height)
      colors: Optional list of colors for each line pair
      styles: Optional list of linestyles for first/second half
      plot_q_values: Boolean to determine whether to plot Q-values (default: True)

  Returns:
      fig: matplotlib figure object
      axs: array of matplotlib axis objects
  """
  # Get all indices if none specified
  all_sf_values = episode_sf_value(e)  # Get full SF values to determine shape
  if idxs is None:
    idxs = list(range(all_sf_values.shape[1]))  # Use all available indices

  # Calculate figure size based on number of panels
  if figsize is None:
    figsize = (7 * len(idxs), 5)

  fig, axs = plt.subplots(1, len(idxs), figsize=figsize)
  if len(idxs) == 1:
    axs = [axs]  # Make iterable for single panel case

  line_mask = line_mask or [True, True, False, False, True, True, False, False]

  line_names = line_names or [
    "main-task",
    "off-task",
    "main2-task",
    "off-task2",
    "main landmark",
    "off-task landmark",
    "main2 landmark",
    "off-task2 landmark feature",
  ]
  # Get first half of line names and take every even index (0, 2)
  first_half = line_names[
    : len(line_names) // 2
  ]  # ['main', 'off-task', 'main2', 'off-task2']
  policy_names = first_half[::2]  # ['main', 'main2']

  colors = colors or ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
  styles = styles or ["-", "--"]

  in_episode = analysis_utils.get_in_episode(e.timesteps)
  if task_w is None:
    task_w = e.timesteps.observation.task_w
    task_w = task_w[in_episode]
  max_value = -1000
  for panel_idx, idx in enumerate(idxs):
    sf_values = all_sf_values[:, idx]
    q_value = (sf_values * task_w).sum(-1)
    max_value = max(max_value, q_value.max(), sf_values.max())
    ax = axs[panel_idx]

    time_steps = np.arange(sf_values.shape[0])
    n_total = sf_values.shape[1]
    n_half = n_total // 2

    for i in range(sf_values.shape[1]):
      if line_mask is not None and not line_mask[i]:
        continue

      color_idx = i % n_half
      style_idx = i // n_half

      # Only show legend in first panel
      label = (
        line_names[i] if line_names and i < len(line_names) and panel_idx == 0 else None
      )
      ax.plot(
        time_steps,
        sf_values[:, i],
        label=label,
        color=colors[color_idx],
        linestyle=styles[style_idx],
      )

    # Add Q-value plot if plot_q_values is True
    if plot_q_values:
      ax.plot(
        time_steps,
        q_value,
        label="Q-value" if panel_idx == 0 else None,
        color="k",
        linestyle="-",
      )

    if len(idxs) > 1:
      try:
        ax.set_title(
          f"Successor Feature Predictions (task={policy_names[idx]})",
          fontsize=DEFAULT_TITLE_SIZE,
        )
      except:
        pass
    else:
      ax.set_title("Successor Feature Predictions", fontsize=DEFAULT_TITLE_SIZE)
    ax.set_xlabel("Time Step", fontsize=DEFAULT_LABEL_SIZE)
    ax.set_ylabel("Value", fontsize=DEFAULT_LABEL_SIZE)
    ax.set_xlim(0, sf_values.shape[0] - 1)
    ax.set_ylim(0, 1.1 * max_value)

  # Only show legend in first panel
  if line_names is not None:
    axs[0].legend()

  # Adjust spacing between subplots
  plt.tight_layout()

  return fig, axs


############################################################################
# Experiment results
############################################################################
def path_reuse_results(
  user_df: DataFrame,
  model_df: DataFrame,
  save_dir: str = None,
  filter_columns: List[str] = [],
  tell_reuse: int = 1,
  display_figs: bool = False,
  save_figs: bool = True,
  verbosity: int = 0,
  n_simulations: int = 1000,
  rereun_analysis: bool = False,
  rt_ylim: Tuple[float, float] = None,
  overlap_threshold: float = 0.7,
):
  """_summary_

  1. Filter out users with less than 16 successes during training

    Args:
      user_df (DataFrame): _description_
      model_df (DataFrame): _description_
  """
  save_dir = save_dir or data_configs.JAXMAZE_RESULTS_DIR
  save_dir = os.path.join(save_dir, f"1.path_reuse_tell_reuse={tell_reuse}")
  os.makedirs(save_dir, exist_ok=True)

  # Open stats file
  stats_file = open(os.path.join(save_dir, "two_paths_stats.txt"), "w")
  stats_file.write("Two Paths Statistical Analysis\n\n")

  ##################
  # Get relevant simulations
  ##################
  mdf = model_df.filter(maze="big_m3_maze1", eval=True)

  ##################
  # get all episodes for users who achieved at least 16 successes during training
  ##################
  sub_df = filter_users_by_success(
    user_df.filter(
      manipulation=3,
      tell_reuse=tell_reuse,
      eval_shares_start_pos=True,
    )
  )

  ##################
  # Create success rate and path reuse plot
  ##################
  fig, ax = analysis_utils.plot_success_rate_path_reuse_metrics(
    df=sub_df,
    model_df=mdf,
    stats_file=stats_file,
    title="Path Reuse & Generalization Success",
    figsize=(6, 4),
    include_raw_data=False,
    legend_loc="center left",
    legend_ncol=1,
    overlap_threshold=overlap_threshold,
  )

  if save_figs:
    fig.savefig(
      os.path.join(save_dir, "success_rate_path_reuse.pdf"), bbox_inches="tight"
    )

  # if display_figs:
  #  from IPython.display import display
  #  display(fig)

  ######################
  # Plot Response times when using new path vs. partial reuse
  ######################
  stats_file.write("\nResponse Time Analysis\n")
  stats_file.write("======================================\n")

  do_analysis = [True, True, False, False, False]
  # measure_to_ylim = {
  #  'max_log_rt': None,
  #  'first_log_rt': None,
  #  #'first_rt': None,
  #  #'max_rt': None,
  #  #'total_rt': None,
  # }
  sub_df = sub_df.filter(pl.col("reuse") != -1)
  for idx, measure in enumerate(
    [
      "max_log_rt",
      "first_log_rt",
      #'first_rt', 'max_rt', 'total_rt'
    ]
  ):
    stats_file.write(f"\n{measure}\n")
    stats_file.write("--------------------\n")

    for use_box_plot in [False]:
      fig, ax = plt.subplots(figsize=(4, 4))
      analysis_utils.plot_bar_rt_comparison(
        sub_df.filter(success=1),
        measure,
        n_simulations=n_simulations if do_analysis[idx] else 1,
        stats_file=stats_file if do_analysis[idx] else None,
        ax=ax,
        rereun_analysis=rereun_analysis,
        ylim=None if use_box_plot else (7, 8),
        use_box_plot=use_box_plot,
        ylabel="Log RT",
      )

      if save_figs:
        fig.savefig(
          os.path.join(save_dir, f"rt_comparison_{measure}_box={use_box_plot}.pdf"),
          bbox_inches="tight",
        )
      if display_figs:
        plt.show()

  # Close stats file at the end
  stats_file.close()
  # if verbosity > 0:
  #  with open(stats_file_, "r") as f:
  #    print(f.read())


def sf_analysis_results(
  model_df: DataFrame,
  save_dir: str = None,
  display_figs: bool = False,
  save_figs: bool = True,
):
  save_dir = save_dir or data_configs.JAXMAZE_RESULTS_DIR
  sf_episodes = model_df.filter(maze="big_m3_maze1", eval=False, algo="usfa")
  fig, ax = plot_sf_values(
    sf_episodes.episodes[0], plot_q_values=False, figsize=(5, 4), idxs=[0]
  )
  if save_figs:
    fig.savefig(os.path.join(save_dir, "sf_predictions_plots.pdf"), bbox_inches="tight")
  if display_figs:
    from IPython.display import display

    display(fig)


def juncture_results(
  user_df: DataFrame,
  # model_df: DataFrame,
  save_dir: str = None,
  filter_columns: List[str] = None,
  display_figs: bool = False,
  save_figs: bool = True,
  verbosity: int = 0,
  tell_reuse_options=[1, 0],
  figsize=(5.5, 4),
  include_raw_data: bool = False,
  show_legend: bool = True,
  options: List[Tuple[str, int]] = None,
  measure="first_log_rt",
  use_median: bool = False,
  ylim=None,
):
  """Analyze results from experiment 4.

  Args:
      user_df (DataFrame): DataFrame containing user data
      model_df (DataFrame): DataFrame containing model data
      save_dir (str): Directory to save figures
      filter_columns (List[str], optional): Columns to use for outlier filtering in RT analysis.
          Defaults to ['avg_rt'].
      display_figs (bool, optional): Whether to display figures. Defaults to False.
      save_figs (bool, optional): Whether to save figures. Defaults to True.
  """
  save_dir = save_dir or data_configs.JAXMAZE_RESULTS_DIR

  save_dir = os.path.join(save_dir, "2.juncture")
  os.makedirs(save_dir, exist_ok=True)
  # Default to ['avg_rt'] if no filter columns specified
  filter_columns = filter_columns or []

  # Open stats file
  stats_filename = os.path.join(save_dir, "juncture_stats.txt")
  stats_file = open(stats_filename, "w")
  stats_file.write("Juncture Manipulation Statistical Analysis\n\n")

  user_df = filter_users_by_success_and_tell_reuse(user_df.filter(manipulation=4))
  first_100_users = []
  for tell_reuse in [1, 0]:
    unique_user_ids = user_df.filter(tell_reuse=tell_reuse)["user_id"].unique()
    unique_user_ids = unique_user_ids[: min(100, len(unique_user_ids))]
    print(f"Adding {len(unique_user_ids)} users for tell_reuse={tell_reuse}")
    first_100_users.extend(unique_user_ids)

  # Filter dataframe to only include rows with those user IDs
  user_df = user_df.filter(pl.col("user_id").is_in(first_100_users))

  ##################
  # Add setting column based on maze name
  ##################
  user_df = analysis_utils.get_polars_df(user_df)  # fancy merging will use regular df
  user_df = user_df.filter(manipulation=4)

  def get_maze_setting(maze_str: str) -> str:
    if "short" in maze_str.lower():
      return "short"
    elif "long" in maze_str.lower():
      return "long"
    raise ValueError(f"Could not determine setting from maze string: {maze_str}")

  # Add setting column based on maze name
  user_df = user_df.with_columns(
    setting=pl.col("maze").map_elements(get_maze_setting, return_dtype=pl.String)
  )

  ############################################
  # Create combined figure with all conditions on one plot
  ############################################
  fig, ax = plt.subplots(figsize=figsize)

  # We'll focus only on first RT

  # Define colors and labels for each condition
  condition_colors = {
    ("short", 1): plot_configs.default_colors["sky blue"],  # Near x Known
    ("long", 1): plot_configs.default_colors["vermillion"],  # Far x Known
    ("short", 0): plot_configs.default_colors["bluish green"],  # Near x Unknown
  }

  condition_labels = {
    ("short", 1): "Near, Known Test goal",
    ("long", 1): "Far, Known Test goal",
    ("short", 0): "Near, Unknown Test goal",
  }

  # Store all data for combined plot
  all_diffs = []
  all_means = []
  all_sems = []
  all_labels = []
  all_colors = []

  options = options or [
    ("short", 1),
    ("short", 0),
    ("long", 1),
  ]

  # Collect data for each condition
  for setting, tell_reuse in options:
    stats_file.write(f"\n\n=================={setting}===================\n")
    difference_df = analysis_utils.compute_condition_difference_df(
      user_df.filter(setting=setting, tell_reuse=tell_reuse),
      measures=[measure],
    )
    stats_file.write(f"\n\nRT Analysis for tell_reuse={tell_reuse}\n")
    stats_file.write("-----------------------------------------\n")

    # Get statistics for this condition
    results = analysis_utils.power_analysis_rt_differences(
      difference_df, measure, stats_file=stats_file
    )

    # Store data for plotting
    all_diffs.append(difference_df[measure].to_numpy())
    all_means.append(results["median"])  # Use median instead of mean
    all_sems.append(results["median_ci"])  # Use bootstrapped CI instead of SE
    all_labels.append(condition_labels[(setting, tell_reuse)])
    all_colors.append(condition_colors[(setting, tell_reuse)])

  # Create bar plot with all conditions
  x_pos = np.arange(len(all_means))

  # Convert from CI to lower/upper error values needed by matplotlib
  all_sems_array = np.array(all_sems)
  all_means_array = np.array(all_means)

  # Calculate asymmetric error bars (lower and upper offsets)
  lower_errors = all_means_array - all_sems_array[:, 0]
  upper_errors = all_sems_array[:, 1] - all_means_array

  ax.bar(
    x_pos,
    all_means,
    yerr=[
      lower_errors,
      upper_errors,
    ],  # Format for asymmetric error bars: [lower_errors, upper_errors]
    capsize=5,
    color=all_colors,
  )

  # Add individual points with jitter
  if include_raw_data:
    for i, diffs in enumerate(all_diffs):
      x_jitter = np.random.normal(i, 0.125, size=len(diffs))
      ax.scatter(x_jitter, diffs, alpha=0.3, color="black", s=20)

  # Add zero line
  ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)

  # Customize plot
  ax.set_xticks(x_pos)
  ax.set_xticklabels([])
  ax.set_ylabel("$\Delta$ Log RT", fontsize=DEFAULT_LABEL_SIZE)
  ax.set_title(
    "Juncture Manipulation\nFirst Response Time Difference", fontsize=DEFAULT_TITLE_SIZE
  )
  ax.tick_params(axis="both", which="major", labelsize=DEFAULT_LABEL_SIZE)
  ax.grid(True, linestyle="--", alpha=0.7)

  # Create legend with colored patches
  legend_elements = [
    mpatches.Patch(color=all_colors[i], label=all_labels[i])
    for i in range(len(all_labels))
  ]
  if show_legend:
    ax.legend(handles=legend_elements, loc="lower right", fontsize=DEFAULT_LEGEND_SIZE)

  # Set y-axis limits based on all data points
  if ylim is None:
    all_data = np.concatenate(all_diffs)
    y_min, y_max = np.percentile(all_data, [1, 99])
  else:
    y_min, y_max = ylim
  y_range = y_max - y_min
  ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

  # Adjust layout
  plt.tight_layout()

  # Save combined figure in multiple formats
  if save_figs:
    base_path = os.path.join(save_dir, "exp4_2_rt_diff_combined")
    fig.savefig(f"{base_path}_{measure}.pdf", bbox_inches="tight")
    # fig.savefig(f"{base_path}_{measure}.png", bbox_inches="tight", dpi=300)
  if display_figs:
    from IPython.display import display

    display(fig)

  # Close stats file at the end
  stats_file.close()
  if verbosity > 0:
    with open(stats_filename, "r") as f:
      print(f.read())


def shortcut_results(
  user_df: DataFrame,
  model_df: DataFrame,
  save_dir: str = None,
  filter_columns: List[str] = None,
  display_figs: bool = False,
  tell_reuse: int = 1,
  save_figs: bool = True,
  verbosity: int = 0,
  overlap_threshold: float = 0.6,
):
  """_summary_

  1. Filter out users with less than 16 successes during training

    Args:
      user_df (DataFrame): _description_
      model_df (DataFrame): _description_
  """
  save_dir = save_dir or data_configs.JAXMAZE_RESULTS_DIR
  save_dir = os.path.join(save_dir, f"4.shortcut_tell_reuse={tell_reuse}")
  os.makedirs(save_dir, exist_ok=True)

  # Open stats file
  stats_file = open(os.path.join(save_dir, "3.shortcut_stats.txt"), "w")
  stats_file.write("Shortcut Manipulation Statistical Analysis\n")

  mdf = model_df.filter(maze="big_m1_maze3_shortcut", eval=True)

  sub_df = filter_users_by_success(
    user_df.filter(manipulation=1, tell_reuse=tell_reuse, eval_shares_start_pos=True)
  )

  ##################
  # Create success rate and path reuse plots
  ##################
  fig, ax = analysis_utils.plot_success_rate_path_reuse_metrics(
    df=sub_df,
    model_df=mdf,
    stats_file=stats_file,
    title="Path Reuse & Generalization Success",
    figsize=(6, 4),
    include_raw_data=False,
    legend_loc="center left",
    overlap_threshold=overlap_threshold,
  )

  if save_figs:
    fig.savefig(os.path.join(save_dir, "exp2_2_success_rate.pdf"), bbox_inches="tight")

  if display_figs:
    from IPython.display import display

    display(fig)

  # Close stats file at the end
  stats_file.close()
  if verbosity > 0:
    with open(os.path.join(save_dir, "stats.txt"), "r") as f:
      print(f.read())


def start_results(
  user_df: DataFrame,
  save_dir: str = None,
  filter_columns: List[str] = None,
  display_figs: bool = False,
  tell_reuse: int = 1,
  save_figs: bool = True,
  verbosity: int = 0,
  ylim: Tuple[float, float] = None,
):
  """_summary_

  1. Filter out users with less than 16 successes during training

    Args:
      user_df (DataFrame): _description_
      save_dir (str): Directory to save figures
      filter_columns (List[str], optional): Columns to use for outlier filtering in RT analysis.
          Defaults to ['avg_rt'].
      display_figs (bool, optional): Whether to display figures. Defaults to False.
      save_figs (bool, optional): Whether to save figures. Defaults to True.
  """
  save_dir = save_dir or data_configs.JAXMAZE_RESULTS_DIR
  save_dir = os.path.join(save_dir, f"3.start_tell_reuse={tell_reuse}")
  os.makedirs(save_dir, exist_ok=True)
  # Default to ['avg_rt'] if no filter columns specified

  stats_file = open(os.path.join(save_dir, "start_stats.txt"), "w")
  stats_file.write("Intermediary Start Manipulation Statistical Analysis\n")
  stats_file.write("===============================\n\n")

  ##################
  # get all episodes for users who achieved at least 16 successes during training
  ##################
  sub_df = filter_users_by_success(
    user_df.filter(manipulation=2, tell_reuse=tell_reuse)
  )

  ##################
  # Create Response time difference plot
  ##################
  # Create filter string for filename
  filter_columns = filter_columns or []
  filter_str = ",".join(filter_columns)
  difference_df = analysis_utils.compute_condition_difference_df(
    analysis_utils.get_polars_df(sub_df),
    measures=[
      "first_log_rt",
      # "max_log_rt",
      # "avg_log_rt",
    ],
  )
  xlabels = [
    "",
    # "Max",
    # "Average",
  ]
  measures = [
    "first_log_rt",
    # "max_log_rt",
    # "avg_log_rt",
  ]
  colors = [
    plot_configs.default_colors["google blue"],
    # plot_configs.default_colors["sky blue"],
    # default_colors["google orange"],
  ]
  fig, ax = plt.subplots(figsize=(3, 4))
  fig, ax = analysis_utils.plot_rt_differences(
    difference_df,
    measures=measures,
    title="Start Manipulation\nFirst Response Time Difference",
    colors=colors,
    ylabel="$\Delta$ Log RT",
    xlabels=xlabels,
    stats_file=stats_file,
    ax=ax,
    ylim=ylim,
  )

  if save_figs:
    fig.savefig(
      os.path.join(save_dir, f"exp3_2_rt_diff_filter_{filter_str}.pdf"),
      bbox_inches="tight",
    )
  if display_figs:
    from IPython.display import display

    display(fig)
  stats_file.close()
  if verbosity > 0:
    with open(os.path.join(save_dir, "stats.txt"), "r") as f:
      print(f.read())


if __name__ == "__main__":
  from data_processing import process_model_data
  from data_processing import process_user_data

  user_df = process_user_data.get_jaxmaze_human_data()
  model_df = process_model_data.get_jaxmaze_model_data()

  save_dir = (data_configs.JAXMAZE_RESULTS_DIR,)
  os.makedirs(save_dir, exist_ok=True)

  path_reuse_results(user_df, model_df, save_dir=save_dir)
  juncture_results(user_df, model_df, save_dir=save_dir)
  start_results(user_df, model_df, save_dir=save_dir)
  shortcut_results(user_df, model_df, save_dir=save_dir)
