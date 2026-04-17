import sys
import os
import pickle
import inspect

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


def num_users(df):
  return len(df["user_id"].unique())


def filter_users_by_success(df, analysis_name=None, **kwargs):
  # Get the calling function name if not provided
  if analysis_name is None:
    analysis_name = inspect.currentframe().f_back.f_code.co_name

  # Create cache file path
  cache_path = os.path.join(
    data_configs.ANALYSIS_CACHE_DIR, f"{analysis_name}_user_ids.pkl"
  )

  # Compute user IDs if cache doesn't exist or failed to load
  print("Num initial users: ", num_users(df))

  # Try to load cached user IDs
  if os.path.exists(cache_path):
    with open(cache_path, "rb") as f:
      print(f"Loading cached user IDs from {cache_path}")
      unique_user_ids = pickle.load(f)

    # Filter dataframe to only include rows with those user IDs
    df_filtered = df.filter(pl.col("user_id").is_in(unique_user_ids))
    print("Num users after cache filter: ", num_users(df_filtered))
    return df_filtered, unique_user_ids

  df = df.filter(min_train_success=True, eval=True)
  print("Num initial users after success filter: ", num_users(df))

  # sort by 'session_start' column so earlier is first
  # columns resemble "2025-03-04T21:37:47.918051"
  df = df.sort("session_start")
  unique_user_ids = df["user_id"].unique(maintain_order=True).to_list()
  unique_user_ids = unique_user_ids[: min(100, len(unique_user_ids))]
  print(f"Adding {len(unique_user_ids)} users")
  print(unique_user_ids[:10])

  # Save to cache
  os.makedirs(os.path.dirname(cache_path), exist_ok=True)
  with open(cache_path, "wb") as f:
    pickle.dump(unique_user_ids, f)
  print(f"Saved user IDs to cache: {cache_path}")

  df = df.filter(pl.col("user_id").is_in(unique_user_ids))
  print("Num initial users after first 100 filter: ", num_users(df))
  return df, unique_user_ids


def get_path_reuse_eval_data(user_df, tell_reuse=1, eval_only=True):
  """Return filtered (user_df, model_df) for path reuse experiment."""
  eval_filter = dict(manipulation="paths", world="big_m3_maze1")
  if eval_only:
    eval_filter["eval"] = True
    eval_filter["eval_shares_start_pos"] = True
  sub_df, _ = filter_users_by_success(
    user_df.filter(tell_reuse=tell_reuse, **eval_filter),
    analysis_name="path_reuse_results",
  )
  return sub_df


def get_shortcut_eval_data(user_df, tell_reuse=1, eval_only=True):
  """Return filtered (user_df, model_df) for shortcut experiment."""
  filter_kwargs = dict(
    manipulation="shortcut",
    # world="big_m1_maze3_shortcut",
    tell_reuse=tell_reuse,
  )
  if eval_only:
    filter_kwargs["eval"] = True
    filter_kwargs["eval_shares_start_pos"] = True
  sub_df, _ = filter_users_by_success(
    user_df.filter(**filter_kwargs),
    analysis_name="shortcut_results",
  )
  return sub_df


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
  tell_reuse: int = 1,
  display_figs: bool = False,
  save_figs: bool = True,
  n_simulations: int = 1000,
  rereun_analysis: bool = False,
  overlap_threshold: float = 0.7,
  mu: float = 0.5,
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

  experiment_name = "1.two_paths_stats"

  ##################
  # Get relevant simulations and filtered user data
  ##################
  eval_filter = dict(manipulation="paths", world="big_m3_maze1", eval=True)
  mdf = model_df.filter(**eval_filter)
  sub_df = get_path_reuse_eval_data(user_df, tell_reuse=tell_reuse)

  ##################
  # Create success rate and path reuse plot (median and mean)
  ##################
  for center in ["median", "mean"]:
    fig, ax, human_data = analysis_utils.plot_success_rate_path_reuse_metrics(
      df=sub_df,
      model_df=mdf,
      experiment_name=experiment_name,
      title="Generalization Success & Path Reuse",
      figsize=(6, 4),
      legend_loc="center left",
      legend_ncol=1,
      overlap_threshold=overlap_threshold,
      center=center,
      mu=mu,
    )

    if save_figs:
      fig.savefig(
        os.path.join(save_dir, f"success_rate_path_reuse_{center}.pdf"),
        bbox_inches="tight",
      )

  if save_figs:
    analysis_utils.plot_human_rate_histograms(
      reuse_rates={"Human": human_data["human"]["reuse_rates"]},
      save_path=os.path.join(save_dir, "human_rate_distributions.pdf"),
    )

  ######################
  # Plot Response times when using new path vs. partial reuse
  ######################
  do_analysis = [True, True, False, False, False]
  for idx, measure in enumerate(
    [
      "first_log_rt",
      "max_log_rt",
    ]
  ):
    for use_box_plot in [False]:
      fig, ax = plt.subplots(figsize=(4, 4))
      analysis_utils.plot_bar_rt_comparison(
        sub_df,
        measure,
        n_simulations=n_simulations if do_analysis[idx] else 1,
        experiment_name=experiment_name if do_analysis[idx] else None,
        ax=ax,
        rereun_analysis=rereun_analysis,
        ylim=None if use_box_plot else (7, 8),
        use_box_plot=use_box_plot,
        overlap_threshold=overlap_threshold,
        ylabel="Log RT",
      )

      if save_figs:
        fig.savefig(
          os.path.join(save_dir, f"rt_comparison_{measure}_box={use_box_plot}.pdf"),
          bbox_inches="tight",
        )
      if display_figs:
        plt.show()


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
  mu: float = 0.5,
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

  mdf = model_df.filter(world="big_m1_maze3_shortcut", eval=True)
  sub_df = get_shortcut_eval_data(
    user_df,
    tell_reuse=tell_reuse,
  )

  ##################
  # Create success rate and path reuse plots (median and mean)
  ##################
  for center in ["median", "mean"]:
    fig, ax, human_data = analysis_utils.plot_success_rate_path_reuse_metrics(
      df=sub_df,
      model_df=mdf,
      title="Generalization Success & Path Reuse",
      figsize=(6, 4),
      legend_loc="center left",
      overlap_threshold=overlap_threshold,
      center=center,
      mu=mu,
    )

    if save_figs:
      fig.savefig(
        os.path.join(save_dir, f"success_rate_path_reuse_{center}.pdf"),
        bbox_inches="tight",
      )

  if save_figs:
    analysis_utils.plot_human_rate_histograms(
      reuse_rates={"Human": human_data["human"]["reuse_rates"]},
      save_path=os.path.join(save_dir, "human_rate_distributions.pdf"),
    )

    analysis_utils.plot_reuse_bar(
      reuse_rates=human_data["human"]["reuse_rates"],
      save_path=os.path.join(save_dir, "reuse_bar.pdf"),
    )

  if display_figs:
    from IPython.display import display

    display(fig)


def start_results(
  user_df: DataFrame,
  save_dir: str = None,
  filter_columns: List[str] = None,
  display_figs: bool = False,
  tell_reuse: int = 1,
  save_figs: bool = True,
  verbosity: int = 0,
  ylim: Tuple[float, float] = None,
  median: bool = True,
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

  ##################
  # get all episodes for users who achieved at least 16 successes during training
  ##################
  sub_df, _ = filter_users_by_success(
    user_df.filter(manipulation="start", eval=True, tell_reuse=tell_reuse),
    analysis_name="start_results",
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
    ax=ax,
    ylim=ylim,
    use_median=median,
  )

  if save_figs:
    fig.savefig(
      os.path.join(save_dir, f"exp3_2_rt_diff_filter_{filter_str}.pdf"),
      bbox_inches="tight",
    )
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

  user_df, first_100_users = analysis_utils.filter_users_by_success_and_tell_reuse(
    user_df.filter(manipulation="juncture"),
    analysis_name="juncture_results",
  )

  ##################
  # Add setting column based on maze name
  ##################
  user_df = analysis_utils.get_polars_df(user_df)  # fancy merging will use regular df
  user_df = user_df.filter(manipulation="juncture")

  def get_maze_setting(maze_str: str) -> str:
    if "short" in maze_str.lower():
      return "short"
    elif "long" in maze_str.lower():
      return "long"
    raise ValueError(f"Could not determine setting from maze string: {maze_str}")

  # Add setting column based on maze name
  user_df = user_df.with_columns(
    setting=pl.col("world").map_elements(get_maze_setting, return_dtype=pl.String)
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
  all_pvalues = []

  options = options or [
    ("short", 1),
    ("short", 0),
    ("long", 1),
  ]

  # Collect data for each condition
  for idx, (setting, tell_reuse) in enumerate(options):
    difference_df = analysis_utils.compute_condition_difference_df(
      user_df.filter(setting=setting, tell_reuse=tell_reuse),
      measures=[measure],
    )

    # Get statistics for this condition
    results = analysis_utils.power_analysis_rt_differences(
      difference_df,
      measure,
      setting=str((idx, setting, tell_reuse)),
    )

    # Store data for plotting - always use median + bootstrapped CI
    all_diffs.append(difference_df[measure].to_numpy())
    all_means.append(results["median"])
    all_sems.append(results["median_ci"])  # Bootstrapped CI (asymmetric)
    all_labels.append(condition_labels[(setting, tell_reuse)])
    all_colors.append(condition_colors[(setting, tell_reuse)])
    all_pvalues.append(results["test"]["p_value"])

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

  shared_top = float(np.max(all_sems_array[:, 1]))
  y_min_cur, y_max_cur = ax.get_ylim()
  y_range_cur = y_max_cur - y_min_cur
  text_y = max(shared_top, y_max_cur) + 0.04 * y_range_cur
  for i, p_value in enumerate(all_pvalues):
    ax.text(
      x_pos[i],
      text_y,
      analysis_utils._p_value_to_text(p_value),
      ha="center",
      va="bottom",
      fontsize=14,
      color="black",
    )
  ax.set_ylim(y_min_cur, text_y + 0.12 * y_range_cur)

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


if __name__ == "__main__":
  import polars as pl

  user_df = pl.read_parquet(data_configs.get_dataframe_path("jaxmaze", "human"))
  model_df = data_configs.load_dataframes("jaxmaze")

  save_dir = (data_configs.JAXMAZE_RESULTS_DIR,)
  os.makedirs(save_dir, exist_ok=True)

  path_reuse_results(user_df, model_df, save_dir=save_dir)
  shortcut_results(user_df, model_df, save_dir=save_dir)
  start_results(user_df, model_df, save_dir=save_dir)
  juncture_results(user_df, model_df, save_dir=save_dir)
