"""
Functions for
(1) plotting experimental results
(2) doing power analysis
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_configs
from typing import List, Tuple, NamedTuple, Union

import yaml

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats
from datetime import datetime
from flax import struct
import seaborn as sns

from math import sqrt, ceil
from statsmodels.stats.power import TTestPower
import pandas as pd
import statsmodels.formula.api as smf
from multiprocessing import Pool
from scipy import stats

from nicewebrl.dataframe import DataFrame
import data_configs
from plot_configs import (
  default_colors,
  model_colors,
  model_names,
  model_order,
  measure_to_title,
  measure_to_ylabel,
)

DEFAULT_TITLE_SIZE = 15
DEFAULT_LABEL_SIZE = 15
DEFAULT_LEGEND_SIZE = 10.5

from tqdm.auto import tqdm


def get_polars_df(df: Union[DataFrame, pl.DataFrame]) -> pl.DataFrame:
  """Check if the dataframe is compatible with the analysis functions."""
  if isinstance(df, pl.DataFrame):
    return df
  elif isinstance(df, DataFrame):
    return df._df
  else:
    raise ValueError(f"Unsupported dataframe type: {type(df)}")


def get_base_filename(filepath):
  # Get filename without path
  filename = os.path.basename(filepath)
  # Remove extension
  base_filename = os.path.splitext(filename)[0]
  return base_filename


def add_to_file(stats_file, algo, label, text):
  # Get the base filename without extension
  base_filename = get_base_filename(stats_file.name)

  # Path to the YAML file
  yaml_file = data_configs.PAPER_STATS_MODEL_FILE

  # Load existing YAML if it exists
  data = {}
  if os.path.exists(yaml_file):
    with open(yaml_file, "r") as f:
      try:
        data = yaml.safe_load(f) or {}
      except yaml.YAMLError:
        # Handle case where file exists but is empty or invalid
        pass

  experiment = base_filename
  experiment_data = data.get(experiment, {})
  data[experiment] = experiment_data

  algo_data = experiment_data.get(algo, {})
  algo_data[label] = text
  experiment_data[algo] = algo_data

  algo_data[label] = text

  # Write back to YAML file
  with open(yaml_file, "w") as f:
    yaml.dump(data, f, default_flow_style=False)


######################################
# Plotting functions
######################################


def plot_reaction_times(
  reaction_times,
  figsize=(6, 3),
  color="lightblue",
  title=None,
  ylabel=True,
  ylim=None,
  remove_last: bool = True,
  show_xlabel=True,
  ax=None,
):
  """
  Creates a bar plot showing the progression of response times over steps.

  Args:
      reaction_times: numpy array of response times
      figsize: tuple specifying figure size (width, height)
      color: color for the plotted bars
      ylim: optional tuple (ymin, ymax) for setting y-axis limits

  Returns:
      fig, ax: matplotlib figure and axis objects
  """
  reaction_times = reaction_times[:-1] if remove_last else reaction_times
  if ax is None:
    fig, ax = plt.subplots(figsize=figsize)
  else:
    fig = ax.figure

  # Plot response times as bars
  steps = np.arange(len(reaction_times))
  ax.bar(steps, reaction_times, color=color, alpha=0.7, edgecolor="black")

  if show_xlabel:
    ax.set_xlabel("Episode timestep", fontsize=DEFAULT_LABEL_SIZE)
  if ylabel is not None:
    ax.set_ylabel("Response Time (s)", fontsize=DEFAULT_LABEL_SIZE)
  if title is not None:
    ax.set_title(title, fontsize=DEFAULT_TITLE_SIZE)
  ax.grid(True, axis="y", linestyle="--", alpha=0.7)

  # Add mean line
  mean_rt = np.mean(reaction_times)
  std_rt = np.std(reaction_times)
  ax.axhline(
    y=mean_rt,
    color="r",
    linestyle="--",
    label=f"Mean: {mean_rt:.3f}s, Std: {std_rt:.3f}s",
  )
  ax.legend(fontsize=DEFAULT_LEGEND_SIZE)

  # Adjust tick label sizes
  ax.tick_params(axis="both", which="major", labelsize=DEFAULT_LABEL_SIZE)

  # Set y-axis limits if provided
  if ylim is not None:
    ax.set_ylim(ylim)

  plt.tight_layout()
  return fig, ax


def bar_plot_error(
  human_data, model_stats=None, ax=None, ylim=None, legend=True, xlabels=False
):
  """Plot bar chart comparing human and model performance with error bars.

  Args:
      human_data: Dict containing human_means and human_se
      model_stats: DataFrame with columns 'algo', 'mean', 'se'
      ax: Optional matplotlib axis
      ylim: Optional y-axis limits tuple
  """
  if ax is None:
    fig, ax = plt.subplots(figsize=(8, 6))
  else:
    fig = ax.figure

  # Combine human and model data
  all_data = {"human": human_data["means"]}
  # Add model data
  yerr = [human_data["se"]]  # Start with human SE
  if model_stats is not None:
    algos = model_stats["algo"].unique().to_list()
    for algo in model_order:
      if not algo in algos:
        continue
      row = model_stats.filter(algo=algo)
      all_data[algo] = row["mean"].to_numpy()[0]
      yerr.append(row["se"].to_numpy()[0])

  # Plot bars
  x_pos = np.arange(len(all_data))
  ordered_keys = [k for k in model_order if k in all_data]
  bars = ax.bar(
    x_pos,
    [all_data[k] for k in ordered_keys],
    yerr=yerr,
    capsize=5,
    color=[model_colors.get(k, "#333333") for k in ordered_keys],
  )

  # Update x-tick labels to match new ordering
  if xlabels:
    ax.set_xticks(x_pos)
    ax.set_xticklabels([model_names[k] for k in ordered_keys], rotation=45, ha="right")
  else:
    ax.set_xticks([])

  # Add individual dots for human data with jitter
  if "raw" in human_data:
    x_jitter = np.random.normal(0, 0.15, size=len(human_data["raw"]))
    ax.scatter(
      [0 + j for j in x_jitter],
      human_data["raw"],
      color="black",
      alpha=0.5,
      zorder=3,
    )

  # Add chance level line
  ax.axhline(y=50, color="r", linestyle="--", alpha=0.5, label="Chance level")

  if ylim is not None:
    ax.set_ylim(ylim)

  if legend:
    legend_elements = bars
    legend_labels = [model_names[k] for k in ordered_keys]
    ax.legend(legend_elements, legend_labels, loc="lower right")

  ax.grid(True, linestyle="--", alpha=0.7)

  return fig, ax


def plot_bar_rt_comparison(
  df,
  rt_column,
  ax=None,
  title=None,
  ylabel=None,
  colors=None,
  reuse_column: str = "reuse",
  stats_file=None,
  n_simulations: int = 1000,
  rereun_analysis: bool = False,
  include_raw_data: bool = False,
  ylim: Tuple[float, float] = None,
  compute_n_for_desired_power: bool = False,
  use_median: bool = True,
  use_box_plot: bool = True,
):
  """Plot comparison of response times between multiple conditions.

  Args:
      episode_sets: List of DataFrames, each containing episodes for one condition
      rt_column: Name of response time column to analyze (e.g. 'first_rt', 'avg_rt')
      ax: Optional matplotlib axis to plot on
      display_stats: Whether to print statistical test results
      title: Custom title
      ylabel: Custom y-axis label
      xlabels: List of labels for x-axis ticks
      colors: List of colors for bars
      use_median: Whether to use median with confidence intervals instead of mean with standard error
      use_box_plot: Whether to use box plots instead of bar plots (only works with use_median=True)

  Returns:
      matplotlib axis
  """

  # Validate use_box_plot parameter
  if use_box_plot and not use_median:
    raise ValueError("Box plots can only be used when use_median=True")

  ylabel = ylabel or measure_to_ylabel[rt_column]
  title = title or measure_to_title[rt_column]
  len_before = len(df)
  df = df.filter(pl.col("success").is_not_null() & pl.col(reuse_column).is_not_null())
  len_after = len(df)
  print(f"Filtered {len_before - len_after} rows with null success or reuse")

  power_results = None
  if stats_file is not None:
    import os
    import pickle

    # Create a unique cache key based on the analysis parameters
    cache_key = f"{rt_column}_{reuse_column}_{n_simulations}"
    cache_path = f"temp/statsfile.{cache_key}.pkl"

    if os.path.exists(cache_path) and not rereun_analysis:
      print(f"Loading cached results from {cache_path}")
      try:
        with open(cache_path, "rb") as f:
          power_results = pickle.load(f)
      except Exception as e:
        print(f"Error loading cache: {e}")
        power_results = None

  # Run analysis if no cached results
  if power_results is None:
    if "rt" in rt_column:
      power_results = power_analysis_rt_across_groups(
        df,
        measure=rt_column,
        reuse_column=reuse_column,
        stats_file=stats_file,
        n_simulations=n_simulations,
        compute_n_for_desired_power=compute_n_for_desired_power,
      )
    elif "path_length" in rt_column:
      power_results = power_analysis_path_length_across_groups(
        df,
        measure=rt_column,
        reuse_column=reuse_column,
        stats_file=stats_file,
        n_simulations=n_simulations,
      )
    else:
      raise ValueError(f"Unknown rt_column: {rt_column}")

    # Save results to cache if cache_file is provided
    if stats_file is not None:
      import os
      import pickle

      # Create directory if it doesn't exist
      os.makedirs(
        os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".",
        exist_ok=True,
      )

      print(f"Saving results to {cache_path}")
      with open(cache_path, "wb") as f:
        pickle.dump(power_results, f)

  if use_median:
    # Use medians and confidence intervals
    values = (
      power_results["descriptive"]["medians"]["no_reuse"],
      power_results["descriptive"]["medians"]["reuse"],
    )

    # Calculate asymmetric error bars for confidence intervals
    ci_low_no_reuse = power_results["descriptive"]["median_cis"]["no_reuse"][0]
    ci_high_no_reuse = power_results["descriptive"]["median_cis"]["no_reuse"][1]
    ci_low_reuse = power_results["descriptive"]["median_cis"]["reuse"][0]
    ci_high_reuse = power_results["descriptive"]["median_cis"]["reuse"][1]

    yerr = np.array(
      [
        [values[0] - ci_low_no_reuse, values[1] - ci_low_reuse],  # lower errors
        [ci_high_no_reuse - values[0], ci_high_reuse - values[1]],  # upper errors
      ]
    )

  else:
    # Use means and standard errors (original behavior)
    values = (
      power_results["descriptive"]["means"]["no_reuse"],
      power_results["descriptive"]["means"]["reuse"],
    )
    yerr = (
      power_results["descriptive"]["ses"]["no_reuse"],
      power_results["descriptive"]["ses"]["reuse"],
    )

  # Create plot (bar or box)
  x_pos = np.arange(len(values))
  colors = [default_colors["nice purple"], default_colors["bluish green"]]

  if use_box_plot and use_median:
    # Create box plots instead of bar plots
    # Prepare data for boxplot - reshape to format boxplot expects
    boxplot_data = []
    for key in ["no_reuse", "reuse"]:
      boxplot_data.append(power_results["raw_means"][key])

    # Create box plot
    bp = ax.boxplot(
      boxplot_data,
      positions=x_pos,
      patch_artist=True,
      widths=0.6,
      showmeans=True,
      meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
    )

    # Customize box plot colors
    for i, box in enumerate(bp["boxes"]):
      box.set(facecolor=colors[i])

  elif use_median:
    # For median with asymmetric CIs
    ax.bar(x_pos, values, color=colors[: len(values)])
    # Add error bars separately
    ax.errorbar(x_pos, values, yerr=yerr, fmt="none", ecolor="black", capsize=5)
  else:
    # For mean with symmetric SEs (original behavior)
    ax.bar(x_pos, values, yerr=yerr, capsize=5, color=colors[: len(values)])

  # Add individual points with jitter
  all_data = []
  for i, key in enumerate(["no_reuse", "reuse"]):
    data = power_results["raw_means"][key]
    print(f"n {key}: {len(data)}")
    if include_raw_data:
      x_jitter = np.random.normal(i, 0.04, size=len(data))
      ax.scatter(x_jitter, data, alpha=0.3, color="black", s=20)
    all_data.append(data)

  # Customize plot
  ax.set_xticks(x_pos)
  xlabels = ["New Path", "Partial Reuse"]
  ax.set_xticklabels(xlabels, ha="center")

  # Update title to indicate whether using mean or median
  ax.set_ylabel(ylabel or f"Log {rt_column}", fontsize=DEFAULT_LABEL_SIZE)
  ax.set_title(title, fontsize=DEFAULT_TITLE_SIZE)
  ax.tick_params(axis="both", which="major", labelsize=DEFAULT_LABEL_SIZE)
  ax.grid(True, linestyle="--", alpha=0.7)

  if ylim is not None:
    y_min, y_max = ylim
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

  return ax


def plot_bar_rt_comparison_columns(
  df,
  rt_columns,
  ax=None,
  title=None,
  ylabel=None,
  xlabels=None,
  colors=None,
):
  """Plot comparison of response times between multiple columns in a DataFrame.

  Args:
      df: DataFrame containing response time data
      rt_columns: List of column names to analyze (e.g. ['first_rt', 'avg_rt'])
      ax: Optional matplotlib axis to plot on
      title: Custom title
      ylabel: Custom y-axis label
      xlabels: List of labels for x-axis ticks (defaults to column names)
      colors: List of colors for bars

  Returns:
      matplotlib axis
  """

  if colors is None:
    colors = [i for i in default_colors.values()]
    # Extend colors if needed
    while len(colors) < len(rt_columns):
      colors.extend(colors)

  if ax is None:
    _, ax = plt.subplots(figsize=(5, 4))

  # Calculate log RTs and group by user for each column
  all_data = []
  for col in rt_columns:
    log_col = f"log_{col}"
    stats = (
      df.with_columns((1000 * pl.col(col)).log().alias(log_col))
      .group_by("user_id")
      .agg(pl.col(log_col).mean())
    )
    all_data.append(stats[log_col].to_numpy())

  # Calculate means and standard errors
  means = [np.mean(data) for data in all_data]
  sems = [np.std(data) / np.sqrt(len(data)) for data in all_data]

  # Create bar plot
  x_pos = np.arange(len(rt_columns))
  bars = ax.bar(x_pos, means, yerr=sems, capsize=5, color=colors[: len(rt_columns)])

  # Add individual points with jitter
  for i, data in enumerate(all_data):
    x_jitter = np.random.normal(i, 0.04, size=len(data))
    ax.scatter(x_jitter, data, alpha=0.3, color="black", s=20)

  # Customize plot
  ax.set_xticks(x_pos)
  if xlabels:
    ax.set_xticklabels(xlabels, ha="center")
  else:
    ax.set_xticklabels(rt_columns, ha="center")
  ax.set_ylabel(ylabel or "Log Response Time", fontsize=DEFAULT_LABEL_SIZE)
  ax.set_title(title or "Response Time Comparison", fontsize=DEFAULT_TITLE_SIZE)
  ax.tick_params(axis="both", which="major", labelsize=DEFAULT_LABEL_SIZE)
  ax.grid(True, linestyle="--", alpha=0.7)

  return ax


def plot_rt_differences(
  difference_df: pl.DataFrame,
  measures: List[str],
  title: str,
  ax: plt.Axes = None,
  colors=None,
  ylabel="Log RT Difference (Cond2 - Cond1)",
  stats_file=None,
  xlabels=None,
  include_raw_data: bool = False,
  ylim: Tuple[float, float] = None,
  use_median: bool = True,
  use_box_plot: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
  """Plot RT differences between conditions.

  First compute a power analysis for each measure, seeing if its statistically significantly above 0.
  - note we have repeated measures per user, corresponding to the 'reversal column'.
  """
  # Calculate statistics for each measure
  means = []
  sems = []
  medians = []
  cis = []
  all_diffs = []
  for measure in measures:
    results = power_analysis_rt_differences(
      difference_df, measure, stats_file=stats_file
    )
    means.append(results["mean"])
    sems.append(results["se"])
    medians.append(results["median"])
    cis.append(results["median_ci"])

    # Get user-wide means instead of all individual data points
    user_means = (
      difference_df.group_by("user_id").agg(pl.col(measure).mean()).select(measure)
    )
    all_diffs.append(user_means.to_numpy().flatten())
    import pdb

    pdb.set_trace()

  # Create/get axis
  if ax is None:
    fig, ax = plt.subplots(figsize=(5, 4))
  else:
    fig = ax.figure

  # Create bar plot or box plot
  x_pos = np.arange(len(measures))

  if use_box_plot and use_median:
    # Create box plots instead of bar plots
    boxplot_data = all_diffs

    # Set box colors if provided
    box_colors = colors if colors else ["#1f77b4"] * len(measures)

    # Create box plot
    bp = ax.boxplot(
      boxplot_data,
      positions=x_pos,
      patch_artist=True,
      widths=0.6,
      showmeans=True,
      meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
    )

    # Customize box plot colors
    for i, box in enumerate(bp["boxes"]):
      box.set(facecolor=box_colors[i % len(box_colors)])

    # Add median lines in black for better visibility
    for i, median in enumerate(bp["medians"]):
      median.set(color="black", linewidth=1.5)

    # Add whiskers and caps in black
    for i, whisker in enumerate(bp["whiskers"]):
      whisker.set(color="black", linewidth=1)
    for i, cap in enumerate(bp["caps"]):
      cap.set(color="black", linewidth=1)

  elif use_median:
    # Use medians and CIs
    values = medians

    # Create bars
    bars = ax.bar(
      x_pos,
      values,
      color=colors,
    )

    # Add error bars separately for asymmetric CIs
    for i, (median, ci) in enumerate(zip(medians, cis)):
      lower_err = median - ci[0]
      upper_err = ci[1] - median
      ax.errorbar(
        i,
        median,
        yerr=np.array([[lower_err, upper_err]]).T,
        fmt="none",
        ecolor="black",
        # ecolor=default_colors.get("vermillion", "red"),
        capsize=5,
      )
  else:
    # Use means and standard errors (original behavior)
    values = means

    # Create bars with symmetric error bars
    bars = ax.bar(
      x_pos,
      values,
      yerr=sems,
      capsize=5,
      color=colors,
    )

  # Add individual points with jitter (now showing user means)
  if include_raw_data and not use_box_plot:
    for i, diffs in enumerate(all_diffs):
      x_jitter = np.random.normal(i, 0.125, size=len(diffs))
      ax.scatter(x_jitter, diffs, alpha=0.3, color="black", s=20)

  # Add zero line
  ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)

  # Customize plot
  ax.set_xticks(x_pos)
  ax.set_xticklabels(xlabels or measures, ha="center")
  ax.set_ylabel(ylabel, fontsize=DEFAULT_LABEL_SIZE)
  if title:
    ax.set_title(title, fontsize=DEFAULT_TITLE_SIZE)
  ax.tick_params(axis="both", which="major", labelsize=DEFAULT_LABEL_SIZE)
  ax.grid(True, linestyle="--", alpha=0.7)

  # Set y-axis limits based on all data points
  if ylim is None:
    all_data = np.concatenate(all_diffs)
    y_min, y_max = np.percentile(all_data, [1, 99])
  else:
    y_min, y_max = ylim
  y_range = y_max - y_min
  ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

  return fig, ax


def compute_condition_difference_df(
  df: pl.DataFrame, measures: List[str]
) -> pl.DataFrame:
  """Compute differences between conditions 1 and 2 for matched reversal conditions.

  Args:
      df: DataFrame containing columns [condition, reversal, setting, {measures}]
      setting: Which setting to filter for ('short' or 'long')
      measures: List of measure column names to compute differences for

  Returns:
      DataFrame with columns [user_id, reversal, diff_{measure1}, diff_{measure2}, ...]
  """

  # Get condition 1 and 2 data separately
  cond1_df = df.filter(condition=1)
  cond2_df = df.filter(condition=2)

  # Join the dataframes on user_id and reversal to match pairs
  diff_df = cond1_df.join(cond2_df, on=["user_id", "reversal"], suffix="_cond2")

  # Compute differences for each measure
  diff_exprs = [
    (pl.col(f"{measure}_cond2") - pl.col(measure)).alias(measure)
    for measure in measures
  ]

  # Select user_id, reversal and all difference columns
  diff_df = diff_df.select(["user_id", "reversal"] + diff_exprs)

  return diff_df


# def get_success_rate_path_reuse_data(
#  df: DataFrame,
#  overlap_threshold: float,
#  model_df: DataFrame = None,
#  reuse_column: str = "reuse",
#  stats_file: str = None,
# ) -> Tuple[DataFrame, DataFrame]:
#  # for both df and model_df, add a boolean "reuse" column if over "overlap" column is greater than overlap_threshold
#  df = df.with_columns((pl.col("overlap") > overlap_threshold).alias(reuse_column))
#  model_df = model_df.with_columns((pl.col("overlap") > overlap_threshold).alias(reuse_column))

#  # Calculate human statistics with consistent ordering
#  # drop all rows where reuse is -1
#  df = df.filter(pl.col(reuse_column) != -1)
#  model_df = model_df.filter(pl.col(reuse_column) != -1)

#  # Calculate human success statistics
#  human_success_stats = compute_binary_measure_statistics(df, "user_id", "success")

#  # Calculate human reuse statistics
#  human_reuse_stats = compute_binary_measure_statistics(df, "user_id", reuse_column)

#  # Add stats to file
#  if stats_file:
#    add_to_file(
#      stats_file,
#      algo="human data",
#      label="success",
#      text=human_success_stats["paper_result"],
#    )

#    add_to_file(
#      stats_file,
#      algo="human data",
#      label="reuse",
#      text=human_reuse_stats["paper_result"],
#    )

#  # Prepare data for plotting - choose metric based on normality
#  all_data = {
#    "human": {
#      # Choose values based on normality test
#      "success": 100
#      * (
#        human_success_stats["mean"]
#        if human_success_stats["normality"]["is_normal"]
#        else human_success_stats["median"]
#      ),
#      "reuse": 100
#      * (
#        human_reuse_stats["mean"]
#        if human_reuse_stats["normality"]["is_normal"]
#        else human_reuse_stats["median"]
#      ),
#      # Choose error metrics based on normality test
#      "success_error": 100
#      * (
#        human_success_stats["se"]
#        if human_success_stats["normality"]["is_normal"]
#        else np.array(human_success_stats["median_ci"])
#      ),
#      "reuse_error": 100
#      * (
#        human_reuse_stats["se"]
#        if human_reuse_stats["normality"]["is_normal"]
#        else np.array(human_reuse_stats["median_ci"])
#      ),
#      # Store normality info for reference
#      "success_is_normal": human_success_stats["normality"]["is_normal"],
#      "reuse_is_normal": human_reuse_stats["normality"]["is_normal"],
#    }
#  }

#  # Add model data if provided
#  if model_df is not None:
#    # Calculate model statistics
#    algos = model_df["algo"].unique().to_list()
#    for algo in model_order:
#      if algo not in algos:
#        continue

#      # Filter for this algorithm
#      algo_df = model_df.filter(algo=algo)

#      # Skip if no data for this algorithm
#      if len(algo_df) == 0:
#        continue

#      # Compute success statistics
#      success_stats = compute_binary_measure_statistics(algo_df, "seed", "success")

#      # Compute reuse statistics
#      reuse_stats = compute_binary_measure_statistics(algo_df, "seed", reuse_column)

#      # Add to stats file
#      if stats_file:
#        add_to_file(
#          stats_file,
#          algo=algo,
#          label="success",
#          text=success_stats["paper_result"],
#        )

#        add_to_file(
#          stats_file,
#          algo=algo,
#          label="reuse",
#          text=reuse_stats["paper_result"],
#        )

#      # Add to all_data dictionary for plotting
#      all_data[algo] = {
#        # Choose values based on normality test
#        "success": 100
#        * (
#          success_stats["mean"]
#          if success_stats["normality"]["is_normal"]
#          else success_stats["median"]
#        ),
#        "reuse": 100
#        * (
#          reuse_stats["mean"]
#          if reuse_stats["normality"]["is_normal"]
#          else reuse_stats["median"]
#        ),
#        # Choose error metrics based on normality test
#        "success_error": 100
#        * (
#          success_stats["se"]
#          if success_stats["normality"]["is_normal"]
#          else np.array(success_stats["median_ci"])
#        ),
#        "reuse_error": 100
#        * (
#          reuse_stats["se"]
#          if reuse_stats["normality"]["is_normal"]
#          else np.array(reuse_stats["median_ci"])
#        ),
#        # Store normality info for reference
#        "success_is_normal": success_stats["normality"]["is_normal"],
#        "reuse_is_normal": reuse_stats["normality"]["is_normal"],
#      }

#  return all_data


def get_human_success_rate_path_reuse_data(
  df: DataFrame,
  overlap_threshold: float,
  reuse_column: str = "reuse",
  stats_file: str = None,
) -> dict:
  # Add boolean "reuse" column based on overlap threshold
  df = df.with_columns((pl.col("overlap") > overlap_threshold).cast(pl.Float64).alias(reuse_column))

  # Filter rows where reuse is -1
  df = df.filter(pl.col(reuse_column) != -1)

  # Calculate human success statistics
  human_success_stats = compute_binary_measure_statistics(df, "user_id", "success")

  # Calculate human reuse statistics
  human_reuse_stats = compute_binary_measure_statistics(df, "user_id", reuse_column)

  # Add stats to file
  if stats_file:
    add_to_file(
      stats_file,
      algo="human data",
      label="success",
      text=human_success_stats["paper_result"],
    )

    add_to_file(
      stats_file,
      algo="human data",
      label="reuse",
      text=human_reuse_stats["paper_result"],
    )

  # Prepare human data for plotting
  return {
    "human": {
      # Choose values based on normality test
      "success": 100
      * (
        human_success_stats["mean"]
        if human_success_stats["normality"]["is_normal"]
        else human_success_stats["median"]
      ),
      "reuse": 100
      * (
        human_reuse_stats["mean"]
        if human_reuse_stats["normality"]["is_normal"]
        else human_reuse_stats["median"]
      ),
      # Choose error metrics based on normality test
      "success_error": 100
      * (
        human_success_stats["se"]
        if human_success_stats["normality"]["is_normal"]
        else np.array(human_success_stats["median_ci"])
      ),
      "reuse_error": 100
      * (
        human_reuse_stats["se"]
        if human_reuse_stats["normality"]["is_normal"]
        else np.array(human_reuse_stats["median_ci"])
      ),
      # Store normality info for reference
      "success_is_normal": human_success_stats["normality"]["is_normal"],
      "reuse_is_normal": human_reuse_stats["normality"]["is_normal"],
    }
  }


def get_model_success_rate_path_reuse_data(
  model_df: DataFrame,
  overlap_threshold: float,
  reuse_column: str = "reuse",
  stats_file: str = None,
) -> dict:
  if model_df is None:
    return {}

  # Add boolean "reuse" column based on overlap threshold
  model_df = model_df.with_columns(
    (pl.col("overlap") > overlap_threshold).alias(reuse_column)
  )

  # Filter rows where reuse is -1
  model_df = model_df.filter(pl.col(reuse_column) != -1)

  model_data = {}

  # Calculate model statistics
  algos = model_df["algo"].unique().to_list()
  for algo in model_order:
    if algo not in algos:
      continue

    # Filter for this algorithm
    algo_df = model_df.filter(algo=algo)

    # Skip if no data for this algorithm
    if len(algo_df) == 0:
      continue

    # Compute success statistics
    success_stats = compute_binary_measure_statistics(algo_df, "seed", "success")

    # Compute reuse statistics
    reuse_stats = compute_binary_measure_statistics(algo_df, "seed", reuse_column)

    # Add to stats file
    if stats_file:
      add_to_file(
        stats_file,
        algo=algo,
        label="success",
        text=success_stats["paper_result"],
      )

      add_to_file(
        stats_file,
        algo=algo,
        label="reuse",
        text=reuse_stats["paper_result"],
      )

    # Add to model_data dictionary for plotting
    model_data[algo] = {
      # Choose values based on normality test
      "success": 100
      * (
        success_stats["mean"]
        if success_stats["normality"]["is_normal"]
        else success_stats["median"]
      ),
      "reuse": 100
      * (
        reuse_stats["mean"]
        if reuse_stats["normality"]["is_normal"]
        else reuse_stats["median"]
      ),
      # Choose error metrics based on normality test
      "success_error": 100
      * (
        success_stats["se"]
        if success_stats["normality"]["is_normal"]
        else np.array(success_stats["median_ci"])
      ),
      "reuse_error": 100
      * (
        reuse_stats["se"]
        if reuse_stats["normality"]["is_normal"]
        else np.array(reuse_stats["median_ci"])
      ),
      # Store normality info for reference
      "success_is_normal": success_stats["normality"]["is_normal"],
      "reuse_is_normal": reuse_stats["normality"]["is_normal"],
    }

  return model_data

def get_center(data, key):
  # Format error bars based on normality
  if data[f"{key}_is_normal"]:
    # For normal data, use symmetric standard error
    return data[f"{key}_mean"]
  else:
    return data[f"{key}_median"]

def get_err(data, key):
  # Format error bars based on normality
  if data[f"{key}_is_normal"]:
    # For normal data, use symmetric standard error
    err = data[f"{key}_error"]
  else:
    # For non-normal data, use asymmetric confidence intervals
    reuse_ci = data[f"{key}_error"]
    # Check if the CI is properly formatted as [lower, upper]
    if isinstance(reuse_ci, np.ndarray) and reuse_ci.size == 2:
      lower_reuse_ci = data[key] - reuse_ci[0]
      upper_reuse_ci = reuse_ci[1] - data[key]
      err = np.array([[lower_reuse_ci, upper_reuse_ci]]).T
    else:
      # Fallback to symmetric error if CI format is unexpected
      err = reuse_ci
  return err

def plot_success_rate_path_reuse_metrics(
  df: DataFrame,
  model_df: DataFrame = None,
  stats_file=None,
  ax=None,
  reuse_column: str = "reuse",
  path_deviance_column: str = None,
  title="Success Rate and Path Reuse",
  figsize=(8, 8),  # Changed to square figure for better 2D visualization
  include_raw_data: bool = True,
  min_circle_size: int = 10,
  max_circle_size: int = 100,
  legend_loc: str = "lower right",
  legend_ncol: int = 1,
  overlap_threshold: float = 0.7,
  point_size=100,
) -> Tuple[plt.Figure, plt.Axes]:
  """Plot success rate vs path reuse as a 2D scatter plot with error bars.

  Args:
      df (DataFrame): DataFrame containing human data
      model_df (DataFrame, optional): DataFrame containing model data. If None, only plots human data.
      stats_file (file, optional): File to write statistics to
      ax (plt.Axes, optional): Matplotlib axes to plot on. If None, creates new figure
      reuse_column (str, optional): Column name for path reuse metric
      path_deviance_column (str, optional): Column name for path length deviance
      title (str, optional): Plot title
      figsize (tuple, optional): Figure size if creating new figure
      include_raw_data (bool, optional): Whether to include individual human data points
      min_circle_size (int, optional): Minimum size for circles
      max_circle_size (int, optional): Maximum size for circles

  Returns:
      tuple: (fig, ax) containing the figure and axes object
  """

  # Create figure if needed
  if ax is None:
    fig, ax = plt.subplots(figsize=figsize)
  else:
    fig = ax.figure

  human_data = get_human_success_rate_path_reuse_data(
    df=df,
    overlap_threshold=overlap_threshold,
    reuse_column=reuse_column,
    stats_file=stats_file,
  )

  model_data = get_model_success_rate_path_reuse_data(
    model_df=model_df,
    overlap_threshold=overlap_threshold,
    reuse_column=reuse_column,
    stats_file=stats_file,
  )

  all_data = {**human_data, **model_data}

  # Plot data points with error bars
  ordered_keys = [k for k in model_order if k in all_data]
  marker_size = point_size  # Default size of the main scatter points

  # Plot each model/human data point with error bars
  for key in ordered_keys:
    data = all_data[key]

    # Format error bars based on normality
    if data["reuse_is_normal"]:
      # For normal data, use symmetric standard error
      xerr = data["reuse_error"]
    else:
      # For non-normal data, use asymmetric confidence intervals
      reuse_ci = data["reuse_error"]
      # Check if the CI is properly formatted as [lower, upper]
      if isinstance(reuse_ci, np.ndarray) and reuse_ci.size == 2:
        lower_reuse_ci = data["reuse"] - reuse_ci[0]
        upper_reuse_ci = reuse_ci[1] - data["reuse"]
        xerr = np.array([[lower_reuse_ci, upper_reuse_ci]]).T
      else:
        # Fallback to symmetric error if CI format is unexpected
        xerr = reuse_ci

    # Similarly for success
    if data["success_is_normal"]:
      # For normal data, use symmetric standard error
      yerr = data["success_error"]
    else:
      # For non-normal data, use asymmetric confidence intervals
      success_ci = data["success_error"]
      # Check if the CI is properly formatted as [lower, upper]
      if isinstance(success_ci, np.ndarray) and success_ci.size == 2:
        lower_success_ci = data["success"] - success_ci[0]
        upper_success_ci = success_ci[1] - data["success"]
        yerr = np.array([[lower_success_ci, upper_success_ci]]).T
      else:
        # Fallback to symmetric error if CI format is unexpected
        yerr = success_ci

    ax.errorbar(
      data["reuse"],
      data["success"],
      xerr=xerr,
      yerr=yerr,
      fmt="none",
      color=model_colors.get(key, "#333333"),
      capsize=5,
      capthick=2,
      elinewidth=2,
      zorder=2,
    )

    ax.scatter(
      data["reuse"],
      data["success"],
      color=model_colors.get(key, "#333333"),
      s=marker_size,
      label=model_names[key],
      zorder=3,
    )

  # Customize axes
  ax.set_xlabel("Path Reuse (%)", fontsize=DEFAULT_LABEL_SIZE)
  ax.set_ylabel("Success Rate (%)", fontsize=DEFAULT_LABEL_SIZE)
  ax.set_title(title, fontsize=DEFAULT_TITLE_SIZE)

  # Add chance level line for success rate
  ax.axhline(y=50, color="r", linestyle="--", alpha=0.5, label="Chance level")
  ax.axvline(x=50, color="r", linestyle="--", alpha=0.5)

  # Set axis limits with some padding
  ax.set_xlim(-5, 105)
  ax.set_ylim(-5, 105)

  ax.set_xticks(range(0, int(ax.get_xlim()[1]) + 1, 10))
  ax.set_yticks(range(0, int(ax.get_ylim()[1]) + 1, 10))
  # Add grid
  ax.grid(True, linestyle="--", alpha=0.7)

  # Add legend
  if model_df is not None:
    ax.legend(
      loc=legend_loc,
      ncol=legend_ncol,
      columnspacing=1,
      handletextpad=0.5,
      fontsize=DEFAULT_LEGEND_SIZE,
    )

  return fig, ax


######################################
# Power Analysis function
######################################


# define functions to run power analysis for linear mixed effects model (LME)
def simulate_mixed_effects_trial(args):
  """Simulate a single mixed effects trial and test for significance.

  Args:
      args: Tuple containing:
          - num_subjects: Number of subjects in simulation
          - trials_per_subject: Number of trials per subject
          - B0: Intercept coefficient
          - B1: Slope coefficient
          - random_effect_var: Variance of random effects
          - residual_var: Variance of residuals
          - alpha: Significance level for test

  Returns:
      bool: True if result is significant at alpha level
  """
  # Unpack arguments
  num_subjects, trials_per_subject, B0, B1, random_effect_var, residual_var, alpha = (
    args
  )

  # Generate data
  user_ids = np.repeat(range(num_subjects), trials_per_subject)
  reuse = np.random.choice([0, 1], num_subjects * trials_per_subject)
  random_intercepts = np.random.normal(0, np.sqrt(random_effect_var), num_subjects)
  random_intercepts = np.repeat(random_intercepts, trials_per_subject)
  residuals = np.random.normal(
    0, np.sqrt(residual_var), num_subjects * trials_per_subject
  )

  # Generate response variable
  RT = B0 + B1 * reuse + random_intercepts + residuals

  # Create and test model
  data = pd.DataFrame({"RT": RT, "reuse": reuse, "user_id": user_ids})
  model = smf.mixedlm("RT ~ reuse", data, groups=data["user_id"])
  result = model.fit(reml=True)

  return result.pvalues["reuse"] < alpha


def mixed_effects_compute_power(
  num_subjects: int,
  trials_per_subject: int,
  B0: float,
  B1: float,
  random_effect_var: float,
  residual_var: float,
  n_simulations: int = 500,
  alpha: float = 0.05,
  parallel: bool = False,
  n_jobs: int = -1,
  verbose: bool = False,
) -> float:
  """Compute power for mixed effects model using parallel or sequential processing.

  Args:
      num_subjects: Number of subjects in each simulation
      trials_per_subject: Number of trials per subject
      B0: Intercept coefficient
      B1: Slope coefficient
      random_effect_var: Variance of random effects
      residual_var: Variance of residuals
      n_simulations: Number of simulations to run (default: 500)
      alpha: Significance level (default: 0.05)
      parallel: Whether to use parallel processing (default: True)
      n_jobs: Number of processes to use if parallel (-1 for all cores, default: -1)
      **kwargs: Additional arguments to pass to simulate_mixed_effects_trial

  Returns:
      float: Computed power (proportion of significant results)
  """
  # Prepare simulation parameters
  sim_args = [
    (
      num_subjects,
      trials_per_subject,
      B0,
      B1,
      random_effect_var,
      residual_var,
      alpha,
    )
  ] * n_simulations

  if parallel:
    # Use all available cores if n_jobs is -1
    n_jobs = None if n_jobs == -1 else n_jobs

    # Run simulations in parallel
    with Pool(processes=n_jobs) as pool:
      results = list(
        tqdm(
          pool.imap(simulate_mixed_effects_trial, sim_args),
          total=n_simulations,
          desc="Simulating data",
          disable=not verbose,
        )
      )
  else:
    # Run simulations sequentially
    results = [
      simulate_mixed_effects_trial(args)
      for args in tqdm(sim_args, desc="Simulating data")
    ]

  return sum(results) / n_simulations


def compute_binary_measure_statistics(
  df: pl.DataFrame,
  group_col: str,
  measure: str,
  mu: float = 0.5,
  alpha: float = 0.05,
  confidence=0.95,
):
  """Calculate statistics for binary proportion data with appropriate normality testing.

  Args:
      df: DataFrame containing the measure column and grouping column
      group_col: Column to group by (e.g. 'user_id' or 'seed')
      measure: Column containing binary measure (0/1 values)
      mu: null hypothesis value (default: 0.5)
      alpha: significance level for tests (default: 0.05)

  Returns:
      dict containing mean, median, standard error, confidence intervals, and test results
  """
  # First aggregate by group to get their mean rate
  group_means = df.group_by(group_col).agg(
    rate=pl.col(measure).mean(), n_trials=pl.col(measure).count()
  )

  rates = group_means["rate"].to_numpy()
  n_groups = len(rates)

  # Test normality using Shapiro-Wilk test
  _, normality_p = stats.shapiro(rates)
  is_normal = normality_p > alpha

  # Calculate median of the group means
  p_median = group_means["rate"].median()

  # Standard deviation of the group means
  sd = group_means["rate"].std()


  # Mean of group means (weighted by n_trials)
  total_trials = group_means["n_trials"].sum()

  p_obs = (group_means["rate"] * group_means["n_trials"]).sum() / total_trials
  se = np.sqrt(p_obs * (1 - p_obs) / total_trials)

  alpha = 0.05
  if is_normal:
      # For normal data: use parametric approach with mean
      se = np.sqrt(p_obs * (1 - p_obs) / total_trials)
      ci_low = p_obs - stats.norm.ppf(1-alpha/2) * se
      ci_high = p_obs + stats.norm.ppf(1-alpha/2) * se
  else:
      # For non-normal data: use bootstrap for median CI
      bootstrap_samples = 10000
      bootstrap_means = []
      n_trials = group_means["n_trials"].to_numpy()
      for _ in range(bootstrap_samples):
          # Sample indices with replacement
          idx = np.random.choice(n_groups, size=n_groups, replace=True)
          # Calculate weighted mean using the sampled rates and their corresponding n_trials
          sampled_rates = rates[idx]
          sampled_trials = n_trials[idx]
          weighted_mean = np.sum(sampled_rates * sampled_trials) / np.sum(sampled_trials)
          bootstrap_means.append(weighted_mean)

      # Percentile method for confidence intervals
      ci_low = np.percentile(bootstrap_means, alpha/2 * 100)
      ci_high = np.percentile(bootstrap_means, (1-alpha/2) * 100)
      if ci_high < p_median:
        ci_high = p_median
      if ci_low > p_median:
        ci_low = p_median

  ci_low = max(0.0, ci_low)  # Lower bound can't be below 0%
  ci_high = min(1.0, ci_high)  # Upper bound can't exceed 100%


  if is_normal:
    # One-sided t-test
    t_stat, p_value = stats.ttest_1samp(rates, mu)
    # Convert to one-sided p-value if t-statistic is in predicted direction
    p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2

    # Calculate Cohen's d effect size
    d = (p_obs - mu) / np.std(rates, ddof=1)
    effect_size = {"name": "Cohen's d", "value": d}

    test_name = "One-sample t-test"
    test_stat = t_stat
    df = n_groups - 1  # Degrees of freedom for one-sample t-test

  else:
    # One-sided Wilcoxon signed-rank test
    w_stat, p_value = stats.wilcoxon(rates - mu, alternative="greater")

    # Calculate r effect size (correlation coefficient) for Wilcoxon test
    z = stats.norm.ppf(1 - p_value)  # Convert p-value to Z score
    r = z / np.sqrt(n_groups)  # Standardize by sample size
    effect_size = {"name": "r", "value": r}

    test_name = "Wilcoxon signed-rank test"
    test_stat = w_stat
    df = n_groups - 1  # Using n-1 for consistency, though Wilcoxon doesn't use df

  # Prepare paper result text
  paper_result = f"Mean={100 * p_obs:.2f}%, Median={100 * p_median:.2f}% [95% CI: {100 * ci_low:.2f}%, {100 * ci_high:.2f}%], t({df})={test_stat:.2f}, p={p_value:.2g}"

  return {
    "n_groups": n_groups,
    "mean": p_obs,
    "median": p_median,
    "median_ci": (ci_low, ci_high),
    "se": se,
    "sd": sd,
    "normality": {"is_normal": is_normal, "p_value": normality_p},
    "test": {"name": test_name, "statistic": test_stat, "p_value": p_value, "df": df},
    "effect_size": effect_size,
    "paper_result": paper_result,
  }


def power_analysis_path_reuse(
  df: pl.DataFrame,
  measure: str = "reuse",
  mu: float = 0.5,
  alpha: float = 0.05,
  plot: bool = False,
  stats_file=None,
):
  """Analyze binary proportion data using appropriate statistical test based on normality.

  Args:
      df: DataFrame containing reuse column (binary 0/1) and user_id
      mu: null hypothesis value (default: 0.5)
      alpha: significance level for tests (default: 0.05)
      plot: whether to show diagnostic plots (default: False)
      stats_file: optional file handle to write stats output
  """
  # Use the new statistics function
  results = compute_binary_measure_statistics(df, "user_id", measure, mu, alpha)

  # Print summary
  if stats_file:
    stats_file.write(results["paper_result"] + "\n")
    add_to_file(
      stats_file, algo="human data", label=measure, text=results["paper_result"]
    )

    # Add W value if it's a Wilcoxon test
    if results["test"]["name"] == "Wilcoxon signed-rank test":
      w_stat = results["test"]["statistic"]
      p_value = results["test"]["p_value"]
      add_to_file(
        stats_file,
        algo="human data",
        label=f"{measure} w-p values",
        text=f"W-value={w_stat:.3f}, p={p_value:.2g}",
      )

  if plot:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram with density plot
    rates = df.group_by("user_id").agg(rate=pl.col(measure).mean())
    sns.histplot(data=rates, x="rate", kde=True, ax=ax1)
    ax1.axvline(mu, color="r", linestyle="--", label=f"Null (μ={mu})")
    ax1.set_title("Distribution of User Rates")
    ax1.set_xlabel(measure.capitalize() + " Rate")
    ax1.legend()

    # Q-Q plot
    stats.probplot(rates["rate"].to_numpy(), dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot")

    plt.tight_layout()
    plt.show()

  # Add additional calculations for power analysis that were in the original function
  # Calculate required sample sizes for different power levels
  power_levels = [0.8, 0.9, 0.95]
  n_required = {}
  effect = results["effect_size"]["value"]
  is_normal = results["normality"]["is_normal"]

  if is_normal:
    # For t-test
    for power in power_levels:
      analysis = TTestPower()
      n_required[power] = analysis.solve_power(
        effect_size=effect, alpha=alpha, power=power, alternative="larger"
      )
      n_required[power] = ceil(n_required[power])
  else:
    # For Wilcoxon test (based on asymptotic relative efficiency)
    # Convert r to d using formula: d = 2r/sqrt(1-r^2)
    if abs(effect) < 1:
      d = 2 * effect / sqrt(1 - effect**2)
    else:
      d = float("inf")

    for power in power_levels:
      analysis = TTestPower()
      n_t = analysis.solve_power(
        effect_size=d, alpha=alpha, power=power, alternative="larger"
      )
      # Adjust for Wilcoxon efficiency
      n_required[power] = ceil(n_t / 0.95)

  return {
    "n_users": results["n_groups"],
    "mean_trials": np.mean(
      df.group_by("user_id").agg(pl.col(measure).count())["count"].to_numpy()
    ),
    "reuse_rates": df.group_by("user_id")
    .agg(pl.col(measure).mean())["mean"]
    .to_numpy(),
    "mean": results["mean"],
    "median": results["median"],
    "median_ci": np.array(results["median_ci"]),
    "se": results["se"],
    "normality": results["normality"],
    "test": results["test"],
    "effect_size": results["effect_size"],
  }


def power_analysis_rt_across_groups(
  df: pl.DataFrame,
  measure: str,
  reuse_column: str = "reuse",
  alpha=0.05,
  stats_file=None,
  n_simulations=500,
  power_levels=[0.8],
  compute_n_for_desired_power: bool = False,
):
  """Perform power analysis for between-groups comparison using linear mixed effects model.

  Args:
      df: DataFrame with columns [user_id, reuse, rt] where:
          - user_id: identifier for each participant
          - reuse: boolean indicating condition
          - rt: response time measurement (in log seconds)
      alpha: Significance level (default: 0.05)
      stats_file: Optional file handle to write stats output

  Returns:
      dict containing analysis results
  """
  # Convert to pandas for statsmodels compatibility
  data = get_polars_df(df).select(["user_id", reuse_column, measure]).to_pandas()
  data.columns = ["user_id", "reuse", "RT"]

  # Calculate descriptive statistics by group
  user_stats = df.group_by(["user_id", reuse_column]).agg(
    mean_val=pl.col(measure).mean()
  )

  reuse_means = user_stats.filter(pl.col(reuse_column) == True)["mean_val"].to_numpy()
  no_reuse_means = user_stats.filter(pl.col(reuse_column) == False)[
    "mean_val"
  ].to_numpy()

  n1, n2 = len(no_reuse_means), len(reuse_means)
  mean1, mean2 = np.mean(no_reuse_means), np.mean(reuse_means)
  var1, var2 = np.var(no_reuse_means, ddof=1), np.var(reuse_means, ddof=1)

  # Calculate medians and bootstrap 95% CIs
  median1, median2 = np.median(no_reuse_means), np.median(reuse_means)

  # Bootstrap 95% confidence intervals for medians
  n_bootstrap = 10000
  bootstrap_medians1, bootstrap_medians2 = [], []

  for _ in range(n_bootstrap):
    bootstrap_sample1 = np.random.choice(no_reuse_means, size=n1, replace=True)
    bootstrap_sample2 = np.random.choice(reuse_means, size=n2, replace=True)
    bootstrap_medians1.append(np.median(bootstrap_sample1))
    bootstrap_medians2.append(np.median(bootstrap_sample2))

  ci_low1, ci_high1 = np.percentile(bootstrap_medians1, [2.5, 97.5])
  ci_low2, ci_high2 = np.percentile(bootstrap_medians2, [2.5, 97.5])

  if stats_file is None:
    return {
      "descriptive": {
        "means": {"no_reuse": mean1, "reuse": mean2},
        "medians": {"no_reuse": median1, "reuse": median2},
        "median_cis": {"no_reuse": (ci_low1, ci_high1), "reuse": (ci_low2, ci_high2)},
        "sds": {"no_reuse": np.sqrt(var1), "reuse": np.sqrt(var2)},
        "ses": {"no_reuse": np.sqrt(var1 / n1), "reuse": np.sqrt(var2 / n2)},
      },
      "raw_means": {"no_reuse": no_reuse_means, "reuse": reuse_means},
    }

  # Fit linear mixed effects model
  model = smf.mixedlm("RT ~ reuse", data, groups=data["user_id"])
  result = model.fit(reml=True)

  # Get effect size (standardized coefficient)
  def get_param_name(result):
    if "reuse[T.True]" in result.params:
      return "reuse[T.True]"
    elif "reuse[T.1]" in result.params:
      return "reuse[T.1]"
    elif "reuse" in result.params:
      return "reuse"
    else:
      raise ValueError(f"No reuse parameter found in result: {result.params}")

  param_name = get_param_name(result)
  effect_size = result.params[param_name] / np.std(data["RT"])

  # Calculate required sample sizes for different power levels
  n_required = {}

  # Binary search for each power level
  if compute_n_for_desired_power:
    for target_power in tqdm(power_levels, desc="Power levels"):
      left = 10  # minimum sample size
      right = 500  # maximum sample size to try

      while left < right:
        n = (left + right) // 2
        power = mixed_effects_compute_power(
          num_subjects=n,
          trials_per_subject=len(data) // len(data["user_id"].unique()),
          B0=result.params["Intercept"],
          B1=result.params[param_name],
          random_effect_var=result.cov_re.iloc[0, 0],
          residual_var=result.scale,
          n_simulations=n_simulations,
        )

        if abs(power - target_power) < 0.01:  # within 1% of target
          break
        elif power < target_power:
          left = n + 1
        else:
          right = n - 1

      n_required[target_power] = n

  # Calculate actual power with current sample size
  current_power = mixed_effects_compute_power(
    num_subjects=len(data["user_id"].unique()),
    trials_per_subject=len(data) // len(data["user_id"].unique()),
    B0=result.params["Intercept"],
    B1=result.params[param_name],
    random_effect_var=result.cov_re.iloc[0, 0],
    residual_var=result.scale,
    n_simulations=n_simulations,
  )

  results = {
    "effect_size": {"name": "Standardized coefficient", "value": effect_size},
    "n_required": n_required,
    "current_power": current_power,
    "B0": result.params["Intercept"],
    "B1": result.params[param_name],
    "test_results": {
      "name": "Linear mixed effects model",
      "statistic": result.tvalues[param_name],
      "p_value": result.pvalues[param_name],
      "n1": n1,
      "n2": n2,
    },
    "descriptive": {
      "means": {"no_reuse": mean1, "reuse": mean2},
      "medians": {"no_reuse": median1, "reuse": median2},
      "median_cis": {"no_reuse": (ci_low1, ci_high1), "reuse": (ci_low2, ci_high2)},
      "sds": {"no_reuse": np.sqrt(var1), "reuse": np.sqrt(var2)},
      "ses": {"no_reuse": np.sqrt(var1 / n1), "reuse": np.sqrt(var2 / n2)},
    },
    "raw_means": {"no_reuse": no_reuse_means, "reuse": reuse_means},
  }

  param_name = get_param_name(result)
  se = result.bse[param_name]
  ci = result.conf_int(alpha=0.05).loc[param_name]
  b1 = result.params[param_name]
  t_val = result.tvalues[param_name]
  p_val = result.pvalues[param_name]
  df = result.df_resid
  paper_result = f"β = {b1:.2f}, SE = {se:.2f}, t({df}) = {t_val:.2f}, p = {p_val:.2g}, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]"
  add_to_file(stats_file, algo="human data", label="rt_difference", text=paper_result)

  # Add medians and CIs to the stats file
  add_to_file(
    stats_file,
    algo="human data",
    label="no_reuse_median",
    text=f"Median = {median1:.2f} [95% CI: {ci_low1:.2f}, {ci_high1:.2f}]",
  )
  add_to_file(
    stats_file,
    algo="human data",
    label="reuse_median",
    text=f"Median = {median2:.2f} [95% CI: {ci_low2:.2f}, {ci_high2:.2f}]",
  )

  if stats_file:
    stats_file.write("\nLinear Mixed Effects Model Results:\n")
    stats_file.write("================================\n")
    stats_file.write(str(result.summary()) + "\n\n")

    stats_file.write("Group Medians with 95% CIs:\n")
    stats_file.write(
      f"No Reuse: {median1:.3f} [95% CI: {ci_low1:.3f}, {ci_high1:.3f}]\n"
    )
    stats_file.write(
      f"Reuse: {median2:.3f} [95% CI: {ci_low2:.3f}, {ci_high2:.3f}]\n\n"
    )

    stats_file.write("<<<<<FOR PAPER>>>>>\n")
    stats_file.write(f"{paper_result}\n")
    stats_file.write(f"Current power: {current_power:.3f}\n")
    for power, n in n_required.items():
      stats_file.write(f"Required sample size for {power * 100}% power: {n}\n")

  return results


def power_analysis_path_length_across_groups(
  df: pl.DataFrame,
  measure: str,
  alpha: float = 0.05,
  stats_file=None,
  reuse_column: str = "reuse",
  n_simulations: int = 500,
):
  """Perform power analysis for between-groups comparison of path lengths using linear mixed effects model.

  Args:
      df: DataFrame with columns [user_id, reuse, path_length] where:
          - user_id: identifier for each participant
          - reuse: boolean indicating condition
          - path_length: length of path taken
      alpha: Significance level (default: 0.05)
      stats_file: Optional file handle to write stats output
      n_simulations: Number of simulations for power analysis

  Returns:
      dict containing analysis results
  """
  # Convert to pandas for statsmodels compatibility
  data = get_polars_df(df).select(["user_id", reuse_column, measure]).to_pandas()
  data.columns = ["user_id", "reuse", "path_length"]

  # Calculate descriptive statistics by group
  user_stats = df.group_by(["user_id", "reuse"]).agg(
    mean_val=pl.col(measure).mean(),
    median_val=pl.col(measure).median(),
    std_val=pl.col(measure).std(),
  )

  reuse_stats = user_stats.filter(pl.col("reuse") == True)
  no_reuse_stats = user_stats.filter(pl.col("reuse") == False)

  n1, n2 = len(no_reuse_stats), len(reuse_stats)
  mean1, mean2 = no_reuse_stats["mean_val"].mean(), reuse_stats["mean_val"].mean()
  median1, median2 = (
    no_reuse_stats["median_val"].median(),
    reuse_stats["median_val"].median(),
  )
  var1, var2 = (
    np.var(no_reuse_stats["mean_val"].to_numpy(), ddof=1),
    np.var(reuse_stats["mean_val"].to_numpy(), ddof=1),
  )

  if stats_file is None:
    return {
      "descriptive": {
        "means": {"no_reuse": mean1, "reuse": mean2},
        "medians": {"no_reuse": median1, "reuse": median2},
        "sds": {"no_reuse": np.sqrt(var1), "reuse": np.sqrt(var2)},
        "ses": {"no_reuse": np.sqrt(var1 / n1), "reuse": np.sqrt(var2 / n2)},
      },
      "raw_means": {
        "no_reuse": no_reuse_stats["mean_val"].to_numpy(),
        "reuse": reuse_stats["mean_val"].to_numpy(),
      },
    }

  # Fit linear mixed effects model
  model = smf.mixedlm("path_length ~ reuse", data, groups=data["user_id"])
  result = model.fit(reml=True)

  # Get effect size (standardized coefficient)
  param_name = "reuse[T.True]" if "reuse[T.True]" in result.params else "reuse"
  effect_size = result.params[param_name] / np.std(data["path_length"])

  # Calculate required sample sizes for different power levels
  power_levels = [0.8, 0.9, 0.95]
  n_required = {}

  # Binary search for each power level
  for target_power in tqdm(power_levels, desc="Power levels"):
    left = 10  # minimum sample size
    right = 200  # maximum sample size to try

    while left < right:
      n = (left + right) // 2
      # Use gamma distribution for simulation to better match path length distribution
      power = mixed_effects_compute_power_gamma(
        num_subjects=n,
        trials_per_subject=len(data) // len(data["user_id"].unique()),
        B0=result.params["Intercept"],
        B1=result.params[param_name],
        random_effect_var=result.cov_re.iloc[0, 0],
        shape=np.mean(data["path_length"]) ** 2 / np.var(data["path_length"]),
        n_simulations=n_simulations,
      )

      if abs(power - target_power) < 0.01:  # within 1% of target
        break
      elif power < target_power:
        left = n + 1
      else:
        right = n - 1

    n_required[target_power] = n

  # Calculate actual power with current sample size
  current_power = mixed_effects_compute_power_gamma(
    num_subjects=len(data["user_id"].unique()),
    trials_per_subject=len(data) // len(data["user_id"].unique()),
    B0=result.params["Intercept"],
    B1=result.params[param_name],
    random_effect_var=result.cov_re.iloc[0, 0],
    shape=np.mean(data["path_length"]) ** 2 / np.var(data["path_length"]),
    n_simulations=n_simulations,
  )

  results = {
    "effect_size": {"name": "Standardized coefficient", "value": effect_size},
    "n_required": n_required,
    "current_power": current_power,
    "test_results": {
      "name": "Linear mixed effects model",
      "statistic": result.tvalues[param_name],
      "p_value": result.pvalues[param_name],
      "n1": n1,
      "n2": n2,
    },
    "descriptive": {
      "means": {"no_reuse": mean1, "reuse": mean2},
      "medians": {"no_reuse": median1, "reuse": median2},
      "sds": {"no_reuse": np.sqrt(var1), "reuse": np.sqrt(var2)},
      "ses": {"no_reuse": np.sqrt(var1 / n1), "reuse": np.sqrt(var2 / n2)},
    },
    "raw_means": {
      "no_reuse": no_reuse_stats["mean_val"].to_numpy(),
      "reuse": reuse_stats["mean_val"].to_numpy(),
    },
  }

  if stats_file:
    stats_file.write("\nLinear Mixed Effects Model Results (Path Length):\n")
    stats_file.write("=========================================\n")
    stats_file.write(str(result.summary()) + "\n\n")

    stats_file.write("Sample Sizes:\n")
    stats_file.write(f"\tNo Reuse: {n1} users\n")
    stats_file.write(f"\tReuse: {n2} users\n")
    stats_file.write(
      f"\tTrials per user: {len(data) // len(data['user_id'].unique())}\n\n"
    )

    stats_file.write("Means:\n")
    stats_file.write(f"\tNo Reuse: {mean1:.3f}\n")
    stats_file.write(f"\tReuse: {mean2:.3f}\n")
    stats_file.write(f"\tReuse: {mean2:.3f}\n")
    stats_file.write(f"\tDifference: {mean2 - mean1:.3f}\n\n")

    stats_file.write("Medians:\n")
    stats_file.write(f"\tNo Reuse: {median1:.3f}\n")
    stats_file.write(f"\tReuse: {median2:.3f}\n")
    stats_file.write(f"\tDifference: {median2 - median1:.3f}\n\n")

    stats_file.write(f"Effect size: {effect_size:.3f}\n\n")

    stats_file.write("Power Analysis:\n")
    stats_file.write(f"Current power: {current_power:.3f}\n")
    for power, n in n_required.items():
      stats_file.write(f"Required sample size for {power * 100}% power: {n}\n")

  return results


def mixed_effects_compute_power_gamma(
  num_subjects: int,
  trials_per_subject: int,
  B0: float,
  B1: float,
  random_effect_var: float,
  shape: float,
  n_simulations: int = 500,
  alpha: float = 0.05,
):
  """Simulate a mixed effects trial with gamma-distributed path lengths and test for significance.

  Args:
      num_subjects: Number of subjects in simulation
      trials_per_subject: Number of trials per subject
      B0: Intercept coefficient
      B1: Slope coefficient
      random_effect_var: Variance of random effects
      shape: Shape parameter for gamma distribution
      n_simulations: Number of simulations to run
      alpha: Significance level for test

  Returns:
      float: Computed power (proportion of significant results)
  """
  significant_results = 0

  for _ in range(n_simulations):
    # Generate data
    user_ids = np.repeat(range(num_subjects), trials_per_subject)
    reuse = np.random.choice([0, 1], num_subjects * trials_per_subject)
    random_intercepts = np.random.normal(0, np.sqrt(random_effect_var), num_subjects)
    random_intercepts = np.repeat(random_intercepts, trials_per_subject)

    # Generate path lengths using gamma distribution
    mean = np.exp(B0 + B1 * reuse + random_intercepts)
    scale = mean / shape  # scale parameter for gamma distribution
    path_length = np.random.gamma(shape, scale)

    # Create and test model
    data = pd.DataFrame(
      {"path_length": path_length, "reuse": reuse, "user_id": user_ids}
    )
    model = smf.mixedlm("path_length ~ reuse", data, groups=data["user_id"])
    try:
      result = model.fit(reml=True)
      param_name = "reuse[T.True]" if "reuse[T.True]" in result.params else "reuse"
      if result.pvalues[param_name] < alpha:
        significant_results += 1
    except:
      continue

  return significant_results / n_simulations


def power_analysis_rt_differences(
  difference_df: pl.DataFrame, measure: str, alpha: float = 0.05, stats_file=None
) -> dict:
  """Analyze RT differences between conditions with appropriate statistical tests.

  Args:
      difference_df: DataFrame containing RT differences and user/reversal info
      measure: Name of RT measure column being analyzed
      alpha: significance level for tests (default: 0.05)
      stats_file: optional file handle to write stats output

  Returns:
      dict containing test results and effect size
  """
  # Get mean difference per user (averaging across reversals)
  user_means = (
    difference_df.group_by("user_id").agg(pl.col(measure).mean()).select(measure)
  )
  differences = user_means.to_numpy().flatten()

  n = len(differences)

  # Test normality using Shapiro-Wilk test
  _, normality_p = stats.shapiro(differences)
  is_normal = normality_p > alpha

  # Calculate mean and standard error
  mean = np.mean(differences)
  se = np.std(differences, ddof=1) / np.sqrt(n)
  sd = np.std(differences, ddof=1)  # Standard deviation for reporting

  # Calculate median
  median = np.median(differences)

  # Bootstrap 95% confidence intervals for the median
  n_bootstrap = 10000
  bootstrap_medians = []
  for _ in range(n_bootstrap):
    bootstrap_sample = np.random.choice(differences, size=n, replace=True)
    bootstrap_medians.append(np.median(bootstrap_sample))

  ci_low, ci_high = np.percentile(bootstrap_medians, [2.5, 97.5])

  if stats_file:
    stats_file.write(f"\n{measure}:\n")
    stats_file.write("=" * (len(measure) + 10) + "\n")
    stats_file.write(f"N = {n} participants\n")
    stats_file.write(f"Mean difference = {mean:.3f} (SE: {se:.3f})\n")
    stats_file.write(
      f"Median difference = {median:.3f} (95% CI: [{ci_low:.3f}, {ci_high:.3f}])\n\n"
    )
    stats_file.write("Normality Test (Shapiro-Wilk):\n")
    stats_file.write(
      f"p = {normality_p:.3f} ({'Normal' if is_normal else 'Non-normal'} distribution)\n\n"
    )

  if is_normal:
    # One-sided paired t-test (testing if condition 1 < condition 2)
    t_stat, p_value = stats.ttest_rel(differences, np.zeros_like(differences))
    # Convert to one-sided p-value if t-statistic is in predicted direction
    p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2
    df = n - 1  # Degrees of freedom for one-sample t-test

    # Calculate Cohen's d effect size for paired differences
    d = mean / np.std(differences, ddof=1)
    effect_size = {"name": "Cohen's d", "value": d}

    test_name = "Paired t-test"
    test_stat = t_stat

    # Power analysis for paired t-test
    analysis = TTestPower()
    actual_power = analysis.power(
      effect_size=abs(d), nobs=n, alpha=alpha, alternative="larger"
    )

    # Calculate required sample sizes for different power levels
    power_levels = [0.8, 0.9, 0.95]
    required_n = {}
    for power in power_levels:
      n_required = analysis.solve_power(
        effect_size=abs(d), alpha=alpha, power=power, alternative="larger"
      )
      required_n[power] = ceil(n_required)

  else:
    # One-sided Wilcoxon signed-rank test
    w_stat, p_value = stats.wilcoxon(differences, alternative="greater")
    test_name = "Wilcoxon signed-rank test"
    test_stat = w_stat
    df = (
      n - 1
    )  # Using n-1 for consistency in reporting, though Wilcoxon technically doesn't use df

    # Calculate r effect size for Wilcoxon test
    # Convert p-value to z-score using inverse normal CDF
    z = stats.norm.ppf(1 - p_value)  # One-sided p-value
    r = z / np.sqrt(n)  # Standardize by sample size
    effect_size = {"name": "r", "value": r}

    # Convert r to d for power analysis
    # Formula: d = 2r/sqrt(1-r^2)
    d = 2 * r / sqrt(1 - r**2) if abs(r) < 1 else float("inf")

    # Power analysis using t-test as approximation (with 95% efficiency adjustment)
    analysis = TTestPower()
    actual_power = (
      analysis.power(effect_size=abs(d), nobs=n, alpha=alpha, alternative="larger")
      * 0.95
    )  # Adjust for Wilcoxon efficiency

    # Calculate required sample sizes for different power levels
    power_levels = [0.8, 0.9, 0.95]
    required_n = {}
    for power in power_levels:
      n_required = analysis.solve_power(
        effect_size=abs(d), alpha=alpha, power=power, alternative="larger"
      )
      # Adjust for Wilcoxon efficiency
      required_n[power] = ceil(n_required / 0.95)

  paper_result = f"Mean={mean:.3f}, Median={median:.3f} [95% CI: {ci_low:.3f}, {ci_high:.3f}], t({df})={test_stat:.3f}, p={p_value:.2g}"
  add_to_file(stats_file, algo="human data", label="rt_difference", text=paper_result)
  if stats_file:
    stats_file.write(f"{test_name}:\n")
    stats_file.write(f"statistic = {test_stat:.3f}\n")
    stats_file.write(f"p = {p_value:.3f}\n\n")
    stats_file.write(f"Effect Size ({effect_size['name']}):\n")
    stats_file.write(f"{effect_size['value']:.3f}\n\n")
    stats_file.write("<<<<<FOR PAPER>>>>>\n")
    stats_file.write(f"{paper_result}\n")

    # Add power analysis results
    stats_file.write("Power Analysis:\n")
    stats_file.write(f"Achieved power with current N={n}: {actual_power:.3f}\n")
    stats_file.write("Required sample sizes:\n")
    for power, n_req in required_n.items():
      stats_file.write(f"  {power * 100:g}% power: N ≥ {n_req}\n")

  return {
    "n": n,
    "mean": mean,
    "median": median,
    "median_ci": np.array([ci_low, ci_high]),
    "se": se,
    "sd": sd,  # Add SD to return value for paper reporting
    "df": df,  # Add degrees of freedom to return value for paper reporting
    "normality": {"is_normal": is_normal, "p_value": normality_p},
    "test": {"name": test_name, "statistic": test_stat, "p_value": p_value},
    "effect_size": effect_size,
    "power_analysis": {"actual_power": actual_power, "required_n": required_n},
  }


def plot_success_rate_efficient_reuse_metrics(
  df: DataFrame,
  model_df: DataFrame = None,
  stats_file=None,
  ax=None,
  reuse_columns: List[str] = [
    "efficient_reuse_1.25",
    "efficient_reuse_1.5",
    "efficient_reuse_1.75",
    "efficient_reuse_2",
  ],
  title="Success Rate and Efficient Path Reuse",
  figsize=(8, 8),
  color=default_colors["orange"],
) -> Tuple[plt.Figure, plt.Axes]:
  """Plot success rate vs different efficient path reuse metrics as a 2D scatter plot with error bars.

  Args:
      df (DataFrame): DataFrame containing human data
      model_df (DataFrame, optional): DataFrame containing model data. If None, only plots human data.
      stats_file (file, optional): File to write statistics to
      ax (plt.Axes, optional): Matplotlib axes to plot on. If None, creates new figure
      reuse_columns (List[str], optional): List of efficient reuse column names to plot
      title (str, optional): Plot title
      figsize (tuple, optional): Figure size if creating new figure
      color (str, optional): Color to use for all data points

  Returns:
      tuple: (fig, ax) containing the figure and axes object
  """

  # Create figure if needed
  if ax is None:
    fig, ax = plt.subplots(figsize=figsize)
  else:
    fig = ax.figure

  # Calculate human success statistics (shared across all reuse metrics)
  human_successes = (
    df.group_by("user_id")
    .agg(pl.col("success").mean())
    .select("success")
    .to_numpy()
    .flatten()
  )
  human_success_mean = np.mean(human_successes)
  human_success_se = np.sqrt(
    (human_success_mean * (1 - human_success_mean)) / len(human_successes)
  )

  # Prepare data for plotting
  reuse_data = []

  # Calculate statistics for each reuse metric
  for reuse_column in reuse_columns:
    results = power_analysis_path_reuse(
      df, measure=reuse_column, mu=0.5, alpha=0.05, plot=False, stats_file=stats_file
    )

    # Extract threshold value from column name (e.g., "efficient_reuse_1.25" -> "1.25")
    threshold = reuse_column.split("_")[-1]

    reuse_data.append(
      {"threshold": threshold, "reuse_mean": results["mean"], "reuse_se": results["se"]}
    )

  # Plot each reuse metric data point with error bars
  marker_size = 100  # Size of the scatter points

  for data in reuse_data:
    ax.errorbar(
      100 * data["reuse_mean"],
      100 * human_success_mean,
      xerr=100 * data["reuse_se"],
      yerr=100 * human_success_se,
      fmt="none",
      color=color,
      capsize=5,
      capthick=2,
      elinewidth=2,
      zorder=2,
    )
    ax.scatter(
      100 * data["reuse_mean"],
      100 * human_success_mean,
      color=color,
      s=marker_size,
      zorder=3,
    )
    # Add threshold label next to each point
    ax.annotate(
      data["threshold"],
      xy=(100 * data["reuse_mean"] + 2, 100 * human_success_mean + 2),
      fontsize=10,
      zorder=4,
    )

  # Customize axes
  ax.set_xlabel("Efficient Path Reuse (%)", fontsize=DEFAULT_LABEL_SIZE)
  ax.set_ylabel("Success Rate (%)", fontsize=DEFAULT_LABEL_SIZE)
  ax.set_title(title, fontsize=DEFAULT_TITLE_SIZE)

  # Add chance level lines
  ax.axhline(y=50, color="r", linestyle="--", alpha=0.5, label="Chance level")
  ax.axvline(x=50, color="r", linestyle="--", alpha=0.5)

  # Set axis limits with some padding
  ax.set_xlim(-5, 105)
  ax.set_ylim(-5, 105)

  # Add grid
  ax.grid(True, linestyle="--", alpha=0.7)

  return fig, ax
