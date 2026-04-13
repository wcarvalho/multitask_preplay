"""
Path Reuse Split Analysis

This script creates a 2-column plot for comparison:
- Left panel: Original 2-condition RT plot (New Path vs Partial Reuse)
- Right panel: 3-condition RT plot (New Path, Reuse Left, Reuse Right)

Usage:
    uv run python random/path_reuse_split.py
"""

import sys
import os

sys.path.insert(0, ".")
sys.path.append("simulations")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.formula.api as smf

from data_processing import process_user_data
from analysis import analysis_utils
import plot_configs

DEFAULT_TITLE_SIZE = 15
DEFAULT_LABEL_SIZE = 15
DEFAULT_LEGEND_SIZE = 10.5
OVERLAP_THRESHOLD = 0.7


def compute_group_statistics(df, measure, group_column, group_value):
  """Compute statistics for a specific group."""
  group_df = df.filter(pl.col(group_column) == group_value)

  if len(group_df) == 0:
    return None

  # Get user-level means
  user_means = (
    group_df.group_by("user_id").agg(pl.col(measure).mean().alias("user_mean"))
  )["user_mean"].to_numpy()

  if len(user_means) == 0:
    return None

  # Compute median and bootstrap CI
  median = np.median(user_means)
  n_bootstrap = 10000
  bootstrap_medians = []
  for _ in range(n_bootstrap):
    sample = np.random.choice(user_means, size=len(user_means), replace=True)
    bootstrap_medians.append(np.median(sample))

  ci_low = np.percentile(bootstrap_medians, 2.5)
  ci_high = np.percentile(bootstrap_medians, 97.5)

  return {
    "median": median,
    "ci": (ci_low, ci_high),
    "user_means": user_means,
    "n_users": len(user_means),
    "n_episodes": len(group_df),
  }


def run_pairwise_comparisons(sub_df, measure):
  """Run LME comparisons: reuse_left vs new_path, reuse_right vs new_path.

  Uses the same linear mixed effects model approach as power_analysis_rt_across_groups.
  """
  results = {}
  for test_condition in ["reuse_left", "reuse_right"]:
    # Filter to new_path and this test condition
    comparison_df = sub_df.filter(
      pl.col("condition").is_in(["new_path", test_condition])
    )

    # Create binary column (True = test condition, False = new_path)
    comparison_df = comparison_df.with_columns(
      (pl.col("condition") == test_condition).alias("is_reuse")
    )

    # Convert to pandas for statsmodels
    data = comparison_df.select(["user_id", "is_reuse", measure]).to_pandas()
    data.columns = ["user_id", "reuse", "RT"]

    # Fit LME model
    model = smf.mixedlm("RT ~ reuse", data, groups=data["user_id"])
    result = model.fit(reml=True)

    # Extract key statistics
    param_name = "reuse[T.True]" if "reuse[T.True]" in result.params else "reuse"
    results[test_condition] = {
      "beta": result.params[param_name],
      "se": result.bse[param_name],
      "t": result.tvalues[param_name],
      "p": result.pvalues[param_name],
      "ci": result.conf_int(alpha=0.05).loc[param_name].values,
      "df": result.df_resid,
    }
  return results


def load_and_prepare_data(tell_reuse=1):
  """Load data and prepare the filtered DataFrame used by both plots.

  Uses the EXACT same filtering logic as path_reuse_results in jaxmaze_analysis.py:
  1. Filter by: tell_reuse, eval_shares_start_pos=True, manipulation="paths",
     world="big_m3_maze1", eval=True
  2. Filter to users with min_train_success=True
  3. Take first 100 users sorted by session_start
  """
  from analysis.jaxmaze_analysis import filter_users_by_success

  print("Loading jaxmaze human data...")
  user_df = process_user_data.get_jaxmaze_human_data(
    debug=False,
    load_df_only=False,
    overwrite_episode_df=False,
  )

  # Check if approach_direction column exists
  df = analysis_utils.get_polars_df(user_df)
  if "approach_direction" not in df.columns:
    raise ValueError(
      "approach_direction column not found. "
      "Please run: python data_processing/process_user_data.py --env jaxmaze --df"
    )

  # Apply EXACT same filtering as path_reuse_results
  eval_filter = dict(
    manipulation="paths",
    world="big_m3_maze1",
    eval=True,
  )

  # Filter using the same approach as path_reuse_results
  filtered_user_df = user_df.filter(
    tell_reuse=tell_reuse, eval_shares_start_pos=True, **eval_filter
  )

  # Apply filter_users_by_success (takes first 100 users with min_train_success)
  sub_df, unique_user_ids = filter_users_by_success(
    filtered_user_df, analysis_name="path_reuse_results"
  )

  # Convert to polars for further processing
  sub_df = analysis_utils.get_polars_df(sub_df)

  print(f"\nFiltered to {len(sub_df)} eval episodes")
  print(f"Unique users: {sub_df['user_id'].n_unique()}")

  # Add reuse column based on overlap threshold (boolean)
  sub_df = sub_df.with_columns(
    (pl.col("overlap") > OVERLAP_THRESHOLD).alias("is_reuse")
  )

  # Create 3-condition column
  def get_condition(row):
    if not row["is_reuse"]:
      return "new_path"
    elif row["approach_direction"] == "left":
      return "reuse_left"
    elif row["approach_direction"] == "right":
      return "reuse_right"
    else:
      return "reuse_unknown"

  sub_df = sub_df.with_columns(
    pl.struct(["is_reuse", "approach_direction"])
    .map_elements(get_condition, return_dtype=pl.String)
    .alias("condition")
  )

  return sub_df


def print_data_summary(sub_df):
  """Print summary statistics about the data."""
  print("\n--- Approach Direction Distribution ---")
  direction_counts = sub_df.group_by("approach_direction").agg(pl.len().alias("count"))
  print(direction_counts)

  print("\n--- Approach Direction by Reuse ---")
  reuse_direction = sub_df.group_by(["is_reuse", "approach_direction"]).agg(
    pl.len().alias("count")
  )
  print(reuse_direction)

  print("\n--- Condition Distribution ---")
  condition_counts = sub_df.group_by("condition").agg(pl.len().alias("count"))
  print(condition_counts)


def plot_3_condition(sub_df, ax, measure):
  """Plot the 3-condition RT comparison (New Path, Reuse Left, Reuse Right)."""
  conditions = ["new_path", "reuse_left", "reuse_right"]
  condition_labels = ["New Path", "Reuse, Left", "Reuse, Right"]
  colors = [
    plot_configs.default_colors["nice purple"],
    plot_configs.default_colors["sky blue"],
    plot_configs.default_colors["bluish green"],
  ]

  all_stats = []
  print(f"\n=== 3-Condition Statistics ({measure}) ===")
  for condition in conditions:
    stats = compute_group_statistics(sub_df, measure, "condition", condition)
    if stats is not None:
      print(
        f"{condition}: n_users={stats['n_users']}, n_episodes={stats['n_episodes']}, "
        f"median={stats['median']:.3f}, CI=[{stats['ci'][0]:.3f}, {stats['ci'][1]:.3f}]"
      )
    all_stats.append(stats)

  # Filter out conditions with no data
  valid_conditions = [
    (cond, label, color, stats)
    for cond, label, color, stats in zip(
      conditions, condition_labels, colors, all_stats
    )
    if stats is not None
  ]

  if not valid_conditions:
    print("No valid conditions found with data!")
    return None

  x_pos = np.arange(len(valid_conditions))
  medians = [s[3]["median"] for s in valid_conditions]
  labels = [s[1] for s in valid_conditions]
  bar_colors = [s[2] for s in valid_conditions]

  # Calculate error bars
  lower_errors = [medians[i] - s[3]["ci"][0] for i, s in enumerate(valid_conditions)]
  upper_errors = [s[3]["ci"][1] - medians[i] for i, s in enumerate(valid_conditions)]

  ax.bar(x_pos, medians, color=bar_colors)
  ax.errorbar(
    x_pos,
    medians,
    yerr=[lower_errors, upper_errors],
    fmt="none",
    ecolor="black",
    capsize=5,
  )

  # Add individual data points with jitter
  for i, (_, _, _, stats) in enumerate(valid_conditions):
    x_jitter = np.random.normal(i, 0.08, size=len(stats["user_means"]))
    ax.scatter(x_jitter, stats["user_means"], alpha=0.3, color="black", s=20)

  ax.set_xticks(x_pos)
  ax.set_xticklabels(labels, ha="center", fontsize=DEFAULT_LABEL_SIZE - 1)
  ax.set_ylabel("Log RT", fontsize=DEFAULT_LABEL_SIZE)
  ax.set_title("Split by Approach Direction", fontsize=DEFAULT_TITLE_SIZE)
  ax.tick_params(axis="both", which="major", labelsize=DEFAULT_LABEL_SIZE)
  ax.grid(True, linestyle="--", alpha=0.7)

  # Return user means for y-axis limits
  all_user_means = np.concatenate([s[3]["user_means"] for s in valid_conditions])
  return all_user_means


def create_comparison_plot(sub_df, measure, save_dir):
  """Create and save the 2-panel comparison plot for a given measure."""
  print(f"\n{'=' * 60}")
  print(f"Creating plot for {measure}")
  print(f"{'=' * 60}")

  # Create 2-column figure
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

  # Left panel: 2-condition plot
  print(f"\n=== 2-Condition Plot ({measure}) ===")
  analysis_utils.plot_bar_rt_comparison(
    sub_df,
    measure,
    ax=ax1,
    overlap_threshold=OVERLAP_THRESHOLD,
    ylabel="Log RT",
    use_box_plot=False,
    use_median=True,
    include_raw_data=True,
  )
  ax1.set_title("Original: 2 Conditions", fontsize=DEFAULT_TITLE_SIZE)

  # Right panel: 3-condition plot
  plot_3_condition(sub_df, ax2, measure)

  # Use same y-limits as path_reuse_results (ylim=(7, 8))
  shared_ylim = (7, 8)
  ax1.set_ylim(shared_ylim)
  ax2.set_ylim(shared_ylim)

  plt.tight_layout()

  # Save figure (PDF only)
  filename = f"path_reuse_split_{measure}.pdf"
  fig.savefig(os.path.join(save_dir, filename), bbox_inches="tight")
  print(f"\nFigure saved to {save_dir}/{filename}")

  plt.close(fig)

  # Run statistical comparisons
  print(f"\n=== Pairwise Statistical Tests ({measure}) ===")
  stats_results = run_pairwise_comparisons(sub_df, measure)
  for condition, stats in stats_results.items():
    label = "Reuse, Left" if condition == "reuse_left" else "Reuse, Right"
    print(
      f"{label} vs New Path: "
      f"beta={stats['beta']:.3f}, SE={stats['se']:.3f}, "
      f"t({stats['df']:.0f})={stats['t']:.2f}, p={stats['p']:.2g}, "
      f"95% CI [{stats['ci'][0]:.3f}, {stats['ci'][1]:.3f}]"
    )


def main():
  # Load and prepare data (same data used for all plots)
  sub_df = load_and_prepare_data()

  # Print summary statistics
  print_data_summary(sub_df)

  # Save directory
  save_dir = "random/plots"
  os.makedirs(save_dir, exist_ok=True)

  # Create plots for both measures
  for measure in ["first_log_rt", "max_log_rt"]:
    create_comparison_plot(sub_df, measure, save_dir)


if __name__ == "__main__":
  main()
