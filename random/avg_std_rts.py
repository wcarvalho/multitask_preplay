import os

import data_configs
from analysis import analysis_utils
from analysis.jaxmaze_analysis import filter_users_by_success
import numpy as np
import matplotlib.pyplot as plt
from analysis import vis_utils
from plot_configs import default_colors
import polars as pl
from tqdm import tqdm
# Configuration

global_df = pl.read_parquet(data_configs.get_dataframe_path("jaxmaze", "human"))

df, _ = filter_users_by_success(
  global_df.filter(
    manipulation="paths",
    world="big_m3_maze1",
    eval=True,
    tell_reuse=1,
    eval_shares_start_pos=True,
  ),
  analysis_name="path_reuse_results",
)

df = analysis_utils.add_reuse_column(df, "reuse", 0.5)

reuse_df = df.filter(reuse=1)
no_reuse_df = df.filter(reuse=0)

# Sort by first_rt and get top 50% (fastest half) for each condition
reuse_sorted = reuse_df.sort("first_rt")
no_reuse_sorted = no_reuse_df.sort("first_rt")

# Get the fastest 50% of participants
prop = 0.9
reuse_top50 = reuse_sorted.head(int(len(reuse_sorted) * prop))
no_reuse_top50 = no_reuse_sorted.head(int(len(no_reuse_sorted) * prop))


# Function to parse reaction times string
def parse_reaction_times(rt_string):
  # Remove brackets and split by whitespace
  rt_string = rt_string.strip("[]")
  arr = np.array([float(x) for x in rt_string.split()])
  return arr
  # return np.log(arr)


# Function to filter outliers based on max reaction time
def filter_outliers_by_max_rt(df):
  # First, compute max RT for each episode
  max_rts = []
  for row in df.iter_rows(named=True):
    rt_array = parse_reaction_times(row["reaction_times"])
    max_rts.append(np.max(rt_array))

  max_rts = np.array(max_rts)

  # Compute quartiles and IQR
  Q1 = np.percentile(max_rts, 25)
  Q3 = np.percentile(max_rts, 75)
  IQR = Q3 - Q1

  # Define outlier bounds
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR

  # Filter out outliers
  valid_indices = (max_rts >= lower_bound) & (max_rts <= upper_bound)

  print(f"Filtering outliers: {len(df)} -> {np.sum(valid_indices)} episodes")
  print(f"Max RT bounds: [{lower_bound:.3f}, {upper_bound:.3f}]")

  # Convert to list and filter dataframe
  valid_rows = []
  for i, row in enumerate(df.iter_rows(named=True)):
    if valid_indices[i]:
      valid_rows.append(row)

  # Create new dataframe from valid rows
  if valid_rows:
    return pl.DataFrame(valid_rows)
  else:
    return pl.DataFrame(schema=df.schema)


# Compute mean and std of reaction times for each episode
def compute_rt_stats(df):
  mean_rts = []
  std_rts = []

  for row in tqdm(df.iter_rows(named=True), "computing rt stats"):
    rt_array = parse_reaction_times(row["reaction_times"])
    mean_rts.append(np.mean(rt_array))
    std_rts.append(np.std(rt_array))

  return np.array(mean_rts), np.array(std_rts)


# Filter outliers based on max reaction time
print("Filtering outliers for reuse condition:")
reuse_filtered = filter_outliers_by_max_rt(reuse_top50)
print("Filtering outliers for no-reuse condition:")
no_reuse_filtered = filter_outliers_by_max_rt(no_reuse_top50)

# Compute statistics for each condition (using filtered data)
reuse_mean_rts, reuse_std_rts = compute_rt_stats(reuse_filtered)
no_reuse_mean_rts, no_reuse_std_rts = compute_rt_stats(no_reuse_filtered)

# Get sample sizes for standard error calculation (after filtering)
n_reuse = len(reuse_filtered)
n_no_reuse = len(no_reuse_filtered)

# Compute overall statistics across episodes
stats = {
  "reuse_mean": {
    "mean_of_means": np.mean(reuse_mean_rts),
    "std_of_means": np.std(reuse_mean_rts),
    "mean_of_stds": np.mean(reuse_std_rts),
    "std_of_stds": np.std(reuse_std_rts),
  },
  "no_reuse_mean": {
    "mean_of_means": np.mean(no_reuse_mean_rts),
    "std_of_means": np.std(no_reuse_mean_rts),
    "mean_of_stds": np.mean(no_reuse_std_rts),
    "std_of_stds": np.std(no_reuse_std_rts),
  },
}

# Print statistics
print("Statistics Summary:")
print(
  f"Reuse=0: Mean RT = {stats['no_reuse_mean']['mean_of_means']:.3f} ± {stats['no_reuse_mean']['std_of_means']:.3f}"
)
print(
  f"Reuse=1: Mean RT = {stats['reuse_mean']['mean_of_means']:.3f} ± {stats['reuse_mean']['std_of_means']:.3f}"
)
print(
  f"Reuse=0: Std RT = {stats['no_reuse_mean']['mean_of_stds']:.3f} ± {stats['no_reuse_mean']['std_of_stds']:.3f}"
)
print(
  f"Reuse=1: Std RT = {stats['reuse_mean']['mean_of_stds']:.3f} ± {stats['reuse_mean']['std_of_stds']:.3f}"
)

# Create 2-panel plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: Mean reaction times
conditions = ["Reuse=0", "Reuse=1"]
means_data = [
  stats["no_reuse_mean"]["mean_of_means"],
  stats["reuse_mean"]["mean_of_means"],
]
means_err = [
  stats["no_reuse_mean"]["std_of_means"] / np.sqrt(n_no_reuse),
  stats["reuse_mean"]["std_of_means"] / np.sqrt(n_reuse),
]

bars1 = ax1.bar(
  conditions,
  means_data,
  yerr=means_err,
  color=[default_colors["nice purple"], default_colors["bluish green"]],
  alpha=0.7,
  capsize=5,
)
ax1.set_ylabel("Mean Reaction Time")
ax1.set_title("Mean of Reaction Times")
ax1.grid(True, alpha=0.3)

# Add sample size annotations on bars
sample_sizes = [n_no_reuse, n_reuse]
for i, (bar, n) in enumerate(zip(bars1, sample_sizes)):
  height = bar.get_height() + means_err[i]
  ax1.text(
    bar.get_x() + bar.get_width() / 2.0, height, f"n={n}", ha="center", va="bottom"
  )

# Panel 2: Std of reaction times
stds_data = [
  stats["no_reuse_mean"]["mean_of_stds"],
  stats["reuse_mean"]["mean_of_stds"],
]
stds_err = [
  stats["no_reuse_mean"]["std_of_stds"] / np.sqrt(n_no_reuse),
  stats["reuse_mean"]["std_of_stds"] / np.sqrt(n_reuse),
]

bars2 = ax2.bar(
  conditions,
  stds_data,
  yerr=stds_err,
  color=[default_colors["nice purple"], default_colors["bluish green"]],
  alpha=0.7,
  capsize=5,
)
ax2.set_ylabel("Std of Reaction Time")
ax2.set_title("Standard Deviation of Reaction Times")
ax2.grid(True, alpha=0.3)

# Add sample size annotations on bars
for i, (bar, n) in enumerate(zip(bars2, sample_sizes)):
  height = bar.get_height() + stds_err[i]
  ax2.text(
    bar.get_x() + bar.get_width() / 2.0, height, f"n={n}", ha="center", va="bottom"
  )

plt.tight_layout()
os.makedirs("random/plots", exist_ok=True)
plt.savefig("random/plots/avg_std_rts.png", dpi=300, bbox_inches="tight")
