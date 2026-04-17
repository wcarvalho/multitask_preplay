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

# Extract path_length and total_rt data for each condition
reuse_path_lengths = reuse_df["path_length"].to_numpy()
reuse_total_rts = reuse_df["total_rt"].to_numpy()

no_reuse_path_lengths = no_reuse_df["path_length"].to_numpy()
no_reuse_total_rts = no_reuse_df["total_rt"].to_numpy()

# Get sample sizes for standard error calculation
n_reuse = len(reuse_df)
n_no_reuse = len(no_reuse_df)

# Compute overall statistics across episodes
stats = {
  "path_length": {
    "reuse_mean": np.mean(reuse_path_lengths),
    "reuse_std": np.std(reuse_path_lengths),
    "reuse_se": np.std(reuse_path_lengths) / np.sqrt(n_reuse),
    "no_reuse_mean": np.mean(no_reuse_path_lengths),
    "no_reuse_std": np.std(no_reuse_path_lengths),
    "no_reuse_se": np.std(no_reuse_path_lengths) / np.sqrt(n_no_reuse),
  },
  "total_rt": {
    "reuse_mean": np.mean(reuse_total_rts),
    "reuse_std": np.std(reuse_total_rts),
    "reuse_se": np.std(reuse_total_rts) / np.sqrt(n_reuse),
    "no_reuse_mean": np.mean(no_reuse_total_rts),
    "no_reuse_std": np.std(no_reuse_total_rts),
    "no_reuse_se": np.std(no_reuse_total_rts) / np.sqrt(n_no_reuse),
  },
}

# Print statistics
print("Statistics Summary:")
print(
  f"Path Length - Reuse=0: {stats['path_length']['no_reuse_mean']:.3f} ± {stats['path_length']['no_reuse_std']:.3f}"
)
print(
  f"Path Length - Reuse=1: {stats['path_length']['reuse_mean']:.3f} ± {stats['path_length']['reuse_std']:.3f}"
)
print(
  f"Total RT - Reuse=0: {stats['total_rt']['no_reuse_mean']:.3f} ± {stats['total_rt']['no_reuse_std']:.3f}"
)
print(
  f"Total RT - Reuse=1: {stats['total_rt']['reuse_mean']:.3f} ± {stats['total_rt']['reuse_std']:.3f}"
)

# Create 2-panel plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: Path Length
conditions = ["New Path", "Partial Reuse"]
path_length_means = [
  stats["path_length"]["no_reuse_mean"],
  stats["path_length"]["reuse_mean"],
]
path_length_errs = [
  stats["path_length"]["no_reuse_se"],
  stats["path_length"]["reuse_se"],
]

bars1 = ax1.bar(
  conditions,
  path_length_means,
  yerr=path_length_errs,
  color=[default_colors["nice purple"], default_colors["bluish green"]],
  alpha=0.7,
  capsize=5,
)
ax1.set_ylabel("Path Length")
ax1.set_title("Path Length by Reuse Condition")
ax1.grid(True, alpha=0.3)

# Add sample size annotations for path length
sample_sizes = [n_no_reuse, n_reuse]
for i, (bar, mean_val) in enumerate(zip(bars1, path_length_means)):
  height = bar.get_height() + path_length_errs[i]
  ax1.text(
    bar.get_x() + bar.get_width() / 2.0,
    height,
    f"{mean_val:.1f}",
    ha="center",
    va="bottom",
  )

# Panel 2: Total RT
total_rt_means = [stats["total_rt"]["no_reuse_mean"], stats["total_rt"]["reuse_mean"]]
total_rt_errs = [stats["total_rt"]["no_reuse_se"], stats["total_rt"]["reuse_se"]]

bars2 = ax2.bar(
  conditions,
  total_rt_means,
  yerr=total_rt_errs,
  color=[default_colors["nice purple"], default_colors["bluish green"]],
  alpha=0.7,
  capsize=5,
)
ax2.set_ylabel("Total Episode Time")
ax2.set_title("Total Episode Time by Reuse Condition")
ax2.grid(True, alpha=0.3)

# Add sample size annotations for total RT
for i, (bar, mean_val) in enumerate(zip(bars2, total_rt_means)):
  height = bar.get_height() + total_rt_errs[i]
  ax2.text(
    bar.get_x() + bar.get_width() / 2.0,
    height,
    f"{mean_val:.1f}",
    ha="center",
    va="bottom",
  )

plt.tight_layout()
os.makedirs("random/plots", exist_ok=True)
plt.savefig("random/plots/avg_time.png", dpi=300, bbox_inches="tight")
plt.close()
