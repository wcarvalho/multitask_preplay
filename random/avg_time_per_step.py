import os

from data_processing import process_user_data
from analysis import analysis_utils
from analysis.jaxmaze_analysis import filter_users_by_success
import numpy as np
import matplotlib.pyplot as plt
from plot_configs import default_colors

# Load data
global_df = process_user_data.get_jaxmaze_human_data(
  load_df_only=True,
)

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


def parse_reaction_times(rt_string):
  rt_string = rt_string.strip("[]")
  return np.array([float(x) for x in rt_string.split()])


def compute_per_user_mean_trajectories(condition_df):
  """For each user, average their per-timestep RTs across trials."""
  user_ids = condition_df["user_id"].unique().to_list()
  user_means = []
  for uid in user_ids:
    user_rows = condition_df.filter(user_id=uid)
    rt_arrays = [
      parse_reaction_times(row["reaction_times"])
      for row in user_rows.iter_rows(named=True)
    ]
    if not rt_arrays:
      continue
    max_len = max(len(a) for a in rt_arrays)
    padded = np.full((len(rt_arrays), max_len), np.nan)
    for i, a in enumerate(rt_arrays):
      padded[i, : len(a)] = a
    user_mean = np.nanmean(padded, axis=0)
    user_means.append(user_mean)
  return user_means


def compute_cross_user_stats(user_means):
  """Stack user means and compute mean/STE across users at each timestep."""
  max_len = max(len(m) for m in user_means)
  stacked = np.full((len(user_means), max_len), np.nan)
  for i, m in enumerate(user_means):
    stacked[i, : len(m)] = m
  count = np.sum(~np.isnan(stacked), axis=0)
  mean = np.nanmean(stacked, axis=0)
  ste = np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(count)
  return mean, ste, count


# Compute trajectories
reuse_user_means = compute_per_user_mean_trajectories(reuse_df)
no_reuse_user_means = compute_per_user_mean_trajectories(no_reuse_df)

reuse_mean, reuse_ste, reuse_count = compute_cross_user_stats(reuse_user_means)
no_reuse_mean, no_reuse_ste, no_reuse_count = compute_cross_user_stats(
  no_reuse_user_means
)

color_no_reuse = default_colors["nice purple"]
color_reuse = default_colors["bluish green"]
timesteps_nr = np.arange(len(no_reuse_mean))
timesteps_r = np.arange(len(reuse_mean))


# Same log transform as process_user_data.py: log(1000 * rt + 1e-5)
def log_rt(rt):
  return np.log(1000 * rt + 1e-5)


# Compute log-transformed per-user means and cross-user stats
log_reuse_user_means = [log_rt(m) for m in reuse_user_means]
log_no_reuse_user_means = [log_rt(m) for m in no_reuse_user_means]

log_reuse_mean, log_reuse_ste, _ = compute_cross_user_stats(log_reuse_user_means)
log_no_reuse_mean, log_no_reuse_ste, _ = compute_cross_user_stats(
  log_no_reuse_user_means
)

os.makedirs("random/plots", exist_ok=True)

for scale, suffix in [("linear", ""), ("log", "_log")]:
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, sharey=True)

  if scale == "log":
    nr_mean, nr_ste = log_no_reuse_mean, log_no_reuse_ste
    r_mean, r_ste = log_reuse_mean, log_reuse_ste
    ylabel = "Log RT"
  else:
    nr_mean, nr_ste = no_reuse_mean, no_reuse_ste
    r_mean, r_ste = reuse_mean, reuse_ste
    ylabel = "RT (seconds)"

  # Top panel: New Path (no reuse)
  ax1.bar(
    timesteps_nr,
    nr_mean,
    color=color_no_reuse,
    alpha=0.7,
    width=1.0,
  )
  ax1.set_title("New Path", fontsize=14)
  ax1.set_ylabel(ylabel)
  ax1.grid(True, alpha=0.3, axis="y")

  # Bottom panel: Partial Reuse
  ax2.bar(
    timesteps_r,
    r_mean,
    color=color_reuse,
    alpha=0.7,
    width=1.0,
  )
  ax2.set_title("Partial Reuse", fontsize=14)
  ax2.set_xlabel("Timestep")
  ax2.set_ylabel(ylabel)
  ax2.grid(True, alpha=0.3, axis="y")

  if scale == "log":
    ax1.set_ylim(4, 8)
    ax2.set_ylim(4, 8)

  plt.tight_layout()
  fname = f"random/plots/avg_time_per_step{suffix}.png"
  plt.savefig(fname, dpi=300, bbox_inches="tight")
  plt.close()
  print(f"Saved to {fname}")

# Side-by-side grouped bar plot
for scale, suffix in [("linear", ""), ("log", "_log")]:
  fig, ax = plt.subplots(1, 1, figsize=(12, 4))

  if scale == "log":
    nr_vals, r_vals = log_no_reuse_mean, log_reuse_mean
    ylabel = "Log RT"
  else:
    nr_vals, r_vals = no_reuse_mean, reuse_mean
    ylabel = "RT (seconds)"

  # Pad to same length
  max_len = max(len(nr_vals), len(r_vals))
  nr_padded = np.full(max_len, np.nan)
  r_padded = np.full(max_len, np.nan)
  nr_padded[: len(nr_vals)] = nr_vals
  r_padded[: len(r_vals)] = r_vals

  bar_width = 0.4
  x = np.arange(max_len)
  ax.bar(
    x - bar_width / 2,
    np.nan_to_num(nr_padded),
    width=bar_width,
    color=color_no_reuse,
    alpha=0.7,
    label="New Path",
  )
  ax.bar(
    x + bar_width / 2,
    np.nan_to_num(r_padded),
    width=bar_width,
    color=color_reuse,
    alpha=0.7,
    label="Partial Reuse",
  )
  ax.set_xlabel("Timestep")
  ax.set_ylabel(ylabel)
  ax.legend()
  ax.grid(True, alpha=0.3, axis="y")

  if scale == "log":
    ax.set_ylim(4, 8)

  plt.tight_layout()
  fname = f"random/plots/avg_time_per_step_paired{suffix}.png"
  plt.savefig(fname, dpi=300, bbox_inches="tight")
  plt.close()
  print(f"Saved to {fname}")
