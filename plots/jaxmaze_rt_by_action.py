"""RT by same/different action analysis.

For each timestep t >= 1, classify as "same" (actions[t] == actions[t-1])
or "different" (actions[t] != actions[t-1]). Pool across episodes per user,
then compute mean RT per user for each bin.

One plot per condition: mean +/- SEM of raw RT for same vs different,
with a horizontal line at the overall mean RT.

Migrated from random/rt_by_action.py.

Usage:
    python plots/jaxmaze_rt_by_action.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Repo-root imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import analysis_utils
from analysis.jaxmaze_analysis import filter_users_by_success
import data_configs

import plot_configs  # noqa: F401 — from same directory

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def parse_reaction_times(rt_string):
  rt_string = rt_string.strip("[]")
  return np.array([float(x) for x in rt_string.split()])


def parse_actions(action_string):
  action_string = action_string.strip("[]")
  return np.array([int(x) for x in action_string.split()])


def compute_same_diff_rt(condition_df):
  """For each user, compute mean RT for same-action vs different-action timesteps.

  Returns:
      same_means: list of per-user mean RT for same-action timesteps
      diff_means: list of per-user mean RT for different-action timesteps
      overall_means: list of per-user overall mean RT (for demeaning)
  """
  user_ids = condition_df["user_id"].unique().to_list()
  same_means = []
  diff_means = []
  overall_means = []

  for uid in user_ids:
    user_rows = condition_df.filter(user_id=uid)
    same_rts = []
    diff_rts = []
    all_rts = []

    for row in user_rows.iter_rows(named=True):
      rts = parse_reaction_times(row["reaction_times"])
      actions = parse_actions(row["actions"])

      # Ensure same length
      min_len = min(len(rts), len(actions))
      rts = rts[:min_len]
      actions = actions[:min_len]

      all_rts.extend(rts.tolist())

      # Classify timesteps t >= 1
      for t in range(1, len(actions)):
        if actions[t] == actions[t - 1]:
          same_rts.append(rts[t])
        else:
          diff_rts.append(rts[t])

    if not same_rts or not diff_rts:
      continue

    same_means.append(np.mean(same_rts))
    diff_means.append(np.mean(diff_rts))
    overall_means.append(np.mean(all_rts))

  return (
    np.array(same_means),
    np.array(diff_means),
    np.array(overall_means),
  )


def plot_bars(same_vals, diff_vals, overall_mean, title, save_path):
  means = [np.mean(same_vals), np.mean(diff_vals)]
  sems = [
    np.std(same_vals, ddof=1) / np.sqrt(len(same_vals)),
    np.std(diff_vals, ddof=1) / np.sqrt(len(diff_vals)),
  ]
  fig, ax = plt.subplots(figsize=(4, 4))
  ax.bar(
    [0, 1],
    means,
    yerr=sems,
    capsize=5,
    color=["steelblue", "darkorange"],
    edgecolor="white",
    width=0.6,
  )
  ax.axhline(overall_mean, color="black", linewidth=0.8, linestyle="--", label="mean")
  ax.set_xticks([0, 1])
  ax.set_xticklabels(["Same action", "Different action"])
  ax.set_ylabel("RT (seconds)")
  ax.set_title(title, fontsize=11)
  ax.legend()
  ax.grid(True, alpha=0.3, axis="y")
  fig.tight_layout()
  fig.savefig(save_path, bbox_inches="tight", dpi=300)
  plt.close(fig)
  n = len(same_vals)
  print(f"  Saved {save_path} (n={n}, same={means[0]:.4f}, diff={means[1]:.4f})")


def get_maze_setting(maze_str: str) -> str:
  if "short" in maze_str.lower():
    return "short"
  elif "long" in maze_str.lower():
    return "long"
  raise ValueError(f"Could not determine setting from maze string: {maze_str}")


def main():
  os.makedirs(OUTPUT_DIR, exist_ok=True)

  # ---- Load data ----
  global_df = pl.read_parquet(data_configs.get_dataframe_path("jaxmaze", "human"))

  # ---- Define conditions ----
  conditions = {}

  # 1. Path Reuse
  print("Filtering: Path Reuse")
  path_df, _ = filter_users_by_success(
    global_df.filter(
      manipulation="paths",
      world="big_m3_maze1",
      eval=True,
      eval_shares_start_pos=True,
      tell_reuse=1,
    ),
    analysis_name="path_reuse_results",
  )
  conditions["Path Reuse"] = analysis_utils.get_polars_df(path_df)

  # 2-3. Start condition 1 & 2
  print("Filtering: Start")
  start_df, _ = filter_users_by_success(
    global_df.filter(manipulation="start", eval=True, tell_reuse=1),
    analysis_name="start_results",
  )
  start_df = analysis_utils.get_polars_df(start_df)
  conditions["Start cond 1"] = start_df.filter(condition=1)
  conditions["Start cond 2"] = start_df.filter(condition=2)

  # 4-9. Juncture conditions x condition 1 & 2
  print("Filtering: Juncture")
  juncture_df, _ = analysis_utils.filter_users_by_success_and_tell_reuse(
    global_df.filter(manipulation="juncture"),
    analysis_name="juncture_results",
  )
  juncture_df = analysis_utils.get_polars_df(juncture_df)
  juncture_df = juncture_df.filter(manipulation="juncture", eval=True)

  juncture_df = juncture_df.with_columns(
    setting=pl.col("world").map_elements(get_maze_setting, return_dtype=pl.String)
  )

  juncture_configs = [
    ("Near+Known", "short", 1),
    ("Near+Unknown", "short", 0),
    ("Far+Known", "long", 1),
  ]
  for label, setting, tell_reuse in juncture_configs:
    sub = juncture_df.filter(setting=setting, tell_reuse=tell_reuse)
    conditions[f"Juncture {label} cond 1"] = sub.filter(condition=1)
    conditions[f"Juncture {label} cond 2"] = sub.filter(condition=2)

  # ---- Compute and plot ----
  for name, df in conditions.items():
    print(f"Computing: {name}")
    same_means, diff_means, overall_means = compute_same_diff_rt(df)

    if len(same_means) == 0:
      print(f"  Skipping {name}: no users with both same and diff timesteps")
      continue

    safe_name = name.replace(" ", "_").replace("+", "")
    overall_mean = np.mean(overall_means)

    plot_bars(
      same_means,
      diff_means,
      overall_mean,
      name,
      os.path.join(OUTPUT_DIR, f"jaxmaze_rt_by_action_{safe_name}.pdf"),
    )

  print("Done!")


if __name__ == "__main__":
  main()
