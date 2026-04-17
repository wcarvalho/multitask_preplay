"""Top-10 demeaned RT analysis across conditions.

For each person, pool all per-timestep RTs across eval episodes in a condition,
sort descending, take top 10, subtract that person's mean RT. Then compute
mean ± SEM across people for each of the 10 ranked positions.

Two plot types:
- difference/: demeaned top-10 (sorted RTs minus person mean)
- relative/: successive differences (RT[i] - RT[i+1] for sorted RTs)

Start and Juncture are split by condition (1 vs 2).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import polars as pl

import data_configs
from analysis.jaxmaze_analysis import filter_users_by_success
from analysis import analysis_utils


def parse_reaction_times(rt_string):
  rt_string = rt_string.strip("[]")
  return np.array([float(x) for x in rt_string.split()])


def get_top10_demeaned(condition_df):
  """For each user, get top-10 slowest timestep RTs, demeaned by user mean."""
  user_ids = condition_df["user_id"].unique().to_list()
  all_top10 = []

  for uid in user_ids:
    user_rows = condition_df.filter(user_id=uid)
    all_rts = []
    for row in user_rows.iter_rows(named=True):
      rts = parse_reaction_times(row["reaction_times"])
      all_rts.extend(rts.tolist())

    if len(all_rts) < 10:
      continue

    all_rts = np.array(all_rts)
    user_mean = np.mean(all_rts)

    sorted_rts = np.sort(all_rts)[::-1][:10]
    demeaned = sorted_rts - user_mean
    all_top10.append(demeaned)

  all_top10 = np.array(all_top10)
  means = np.mean(all_top10, axis=0)
  sems = np.std(all_top10, axis=0, ddof=1) / np.sqrt(len(all_top10))
  print(f"  {len(all_top10)} users with >=10 timesteps")
  return means, sems


def get_max_rt_position_stats(condition_df, thresholds=(1, 10)):
  """For each user, compute % of episodes where max RT is in the first N timesteps."""
  user_ids = condition_df["user_id"].unique().to_list()
  user_pcts = {t: [] for t in thresholds}

  for uid in user_ids:
    user_rows = condition_df.filter(user_id=uid)
    n_episodes = 0
    counts = {t: 0 for t in thresholds}
    for row in user_rows.iter_rows(named=True):
      rts = parse_reaction_times(row["reaction_times"])
      if len(rts) == 0:
        continue
      n_episodes += 1
      max_idx = np.argmax(rts)
      for t in thresholds:
        if max_idx < t:
          counts[t] += 1

    if n_episodes == 0:
      continue
    for t in thresholds:
      user_pcts[t].append(counts[t] / n_episodes)

  arrays = {t: np.array(user_pcts[t]) for t in thresholds}
  means = np.array([np.mean(arrays[t]) for t in thresholds])
  sems = np.array(
    [np.std(arrays[t], ddof=1) / np.sqrt(len(arrays[t])) for t in thresholds]
  )
  print(f"  {len(next(iter(arrays.values())))} users")
  return means, sems


def get_top10_relative(condition_df):
  """For each user, get successive differences of top-10 sorted RTs.

  Value at index i = RT[i] - RT[i+1] for sorted descending RTs.
  No demeaning. Returns 9 values (differences between 10 ranks).
  """
  user_ids = condition_df["user_id"].unique().to_list()
  all_diffs = []

  for uid in user_ids:
    user_rows = condition_df.filter(user_id=uid)
    all_rts = []
    for row in user_rows.iter_rows(named=True):
      rts = parse_reaction_times(row["reaction_times"])
      all_rts.extend(rts.tolist())

    if len(all_rts) < 10:
      continue

    all_rts = np.array(all_rts)
    sorted_rts = np.sort(all_rts)[::-1][:10]
    diffs = sorted_rts[:-1] - sorted_rts[1:]
    all_diffs.append(diffs)

  all_diffs = np.array(all_diffs)
  means = np.mean(all_diffs, axis=0)
  sems = np.std(all_diffs, axis=0, ddof=1) / np.sqrt(len(all_diffs))
  print(f"  {len(all_diffs)} users with >=10 timesteps")
  return means, sems


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
conditions["Start\ncond 1"] = start_df.filter(condition=1)
conditions["Start\ncond 2"] = start_df.filter(condition=2)

# 4-9. Juncture conditions × condition 1 & 2
print("Filtering: Juncture")
juncture_df, _ = analysis_utils.filter_users_by_success_and_tell_reuse(
  global_df.filter(manipulation="juncture"),
  analysis_name="juncture_results",
)
juncture_df = analysis_utils.get_polars_df(juncture_df)
juncture_df = juncture_df.filter(manipulation="juncture", eval=True)


def get_maze_setting(maze_str: str) -> str:
  if "short" in maze_str.lower():
    return "short"
  elif "long" in maze_str.lower():
    return "long"
  raise ValueError(f"Could not determine setting from maze string: {maze_str}")


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
  conditions[f"Juncture\n{label}\ncond 1"] = sub.filter(condition=1)
  conditions[f"Juncture\n{label}\ncond 2"] = sub.filter(condition=2)

# ---- Compute and plot ----
os.makedirs("random/plots/top10/difference", exist_ok=True)
os.makedirs("random/plots/top10/relative", exist_ok=True)

# --- Combined figure: difference ---
n_conditions = len(conditions)
fig, axes = plt.subplots(1, n_conditions, figsize=(3 * n_conditions, 4), sharey=True)
ranks = np.arange(1, 11)

difference_results = {}
for name, df in conditions.items():
  print(f"Computing difference: {name.replace(chr(10), ' ')}")
  difference_results[name] = get_top10_demeaned(df)

for ax, (name, (means, sems)) in zip(axes, difference_results.items()):
  ax.bar(ranks, means, yerr=sems, capsize=3, color="steelblue", edgecolor="white")
  ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
  # Add horizontal line at average person's SEM
  avg_sem = np.mean(sems)
  ax.axhline(avg_sem, color="red", linewidth=0.8, linestyle=":", alpha=0.7)
  ax.set_title(name, fontsize=10)
  ax.set_xlabel("Rank (1=slowest)")
  ax.set_xticks(ranks)

axes[0].set_ylabel("Demeaned RT (seconds)")
fig.suptitle("Top-10 Slowest Timestep RTs (Demeaned by Person)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(
  "random/plots/top10/difference/all_conditions.png", dpi=150, bbox_inches="tight"
)
print("Saved to random/plots/top10/difference/all_conditions.png")
plt.close(fig)

# Per-condition difference plots
for name, (means, sems) in difference_results.items():
  fig, ax = plt.subplots(figsize=(5, 4))
  ax.bar(ranks, means, yerr=sems, capsize=3, color="steelblue", edgecolor="white")
  ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
  avg_sem = np.mean(sems)
  ax.axhline(avg_sem, color="red", linewidth=0.8, linestyle=":", alpha=0.7)
  ax.set_title(name, fontsize=12)
  ax.set_xlabel("Rank (1=slowest)")
  ax.set_ylabel("Demeaned RT (seconds)")
  ax.set_xticks(ranks)
  fig.tight_layout()
  safe_name = name.replace("\n", "_").replace(" ", "_").replace("+", "")
  fig.savefig(
    f"random/plots/top10/difference/{safe_name}.png", dpi=150, bbox_inches="tight"
  )
  plt.close(fig)

# --- Combined figure: relative ---
fig, axes = plt.subplots(1, n_conditions, figsize=(3 * n_conditions, 4), sharey=True)
rel_ranks = np.arange(1, 10)

relative_results = {}
for name, df in conditions.items():
  print(f"Computing relative: {name.replace(chr(10), ' ')}")
  relative_results[name] = get_top10_relative(df)

for ax, (name, (means, sems)) in zip(axes, relative_results.items()):
  ax.bar(rel_ranks, means, yerr=sems, capsize=3, color="darkorange", edgecolor="white")
  ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
  ax.set_title(name, fontsize=10)
  ax.set_xlabel("Gap (1→2, 2→3, ...)")
  ax.set_xticks(rel_ranks)

axes[0].set_ylabel("RT[i] - RT[i+1] (seconds)")
fig.suptitle("Successive Differences of Top-10 Sorted RTs", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(
  "random/plots/top10/relative/all_conditions.png", dpi=150, bbox_inches="tight"
)
print("Saved to random/plots/top10/relative/all_conditions.png")
plt.close(fig)

# Per-condition relative plots
for name, (means, sems) in relative_results.items():
  fig, ax = plt.subplots(figsize=(5, 4))
  ax.bar(rel_ranks, means, yerr=sems, capsize=3, color="darkorange", edgecolor="white")
  ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
  ax.set_title(name, fontsize=12)
  ax.set_xlabel("Gap (1→2, 2→3, ...)")
  ax.set_ylabel("RT[i] - RT[i+1] (seconds)")
  ax.set_xticks(rel_ranks)
  fig.tight_layout()
  safe_name = name.replace("\n", "_").replace(" ", "_").replace("+", "")
  fig.savefig(
    f"random/plots/top10/relative/{safe_name}.png", dpi=150, bbox_inches="tight"
  )
  plt.close(fig)

# --- Combined figure: max RT position ---
os.makedirs("random/plots/top10/max_rt_position", exist_ok=True)

fig, axes = plt.subplots(1, n_conditions, figsize=(3 * n_conditions, 4), sharey=True)

max_rt_thresholds = (1, 10)
position_results = {}
for name, df in conditions.items():
  print(f"Computing max RT position: {name.replace(chr(10), ' ')}")
  position_results[name] = get_max_rt_position_stats(df, thresholds=max_rt_thresholds)

bar_labels = [
  f"{'1st' if t == 1 else f'First {t}'}\ntimestep{'s' if t > 1 else ''}"
  for t in max_rt_thresholds
]
bar_x = np.arange(len(bar_labels))
for ax, (name, (means, sems)) in zip(axes, position_results.items()):
  ax.bar(
    bar_x,
    means * 100,
    yerr=sems * 100,
    capsize=4,
    color=["steelblue", "darkorange"],
    edgecolor="white",
  )
  ax.set_title(name, fontsize=10)
  ax.set_xticks(bar_x)
  ax.set_xticklabels(bar_labels)
  ax.set_ylim(0, 100)

axes[0].set_ylabel("% of episodes")
fig.suptitle("% of Episodes Where Max RT Is on...", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(
  "random/plots/top10/max_rt_position/all_conditions.png", dpi=150, bbox_inches="tight"
)
print("Saved to random/plots/top10/max_rt_position/all_conditions.png")
plt.close(fig)

# Per-condition max RT position plots
for name, (means, sems) in position_results.items():
  fig, ax = plt.subplots(figsize=(4, 4))
  ax.bar(
    bar_x,
    means * 100,
    yerr=sems * 100,
    capsize=4,
    color=["steelblue", "darkorange"],
    edgecolor="white",
  )
  ax.set_title(name, fontsize=12)
  ax.set_xticks(bar_x)
  ax.set_xticklabels(bar_labels)
  ax.set_ylabel("% of episodes")
  ax.set_ylim(0, 100)
  fig.tight_layout()
  safe_name = name.replace("\n", "_").replace(" ", "_").replace("+", "")
  fig.savefig(
    f"random/plots/top10/max_rt_position/{safe_name}.png", dpi=150, bbox_inches="tight"
  )
  plt.close(fig)

print("Done!")
