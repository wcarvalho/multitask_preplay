"""Max RT position analysis across conditions.

For each person, compute the percentage of episodes where the maximum reaction time
occurs within the first N timesteps. Thresholds: 1, 5, 10, 20.

Conditions are merged (not split by sub-condition):
- Path Reuse: all path reuse data
- Start: all start manipulation data (conditions merged)
- Juncture: all juncture manipulation data (settings/tell_reuse/conditions merged)
- All: combined data from all three conditions
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


def get_max_rt_position_stats(condition_df, thresholds=(1, 5, 10, 20)):
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


def main():
  # ---- Load data ----
  global_df = pl.read_parquet(data_configs.get_dataframe_path("jaxmaze", "human"))

  # ---- Define conditions (merged) ----
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

  # 2. Start (merged - all conditions combined)
  print("Filtering: Start")
  start_df, _ = filter_users_by_success(
    global_df.filter(manipulation="start", eval=True, tell_reuse=1),
    analysis_name="start_results",
  )
  conditions["Start"] = analysis_utils.get_polars_df(start_df)

  # 3. Juncture (merged - all settings/tell_reuse/conditions combined)
  print("Filtering: Juncture")
  juncture_df, _ = analysis_utils.filter_users_by_success_and_tell_reuse(
    global_df.filter(manipulation="juncture"),
    analysis_name="juncture_results",
  )
  juncture_df = analysis_utils.get_polars_df(juncture_df)
  conditions["Juncture"] = juncture_df.filter(manipulation="juncture", eval=True)

  # 4. All - combine all three datasets
  print("Combining: All")
  conditions["All"] = pl.concat(
    [conditions["Path Reuse"], conditions["Start"], conditions["Juncture"]]
  )

  # ---- Output directory ----
  output_dir = "random/plots/max_rt_position"
  os.makedirs(output_dir, exist_ok=True)

  # ---- Compute stats ----
  thresholds = (1, 5, 10, 20)
  position_results = {}
  for name, df in conditions.items():
    print(f"Computing max RT position: {name}")
    position_results[name] = get_max_rt_position_stats(df, thresholds=thresholds)

  # ---- Create bar labels ----
  bar_labels = [
    f"{'1st' if t == 1 else f'First {t}'}\ntimestep{'s' if t > 1 else ''}"
    for t in thresholds
  ]
  bar_x = np.arange(len(bar_labels))
  colors = ["steelblue", "darkorange", "seagreen", "mediumpurple"]

  # ---- Per-condition plots ----
  for name, (means, sems) in position_results.items():
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
      bar_x,
      means * 100,
      yerr=sems * 100,
      capsize=4,
      color=colors,
      edgecolor="white",
    )
    ax.set_title(name, fontsize=12)
    ax.set_xticks(bar_x)
    ax.set_xticklabels(bar_labels)
    ax.set_ylabel("% of episodes")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    safe_name = name.replace(" ", "_")
    fig.savefig(f"{output_dir}/{safe_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output_dir}/{safe_name}.png")

  print("Done!")


if __name__ == "__main__":
  main()
