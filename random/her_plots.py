"""
python random/her_plots.py
"""

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "plots", "her_plots")
CACHE_DIR = os.path.join(SAVE_DIR, "cache")

ENTITY = "wcarvalho92"
PROJECT = "housemaze"

# Big experiment keys
TRAIN_KEY_BIG = "evaluator_performance/0.1 1.train \n her_test_big - AvgReturn"
TEST_KEY_BIG = "evaluator_performance/0.1 0.test \n her_test_big - AvgReturn"

# Small experiment keys (note: uses "her_test" not "her_test_small")
TRAIN_KEY_SMALL = "evaluator_performance/0.1 1.train \n her_test - AvgReturn"
TEST_KEY_SMALL = "evaluator_performance/0.1 0.test \n her_test - AvgReturn"

STEP_KEY = "evaluator_performance/num_actor_steps"

# All keys for fetching
KEYS_BIG = [TRAIN_KEY_BIG, TEST_KEY_BIG, STEP_KEY]
KEYS_SMALL = [TRAIN_KEY_SMALL, TEST_KEY_SMALL, STEP_KEY]
KEYS_ALL = [TRAIN_KEY_BIG, TEST_KEY_BIG, TRAIN_KEY_SMALL, TEST_KEY_SMALL, STEP_KEY]

COLORS = {
  "HER": (0.0, 114 / 255, 178 / 255),  # blue
  "HER + All-goals backup": (213 / 255, 94 / 255, 0.0),  # vermillion
}


def get_group_history(
  group,
  keys,
  name=None,
  entity=ENTITY,
  project=PROJECT,
  cache_dir=CACHE_DIR,
  window_size=20,
  refresh=False,
  seeds=None,
):
  """Fetch time-series data for a wandb group, with caching and smoothing."""
  os.makedirs(cache_dir, exist_ok=True)
  parts = [group]
  if name:
    parts.append(name.replace(",", "_").replace("=", ""))
  if seeds:
    parts.append(f"seeds{'_'.join(str(s) for s in sorted(seeds))}")
  cache_file = os.path.join(cache_dir, f"{'_'.join(parts)}_timeseries.json")

  if refresh and os.path.exists(cache_file):
    os.remove(cache_file)

  if os.path.exists(cache_file):
    print(f"Loaded {cache_file}")
    return pd.read_json(cache_file)

  print(f"Fetching runs for group={group}" + (f", name={name}" if name else ""))
  api = wandb.Api()
  filters = {"group": group}
  if name:
    filters["display_name"] = name
  runs = api.runs(f"{entity}/{project}", filters=filters)

  run_data = []
  for run in tqdm(runs, desc=f"Processing {group} runs"):
    if seeds is not None:
      seed = run.config.get("SEED")
      if seed not in seeds:
        continue
    try:
      # Use scan_history for more reliable data fetching
      history_rows = list(run.scan_history())
      if not history_rows:
        print(f"  No data for run {run.id}")
        continue
      history = pd.DataFrame(history_rows)
      # Filter to needed columns
      cols_to_keep = ["_step", "run_id"] + [k for k in keys if k in history.columns]
      history["run_id"] = run.id
      history = history[[c for c in cols_to_keep if c in history.columns]]
      run_data.append(history)
    except Exception as e:
      print(f"  Error processing run {run.id}: {e}")

  if not run_data:
    print(f"  No data collected for {group}")
    return pd.DataFrame()

  df = pd.concat(run_data, ignore_index=True)

  # Rolling-window smoothing per run
  smooth_dfs = []
  for run_id in df["run_id"].unique():
    run_slice = df[df["run_id"] == run_id].sort_values("_step")
    if len(run_slice) >= window_size:
      smooth_cols = [
        c
        for c in run_slice.select_dtypes(include=[np.number]).columns
        if c not in ("_step", STEP_KEY)
      ]
      run_slice[smooth_cols] = (
        run_slice[smooth_cols].rolling(window=window_size, min_periods=1).mean()
      )
    smooth_dfs.append(run_slice)

  df = pd.concat(smooth_dfs, ignore_index=True)
  df.to_json(cache_file)
  print(f"Saved {cache_file}")
  return df


def _plot_panel(ax, datasets, key, title, xlim=None):
  """Plot mean +/- SEM for one or more groups on a single axes."""
  for label, df in datasets.items():
    if df.empty:
      continue
    grouped = df.groupby(STEP_KEY).agg({key: ["mean", "sem"]})
    steps = grouped.index.values
    means = grouped[(key, "mean")].values
    sems = grouped[(key, "sem")].values

    color = COLORS.get(label, "gray")
    ax.plot(steps, means, label=label, color=color, linewidth=2)
    ax.fill_between(steps, means - sems, means + sems, color=color, alpha=0.3)

  ax.set_title(title, fontsize=14)
  ax.set_xlabel("Actor Steps", fontsize=12)
  ax.set_ylabel("AvgReturn", fontsize=12)
  ax.grid(True, alpha=0.3)
  ax.set_ylim(-0.2, 1.2)
  if xlim is not None:
    ax.set_xlim(xlim[0] * 1e6, xlim[1] * 1e6)
  ax.legend(fontsize=11)
  ax.xaxis.set_major_formatter(lambda x, pos: f"{x / 1e6:.0f}M")


def plot_her(
  save_dir=SAVE_DIR,
  her_group="her-exp-8",
  her_name=None,
  her_seeds=None,
  xlim=None,
  refresh=False,
  experiment="big",
  **fetch_kwargs,
):
  """1x2 plot with HER only."""
  if experiment == "big":
    keys = KEYS_BIG
    train_key, test_key = TRAIN_KEY_BIG, TEST_KEY_BIG
  else:
    keys = KEYS_SMALL
    train_key, test_key = TRAIN_KEY_SMALL, TEST_KEY_SMALL

  her_df = get_group_history(
    group=her_group,
    name=her_name,
    keys=keys,
    seeds=her_seeds,
    refresh=refresh,
    **fetch_kwargs,
  )
  datasets = {"HER": her_df}

  fig, axs = plt.subplots(1, 2, figsize=(14, 5))
  _plot_panel(axs[0], datasets, train_key, f"Train ({experiment})", xlim=xlim)
  _plot_panel(axs[1], datasets, test_key, f"Test ({experiment})", xlim=xlim)
  fig.tight_layout()

  os.makedirs(save_dir, exist_ok=True)
  path = os.path.join(save_dir, f"her_{experiment}.png")
  fig.savefig(path, bbox_inches="tight", dpi=300)
  print(f"Saved {path}")
  plt.close(fig)


def plot_her_both(
  save_dir=SAVE_DIR,
  her_group_big="her-exp-8",
  her_name_big=None,
  her_group_small="her-test-small-1",
  her_name_small=None,
  her_seeds=None,
  xlim=None,
  refresh=False,
  **fetch_kwargs,
):
  """2x2 plot with HER for both big and small experiments."""
  her_df_big = get_group_history(
    group=her_group_big,
    name=her_name_big,
    keys=KEYS_BIG,
    seeds=her_seeds,
    refresh=refresh,
    **fetch_kwargs,
  )
  her_df_small = get_group_history(
    group=her_group_small,
    name=her_name_small,
    keys=KEYS_SMALL,
    seeds=her_seeds,
    refresh=refresh,
    **fetch_kwargs,
  )

  fig, axs = plt.subplots(2, 2, figsize=(14, 10))

  # Big experiment (top row)
  _plot_panel(axs[0, 0], {"HER": her_df_big}, TRAIN_KEY_BIG, "Train (big)", xlim=xlim)
  _plot_panel(axs[0, 1], {"HER": her_df_big}, TEST_KEY_BIG, "Test (big)", xlim=xlim)

  # Small experiment (bottom row)
  _plot_panel(
    axs[1, 0], {"HER": her_df_small}, TRAIN_KEY_SMALL, "Train (small)", xlim=xlim
  )
  _plot_panel(
    axs[1, 1], {"HER": her_df_small}, TEST_KEY_SMALL, "Test (small)", xlim=xlim
  )

  fig.tight_layout()

  os.makedirs(save_dir, exist_ok=True)
  path = os.path.join(save_dir, "her_both.png")
  fig.savefig(path, bbox_inches="tight", dpi=300)
  print(f"Saved {path}")
  plt.close(fig)


def plot_her_comparison(
  save_dir=SAVE_DIR,
  her_group="her-exp-8",
  her_name=None,
  her_seeds=None,
  allgoals_group="her-debug-2",
  allgoals_name=None,
  allgoals_seeds=None,
  xlim=None,
  refresh=False,
  **fetch_kwargs,
):
  """1x2 plot comparing HER vs HER + All-goals backup."""
  her_df = get_group_history(
    group=her_group,
    name=her_name,
    keys=KEYS_BIG,
    seeds=her_seeds,
    refresh=refresh,
    **fetch_kwargs,
  )
  her_allgoals_df = get_group_history(
    group=allgoals_group,
    name=allgoals_name,
    keys=KEYS_BIG,
    seeds=allgoals_seeds,
    refresh=refresh,
    **fetch_kwargs,
  )
  datasets = {"HER": her_df, "HER + All-goals backup": her_allgoals_df}

  fig, axs = plt.subplots(1, 2, figsize=(14, 5))
  _plot_panel(axs[0], datasets, TRAIN_KEY_BIG, "Train", xlim=xlim)
  _plot_panel(axs[1], datasets, TEST_KEY_BIG, "Test", xlim=xlim)
  fig.tight_layout()

  os.makedirs(save_dir, exist_ok=True)
  path = os.path.join(save_dir, "her_big.png")
  fig.savefig(path, bbox_inches="tight", dpi=300)
  print(f"Saved {path}")
  plt.close(fig)


if __name__ == "__main__":
  # Big experiment - show HER vs HER + auxiliary task
  plot_her_comparison(
    her_group="her-exp-8",
    her_name="alg=her,num_=1,exp=her_test_big",
    her_seeds=None,
    allgoals_group="her-debug-2",
    allgoals_name="alg=her,all_=0.6,td_l=0.9,new_=False,exp=her_test_big",
    allgoals_seeds=[1, 2],
    xlim=[0, 20],
  )

  # Small experiment - single HER line
  plot_her(
    her_group="her-test-small-1",
    her_name="alg=her,tota=10000000,exp=her_test_small",
    her_seeds=None,
    xlim=[0, 10],
    experiment="small",
  )
