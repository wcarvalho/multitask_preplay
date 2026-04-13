"""
python random/preplay_conservative_ablation.py
"""

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "plots", "preplay_ablation")
CACHE_DIR = os.path.join(SAVE_DIR, "cache")

ENTITY = "wcarvalho92"
PROJECT = "housemaze"

TRAIN_KEY = "evaluator_performance/0.1 1.train \n two_paths - AvgReturn"
TEST_KEY = "evaluator_performance/0.1 0.test \n two_paths - AvgReturn"
STEP_KEY = "evaluator_performance/num_actor_steps"
KEYS = [TRAIN_KEY, TEST_KEY, STEP_KEY]

# Groups and labels
GROUPS = {
  "preplay-pnas-revision-5": "Multitask Preplay",
  "preplay-peng-ablation-pnas-revision-2": "w/o Off-task Q(λ)",
  "preplay-cql-ablation-pnas-revision-2": "w/o Conservative reg.",
  "preplay-all-goals-ablation-pnas-revision-2": "w/o All-goals learning",
}

COLORS = {
  "Multitask Preplay": (213 / 255, 94 / 255, 0.0),  # vermillion
  "w/o Off-task Q(λ)": (0.0, 114 / 255, 178 / 255),  # blue
  "w/o Conservative reg.": (0.0, 158 / 255, 115 / 255),  # bluish green
  "w/o All-goals learning": "#666666",  # dark gray
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
      history_rows = list(run.scan_history())
      if not history_rows:
        print(f"  No data for run {run.id}")
        continue
      history = pd.DataFrame(history_rows)
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
  ax.legend(fontsize=10)
  ax.xaxis.set_major_formatter(lambda x, pos: f"{x / 1e6:.0f}M")


def plot_lines(datasets, save_dir=SAVE_DIR, xlim=None):
  """1x2 line plot: training curves for train and test."""
  fig, axs = plt.subplots(1, 2, figsize=(14, 5))
  _plot_panel(axs[0], datasets, TRAIN_KEY, "Train (two_paths)", xlim=xlim)
  _plot_panel(axs[1], datasets, TEST_KEY, "Test (two_paths)", xlim=xlim)
  fig.tight_layout()

  os.makedirs(save_dir, exist_ok=True)
  path = os.path.join(save_dir, "ablation_lines.png")
  fig.savefig(path, bbox_inches="tight", dpi=300)
  print(f"Saved {path}")
  plt.close(fig)


def plot_bars(datasets, save_dir=SAVE_DIR):
  """1x2 bar plot: final performance (last timestep per run) for train and test."""
  labels = []
  train_means = []
  train_sems = []
  test_means = []
  test_sems = []

  for label, df in datasets.items():
    if df.empty:
      continue
    labels.append(label)

    # Take the final timestep per run
    run_finals_train = []
    run_finals_test = []
    for run_id in df["run_id"].unique():
      run_slice = df[df["run_id"] == run_id].sort_values("_step")
      last_row = run_slice.iloc[-1]
      if TRAIN_KEY in run_slice.columns:
        run_finals_train.append(last_row[TRAIN_KEY])
      if TEST_KEY in run_slice.columns:
        run_finals_test.append(last_row[TEST_KEY])

    run_finals_train = np.array(run_finals_train)
    run_finals_test = np.array(run_finals_test)

    train_means.append(run_finals_train.mean())
    train_sems.append(
      run_finals_train.std() / np.sqrt(len(run_finals_train))
      if len(run_finals_train) > 1
      else 0
    )
    test_means.append(run_finals_test.mean())
    test_sems.append(
      run_finals_test.std() / np.sqrt(len(run_finals_test))
      if len(run_finals_test) > 1
      else 0
    )

  x = np.arange(len(labels))
  bar_width = 0.6
  colors = [COLORS.get(l, "gray") for l in labels]

  fig, axs = plt.subplots(1, 2, figsize=(14, 5))
  for ax, means, sems, title in [
    (axs[0], train_means, train_sems, "Train (two_paths)"),
    (axs[1], test_means, test_sems, "Test (two_paths)"),
  ]:
    bars = ax.bar(x, means, bar_width, yerr=sems, capsize=5, color=colors)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("AvgReturn", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.grid(True, axis="y", alpha=0.3)

  fig.tight_layout()

  os.makedirs(save_dir, exist_ok=True)
  path = os.path.join(save_dir, "ablation_bars.png")
  fig.savefig(path, bbox_inches="tight", dpi=300)
  print(f"Saved {path}")
  plt.close(fig)


if __name__ == "__main__":
  # Fetch all groups
  datasets = {}
  for group, label in GROUPS.items():
    df = get_group_history(group=group, keys=KEYS)
    datasets[label] = df

  # Line plots (training curves)
  plot_lines(datasets, xlim=[0, 20])

  # Bar plots (final performance)
  plot_bars(datasets)
