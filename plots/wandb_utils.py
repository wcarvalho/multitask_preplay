"""Shared WandB data fetching and plotting utilities for all plot scripts."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from tqdm.auto import tqdm

import plot_configs

ENTITY = "wcarvalho92"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "_wandb_cache")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def get_group_history(
  group,
  keys_dict,
  entity=ENTITY,
  project="craftax-multigoal",
  cache_dir=CACHE_DIR,
  window_size=20,
  refresh=False,
  filter_key="group",
  extra_filters=None,
):
  """Fetch time-series data for a wandb group, with caching and smoothing.

  Args:
    group: The group name (or display_name) to filter runs by.
    keys_dict: Dict with keys: train_key, test_key (or test_keys),
               train_step_key, test_step_key.
    entity: WandB entity.
    project: WandB project name.
    cache_dir: Directory for cached JSON files.
    window_size: Rolling window size for smoothing.
    refresh: If True, re-fetch from WandB ignoring cache.
    filter_key: WandB filter field ("group" or "display_name").
    extra_filters: Additional WandB filter dict to merge (e.g. {"group": "..."}).
  """
  os.makedirs(cache_dir, exist_ok=True)
  safe_name = group.replace("/", "_").replace(",", "_").replace("=", "_")
  cache_file = os.path.join(cache_dir, f"{project}_{safe_name}_timeseries.json")

  if refresh and os.path.exists(cache_file):
    os.remove(cache_file)

  if os.path.exists(cache_file):
    print(f"Loaded {cache_file}")
    return pd.read_json(cache_file)

  # Extract all keys to fetch
  keys = [
    keys_dict["train_step_key"],
    keys_dict["test_step_key"],
  ]
  if "train_key" in keys_dict:
    keys.append(keys_dict["train_key"])
  if "train_keys" in keys_dict:
    keys.extend([k for k, _ in keys_dict["train_keys"]])
  if "test_key" in keys_dict:
    keys.append(keys_dict["test_key"])
  if "test_keys" in keys_dict:
    keys.extend([k for k, _ in keys_dict["test_keys"]])
  keys = list(dict.fromkeys(keys))  # deduplicate, preserve order

  print(f"Fetching runs for {filter_key}={group}")
  api = wandb.Api()
  filters = {filter_key: group, "state": {"$in": ["finished", "crashed"]}}
  if extra_filters:
    filters.update(extra_filters)
  runs = api.runs(f"{entity}/{project}", filters=filters)

  run_data = []
  for run in tqdm(runs, desc=f"Processing {group} runs"):
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

  # Rolling-window smoothing per run (exclude step keys from smoothing)
  step_keys = {keys_dict["train_step_key"], keys_dict["test_step_key"]}
  smooth_dfs = []
  for run_id in df["run_id"].unique():
    run_slice = df[df["run_id"] == run_id].sort_values("_step")
    if len(run_slice) >= window_size:
      smooth_cols = [
        c
        for c in run_slice.select_dtypes(include=[np.number]).columns
        if c not in step_keys and c != "_step"
      ]
      run_slice[smooth_cols] = (
        run_slice[smooth_cols].rolling(window=window_size, min_periods=1).mean()
      )
    smooth_dfs.append(run_slice)

  df = pd.concat(smooth_dfs, ignore_index=True)
  df.to_json(cache_file)
  print(f"Saved {cache_file}")
  return df


# ---------------------------------------------------------------------------
# Shared plotting helpers
# ---------------------------------------------------------------------------


def plot_panel(
  ax,
  datasets,
  key,
  step_key,
  title,
  xlim=None,
  show_legend=False,
  show_ylabel=True,
  legend_loc="lower right",
):
  """Plot mean +/- SEM for one or more groups on a single axes."""
  TITLE_SIZE = 24
  LABEL_SIZE = 20
  TICK_SIZE = 18
  LEGEND_SIZE = 18
  LINE_WIDTH = 3

  for model_key, df in datasets.items():
    if df.empty:
      continue
    if key not in df.columns or step_key not in df.columns:
      print(f"  Warning: {model_key} missing columns {key} or {step_key}")
      continue
    valid_df = df.dropna(subset=[key, step_key])
    if valid_df.empty:
      continue
    grouped = valid_df.groupby(step_key).agg({key: ["mean", "sem"]})
    steps = grouped.index.values
    means = grouped[(key, "mean")].values
    sems = grouped[(key, "sem")].values

    color = plot_configs.model_colors.get(model_key, "gray")
    label = plot_configs.model_names.get(model_key, model_key)
    ax.plot(steps, means, label=label, color=color, linewidth=LINE_WIDTH)
    ax.fill_between(steps, means - sems, means + sems, color=color, alpha=0.25)

  ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
  ax.set_xlabel("Environment Steps", fontsize=LABEL_SIZE)
  if show_ylabel:
    ax.set_ylabel("Success Rate", fontsize=LABEL_SIZE)
  ax.tick_params(axis="both", labelsize=TICK_SIZE)
  ax.grid(True, alpha=0.3, linewidth=0.5)
  if xlim is not None:
    ax.set_xlim(xlim[0], xlim[1])
  ax.set_ylim(0, 1.1)
  ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
  if show_legend:
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9, loc=legend_loc)
  ax.xaxis.set_major_formatter(lambda x, pos: f"{x / 1e6:.0f}M")


def plot_bar_panel(
  ax,
  datasets,
  key,
  step_key,
  title,
  show_legend=False,
  show_ylabel=True,
  legend_loc="upper right",
):
  """Plot bar chart of best (peak) mean performance for each model."""
  TITLE_SIZE = 24
  LABEL_SIZE = 20
  TICK_SIZE = 18
  LEGEND_SIZE = 18
  LINE_WIDTH = 2

  ordered_keys = [k for k in plot_configs.model_order if k in datasets]
  ordered_keys += [k for k in datasets if k not in ordered_keys]

  labels = []
  means = []
  sems = []
  colors = []

  for model_key in ordered_keys:
    df = datasets[model_key]
    if df.empty:
      continue
    if key not in df.columns or step_key not in df.columns:
      print(f"  Warning: {model_key} missing columns {key} or {step_key}")
      continue
    valid_df = df.dropna(subset=[key, step_key])
    if valid_df.empty:
      continue

    grouped = valid_df.groupby(step_key).agg({key: ["mean", "sem"]})
    best_idx = grouped[(key, "mean")].idxmax()
    best_mean = grouped.loc[best_idx, (key, "mean")]
    best_sem = grouped.loc[best_idx, (key, "sem")]

    labels.append(plot_configs.model_names.get(model_key, model_key))
    means.append(best_mean)
    sems.append(best_sem)
    colors.append(plot_configs.model_colors.get(model_key, "gray"))

  if not labels:
    return

  x = np.arange(len(labels))
  bars = ax.bar(
    x,
    means,
    yerr=sems,
    capsize=5,
    color=colors,
    edgecolor="black",
    linewidth=LINE_WIDTH,
    width=0.6,
  )
  for bar, label in zip(bars, labels):
    bar.set_label(label)

  ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
  ax.set_xticks(x)
  ax.set_xticklabels([""] * len(labels))
  if show_legend and len(labels) > 1:
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9, loc=legend_loc)
  if show_ylabel:
    ax.set_ylabel("Success Rate", fontsize=LABEL_SIZE)
  ax.tick_params(axis="y", labelsize=TICK_SIZE)
  ax.grid(True, alpha=0.3, linewidth=0.5, axis="y")
  ax.set_ylim(0, 1.1)
  ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))


def plot_train_test(
  project,
  groups,
  keys_dict,
  save_dir=OUTPUT_DIR,
  xlim=None,
  refresh=False,
  output_name=None,
  bar_plot=True,
  legend_ax=0,
  legend_loc=None,
):
  """Plot Train and Test learning curves. Supports multiple test panels."""
  if xlim is None:
    xlim = [0, 40e6]

  datasets = {}
  for model_key, raw_name in groups.items():
    # Strip "Group: " or "Name: " prefixes from config values
    group_name = raw_name.split(": ", 1)[1] if ": " in raw_name else raw_name
    df = get_group_history(
      group=group_name,
      keys_dict=keys_dict,
      project=project,
      refresh=refresh,
    )
    datasets[model_key] = df

  train_step_key = keys_dict["train_step_key"]
  test_step_key = keys_dict["test_step_key"]

  # Build train panels
  if "train_keys" in keys_dict:
    train_panels = [(k, label, train_step_key) for k, label in keys_dict["train_keys"]]
  else:
    train_panels = [(keys_dict["train_key"], "Train", train_step_key)]

  # Build test panels
  if "test_keys" in keys_dict:
    test_panels = [(k, label, test_step_key) for k, label in keys_dict["test_keys"]]
  else:
    test_panels = [(keys_dict["test_key"], "Test", test_step_key)]

  # Layout: if both train and test have multiple panels, use grid
  # Columns = [Train, Test], Rows = environments
  n_rows = max(len(train_panels), len(test_panels))
  if n_rows > 1:
    n_cols = 2  # Train column, Test column
    fig, axs = plt.subplots(
      n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows), sharey=True
    )
    if n_rows == 1:
      axs = axs.reshape(1, -1)
  else:
    n_cols_flat = len(train_panels) + len(test_panels)
    fig, axs_flat = plt.subplots(
      1, n_cols_flat, figsize=(8 * n_cols_flat, 6), sharey=True
    )
    if n_cols_flat == 1:
      axs_flat = [axs_flat]

  plot_fn = plot_bar_panel if bar_plot else plot_panel
  panel_idx = 0

  if n_rows > 1:
    # Grid: rows = environments, col 0 = Train, col 1 = Test
    for row in range(n_rows):
      for col, panels in enumerate([train_panels, test_panels]):
        if row < len(panels):
          k, label, sk = panels[row]
          ax = axs[row, col]
          kwargs = {} if bar_plot else {"xlim": xlim}
          legend_kwargs = {"legend_loc": legend_loc} if legend_loc else {}
          plot_fn(
            ax,
            datasets,
            k,
            sk,
            label,
            show_ylabel=(col == 0),
            show_legend=(
              (row, col) == legend_ax
              if isinstance(legend_ax, tuple)
              else row * 2 + col == legend_ax
            ),
            **kwargs,
            **legend_kwargs,
          )
        else:
          axs[row, col].set_visible(False)
  else:
    # Single row: train panels then test panels
    all_panels = train_panels + test_panels
    for i, (k, label, sk) in enumerate(all_panels):
      kwargs = {} if bar_plot else {"xlim": xlim}
      legend_kwargs = {"legend_loc": legend_loc} if legend_loc else {}
      plot_fn(
        axs_flat[i],
        datasets,
        k,
        sk,
        label,
        show_ylabel=(i == 0),
        show_legend=(i == legend_ax),
        **legend_kwargs,
        **kwargs,
      )

  fig.tight_layout()

  os.makedirs(save_dir, exist_ok=True)
  filename = output_name or f"train_test_{project.replace('-', '_')}.pdf"
  path = os.path.join(save_dir, filename)
  fig.savefig(path, bbox_inches="tight", dpi=300)
  print(f"Saved {path}")
  plt.close(fig)
