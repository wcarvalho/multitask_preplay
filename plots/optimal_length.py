"""Bar plot of average deviation from optimal path length per agent.

JaxMaze: BFS serves as the optimal baseline (training episodes).
Craftax: A* optimal path lengths are precomputed (eval episodes).

Migrated from random/optimal_length_deviation.py.

Usage:
    python plots/optimal_length.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Repo-root imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.get_experiment_data import get_experiment_data
from analysis.vis_utils import visualize_jaxmaze_row
import data_configs

import plot_configs  # noqa: F401 — from same directory
from plot_configs import model_colors, model_names, model_order

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
TARGET_BLOCK_NAME = "reverse(Y=False,X=False)"


# ---------------------------------------------------------------------------
# JaxMaze helpers
# ---------------------------------------------------------------------------


def get_jaxmaze_train_data(user_df, model_df, experiment):
  """Get filtered human and model *training* data for a JaxMaze experiment."""
  data = get_experiment_data(
    f"jaxmaze_{experiment}", user_df=user_df, model_df=model_df
  )
  user_ids = data["user_ids"]

  if experiment == "path_reuse":
    train_filters = dict(manipulation="paths", world="big_m3_maze1")
  elif experiment == "shortcut":
    train_filters = dict(manipulation="shortcut", world="big_m1_maze3")
  else:
    raise ValueError(f"Unknown JaxMaze experiment: {experiment}")

  human_train = user_df.filter(
    pl.col("user_id").is_in(user_ids),
    eval=False,
    success=1,
    shares_start_with_eval=True,
    task_set=0,
    **train_filters,
  )

  model_train = model_df.filter(
    eval=False,
    success=1,
    task_set=0,
    **train_filters,
  )

  return human_train, model_train, user_ids


def get_jaxmaze_eval_data(user_df, model_df, experiment, user_ids):
  """Get filtered human and model *eval* data for a JaxMaze experiment."""
  if experiment == "path_reuse":
    eval_filters = dict(manipulation="paths", world="big_m3_maze1")
  elif experiment == "shortcut":
    # Eval world has a different name with the shortcut modification
    eval_filters = dict(manipulation="shortcut", world="big_m1_maze3_shortcut")
  else:
    raise ValueError(f"Unknown JaxMaze experiment: {experiment}")

  human_eval = user_df.filter(
    pl.col("user_id").is_in(user_ids),
    eval=True,
    success=1,
    task_set=0,
    **eval_filters,
  )

  model_eval = model_df.filter(
    eval=True,
    success=1,
    task_set=0,
    **eval_filters,
  )

  return human_eval, model_eval


def _compute_jaxmaze_stats(human_df, model_df, bfs_optimal):
  """Compute deviation stats for JaxMaze human and model data."""
  agent_stats = {}

  # Human: deviation per episode, mean per user, then mean/SEM across users
  if len(human_df) > 0:
    human_dev = human_df.join(bfs_optimal, on=["world", "block_name"]).with_columns(
      (pl.col("path_length") - pl.col("optimal_path_length")).alias("deviation")
    )
    user_means = human_dev.group_by("user_id").agg(pl.col("deviation").mean())[
      "deviation"
    ]
    agent_stats["human"] = (
      user_means.mean(),
      user_means.std() / np.sqrt(len(user_means)),
    )

  # Models: deviation per episode, then mean/SEM
  if len(model_df) > 0:
    model_dev = model_df.join(bfs_optimal, on=["world", "block_name"]).with_columns(
      (pl.col("path_length") - pl.col("optimal_path_length")).alias("deviation")
    )
    for algo in model_dev["algo"].unique().sort().to_list():
      devs = model_dev.filter(algo=algo)["deviation"]
      agent_stats[algo] = (
        devs.mean(),
        devs.std() / np.sqrt(len(devs)),
      )

  return agent_stats


def make_jaxmaze_plot(
  experiment,
  human_train,
  model_train,
  human_eval,
  model_eval,
  bfs_optimal_train,
  bfs_optimal_eval,
):
  """Create bar plot of average deviation from BFS optimal path length."""
  train_stats = _compute_jaxmaze_stats(human_train, model_train, bfs_optimal_train)
  test_stats = _compute_jaxmaze_stats(human_eval, model_eval, bfs_optimal_eval)

  if not train_stats and not test_stats:
    print(f"  No data found for {experiment}, skipping.")
    return

  _save_bar_plot(train_stats, test_stats, f"jaxmaze_{experiment}")


def compute_bfs_optimal(model_df, is_eval=False):
  """Compute mean BFS path_length per (world, block_name) for task_set=0."""
  label = "eval" if is_eval else "train"
  bfs_data = model_df.filter(
    algo="bfs",
    block_name=TARGET_BLOCK_NAME,
    task_set=0,
    eval=is_eval,
  )
  optimal = bfs_data.group_by("world", "block_name").agg(
    pl.col("path_length").mean().alias("optimal_path_length")
  )
  print(f"  BFS optimal ({label}) rows: {len(optimal)}")
  for row in optimal.iter_rows(named=True):
    print(
      f"    world={row['world']}, block={row['block_name']}, "
      f"optimal_path_length={row['optimal_path_length']:.2f}"
    )
  return optimal


def find_anomalous_episodes(human_train, bfs_optimal):
  """Find human training episodes shorter than BFS optimal."""
  joined = human_train.join(bfs_optimal, on=["world", "block_name"])
  anomalous = joined.filter(pl.col("path_length") < pl.col("optimal_path_length"))
  print(
    f"  Human training episodes (block={TARGET_BLOCK_NAME}, task_set=0): {len(human_train)}"
  )
  print(f"  Anomalous (path_length < BFS optimal): {len(anomalous)}")
  return anomalous


def visualize_anomalous_episodes(anomalous_df, experiment):
  """Save visualizations of anomalous episodes."""
  out_dir = os.path.join(OUTPUT_DIR, f"optimal_length_{experiment}_anomalous")
  os.makedirs(out_dir, exist_ok=True)
  for i, row in enumerate(anomalous_df.iter_rows(named=True)):
    user_id = row["user_id"]
    path_len = row["path_length"]
    title = f"user={user_id}, path_length={path_len}"
    fig, _ = visualize_jaxmaze_row(row, title=title)
    save_path = os.path.join(out_dir, f"anomalous_{i:03d}_user={user_id}.pdf")
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
  print(f"  Saved {len(anomalous_df)} anomalous episode images to {out_dir}/")


def save_example_train_paths(human_train, model_train, experiment):
  """Save one example training path per agent (human + each model algo)."""
  out_dir = os.path.join(OUTPUT_DIR, f"optimal_length_{experiment}_example_train")
  os.makedirs(out_dir, exist_ok=True)

  # Human example
  if len(human_train) > 0:
    row = human_train.row(0, named=True)
    title = f"human, success={row['success']}, path_length={row['path_length']}"
    fig, _ = visualize_jaxmaze_row(row, title=title)
    fig.savefig(os.path.join(out_dir, "human.pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved human example to {out_dir}/human.pdf")

  # Model examples
  for algo in model_train["algo"].unique().sort().to_list():
    algo_df = model_train.filter(algo=algo)
    if len(algo_df) == 0:
      continue
    row = algo_df.row(0, named=True)
    title = f"{algo}, success={row['success']}, path_length={row['path_length']}"
    fig, _ = visualize_jaxmaze_row(row, title=title)
    fig.savefig(os.path.join(out_dir, f"{algo}.pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {algo} example to {out_dir}/{algo}.pdf")


# ---------------------------------------------------------------------------
# Craftax helpers
# ---------------------------------------------------------------------------


def _compute_craftax_stats(human_df, model_df, is_eval=True):
  """Compute deviation stats for Craftax human and model data."""
  from data_processing.utils_craftax import OPTIMAL_TEST_LENGTHS

  agent_stats = {}

  # Only eval has precomputed optimal lengths; skip train deviation
  if not is_eval:
    return agent_stats

  optimal_lengths = OPTIMAL_TEST_LENGTHS

  # --- Human ---
  if human_df is not None and len(human_df) > 0:
    if "path_length_deviance" in human_df.columns:
      # Human df already has path_length_deviance from data processing
      user_means = human_df.group_by("user_id").agg(
        pl.col("path_length_deviance").mean()
      )["path_length_deviance"]
    else:
      # Compute deviance manually (join on world column as string)
      optimal_rows = [
        {"world": str(ws), "optimal_length": ol} for ws, ol in optimal_lengths.items()
      ]
      optimal_df = pl.DataFrame(optimal_rows)
      human_with_opt = human_df.join(optimal_df, on="world")
      human_with_opt = human_with_opt.with_columns(
        (pl.col("path_length") - pl.col("optimal_length"))
        .clip(lower_bound=0)
        .alias("path_length_deviance")
      )
      user_means = human_with_opt.group_by("user_id").agg(
        pl.col("path_length_deviance").mean()
      )["path_length_deviance"]
    agent_stats["human"] = (
      user_means.mean(),
      user_means.std() / np.sqrt(len(user_means)),
    )

  # --- Models ---
  if model_df is not None and len(model_df) > 0:
    # Add optimal_length to model data via world join (world is string)
    optimal_rows = [
      {"world": str(ws), "optimal_length": ol} for ws, ol in optimal_lengths.items()
    ]
    optimal_df = pl.DataFrame(optimal_rows)
    model_with_opt = model_df.join(optimal_df, on="world")
    model_with_opt = model_with_opt.with_columns(
      (pl.col("path_length") - pl.col("optimal_length"))
      .clip(lower_bound=0)
      .alias("path_length_deviance")
    )

    for algo in model_with_opt["algo"].unique().sort().to_list():
      devs = model_with_opt.filter(algo=algo)["path_length_deviance"]
      agent_stats[algo] = (
        devs.mean(),
        devs.std() / np.sqrt(len(devs)),
      )

  return agent_stats


def make_craftax_plot(
  human_train, model_train, human_eval, model_eval, experiment_label
):
  """Bar plot of avg deviation from A* optimal, split by agent.

  Both DataFrames should be filtered to ``success=1``.
  """
  train_stats = _compute_craftax_stats(human_train, model_train, is_eval=False)
  test_stats = _compute_craftax_stats(human_eval, model_eval, is_eval=True)

  if not train_stats and not test_stats:
    print(f"  No data found for {experiment_label}, skipping.")
    return

  _save_bar_plot(train_stats, test_stats, experiment_label)


# ---------------------------------------------------------------------------
# Shared plotting
# ---------------------------------------------------------------------------


def _save_bar_plot(train_stats, test_stats, experiment_label):
  """Render and save a deviation bar plot with train/test side-by-side."""
  from matplotlib.patches import Patch

  # Get ordered list of agents present in either train or test
  all_agents = set(train_stats.keys()) | set(test_stats.keys())
  ordered_agents = [a for a in model_order if a in all_agents]
  for a in all_agents:
    if a not in ordered_agents:
      ordered_agents.append(a)

  colors = [model_colors.get(a, "gray") for a in ordered_agents]
  labels = [model_names.get(a, a) for a in ordered_agents]

  # Extract train and test means/SEMs (use NaN for missing)
  train_means = [train_stats.get(a, (np.nan, np.nan))[0] for a in ordered_agents]
  train_sems = [train_stats.get(a, (np.nan, np.nan))[1] for a in ordered_agents]
  test_means = [test_stats.get(a, (np.nan, np.nan))[0] for a in ordered_agents]
  test_sems = [test_stats.get(a, (np.nan, np.nan))[1] for a in ordered_agents]

  print("  Train stats:")
  for agent, m, s in zip(ordered_agents, train_means, train_sems):
    if not np.isnan(m):
      print(f"    {agent}: avg deviation = {m:.2f} +/- {s:.2f}")
  print("  Test stats:")
  for agent, m, s in zip(ordered_agents, test_means, test_sems):
    if not np.isnan(m):
      print(f"    {agent}: avg deviation = {m:.2f} +/- {s:.2f}")

  fig, ax = plt.subplots(figsize=(10, 5))
  x = np.arange(len(ordered_agents))
  bar_width = 0.35

  # Train bars (solid)
  ax.bar(
    x - bar_width / 2,
    train_means,
    bar_width,
    yerr=train_sems,
    color=colors,
    capsize=4,
    edgecolor="black",
    linewidth=0.5,
  )

  # Test bars (hatched)
  ax.bar(
    x + bar_width / 2,
    test_means,
    bar_width,
    yerr=test_sems,
    color=colors,
    capsize=4,
    edgecolor="black",
    linewidth=0.5,
    hatch="//",
  )

  # Create legend with model names and train/test distinction
  model_patches = [
    Patch(facecolor=c, edgecolor="black", label=lbl) for c, lbl in zip(colors, labels)
  ]
  train_patch = Patch(facecolor="white", edgecolor="black", label="Train")
  test_patch = Patch(
    facecolor="white", edgecolor="black", hatch="//", label="Test (eval)"
  )
  ax.legend(
    handles=model_patches + [train_patch, test_patch],
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0,
  )

  ax.set_xticks([])  # Remove x-axis labels
  ax.set_ylabel("Avg deviation from optimal (steps)")
  ax.set_title(
    f"{experiment_label.replace('_', ' ').title()} — Avg deviation from optimal"
  )
  fig.tight_layout()

  save_path = os.path.join(OUTPUT_DIR, f"optimal_length_{experiment_label}.pdf")
  fig.savefig(save_path, dpi=300)
  print(f"  Saved {save_path}")
  plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
  os.makedirs(OUTPUT_DIR, exist_ok=True)

  # ---- JaxMaze ----
  jaxmaze_user_df = pl.read_parquet(data_configs.get_dataframe_path("jaxmaze", "human"))
  jaxmaze_model_df = data_configs.load_dataframes("jaxmaze")

  for experiment in ["shortcut", "path_reuse"]:
    print(f"\n=== jaxmaze_{experiment} ===")
    human_train, model_train, user_ids = get_jaxmaze_train_data(
      jaxmaze_user_df, jaxmaze_model_df, experiment
    )
    human_eval, model_eval = get_jaxmaze_eval_data(
      jaxmaze_user_df, jaxmaze_model_df, experiment, user_ids
    )

    print(f"\n--- BFS optimal baselines ({experiment}) ---")
    bfs_optimal_train = compute_bfs_optimal(jaxmaze_model_df, is_eval=False)
    bfs_optimal_eval = compute_bfs_optimal(jaxmaze_model_df, is_eval=True)

    make_jaxmaze_plot(
      experiment,
      human_train,
      model_train,
      human_eval,
      model_eval,
      bfs_optimal_train,
      bfs_optimal_eval,
    )

    print(f"\n--- Anomalous episodes ({experiment}) ---")
    anomalous = find_anomalous_episodes(human_train, bfs_optimal_train)
    if len(anomalous) > 0:
      visualize_anomalous_episodes(anomalous, experiment)

    print(f"\n--- Example train paths ({experiment}) ---")
    save_example_train_paths(human_train, model_train, experiment)

  # ---- Craftax ----
  print("\n=== craftax_path_reuse ===")
  craftax_data = get_experiment_data("craftax_path_reuse")
  craftax_user_df = craftax_data["user_df"]
  craftax_model_df = craftax_data["model_df"]

  human_train_all = craftax_user_df.filter(eval=False, success=1)
  model_train_all = craftax_model_df.filter(eval=False, success=1)
  human_eval_all = craftax_user_df.filter(eval=True, success=1)
  model_eval_all = craftax_model_df.filter(eval=True, success=1)

  for tell_reuse in [1, 0]:
    label = "known" if tell_reuse == 1 else "unknown"
    print(f"\n--- Craftax {label} eval goals (tell_reuse={tell_reuse}) ---")
    h_train = human_train_all.filter(tell_reuse=tell_reuse)
    h_eval = human_eval_all.filter(tell_reuse=tell_reuse)
    make_craftax_plot(
      h_train, model_train_all, h_eval, model_eval_all, f"craftax_{label}"
    )

  print("\nDone!")


if __name__ == "__main__":
  main()
