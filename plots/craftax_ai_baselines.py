"""
Craftax AI baselines comparison: Multitask Preplay vs external baselines (TWM, PPO-RNN).

Migrated from figures_supplemental/craftax_baselines_comparison.ipynb.

Usage:
    python plots/craftax_ai_baselines.py
    python plots/craftax_ai_baselines.py --refresh
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from tqdm.auto import tqdm

# Repo-root imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_configs import CRAFTAX_AI_DIR

import plot_configs as configs
from plot_configs import default_colors

# Directories
DIRECTORY = os.path.join(CRAFTAX_AI_DIR, "main")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Plot styling constants
DEFAULT_TITLE_SIZE = 16
DEFAULT_XLABEL_SIZE = 12
DEFAULT_YLABEL_SIZE = 14
DEFAULT_LEGEND_SIZE = 12

model_colors = {
  "preplay": configs.model_colors["preplay"],
}


# ---------------------------------------------------------------------------
# WandB data helpers
# ---------------------------------------------------------------------------


def get_runs(group, name=None, entity="wcarvalho92", project="craftax"):
  """Get runs from WandB based on group and optional name."""
  api = wandb.Api()
  return api.runs(
    f"{entity}/{project}",
    filters={
      "group": group,
      **({"display_name": name} if name else {}),
    },
  )


def get_metric_data_by_group(model_to_group, overwrite=False, debug=False):
  """Retrieve raw achievement data from WandB experiments by group.

  Returns a DataFrame with columns: model, setting, group, name, metric,
  value, run_id.
  """
  data = []
  os.makedirs(DIRECTORY, exist_ok=True)

  for model, group in tqdm(model_to_group.items(), desc="Models", leave=True):
    suffix = "_debug_raw.json" if debug else "_raw.json"
    cache_file = os.path.join(DIRECTORY, f"{model}_{group}{suffix}")

    if os.path.exists(cache_file) and not overwrite:
      with open(cache_file, "r") as f:
        print(f"Loaded {cache_file}")
        data.extend(json.load(f))
      continue

    model_data = []
    print(group)
    runs = get_runs(group, name=None)

    for run in tqdm(runs, desc=f"Processing {model} runs", leave=True):
      history = run.history()
      keys = sorted(run.summary.keys())
      if len(keys) == 0:
        print(f"No keys found for {run.group}/{run.name}")
        continue
      for larger_setting in ["eval", "actor"]:
        score_keys = [k for k in keys if larger_setting in k and "0.score" in k]
        try:
          best_idx = np.nanargmax(history[score_keys].to_numpy())
        except Exception:
          best_idx = len(history[score_keys]) - 1

        for key in keys:
          if larger_setting not in key:
            continue
          if "Achievements" in key:
            parts = key.split("/")
            setting = parts[0]
            metric = "/".join(parts[1:])
          elif "0.score" in key:
            setting, metric = key.split("/")
          else:
            continue
          value = history[key][best_idx]
          model_data.append(
            {
              "model": model,
              "setting": setting,
              "group": group,
              "name": run.name,
              "metric": metric,
              "value": value,
              "run_id": run.id,
            }
          )
          if debug:
            break
      if debug:
        break

    if model_data:
      with open(cache_file, "w") as f:
        json.dump(model_data, f)
        print(f"Saved {cache_file}")
      data.extend(model_data)

  return pd.DataFrame(data)


def get_best_scores_from_group(group_name, model_name="preplay", overwrite=False):
  """Extract best score and geometric score from a WandB group.

  Returns dict with 'score' and 'geometric_score' keys, each containing
  'mean' and 'sem'.
  """
  model_to_group = {model_name: group_name}
  df = get_metric_data_by_group(model_to_group=model_to_group, overwrite=overwrite)

  eval_df = df[df["setting"].str.contains("eval")]
  run_ids = eval_df["run_id"].unique()

  scores = []
  geometric_scores = []

  for run_id in run_ids:
    run_data = eval_df[eval_df["run_id"] == run_id]

    score_data = run_data[run_data["metric"] == "0.score"]
    if len(score_data) > 0:
      scores.append(score_data["value"].iloc[0])

    achievement_data = run_data[run_data["metric"].str.startswith("Achievements/")]
    if len(achievement_data) > 0:
      achievement_scores = achievement_data["value"].values
      log_scores = np.log(1 + achievement_scores)
      geometric_mean = np.exp(np.mean(log_scores)) - 1
      geometric_scores.append(geometric_mean)

  return {
    "score": {
      "mean": np.mean(scores) if scores else 0,
      "sem": np.std(scores) / np.sqrt(len(scores)) if scores else 0,
    },
    "geometric_score": {
      "mean": np.mean(geometric_scores) if geometric_scores else 0,
      "sem": (
        np.std(geometric_scores) / np.sqrt(len(geometric_scores))
        if geometric_scores
        else 0
      ),
    },
  }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_baseline_craftax_comparison(
  group_name,
  baseline_scores,
  model_name="Multitask Preplay",
  title="CraftAx Performance Comparison",
  figsize=(10, 5),
  save_path=None,
  overwrite=False,
):
  """Create a 2-panel bar graph comparing model performance with baselines.

  Args:
      group_name: WandB group name to fetch our model's data from.
      baseline_scores: Dict of baseline scores, e.g.
          {'TWM': {'score': {'mean': 7.2, 'sem': 0.09},
                   'geometric_score': {'mean': 2.31, 'sem': 0.04},
                   'color': ..., 'name': ...}}
      model_name: Display name for our model.
      title: Overall figure title.
      figsize: Figure size.
      save_path: Path to save the figure (optional).
      overwrite: Whether to re-fetch WandB data.

  Returns:
      fig, (ax1, ax2)
  """
  model_scores = get_best_scores_from_group(group_name, overwrite=overwrite)

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

  # --- Prepare data ---
  all_model_names = [model_name] + [
    baseline_scores[m].get("name", m) for m in baseline_scores
  ]
  scores_mean = [model_scores["score"]["mean"]] + [
    baseline_scores[m]["score"]["mean"] for m in baseline_scores
  ]
  scores_sem = [model_scores["score"]["sem"]] + [
    baseline_scores[m]["score"].get("sem", 0) for m in baseline_scores
  ]

  geometric_mean = [model_scores["geometric_score"]["mean"]]
  geometric_sem = [model_scores["geometric_score"]["sem"]]
  geometric_model_names = [model_name]
  geometric_unavailable = [False]

  for m in baseline_scores:
    if baseline_scores[m]["geometric_score"]["mean"] is not None:
      geometric_mean.append(baseline_scores[m]["geometric_score"]["mean"])
      geometric_sem.append(baseline_scores[m]["geometric_score"].get("sem", 0))
      geometric_unavailable.append(False)
    else:
      geometric_mean.append(0)
      geometric_sem.append(0)
      geometric_unavailable.append(True)
    geometric_model_names.append(baseline_scores[m].get("name", m))

  # Colors
  colors = [model_colors.get("preplay", default_colors["nice purple"])]
  for m in baseline_scores:
    colors.append(baseline_scores[m].get("color", default_colors["light gray"]))

  geometric_colors = [model_colors.get("preplay", default_colors["nice purple"])]
  for m in baseline_scores:
    geometric_colors.append(
      baseline_scores[m].get("color", default_colors["light gray"])
    )

  # --- Panel A: % Maximum Reward ---
  x_pos = np.arange(len(all_model_names))
  bars1 = ax1.bar(
    x_pos, scores_mean, yerr=scores_sem, capsize=5, color=colors, alpha=0.8
  )
  ax1.set_xticks(x_pos)
  ax1.set_xticklabels(all_model_names, rotation=45, ha="right")
  ax1.set_title("(A) % Maximum Reward", fontsize=DEFAULT_YLABEL_SIZE)
  ax1.set_ylim(0, 8)
  ax1.grid(True, alpha=0.3, axis="y")

  for bar, mean, sem in zip(bars1, scores_mean, scores_sem):
    height = bar.get_height()
    ax1.text(
      bar.get_x() + bar.get_width() / 2.0,
      height + sem + 0.1,
      f"{mean:.2f}",
      ha="center",
      va="bottom",
      fontsize=10,
    )

  # --- Panel B: Geometric Score ---
  x_pos_geom = np.arange(len(geometric_model_names))
  bars2 = ax2.bar(
    x_pos_geom,
    geometric_mean,
    yerr=geometric_sem,
    capsize=5,
    color=geometric_colors,
    alpha=0.8,
  )
  ax2.set_xticks(x_pos_geom)
  ax2.set_xticklabels(geometric_model_names, rotation=45, ha="right")
  ax2.set_title("(B) Geometric Score", fontsize=DEFAULT_YLABEL_SIZE)
  ax2.set_ylim(0, 3)
  ax2.grid(True, alpha=0.3, axis="y")

  for bar, mean, sem, unavail in zip(
    bars2, geometric_mean, geometric_sem, geometric_unavailable
  ):
    height = bar.get_height()
    if unavail:
      ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        0.05,
        "unavailable",
        ha="center",
        va="bottom",
        fontsize=10,
        style="italic",
      )
    else:
      ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + sem + 0.05,
        f"{mean:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
      )

  fig.suptitle(title, fontsize=DEFAULT_TITLE_SIZE + 2)
  plt.tight_layout()

  if save_path:
    save_dir = os.path.dirname(save_path)
    if save_dir:
      os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to {save_path}")

  return fig, (ax1, ax2)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


def save_figure(fig, filename, directory=OUTPUT_DIR):
  """Save figure as PDF."""
  os.makedirs(directory, exist_ok=True)
  path = os.path.join(directory, f"{filename}.pdf")
  plt.savefig(path, bbox_inches="tight", dpi=300)
  print(f"Saved figure to {path}")
  plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
  parser = argparse.ArgumentParser(
    description="Craftax baselines comparison: Multitask Preplay vs TWM, PPO-RNN"
  )
  parser.add_argument(
    "--refresh",
    action="store_true",
    help="Re-fetch data from WandB (ignore cache)",
  )
  args = parser.parse_args()

  baseline_scores = {
    "TWM": {
      "score": {"mean": 7.2, "sem": 0.09},
      "geometric_score": {"mean": 2.31, "sem": 0.04},
      "color": default_colors["google blue"],
      "name": "Transformer World Model\n(Dedieu et al. 2025)",
    },
    "PPO-RNN": {
      "score": {"mean": 2.3, "sem": 0},
      "geometric_score": {"mean": None, "sem": None},
      "color": default_colors["bluish green"],
      "name": "PPO-RNN\n(Matthews et al. 2024)",
    },
  }

  fig, axes = plot_baseline_craftax_comparison(
    group_name="preplay-benchmark-10k-5",
    baseline_scores=baseline_scores,
    model_name="Multitask Preplay",
    title="Craftax-full Baselines Comparisons in Standard Setting",
    overwrite=args.refresh,
  )

  save_figure(fig, "craftax_ai_baselines")


if __name__ == "__main__":
  main()
