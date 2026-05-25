"""
python random/dyna_multigoal_craftax.py
python random/dyna_multigoal_craftax.py --overwrite
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_configs import default_colors

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "plots", "dyna_multigoal_craftax")
CACHE_DIR = os.path.join(SAVE_DIR, "cache")

ENTITY = "wcarvalho92"
PROJECT = "craftax"
GROUP = "dyna-multigoal-pnas-1"
REF_GROUP = "dyna-final-5"

# Achievement metrics (same as figures/craftax_AI_results.py)
CRAFTER_ACHIEVEMENTS = [
  "collect_coal",
  "collect_drink",
  "collect_iron",
  "collect_stone",
  "defeat_skeleton",
  "defeat_zombie",
  "eat_cow",
  "make_stone_pickaxe",
  "make_stone_sword",
  "make_wood_pickaxe",
  "make_arrow",
  "place_torch",
  "make_torch",
  "collect_diamond",
]
ACHIEVEMENT_METRICS = [f"Achievements/{a}" for a in CRAFTER_ACHIEVEMENTS]
ALL_METRICS = ["0.score"] + ACHIEVEMENT_METRICS

# Colorblind-friendly palette for bars
BAR_COLORS = [
  default_colors["vermillion"],
  default_colors["blue"],
  default_colors["bluish green"],
  default_colors["sky blue"],
  default_colors["orange"],
  default_colors["reddish purple"],
  default_colors["nice purple"],
  default_colors["pretty blue"],
  default_colors["dark gray"],
  default_colors["light gray"],
]


import re


def extract_dyna_value(name):
  """Extract the dyna=X value from a run name, return as string label."""
  match = re.search(r"dyna=([\d.]+)", name)
  if match:
    return match.group(1)
  return name


def add_dyna_label(df):
  """Add a 'label' column with the extracted dyna value."""
  df = df.copy()
  df["label"] = df["name"].apply(extract_dyna_value)
  return df


def get_group_data(
  group=GROUP,
  entity=ENTITY,
  project=PROJECT,
  cache_dir=CACHE_DIR,
  overwrite=False,
):
  """Fetch best-eval-timepoint data for all runs in a wandb group, with caching."""
  os.makedirs(cache_dir, exist_ok=True)
  cache_file = os.path.join(cache_dir, f"{group}_data.json")

  if overwrite and os.path.exists(cache_file):
    os.remove(cache_file)

  if os.path.exists(cache_file):
    print(f"Loaded {cache_file}")
    with open(cache_file, "r") as f:
      return pd.DataFrame(json.load(f))

  print(f"Fetching runs for group={group}")
  api = wandb.Api()
  runs = api.runs(f"{entity}/{project}", filters={"group": group})

  data = []
  for run in tqdm(list(runs), desc=f"Processing {group} runs"):
    try:
      history = run.history()
      keys = sorted(run.summary.keys())
      if len(keys) == 0:
        print(f"  No keys for {run.name}")
        continue

      for larger_setting in ["eval", "actor"]:
        score_keys = [k for k in keys if larger_setting in k and "0.score" in k]
        if not score_keys:
          continue
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
          data.append(
            {
              "name": run.name,
              "setting": setting,
              "group": group,
              "metric": metric,
              "value": value,
              "run_id": run.id,
            }
          )
    except Exception as e:
      print(f"  Error processing run {run.name}: {e}")

  if data:
    with open(cache_file, "w") as f:
      json.dump(data, f)
    print(f"Saved {cache_file}")

  return pd.DataFrame(data)


def plot_score_by_name(df, save_dir=SAVE_DIR):
  """Bar plot of % Maximum Reward (0.score) grouped by dyna value label."""
  eval_df = df[(df["setting"].str.contains("eval")) & (df["metric"] == "0.score")]
  if eval_df.empty:
    print("No eval score data found")
    return

  grouped = eval_df.groupby("label")["value"].agg(["mean", "sem"]).reset_index()
  grouped = grouped.sort_values("label", key=lambda s: s.astype(float))

  fig, ax = plt.subplots(figsize=(max(8, len(grouped) * 1.5), 5))
  x = np.arange(len(grouped))
  colors = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(grouped))]

  bars = ax.bar(
    x,
    grouped["mean"],
    yerr=grouped["sem"],
    capsize=5,
    color=colors,
    alpha=0.85,
  )

  # Value labels on bars
  for bar, mean, sem in zip(bars, grouped["mean"], grouped["sem"]):
    ax.text(
      bar.get_x() + bar.get_width() / 2.0,
      bar.get_height() + sem + 0.1,
      f"{mean:.2f}",
      ha="center",
      va="bottom",
      fontsize=10,
    )

  ax.set_xticks(x)
  ax.set_xticklabels(grouped["label"], fontsize=11)
  ax.set_xlabel("Random Goal Replacement Rate", fontsize=12)
  ax.set_ylabel("% Maximum Reward", fontsize=12)
  ax.set_title("Dyna Multigoal: % Maximum Reward by Condition", fontsize=14)
  ax.grid(True, alpha=0.3, axis="y")
  fig.tight_layout()

  os.makedirs(save_dir, exist_ok=True)
  for ext in ["png", "pdf"]:
    path = os.path.join(save_dir, f"score_by_name.{ext}")
    fig.savefig(path, bbox_inches="tight", dpi=300)
  print(f"Saved score_by_name plots to {save_dir}")
  plt.close(fig)


def plot_geometric_score_by_name(df, save_dir=SAVE_DIR):
  """Bar plot of geometric score grouped by dyna value label."""
  eval_df = df[
    (df["setting"].str.contains("eval"))
    & (df["metric"].str.startswith("Achievements/"))
  ]
  if eval_df.empty:
    print("No eval achievement data found")
    return

  # Compute geometric score per run
  geo_scores = []
  for (label, run_id), run_data in eval_df.groupby(["label", "run_id"]):
    scores = run_data["value"].values
    if len(scores) > 0:
      geo = np.exp(np.mean(np.log(1 + scores))) - 1
      geo_scores.append({"label": label, "geometric_score": geo})

  if not geo_scores:
    print("No geometric scores computed")
    return

  geo_df = pd.DataFrame(geo_scores)
  grouped = (
    geo_df.groupby("label")["geometric_score"].agg(["mean", "sem"]).reset_index()
  )
  grouped = grouped.sort_values("label", key=lambda s: s.astype(float))

  fig, ax = plt.subplots(figsize=(max(8, len(grouped) * 1.5), 5))
  x = np.arange(len(grouped))
  colors = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(grouped))]

  bars = ax.bar(
    x,
    grouped["mean"],
    yerr=grouped["sem"],
    capsize=5,
    color=colors,
    alpha=0.85,
  )

  for bar, mean, sem in zip(bars, grouped["mean"], grouped["sem"]):
    ax.text(
      bar.get_x() + bar.get_width() / 2.0,
      bar.get_height() + sem + 0.02,
      f"{mean:.2f}",
      ha="center",
      va="bottom",
      fontsize=10,
    )

  ax.set_xticks(x)
  ax.set_xticklabels(grouped["label"], fontsize=11)
  ax.set_xlabel("Random Goal Replacement Rate", fontsize=12)
  ax.set_ylabel("Geometric Score", fontsize=12)
  ax.set_title("Dyna Multigoal: Geometric Score by Condition", fontsize=14)
  ax.grid(True, alpha=0.3, axis="y")
  fig.tight_layout()

  os.makedirs(save_dir, exist_ok=True)
  for ext in ["png", "pdf"]:
    path = os.path.join(save_dir, f"geometric_score_by_name.{ext}")
    fig.savefig(path, bbox_inches="tight", dpi=300)
  print(f"Saved geometric_score_by_name plots to {save_dir}")
  plt.close(fig)


def plot_both(df, save_dir=SAVE_DIR):
  """Side-by-side bar plots: % Maximum Reward and Geometric Score."""
  # --- Score data ---
  eval_score_df = df[(df["setting"].str.contains("eval")) & (df["metric"] == "0.score")]
  # --- Achievement data ---
  eval_ach_df = df[
    (df["setting"].str.contains("eval"))
    & (df["metric"].str.startswith("Achievements/"))
  ]

  if eval_score_df.empty and eval_ach_df.empty:
    print("No eval data found")
    return

  score_grouped = (
    eval_score_df.groupby("label")["value"]
    .agg(["mean", "sem"])
    .reset_index()
    .sort_values("label", key=lambda s: s.astype(float))
  )

  geo_scores = []
  for (label, run_id), run_data in eval_ach_df.groupby(["label", "run_id"]):
    scores = run_data["value"].values
    if len(scores) > 0:
      geo = np.exp(np.mean(np.log(1 + scores))) - 1
      geo_scores.append({"label": label, "geometric_score": geo})

  geo_grouped = (
    pd.DataFrame(geo_scores)
    .groupby("label")["geometric_score"]
    .agg(["mean", "sem"])
    .reset_index()
    .sort_values("label", key=lambda s: s.astype(float))
  )

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, len(score_grouped) * 2), 5))

  # Panel A: Score
  x = np.arange(len(score_grouped))
  colors = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(score_grouped))]
  bars1 = ax1.bar(
    x,
    score_grouped["mean"],
    yerr=score_grouped["sem"],
    capsize=5,
    color=colors,
    alpha=0.85,
  )
  for bar, mean, sem in zip(bars1, score_grouped["mean"], score_grouped["sem"]):
    ax1.text(
      bar.get_x() + bar.get_width() / 2.0,
      bar.get_height() + sem + 0.1,
      f"{mean:.2f}",
      ha="center",
      va="bottom",
      fontsize=10,
    )
  ax1.set_xticks(x)
  ax1.set_xticklabels(score_grouped["label"], fontsize=11)
  ax1.set_xlabel("Random Goal Replacement Rate", fontsize=12)
  ax1.set_title("(A) % Maximum Reward", fontsize=14)
  ax1.grid(True, alpha=0.3, axis="y")

  # Panel B: Geometric Score
  x2 = np.arange(len(geo_grouped))
  colors2 = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(geo_grouped))]
  bars2 = ax2.bar(
    x2,
    geo_grouped["mean"],
    yerr=geo_grouped["sem"],
    capsize=5,
    color=colors2,
    alpha=0.85,
  )
  for bar, mean, sem in zip(bars2, geo_grouped["mean"], geo_grouped["sem"]):
    ax2.text(
      bar.get_x() + bar.get_width() / 2.0,
      bar.get_height() + sem + 0.02,
      f"{mean:.2f}",
      ha="center",
      va="bottom",
      fontsize=10,
    )
  ax2.set_xticks(x2)
  ax2.set_xticklabels(geo_grouped["label"], fontsize=11)
  ax2.set_xlabel("Random Goal Replacement Rate", fontsize=12)
  ax2.set_title("(B) Geometric Score", fontsize=14)
  ax2.grid(True, alpha=0.3, axis="y")

  fig.suptitle("Dyna Multigoal Craftax: Performance by Condition", fontsize=16)
  fig.tight_layout()

  os.makedirs(save_dir, exist_ok=True)
  for ext in ["png", "pdf"]:
    path = os.path.join(save_dir, f"both_scores_by_name.{ext}")
    fig.savefig(path, bbox_inches="tight", dpi=300)
  print(f"Saved both_scores_by_name plots to {save_dir}")
  plt.close(fig)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--overwrite", action="store_true", help="Refresh cached data")
  parser.add_argument("--group", default=GROUP, help="W&B group name")
  parser.add_argument("--ref_group", default=REF_GROUP, help="Reference W&B group (0)")
  args = parser.parse_args()

  # Fetch multigoal experiment data
  df = get_group_data(group=args.group, overwrite=args.overwrite)
  df = add_dyna_label(df)

  # Fetch reference group (standard dyna = 0 random goal replacement)
  ref_df = get_group_data(group=args.ref_group, overwrite=args.overwrite)
  ref_df = ref_df.copy()
  ref_df["label"] = "0"

  # Combine
  combined = pd.concat([ref_df, df], ignore_index=True)

  if combined.empty:
    print("No data fetched. Check group names and wandb access.")
  else:
    labels = sorted(combined["label"].unique(), key=float)
    print(f"Fetched {len(combined)} rows, labels: {labels}")
    plot_score_by_name(combined)
    plot_geometric_score_by_name(combined)
    plot_both(combined)
