"""
Dyna multigoal Craftax analysis plots.

Migrated from random/dyna_multigoal_craftax.py.

Usage:
    python plots/craftax_ai_dyna_multigoal.py
    python plots/craftax_ai_dyna_multigoal.py --refresh
    python plots/craftax_ai_dyna_multigoal.py --overwrite
"""

import argparse
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import wandb

# Repo-root import for data_configs (not used yet but kept for consistency)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plot_configs import default_colors

# Directories
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
CACHE_DIR = os.path.join(OUTPUT_DIR, "dyna_multigoal_craftax_cache")

ENTITY = "wcarvalho92"
PROJECT = "craftax"
GROUP = "dyna-multigoal-pnas-1"
REF_GROUP = "dyna-final-5"
PREPLAY_GROUP = "preplay-final-5"

# Achievement metrics
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


def plot_score_by_name(df, save_dir=OUTPUT_DIR):
  """Bar plot of % Maximum Reward (0.score) grouped by dyna value label."""
  eval_df = df[(df["setting"].str.contains("eval")) & (df["metric"] == "0.score")]
  if eval_df.empty:
    print("No eval score data found")
    return

  grouped = eval_df.groupby("label")["value"].agg(["mean", "sem"]).reset_index()
  # Sort: preplay first, then dyna, then numeric labels in order
  label_order = ["preplay", "dyna"]

  def sort_key(label):
    if label in label_order:
      return (0, label_order.index(label))
    return (1, float(label))

  grouped = grouped.iloc[grouped["label"].map(sort_key).argsort()]

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

  ax.set_xticks(x)
  ax.set_xticklabels(grouped["label"], fontsize=11)
  ax.set_xlabel("Random Goal Replacement Rate", fontsize=12)
  ax.set_ylabel("% Maximum Reward", fontsize=12)
  ax.set_title("Dyna Multigoal: % Maximum Reward by Condition", fontsize=14)
  ax.grid(True, alpha=0.3, axis="y")
  fig.tight_layout()

  os.makedirs(save_dir, exist_ok=True)
  path = os.path.join(save_dir, "craftax_ai_dyna_multigoal_score.pdf")
  fig.savefig(path, bbox_inches="tight", dpi=300)
  print(f"Saved {path}")
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser(description="Dyna multigoal Craftax analysis plots")
  parser.add_argument(
    "--overwrite",
    "--refresh",
    action="store_true",
    dest="overwrite",
    help="Clear cached data and re-fetch from wandb",
  )
  parser.add_argument("--group", default=GROUP, help="W&B group name")
  parser.add_argument("--ref_group", default=REF_GROUP, help="Reference W&B dyna group")
  parser.add_argument(
    "--preplay_group", default=PREPLAY_GROUP, help="Reference W&B preplay group"
  )
  args = parser.parse_args()

  # Fetch multigoal experiment data
  df = get_group_data(group=args.group, overwrite=args.overwrite)
  df = add_dyna_label(df)

  # Fetch preplay reference group (filter to 512 training envs)
  preplay_df = get_group_data(group=args.preplay_group, overwrite=args.overwrite)
  preplay_df = preplay_df[preplay_df["setting"].str.contains("512")].copy()
  preplay_df["label"] = "preplay"

  # Fetch dyna reference group (filter to 512 training envs)
  ref_df = get_group_data(group=args.ref_group, overwrite=args.overwrite)
  ref_df = ref_df[ref_df["setting"].str.contains("512")].copy()
  ref_df["label"] = "dyna"

  # Combine: preplay, dyna, then multigoal conditions
  combined = pd.concat([preplay_df, ref_df, df], ignore_index=True)

  if combined.empty:
    print("No data fetched. Check group names and wandb access.")
  else:
    labels = combined["label"].unique().tolist()
    print(f"Fetched {len(combined)} rows, labels: {labels}")
    plot_score_by_name(combined)


if __name__ == "__main__":
  main()
