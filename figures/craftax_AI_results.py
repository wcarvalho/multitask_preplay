import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json

from tqdm.notebook import tqdm
from data_configs import default_colors, model_colors, DIRECTORY

DIRECTORY = os.path.join(DIRECTORY, "craftax_AI_results", "data")
DEFAULT_TITLE_SIZE = 16
DEFAULT_XLABEL_SIZE = 12
DEFAULT_YLABEL_SIZE = 14
DEFAULT_LEGEND_SIZE = 12

model_colors = {
  "ql": default_colors["purple"],
  "ql_sf": default_colors["nice purple"],
  "dyna": model_colors["dyna"],
  "preplay": model_colors["preplay"],
}

extra_baseline_scores = {
  "$M^3$ (1M, IID)": 6.6,
  "EfficientMBRL (1M, IID)": 5.4,
  "PPO (1M, IID)": 2.3,
  "PPO (1B, IID)": 15.3,
  "PQO (1B, IID)": 15.3,
}
extra_baseline_colors = {
  "$M^3$ (1M, IID)": default_colors["reddish purple"],
  "EfficientMBRL (1M, IID)": default_colors["google blue"],
  "PPO (1M, IID)": default_colors["light gray"],
  "PPO (1B, IID)": default_colors["black"],
}


model_names = {
  "ql": "QL + 1-step (10M)",
  "ql_sf": "QL + SF (10M)",
  "dyna": "Dyna (1M)",
  "preplay": "Multitask preplay (1M)",
}

model_order = [
  "qlearning",
  "usfa",
  "dyna",
  "preplay",
]

crafter_achievements_names = [
  "Collect Coal",
  "Collect Drink",
  "Collect Iron",
  "Collect Stone",
  "Defeat Skeleton",
  "Defeat Zombie",
  "Eat Cow",
  "Make Stone Pickaxe",
  "Make Stone Sword",
  "Make Wood Pickaxe",
  # "Make Wood Sword",
  # "Place Furnace",
  # "Place Stone",
  # "Place Table",
  "Make Arrow",
  "Place Torch",
  "Make Torch",
  "Collect Diamond",
  # "Collect Sapling",
  # "Collect Wood",
  # "Eat Plant",
  # "Make Iron Pickaxe",
  # "Make Iron Sword",
  # "Place Plant",
  # "Wake Up",
]
crafter_achievements = [k.lower().replace(" ", "_") for k in crafter_achievements_names]
crafter_achievement_metrics = [f"Achievements/{a}" for a in crafter_achievements]
metrics = ["0.score"] + crafter_achievement_metrics


def get_runs(group, name, entity="wcarvalho92", project="craftax"):
  api = wandb.Api()
  return api.runs(
    f"{entity}/{project}",
    filters={
      "group": group,
      **({"display_name": name} if name else {}),
    },
  )


def get_metric_data_by_group(model_to_group=None, debug=False):
  """Retrieves raw achievement data from Weights & Biases experiments by group.

  Args:
      setting (str, optional): The metric prefix used in W&B logging.
          Defaults to "evaluator_performance-achievements-64".
      info (dict, optional): Dictionary mapping model names to group names.
          If None, uses default configuration.
          Format: {
              'model_key': 'group_name',
              ...
          }

  Returns:
      pandas.DataFrame: A dataframe containing raw achievement data with columns:
          - model: The model identifier
          - setting: The metric setting used
          - metric: The metric name
          - value: The metric value
          - run_id: The W&B run ID
  """
  # Initialize empty lists to store data
  data = []
  os.makedirs(DIRECTORY, exist_ok=True)

  # Collect data for each model and achievement
  for model, group in tqdm(model_to_group.items(), desc="Models", leave=True):
    # Create cache filename
    if debug:
      cache_file = os.path.join(DIRECTORY, f"{model}_{group}_debug_raw.json")
    else:
      cache_file = os.path.join(DIRECTORY, f"{model}_{group}_raw.json")

    # Try to load cached data
    if os.path.exists(cache_file):
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
      for key in keys:
        if "Achievements" in key:
          parts = key.split("/")
          setting = parts[0]
          metric = "/".join(parts[1:])  # Join remaining parts with '/'
        elif "0.score" in key:
          setting, metric = key.split("/")
        else:
          continue
        value = history[key].max()
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

    # Save this model's data to cache
    if model_data:
      with open(cache_file, "w") as f:
        json.dump(model_data, f)
        print(f"Saved {cache_file}")
      data.extend(model_data)

  return pd.DataFrame(data)


def plot_achievement_bars(
  df, n=64, fig=None, ax=None, figsize=(12, 5), show_legend=True
):
  # Create figure and axis if not provided
  if fig is None or ax is None:
    fig, ax = plt.subplots(figsize=figsize)

  # Filter data for specific setting
  setting = f"evaluator_performance-achievements-{n}"
  df = df[df["setting"] == setting]

  # Get unique models and achievements
  models = df["model"].unique()
  n_models = len(models)
  n_achievements = len(crafter_achievement_metrics)

  # Set up positions for grouped bars
  bar_width = 0.8 / n_models
  x_pos = np.arange(n_achievements)

  # Sort achievements based on preplay's mean values
  preplay_data = df[df["model"] == "preplay"]
  achievement_means = {
    metric: preplay_data[preplay_data["metric"] == metric]["value"].mean()
    for metric in crafter_achievement_metrics
  }
  sorted_metrics = sorted(
    crafter_achievement_metrics, key=lambda x: achievement_means[x], reverse=True
  )

  # Plot bars for each model
  bars = {}  # Store bar containers for each model
  for i, model in enumerate(models):
    data = df[df["model"] == model]
    means = []
    sems = []
    for metric in sorted_metrics:
      metric_data = data[data["metric"] == metric]["value"]
      means.append(metric_data.mean())
      sems.append(metric_data.sem())

    container = ax.bar(
      x_pos + i * bar_width - (n_models - 1) * bar_width / 2,
      means,
      bar_width,
      yerr=sems,
      label=model_names[model.replace("-", "_")],
      color=model_colors[model.replace("-", "_")],
      capsize=3,
    )
    bars[model] = {"means": means, "sems": sems, "container": container}

  # Add stars for non-overlapping error bars between preplay and dyna
  for idx, metric in enumerate(sorted_metrics):
    preplay_mean = bars["preplay"]["means"][idx]
    preplay_sem = bars["preplay"]["sems"][idx]
    dyna_mean = bars["dyna"]["means"][idx]
    dyna_sem = bars["dyna"]["sems"][idx]

    # Check if error bars don't overlap
    preplay_low = preplay_mean - preplay_sem
    preplay_high = preplay_mean + preplay_sem
    dyna_low = dyna_mean - dyna_sem
    dyna_high = dyna_mean + dyna_sem

    # if (preplay_low > dyna_high) or (dyna_low > preplay_high):
    #    # Place star above the higher bar
    #    higher_mean = max(preplay_mean, dyna_mean)
    #    plt.text(idx, higher_mean + max(preplay_sem, dyna_sem), '*',
    #            ha='center', va='bottom', fontsize=12)

  # Customize plot
  ax.set_xticks(x_pos)

  # Format achievement names - split long ones into two lines
  x_labels = []
  for metric in sorted_metrics:
    name = metric.split("/")[-1].replace("_", " ").title()
    words = name.split()
    if len(words) >= 3:
      # Split into roughly equal parts
      mid = max(len(words) // 2, 2)
      name = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
    x_labels.append(name)

  ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)
  ax.set_ylabel("Success Rate", fontsize=12)

  # Set y-axis limit
  ymax = df["value"].max()
  ymin = df["value"].min()
  ax.set_ylim(ymin, 105)

  # Set x-axis limit to reduce right side space
  ax.set_xlim(-0.75, len(x_pos) - 0.5)

  ax.grid(True, alpha=0.3)
  if show_legend:
    ax.legend(loc="upper right", ncol=1, fontsize=10)
  ax.set_title(
    f"Per-Achievement Generalization Success Rates given {n} Unique Training Environments",
    fontsize=14,
    pad=5,
  )

  # Adjust layout to minimize empty space
  fig.tight_layout()
  return fig, ax


def plot_training_envs_score(
  df,
  ntraining_envs,
  ax=None,
  figsize=(5, 5),
  show_legend=True,
  evaluation=True,
  ylim=None,
  extra_baselines=None,
  extra_baseline_colors=None,
):
  # Create figure and axis if not provided
  if ax is None:
    fig, ax = plt.subplots(figsize=figsize)
  else:
    fig = ax.get_figure()

  key = "evaluator" if evaluation else "actor"
  # Get unique models
  models = df["model"].unique()

  # Plot lines and points for each model
  for model in models:
    # Plot evaluator performance (original functionality)
    model_data = []
    model_sems = []
    x_pos = []

    for n_env in ntraining_envs:
      setting = f"{key}_performance-{n_env}"
      data = df[
        (df["model"] == model)
        & (df["setting"] == setting)
        & (df["metric"] == "0.score")
      ]

      mean = data["value"].mean()
      sem = data["value"].sem()

      if len(data) == 0:
        continue
      model_data.append(mean)
      model_sems.append(sem)
      x_pos.append(n_env)

    if len(model_data) > 0:
      # Plot line connecting points
      ax.plot(
        x_pos,
        model_data,
        "-o",  # Line style with circles for points
        label=model_names[model.replace("-", "_")],
        color=model_colors[model.replace("-", "_")],
        linewidth=2,
        markersize=8,
      )

      # Add error bars
      ax.errorbar(
        x_pos,
        model_data,
        yerr=model_sems,
        fmt="none",  # No connecting line
        color=model_colors[model.replace("-", "_")],
        capsize=5,
      )

  # Add extra baseline scores as horizontal dashed lines
  if extra_baselines is not None and extra_baseline_colors is not None:
    x_min, x_max = min(ntraining_envs) / 1.5, max(ntraining_envs) * 1.5
    for name, score in extra_baselines.items():
      color = extra_baseline_colors.get(
        name, "gray"
      )  # Default to gray if color not found
      ax.hlines(
        y=score,
        xmin=x_min,
        xmax=x_max,
        linestyles="dashed",
        colors=color,
        label=name,
        linewidth=2,
      )

  # Customize plot
  ax.set_xlabel("Number of Unique Training Environments", fontsize=DEFAULT_XLABEL_SIZE)
  ax.set_ylabel("% Maximum Score", fontsize=DEFAULT_YLABEL_SIZE)

  if key == "evaluator":
    title = "Generalization Performance to \n10,000 Unique Environments"
  else:
    title = "Training Performance"
  ax.set_title(title, fontsize=DEFAULT_TITLE_SIZE)

  # Set x-ticks to match ntraining_envs values
  ax.set_xscale("log", base=2)
  ax.set_xticks(ntraining_envs)
  ax.set_xticklabels(ntraining_envs, fontsize=DEFAULT_XLABEL_SIZE)
  ax.set_xlim(min(ntraining_envs) / 1.5, max(ntraining_envs) * 1.5)

  # Add grid
  ax.grid(True, alpha=0.3)

  # Add legend if both types of data are shown
  if show_legend:
    ax.legend(fontsize=DEFAULT_LEGEND_SIZE)

  # Set ylims if provided
  if ylim is not None:
    ax.set_ylim(*ylim)

  if ax is None:
    plt.tight_layout()

  return fig, ax


if __name__ == "__main__":
  model_to_group = {
    "ql": "ql-final-5",
    "ql-sf": "ql-sf-final-5",
    "dyna": "dyna-final-5",
    "preplay": "preplay-final-5",
  }
  df = get_metric_data_by_group(
    model_to_group=model_to_group,
    debug=False,
  )

  directory = DIRECTORY

  def save_figure(
    fig,
    filename,
  ):
    os.makedirs(directory, exist_ok=True)
    # plt.savefig(os.path.join(directory, f"{filename}.png"), bbox_inches='tight', dpi=300)
    plt.savefig(
      os.path.join(directory, f"{filename}.pdf"), bbox_inches="tight", dpi=300
    )
    print(f"Saved figure to {directory}/{filename}.pdf")
    plt.close()

  # Save seed counts at the beginning
  options = [8, 16, 32, 64, 128, 256, 512]
  with open(os.path.join(directory, "seed_counts.txt"), "w") as f:
    f.write("Number of seeds per model and training environment count:\n")
    f.write("-" * 60 + "\n\n")

    # Get all models
    models = sorted(df["model"].unique())

    # For each n value
    for n in options:
      f.write(f"\nFor {n} training environments:\n")
      f.write("-" * 30 + "\n")

      # Filter data for this n
      setting = f"evaluator_performance-achievements-{n}"
      n_df = df[df["setting"] == setting]

      # For each model
      for model in models:
        # Count unique run_ids for this model
        model_data = n_df[n_df["model"] == model]
        n_seeds = len(model_data["run_id"].unique())

        f.write(f"{model}: {n_seeds} seeds\n")

      f.write("\n")

  print(f"Saved seed counts to {directory}/seed_counts.txt")

  # First plot train and eval together
  fig, ax = plt.subplots(1, 2, figsize=(10, 5))
  plot_training_envs_score(
    df,
    ntraining_envs=[8, 16, 32, 64, 128, 256, 512],
    show_legend=False,
    evaluation=False,
    ax=ax[0],
    ylim=(1.5, 16),
    extra_baselines=extra_baseline_scores,
    extra_baseline_colors=extra_baseline_colors,
  )
  plot_training_envs_score(
    df,
    ntraining_envs=[8, 16, 32, 64, 128, 256, 512],
    show_legend=True,
    evaluation=True,
    ax=ax[1],
    ylim=(1.5, 16),
    extra_baselines=extra_baseline_scores,
    extra_baseline_colors=extra_baseline_colors,
  )
  save_figure(fig, "train_eval")

  # then plot eval by itself
  fig, ax = plot_training_envs_score(
    df,
    ntraining_envs=[8, 16, 32, 64, 128, 256, 512],
    show_legend=True,
    evaluation=True,
    ylim=(1.5, 16),
    extra_baselines=extra_baseline_scores,
    extra_baseline_colors=extra_baseline_colors,
  )
  save_figure(fig, "eval")

  # then plot per achievement for 512 (in main paper)
  fig, ax = plot_achievement_bars(df, n=512, figsize=(12, 5), show_legend=True)
  save_figure(fig, "achievement_bars_512")

  # then plot per achievement for all
  nrows = (len(options) + 1) // 2  # Ceiling division to handle odd number of plots
  fig, ax = plt.subplots(nrows, 2, figsize=(20, 5 * nrows))
  ax = ax.flatten()  # Flatten to make indexing easier
  for i, n in enumerate(options):
    plot_achievement_bars(df, n=n, show_legend=i == 0, fig=fig, ax=ax[i])
  # Hide any empty subplots
  for j in range(len(options), len(ax)):
    ax[j].set_visible(False)
  save_figure(fig, "achievement_bars_all")
