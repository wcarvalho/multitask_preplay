import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json

from tqdm.notebook import tqdm
from plot_configs import default_colors, model_colors
from data_configs import CRAFTAX_AI_DIR

DIRECTORY = os.path.join(CRAFTAX_AI_DIR, "main")
DEFAULT_TITLE_SIZE = 16
DEFAULT_XLABEL_SIZE = 12
DEFAULT_YLABEL_SIZE = 14
DEFAULT_LEGEND_SIZE = 12

model_colors = {
  # "ql": default_colors["purple"],
  # "ql_sf": default_colors["nice purple"],
  "ql": model_colors["qlearning"],
  "ql_sf": model_colors["usfa"],
  "dyna": model_colors["dyna"],
  "preplay": model_colors["preplay"],
}

extra_baseline_scores = {
  "$Simulus$ (1M, IID)": 6.6,
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


def get_metric_data_by_group(model_to_group=None, overwrite=False, debug=False):
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
        except Exception as e:
          best_idx = len(history[score_keys]) - 1

        for key in keys:
          if larger_setting not in key:
            continue
          if "Achievements" in key:
            parts = key.split("/")
            setting = parts[0]
            metric = "/".join(parts[1:])  # Join remaining parts with '/'
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
  ax.set_ylabel("% Maximum Reward", fontsize=DEFAULT_YLABEL_SIZE)

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


def plot_training_envs_geometric_score(
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
  """
  Plot geometric mean of achievement scores across training environments.

  Calculates the geometric mean score using the formula:
  S = exp(1/N * Σ ln(1 + s_i)) - 1
  where s_i is the success percentage for achievement i (0-100 scale).

  Args:
    df: DataFrame containing the metric data
    ntraining_envs: List of training environment counts to plot
    ax: Matplotlib axes object (optional)
    figsize: Figure size if creating new figure
    show_legend: Whether to show the legend
    evaluation: Whether to use evaluator or actor performance
    ylim: Y-axis limits (optional)
    extra_baselines: Dict of extra baseline scores (optional)
    extra_baseline_colors: Dict of colors for extra baselines (optional)

  Returns:
    fig, ax: Matplotlib figure and axes objects
  """
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
    model_data = []
    model_sems = []
    x_pos = []

    for n_env in ntraining_envs:
      setting = f"{key}_performance-achievements-{n_env}"

      # Get all achievement data for this model/setting combination
      achievement_data = df[
        (df["model"] == model)
        & (df["setting"] == setting)
        & (df["metric"].str.startswith("Achievements/"))
      ]

      if len(achievement_data) == 0:
        continue

      # Get all unique achievement metrics for this setting
      available_metrics = achievement_data["metric"].unique()

      # Group by run_id to compute geometric mean for each run
      geometric_scores = []
      for run_id in achievement_data["run_id"].unique():
        run_data = achievement_data[achievement_data["run_id"] == run_id]

        # Get achievement scores for this run (should have one score per achievement)
        achievement_scores = []
        for metric in available_metrics:
          metric_data = run_data[run_data["metric"] == metric]["value"]
          if len(metric_data) > 0:
            achievement_scores.append(
              metric_data.iloc[0]
            )  # Take first (should be only) value

        if len(achievement_scores) > 0:
          # Convert percentages (0-100) to 0-1 scale
          # import pdb; pdb.set_trace()
          achievement_scores = np.array(achievement_scores)

          # Apply geometric mean formula: exp(mean(ln(1 + s_i))) - 1
          # Add small epsilon to avoid log(1) = 0 issues with perfect scores
          log_scores = np.log(1 + achievement_scores)
          geometric_mean = np.exp(np.mean(log_scores)) - 1

          # Convert back to percentage scale (0-100)
          geometric_scores.append(geometric_mean)

      if len(geometric_scores) > 0:
        mean_score = np.mean(geometric_scores)
        sem_score = np.std(geometric_scores) / np.sqrt(len(geometric_scores))

        model_data.append(mean_score)
        model_sems.append(sem_score)
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
  ax.set_ylabel("Geometric Score", fontsize=DEFAULT_YLABEL_SIZE)

  if key == "evaluator":
    title = "Generalization Performance \nto 10,000 Unique Environments"
  else:
    title = "Training Performance (Geometric Mean)"
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


def get_best_scores_from_group(group_name, model_name="preplay", overwrite=False):
  """Extract the best score and compute geometric score from a group at the best eval timepoint.
  
  Args:
      group_name (str): The W&B group name to download data from
      model_name (str): The model name to use in the dataframe
      overwrite (bool): Whether to overwrite cached data
      
  Returns:
      dict: Dictionary with 'score' and 'geometric_score' keys, each containing
            'mean' and 'sem' values
  """
  # Download data using existing function - this already gets best timepoint data
  model_to_group = {model_name: group_name}
  df = get_metric_data_by_group(model_to_group=model_to_group, overwrite=overwrite)
  
  # Filter for eval data only
  eval_df = df[df["setting"].str.contains("eval")]
  
  # Get unique run IDs
  run_ids = eval_df["run_id"].unique()
  
  scores = []
  geometric_scores = []
  
  for run_id in run_ids:
    # Get data for this run
    run_data = eval_df[eval_df["run_id"] == run_id]
    
    # Get the score
    score_data = run_data[run_data["metric"] == "0.score"]
    if len(score_data) > 0:
      scores.append(score_data["value"].iloc[0])
    
    # Get all achievement values
    achievement_data = run_data[run_data["metric"].str.startswith("Achievements/")]
    if len(achievement_data) > 0:
      achievement_scores = achievement_data["value"].values
      
      # Compute geometric score
      log_scores = np.log(1 + achievement_scores)
      geometric_mean = np.exp(np.mean(log_scores)) - 1
      geometric_scores.append(geometric_mean)
  
  # Compute means and SEMs
  results = {
    'score': {
      'mean': np.mean(scores) if scores else 0,
      'sem': np.std(scores) / np.sqrt(len(scores)) if scores else 0
    },
    'geometric_score': {
      'mean': np.mean(geometric_scores) if geometric_scores else 0,
      'sem': np.std(geometric_scores) / np.sqrt(len(geometric_scores)) if geometric_scores else 0
    }
  }
  
  return results


def plot_baseline_craftax_comparison(group_name, baseline_scores, model_name="Multitask Preplay", 
                                   xlabel_score="Score", xlabel_geometric="Geometric Score",
                                   title="CraftAx Performance Comparison", figsize=(10, 5),
                                   save_path=None):
  """Create a 2-panel bar graph comparing model performance with baselines.
  
  Args:
      group_name (str): The W&B group name to download data from
      baseline_scores (dict): Dictionary of baseline scores with format:
                             {
                               'baseline_name': {
                                 'score': {'mean': float, 'sem': float},
                                 'geometric_score': {'mean': float, 'sem': float or None},
                                 'color': str (optional) - color for the baseline bars,
                                 'name': str (optional) - display name for the baseline
                               }
                             }
      model_name (str): Display name for the model
      xlabel_score (str): X-axis label for score panel
      xlabel_geometric (str): X-axis label for geometric score panel
      title (str): Overall figure title
      figsize (tuple): Figure size
      save_path (str): Path to save the figure (optional)
      
  Returns:
      fig, (ax1, ax2): Figure and axes objects
  """
  # Get scores from the group
  model_scores = get_best_scores_from_group(group_name)
  
  # Create figure with 2 subplots
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
  
  # Prepare data for plotting
  # Use custom names if provided, otherwise use keys
  all_model_names = [model_name] + [baseline_scores[m].get('name', m) for m in baseline_scores]
  scores_mean = [model_scores['score']['mean']] + [baseline_scores[m]['score']['mean'] for m in baseline_scores]
  scores_sem = [model_scores['score']['sem']] + [baseline_scores[m]['score'].get('sem', 0) for m in baseline_scores]
  
  geometric_mean = [model_scores['geometric_score']['mean']]
  geometric_sem = [model_scores['geometric_score']['sem']]
  geometric_model_names = [model_name]
  geometric_unavailable = [False]  # Track which values are unavailable
  
  for m in baseline_scores:
    if baseline_scores[m]['geometric_score']['mean'] is not None:
      geometric_mean.append(baseline_scores[m]['geometric_score']['mean'])
      geometric_sem.append(baseline_scores[m]['geometric_score'].get('sem', 0))
      geometric_unavailable.append(False)
    else:
      # Include models without geometric score but set to 0
      geometric_mean.append(0)
      geometric_sem.append(0)
      geometric_unavailable.append(True)
    geometric_model_names.append(baseline_scores[m].get('name', m))
  
  # Define colors - use custom colors if provided
  colors = [model_colors.get('preplay', default_colors['nice purple'])]
  for m in baseline_scores:
    colors.append(baseline_scores[m].get('color', default_colors['light gray']))
  
  geometric_colors = [model_colors.get('preplay', default_colors['nice purple'])]
  for m in baseline_scores:
    geometric_colors.append(baseline_scores[m].get('color', default_colors['light gray']))
  
  # Plot score panel
  x_pos = np.arange(len(all_model_names))
  bars1 = ax1.bar(x_pos, scores_mean, yerr=scores_sem, capsize=5, color=colors, alpha=0.8)
  ax1.set_xticks(x_pos)
  ax1.set_xticklabels(all_model_names, rotation=45, ha='right')
  ax1.set_title('(A) % Maximum Reward', fontsize=DEFAULT_YLABEL_SIZE)
  #ax1.set_title(xlabel_score, fontsize=DEFAULT_TITLE_SIZE)
  ax1.set_ylim(0, 8)
  ax1.grid(True, alpha=0.3, axis='y')
  
  # Add value labels on top of bars
  for i, (bar, mean, sem) in enumerate(zip(bars1, scores_mean, scores_sem)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + sem + 0.1,
             f'{mean:.2f}', ha='center', va='bottom', fontsize=10)
  
  # Plot geometric score panel
  x_pos_geom = np.arange(len(geometric_model_names))
  bars2 = ax2.bar(x_pos_geom, geometric_mean, yerr=geometric_sem, capsize=5, color=geometric_colors, alpha=0.8)
  ax2.set_xticks(x_pos_geom)
  ax2.set_xticklabels(geometric_model_names, rotation=45, ha='right')
  ax2.set_title('(B) Geometric Score', fontsize=DEFAULT_YLABEL_SIZE)
  #ax2.set_title(xlabel_geometric, fontsize=DEFAULT_TITLE_SIZE)
  ax2.set_ylim(0, 3)
  ax2.grid(True, alpha=0.3, axis='y')
  
  # Add value labels on top of bars
  for i, (bar, mean, sem, unavail) in enumerate(zip(bars2, geometric_mean, geometric_sem, geometric_unavailable)):
    height = bar.get_height()
    if unavail:
      # Show "unavailable" text for missing data
      ax2.text(bar.get_x() + bar.get_width()/2., 0.05,
               'unavailable', ha='center', va='bottom', fontsize=10, style='italic')
    else:
      ax2.text(bar.get_x() + bar.get_width()/2., height + sem + 0.05,
               f'{mean:.2f}', ha='center', va='bottom', fontsize=10)
  
  # Add overall title
  fig.suptitle(title, fontsize=DEFAULT_TITLE_SIZE + 2)
  
  plt.tight_layout()
  
  # Save if path provided
  if save_path:
    # Create directory if it doesn't exist
    save_dir = os.path.dirname(save_path)
    if save_dir:
      os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved figure to {save_path}")
  
  return fig, (ax1, ax2)


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
    ylim=(1.5, 7),
  )
  plot_training_envs_score(
    df,
    ntraining_envs=[8, 16, 32, 64, 128, 256, 512],
    show_legend=True,
    evaluation=True,
    ax=ax[1],
    ylim=(1.5, 7),
  )
  save_figure(fig, "train_eval")

  # then plot eval by itself
  fig, ax = plot_training_envs_score(
    df,
    ntraining_envs=[8, 16, 32, 64, 128, 256, 512],
    show_legend=True,
    evaluation=True,
    ylim=(1.5, 7),
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

  # Example usage of the new baseline comparison function
  # Define baseline scores
  baseline_scores = {
    'TWM': {
      'score': {'mean': 7.2, 'sem': 0.09},
      'geometric_score': {'mean': 2.31, 'sem': 0.04},
      'color': default_colors["google blue"],
      'name': "Transformer World Model\n(Dedieu et al. 2025)",
    },
    'PPO-RNN': {
      'score': {'mean': 2.3, 'sem': 0},
      'geometric_score': {'mean': None, 'sem': None},  # Unavailable
      'color': default_colors["bluish green"],
      'name': "PPO-RNN\n(Matthews et al. 2024)",
    }
  }
  
  # Create baseline comparison plot
  fig, axes = plot_baseline_craftax_comparison(
    group_name="preplay-benchmark-10k-5",
    baseline_scores=baseline_scores,
    model_name="Multitask Preplay",
    title="CraftAx Performance: Multitask Preplay vs Baselines",
    save_path=os.path.join(directory, "baseline_comparison.pdf")
  )
