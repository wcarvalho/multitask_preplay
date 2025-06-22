import sys
import os

# add this directory to the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
from analysis import vis_utils

# from data_processing import utils_jaxmaze as utils
from data_processing.utils_jaxmaze import create_maps, compute_overlap
from figures import figure_utils
import data_configs
import nicewebrl
import numpy as np


def visualize_examples_by_reuse(
  df: nicewebrl.DataFrame,
  manipulation_id: int,
  threshold: float,
  model: str,
  n_examples: int = 10,
  train_maze: str = None,
  test_maze: str = None,
):
  """
  Visualize examples classified by reuse (1 or 0) based on threshold.

  Args:
    df: Full dataframe (for humans) or filtered dataframe (for models)
    manipulation_id: The manipulation ID (1 for shortcut, 3 for two_paths)
    threshold: Threshold value for determining reuse
    model: The model or 'human'
    n_examples: Number of examples to generate for each reuse value
    train_maze: Maze name for training data (for models)
    test_maze: Maze name for test data (for models)
  """
  manipulation_name = "shortcut" if manipulation_id == 1 else "two_paths"

  # Create directory structure
  output_dir = f"{data_configs.DIRECTORY}/figures/jaxmaze_overlap_analysis/{manipulation_name}_{threshold}/{model}"

  # Delete existing directory to remove old data
  import shutil

  if os.path.exists(output_dir):
    print(f"Removing old data from {output_dir}...")
    shutil.rmtree(output_dir)

  # Create fresh directory
  os.makedirs(output_dir, exist_ok=True)

  # Different handling for human data vs model data
  samples = []

  if model == "human":
    # For humans, we need to carefully match train and test examples
    user_ids = df["user_id"].unique().to_list()

    for user_id in user_ids:
      # Get test examples for this user
      test_initial = df.filter(
        eval=True, user_id=user_id, manipulation=manipulation_id
      ).sort("overlap", descending=True)

      if len(test_initial.episodes) == 0:
        continue

      # Process each test episode one by one
      for episode_idx in range(len(test_initial.episodes)):
        # Get values for filtering
        start_pos_val = test_initial["start_pos"].to_list()[episode_idx]
        block_name_val = test_initial["block_name"].to_list()[episode_idx]
        room_val = test_initial["room"].to_list()[episode_idx]

        filters = dict(
          start_pos=start_pos_val,
          block_name=block_name_val,
          user_id=user_id,
          room=room_val,
        )

        # Get properly filtered test and train
        test = df.filter(eval=True, manipulation=manipulation_id, **filters)
        train = df.filter(
          eval=False, manipulation=manipulation_id, success=1, **filters
        )

        if len(train.episodes) == 0 or len(test.episodes) == 0:
          continue

        # Calculate overlap
        train_map = create_maps([train.episodes[0]]).sum(0)
        test_map = create_maps([test.episodes[0]]).sum(0)

        overlap_value = compute_overlap(train_map, test_map).mean()

        reuse = int(overlap_value >= threshold)

        samples.append(
          {
            "key": user_id,
            "train": train,
            "test": test,
            "overlap": overlap_value,
            "reuse": reuse,
            "filters": filters,  # Store filters for reference
            "block_name": block_name_val,
          }
        )
  else:
    # For models, process as before - each seed has one train and one test
    train = df.filter(eval=False, manipulation=manipulation_id, maze=train_maze)
    test = df.filter(eval=True, manipulation=manipulation_id, maze=test_maze)
    # Get available keys (seeds)
    model_key = "seed"
    keys = train[model_key].unique().to_list()

    block_names = train["block_name"].unique().to_list()
    for block_name in block_names:
      for k in keys:
        train_filtered = train.filter(block_name=block_name, **{model_key: k})
        test_filtered = test.filter(block_name=block_name, **{model_key: k})

        if len(train_filtered.episodes) == 0 or len(test_filtered.episodes) == 0:
          continue

        # Calculate overlap
        train_map = create_maps([train_filtered.episodes[0]]).sum(0)
        test_map = create_maps([test_filtered.episodes[0]]).sum(0)

        overlap_value = compute_overlap(train_map, test_map).mean()

        reuse = int(overlap_value >= threshold)

        samples.append(
          {
            "key": k,
            "block_name": block_name,
            "train": train_filtered,
            "test": test_filtered,
            "overlap": overlap_value,
            "reuse": reuse,
          }
        )

  # Separate into above and below threshold examples (regardless of actual reuse classification)
  above_threshold_samples = [s for s in samples if s["overlap"] >= threshold]
  below_threshold_samples = [s for s in samples if s["overlap"] < threshold]

  # Sort by overlap for better visualization
  above_threshold_samples.sort(
    key=lambda x: x["overlap"], reverse=False
  )  # Lower overlap first
  below_threshold_samples.sort(
    key=lambda x: x["overlap"], reverse=True
  )  # Highest overlap first

  print(f"{model} - {manipulation_name} - threshold {threshold}:")
  print(f"  Found {len(above_threshold_samples)} examples with overlap >= {threshold}")
  print(f"  Found {len(below_threshold_samples)} examples with overlap < {threshold}")

  # Limit to n_examples
  above_threshold_samples = above_threshold_samples[:n_examples]
  below_threshold_samples = below_threshold_samples[:n_examples]

  # Create individual figures for above threshold examples
  for idx, sample in enumerate(above_threshold_samples):
    k = sample["key"]
    train_filtered = sample["train"]
    test_filtered = sample["test"]
    overlap_value = sample["overlap"]
    reuse_value = sample["reuse"]
    block_name = sample["block_name"]

    # Create a new figure for each example
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    vis_utils.render_path(train_filtered.episodes[0], ax=axs[0])
    vis_utils.render_path(test_filtered.episodes[0], ax=axs[1])

    train_success = train_filtered["success"].to_list()[0]
    test_success = test_filtered["success"].to_list()[0]

    if model == "human":
      axs[0].set_title(f"Train (user: {k}, success: {train_success})")
    else:
      model_key = "seed"
      axs[0].set_title(f"Train ({model_key}: {k}, success: {train_success})")

    axs[1].set_title(
      f"Test (success: {test_success}, overlap: {overlap_value:.3f}, reuse: {reuse_value})"
    )

    fig.tight_layout()
    # Save with numbered filename including overlap value
    figure_utils.save_figure(
      fig,
      f"above_threshold_example_{idx + 1:02d}_{block_name}_overlap_{overlap_value:.3f}_reuse_{reuse_value}",
      directory=output_dir,
    )
    plt.close(fig)  # Close the figure to free memory

  # Create individual figures for below threshold examples
  for idx, sample in enumerate(below_threshold_samples):
    k = sample["key"]
    train_filtered = sample["train"]
    test_filtered = sample["test"]
    overlap_value = sample["overlap"]
    reuse_value = sample["reuse"]
    block_name = sample["block_name"]

    # Create a new figure for each example
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    vis_utils.render_path(train_filtered.episodes[0], ax=axs[0])
    vis_utils.render_path(test_filtered.episodes[0], ax=axs[1])

    train_success = train_filtered["success"].to_list()[0]
    test_success = test_filtered["success"].to_list()[0]

    if model == "human":
      axs[0].set_title(f"Train (user: {k}, success: {train_success})")
    else:
      model_key = "seed"
      axs[0].set_title(f"Train ({model_key}: {k}, success: {train_success})")

    axs[1].set_title(
      f"Test (success: {test_success}, overlap: {overlap_value:.3f}, reuse: {reuse_value})"
    )

    fig.tight_layout()
    # Save with numbered filename including overlap value
    figure_utils.save_figure(
      fig,
      f"below_threshold_example_{idx + 1:02d}_{block_name}_overlap_{overlap_value:.3f}_reuse_{reuse_value}",
      directory=output_dir,
    )
    plt.close(fig)  # Close the figure to free memory


if __name__ == "__main__":
  from data_processing import process_model_data
  from data_processing import process_user_data

  # Define thresholds to analyze
  thresholds = [
    0.25,
    0.5,
    0.7,
  ]

  # Define manipulations
  manipulations = [
    {
      "name": "two_paths",
      "manipulation_id": 3,
      "train_maze": "big_m3_maze1",
      "test_maze": "big_m3_maze1",
    },
    {
      "name": "shortcut",
      "manipulation_id": 1,
      "train_maze": "big_m1_maze3",
      "test_maze": "big_m1_maze3_shortcut",
    },
  ]

  # Define models
  models = [
    # "qlearning",
    # "bfs",
    "dfs",
    # "usfa",
    # "dyna",
    # "preplay"
    "preplay_new",
  ]
  model2algo = dict(
    preplay_new="preplay",
  )

  print("Loading model data...")
  # Load model data
  model_df = process_model_data.get_jaxmaze_model_data(
    load_df_only=False,
    models=models,
  )
  # Process model data for each manipulation, threshold, and model
  for manipulation in manipulations:
    manip_name = manipulation["name"]
    manip_id = manipulation["manipulation_id"]
    train_maze = manipulation["train_maze"]
    test_maze = manipulation["test_maze"]

  # Process each model
  for model in models:
    print(f"\nProcessing {model} model for {manip_name}")

    # Process each threshold for this model
    for threshold in thresholds:
      print(f"Processing {manip_name} with threshold {threshold} for {model}")
      visualize_examples_by_reuse(
        df=model_df.filter(algo=model2algo.get(model, model)),
        manipulation_id=manip_id,
        threshold=threshold,
        model=model,
        train_maze=train_maze,
        test_maze=test_maze,
      )

  # Load human data
  print("Loading human data...")
  user_df = process_user_data.get_jaxmaze_human_data(
    # overwrite_episode_df=True,
    load_df_only=False,
  )

  # Process human data for each manipulation and threshold
  for manipulation in manipulations:
    manip_name = manipulation["name"]
    manip_id = manipulation["manipulation_id"]
    train_maze = manipulation["train_maze"]
    test_maze = manipulation["test_maze"]

    # Process each threshold for human data
    for threshold in thresholds:
      print(f"\nProcessing {manip_name} with threshold {threshold} for humans")
      visualize_examples_by_reuse(
        df=user_df,  # Pass the full dataframe for humans
        manipulation_id=manip_id,
        threshold=threshold,
        model="human",
        train_maze=train_maze,
        test_maze=test_maze,
      )
