"""
This script generates figures for the JaxMaze overlap analysis.

Call from root directory with:
python figures_supplemental/jaxmaze_overlap_analysis.py

The figures are saved in the {data_configs.JAXMAZE_OVERLAP_ANALYSIS_DIR} directory.
"""

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
  n_search: int = 10,
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
  manipulation_name = manipulation_id

  # Create directory structure
  output_dir = f"{data_configs.JAXMAZE_OVERLAP_ANALYSIS_DIR}/{manipulation_name}_{threshold}/{model}"

  # Delete existing directory to remove old data
  import shutil

  if os.path.exists(output_dir):
    print(f"Removing old data from {output_dir}...")
    shutil.rmtree(output_dir)

  # Create fresh directory
  os.makedirs(output_dir, exist_ok=True)

  # Different handling for human data vs model data
  samples = []

  # if model == "human":
  # For humans, we need to carefully match train and test examples
  user_ids = df["user_id"].unique().to_list()
  block_names = df["block_name"].unique().to_list()
  for user_id in user_ids:
    # Get test examples for this user
    for block_name in block_names:
      test_initial = df.filter(
        eval=True,
        user_id=user_id,
        manipulation=manipulation_id,
        block_name=block_name,  # easiest to visualize
        world=test_maze,
      ).sort("overlap", descending=True)

      if len(test_initial.episodes) == 0:
        continue
      # Process each test episode one by one
      ntest = len(test_initial.episodes)
      for test_episode_idx in range(min(n_search, ntest)):
        # Get values for filtering
        start_pos_val = test_initial["start_pos"].to_list()[test_episode_idx]
        block_name_val = test_initial["block_name"].to_list()[test_episode_idx]
        task_set_val = test_initial["task_set"].to_list()[test_episode_idx]

        filters = dict(
          start_pos=start_pos_val,
          block_name=block_name_val,
          user_id=user_id,
          task_set=task_set_val,
        )
        # Get properly filtered test and train
        test = df.filter(
          eval=True, manipulation=manipulation_id, world=test_maze, **filters
        )
        corresponding_train_episode_idx = test[
          "corresponding_train_episode_idx"
        ].to_list()[test_episode_idx]

        train = df.filter(
          # eval=False,
          # manipulation=manipulation_id,
          # world=train_maze,
          global_episode_idx=corresponding_train_episode_idx,
        )

        if len(train.episodes) == 0 or len(test.episodes) == 0:
          continue

        # Calculate overlap
        # just use 1 example
        train_map = create_maps([train.episodes[0]]).sum(0)
        test_map = create_maps([test.episodes[0]]).sum(0)

        overlap_value = compute_overlap(train_map, test_map).mean()
        df_overlap_value = test["overlap"].to_list()[0]
        reuse = int(overlap_value >= threshold)

        samples.append(
          {
            "key": user_id,
            "train": train,
            "test": test,
            "overlap": overlap_value,
            "df_overlap": df_overlap_value,
            "reuse": reuse,
            "filters": filters,  # Store filters for reference
            "block_name": block_name_val,
          }
        )

  # Separate into above and below threshold examples (regardless of actual reuse classification)
  above_threshold_samples = [s for s in samples if s["overlap"] >= threshold]
  below_threshold_samples = [s for s in samples if s["overlap"] < threshold]

  if len(above_threshold_samples) == 0 and len(below_threshold_samples) == 0:
    raise RuntimeError("No examples found")

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
    df_overlap_value = sample["df_overlap"]
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
      f"Test (success: {test_success}, overlap: {overlap_value:.3f}, df_overlap: {df_overlap_value:.3f}, reuse: {reuse_value})"
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
    df_overlap_value = sample["df_overlap"]
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
      f"Test (success: {test_success}, overlap: {overlap_value:.3f}, df_overlap: {df_overlap_value:.3f}, reuse: {reuse_value})"
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
  import argparse
  from data_processing import process_model_data
  from data_processing import process_user_data

  # Parse command line arguments
  parser = argparse.ArgumentParser(
    description="Generate JaxMaze overlap analysis visualizations"
  )
  parser.add_argument(
    "--models",
    nargs="+",
    default=["preplay", "dfs", "human"],
    choices=["human", "preplay", "usfa", "dyna", "qlearning", "bfs", "dfs"],
    help="Models to analyze (can specify multiple). Use 'human' for human data.",
  )
  parser.add_argument(
    "--manipulations",
    nargs="+",
    default=["two_paths", "shortcut"],
    choices=["two_paths", "shortcut"],
    help="Models to analyze (can specify multiple). Use 'human' for human data.",
  )
  parser.add_argument(
    "--thresholds",
    nargs="+",
    type=float,
    default=[0.3, 0.5, 0.6, 0.7],
    help="Overlap thresholds to analyze (default: 0.25 0.5 0.7)",
  )
  parser.add_argument(
    "--n",
    type=int,
    default=10,
    help="Number of examples to generate for each reuse category (default: 10)",
  )

  args = parser.parse_args()

  # Define thresholds to analyze
  thresholds = args.thresholds

  # Define manipulations
  manipulations = [
    {
      "name": "two_paths",
      "manipulation_id": "paths",
      "train_maze": "big_m3_maze1",
      "test_maze": "big_m3_maze1",
    },
    {
      "name": "shortcut",
      "manipulation_id": "shortcut",
      "train_maze": "big_m1_maze3",
      "test_maze": "big_m1_maze3_shortcut",
    },
  ]
  if args.manipulations:
    manipulations = [m for m in manipulations if m["name"] in args.manipulations]

  # Define models available for JaxMaze (excluding human which is handled separately)
  available_models = ["preplay", "usfa", "dyna", "qlearning", "bfs", "dfs"]
  requested_models = [m for m in args.models if m != "human"]

  model2algo = dict(
    preplay_new="preplay",
  )

  # Load and process model data if any models were requested
  if requested_models:
    print("Loading model data...")
    model_df = process_model_data.get_jaxmaze_model_data(
      load_df_only=False,
      models=requested_models,
    )

    # Process model data for each manipulation, threshold, and model
    for manipulation in manipulations:
      manip_name = manipulation["name"]
      manip_id = manipulation["manipulation_id"]
      train_maze = manipulation["train_maze"]
      test_maze = manipulation["test_maze"]

      # Process each model
      for model in requested_models:
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
            n_examples=args.n,
          )

  # Load and process human data if requested
  if "human" in args.models:
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
          n_examples=args.n,
        )

  print("Analysis complete!")
