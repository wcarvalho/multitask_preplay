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
import figure_utils
import data_configs
import nicewebrl
import numpy as np

# def compute_overlap_df(train, test):
#  train_map = utils.create_maps(train.episodes, start_pos=4).sum(0)
#  test_map = utils.create_maps(test.episodes, start_pos=4).sum(0)
#  print("n episodes", len(train.episodes), len(test.episodes))
#  print("path lengths", len(np.argwhere(train_map > 0)), len(np.argwhere(test_map > 0)))
#  overlap = utils.compute_overlap(train_map, test_map)
#  return overlap

# def visualize_user_overlap(df: nicewebrl.DataFrame, n=3):
#  user_ids = df['user_id'].unique().to_list()
#  fig, axs = plt.subplots(n, 2, figsize=(10, 5*n))

#  if n == 1:
#    axs = axs.reshape(1, 2)

#  idx = -1
#  success_count = 0
#  while True:
#    idx += 1

#    user_id = user_ids[idx]
#    test = df.filter(eval=True,
#                    user_id=user_id,
#                    #eval_shares_start_pos=True
#                    )
#    # test.head()
#    filters = dict(
#        start_pos=test["start_pos"].to_list()[0],
#        block_name=test['block_name'].to_list()[0],
#        user_id=test['user_id'].to_list()[0],
#        room=test['room'].to_list()[0],
#    )
#    # print(filters)
#    test = df.filter(eval=True, manipulation=1,
#      #eval_shares_start_pos=True,
#      **filters)
#    train = df.filter(eval=False, manipulation=1, success=1, **filters)

#    if len(train.episodes) == 0 or len(test.episodes) == 0:
#      print(f"No episodes for user {user_id}")
#      continue
#    print(f"Making plot for user {user_id}")


#    # Use the first episode for each
#    #train_map = create_maps(train.episodes).sum(0)
#    #test_map = create_maps(test.episodes).sum(0)

#    #overlap = compute_overlap(train_map, test_map)
#    overlap = compute_overlap_df(train, test)
#    overlap_value = overlap.mean()

#    if 'reuse' in test.columns:
#      reuse = test['reuse'].to_list()[0]
#      overlap_df = test['overlap'].to_list()[0]
#    else:
#      reuse = None
#      overlap_df = None

#    vis_utils.render_path(train.episodes[0], ax=axs[success_count, 0])
#    vis_utils.render_path(test.episodes[0], ax=axs[success_count, 1])

#    train_success = train['success'].to_list()[0]
#    test_success = test['success'].to_list()[0]
#    axs[success_count, 0].set_title(f"Train (user: {user_id}, success: {train_success})")
#    axs[success_count, 1].set_title(f"Test (success: {test_success}, overlap: {overlap_value:.4}\noverlap_df: {overlap_df}, reuse: {reuse})")
#    success_count += 1
#    if success_count >= n:
#      break

#  #fig.tight_layout()
#  figure_utils.save_figure(
#    fig,
#    "human_overlap",
#    directory=f"{data_configs.DIRECTORY}/figures/jaxmaze_overlap_reuse",
#  )
#  return fig, train, test


def calculate_reuse(overlap_value, threshold):
  """Calculate reuse based on overlap and threshold."""
  return 1 if overlap_value >= threshold else 0


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
      test_initial = df.filter(eval=True, user_id=user_id, manipulation=manipulation_id)

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

        overlap = compute_overlap(train_map, test_map)
        overlap_value = overlap.mean()

        reuse = calculate_reuse(overlap_value, threshold)

        samples.append(
          {
            "key": user_id,
            "train": train,
            "test": test,
            "overlap": overlap_value,
            "reuse": reuse,
            "filters": filters,  # Store filters for reference
          }
        )
  else:
    # For models, process as before - each seed has one train and one test
    train = df.filter(eval=False, manipulation=manipulation_id, maze=train_maze)
    test = df.filter(eval=True, manipulation=manipulation_id, maze=test_maze)

    # Get available keys (seeds)
    model_key = "seed"
    keys = train[model_key].unique().to_list()

    for k in keys:
      train_filtered = train.filter(**{model_key: k})
      test_filtered = test.filter(**{model_key: k})

      if len(train_filtered.episodes) == 0 or len(test_filtered.episodes) == 0:
        continue

      # Calculate overlap
      train_map = create_maps([train_filtered.episodes[0]]).sum(0)
      test_map = create_maps([test_filtered.episodes[0]]).sum(0)

      overlap = compute_overlap(train_map, test_map)
      overlap_value = overlap.mean()

      reuse = calculate_reuse(overlap_value, threshold)

      samples.append(
        {
          "key": k,
          "train": train_filtered,
          "test": test_filtered,
          "overlap": overlap_value,
          "reuse": reuse,
        }
      )

  # Separate into reuse=1 and reuse=0 examples
  reuse_1_samples = [s for s in samples if s["reuse"] == 1]
  reuse_0_samples = [s for s in samples if s["reuse"] == 0]

  print(f"{model} - {manipulation_name} - threshold {threshold}:")
  print(f"  Found {len(reuse_1_samples)} examples with reuse=1")
  print(f"  Found {len(reuse_0_samples)} examples with reuse=0")

  # Limit to n_examples
  reuse_1_samples = reuse_1_samples[:n_examples]
  reuse_0_samples = reuse_0_samples[:n_examples]

  # Create individual figures for reuse=1 examples
  for idx, sample in enumerate(reuse_1_samples):
    k = sample["key"]
    train_filtered = sample["train"]
    test_filtered = sample["test"]
    overlap_value = sample["overlap"]

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
      f"Test (success: {test_success}, overlap: {overlap_value:.3f}, reuse: 1)"
    )

    fig.tight_layout()
    # Save with numbered filename including overlap value
    figure_utils.save_figure(
      fig,
      f"reuse_1_example_{idx + 1:02d}_overlap_{overlap_value:.3f}",
      directory=output_dir,
    )
    plt.close(fig)  # Close the figure to free memory

  # Create individual figures for reuse=0 examples
  for idx, sample in enumerate(reuse_0_samples):
    k = sample["key"]
    train_filtered = sample["train"]
    test_filtered = sample["test"]
    overlap_value = sample["overlap"]

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
      f"Test (success: {test_success}, overlap: {overlap_value:.3f}, reuse: 0)"
    )

    fig.tight_layout()
    # Save with numbered filename including overlap value
    figure_utils.save_figure(
      fig,
      f"reuse_0_example_{idx + 1:02d}_overlap_{overlap_value:.3f}",
      directory=output_dir,
    )
    plt.close(fig)  # Close the figure to free memory


# def visualize_model_overlap(
#  train: nicewebrl.DataFrame, test: nicewebrl.DataFrame, name: str, key: str = 'seed', seeds=None
# ):
#  # Get seeds if not provided, limited to first 3
#  seeds = seeds or train[key].unique().to_list()[:3]

#  fig, axs = plt.subplots(len(seeds), 2, figsize=(10, 5*len(seeds)))

#  if len(seeds) == 1:
#    axs = axs.reshape(1, 2)

#  for idx, seed in enumerate(seeds):
#    if idx >= 3:
#      break

#    # Filter data for this seed
#    train_seed_df = train.filter(**{key: seed})
#    test_seed_df = test.filter(**{key: seed})

#    if len(train_seed_df.episodes) == 0 or len(test_seed_df.episodes) == 0:
#      continue

#    # Use the first episode for each
#    train_map = create_maps([train_seed_df.episodes[0]]).sum(0)
#    test_map = create_maps([test_seed_df.episodes[0]]).sum(0)

#    overlap = compute_overlap(train_map, test_map)
#    overlap_value = overlap.mean()

#    reuse = test_seed_df['reuse'].to_list()[0] if 'reuse' in test_seed_df.columns else None

#    vis_utils.render_path(train_seed_df.episodes[0], ax=axs[idx, 0])
#    vis_utils.render_path(test_seed_df.episodes[0], ax=axs[idx, 1])

#    train_success = train_seed_df['success'].to_list()[0]
#    test_success = test_seed_df['success'].to_list()[0]
#    axs[idx, 0].set_title(f"Train (seed: {seed}, success: {train_success})")
#    axs[idx, 1].set_title(f"Test (success: {test_success}, overlap: {overlap_value:.3f}, reuse: {reuse})")
#  fig.tight_layout()
#  figure_utils.save_figure(
#    fig,
#    f"{name}_model_overlap",
#    directory=f"{data_configs.DIRECTORY}/figures/jaxmaze_overlap_reuse",
#  )
#  return fig

if __name__ == "__main__":
  from data_processing import process_model_data
  from data_processing import process_user_data

  # Define thresholds to analyze
  thresholds = [0.15, 0.5, 0.7]

  # Define manipulations
  manipulations = [
    {
      "name": "shortcut",
      "manipulation_id": 1,
      "train_maze": "big_m1_maze3",
      "test_maze": "big_m1_maze3_shortcut",
    },
    {
      "name": "two_paths",
      "manipulation_id": 3,
      "train_maze": "big_m3_maze1",
      "test_maze": "big_m3_maze1",
    },
  ]

  # Define models
  models = ["qlearning", "bfs", "dfs", "usfa", "dyna", "preplay"]

  print("Loading model data...")
  # Load model data
  model_df = process_model_data.get_jaxmaze_model_data(
    load_df_only=False,
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
          df=model_df.filter(algo=model),
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
