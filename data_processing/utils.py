import os
import os.path
import sys
import json
from glob import glob
from typing import NamedTuple, Callable
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
import polars as pl
from datasets import load_dataset
from absl import logging
from flax import serialization, struct
from serialization import SerializationWrapper
import jax
import jax.numpy as jnp

import numpy as np
import data_configs


def download_data(
  data_dir: str,
  dataset_name: str,
):
  dataset = load_dataset(f"wcarvalho/{dataset_name}")

  # Create directories
  os.makedirs(os.path.join(data_dir, "final"), exist_ok=True)

  # Save each split as CSV
  for split_name, split_data in dataset.items():
    filename = os.path.join(data_dir, "final", f"{split_name}_episode_df.parquet")
    if os.path.exists(filename):
      print(f"Skipping {split_name} data because it already exists")
      continue
    split_data.to_pandas().to_parquet(filename)
    print(f"Saved {split_name} data to {filename}")


def download_jaxmaze_data():
  print("Data will be downloaded or saved to", data_configs.JAXMAZE_MODEL_DATA_DIR)
  download_data(
    data_dir=data_configs.JAXMAZE_MODEL_DATA_DIR,
    dataset_name=f"{data_configs.HUGGINGFACE_JAXMAZE_DATASET_NAME}_human",
  )
  download_data(
    data_dir=data_configs.JAXMAZE_MODEL_DATA_DIR,
    dataset_name=f"{data_configs.HUGGINGFACE_JAXMAZE_DATASET_NAME}_models",
  )


def download_craftax_data():
  print("Data will be downloaded or saved to", data_configs.CRAFTAX_MODEL_DATA_DIR)
  download_data(
    data_dir=data_configs.CRAFTAX_MODEL_DATA_DIR,
    dataset_name=f"{data_configs.HUGGINGFACE_CRAFTAX_DATASET_NAME}_human",
  )
  download_data(
    data_dir=data_configs.CRAFTAX_MODEL_DATA_DIR,
    dataset_name=f"{data_configs.HUGGINGFACE_CRAFTAX_DATASET_NAME}_models",
  )


class EpisodeData(NamedTuple):
  actions: jax.Array
  timesteps: struct.PyTreeNode
  positions: jax.Array = None
  reaction_times: jax.Array = None
  transitions: struct.PyTreeNode = None


def load_episode_data(filename: str, example_timestep: struct.PyTreeNode):
  """Load list of episodes from safetensor file."""
  start_time = time.time()
  with open(filename, "rb") as f:
    serialized_data = f.read()
    # Create template episode for deserialization
    example_episode = EpisodeData(
      actions=jnp.zeros((1,)),
      positions=jnp.zeros((1, 2)),
      timesteps=example_timestep,
      reaction_times=None,
      transitions=None,
    )
    # Two-step deserialization
    attempt1 = serialization.from_bytes(None, serialized_data)
    nepisodes = len(attempt1)
    episode_data = serialization.from_bytes(
      [SerializationWrapper(example_episode)] * nepisodes, serialized_data
    )
  logging.info(
    f"Loaded episode data for {os.path.basename(filename)} in {time.time() - start_time} seconds"
  )
  return episode_data


def get_in_episode(timestep):
  # get mask for within episode
  non_terminal = timestep.discount
  is_last = timestep.last()
  term_cumsum = jnp.cumsum(is_last, -1)
  in_episode = (term_cumsum + non_terminal) < 2
  return in_episode


def success(e: EpisodeData):
  in_episode = get_in_episode(e.timesteps)
  rewards = e.timesteps.reward[in_episode]
  # return rewards
  assert rewards.ndim == 1, "this is only defined over vector, e.g. 1 episode"
  success = rewards > 0.5
  return success.any().astype(jnp.float32)


def path_length(e: EpisodeData):
  in_episode = get_in_episode(e.timesteps)
  return sum(in_episode)


def total_reward(e: EpisodeData):
  in_episode = get_in_episode(e.timesteps)
  return e.timesteps.reward[in_episode].sum()


def compute_overlap(map1: np.ndarray, map2: np.ndarray):
  """map1: HxW, map2: HxW"""
  """Calculate the overlap between two maps."""
  shared_length = ((map2 + map1) * map2 > 1).sum()
  map2_length = (map2 > 0).sum()
  overlap = shared_length / map2_length
  return overlap


def best_path_overlap(maps: np.ndarray, test_maps: np.ndarray):
  """maps: NxHxW, test_map: HxW"""
  """Calculate the overlap between two maps."""
  # Find the train episode with highest overlap
  overlaps = []
  for i, map in enumerate(maps):
    if test_maps.ndim == 2:
      overlap = compute_overlap(map, test_maps).mean()
      overlaps.append(overlap)
    else:
      test_overlaps = []
      for test_map in test_maps:
        test_overlap = compute_overlap(map, test_map).mean()
        test_overlaps.append(test_overlap)
      min_overlap = np.min(test_overlaps)
      if min_overlap < 0.15:
        overlaps.append(0)
      else:
        overlaps.append(min_overlap)

  # Select the train episode/map with highest overlap
  best_idx = np.argmax(overlaps)
  best_map = maps[best_idx]
  best_overlap = overlaps[best_idx]
  return best_idx, best_map, best_overlap


def compute_average_vector_from_end(positions):
  """Compute the average vector from the last half of the trajectory.

  Args:
    positions: List of (x, y) positions

  Returns:
    Tuple of (average_vector, vectors) where average_vector is (dx, dy)
  """

  if len(positions) < 2:
    return (0, 0), []

  # Calculate pairwise vectors from adjacent positions
  vectors = []
  for i in range(len(positions) - 1):
    dy = -(positions[i + 1][0] - positions[i][0])
    dx = positions[i + 1][1] - positions[i][1]
    vectors.append((dx, dy))

  # Get the last half of vectors
  vectors = np.array(vectors)

  last_n = min(10, len(vectors) // 3)
  vector_end = vectors[-last_n:]

  # Calculate average vector
  avg_vector = np.mean(vector_end, axis=0)
  avg_dx = avg_vector[0]
  avg_dy = avg_vector[1]

  return (avg_dx, avg_dy), vector_end


def compute_cosine_similarity(train_positions, test_positions):
  train_avg_vector, train_vectors = compute_average_vector_from_end(train_positions)
  test_avg_vector, test_vectors = compute_average_vector_from_end(test_positions)
  cosine = np.dot(train_avg_vector, test_avg_vector) / (
    np.linalg.norm(train_avg_vector) * np.linalg.norm(test_avg_vector) + 1e-5
  )
  return cosine


def get_overlap_dicts_human(
  df,
  train_maze,
  test_maze,
  overlap_threshold,
  create_train_maps_fn=None,
  create_test_maps_fn=None,
):
  """Get reuse and overlap dictionaries for a given DataFrame.

  In human case, a block will have a single test episode but could have multiple corresponding train episodes.
  """
  overlap_dict = {}
  reuse_dict = {}
  corresponding_train_episode_idx = {}
  cosine_dict = {}

  if create_test_maps_fn is None:
    create_test_maps_fn = create_train_maps_fn

  # Get unique users
  block_names = df.filter(world=test_maze)["block_name"].unique().to_list()
  for block_name in block_names:
    # Get test episodes
    test = df.filter(world=test_maze, block_name=block_name, eval=True)
    if len(test) == 0:
      continue
    start_pos = test["start_pos"].to_list()[0]

    # Get train episodes
    train = df.filter(
      world=train_maze,
      block_name=block_name,
      task_set=0,
      eval=False,
      success=1,
      start_pos=start_pos,
    )

    if len(train.episodes) == 0:
      continue

    # Create map for training episodes
    train_maps = create_train_maps_fn(train.episodes)

    # Process each test episode
    for idx, row in enumerate(test._df.iter_rows(named=True)):
      global_index = row["global_episode_idx"]
      episode = test.episodes[idx]
      # Create map for single test episode
      test_map = create_test_maps_fn([episode]).sum(0)
      best_idx, best_map, best_overlap = best_path_overlap(train_maps, test_map)
      overlap_mean = best_overlap

      # Store the reuse value
      episode_id = (test_maze, global_index)
      corresponding_train_episode_idx[episode_id] = train[
        "global_episode_idx"
      ].to_list()[best_idx]
      overlap_dict[episode_id] = overlap_mean
      reuse_dict[episode_id] = int(overlap_mean > overlap_threshold)
      cosine_dict[episode_id] = compute_cosine_similarity(
        train.episodes[best_idx].positions, test.episodes[idx].positions
      )

  return reuse_dict, overlap_dict, corresponding_train_episode_idx, cosine_dict


def _debug_plot_nan_overlap(
  train_map,
  test_map,
  train_episode,
  test_episode,
  row,
  overlap,
  idx,
  block_name,
  train_maze,
  test_maze,
):
  """Debug plot when overlap is NaN during processing."""
  import matplotlib.pyplot as plt
  from matplotlib.patches import Patch

  fig, axes = plt.subplots(1, 3, figsize=(15, 5))

  algo = row.get("algo", "?")
  seed = row.get("user_id", "?")
  world = row.get("world", "?")
  episode_idx = row.get("global_episode_idx", "?")

  # Panel 1: train map
  ax = axes[0]
  ax.imshow(np.array(train_map), cmap="Blues", interpolation="nearest")
  train_cells = int((np.array(train_map) > 0).sum())
  ax.set_title(
    f"Train map (cells={train_cells})\npositions={len(train_episode.positions)}"
  )

  # Panel 2: test map
  ax = axes[1]
  ax.imshow(np.array(test_map), cmap="Reds", interpolation="nearest")
  test_cells = int((np.array(test_map) > 0).sum())
  ax.set_title(
    f"Test map (cells={test_cells})\npositions={len(test_episode.positions)}"
  )

  # Panel 3: combined overlap
  ax = axes[2]
  combined = np.zeros_like(np.array(train_map), dtype=np.int32)
  combined += (np.array(train_map) > 0).astype(np.int32) * 1
  combined += (np.array(test_map) > 0).astype(np.int32) * 2
  cmap = plt.cm.colors.ListedColormap(["white", "blue", "red", "purple"])
  bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
  norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
  ax.imshow(combined, cmap=cmap, norm=norm, interpolation="nearest")
  ax.set_title(f"Overlap={overlap:.4f}\nidx={idx}, block={block_name}")
  ax.legend(
    handles=[
      Patch(facecolor="blue", label="Train only"),
      Patch(facecolor="red", label="Test only"),
      Patch(facecolor="purple", label="Shared"),
    ],
    loc="upper right",
    fontsize=8,
  )

  fig.suptitle(
    f"NaN DEBUG | algo={algo} | seed={seed} | world={world} | ep={episode_idx}\n"
    f"train_maze={train_maze} | test_maze={test_maze}",
    fontsize=11,
  )
  plt.tight_layout()
  plt.show()


def get_overlap_dicts_model(
  df,
  train_maze,
  test_maze,
  overlap_threshold,
  create_train_maps_fn=None,
  create_test_maps_fn=None,
):
  """Get reuse and overlap dictionaries for a given DataFrame.

  In model case, a block will have a single test episode and a single corresponding train episode.
  """
  overlap_dict = {}
  reuse_dict = {}
  corresponding_train_episode_idx = {}
  cosine_dict = {}

  if create_test_maps_fn is None:
    create_test_maps_fn = create_train_maps_fn
  # Get unique users
  block_names = df.filter(world=test_maze)["block_name"].unique().to_list()
  for block_name in block_names:
    # Get test episodes
    test = df.filter(world=test_maze, block_name=block_name, eval=True)
    if len(test) == 0:
      continue
    start_pos = test["start_pos"].to_list()[0]

    # Get train episodes
    train = df.filter(
      world=train_maze,
      block_name=block_name,
      task_set=0,
      eval=False,
      success=1,
      start_pos=start_pos,
    )

    if len(train.episodes) == 0:
      continue

    # Create map for training episodes
    train_maps = create_train_maps_fn(train.episodes)
    test_maps = create_test_maps_fn(test.episodes)
    import ipdb; ipdb.set_trace()
    # Process each test episode
    for idx, row in enumerate(test._df.iter_rows(named=True)):
      global_index = row["global_episode_idx"]
      if not (idx < len(test_maps) and idx < len(train_maps)):
        continue
      test_map = test_maps[idx]
      train_map = train_maps[idx]
      overlap = compute_overlap(train_map, test_map)
      overlap_mean = overlap.mean()
      if np.isnan(overlap):
        _debug_plot_nan_overlap(
          train_map=train_map,
          test_map=test_map,
          train_episode=train.episodes[idx],
          test_episode=test.episodes[idx],
          row=row,
          overlap=overlap,
          idx=idx,
          block_name=block_name,
          train_maze=train_maze,
          test_maze=test_maze,
        )
        import ipdb

        ipdb.set_trace()

      # Store the reuse value
      episode_id = (test_maze, global_index)
      corresponding_train_episode_idx[episode_id] = train[
        "global_episode_idx"
      ].to_list()[idx]
      overlap_dict[episode_id] = overlap_mean
      reuse_dict[episode_id] = int(overlap_mean > overlap_threshold)
      cosine_dict[episode_id] = compute_cosine_similarity(
        train.episodes[idx].positions, test.episodes[idx].positions
      )

  return reuse_dict, overlap_dict, corresponding_train_episode_idx, cosine_dict


def add_reuse_dicts_to_df(
  df,
  all_reuse_dicts,
  all_overlap_dicts,
  all_corresponding_train_episode_idx,
  all_cosine_dicts,
  all_approach_direction_dicts=None,
):
  """Add reuse and overlap columns to a DataFrame using the provided dictionaries.

  Args:
      df (pl.DataFrame): The DataFrame to modify
      all_reuse_dicts (list of dicts): List of dictionaries mapping (maze, global_episode_idx) to reuse values
      all_overlap_dicts (list of dicts): List of dictionaries mapping (maze, global_episode_idx) to overlap values
      all_corresponding_train_episode_idx (list of dicts): List of dictionaries mapping to train episode index
      all_cosine_dicts (list of dicts): List of dictionaries mapping to cosine similarity
      all_approach_direction_dicts (list of dicts, optional): List of dictionaries mapping to approach direction

  Returns:
      pl.DataFrame: The modified DataFrame with reuse and overlap columns
  """

  # Combine all dictionaries
  final_reuse_dict = {k: v for d in all_reuse_dicts for k, v in d.items()}
  final_overlap_dict = {k: v for d in all_overlap_dicts for k, v in d.items()}
  final_corresponding_train_episode_idx = {
    k: v for d in all_corresponding_train_episode_idx for k, v in d.items()
  }
  final_cosine_dict = {k: v for d in all_cosine_dicts for k, v in d.items()}

  # Handle approach direction (optional, only for jaxmaze human data)
  if all_approach_direction_dicts is not None:
    final_approach_direction_dict = {
      k: v for d in all_approach_direction_dicts for k, v in d.items()
    }
  else:
    final_approach_direction_dict = {}

  columns = [
    # ===========
    # For reuse column
    # ===========
    pl.struct(["world", "global_episode_idx"])
    .map_elements(
      lambda s: final_reuse_dict.get((s["world"], s["global_episode_idx"]), -1),
      return_dtype=pl.Int32,
    )
    .alias("reuse"),
    # ===========
    # For overlap column
    # ===========
    pl.struct(["world", "global_episode_idx"])
    .map_elements(
      lambda s: final_overlap_dict.get(
        (s["world"], s["global_episode_idx"]), float("nan")
      ),
      return_dtype=pl.Float64,
    )
    .alias("overlap"),
    # ===========
    # For corresponding train episode index column
    # ===========
    pl.struct(["world", "global_episode_idx"])
    .map_elements(
      lambda s: final_corresponding_train_episode_idx.get(
        (s["world"], s["global_episode_idx"]), -1
      ),
      return_dtype=pl.Int32,
    )
    .alias("corresponding_train_episode_idx"),
    # ===========
    # For cosine similarity column
    # ===========
    pl.struct(["world", "global_episode_idx"])
    .map_elements(
      lambda s: final_cosine_dict.get(
        (s["world"], s["global_episode_idx"]), float("nan")
      ),
      return_dtype=pl.Float64,
    )
    .alias("train_test_cosine"),
  ]

  # Add approach direction column if we have data for it
  if final_approach_direction_dict:
    columns.append(
      pl.struct(["world", "global_episode_idx"])
      .map_elements(
        lambda s: final_approach_direction_dict.get(
          (s["world"], s["global_episode_idx"]), "unknown"
        ),
        return_dtype=pl.String,
      )
      .alias("approach_direction")
    )

  return df.with_columns(columns)


def reorder_columns(df, front_cols):
  remaining_cols = [col for col in df.columns if col not in front_cols]
  return df.select(front_cols + remaining_cols)
