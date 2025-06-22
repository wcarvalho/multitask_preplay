import os
import os.path
import json
from glob import glob
from typing import NamedTuple, Callable
import time

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
    filename = os.path.join(data_dir, "final", f"{split_name}_episode_df.csv")
    if os.path.exists(filename):
      print(f"Skipping {split_name} data because it already exists")
      continue
    split_data.to_pandas().to_csv(filename, index=False)
    print(f"Saved {split_name} data to {filename}")


def download_jaxmaze_data():
  download_data(
    data_dir=data_configs.JAXMAZE_DATA_DIR,
    dataset_name=data_configs.HUGGINGFACE_JAXMAZE_DATASET_NAME,
  )


def download_craftax_data():
  download_data(
    data_dir=data_configs.CRAFTAX_DATA_DIR,
    dataset_name=data_configs.HUGGINGFACE_CRAFTAX_DATASET_NAME,
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


def create_maps(episode_data_list, start_pos=0):
  maps = []
  for episode_data in episode_data_list:
    timesteps = episode_data.timesteps

    # [T, H, W, 1]
    # Assuming grid is 3D with time as first dimension
    grid_shape = timesteps.state.grid.shape

    # skip the time dimension and final channel dimension
    grid = jnp.zeros(grid_shape[1:-1], dtype=jnp.int32)

    # go through each position and set the corresponding index to 1
    for pos in episode_data.positions[start_pos:]:
      grid = grid.at[pos[0], pos[1]].set(1)
    maps.append(grid)
  return np.array(maps)


def compute_overlap(map1: np.ndarray, map2: np.ndarray):
  """map1: HxW, map2: HxW"""
  """Calculate the overlap between two maps."""
  nonzero_indices = np.argwhere(map2 > 0)
  values_map1 = (map1[nonzero_indices[:, 0], nonzero_indices[:, 1]] > 0).astype(
    np.float32
  )
  values_map2 = (map2[nonzero_indices[:, 0], nonzero_indices[:, 1]] > 0).astype(
    np.float32
  )
  overlap = (values_map1 + values_map2) > 1
  return overlap


def add_reuse_dicts_to_df(df, all_reuse_dicts, all_overlap_dicts):
  """Add reuse and overlap columns to a DataFrame using the provided dictionaries.

  Args:
      df (pl.DataFrame): The DataFrame to modify
      all_reuse_dicts (list of dicts): List of dictionaries mapping (maze, global_episode_idx) to reuse values
      all_overlap_dicts (list of dicts): List of dictionaries mapping (maze, global_episode_idx) to overlap values

  Returns:
      pl.DataFrame: The modified DataFrame with reuse and overlap columns
  """

  # Combine all dictionaries
  final_reuse_dict = {k: v for d in all_reuse_dicts for k, v in d.items()}
  final_overlap_dict = {k: v for d in all_overlap_dicts for k, v in d.items()}

  return df.with_columns(
    [
      # For reuse column
      pl.struct(["maze", "global_episode_idx"])
      .map_elements(
        lambda s: final_reuse_dict.get((s["maze"], s["global_episode_idx"]), -1),
        return_dtype=pl.Int32,
      )
      .alias("reuse"),
      # For overlap column
      pl.struct(["maze", "global_episode_idx"])
      .map_elements(
        lambda s: final_overlap_dict.get(
          (s["maze"], s["global_episode_idx"]), float("nan")
        ),
        return_dtype=pl.Float64,
      )
      .alias("overlap"),
    ]
  )
