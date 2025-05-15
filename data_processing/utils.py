import os
import os.path
import json
from glob import glob
from typing import NamedTuple, Callable
import polars as pl

# Third-party imports
import time
from absl import logging
from flax import serialization, struct
import jax
import jax.numpy as jnp

import numpy as np
import nicewebrl


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
      [example_episode] * nepisodes, serialized_data
    )
  logging.info(
    f"Loaded episode data for {os.path.basename(filename)} in {time.time() - start_time} seconds"
  )
  return episode_data


# def get_model_data(
#  input_glob_pattern: str,
#  output_data_path: str,
#  example_timestep,
#  env_name: str,
#  model_name: str,
#  generate_all_episodes_data: Callable,
#  generate_all_episodes_df: Callable,
#  overwrite_episode_data=False,
#  overwrite_episode_df=False,
#  load_df_only=True,
#  debug=False,

# ):
#  """Process human data for a specific environment.

#  This function handles both loading pre-processed data and processing raw data.

#  Args:
#    data_path: Path to the raw data files
#    human_data_filebase: Base path for saving/loading processed data
#    example_timestep: Example timestep for deserialization
#    env_name: Environment name ("jaxmaze" or "craftax")
#    overwrite_episode_data: Whether to reprocess episode data
#    overwrite_episode_df: Whether to reprocess episode info
#    load_df_only: Whether to return only the DataFrame (not episodes)
#    debug: Whether to run in debug mode (processing fewer files)
#    parallel: Whether to process files in parallel

#  Returns:
#    Either a DataFrame of episode info or a tuple of (DataFrame, failed_files)
#  """
#  ################################################################
#  # Load data paths
#  ################################################################
#  failed_files = []

#  if debug:
#    all_episodes_data_filename = os.path.join(output_data_path, f"{model_name}_episodes.safetensor")
#    all_episodes_metadata_filename = os.path.join(output_data_path, f"{model_name}_episode_metadata.json")
#    all_episodes_df_filename = os.path.join(output_data_path, f"{model_name}_episode_df.csv")
#  else:
#    all_episodes_data_filename = os.path.join(output_data_path, 'debug', f"{model_name}_episodes.safetensor")
#    all_episodes_metadata_filename = os.path.join(output_data_path, 'debug', f"{model_name}_episode_metadata.json")
#    all_episodes_df_filename = os.path.join(output_data_path, 'debug', f"{model_name}_episode_df.csv")

#  #--------------------------------
#  # don't want to overwrite anything
#  #--------------------------------
#  if (not (overwrite_episode_data and overwrite_episode_df)
#      and os.path.exists(all_episodes_data_filename)
#      and os.path.exists(all_episodes_df_filename)):
#      if load_df_only:
#        all_episodes_df = pl.read_csv(all_episodes_df_filename)
#        return all_episodes_df, failed_files
#      else:
#        all_episodes_df = pl.read_csv(all_episodes_df_filename)
#        all_episode_data = load_episode_data(
#          filename=all_episodes_data_filename,
#          example_timestep=example_timestep)
#        return nicewebrl.DataFrame(all_episodes_df, all_episode_data), failed_files
#  #--------------------------------
#  # don't want to overwrite episode data but want to overwrite episode info
#  #--------------------------------
#  elif (not overwrite_episode_data
#        and overwrite_episode_df
#        and os.path.exists(all_episodes_data_filename)
#        and os.path.exists(all_episodes_metadata_filename)):

#    # load all episode data from single safetensor file
#    all_episode_data = load_episode_data(
#      filename=all_episodes_data_filename,
#      example_timestep=example_timestep)

#    # load all episode metadata from single json file
#    with open(all_episodes_metadata_filename, "r") as f:
#      all_episodes_metadata = json.load(f)
#      import ipdb; ipdb.set_trace()

#    all_episodes_df = generate_all_episodes_df(all_episode_data, all_episodes_metadata)

#    # Save updated episode info
#    all_episodes_df.write_csv(all_episodes_df_filename)

#    if load_df_only:
#      return all_episodes_df, failed_files
#    else:
#      return nicewebrl.DataFrame(all_episodes_df, all_episode_data), failed_files
#  #--------------------------------
#  # overwrite everything
#  #--------------------------------
#  elif overwrite_episode_data and overwrite_episode_df:
#    files = glob(input_glob_pattern)
#    if debug:
#      files = files[: max(int(len(files) * 0.1), 10)]

#    all_episode_data, all_episode_metadata = generate_all_episodes_data(
#      files=files,
#      example_timestep=example_timestep,
#      env_name=env_name,
#      debug=debug,
#      parallel=True,
#    )

#    all_episodes_df = generate_all_episodes_df(
#      all_episode_data,
#      all_episodes_metadata,
#      env_name=env_name,
#    )

#    # Save episode dataframe
#    all_episodes_df.write_csv(all_episodes_df_filename)

#    if load_df_only:
#      return all_episodes_df, failed_files
#    else:
#      return nicewebrl.DataFrame(all_episodes_df, all_episode_data), failed_files

#  else:
#    raise ValueError("Invalid overwrite_episode_data and overwrite_episode_info")


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
