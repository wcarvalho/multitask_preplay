"""
This script processes the user data and saves it to a parquet file.

Call from root directory with:
python data_processing/process_user_data.py --env jaxmaze --df
python data_processing/process_user_data.py --env craftax --df

"""

import os
import os.path
import sys

sys.path.append("simulations")


import json
from glob import glob
from collections import defaultdict
from typing import List
from datetime import datetime
import shutil

# Third-party imports
import time
from absl import logging
from flax import serialization
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from joblib import Parallel, delayed
import numpy as np
import polars as pl
import nicewebrl
from tqdm import tqdm

# Local application/library specific imports
from housemaze import utils
from housemaze.human_dyna import multitask_env
from housemaze.human_dyna import mazes
from housemaze.human_dyna import web_env
import data_configs

from data_processing.utils import add_reuse_dicts_to_df
from data_processing import utils_jaxmaze
from data_processing import utils_craftax
from data_processing.utils import EpisodeData, load_episode_data


def dict_to_string(data):
  # Convert each key-value pair to "key=value" format
  pairs = [f"{key}={value}" for key, value in data.items()]

  # Join all pairs with ", " separator
  return ", ".join(pairs)


def seperate_data_into_episodes(data: List[dict], env_name: str):
  """This function will group episodes by the values from get_block_stage_description

  The input i
  So for example, each episode with {'stage': "'not obvious' shortcut",
   'block': 'shortcut',
   'manipulation': 1,
   'episode_idx': 1,
   'eval': True}
   with go into its own list.
  """
  key_to_episodes_unprocessed = defaultdict(list)
  episode_idx = -1
  keys = set()
  episode_information = dict()
  # first group all of the data based on which (stage, block) its in
  for datum in data:
    # This function will be used to group datapoints of an individual episode
    if env_name == "jaxmaze":
      info = utils_jaxmaze.get_block_stage_description(datum)
    elif env_name == "craftax":
      info = utils_craftax.get_block_stage_description(datum)
    else:
      raise ValueError(f"Unknown environment: {env_name}")
    key = dict_to_string(info)
    if key not in keys:
      episode_idx += 1
      keys.add(key)
    info["user_episode_idx"] = episode_idx

    updated_key = dict_to_string(info)
    key_to_episodes_unprocessed[updated_key].append(datum)
    episode_information[updated_key] = info
  return key_to_episodes_unprocessed, episode_information


def time_diff(t1, t2) -> float:
  # Convert string timestamps to datetime objects
  if t1 is None or t2 is None:
    return np.nan
  t1 = datetime.strptime(t1, "%Y-%m-%dT%H:%M:%S.%fZ")
  t2 = datetime.strptime(t2, "%Y-%m-%dT%H:%M:%S.%fZ")

  # Calculate the time difference
  time_difference = t2 - t1

  # Convert the time difference to milliseconds
  return time_difference.total_seconds()


def compute_reaction_time(datum) -> float:
  # Calculate the time difference
  return time_diff(datum["data"]["image_seen_time"], datum["data"]["action_taken_time"])


def generate_file_episodes_data(file, example_timestep, env_name: str, debug=False):
  assert env_name in ["jaxmaze", "craftax"], f"Unknown environment data: {env_name}"

  try:
    data = nicewebrl.read_all_records_sync(file)
  except Exception as e:
    logging.warning(f"Failed to read records from {file}: {str(e)}")
    return None, None

  if len(data) < 2:
    return None, None

  file_metadata = data[-1]
  finished = file_metadata.get("finished", False)
  if not finished:
    return None, None

  # Set up environment utils
  if env_name == "jaxmaze":
    env_utils = utils_jaxmaze
  elif env_name == "craftax":
    env_utils = utils_craftax
  else:
    raise ValueError(f"Unknown environment: {env_name}")

  #####################
  # filenames
  #####################
  user_filename = file.split("/")[-1].split(".json")[0]
  base_path = file.split(user_filename)[0]

  # Ensure cache directory exists
  cache_dir = f"{base_path}/cache/"
  os.makedirs(cache_dir, exist_ok=True)

  if debug:
    episode_data_filename = (
      f"{base_path}/cache/{user_filename}_debug_episode_data.bytes"
    )
    episode_metadata_filename = (
      f"{base_path}/cache/{user_filename}_debug_episode_info.json"
    )
  else:
    episode_data_filename = f"{base_path}/cache/{user_filename}_episode_data.bytes"
    episode_metadata_filename = f"{base_path}/cache/{user_filename}_episode_info.json"

  if (
    os.path.exists(episode_data_filename)
    and os.path.exists(episode_metadata_filename)
    and not debug
  ):
    episode_data = load_episode_data(episode_data_filename, example_timestep)
    episode_metadata = json.load(open(episode_metadata_filename))
    return episode_data, episode_metadata

  #####################
  # filter out practice not-manipulation data
  #####################
  def filter_fn(datum):
    if "metadata" not in datum:
      return True
    desc = datum["metadata"]["block_metadata"]["desc"]
    manipulation = datum["metadata"]["block_metadata"].get("manipulation", None)
    if manipulation is None:
      return True
    if "practice" in desc:
      return True

    return False

  nbefore = len(data)
  data = [datum for datum in data if not filter_fn(datum)]

  if len(data) == 0:
    return None, None

  logging.info(f"Filtered {nbefore - len(data)} data points")
  #####################
  # separate data by block/stage
  #####################
  key_to_episodes_unprocessed, episode_information = seperate_data_into_episodes(
    data, env_name
  )
  idxs = [k["user_episode_idx"] for k in episode_information.values()]

  assert len(idxs) == len(set(idxs)), (
    f"user_episode_idx is not unique. {len(idxs)} vs {len(set(idxs))}. max={max(idxs)}"
  )

  start_time = time.time()
  episode_data = [None] * len(key_to_episodes_unprocessed.keys())
  episode_metadata = [None] * len(key_to_episodes_unprocessed.keys())

  for episode_key in key_to_episodes_unprocessed.keys():
    episode_idx = episode_information[episode_key]["user_episode_idx"]
    raw_data = key_to_episodes_unprocessed[episode_key]

    actions = jnp.asarray([datum["data"]["action_idx"] for datum in raw_data])
    nactions = len(actions)
    if nactions <= 1:
      logging.warning(f"Skipping episode {episode_idx} because it has <= 1 actions")
      continue

    # Compute reaction times
    reaction_times = [compute_reaction_time(datum) for datum in raw_data]
    reaction_times = jnp.asarray(reaction_times)

    # --------------------------------
    # generate episode metadata
    # --------------------------------
    episode_metadata[episode_idx] = dict(
      **raw_data[0]["metadata"],
      file=file,
      user_storage=file_metadata.get("user_storage", {}),
      name=raw_data[0]["name"],
      user_data=raw_data[0]["user_data"],
    )
    episode_metadata[episode_idx].update(episode_information[episode_key])

    # --------------------------------
    # generate episode data
    # --------------------------------
    timesteps = [
      env_utils.deserialize_timestep(datum, example_timestep) for datum in raw_data
    ]
    timesteps = jtu.tree_map(lambda *v: jnp.stack(v), *timesteps)

    step_nums = env_utils.get_step_number(timesteps)
    expected_step_num = jnp.arange(len(step_nums))
    timestep_order_correct = jnp.all(step_nums == expected_step_num)
    if not timestep_order_correct:
      # Get sorting indices
      sort_indices = jnp.argsort(step_nums)

      # Check if sorting fixes the sequence
      sorted_steps = step_nums[sort_indices]
      if jnp.all(sorted_steps == expected_step_num):
        # Fix the ordering of all relevant data
        actions = actions[sort_indices]
        timesteps = jtu.tree_map(
          lambda x: (
            x[sort_indices] if isinstance(x, (jnp.ndarray, np.ndarray)) else x
          ),
          timesteps,
        )
        logging.info(
          f"{user_filename}: Fixed step indices for episode {episode_key} through sorting"
        )
      else:
        logging.warning(
          f"{user_filename}: Skipping episode {episode_key} due to invalid step indices that cannot be fixed"
        )
        raise RuntimeError(
          f"{user_filename}: episode {episode_key} has faulty step indices: {step_nums}"
        )

    positions = env_utils.get_agent_position(timesteps)

    episode_data[episode_idx] = EpisodeData(
      actions=actions,
      positions=positions,
      timesteps=timesteps,
      reaction_times=reaction_times,
    )

  # filter out episodes with no actions
  episode_data = [e for e in episode_data if e is not None]
  episode_metadata = [e for e in episode_metadata if e is not None]

  # Save using Flax serialization
  with open(episode_data_filename, "wb") as f:
    serialized_data = serialization.to_bytes(episode_data)
    f.write(serialized_data)

  # Save episode metadata
  with open(episode_metadata_filename, "w") as f:
    json.dump(episode_metadata, f)

  logging.info(
    f"Saved episode data for {os.path.basename(file)} in {time.time() - start_time} seconds"
  )

  return episode_data, episode_metadata


def generate_all_episodes_data(
  paths,
  example_timestep,
  env_name: str,
  debug=False,
  parallel=True,
  n_jobs: int = -1,
):
  failed_files = []

  def process_file(file):
    try:
      episode_data, episode_metadata = generate_file_episodes_data(
        file,
        example_timestep,
        env_name=env_name,
        debug=debug,
      )
      if episode_data is None or episode_metadata is None:
        print(f"Failed to process file {file}")
        failed_files.append(file)
      return episode_data, episode_metadata
    except Exception as e:
      logging.error(f"Error processing file {file}: {e}")
      failed_files.append(file)
      if debug:
        raise e
      return None, None

  if parallel:
    results = Parallel(n_jobs=n_jobs)(delayed(process_file)(file) for file in paths)
  else:
    results = [process_file(file) for file in paths]

  all_episode_data = []
  all_episode_metadata = []

  for episode_data, episode_metadata in results:
    if episode_data is not None and episode_metadata is not None:
      all_episode_data.extend(episode_data)
      all_episode_metadata.extend(episode_metadata)

  return all_episode_data, all_episode_metadata, failed_files


def generate_all_episodes_df(
  all_episode_data, all_episode_metadata, env_name: str, debug=False
):
  assert env_name in ["jaxmaze", "craftax"], f"Unknown environment data: {env_name}"

  # Set up environment utils
  if env_name == "jaxmaze":
    env_utils = utils_jaxmaze
  elif env_name == "craftax":
    env_utils = utils_craftax
  else:
    raise ValueError(f"Unknown environment: {env_name}")

  all_episode_row_data = []
  for episode_data, episode_metadata in zip(all_episode_data, all_episode_metadata):
    episode_row_data = env_utils.make_human_episode_row_data(
      metadata=episode_metadata,
      timesteps=episode_data.timesteps,
    )
    episode_row_data["global_episode_idx"] = len(all_episode_row_data)
    all_episode_row_data.append(episode_row_data)

  all_episode_df = pl.DataFrame(all_episode_row_data)

  def termination(e):
    return env_utils.any_feature_achieved(e)

  # Use provided get_reaction_time_fn or fallback to default
  def get_log_rt(e: EpisodeData):
    return np.log(1000 * e.reaction_times + 1e-5)

  def path_length(e: EpisodeData):
    return len(e.actions[:-1])

  # Define measures dictionary
  measure_to_fn = {
    "success": env_utils.success,
    "path_length": path_length,
    "termination": termination,
    "first_rt": lambda e: e.reaction_times[0],
    "first_log_rt": lambda e: get_log_rt(e)[0],
    "avg_rt": lambda e: np.mean(e.reaction_times[:-1]),
    "avg_log_rt": lambda e: np.mean(get_log_rt(e)[:-1]),
    "total_rt": lambda e: np.sum(e.reaction_times[:-1]),
    "total_log_rt": lambda e: np.sum(get_log_rt(e)[:-1]),
    "avg_post_rt": lambda e: np.mean(e.reaction_times[1:-1]),
    "avg_log_post_rt": lambda e: np.mean(get_log_rt(e)[1:-1]),
    "max_rt": lambda e: np.max(e.reaction_times[:-1]),
    "max_log_rt": lambda e: np.max(get_log_rt(e)[:-1]),
    "final_rt": lambda e: e.reaction_times[-1],
    "final_log_rt": lambda e: get_log_rt(e)[-1],
    "reaction_times": lambda e: str(e.reaction_times[:-1]),
  }

  # Initialize computed values
  computed_values = {key: [] for key in measure_to_fn}

  # Calculate values for each episode with error handling
  for i, episode in enumerate(all_episode_data):
    for key, fn in measure_to_fn.items():
      try:
        value = fn(episode)
        computed_values[key].append(value)
      except Exception as e:
        logging.warning(f"Failed to compute {key} for episode {i}: {str(e)}")
        computed_values[key].append(None)  # Use None for failed computations
        if debug:
          raise e

  # Add computed columns
  all_episode_df = all_episode_df.with_columns(
    [pl.Series(key, values) for key, values in computed_values.items()]
  )

  ######################################
  # Add eval_shares_start_pos column
  ######################################
  def determine_eval_shares_start_pos(group_df):
    # Extract unique start_pos for eval=True and eval=False (non-eval)
    eval_start_positions = group_df.filter(eval=True)["start_pos"].unique()
    train_start_positions = group_df.filter(eval=False)["start_pos"].unique()

    shares_start_pos = False  # Default to False
    # Only proceed if both series are non-empty
    if len(eval_start_positions) > 0 and len(train_start_positions) > 0:
      # Check if any eval_start_position is present in train_start_positions
      shares_start_pos = any(
        pos in train_start_positions for pos in eval_start_positions
      )

    return group_df.with_columns(
      pl.lit(shares_start_pos).alias("eval_shares_start_pos")
    )

  # Use apply instead of map_groups
  group_keys = ["manipulation", "user_id", "block_name"]
  # Add row index to preserve original order
  result = all_episode_df.partition_by(group_keys, maintain_order=True)
  result = [determine_eval_shares_start_pos(group) for group in result]
  all_episode_df = pl.concat(result).sort("global_episode_idx")

  ######################################
  # Add success_at_min_success column
  ######################################

  # Create a mapping of which blocks passed the threshold
  block_success_counts = (
    all_episode_df.filter(eval=False)  # Only look at training mazes
    .group_by("manipulation", "user_id", "block_name")
    .agg(pl.col("success").sum().alias("train_success_count"))
  )

  block_passed_df = env_utils.compute_if_block_passed(block_success_counts)

  # Join train_success_count back to the main DataFrame
  all_episode_df = all_episode_df.join(
    block_success_counts, on=["manipulation", "user_id", "block_name"], how="left"
  ).sort("global_episode_idx")

  # Add min_train_success column using an efficient join
  all_episode_df = (
    all_episode_df.join(
      block_passed_df, on=["manipulation", "user_id", "block_name"], how="left"
    )
    .with_columns(pl.col("passed").fill_null(False).alias("min_train_success"))
    .drop("passed")
  ).sort("global_episode_idx")

  temp_df = nicewebrl.DataFrame(
    all_episode_df,
    all_episode_data,
  )

  all_episode_df = all_episode_df.with_columns(
    [
      pl.lit(None, dtype=pl.Int32).alias("reuse"),
      pl.lit(None, dtype=pl.Float64).alias("overlap"),
    ]
  )

  all_reuse_dicts = []
  all_overlap_dicts = []

  for user_id in tqdm(
    all_episode_df["user_id"].unique().to_list(), desc="Processing reuse per user"
  ):
    user_df_nicewebrl = temp_df.filter(user_id=user_id)
    reuse_dict, overlap_dict = env_utils.add_reuse_columns(user_df_nicewebrl)
    all_reuse_dicts.append(reuse_dict)
    all_overlap_dicts.append(overlap_dict)

  # Use our utility function to add the columns
  all_episode_df = add_reuse_dicts_to_df(
    all_episode_df, all_reuse_dicts, all_overlap_dicts
  )

  if hasattr(env_utils, "finish_preparing_human_dataframe"):
    all_episode_df = env_utils.finish_preparing_human_dataframe(all_episode_df)

  return all_episode_df


def get_human_data(
  input_glob_pattern: str,
  output_data_path: str,
  example_timestep,
  env_name: str,
  overwrite_episode_data=False,
  overwrite_episode_df=False,
  load_df_only=True,
  debug=False,
  n_jobs: int = -1,
):
  """Process human data for a specific environment.

  This function handles both loading pre-processed data and processing raw data.

  Args:
    data_path: Path to the raw data files
    human_data_filebase: Base path for saving/loading processed data
    example_timestep: Example timestep for deserialization
    env_name: Environment name ("jaxmaze" or "craftax")
    overwrite_episode_data: Whether to reprocess episode data
    overwrite_episode_df: Whether to reprocess episode info
    load_df_only: Whether to return only the DataFrame (not episodes)
    debug: Whether to run in debug mode (processing fewer files)
    parallel: Whether to process files in parallel

  Returns:
    Either a DataFrame of episode information
  """
  ################################################################
  # Load data paths
  ################################################################
  failed_files = []

  if debug:
    all_episodes_data_filename = os.path.join(
      output_data_path, "debug", "human_data_episodes.safetensor"
    )
    all_episodes_metadata_filename = os.path.join(
      output_data_path, "debug", "human_data_episode_metadata.json"
    )
    all_episodes_df_filename = os.path.join(
      output_data_path, "debug", "human_data_episode_df.csv"
    )
  else:
    all_episodes_data_filename = os.path.join(
      output_data_path, "final", "human_data_episodes.safetensor"
    )
    all_episodes_metadata_filename = os.path.join(
      output_data_path, "final", "human_data_episode_metadata.json"
    )
    all_episodes_df_filename = os.path.join(
      output_data_path, "final", "human_data_episode_df.csv"
    )

  print(f"GENERATING {all_episodes_df_filename}")
  os.makedirs(os.path.dirname(all_episodes_data_filename), exist_ok=True)

  # --------------------------------
  # don't want to overwrite anything
  # --------------------------------
  data_file_exists = os.path.exists(all_episodes_data_filename)
  df_file_exists = os.path.exists(all_episodes_df_filename)
  overwrite_episode_df = (
    overwrite_episode_df or overwrite_episode_data
  )  # must redo df if data is overwritten
  if (
    not (overwrite_episode_data or overwrite_episode_df)
    and data_file_exists
    and df_file_exists
  ):
    print(f"LOADING {all_episodes_df_filename}")
    all_episodes_df = pl.read_csv(all_episodes_df_filename)
    if load_df_only:
      return all_episodes_df
    else:
      print(f"LOADING {all_episodes_data_filename}")
      all_episode_data = load_episode_data(
        filename=all_episodes_data_filename, example_timestep=example_timestep
      )
      return nicewebrl.DataFrame(all_episodes_df, all_episode_data)
  # --------------------------------
  # don't want to overwrite episode data but want to overwrite episode info
  # --------------------------------
  elif (not overwrite_episode_data and data_file_exists) and (
    overwrite_episode_df or not df_file_exists
  ):
    # load all episode data from single safetensor file
    all_episode_data = load_episode_data(
      filename=all_episodes_data_filename, example_timestep=example_timestep
    )
    print(f"LOADING {all_episodes_data_filename}")
    # load all episode metadata from single json file
    with open(all_episodes_metadata_filename, "r") as f:
      all_episode_metadata = json.load(f)

    all_episodes_df = generate_all_episodes_df(
      all_episode_data,
      all_episode_metadata,
      env_name=env_name,
      debug=debug,
    )
    # Save updated episode info
    all_episodes_df.write_csv(all_episodes_df_filename)

    if load_df_only:
      return all_episodes_df
    else:
      return nicewebrl.DataFrame(all_episodes_df, all_episode_data)
  # --------------------------------
  # overwrite everything
  # --------------------------------
  elif (
    (overwrite_episode_data and overwrite_episode_df)
    or not data_file_exists
    or not df_file_exists
  ):
    files = glob(input_glob_pattern)
    if not files:
      raise ValueError(f"No files found for input_glob_pattern: {input_glob_pattern}")
    if debug:
      files = files[: max(int(len(files) * 0.1), 3)][::-1]

    all_episode_data, all_episode_metadata, failed_files = generate_all_episodes_data(
      paths=files,
      example_timestep=example_timestep,
      env_name=env_name,
      debug=debug,
      parallel=not debug,
      n_jobs=n_jobs,
    )

    if failed_files:
      # Create a 'failed' subdirectory if it doesn't exist
      failed_dir = os.path.join(output_data_path, "human_data", "failed")
      os.makedirs(failed_dir, exist_ok=True)

      # Move each failed file to the failed directory
      print(f"Moving {len(failed_files)} files to {failed_dir}")
      for file_path in failed_files:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(failed_dir, filename)
        # If destination file already exists, remove it
        if os.path.exists(dest_path):
          os.remove(dest_path)
        # Move the file to the failed directory
        shutil.move(file_path, dest_path)

    # Save using Flax serialization
    with open(all_episodes_data_filename, "wb") as f:
      serialized_data = serialization.to_bytes(all_episode_data)
      f.write(serialized_data)
      print(f"Saved {len(all_episode_data)} episodes to {all_episodes_data_filename}")
    with open(all_episodes_metadata_filename, "w") as f:
      json.dump(all_episode_metadata, f)
      print("Saved metadata")

    all_episodes_df = generate_all_episodes_df(
      all_episode_data,
      all_episode_metadata,
      env_name=env_name,
      debug=debug,
    )

    # Save episode dataframe
    all_episodes_df.write_csv(all_episodes_df_filename)

    if load_df_only:
      return all_episodes_df
    else:
      return nicewebrl.DataFrame(all_episodes_df, all_episode_data)

  else:
    raise ValueError("Invalid overwrite_episode_data and overwrite_episode_df")


def get_jaxmaze_human_data(
  input_data_path: str = os.path.join(
    data_configs.JAXMAZE_USER_DIR, data_configs.JAXMAZE_HUMAN_DATA_PATTERN
  ),
  output_data_path: str = data_configs.JAXMAZE_DATA_DIR,
  overwrite_episode_data: bool = False,
  overwrite_episode_df: bool = False,
  load_df_only: bool = True,
  debug: bool = False,
):
  """Get human data for JaxMaze environment."""
  from experiments.jaxmaze.experiment_utils import SuccessTrackingAutoResetWrapper

  ################################################################
  # Setup environment
  ################################################################
  image_data = utils.load_image_dict()
  image_keys = image_data["keys"]

  groups = [
    # room 1
    [image_keys.index("orange"), image_keys.index("potato")],
    # room 2
    [image_keys.index("knife"), image_keys.index("spoon")],
    # room 3
    # [image_keys.index('tomato'), image_keys.index('lettuce')],
  ]
  groups = np.array(groups, dtype=np.int32)
  task_objects = groups.reshape(-1)

  char2idx = mazes.groups_to_char2key(groups)

  dummy_rng = jax.random.PRNGKey(42)

  dummy_env_params = mazes.get_maze_reset_params(
    groups=groups,
    char2key=char2idx,
    maze_str=mazes.big_practice_maze,
    randomize_agent=False,
    make_env_params=True,
  )
  task_runner = multitask_env.TaskRunner(task_objects=task_objects)
  base_env = web_env.HouseMaze(
    task_runner=task_runner,
    num_categories=200,
  )
  env = SuccessTrackingAutoResetWrapper(base_env)
  example_web_timestep = env.reset(dummy_rng, dummy_env_params)

  # Call the common human data function
  return get_human_data(
    env_name="jaxmaze",
    input_glob_pattern=input_data_path,
    output_data_path=output_data_path,
    example_timestep=example_web_timestep,
    overwrite_episode_data=overwrite_episode_data,
    overwrite_episode_df=overwrite_episode_df,
    load_df_only=load_df_only,
    debug=debug,
  )


def get_craftax_human_data(
  input_data_path: str,
  output_data_path: str = data_configs.CRAFTAX_DATA_DIR,
  overwrite_episode_data=False,
  overwrite_episode_df=False,
  load_df_only: bool = True,
  debug=False,
  n_jobs: int = 2,
):
  """Get human data for Craftax environment."""
  import experiments.craftax.craftax_experiment_structure as experiment

  example_web_timestep = experiment.jax_web_env.reset(
    jax.random.PRNGKey(0), experiment.dummy_params
  )

  # Call the common human data function
  return get_human_data(
    env_name="craftax",
    input_glob_pattern=input_data_path,
    output_data_path=output_data_path,
    example_timestep=example_web_timestep,
    overwrite_episode_data=overwrite_episode_data,
    overwrite_episode_df=overwrite_episode_df,
    load_df_only=load_df_only,
    debug=debug,
    n_jobs=n_jobs,
  )


# Update the main function to handle the returned failed_files
if __name__ == "__main__":
  # Parse command line arguments
  import argparse

  parser = argparse.ArgumentParser(description="Process user data for experiments")
  parser.add_argument(
    "--env",
    choices=["jaxmaze", "craftax", "both"],
    default="both",
    help="Which environment to process (jaxmaze, craftax, or both)",
  )
  parser.add_argument("--episodes", action="store_true", help="Overwrite episode data")
  parser.add_argument("--df", action="store_true", help="Overwrite episode dataframe")
  parser.add_argument(
    "--debug", action="store_true", help="Run in debug mode with reduced data"
  )

  args = parser.parse_args()

  overwrite_episode_data = args.episodes
  overwrite_episode_df = args.df
  debug_mode = args.debug

  # Process JaxMaze data if requested
  if args.env in ["jaxmaze", "both"]:
    print(
      f"Processing JaxMaze data (episodes={overwrite_episode_data}, df={overwrite_episode_df}, debug={debug_mode})"
    )
    human_data_df = get_jaxmaze_human_data(
      overwrite_episode_data=overwrite_episode_data,
      overwrite_episode_df=overwrite_episode_df,
      load_df_only=True,
      debug=debug_mode,
    )
    # Compute mean success rates for Craftax
    print("\nJaxMaze Success Rates:")
    success_rates = (
      human_data_df.group_by(["user_id", "eval"])
      .agg(pl.col("success").mean())
      .sort(["eval", "user_id"])
    )
    print(success_rates)

  # Process Craftax data if requested
  if args.env in ["craftax", "both"]:
    print(
      f"Processing Craftax data (episodes={overwrite_episode_data}, df={overwrite_episode_df}, debug={debug_mode})"
    )
    human_data_path = os.path.join(
      data_configs.CRAFTAX_USER_DIR, data_configs.CRAFTAX_HUMAN_DATA_PATTERN
    )
    human_data_df = get_craftax_human_data(
      input_data_path=human_data_path,
      overwrite_episode_data=overwrite_episode_data,
      overwrite_episode_df=overwrite_episode_df,
      load_df_only=True,
      debug=debug_mode,
    )
    # Compute mean success rates for Craftax
    print("\nCraftax Success Rates:")
    success_rates = (
      human_data_df.group_by(["user_id", "eval"])
      .agg(pl.col("success").mean())
      .sort(["eval", "user_id"])
    )
    print(success_rates)
  import ipdb

  ipdb.set_trace()
