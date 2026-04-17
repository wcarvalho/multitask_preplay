from functools import partial
import os
import os.path
import sys

sys.path.insert(0, ".")
sys.path.append("simulations")

try:
  from experiments.craftax.configs import check_cache

  check_cache()
except ModuleNotFoundError:
  pass

import jax
import jax.numpy as jnp
import nicewebrl
import numpy as np
from flax import serialization
import polars as pl
from data_processing.utils import (
  get_in_episode,
  total_reward,
  success,
  path_length,
  get_overlap_dicts_model,
  get_overlap_dicts_human,
)

################################################
# Cache Optimal Test Paths and lengths (lazy-loaded)
################################################

_OPTIMAL_TEST_PATHS = None
_OPTIMAL_TEST_LENGTHS = None
_OPTIMAL_TRAIN_PATHS = None
_OPTIMAL_TRAIN_LENGTHS = None
_world_seed_to_idx = None


def make_start_position(start_positions):
  start_position = jnp.zeros((10, 2), dtype=jnp.int32)
  return start_position.at[: len(start_positions)].set(jnp.asarray(start_positions))


def _compute_optimal_test_paths():
  global _OPTIMAL_TEST_PATHS, _OPTIMAL_TEST_LENGTHS
  if _OPTIMAL_TEST_PATHS is not None:
    return
  from simulations import craftax_experiment_configs
  from simulations.craftax_web_env import EnvParams, CraftaxSymbolicWebEnvNoAutoReset
  from simulations.craftax_utils import astar

  _OPTIMAL_TEST_PATHS = {}
  for config in craftax_experiment_configs.PATHS_CONFIGS:
    cache_dir = os.path.join(
      os.path.dirname(os.path.dirname(__file__)),
      "simulations",
      "craftax_cache",
      "optimal_paths",
    )
    os.makedirs(cache_dir, exist_ok=True)
    world_seed = config.world_seed
    cache_file = os.path.join(cache_dir, f"path_world={world_seed}_eval.npy")

    if os.path.exists(cache_file):
      path = np.load(cache_file)
    else:
      env_params = craftax_experiment_configs.make_block_env_params(
        config, EnvParams()
      ).replace(
        start_positions=make_start_position(config.start_eval_positions),
      )
      env = CraftaxSymbolicWebEnvNoAutoReset()
      obs, state = env.reset(jax.random.PRNGKey(0), env_params)
      goal_position = config.test_object_location
      path, _ = astar(state, goal_position)
      path = np.array(path)

      np.save(cache_file, path)
      print(f"Cached optimal path for {config.world_seed} in {cache_file}")
    _OPTIMAL_TEST_PATHS[config.world_seed] = path
  _OPTIMAL_TEST_LENGTHS = {k: len(v) - 1 for k, v in _OPTIMAL_TEST_PATHS.items()}


def _compute_optimal_train_paths():
  global _OPTIMAL_TRAIN_PATHS, _OPTIMAL_TRAIN_LENGTHS
  if _OPTIMAL_TRAIN_PATHS is not None:
    return
  from simulations import craftax_experiment_configs

  cache_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "simulations",
    "craftax_cache",
    "training_paths",
  )
  _OPTIMAL_TRAIN_PATHS = {}
  for config in craftax_experiment_configs.PATHS_CONFIGS:
    world_seed = config.world_seed
    for start_pos in config.start_train_positions:
      for train_obj in config.train_objects:
        cache_file = os.path.join(
          cache_dir, f"path_{world_seed}_{start_pos}_{train_obj}.npy"
        )
        if os.path.exists(cache_file):
          path = np.load(cache_file)
          _OPTIMAL_TRAIN_PATHS[(world_seed, start_pos, train_obj)] = path
        else:
          print(f"Warning: missing cached train path {cache_file}")
  _OPTIMAL_TRAIN_LENGTHS = {k: len(v) - 1 for k, v in _OPTIMAL_TRAIN_PATHS.items()}


def _compute_world_seed_to_idx():
  global _world_seed_to_idx
  if _world_seed_to_idx is not None:
    return
  from simulations import craftax_experiment_configs

  _world_seed_to_idx = {
    config.world_seed: idx
    for idx, config in enumerate(craftax_experiment_configs.PATHS_CONFIGS)
  }


def __getattr__(name):
  if name == "OPTIMAL_TEST_PATHS":
    _compute_optimal_test_paths()
    return _OPTIMAL_TEST_PATHS
  if name == "OPTIMAL_TEST_LENGTHS":
    _compute_optimal_test_paths()
    return _OPTIMAL_TEST_LENGTHS
  if name == "OPTIMAL_TRAIN_PATHS":
    _compute_optimal_train_paths()
    return _OPTIMAL_TRAIN_PATHS
  if name == "OPTIMAL_TRAIN_LENGTHS":
    _compute_optimal_train_paths()
    return _OPTIMAL_TRAIN_LENGTHS
  if name == "world_seed_to_idx":
    _compute_world_seed_to_idx()
    return _world_seed_to_idx
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


################################################
# Human Data
################################################
def get_block_stage_description(datum):
  """There were 4 blocks, each with a maze identified by its own world seed.

  This function will be used to group datapoints of an individual episode."""
  ####################
  # block information
  ####################
  block_metadata = datum["metadata"]["block_metadata"]
  # e.g. manipulation = 4
  block_manipulation = block_metadata.get("manipulation", -1)
  # e.g. desc = 'off-task object regular'
  # block_desc = block_metadata.get("desc", "unknown")

  ####################
  # stage information
  ####################
  return dict(
    world_seed=datum["metadata"].get("world_seed"),
    maze=datum["metadata"].get("world_seed"),
    condition=datum["metadata"].get("condition", 0),
    name=datum["name"],
    # block=block_desc,
    manipulation=block_manipulation,
    episode_idx=datum["metadata"]["nepisodes"],
    eval=datum["metadata"]["eval"],
    block_identifier=datum["metadata"].get("world_seed"),
  )


def reduce_timestep_size(t: nicewebrl.TimeStep):
  state = t.state

  def make_uint(x):
    return x.astype(jnp.uint8)
    # return x

  new_state = state.replace(
    # remove all levels after 1st
    map=jax.tree_util.tree_map(lambda t: make_uint(t[:1]), state.map),
    item_map=jax.tree_util.tree_map(lambda t: make_uint(t[:1]), state.item_map),
    mob_map=jax.tree_util.tree_map(lambda t: make_uint(t[:1]), state.mob_map),
    light_map=jax.tree_util.tree_map(lambda t: make_uint(t[:1]), state.light_map),
  )
  return t.replace(
    observation=None,
    state=new_state,
  )


def deserialize_timestep(datum, example_timestep):
  timestep = datum["data"]["timestep"]
  timestep = serialization.from_bytes(example_timestep, timestep)
  timestep = reduce_timestep_size(timestep)

  return timestep


def get_task_object(timesteps: nicewebrl.TimeStep):
  from simulations import craftax_experiment_configs

  goal = int(timesteps.state.current_goal[0])
  return craftax_experiment_configs.GOAL_TO_BLOCK[goal]


def make_task_vector(timesteps: nicewebrl.TimeStep):
  from simulations.craftax_web_env import task_onehot

  current_goal = timesteps.state.current_goal[0]
  task_w = task_onehot(current_goal)
  return task_w


def get_step_number(timesteps: nicewebrl.TimeStep):
  return timesteps.state.timestep


def make_human_episode_row_data(
  metadata: dict,
  timesteps: nicewebrl.TimeStep,
):
  """THIS IS WHERE YOU'LL WANT TO INSERT OTHER EPISODE LEVEL INFO TO TRACK IN DATAFRAME!!!

  Args:
      datum (dict): _description_
      timesteps (TimeStep): _description_
      file (str): _description_

  Returns:
      _type_: _description_
  """

  is_eval = metadata["eval"]
  task_object = int(get_task_object(timesteps))
  if is_eval:
    task_set = 0
  else:
    train_objects = metadata["block_metadata"]["train_objects"]
    task_set = train_objects.index(task_object)

  user_storage = metadata["user_storage"]
  row = dict(
    # shared across {human, model}, {craftax, jaxmaze}
    domain="craftax",
    algo="human",
    block_name=metadata.get("world_seed"),
    condition=int(metadata.get("condition", 0)),
    eval=metadata["eval"],
    start_pos=str(get_agent_position(timesteps)[0]),
    manipulation=metadata["block_metadata"].get("manipulation", None),
    # global_episode_idx=metadata["user_episode_idx"],
    task=int(get_task_object(timesteps)),
    task_set=task_set,
    room=task_set,  # for compatibility with prior code
    maze=metadata["name"],
    # idiosyncratic
    world_seed=metadata.get("world_seed"),
    # world_seed=metadata.get("world_seed"),
    block=metadata["block_metadata"]["desc"],
    episode_idx=metadata["nepisodes"],
    task_vector=str(make_task_vector(timesteps)),
    # Human specific
    session_start=user_storage["session_start"],
    session_duration=user_storage["session_duration"],
    timezone=user_storage["env_vars"]["TIMEZONE"],
  )
  row.update(metadata["user_data"])

  user_storage = metadata["user_storage"]
  row.update(user_storage["user_info"])

  if row["condition"] > 0:
    row["eval"] = True

  ##########
  # get experiment name from file
  ##########
  row.update(
    exp_name=user_storage["env_vars"]["NAME"],
    tell_reuse=user_storage["env_vars"]["SAY_REUSE"],
    eval_map=user_storage["env_vars"]["EVAL_SHOW_MAP"],
    timer=0,
  )
  ####################
  # add version, tell_reuse, timer
  ####################
  name = row.get("exp_name")
  if name is not None:
    # example 'exp4-v1-r1-t0-plan'
    # split on '-' and take the first element
    # if v--> version
    # if there's a word at the end, it's the manipulation
    # create a dictionary according to this legend
    legend = dict(v="version")
    name_info = dict()
    for k, v in legend.items():
      if k in name:
        name_info[v] = name.split(k)[1].split("-")[0]
    row.update(name_info)

  # Convert all numeric strings to integers
  for key, value in row.items():
    if isinstance(value, str) and value.isdigit():
      row[key] = int(value)

  #####################
  ## add optimal path length - with caching
  #####################
  _compute_optimal_test_paths()
  optimal_length = _OPTIMAL_TEST_LENGTHS[int(row["world_seed"])]
  row["optimal_length"] = optimal_length
  path_length = len(get_agent_position(timesteps)) - 1
  row["suboptimal_path"] = path_length >= 2 * optimal_length
  row["efficient_1.25"] = path_length <= 1.25 * optimal_length
  row["efficient_1.5"] = path_length <= 1.5 * optimal_length
  row["efficient_1.75"] = path_length <= 1.75 * optimal_length
  row["efficient_2"] = path_length <= 2 * optimal_length
  deviation = path_length - optimal_length

  row["path_length"] = path_length
  row["path_length_deviance"] = max(0, deviation)

  row = fix_row(row)
  return row


def fix_row(row):
  new_row = dict(row)
  if "world_seed" in row:
    world_seed = new_row["world_seed"]
    new_row["world"] = str(world_seed)

  # same for this domain
  new_row["block_name"] = str(new_row["world"])

  if "task" in new_row:
    new_row["task_object_id"] = new_row["task"]

  new_row.pop("room", None)

  if "seed" in new_row:
    new_row["user_id"] = new_row["seed"]

  if "manipulation" in new_row:
    if isinstance(new_row["manipulation"], int):
      new_row["manipulation"] = {
        1: "shortcut",
        2: "start",
        3: "paths",
        4: "juncture",
        5: "paths",
      }[new_row["manipulation"]]
    elif isinstance(new_row["manipulation"], str):
      assert new_row["manipulation"] in ["paths"]
    else:
      raise ValueError(f"Invalid manipulation: {new_row['manipulation']}")

  new_row.pop("seed", None)
  new_row.pop("maze", None)

  desired_types = dict(
    domain=str,
    algo=str,
    block_name=str,
    world=str,
    condition=int,
    eval=bool,
    start_pos=str,
    manipulation=str,
    task_object_id=int,
    task_set=int,
    task_vector=str,
  )
  for k, v in desired_types.items():
    assert isinstance(new_row[k], v), f"Expected {k} to be {v}, got {type(new_row[k])}"

  keys_to_remove = [
    "debug",
    "git_version",
    "name",
    "seed",
    "maze",
    "reversal",
    "user",
    "world_seed",
    "block",
    "exp_name",
    "version",
    "task",
  ]
  for k in keys_to_remove:
    new_row.pop(k, None)

  return new_row


def get_agent_position(timesteps: nicewebrl.TimeStep):
  return timesteps.state.player_position


def any_feature_achieved(episode_data):
  features = episode_data.timesteps.state.achievements
  achieved = features.sum(-1) > 0
  return achieved.any().astype(np.float32)


# def best_path_overlap(maps: np.ndarray, test_maps: np.ndarray):
#  """maps: NxHxW, test_map: HxW"""
#  """Calculate the overlap between two maps."""
#  # Find the train episode with highest overlap
#  overlaps = []
#  for i, map in enumerate(maps):
#    if test_maps.ndim == 2:
#      overlap = compute_overlap(map, test_maps).mean()
#      overlaps.append(overlap)
#    else:
#      test_overlaps = []
#      for test_map in test_maps:
#        test_overlap = compute_overlap(map, test_map).mean()
#        test_overlaps.append(test_overlap)
#      min_overlap = np.min(test_overlaps)
#      if min_overlap < 0.15:
#        overlaps.append(0)
#      else:
#        overlaps.append(min_overlap)

#  # Select the train episode/map with highest overlap
#  best_idx = np.argmax(overlaps)
#  best_map = maps[best_idx]
#  best_overlap = overlaps[best_idx]
#  return best_idx, best_map, best_overlap


# def compute_overlap(map1: np.ndarray, map2: np.ndarray):
#  """map1: HxW, map2: HxW"""
#  """Calculate the overlap between two maps."""

#  shared_length = ((map2 + map1) * map2 > 1).sum()
#  map2_length = (map2 > 0).sum()
#  overlap = shared_length / map2_length
#  return overlap


def create_maps(
  episode_data_list,
  add_error=False,
):
  maps = []

  for episode_data in episode_data_list:
    timesteps = episode_data.timesteps

    # [T, N, H, W]
    # Assuming grid is 3D with time as first dimension
    grid_shape = timesteps.state.map.shape

    def make_grid(positions):
      # skip the time dimension and final channel dimension
      grid = jnp.zeros(grid_shape[2:], dtype=jnp.int32)
      for pos in positions:
        if add_error:
          y, x = pos[0], pos[1]
          for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
              grid = grid.at[y + dy, x + dx].set(1)
        else:
          grid = grid.at[pos[0], pos[1]].set(1)
      return grid

    # go through each position and set the corresponding index to 1
    positions = episode_data.positions[:-1]
    grid = make_grid(positions)
    maps.append(grid)

  return np.array(maps)


def add_reuse_columns(df: nicewebrl.DataFrame, overlap_threshold=0.15) -> tuple:
  """Add 'reuse' and 'overlap' columns to the DataFrame.

  Args:
      df (DataFrame): Input DataFrame
      overlap_threshold (float, optional): Threshold for path reuse. Defaults to 0.15.

  Returns:
      tuple: (reuse_dict, overlap_dict) dictionaries mapping (maze, global_episode_idx) to values
  """
  # Create dictionaries to store values
  all_reuse_dict = {}
  all_overlap_dict = {}
  all_corresponding_train_episode_idx = {}
  all_cosine_dict = {}
  all_skipped_blocks = []

  for world in df["world"].unique():
    (
      reuse_dict,
      overlap_dict,
      corresponding_train_episode_idx,
      cosine_dict,
      skipped_blocks,
    ) = get_overlap_dicts_human(
      df,
      train_maze=world,
      test_maze=world,
      overlap_threshold=0.15,
      create_train_maps_fn=partial(create_maps, add_error=True),
      create_test_maps_fn=partial(create_maps, add_error=False),
    )
    all_reuse_dict.update(reuse_dict)
    all_overlap_dict.update(overlap_dict)
    all_corresponding_train_episode_idx.update(corresponding_train_episode_idx)
    all_cosine_dict.update(cosine_dict)
    all_skipped_blocks.extend(skipped_blocks)

  # Return the dictionaries directly instead of converting to Series
  return (
    all_reuse_dict,
    all_overlap_dict,
    all_corresponding_train_episode_idx,
    all_cosine_dict,
    all_skipped_blocks,
  )


def compute_if_block_passed(block_success_counts):
  # Return a polars DataFrame with user_id, manipulation, block_name, and passed columns
  # instead of a dictionary to enable efficient joining
  return block_success_counts.with_columns(
    (pl.col("train_success_count") >= 16).alias("passed")
  ).select(["user_id", "manipulation", "block_name", "passed"])


def finish_preparing_human_dataframe(df: pl.DataFrame) -> pl.DataFrame:
  df = df.with_columns(
    (pl.col("reuse").eq(1) & pl.col("efficient_1.25").eq(True)).alias(
      "efficient_reuse_1.25"
    )
  )
  df = df.with_columns(
    (pl.col("reuse").eq(1) & pl.col("efficient_1.5").eq(True)).alias(
      "efficient_reuse_1.5"
    )
  )
  df = df.with_columns(
    (pl.col("reuse").eq(1) & pl.col("efficient_1.75").eq(True)).alias(
      "efficient_reuse_1.75"
    )
  )
  df = df.with_columns(
    (pl.col("reuse").eq(1) & pl.col("efficient_2").eq(True)).alias("efficient_reuse_2")
  )
  return df


################################################
# Model Data
################################################


def dummy_config():
  from simulations import craftax_simulation_configs

  return dict(
    task_config=craftax_simulation_configs.TRAIN_EVAL_CONFIGS,
    eval=True,
    manipulation="string",
    algo="string",
    seed=0,
  )


def generate_algorithm_episodes(algorithm, rng, extras: dict = None, debug=False):
  from simulations import craftax_simulation_configs

  train_configs = craftax_simulation_configs.TRAIN_EVAL_CONFIGS
  test_configs = craftax_simulation_configs.TEST_CONFIGS

  def generate_craftax_episodes(configs, rng, is_eval=False):
    episodes_list = []
    configs_list = []
    nparams = configs.world_seed.shape[0]
    for i in range(nparams):
      rng, rng_subkey = jax.random.split(rng)
      env_params = craftax_simulation_configs.make_multigoal_env_params(
        jax.tree_util.tree_map(lambda x: x[i : i + 1], configs)
      )
      episodes = algorithm.eval_fn(rng_subkey, env_params)
      episodes = episodes._replace(positions=get_agent_position(episodes.timesteps))
      # Split episodes
      task_config = jax.tree_util.tree_map(lambda x: x[0], env_params.task_configs)
      nepisodes = episodes.actions.shape[0]

      for j in range(nepisodes):
        # minimize space requirements
        episode = jax.tree_util.tree_map(lambda x: x[j], episodes)
        in_episode = get_in_episode(episode.timesteps)
        episode = jax.tree_util.tree_map(lambda x: x[in_episode], episode)
        # strip transitions to save memory (duplicates timesteps + extras not used downstream)
        episode = episode._replace(transitions=None)
        # reduce map arrays from 9 levels to 1, int32→uint8, drop observation
        episode = episode._replace(timesteps=reduce_timestep_size(episode.timesteps))
        episodes_list.append(episode)
        configs_list.append(
          dict(
            task_config=task_config,
            eval=is_eval,
            manipulation=5,
            algo=algorithm.model_name or algorithm.model_filename,
            seed=extras["seed"],
          )
        )
    return episodes_list, configs_list

  # Generate episodes from both train and test configs
  rng, rng_train = jax.random.split(rng)
  train_episodes, train_configs_info = generate_craftax_episodes(
    train_configs, rng_train, is_eval=False
  )
  rng, rng_test = jax.random.split(rng)
  test_episodes, test_configs_info = generate_craftax_episodes(
    test_configs, rng_test, is_eval=True
  )
  all_episodes = train_episodes + test_episodes
  all_configs = train_configs_info + test_configs_info
  return all_episodes, all_configs


def make_model_episode_row_data(episode, metadata):
  _compute_world_seed_to_idx()
  task_config = metadata["task_config"]

  def make_name(world_seed, eval):
    name = f"paths_{_world_seed_to_idx[world_seed]}"
    if eval:
      name += "_eval1"
    else:
      name += "_training"
    return name

  row = dict(
    # shared across {human, model}, {craftax, jaxmaze}
    domain="craftax",
    algo=metadata["algo"],
    block_name=metadata.get("world_seed"),
    condition=int(metadata.get("condition", 0)),
    eval=metadata["eval"],
    start_pos=str(get_agent_position(episode.timesteps)[0]),
    manipulation=metadata["manipulation"],
    task=int(get_task_object(episode.timesteps)),
    task_set=0,
    room=0,  # for compatibility with prior code
    total_reward=float(total_reward(episode)),
    success=float(success(episode)),
    path_length=int(path_length(episode)),
    positions=str(np.asarray(episode.positions)),
    actions=str(np.asarray(episode.actions[:-1])),
    seed=metadata["seed"],
    user_id=metadata["seed"],  # reflecting human data format
    maze=make_name(int(task_config.world_seed), metadata["eval"]),
    # idiosyncratic
    world_seed=int(task_config.world_seed),
    # maze=int(task_config.world_seed),
    task_vector=str(make_task_vector(episode.timesteps)),
  )
  row = fix_row(row)
  return row


def add_model_reuse_columns(df: nicewebrl.DataFrame, overlap_threshold=0.15) -> tuple:
  """Add 'reuse' and 'overlap' columns to the DataFrame.

  Args:
      df (DataFrame): Input DataFrame
      overlap_threshold (float, optional): Threshold for path reuse. Defaults to 0.15.

  Returns:
      tuple: (reuse_dict, overlap_dict) dictionaries mapping (maze, global_episode_idx) to values
  """
  # Create dictionaries to store values
  all_reuse_dict = {}
  all_overlap_dict = {}
  all_corresponding_train_episode_idx = {}
  all_cosine_dict = {}

  for world in df["world"].unique():
    reuse_dict, overlap_dict, corresponding_train_episode_idx, cosine_dict = (
      get_overlap_dicts_model(
        df,
        train_maze=world,
        test_maze=world,
        overlap_threshold=0.15,
        create_train_maps_fn=partial(create_maps, add_error=True),
        create_test_maps_fn=partial(create_maps, add_error=False),
      )
    )
    all_reuse_dict.update(reuse_dict)
    all_overlap_dict.update(overlap_dict)
    all_corresponding_train_episode_idx.update(corresponding_train_episode_idx)
    all_cosine_dict.update(cosine_dict)

  return (
    all_reuse_dict,
    all_overlap_dict,
    all_corresponding_train_episode_idx,
    all_cosine_dict,
  )
