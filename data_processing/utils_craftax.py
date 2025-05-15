import os
import os.path
import sys

sys.path.append("simulations")

import jax
import jax.numpy as jnp
import nicewebrl
import numpy as np
from flax import serialization
import polars as pl
from simulations import craftax_experiment_configs  # human
from simulations import craftax_simulation_configs  # model. derivative of human configs
from data_processing.utils import get_in_episode, total_reward, success, path_length
from simulations.craftax_web_env import EnvParams, CraftaxSymbolicWebEnvNoAutoReset
from simulations.craftax_utils import astar

################################################
# Cache Optimal Test Paths and lengths
################################################

OPTIMAL_TEST_PATHS = {}


def make_start_position(start_positions):
  start_position = jnp.zeros((10, 2), dtype=jnp.int32)
  return start_position.at[: len(start_positions)].set(jnp.asarray(start_positions))


for config in craftax_experiment_configs.PATHS_CONFIGS:
  # Create cache path
  cache_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "simulations",
    "craftax_cache",
    "optimal_paths",
  )
  os.makedirs(cache_dir, exist_ok=True)
  world_seed = config.world_seed
  cache_file = os.path.join(cache_dir, f"path_world={world_seed}_eval.npy")

  # Try to load from cache
  if os.path.exists(cache_file):
    path = np.load(cache_file)
  else:
    # Calculate path and save to cache
    env_params = craftax_experiment_configs.make_block_env_params(
      config, EnvParams()
    ).replace(
      # goal_locations=config.test_object_location,
      # current_goal=jnp.asarray(config.test_objects[0], dtype=jnp.int32),
      start_positions=make_start_position(config.start_eval_positions),
    )
    env = CraftaxSymbolicWebEnvNoAutoReset()
    obs, state = env.reset(jax.random.PRNGKey(0), env_params)
    goal_position = config.test_object_location
    path, _ = astar(state, goal_position)
    path = np.array(path)

    np.save(cache_file, path)
    print(f"Cached optimal path for {config.world_seed} in {cache_file}")
  OPTIMAL_TEST_PATHS[config.world_seed] = path
OPTIMAL_TEST_LENGTHS = {k: len(v) - 1 for k, v in OPTIMAL_TEST_PATHS.items()}


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
    map=jax.tree_map(lambda t: make_uint(t[:1]), state.map),
    item_map=jax.tree_map(lambda t: make_uint(t[:1]), state.item_map),
    mob_map=jax.tree_map(lambda t: make_uint(t[:1]), state.mob_map),
    light_map=jax.tree_map(lambda t: make_uint(t[:1]), state.light_map),
  )
  return t.replace(
    observation=None,  # remove observation
    state=new_state,
  )


def deserialize_timestep(datum, example_timestep):
  timestep = datum["data"]["timestep"]
  timestep = serialization.from_bytes(example_timestep, timestep)
  timestep = reduce_timestep_size(timestep)

  return timestep


def get_task_object(timesteps: nicewebrl.TimeStep):
  goal = int(timesteps.state.current_goal[0])
  return craftax_experiment_configs.GOAL_TO_BLOCK[goal]


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
  optimal_length = OPTIMAL_TEST_LENGTHS[int(row["world_seed"])]
  row["optimal_length"] = optimal_length
  path_length = len(get_agent_position(timesteps)) - 1
  row["suboptimal_path"] = path_length >= 2 * optimal_length
  row["efficient_1.25"] = path_length <= 1.25 * optimal_length
  row["efficient_1.5"] = path_length <= 1.5 * optimal_length
  row["efficient_1.75"] = path_length <= 1.75 * optimal_length
  row["efficient_2"] = path_length <= 2 * optimal_length
  deviation = path_length - optimal_length

  row["path_length_deviance"] = max(0, deviation)

  return row


def get_agent_position(timesteps: nicewebrl.TimeStep):
  return timesteps.state.player_position


def any_feature_achieved(episode_data):
  features = episode_data.timesteps.state.achievements
  achieved = features.sum(-1) > 0
  return achieved.any().astype(np.float32)


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


def create_maps(episode_data_list, start_pos=0):
  maps = []

  for episode_data in episode_data_list:
    timesteps = episode_data.timesteps

    # [T, N, H, W]
    # Assuming grid is 3D with time as first dimension
    grid_shape = timesteps.state.map.shape

    # skip the time dimension and final channel dimension
    grid = jnp.zeros(grid_shape[2:], dtype=jnp.int32)

    # go through each position and set the corresponding index to 1
    for pos in episode_data.positions[start_pos:]:
      grid = grid.at[pos[0], pos[1]].set(1)
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
  reuse_dict = {}
  overlap_dict = {}

  def update_reuse_dict(train_mazes, test_mazes):
    for train_maze, test_maze in zip(train_mazes, test_mazes):
      # Get train episodes
      test = df.filter(maze=test_maze, eval=True)
      if len(test) == 0:
        print(f"No test episodes for {(train_maze, test_maze)}")
        continue
      start_pos = test["start_pos"].to_list()[0]

      train = df.filter(
        maze=train_maze, room=0, eval=False, success=1, start_pos=start_pos
      )

      if len(train.episodes) == 0:
        print(f"No successful training episodes for {(train_maze, test_maze)}")
        continue

      # Create map for training episodes
      train_map = create_maps(train.episodes).sum(0)

      # Get test episodes

      # Process each test episode
      for idx, row in enumerate(test._df.iter_rows(named=True)):
        global_index = row["global_episode_idx"]
        episode = test.episodes[idx]
        # Create map for single test episode
        test_map = create_maps([episode]).sum(0)
        overlap = compute_overlap(train_map, test_map)
        overlap_mean = overlap.mean()

        # Store both raw overlap and binary reuse values
        episode_id = (test_maze, global_index)
        overlap_dict[episode_id] = overlap_mean
        reuse_dict[episode_id] = int(overlap_mean > overlap_threshold)

  # all_mazes = df["name"].unique()
  train_mazes = [f"paths_{i}_training" for i in range(4)]
  test_mazes = [f"paths_{i}_eval1" for i in range(4)]
  # train_mazes = sorted([m for m in all_mazes if "training" in m])
  # test_mazes = sorted([m for m in all_mazes if "eval" in m])

  # assert len(train_mazes) + len(test_mazes) == len(all_mazes)

  update_reuse_dict(train_mazes, test_mazes)

  # Return the dictionaries directly instead of converting to Series
  return reuse_dict, overlap_dict


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
  import ipdb

  ipdb.set_trace()
  return df


################################################
# Model Data
################################################
world_seed_to_idx = {}
for idx, config in enumerate(craftax_experiment_configs.PATHS_CONFIGS):
  world_seed_to_idx[config.world_seed] = idx


def dummy_config():
  return dict(
    task_config=craftax_simulation_configs.TRAIN_EVAL_CONFIGS,
    eval=True,
    manipulation="string",
    algo="string",
    seed=0,
  )


def generate_algorithm_episodes(algorithm, rng, extras: dict = None):
  train_configs = craftax_simulation_configs.TRAIN_EVAL_CONFIGS
  test_configs = craftax_simulation_configs.TEST_CONFIGS

  def generate_craftax_episodes(configs, is_eval=False, nepisodes=1):
    episodes_list = []
    configs_list = []
    nparams = configs.world_seed.shape[0]
    for i in range(nparams):
      env_params = craftax_simulation_configs.make_multigoal_env_params(
        jax.tree_map(lambda x: x[i : i + 1], configs)
      )
      episodes = algorithm.eval_fn(rng, env_params)
      episodes = episodes._replace(positions=get_agent_position(episodes.timesteps))
      # Split episodes
      task_config = jax.tree_map(lambda x: x[0], env_params.task_configs)
      for j in range(nepisodes):
        # minimize space requirements
        episode = jax.tree_util.tree_map(lambda x: x[j], episodes)
        in_episode = get_in_episode(episode.timesteps)
        episode = jax.tree_util.tree_map(lambda x: x[in_episode], episode)
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
  train_episodes, train_configs_info = generate_craftax_episodes(
    train_configs, is_eval=False
  )
  test_episodes, test_configs_info = generate_craftax_episodes(
    test_configs, is_eval=True
  )
  all_episodes = train_episodes + test_episodes
  all_configs = train_configs_info + test_configs_info
  return all_episodes, all_configs


def make_model_episode_row_data(episode, metadata):
  task_config = metadata["task_config"]

  def make_name(world_seed, eval):
    name = f"paths_{world_seed_to_idx[world_seed]}"
    if eval:
      name += "_eval1"
    else:
      name += "_training"
    return name

  return dict(
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
    seed=metadata["seed"],
    user_id=metadata["seed"],  # reflecting human data format
    maze=make_name(int(task_config.world_seed), metadata["eval"]),
    # idiosyncratic
    world_seed=int(task_config.world_seed),
    # maze=int(task_config.world_seed),
    task_vector=str(episode.timesteps.observation.task_w[0]),
  )


add_model_reuse_columns = add_reuse_columns  # same in this env for both
