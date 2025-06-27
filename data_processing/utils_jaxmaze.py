import re
import os.path
import ast
import jax
import jax.numpy as jnp
import nicewebrl
import numpy as np
from flax import serialization
from serialization import SerializationWrapper
import polars as pl
import functools
from experiments.jaxmaze.experiment_utils import SuccessTrackingAutoResetWrapper

from housemaze.human_dyna import utils
from housemaze.human_dyna import mazes
from housemaze.human_dyna import web_env
from housemaze.human_dyna import multitask_env
from housemaze.human_dyna import experiments as housemaze_experiments

from data_processing.utils import (
  get_in_episode,
  total_reward,
  success,
  path_length,
  get_overlap_dicts_model,
  get_overlap_dicts_human,
)

from tqdm.auto import tqdm

manipulation_int_to_str = {
  1: "shortcut",
  2: "start",
  3: "paths",
  4: "juncture",
}
manipulation_str_to_int = {v: k for k, v in manipulation_int_to_str.items()}


################################################
# Human Data
################################################
def get_block_stage_description(datum):
  """There were 4 blocks, each with the same maze but a different rotation of it identified by "reversal"."""
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
  reversal = datum["metadata"]["block_metadata"].get("reversal", [False, False])
  return dict(
    maze=datum["metadata"].get("maze"),
    condition=datum["metadata"].get("condition", 0),
    name=datum["name"],
    # block=block_desc,
    manipulation=block_manipulation,
    episode_idx=datum["metadata"]["nepisodes"],
    eval=datum["metadata"]["eval"],
    reversal=str(reversal),
    block_identifier=str(reversal),
  )


def deserialize_timestep(datum, example_timestep):
  timestep = datum["data"]["timestep"]

  timestep = serialization.from_bytes(SerializationWrapper(example_timestep), timestep)

  return timestep


def get_task_object(timesteps: nicewebrl.TimeStep):
  return timesteps.state.task_object[0]


def get_task_room(timesteps: nicewebrl.TimeStep, task_groups):
  task_object = get_task_object(timesteps)
  # Find the room (row) that contains the task object
  task_room = next((i for i, row in enumerate(task_groups) if task_object in row), None)
  return task_room


def get_step_number(timesteps: nicewebrl.TimeStep):
  return timesteps.state.step_num


def get_agent_position(timesteps: nicewebrl.TimeStep):
  return timesteps.state.agent_pos


def any_feature_achieved(episode_data):
  features = episode_data.timesteps.state.task_state.features
  achieved = features.sum(-1) > 0
  return achieved.any().astype(np.float32)


def went_to_junction(episode_data, junction=(0, 11)):
  positions = get_agent_position(episode_data.timesteps)
  match = jnp.array(junction) == positions
  match = (match).sum(-1) == 2  # both x and y matches
  return match.any().astype(jnp.float32)  # if any matched


def make_human_episode_row_data(
  metadata: dict,
  timesteps: nicewebrl.TimeStep,
):
  """THIS IS WHERE YOU'LL WANT TO INSERT OTHER EPISODE LEVEL INFO TO TRACK IN DATAFRAME!!!

  Args:
      datum (dict): _description_
      timesteps (multitask_env.TimeStep): _description_
      file (str): _description_

  Returns:
      _type_: _description_
  """

  groups = metadata["block_metadata"].get("groups")

  user_storage = metadata["user_storage"]
  env_vars = user_storage.get("env_vars", {})
  tell_reuse = int(env_vars.get("SAY_REUSE", 1))
  reversal = metadata["block_metadata"]["reversal"]

  maze = metadata.get("maze")
  # Extract world_seed by removing the block identifier pattern
  world_seed = re.sub(r"_\([TF],[TF]\)", "", maze)
  block_name = tuple([int(i) for i in reversal])
  condition = int(metadata.get("condition", 1 if metadata["eval"] else 0))

  row = dict(
    domain="jaxmaze",
    algo="human",
    maze=maze,
    world_seed=world_seed,
    block_name=str(block_name),
    condition=condition,
    name=metadata["name"],
    block=metadata["block_metadata"]["desc"],
    manipulation=metadata["block_metadata"].get("manipulation", None),
    # global_episode_idx=metadata["user_episode_idx"],
    episode_idx=metadata["nepisodes"],
    eval=metadata["eval"],
    task=int(get_task_object(timesteps)),
    room=int(get_task_room(timesteps, task_groups=groups)),
    start_pos=str(get_agent_position(timesteps)[0]),
    tell_reuse=tell_reuse,
    reversal=str(reversal),
    task_vector=str(timesteps.state.task_w[0]),
  )
  row.update(metadata["user_data"])
  row.update(user_storage["user_info"])
  ##########
  # get experiment name from file
  ##########
  file = metadata["file"]
  # '/path/data_user=3712207029_name=exp3-v2-r1-t30_exp=3_debug=0.json'
  # e.g. ['data', 'user=3712207029', 'name=exp3-v2-r1-t30', 'exp=3', 'debug=0.json']
  pieces = os.path.splitext(os.path.basename(file))[0].split("_")
  pieces = [p.split("=") for p in pieces if "=" in p]
  new_vals = {p[0]: p[1] for p in pieces}
  # Rename 'name' key to 'exp_name' if it exists
  if "name" in new_vals:
    new_vals["exp_name"] = new_vals.pop("name")
  row.update(new_vals)

  # fix eval for condition 2
  if row["condition"] > 0:
    row["eval"] = True

  #####################
  ## add optimal path length
  #####################
  # path = utils.find_optimal_path(
  #  grid=timesteps.state.grid[0],  # first time-step
  #  agent_pos=tuple([int(i) for i in timesteps.state.agent_pos[0]]),
  #  goal=timesteps.state.task_object[0],
  # )
  # row["optimal_length"] = len(path) - 1  # includes done

  row = fix_row(row)
  return row


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
  shared_length = ((map2 + map1) * map2 > 1).sum()
  map2_length = (map2 > 0).sum()
  overlap = shared_length / map2_length
  return overlap


def add_reuse_columns(df: nicewebrl.DataFrame) -> tuple[dict, dict]:
  """Add a 'reuse' column to the DataFrame indicating whether each episode reused paths.

  Args:
      df (DataFrame): Input DataFrame

  Returns:
      tuple[dict, dict]: Dictionaries for reuse and overlap values.
  """

  # Create a dictionary to store reuse values
  all_reuse_dict = {}
  all_overlap_dict = {}
  all_corresponding_train_episode_idx = {}

  reuse_dict, overlap_dict, corresponding_train_episode_idx = get_overlap_dicts_human(
    df,
    train_maze="big_m3_maze1",
    test_maze="big_m3_maze1",
    overlap_threshold=0.5,
  )
  all_reuse_dict.update(reuse_dict)
  all_overlap_dict.update(overlap_dict)
  all_corresponding_train_episode_idx.update(corresponding_train_episode_idx)

  reuse_dict, overlap_dict, corresponding_train_episode_idx = get_overlap_dicts_human(
    df,
    train_maze="big_m1_maze3",
    test_maze="big_m1_maze3_shortcut",
    overlap_threshold=0.7,
  )
  all_reuse_dict.update(reuse_dict)
  all_overlap_dict.update(overlap_dict)
  all_corresponding_train_episode_idx.update(corresponding_train_episode_idx)

  return all_reuse_dict, all_overlap_dict, all_corresponding_train_episode_idx


def compute_if_block_passed(block_success_counts):
  min_success_needed = {
    "shortcut": 16,  # shortcut, 1
    "start": 16,  # start, 2
    "paths": 16,  # paths, 3
    "juncture": 8,  # plan/juncture, 4
  }

  # First add a column with the minimum success needed for each manipulation
  df_with_min = block_success_counts.with_columns(
    pl.col("manipulation")
    .map_elements(lambda x: min_success_needed.get(x), return_dtype=pl.Int64)
    .alias("min_success_needed")
  )

  # Now use that column for comparison
  return df_with_min.with_columns(
    (pl.col("train_success_count") >= pl.col("min_success_needed")).alias("passed")
  ).select(["user_id", "manipulation", "block_name", "passed"])


################################################
# Model Data
################################################
maze_to_manipulation = dict(
  big_m1_maze3=1,
  big_m1_maze3_shortcut=1,
  big_m2_maze2=2,
  big_m2_maze2_onpath=2,
  big_m2_maze2_offpath=2,
  big_m3_maze1=3,
  big_m4_maze_short=4,
  big_m4_maze_short_eval_same=4,
  big_m4_maze_short_eval_diff=4,
  big_m4_maze_short_blind=4,
  big_m4_maze_short_eval_same_blind=4,
  big_m4_maze_long=4,
  big_m4_maze_long_eval_same=4,
  big_m4_maze_long_eval_diff=4,
  big_m4_maze_long_blind=4,
  big_m4_maze_long_eval_same_blind=4,
)


def dummy_config():
  char2idx, groups, task_objects = mazes.get_group_set()
  env_params = mazes.get_maze_reset_params(
    groups=groups,
    char2key=char2idx,
    maze_str=mazes.maze0,  # random maze
    randomize_agent=False,
    make_env_params=True,
  )
  env_params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), env_params)
  return dict(
    env_params=env_params,
    maze_name="string",
    task=1,
    eval=True,
    algo="string",
    seed=0,
    episode_idx=0,
  )


def generate_algorithm_episodes(algorithm, rng, extras: dict = None, debug=False):
  task_runner = extras.get("task_runner", None)

  char2idx, groups, task_objects = mazes.get_group_set()
  _, test_params, _, label2name = housemaze_experiments.exp4(
    algorithm.config, analysis_eval=True
  )

  all_episodes = []
  all_configs = []

  def collect_data(env_params, task, rng_, is_eval):
    task_vector = task_runner.task_vector(task)
    task_env_params = env_params.replace(task_probs=task_vector.astype(jnp.float32))
    episodes = algorithm.eval_fn(rng_, task_env_params)
    episodes = episodes._replace(positions=get_agent_position(episodes.timesteps))
    nepisodes = episodes.actions.shape[0]
    maze_name = label2name[int(env_params.reset_params.label[0])]

    # Split episodes
    for i in range(nepisodes):
      # minimize space requirements
      episode = jax.tree_util.tree_map(lambda x: x[i], episodes)
      in_episode = get_in_episode(episode.timesteps)
      episode = jax.tree_util.tree_map(lambda x: x[in_episode], episode)
      all_episodes.append(episode)
      all_configs.append(
        dict(
          env_params=task_env_params,
          maze_name=maze_name,
          task=int(task),
          eval=is_eval,
          algo=algorithm.model_name,
          seed=extras["seed"],
          episode_idx=i,
        )
      )

  test_params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), test_params)
  nparams = test_params.reset_params.train_objects.shape[0]
  index = lambda x: jax.tree_util.tree_map(lambda x: x[idx][None], x)
  for idx in tqdm(
    range(nparams), desc=f"{algorithm.model_name}: Generating maze episodes"
  ):
    env_params = test_params.replace(reset_params=index(test_params.reset_params))
    train_object = env_params.reset_params.train_objects[0, 0]
    test_object = env_params.reset_params.test_objects[0, 0]

    is_train = env_params.reset_params.train
    rng, rng_subkey = jax.random.split(rng)
    if is_train:
      if debug:
        continue  # skip train episodes in debug mode
      collect_data(env_params, train_object, rng_=rng_subkey, is_eval=False)
    else:
      collect_data(env_params, test_object, rng_=rng_subkey, is_eval=True)

  return all_episodes, all_configs


def make_model_episode_row_data(episode, metadata):
  maze_name = metadata["maze_name"]
  algo = metadata["algo"]
  if algo == "dynaq_shared":
    algo = "preplay"

  env_rotation = jnp.asarray(
    metadata["env_params"].reset_params.rotation
  ).squeeze()  # [2,1] --> 2
  block_name = tuple([int(i) for i in env_rotation])
  condition = int(metadata.get("condition", 1 if metadata["eval"] else 0))

  row = dict(
    # shared across {human, model}, {craftax, jaxmaze}
    domain="jaxmaze",
    algo=algo,
    world_seed=maze_name,
    block_name=str(block_name),
    condition=condition,
    eval=metadata["eval"],
    start_pos=str(get_agent_position(episode.timesteps)[0]),
    manipulation=maze_to_manipulation.get(maze_name),
    task=int(get_task_object(episode.timesteps)),
    task_set=0,
    room=0,  # for compatibility with prior code
    total_reward=float(total_reward(episode)),
    success=float(success(episode)),
    path_length=int(path_length(episode)),
    seed=metadata["seed"],
    user_id=metadata["seed"],  # reflecting human data format
    # idiosyncratic
    maze=maze_name,
    task_vector=str(episode.timesteps.state.task_w[0]),
  )
  row = fix_row(row)
  return row


def fix_row(row):
  new_row = dict(row)

  new_row["world"] = new_row.pop("world_seed")
  reversal = ast.literal_eval(new_row["block_name"])
  new_row["block_name"] = f"reverse(H={bool(reversal[0])},W={bool(reversal[1])})"

  new_row["task_object_id"] = new_row.pop("task")
  new_row["task_set"] = new_row.pop("room")

  if "seed" in new_row and new_row["algo"] != "human":
    new_row["user_id"] = new_row.pop("seed")

  if "manipulation" in new_row:
    if isinstance(new_row["manipulation"], int):
      new_row["manipulation"] = {
        1: "shortcut",
        2: "start",
        3: "paths",
        4: "juncture",
      }[new_row["manipulation"]]
    elif isinstance(new_row["manipulation"], str):
      assert new_row["manipulation"] in ["shortcut", "start", "paths", "juncture"]
    else:
      raise ValueError(f"Invalid manipulation: {new_row['manipulation']}")

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
    "block",
  ]
  for k in keys_to_remove:
    new_row.pop(k, None)

  return new_row


def add_model_reuse_columns(df: nicewebrl.DataFrame) -> tuple:
  """Add a 'reuse' column to the DataFrame indicating whether each episode reused paths.

  Args:
      df (DataFrame): Input DataFrame

  Returns:
      tuple[dict, dict]: Dictionaries for reuse and overlap values.
  """

  # Create a dictionary to store reuse values
  all_reuse_dict = {}
  all_overlap_dict = {}
  all_corresponding_train_episode_idx = {}

  reuse_dict, overlap_dict, corresponding_train_episode_idx = get_overlap_dicts_model(
    df,
    train_maze="big_m3_maze1",
    test_maze="big_m3_maze1",
    overlap_threshold=0.5,
  )
  all_reuse_dict.update(reuse_dict)
  all_overlap_dict.update(overlap_dict)
  all_corresponding_train_episode_idx.update(corresponding_train_episode_idx)

  reuse_dict, overlap_dict, corresponding_train_episode_idx = get_overlap_dicts_model(
    df,
    train_maze="big_m1_maze3",
    test_maze="big_m1_maze3_shortcut",
    overlap_threshold=0.7,
  )
  all_reuse_dict.update(reuse_dict)
  all_overlap_dict.update(overlap_dict)
  all_corresponding_train_episode_idx.update(corresponding_train_episode_idx)

  return all_reuse_dict, all_overlap_dict, all_corresponding_train_episode_idx
