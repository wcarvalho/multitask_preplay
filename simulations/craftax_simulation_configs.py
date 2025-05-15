from typing import Tuple, List
from craftax_web_env import (
  CraftaxSymbolicWebEnvNoAutoReset,
  EnvParams,
  MultigoalEnvParams,
)
from craftax_experiment_configs import (
  PATHS_CONFIGS,
  make_block_env_params,
  BLOCK_TO_GOAL,
)
import jax.random
import jax.numpy as jnp
import jax.tree_util as jtu
from flax import struct
import numpy as np
import os
import jax
import craftax_utils

MAX_START_POSITIONS = 5


class TaskConfig(struct.PyTreeNode):
  """Configuration for a single experimental block"""

  world_seed: int
  start_position: Tuple[int, int]
  goal_object: int = None
  placed_goals: List[Tuple[int, int]] = None
  placed_achievements: List[Tuple[int, int]] = None
  goal_locations: List[Tuple[int, int]] = None
  train_objects: List[int] = None
  test_objects: List[int] = None


def make_start_position(start_positions):
  start_position = jnp.zeros((MAX_START_POSITIONS, 2), dtype=jnp.int32)
  return start_position.at[: len(start_positions)].set(jnp.asarray(start_positions))


OPTIMAL_PATHS = {}
env = CraftaxSymbolicWebEnvNoAutoReset()
env_params = EnvParams()

TRAIN_CONFIGS = []
TRAIN_EVAL_CONFIGS = []
TEST_CONFIGS = []
# Create cache path

filepath = os.path.dirname(__file__)
cache_dir = os.path.join(filepath, "craftax_cache/optimal_paths")
os.makedirs(cache_dir, exist_ok=True)
for block_config in PATHS_CONFIGS:
  block_env_params = make_block_env_params(block_config, env_params)

  def get_params_and_path(
    world_seed,
    start_position,
    goal_object,
    recompute=False,
  ):
    cache_file = f"path_{world_seed}_{start_position}_{goal_object}.npy"
    cache_file = os.path.join(cache_dir, cache_file)
    params = block_env_params.replace(
      start_positions=make_start_position(start_position),
    )
    assert len(params.placed_goals) == 3, "missing?"
    if os.path.exists(cache_file) and not recompute:
      path = np.load(cache_file)
    else:
      # Calculate path for each start position + test object location
      obs, state = env.reset(jax.random.PRNGKey(0), params)

      path, _ = craftax_utils.astar(state, goal_object)
      path = np.array(path)
      np.save(cache_file, path)

    return params, path

  def get_path_waypoints(path, num_segments=10):
    indices = np.linspace(0, len(path) - 1, num_segments + 1, dtype=int)

    indices = indices[:-1]
    # Get waypoints at those indices
    positions = path[indices]
    return positions

  def add_to_configs(
    world_seed, start_position, goal_object, configs, evaluation=False
  ):
    params, path = get_params_and_path(
      world_seed=world_seed,
      start_position=start_position,
      goal_object=goal_object,
    )
    if len(path) == 0:
      raise RuntimeError("Empty path?")

    waypoints = get_path_waypoints(path)
    if evaluation:
      waypoints = waypoints[:1]

    def blocks_to_goals(blocks):
      return jnp.asarray([BLOCK_TO_GOAL[i] for i in blocks])

    for waypoint in waypoints:
      configs.append(
        TaskConfig(
          world_seed=world_seed,
          start_position=waypoint,
          goal_object=BLOCK_TO_GOAL[goal_object],
          placed_goals=jnp.asarray(params.placed_goals),
          placed_achievements=jnp.asarray(params.placed_achievements),
          goal_locations=jnp.asarray(params.goal_locations),
          train_objects=blocks_to_goals(block_config.train_objects),
          test_objects=blocks_to_goals(block_config.test_objects),
        )
      )

  for start_position in block_config.start_eval_positions:
    # first test
    add_to_configs(
      world_seed=block_config.world_seed,
      start_position=start_position,
      goal_object=block_config.test_objects[0],
      configs=TEST_CONFIGS,
      evaluation=True,
    )

    # then train main eval
    add_to_configs(
      world_seed=block_config.world_seed,
      start_position=start_position,
      goal_object=block_config.train_objects[0],
      configs=TRAIN_EVAL_CONFIGS,
      evaluation=True,
    )

  for start_position in (
    block_config.start_eval_positions + block_config.start_train_positions
  ):
    # train main
    add_to_configs(
      world_seed=block_config.world_seed,
      start_position=start_position,
      goal_object=block_config.train_objects[0],
      configs=TRAIN_CONFIGS,
    )

    # train distractor
    add_to_configs(
      world_seed=block_config.world_seed,
      start_position=start_position,
      goal_object=block_config.train_objects[1],
      configs=TRAIN_CONFIGS,
    )

TRAIN_CONFIGS = jtu.tree_map(lambda *x: jnp.stack(x), *TRAIN_CONFIGS)
TRAIN_EVAL_CONFIGS = jtu.tree_map(lambda *x: jnp.stack(x), *TRAIN_EVAL_CONFIGS)
TEST_CONFIGS = jtu.tree_map(lambda *x: jnp.stack(x), *TEST_CONFIGS)
dummy_config = jax.tree_map(lambda x: x[0], TRAIN_CONFIGS)


default_params = MultigoalEnvParams().replace(
  world_seeds=(dummy_config.world_seed,),
  current_goal=dummy_config.goal_object.astype(jnp.int32),
  start_positions=dummy_config.start_position.astype(jnp.int32),
  placed_goals=dummy_config.placed_goals.astype(jnp.int32),
  placed_achievements=dummy_config.placed_achievements.astype(jnp.int32),
  goal_locations=dummy_config.goal_locations.astype(jnp.int32),
  train_objects=dummy_config.train_objects.astype(jnp.int32),
  test_objects=dummy_config.test_objects.astype(jnp.int32),
  task_configs=TRAIN_CONFIGS,
)


def make_multigoal_env_params(configs):
  return default_params.replace(
    task_configs=configs,
  )
