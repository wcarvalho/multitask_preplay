"""
Based off of: craftax.craftax.envs.craftax_symbolic_env.CraftaxSymbolicEnvNoAutoReset

Changes:
- Added a goal-conditioned reward
- Humans start with 20 strength to more easily kill enemies
- structured obs with {image, task-vector}
- sample world set from pre-defined set of seeds
- goal-conditioned reward
- Current goal is a part of the state to effect the reward function.
- Player starts with pickaxe to mine stone easily
- made stones passable (i.e. can walk over them)
"""

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from jax import lax
from gymnax.environments import spaces, environment
from typing import Tuple, Optional, Union
import chex
from flax import struct
from functools import partial
import distrax
from typing import List

from craftax.craftax.envs.common import log_achievements_to_info
from craftax.environment_base.environment_bases import EnvironmentNoAutoReset
from craftax.craftax.constants import *
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from craftax.craftax.renderer import render_craftax_symbolic
from craftax.craftax.world_gen.world_gen import generate_world


# NOTE: slight hack to import all game logic functions
from craftax.craftax.world_gen.world_gen import (
  generate_world,
  generate_smoothworld,
  generate_dungeon,
  ALL_SMOOTHGEN_CONFIGS,
  ALL_DUNGEON_CONFIGS,
  Mobs,
  get_new_full_inventory,
  get_new_empty_inventory,
)
from craftax.craftax.game_logic import *
from craftax.craftax.util.game_logic_utils import *

try:
  from craftax_game_logic import craftax_step
except ImportError:
  from simulations.craftax_game_logic import craftax_step
except Exception as e:
  raise e


# much smaller action space for web env
class Action(Enum):
  NOOP = 0
  LEFT = 1
  RIGHT = 2
  UP = 3
  DOWN = 4
  DO = 5


# Helper function to get the map view
def _get_map_view(state: EnvState) -> jnp.ndarray:
  """Extracts the agent's current map view."""
  map_level = state.map[state.player_level]
  obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)
  padding = MAX_OBS_DIM + 2  # Padding added in rendering
  tl_corner = state.player_position - obs_dim_array // 2 + padding

  padded_map = jnp.pad(
    map_level,
    (padding, padding),
    constant_values=BlockType.OUT_OF_BOUNDS.value,
  )
  map_view = jax.lax.dynamic_slice(padded_map, tl_corner, OBS_DIM)
  return map_view


def is_game_over(state, params, static_env_params):
  done_steps = state.timestep >= params.max_timesteps
  is_dead = state.player_health <= 0

  return jnp.logical_or(done_steps, is_dead)


@struct.dataclass
class EnvParams:
  max_timesteps: int = 100000
  day_length: int = 300

  always_diamond: bool = False

  mob_despawn_distance: int = 100000
  max_attribute: int = 5

  god_mode: bool = False
  world_seeds: Tuple[int, ...] = tuple()
  # possible_goals: Tuple[int, ...] = tuple()
  current_goal: int = 0
  start_positions: Optional[Union[Tuple[Tuple[int, int]], Tuple[int, int]]] = None
  goal_locations: Tuple[Tuple[int, int]] = ((-1, -1),)
  placed_goals: Tuple[int] = (-1,)
  placed_achievements: Tuple[int] = (-1,)
  fractal_noise_angles: tuple[int, int, int, int] = (None, None, None, None)
  ## for env wrapper
  # active_goals: Tuple[int, ...] = tuple()
  # num_success: int = 5
  train_objects: Tuple[int] = tuple()
  test_objects: Tuple[int] = tuple()


@struct.dataclass
class StaticEnvParams:
  map_size: Tuple[int, int] = (48, 48)
  num_levels: int = 1

  # Mobs
  max_melee_mobs: int = 1
  max_passive_mobs: int = 10
  max_growing_plants: int = 10
  max_ranged_mobs: int = 1
  max_mob_projectiles: int = 3
  max_player_projectiles: int = 3
  use_precondition: bool = False

  # Custom addiitons for human experiments
  initial_crafting_tables: bool = True
  initial_strength: int = 20
  landmark_features: bool = False


@struct.dataclass
class EnvState:
  map: jnp.ndarray
  item_map: jnp.ndarray
  mob_map: jnp.ndarray
  light_map: jnp.ndarray
  down_ladders: jnp.ndarray
  up_ladders: jnp.ndarray
  chests_opened: jnp.ndarray
  monsters_killed: jnp.ndarray

  player_position: jnp.ndarray
  player_level: int
  player_direction: int

  # Intrinsics
  player_health: float
  player_food: int
  player_drink: int
  player_energy: int
  player_mana: int
  is_sleeping: bool
  is_resting: bool

  # Second order intrinsics
  player_recover: float
  player_hunger: float
  player_thirst: float
  player_fatigue: float
  player_recover_mana: float

  # Attributes
  player_xp: int
  player_dexterity: int
  player_strength: int
  player_intelligence: int

  inventory: Inventory

  melee_mobs: Mobs
  passive_mobs: Mobs
  ranged_mobs: Mobs

  mob_projectiles: Mobs
  mob_projectile_directions: jnp.ndarray
  player_projectiles: Mobs
  player_projectile_directions: jnp.ndarray

  growing_plants_positions: jnp.ndarray
  growing_plants_age: jnp.ndarray
  growing_plants_mask: jnp.ndarray

  potion_mapping: jnp.ndarray
  learned_spells: jnp.ndarray

  sword_enchantment: int
  bow_enchantment: int
  armour_enchantments: jnp.ndarray

  boss_progress: int
  boss_timesteps_to_spawn_this_round: int

  light_level: float

  achievements: jnp.ndarray

  state_rng: Any

  timestep: int

  # ADDED FOR EXPERIMENT
  current_goal: int
  start_position: jnp.ndarray
  goal_location: jnp.ndarray
  fractal_noise_angles: tuple[int, int, int, int] = (None, None, None, None)


class Observation(struct.PyTreeNode):
  image: chex.Array
  goal: chex.Array


def get_map_obs_shape():
  num_mob_classes = 5
  num_mob_types = 8
  num_blocks = len(BlockType)
  num_items = len(ItemType)

  return (
    OBS_DIM[0],
    OBS_DIM[1],
    num_blocks + num_items + num_mob_classes * num_mob_types + 1,
  )


def get_flat_map_obs_shape():
  map_obs_shape = get_map_obs_shape()
  return map_obs_shape[0] * map_obs_shape[1] * map_obs_shape[2]


def get_inventory_obs_shape():
  return 51


def generate_world(rng, rng_select, params, static_params):
  # Get default or random start position
  player_position = jnp.array(
    [static_params.map_size[0] // 2, static_params.map_size[1] // 2]
  )

  # Generate smoothgens (overworld, caves, elemental levels, boss level)
  rngs = jax.random.split(rng, 7)
  rng, _rng = rngs[0], rngs[1:]

  smoothgens = jax.vmap(generate_smoothworld, in_axes=(0, None, None, 0, None))(
    _rng, static_params, player_position, ALL_SMOOTHGEN_CONFIGS, params
  )

  # Generate dungeons
  rngs = jax.random.split(rng, 4)
  rng, _rng = rngs[0], rngs[1:]
  dungeons = jax.vmap(generate_dungeon, in_axes=(0, None, 0))(
    _rng, static_params, ALL_DUNGEON_CONFIGS
  )

  # Splice smoothgens and dungeons in order of levels
  map, item_map, light_map, ladders_down, ladders_up = tree_map(
    lambda x, y: jnp.stack(
      (x[0], y[0], x[1], y[1], y[2], x[2], x[3], x[4], x[5]), axis=0
    ),
    smoothgens,
    dungeons,
  )

  # Mobs
  def generate_empty_mobs(max_mobs):
    return Mobs(
      position=jnp.zeros((static_params.num_levels, max_mobs, 2), dtype=jnp.int32),
      health=jnp.ones((static_params.num_levels, max_mobs), dtype=jnp.float32),
      mask=jnp.zeros((static_params.num_levels, max_mobs), dtype=bool),
      attack_cooldown=jnp.zeros((static_params.num_levels, max_mobs), dtype=jnp.int32),
      type_id=jnp.zeros((static_params.num_levels, max_mobs), dtype=jnp.int32),
    )

  melee_mobs = generate_empty_mobs(static_params.max_melee_mobs)
  ranged_mobs = generate_empty_mobs(static_params.max_ranged_mobs)
  passive_mobs = generate_empty_mobs(static_params.max_passive_mobs)

  # Projectiles
  def _create_projectiles(max_num):
    projectiles = generate_empty_mobs(max_num)

    projectile_directions = jnp.ones(
      (static_params.num_levels, max_num, 2), dtype=jnp.int32
    )

    return projectiles, projectile_directions

  mob_projectiles, mob_projectile_directions = _create_projectiles(
    static_params.max_mob_projectiles
  )
  player_projectiles, player_projectile_directions = _create_projectiles(
    static_params.max_player_projectiles
  )

  # Plants
  growing_plants_positions = jnp.zeros(
    (static_params.max_growing_plants, 2), dtype=jnp.int32
  )
  growing_plants_age = jnp.zeros(static_params.max_growing_plants, dtype=jnp.int32)
  growing_plants_mask = jnp.zeros(static_params.max_growing_plants, dtype=bool)

  # Potion mapping for episode
  rng, _rng = jax.random.split(rng)
  potion_mapping = jax.random.permutation(_rng, jnp.arange(6))

  # Inventory
  inventory = tree_map(
    lambda x, y: jax.lax.select(params.god_mode, x, y),
    get_new_full_inventory(),
    get_new_empty_inventory(),
  )

  rng, _rng = jax.random.split(rng)

  # ========================================
  # Add 2 random crafting tables
  # ========================================
  def find_empty_position(rng, current_map, diamond_pos=None, max_distance=5):
    # Generate multiple candidate positions around the diamond
    rng, _rng = jax.random.split(rng)
    num_candidates = 10  # Try multiple positions to increase success chance

    # If diamond position is provided, generate positions within max_distance
    def generate_positions(rng):
      if diamond_pos is not None:
        # Generate offsets from -max_distance to +max_distance
        offsets = jax.random.randint(
          rng,
          shape=(num_candidates, 2),
          minval=-max_distance,
          maxval=max_distance + 1,
        )
        # Add offsets to diamond position
        positions = diamond_pos + offsets
        # Clip to ensure within map bounds
        positions = jnp.clip(
          positions,
          a_min=2,
          a_max=jnp.array(
            [static_params.map_size[0] - 2, static_params.map_size[1] - 2]
          ),
        )
      else:
        # Original random position generation if no diamond_pos
        positions = jax.random.randint(
          rng,
          shape=(num_candidates, 2),
          minval=2,
          maxval=jnp.array(
            [static_params.map_size[0] - 2, static_params.map_size[1] - 2]
          ),
        )
      return positions

    positions = generate_positions(_rng)

    # Check which positions are empty (0 typically represents empty space)
    is_empty = current_map[positions[:, 0], positions[:, 1]] == 0

    # Take the first empty position found
    valid_pos = positions[jnp.argmax(is_empty)]
    return valid_pos, rng

  # ========================================
  # NOTE: MAIN DIFFERENCE:
  # Add 2 random crafting tables
  # ========================================
  if static_params.initial_crafting_tables:
    # Place first table
    # Find diamond position in overworld (level 0)
    flat_index = jnp.argmax(map[0] == BlockType.DIAMOND.value)
    y = flat_index // map[0].shape[1]
    x = flat_index % map[0].shape[1]
    diamond_pos = jnp.array([y, x])

    rng, _rng = jax.random.split(rng)
    pos, _rng = find_empty_position(
      _rng, map[0], diamond_pos=diamond_pos, max_distance=5
    )
    map = map.at[0, pos[0], pos[1]].set(BlockType.CRAFTING_TABLE.value)

    # Place second table
    rng, _rng = jax.random.split(_rng)
    pos, _rng = find_empty_position(
      _rng, map[0], diamond_pos=diamond_pos, max_distance=5
    )
    map = map.at[0, pos[0], pos[1]].set(BlockType.CRAFTING_TABLE.value)

  # ========================================
  # NOTE: give agent pickaxe to mine stone easily
  # ========================================
  inventory = inventory.replace(pickaxe=5)

  if params.start_positions is not None:
    start_positions = jnp.array(params.start_positions)
    if start_positions.ndim == 2:
      # [N, 2]
      # Instead of boolean masking, use a scan to find valid positions
      valid_positions = start_positions.sum(axis=1) > 0

      # Sample from valid positions using distrax
      rng_select, _rng = jax.random.split(rng_select)
      probs = valid_positions.astype(jnp.float32)
      probs = probs / probs.sum()
      chosen_idx = distrax.Categorical(probs=probs).sample(seed=_rng)
      player_position = jax.lax.dynamic_index_in_dim(
        start_positions,
        chosen_idx,
        keepdims=False,
      )
    elif start_positions.ndim == 1:
      # [2]
      assert start_positions.shape == (2,), "should be (y, z)"
      player_position = start_positions
    else:
      raise NotImplementedError(start_positions.ndim)

  ########################################################
  # NOTE: MAIN DIFFERENCE:
  # Place multiple goals at locations
  ########################################################
  index = lambda x, i: jax.lax.dynamic_index_in_dim(
    x,
    i,
    keepdims=False,
  )
  goal_locations = jnp.asarray(params.goal_locations)
  placed_goals = jnp.asarray(params.placed_goals)

  def place_goal_at_index(i, m):
    goal_pos = goal_locations[i]

    def place_goal(m_):
      goal_value = index(placed_goals, i).astype(jnp.int32)
      return m_.at[0, goal_pos[0], goal_pos[1]].set(goal_value)

    return jax.lax.cond(jnp.asarray(goal_pos).sum() > 0, place_goal, lambda m_: m_, m)

  map = jax.lax.fori_loop(0, len(params.goal_locations), place_goal_at_index, map)

  flat_index = jnp.argmax(map[0] == params.current_goal)
  y = flat_index // map[0].shape[1]
  x = flat_index % map[0].shape[1]
  goal_pos = jnp.array([y, x], dtype=jnp.int32)
  state = EnvState(
    map=map,
    item_map=item_map,
    mob_map=jnp.zeros((static_params.num_levels, *static_params.map_size), dtype=bool),
    light_map=light_map,
    down_ladders=ladders_down,
    up_ladders=ladders_up,
    chests_opened=jnp.zeros(static_params.num_levels, dtype=bool),
    monsters_killed=jnp.zeros(static_params.num_levels, dtype=jnp.int32)
    .at[0]
    .set(10),  # First ladder starts open
    player_position=player_position,
    player_direction=jnp.asarray(Action.UP.value, dtype=jnp.int32),
    player_level=jnp.asarray(0, dtype=jnp.int32),
    player_health=jnp.asarray(9.0, dtype=jnp.float32),
    player_food=jnp.asarray(9, dtype=jnp.int32),
    player_drink=jnp.asarray(9, dtype=jnp.int32),
    player_energy=jnp.asarray(9, dtype=jnp.int32),
    player_mana=jnp.asarray(9, dtype=jnp.int32),
    player_recover=jnp.asarray(0.0, dtype=jnp.float32),
    player_hunger=jnp.asarray(0.0, dtype=jnp.float32),
    player_thirst=jnp.asarray(0.0, dtype=jnp.float32),
    player_fatigue=jnp.asarray(0.0, dtype=jnp.float32),
    player_recover_mana=jnp.asarray(0.0, dtype=jnp.float32),
    is_sleeping=False,
    is_resting=False,
    player_xp=jnp.asarray(0, dtype=jnp.int32),
    player_dexterity=jnp.asarray(1, dtype=jnp.int32),
    # NOTE: MAIN DIFFERENCE: Humans start with 20 strength
    player_strength=jnp.asarray(static_params.initial_strength, dtype=jnp.int32),
    player_intelligence=jnp.asarray(1, dtype=jnp.int32),
    inventory=inventory,
    sword_enchantment=jnp.asarray(0, dtype=jnp.int32),
    bow_enchantment=jnp.asarray(0, dtype=jnp.int32),
    armour_enchantments=jnp.array([0, 0, 0, 0], dtype=jnp.int32),
    melee_mobs=melee_mobs,
    ranged_mobs=ranged_mobs,
    passive_mobs=passive_mobs,
    mob_projectiles=mob_projectiles,
    mob_projectile_directions=mob_projectile_directions,
    player_projectiles=player_projectiles,
    player_projectile_directions=player_projectile_directions,
    growing_plants_positions=growing_plants_positions,
    growing_plants_age=growing_plants_age,
    growing_plants_mask=growing_plants_mask,
    potion_mapping=potion_mapping,
    learned_spells=jnp.array([False, False], dtype=bool),
    boss_progress=jnp.asarray(0, dtype=jnp.int32),
    boss_timesteps_to_spawn_this_round=jnp.asarray(
      BOSS_FIGHT_SPAWN_TURNS, dtype=jnp.int32
    ),
    achievements=jnp.zeros((len(Achievement),), dtype=bool),
    light_level=jnp.asarray(calculate_light_level(0, params), dtype=jnp.float32),
    state_rng=_rng,
    timestep=jnp.asarray(0, dtype=jnp.int32),
    start_position=jnp.asarray(player_position, dtype=jnp.int32),
    current_goal=jnp.asarray(params.current_goal, dtype=jnp.int32),
    goal_location=goal_pos,
  )

  return state


class CraftaxSymbolicWebEnvNoAutoReset(EnvironmentNoAutoReset):
  def __init__(self, static_env_params: Optional[StaticEnvParams] = None):
    super().__init__()

    if static_env_params is None:
      static_env_params = self.default_static_params()
    self.static_env_params = static_env_params

  @property
  def default_params(self) -> EnvParams:
    return EnvParams()

  @staticmethod
  def default_static_params() -> StaticEnvParams:
    return StaticEnvParams()

  def step(
    self,
    key: chex.PRNGKey,
    state,
    action: Union[int, float],
    params=None,
  ):
    """Performs step transitions in the environment."""
    # Use default env parameters if no others specified
    obs, state, reward, done, info = self.step_env(key, state, action, params)
    return obs, state, reward, done, info

  def step_env(
    self, rng: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
  ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
    """

    Changes:
     1. have single achievement complete task.
     2. Stopped having params be a static parameter. It will change as a function of the goal.
    """

    def step(state: EnvState, action: int, params: EnvParams):
      state, _ = craftax_step(rng, state, action, params, self.static_env_params)
      goal_achieved = jax.lax.dynamic_index_in_dim(
        state.achievements,
        state.current_goal.astype(jnp.int32),
        keepdims=False,
      )
      goal_achieved = goal_achieved.astype(jnp.float32)
      return state, goal_achieved

    state, reward = step(state, action, params)

    of_interest = jnp.asarray(
      (
        Achievement.COLLECT_DIAMOND.value,
        Achievement.COLLECT_SAPPHIRE.value,
        Achievement.COLLECT_RUBY.value,
        Achievement.COLLECT_IRON.value,
        Achievement.COLLECT_COAL.value,
      ),
      dtype=jnp.int32,
    )
    of_interest = state.achievements[of_interest]
    done = jnp.logical_or(of_interest.sum() > 0, reward > 0)
    done = jnp.logical_or(done, self.is_terminal(state, params))

    info = log_achievements_to_info(state, done)
    info["discount"] = self.discount(state, params)

    return (
      lax.stop_gradient(self.get_obs(state=state, action=action, params=params)),
      lax.stop_gradient(state),
      reward,
      done,
      info,
    )

  def reset(self, key: chex.PRNGKey, params):
    """Performs resetting of environment."""
    obs, state = self.reset_env(key, params)
    return obs, state

  def reset_env(
    self, rng: chex.PRNGKey, params: EnvParams
  ) -> Tuple[chex.Array, EnvState]:
    """NOTE: main change is to select world seed from a set of seeds"""
    if params.world_seeds is not None and len(params.world_seeds) > 0:
      reset_seeds = jnp.asarray(params.world_seeds)
      rng, _rng = jax.random.split(rng)
      selected_seed = jax.random.choice(_rng, reset_seeds)
      world_rng = jax.random.PRNGKey(selected_seed)
    else:
      rng, world_rng = jax.random.split(rng)

    rng, rng_select = jax.random.split(rng)
    state = generate_world(world_rng, rng_select, params, self.static_env_params)

    obs = self.get_obs(
      state=state,
      action=jnp.zeros((), dtype=jnp.int32),  # scalar
      params=params,
    )
    return obs, state

  def get_obs(self, state: EnvState, action: int, params: EnvParams):
    del params
    del action
    return render_craftax_symbolic(state)

  def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
    return is_game_over(state, params, self.static_env_params)

  @property
  def name(self) -> str:
    return "Craftax-Symbolic-NoAutoReset-v1"

  @property
  def num_actions(self) -> int:
    return len(Action)

  def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
    return spaces.Discrete(len(Action))

  def observation_space(self, params: EnvParams) -> spaces.Box:
    flat_map_obs_shape = get_flat_map_obs_shape()
    inventory_obs_shape = get_inventory_obs_shape()

    obs_shape = flat_map_obs_shape + inventory_obs_shape

    return spaces.Box(
      0.0,
      1.0,
      (obs_shape,),
      dtype=jnp.float32,
    )


########################################################
# Multi-goal version of the Craftax environment
########################################################
Achiement_to_idx = {
  Achievement.COLLECT_DIAMOND.value: 0,
  Achievement.COLLECT_SAPPHIRE.value: 1,
  Achievement.COLLECT_RUBY.value: 2,
}

IDX_to_Achievement = jnp.asarray(
  (
    Achievement.COLLECT_DIAMOND.value,
    Achievement.COLLECT_SAPPHIRE.value,
    Achievement.COLLECT_RUBY.value,
  )
)
task_vectors = jnp.zeros((len(Achievement), len(Achiement_to_idx)))
active_task_vectors = []
for achievement, i in Achiement_to_idx.items():
  task_vectors = task_vectors.at[achievement, i].set(1)
  active_task_vectors.append(task_vectors[achievement])
active_task_vectors = jnp.stack(active_task_vectors)


def task_onehot(goal):
  return jax.lax.dynamic_index_in_dim(task_vectors, goal, keepdims=False)


Block_to_idx = jnp.zeros((len(BlockType),), dtype=jnp.int32)
for idx, block_type in enumerate(
  (
    BlockType.DIAMOND.value,
    BlockType.SAPPHIRE.value,
    BlockType.RUBY.value,
  )
):
  Block_to_idx = Block_to_idx.at[block_type].set(idx)


@struct.dataclass
class MultigoalEnvParams(EnvParams):
  task_configs: struct.PyTreeNode = None


@struct.dataclass
class MultiGoalObservation(struct.PyTreeNode):
  image: chex.Array
  task_w: chex.Array
  train_tasks: chex.Array
  # goals: chex.Array
  state_features: chex.Array
  previous_action: int = None


class CraftaxMultiGoalSymbolicWebEnvNoAutoReset(CraftaxSymbolicWebEnvNoAutoReset):
  """
  This is a multi-goal version of the Craftax environment.
  """

  @property
  def default_params(self) -> EnvParams:
    return MultigoalEnvParams()

  def reset(self, key: chex.PRNGKey, params: MultigoalEnvParams):
    """
    Sample a task config from the list of task configs.
    Fill in information
    """
    n_tasks = len(params.task_configs.world_seed)
    task_idx = jax.random.randint(key, (), 0, n_tasks)
    index = lambda x: jax.lax.dynamic_index_in_dim(x, task_idx, keepdims=False)
    task_config = jax.tree_map(index, params.task_configs)

    params = params.replace(
      world_seeds=(task_config.world_seed,),
      current_goal=task_config.goal_object.astype(jnp.int32),
      start_positions=task_config.start_position.astype(jnp.int32),
      placed_goals=task_config.placed_goals.astype(jnp.int32),  # Block type
      placed_achievements=task_config.placed_achievements.astype(
        jnp.int32
      ),  # Achievement type
      goal_locations=task_config.goal_locations.astype(jnp.int32),
      train_objects=task_config.train_objects.astype(jnp.int32),
      test_objects=task_config.test_objects.astype(jnp.int32),
    )

    return super().reset(key, params)

  def get_obs(self, state: EnvState, action: int, params: EnvParams):
    # [D]
    image = render_craftax_symbolic(state)

    # [G]
    task_w = task_onehot(state.current_goal)

    # [N, G]
    # compute state features as whether any params.placed_goal is achieved
    def achieved(achievement):
      complete = jax.lax.dynamic_index_in_dim(
        state.achievements, achievement.astype(jnp.int32), keepdims=False
      ).astype(jnp.float32)

      return task_onehot(achievement) * complete

    # N 1-hots
    achievement_state_features = jax.vmap(achieved)(params.placed_achievements)
    # place them all in same vector of length [N]
    achievement_state_features = achievement_state_features.sum(axis=0)

    map_view = _get_map_view(state)

    def visibility_feature(b):
      visible = jnp.any(map_view == b)
      idx = jax.lax.dynamic_index_in_dim(Block_to_idx, b, keepdims=False).astype(
        jnp.int32
      )
      return jnp.zeros((len(achievement_state_features))).at[idx].set(visible)

    visibility_state_features = jax.vmap(visibility_feature)(params.placed_goals)
    visibility_state_features = visibility_state_features.sum(axis=0)

    if self.static_env_params.landmark_features:
      state_features = jnp.concatenate(
        [achievement_state_features, visibility_state_features], axis=0
      )
    else:
      state_features = achievement_state_features

    # compute train tasks
    def task_onehot_prime(w):
      if self.static_env_params.landmark_features:
        return jnp.concatenate([w, jnp.zeros_like(visibility_state_features)], axis=0)
      else:
        return w

    # add dummy dimensions for visibility
    task_w = task_onehot_prime(task_w)

    train_tasks = jax.vmap(task_onehot)(params.train_objects)
    train_tasks = jax.vmap(task_onehot_prime)(train_tasks)

    return MultiGoalObservation(
      image=image,
      task_w=task_w,
      previous_action=action,
      train_tasks=train_tasks,
      # goals=active_task_vectors,
      state_features=state_features,
    )


########################################################
# Dummy version of the Craftax environment
########################################################
class CraftaxSymbolicWebEnvNoAutoResetDummy(EnvironmentNoAutoReset):
  """
  A dummy version of the Craftax environment with empty core functions.
  """

  def __init__(self, static_env_params: StaticEnvParams = None):
    super().__init__()
    if static_env_params is None:
      static_env_params = self.default_static_params()
    self.static_env_params = static_env_params

  @property
  def default_params(self) -> EnvParams:
    return EnvParams()

  @staticmethod
  def default_static_params() -> StaticEnvParams:
    return StaticEnvParams()

  def step_env(
    self, key: jnp.ndarray, state: EnvState, action: int, params: EnvParams
  ) -> Tuple[jnp.ndarray, EnvState, float, bool, dict]:
    """Simple step function that moves the player based on action"""
    # Get map dimensions from static params
    map_height, map_width = self.static_env_params.map_size

    # Update player position based on action
    y, x = state.player_position

    def up(xy):
      return (xy[0], jnp.clip(xy[1] - 1, 0, map_height - 1))

    def down(xy):
      return (xy[0], jnp.clip(xy[1] + 1, 0, map_height - 1))

    def left(xy):
      return (jnp.clip(xy[0] - 1, 0, map_width - 1), xy[1])

    def right(xy):
      return (jnp.clip(xy[0] + 1, 0, map_width - 1), xy[1])

    def noop(xy):
      return xy

    x, y = jax.lax.switch(action, [noop, left, right, up, down], (x, y))

    # Update state with new position
    new_state = state.replace(player_position=jnp.array([y, x]))

    reward = 1.0
    done = True
    info = {}
    obs = self.get_obs(new_state, params)
    return obs, new_state, reward, done, info

  def reset_env(
    self, key: jnp.ndarray, params: EnvParams
  ) -> Tuple[jnp.ndarray, EnvState]:
    """Reset function that initializes player at params-determined position"""
    static_params = self.static_env_params
    map_height, map_width = static_params.map_size

    player_position = jnp.array(
      [static_params.map_size[0] // 2, static_params.map_size[1] // 2]
    )
    if params.start_positions is not None:
      start_positions = jnp.array(params.start_positions)
      if start_positions.ndim == 2:
        # [N, 2]
        # Instead of boolean masking, use a scan to find valid positions
        valid_positions = start_positions.sum(axis=1) > 0

        # Sample from valid positions using distrax
        rng, _rng = jax.random.split(key)
        probs = valid_positions.astype(jnp.float32)
        probs = probs / probs.sum()
        chosen_idx = distrax.Categorical(probs=probs).sample(seed=_rng)
        player_position = jax.lax.dynamic_index_in_dim(
          start_positions,
          chosen_idx,
          keepdims=False,
        )
      elif start_positions.ndim == 1:
        # [2]
        assert start_positions.shape == (2,), "should be (y, z)"
        player_position = start_positions
      else:
        raise NotImplementedError(start_positions.ndim)

    current_goal = jnp.array(params.current_goal, dtype=jnp.int32)
    placed_goals = jnp.array(params.placed_goals, dtype=jnp.int32)
    match = placed_goals == current_goal

    goal_locations = jnp.array(params.goal_locations, dtype=jnp.int32)
    goal_location = jax.lax.dynamic_index_in_dim(
      goal_locations, jnp.argmax(match), keepdims=False
    )

    state = EnvState(
      map=jnp.full(
        static_params.map_size, params.world_seeds[0]
      ),  # Fill map with world seed value
      item_map=jnp.zeros(static_params.map_size),
      mob_map=jnp.zeros(static_params.map_size),
      light_map=jnp.zeros(static_params.map_size),
      down_ladders=jnp.zeros(static_params.map_size),
      up_ladders=jnp.zeros(static_params.map_size),
      chests_opened=jnp.zeros(static_params.map_size),
      monsters_killed=jnp.zeros(static_params.map_size),
      player_position=player_position,  # Set starting position from params
      player_direction=jnp.asarray(0, dtype=jnp.int32),
      player_level=jnp.asarray(0, dtype=jnp.int32),
      player_health=jnp.asarray(9.0, dtype=jnp.float32),
      player_food=jnp.asarray(9, dtype=jnp.int32),
      player_drink=jnp.asarray(9, dtype=jnp.int32),
      player_energy=jnp.asarray(9, dtype=jnp.int32),
      player_mana=jnp.asarray(9, dtype=jnp.int32),
      player_recover=jnp.asarray(0.0, dtype=jnp.float32),
      player_hunger=jnp.asarray(0.0, dtype=jnp.float32),
      player_thirst=jnp.asarray(0.0, dtype=jnp.float32),
      player_fatigue=jnp.asarray(0.0, dtype=jnp.float32),
      player_recover_mana=jnp.asarray(0.0, dtype=jnp.float32),
      is_sleeping=False,
      is_resting=False,
      player_xp=jnp.asarray(0, dtype=jnp.int32),
      player_dexterity=jnp.asarray(1, dtype=jnp.int32),
      player_strength=jnp.asarray(static_params.initial_strength, dtype=jnp.int32),
      player_intelligence=jnp.asarray(1, dtype=jnp.int32),
      inventory=jnp.zeros((1)),
      sword_enchantment=jnp.asarray(0, dtype=jnp.int32),
      bow_enchantment=jnp.asarray(0, dtype=jnp.int32),
      armour_enchantments=jnp.array([0, 0, 0, 0], dtype=jnp.int32),
      melee_mobs=jnp.zeros((1)),
      ranged_mobs=jnp.zeros((1)),
      passive_mobs=jnp.zeros((1)),
      mob_projectiles=jnp.zeros((1)),
      mob_projectile_directions=jnp.zeros((1)),
      player_projectiles=jnp.zeros((1)),
      player_projectile_directions=jnp.zeros((1)),
      growing_plants_positions=jnp.zeros((1)),
      growing_plants_age=jnp.zeros((1)),
      growing_plants_mask=jnp.zeros((1)),
      potion_mapping=jnp.zeros((1)),
      learned_spells=jnp.array([False, False], dtype=bool),
      boss_progress=jnp.asarray(0, dtype=jnp.int32),
      boss_timesteps_to_spawn_this_round=jnp.asarray(0, dtype=jnp.int32),
      achievements=jnp.zeros((1)),
      light_level=jnp.asarray(0.0, dtype=jnp.float32),
      state_rng=jnp.zeros((1)),
      timestep=jnp.asarray(0, dtype=jnp.int32),
      current_goal=jnp.asarray(params.current_goal, dtype=jnp.int32),
      start_position=jnp.asarray(player_position, dtype=jnp.int32),
      goal_location=goal_location,
    )
    obs = self.get_obs(state, params)
    return obs, state

  def get_obs(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
    """Returns the map with player position marked as a 3-channel observation"""
    # Create a 3-channel observation array
    obs = jnp.zeros((*state.map.shape, 3), dtype=jnp.uint8)

    # Mark player position in the second channel
    y, x = state.player_position
    obs = obs.at[y, x, :].set(255)

    return obs

  def observation_space(self, params: EnvParams) -> spaces.Box:
    """Define observation space as map size"""
    map_height, map_width = self.static_env_params.map_size
    return spaces.Box(0, 255, (map_height, map_width), dtype=jnp.uint8)

  def action_space(self, params: EnvParams) -> spaces.Discrete:
    """Define action space"""
    return spaces.Discrete(5)  # Assuming 5 actions as in original

  @property
  def name(self) -> str:
    return "Craftax-Symbolic-NoAutoReset-v1"

  @property
  def num_actions(self) -> int:
    return 5  # Assuming 5 actions as in original
