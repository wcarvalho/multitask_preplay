"""
Based off of: craftax.craftax.envs.craftax_symbolic_env.CraftaxSymbolicEnvNoAutoReset

Changes:
1. Added a set of seeds to EnvParams so that train + env can have different seeds

"""

import jax
from jax import lax
from jax.tree_util import tree_map
from gymnax.environments import spaces, environment
from typing import Tuple, Optional
import chex
from flax import struct

from craftax.craftax.envs.common import log_achievements_to_info
from craftax.environment_base.environment_bases import EnvironmentNoAutoReset
from craftax.craftax.constants import *
from craftax.craftax.game_logic import craftax_step, is_game_over
from craftax.craftax.craftax_state import EnvState
from craftax.craftax.renderer import render_craftax_symbolic
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

# NOTE: slight hack to import all game logic functions
from craftax.craftax.game_logic import *
from craftax.craftax.util.game_logic_utils import *

MAX_ACHIEVEMENT = max([i.value for i in Achievement.__members__.values()])
assert MAX_ACHIEVEMENT + 1 == len(ACHIEVEMENT_REWARD_MAP), (
  "achievements are not contiguous"
)


@struct.dataclass
class EnvParams:
  max_timesteps: int = 100000
  day_length: int = 300

  always_diamond: bool = False

  mob_despawn_distance: int = 14
  max_attribute: int = 5

  god_mode: bool = False
  world_seeds: Tuple[int, ...] = tuple()

  fractal_noise_angles: tuple[int, int, int, int] = (None, None, None, None)


@struct.dataclass
class StaticEnvParams:
  map_size: Tuple[int, int] = (48, 48)
  num_levels: int = 9

  # Mobs
  max_melee_mobs: int = 3
  max_passive_mobs: int = 3
  max_growing_plants: int = 10
  max_ranged_mobs: int = 2
  max_mob_projectiles: int = 3
  max_player_projectiles: int = 3
  use_precondition: bool = False


dummy_achievements = jnp.zeros(len(ACHIEVEMENT_REWARD_MAP) + 1, dtype=jnp.float32)


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


class Observation(struct.PyTreeNode):
  image: chex.Array
  achievable: chex.Array
  achievements: chex.Array
  task_w: chex.Array
  previous_action: Optional[chex.Array] = None


def get_possible_achievements(
  state: EnvState, use_precondition: bool = False
) -> jnp.ndarray:
  """Returns a binary vector indicating which achievements are currently possible given the agent's observation."""

  # Get the visible map and items within observation range
  obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)
  tl_corner = state.player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2

  # Pad and slice the maps similar to render_craftax_symbolic
  padded_grid = jnp.pad(
    state.map[state.player_level],
    (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
    constant_values=BlockType.OUT_OF_BOUNDS.value,
  )
  padded_items = jnp.pad(
    state.item_map[state.player_level],
    (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
    constant_values=ItemType.NONE.value,
  )

  visible_map = jax.lax.dynamic_slice(padded_grid, tl_corner, OBS_DIM)
  visible_items = jax.lax.dynamic_slice(padded_items, tl_corner, OBS_DIM)

  # Get light map
  padded_light = jnp.pad(
    state.light_map[state.player_level],
    (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
    constant_values=0.0,
  )
  light_map = jax.lax.dynamic_slice(padded_light, tl_corner, OBS_DIM) > 0.05

  # Initialize achievement possibilities array (using max value from Achievement enum + 1)
  max_achievement = max(achievement.value for achievement in Achievement)
  possible_achievements = jnp.zeros(max_achievement + 1, dtype=jnp.int32)

  # Check for gatherable resources in visible and lit areas
  visible_blocks = visible_map * light_map
  visible_items = visible_items * light_map

  # Resource gathering possibilities with tool requirements
  has_tree_nearby = jnp.any(
    jnp.logical_or(
      visible_blocks == BlockType.TREE.value,
      jnp.logical_or(
        visible_blocks == BlockType.FIRE_TREE.value,
        visible_blocks == BlockType.ICE_SHRUB.value,
      ),
    )
  )
  has_stone_nearby = jnp.logical_and(
    jnp.any(visible_blocks == BlockType.STONE.value),
    state.inventory.pickaxe >= 1 if use_precondition else True,
  )
  has_coal_nearby = jnp.logical_and(
    jnp.any(visible_blocks == BlockType.COAL.value),
    state.inventory.pickaxe >= 1 if use_precondition else True,
  )
  has_iron_nearby = jnp.logical_and(
    jnp.any(visible_blocks == BlockType.IRON.value),
    state.inventory.pickaxe >= 2 if use_precondition else True,
  )
  has_diamond_nearby = jnp.logical_and(
    jnp.any(visible_blocks == BlockType.DIAMOND.value),
    state.inventory.pickaxe >= 3 if use_precondition else True,
  )
  has_sapphire_nearby = jnp.logical_and(
    jnp.any(visible_blocks == BlockType.SAPPHIRE.value),
    state.inventory.pickaxe >= 4 if use_precondition else True,
  )
  has_ruby_nearby = jnp.logical_and(
    jnp.any(visible_blocks == BlockType.RUBY.value),
    state.inventory.pickaxe >= 4 if use_precondition else True,
  )

  # Check for mobs in visible range
  def check_mob_nearby(mob_type_id, mob_collection):
    mob_collection = tree_map(lambda x: x[state.player_level], mob_collection)
    mob_positions = mob_collection.position
    mob_types = mob_collection.type_id
    mob_masks = mob_collection.mask

    # lower radius a bit
    low = state.player_position - obs_dim_array // 2 + 2
    high = state.player_position + obs_dim_array // 2 - 2

    is_visible = jnp.logical_and(
      jnp.logical_and(
        jnp.logical_and(
          jnp.logical_and(
            jnp.logical_and(
              mob_positions[:, 0] < high[0], mob_positions[:, 0] >= low[0]
            ),
            mob_positions[:, 1] < high[1],
          ),
          mob_positions[:, 1] >= low[1],
        ),
        mob_masks,
      ),
      mob_types == mob_type_id,
    )
    is_visible = jnp.any(is_visible)

    return jnp.any(is_visible)

  # Resource collection
  possible_achievements = possible_achievements.at[Achievement.COLLECT_WOOD.value].set(
    has_tree_nearby
  )
  possible_achievements = possible_achievements.at[Achievement.COLLECT_STONE.value].set(
    has_stone_nearby
  )
  possible_achievements = possible_achievements.at[Achievement.COLLECT_COAL.value].set(
    has_coal_nearby
  )
  possible_achievements = possible_achievements.at[Achievement.COLLECT_IRON.value].set(
    has_iron_nearby
  )
  possible_achievements = possible_achievements.at[
    Achievement.COLLECT_DIAMOND.value
  ].set(has_diamond_nearby)
  possible_achievements = possible_achievements.at[
    Achievement.COLLECT_SAPPHIRE.value
  ].set(has_sapphire_nearby)
  possible_achievements = possible_achievements.at[Achievement.COLLECT_RUBY.value].set(
    has_ruby_nearby
  )

  # Helper function to check for crafting table
  def is_near_crafting_table():
    return jnp.any(visible_blocks == BlockType.CRAFTING_TABLE.value)

  def is_near_furnace():
    return jnp.any(visible_blocks == BlockType.FURNACE.value)

  # Crafting possibilities (based on inventory and crafting table)
  has_crafting_table = is_near_crafting_table()
  has_furnace = is_near_furnace()

  possible_achievements = possible_achievements.at[
    Achievement.MAKE_WOOD_PICKAXE.value
  ].set(
    jnp.logical_and(
      jnp.logical_and(
        state.inventory.wood >= 1 if use_precondition else True,
        has_crafting_table,
      ),
      state.inventory.pickaxe < 1,
    )
  )
  possible_achievements = possible_achievements.at[
    Achievement.MAKE_WOOD_SWORD.value
  ].set(
    jnp.logical_and(
      jnp.logical_and(
        state.inventory.wood >= 1 if use_precondition else True,
        has_crafting_table,
      ),
      state.inventory.sword < 1,
    )
  )
  possible_achievements = possible_achievements.at[
    Achievement.MAKE_STONE_PICKAXE.value
  ].set(
    jnp.logical_and(
      jnp.logical_and(
        jnp.logical_and(
          state.inventory.wood >= 1 if use_precondition else True,
          state.inventory.stone >= 1 if use_precondition else True,
        ),
        has_crafting_table,
      ),
      state.inventory.pickaxe < 2,
    )
  )
  possible_achievements = possible_achievements.at[
    Achievement.MAKE_STONE_SWORD.value
  ].set(
    jnp.logical_and(
      jnp.logical_and(
        jnp.logical_and(
          state.inventory.wood >= 1 if use_precondition else True,
          state.inventory.stone >= 1 if use_precondition else True,
        ),
        has_crafting_table,
      ),
      state.inventory.sword < 2,
    )
  )
  possible_achievements = possible_achievements.at[
    Achievement.MAKE_IRON_PICKAXE.value
  ].set(
    jnp.logical_and(
      jnp.logical_and(
        jnp.logical_and(
          jnp.logical_and(
            jnp.logical_and(
              state.inventory.wood >= 1 if use_precondition else True,
              state.inventory.stone >= 1 if use_precondition else True,
            ),
            jnp.logical_and(
              state.inventory.iron >= 1 if use_precondition else True,
              state.inventory.coal >= 1 if use_precondition else True,
            ),
          ),
          has_crafting_table,
        ),
        has_furnace,
      ),
      state.inventory.pickaxe < 3,
    )
  )
  possible_achievements = possible_achievements.at[
    Achievement.MAKE_IRON_SWORD.value
  ].set(
    jnp.logical_and(
      jnp.logical_and(
        jnp.logical_and(
          jnp.logical_and(
            jnp.logical_and(
              state.inventory.wood >= 1 if use_precondition else True,
              state.inventory.stone >= 1 if use_precondition else True,
            ),
            jnp.logical_and(
              state.inventory.iron >= 1 if use_precondition else True,
              state.inventory.coal >= 1 if use_precondition else True,
            ),
          ),
          has_crafting_table,
        ),
        has_furnace,
      ),
      state.inventory.sword < 3,
    )
  )

  # Placement possibilities
  possible_achievements = possible_achievements.at[Achievement.PLACE_TABLE.value].set(
    state.inventory.wood >= 4
  )
  possible_achievements = possible_achievements.at[Achievement.PLACE_STONE.value].set(
    state.inventory.stone > 0
  )
  possible_achievements = possible_achievements.at[Achievement.PLACE_FURNACE.value].set(
    state.inventory.stone >= 8
  )
  possible_achievements = possible_achievements.at[Achievement.PLACE_TORCH.value].set(
    state.inventory.torches > 0
  )

  # Helper function to check for blocks in visible and lit areas
  def check_block_nearby(block_type, visible_blocks):
    return jnp.any(visible_blocks == block_type.value)

  # Check for chests and plants
  has_chest_nearby = check_block_nearby(BlockType.CHEST, visible_blocks)
  has_plant_nearby = check_block_nearby(BlockType.RIPE_PLANT, visible_blocks)

  # Interaction possibilities
  possible_achievements = possible_achievements.at[Achievement.OPEN_CHEST.value].set(
    has_chest_nearby
  )
  possible_achievements = possible_achievements.at[Achievement.EAT_PLANT.value].set(
    has_plant_nearby
  )
  possible_achievements = possible_achievements.at[Achievement.DRINK_POTION.value].set(
    jnp.any(state.inventory.potions > 0)
  )

  # Combat-related possibilities (including spells)
  # has_weapon = jnp.logical_or(
  #    state.inventory.sword > 0,
  #    jnp.logical_or(
  #        jnp.logical_and(
  #            state.inventory.bow > 0,
  #            state.inventory.arrows > 0
  #        ),
  #        jnp.logical_and(
  #            jnp.logical_or(
  #                state.learned_spells[0],
  #                state.learned_spells[1]
  #            ),
  #            state.player_mana >= 1
  #        )
  #    )
  # )

  # Melee mobs (using exact Achievement enum values)
  possible_achievements = possible_achievements.at[Achievement.DEFEAT_ZOMBIE.value].set(
    check_mob_nearby(0, state.melee_mobs)
  )
  possible_achievements = possible_achievements.at[
    Achievement.DEFEAT_GNOME_WARRIOR.value
  ].set(check_mob_nearby(1, state.melee_mobs))
  possible_achievements = possible_achievements.at[
    Achievement.DEFEAT_ORC_SOLIDER.value
  ].set(check_mob_nearby(2, state.melee_mobs))
  possible_achievements = possible_achievements.at[Achievement.DEFEAT_LIZARD.value].set(
    check_mob_nearby(3, state.melee_mobs)
  )
  possible_achievements = possible_achievements.at[Achievement.DEFEAT_KNIGHT.value].set(
    check_mob_nearby(4, state.melee_mobs)
  )
  possible_achievements = possible_achievements.at[Achievement.DEFEAT_TROLL.value].set(
    check_mob_nearby(5, state.melee_mobs)
  )
  possible_achievements = possible_achievements.at[Achievement.DEFEAT_PIGMAN.value].set(
    check_mob_nearby(6, state.melee_mobs)
  )
  possible_achievements = possible_achievements.at[
    Achievement.DEFEAT_FROST_TROLL.value
  ].set(check_mob_nearby(7, state.melee_mobs))

  # Ranged mobs
  possible_achievements = possible_achievements.at[
    Achievement.DEFEAT_SKELETON.value
  ].set(check_mob_nearby(0, state.ranged_mobs))
  possible_achievements = possible_achievements.at[
    Achievement.DEFEAT_GNOME_ARCHER.value
  ].set(check_mob_nearby(1, state.ranged_mobs))
  possible_achievements = possible_achievements.at[
    Achievement.DEFEAT_ORC_MAGE.value
  ].set(check_mob_nearby(2, state.ranged_mobs))
  possible_achievements = possible_achievements.at[Achievement.DEFEAT_KOBOLD.value].set(
    check_mob_nearby(3, state.ranged_mobs)
  )
  possible_achievements = possible_achievements.at[Achievement.DEFEAT_ARCHER.value].set(
    check_mob_nearby(4, state.ranged_mobs)
  )
  possible_achievements = possible_achievements.at[
    Achievement.DEFEAT_DEEP_THING.value
  ].set(check_mob_nearby(5, state.ranged_mobs))
  possible_achievements = possible_achievements.at[
    Achievement.DEFEAT_FIRE_ELEMENTAL.value
  ].set(check_mob_nearby(6, state.ranged_mobs))
  possible_achievements = possible_achievements.at[
    Achievement.DEFEAT_ICE_ELEMENTAL.value
  ].set(check_mob_nearby(7, state.ranged_mobs))

  # Passive mobs (for eating)
  possible_achievements = possible_achievements.at[Achievement.EAT_COW.value].set(
    check_mob_nearby(0, state.passive_mobs)
  )
  possible_achievements = possible_achievements.at[Achievement.EAT_BAT.value].set(
    check_mob_nearby(1, state.passive_mobs)
  )
  possible_achievements = possible_achievements.at[Achievement.EAT_SNAIL.value].set(
    check_mob_nearby(2, state.passive_mobs)
  )
  possible_achievements = jnp.concatenate(
    (
      possible_achievements,
      # 1 more for health (will always be 0)
      jnp.zeros([1], dtype=jnp.int32),
    )
  )

  # Add missing resource collection achievements
  has_sapling_nearby = check_block_nearby(BlockType.PLANT, visible_blocks)
  has_fountain_nearby = check_block_nearby(BlockType.FOUNTAIN, visible_blocks)

  possible_achievements = possible_achievements.at[
    Achievement.COLLECT_SAPLING.value
  ].set(has_sapling_nearby)
  possible_achievements = possible_achievements.at[Achievement.COLLECT_DRINK.value].set(
    has_fountain_nearby
  )

  # Add missing placement achievements
  possible_achievements = possible_achievements.at[Achievement.PLACE_PLANT.value].set(
    state.inventory.sapling > 0
  )

  # Add sleeping/resting related achievements
  possible_achievements = possible_achievements.at[Achievement.WAKE_UP.value].set(
    jnp.logical_or(state.is_sleeping, state.is_resting)
  )

  # Add crafting related achievements
  possible_achievements = possible_achievements.at[Achievement.MAKE_ARROW.value].set(
    jnp.logical_and(
      state.inventory.wood >= 1 if use_precondition else True, has_crafting_table
    )
  )
  possible_achievements = possible_achievements.at[Achievement.MAKE_TORCH.value].set(
    jnp.logical_and(
      jnp.logical_and(
        state.inventory.wood >= 1 if use_precondition else True,
        state.inventory.coal >= 1 if use_precondition else True,
      ),
      has_crafting_table,
    )
  )

  # Add advanced equipment crafting
  possible_achievements = possible_achievements.at[
    Achievement.MAKE_DIAMOND_PICKAXE.value
  ].set(
    jnp.logical_and(
      jnp.logical_and(
        state.inventory.wood >= 1 if use_precondition else True,
        state.inventory.diamond >= 1 if use_precondition else True,
      ),
      jnp.logical_and(has_crafting_table, state.inventory.pickaxe < 4),
    )
  )

  # Add advanced equipment crafting
  possible_achievements = possible_achievements.at[
    Achievement.MAKE_DIAMOND_SWORD.value
  ].set(
    jnp.logical_and(
      jnp.logical_and(
        state.inventory.wood >= 2 if use_precondition else True,
        state.inventory.diamond >= 1 if use_precondition else True,
      ),
      jnp.logical_and(has_crafting_table, state.inventory.sword < 4),
    )
  )

  possible_achievements = possible_achievements.at[
    Achievement.MAKE_IRON_ARMOUR.value
  ].set(
    jnp.logical_and(
      jnp.logical_and(
        state.inventory.iron >= 3 if use_precondition else True,
        state.inventory.coal >= 3 if use_precondition else True,
      ),
      jnp.logical_and(has_crafting_table, has_furnace),
    )
  )
  possible_achievements = possible_achievements.at[
    Achievement.MAKE_DIAMOND_ARMOUR.value
  ].set(
    jnp.logical_and(
      state.inventory.diamond >= 3 if use_precondition else True,
      has_crafting_table,
    )
  )

  # Add necromancer related achievements
  has_necromancer_nearby = check_block_nearby(BlockType.NECROMANCER, visible_blocks)
  has_vulnerable_necromancer_nearby = check_block_nearby(
    BlockType.NECROMANCER_VULNERABLE, visible_blocks
  )

  possible_achievements = possible_achievements.at[
    Achievement.DAMAGE_NECROMANCER.value
  ].set(has_necromancer_nearby)
  possible_achievements = possible_achievements.at[
    Achievement.DEFEAT_NECROMANCER.value
  ].set(has_vulnerable_necromancer_nearby)

  # Add bow related achievements
  possible_achievements = possible_achievements.at[Achievement.FIND_BOW.value].set(
    jnp.logical_and(
      state.inventory.bow == 0,  # Only possible if you don't already have a bow
      has_chest_nearby,
    )
  )
  possible_achievements = possible_achievements.at[Achievement.FIRE_BOW.value].set(
    jnp.logical_and(state.inventory.bow > 0, state.inventory.arrows > 0)
  )

  # Add spell related achievements
  possible_achievements = possible_achievements.at[
    Achievement.LEARN_FIREBALL.value
  ].set(jnp.logical_and(state.player_intelligence >= 2, ~state.learned_spells[0]))
  possible_achievements = possible_achievements.at[Achievement.CAST_FIREBALL.value].set(
    jnp.logical_and(state.learned_spells[0], state.player_mana >= 2)
  )
  possible_achievements = possible_achievements.at[Achievement.LEARN_ICEBALL.value].set(
    jnp.logical_and(
      state.inventory.books >= 1, jnp.logical_not(state.learned_spells[1])
    )
  )
  possible_achievements = possible_achievements.at[Achievement.CAST_ICEBALL.value].set(
    jnp.logical_and(state.learned_spells[1], state.player_mana >= 2)
  )

  # Add enchantment achievements
  has_fire_enchant_table = check_block_nearby(
    BlockType.ENCHANTMENT_TABLE_FIRE, visible_blocks
  )
  has_ice_enchant_table = check_block_nearby(
    BlockType.ENCHANTMENT_TABLE_ICE, visible_blocks
  )

  possible_achievements = possible_achievements.at[Achievement.ENCHANT_SWORD.value].set(
    jnp.logical_and(
      jnp.logical_or(has_fire_enchant_table, has_ice_enchant_table),
      state.inventory.sword > 0,
    )
  )
  possible_achievements = possible_achievements.at[
    Achievement.ENCHANT_ARMOUR.value
  ].set(
    jnp.logical_and(
      jnp.logical_or(has_fire_enchant_table, has_ice_enchant_table),
      jnp.any(state.inventory.armour > 0),
    )
  )

  ## Add level entry achievements
  # has_ladder_nearby = check_block_nearby(
  #    ItemType.LADDER_DOWN, visible_blocks)

  ## Map achievements to level numbers (based on game structure)
  ## Level 0 is surface, descending ladder on level 0 takes you to level 1 (Gnomish Mines)
  # possible_achievements = possible_achievements.at[Achievement.ENTER_GNOMISH_MINES.value].set(
  #    jnp.logical_and(
  #        has_ladder_nearby,
  #        state.player_level == 0
  #    )
  # )

  # possible_achievements = possible_achievements.at[Achievement.ENTER_DUNGEON.value].set(
  #    jnp.logical_and(
  #        has_ladder_nearby,
  #        state.player_level == 1
  #    )
  # )

  # possible_achievements = possible_achievements.at[Achievement.ENTER_SEWERS.value].set(
  #    jnp.logical_and(
  #        has_ladder_nearby,
  #        state.player_level == 2
  #    )
  # )

  # possible_achievements = possible_achievements.at[Achievement.ENTER_VAULT.value].set(
  #    jnp.logical_and(
  #        has_ladder_nearby,
  #        state.player_level == 3
  #    )
  # )

  # possible_achievements = possible_achievements.at[Achievement.ENTER_TROLL_MINES.value].set(
  #    jnp.logical_and(
  #        has_ladder_nearby,
  #        state.player_level == 4
  #    )
  # )

  # possible_achievements = possible_achievements.at[Achievement.ENTER_FIRE_REALM.value].set(
  #    jnp.logical_and(
  #        has_ladder_nearby,
  #        state.player_level == 5
  #    )
  # )

  # possible_achievements = possible_achievements.at[Achievement.ENTER_ICE_REALM.value].set(
  #    jnp.logical_and(
  #        has_ladder_nearby,
  #        state.player_level == 6
  #    )
  # )

  # possible_achievements = possible_achievements.at[Achievement.ENTER_GRAVEYARD.value].set(
  #    jnp.logical_and(
  #        has_ladder_nearby,
  #        state.player_level == 7
  #    )
  # )

  return possible_achievements


def print_possible_achievements(
  possible_achievements: jnp.ndarray, return_list: bool = False
) -> None:
  """
  Prints a readable list of currently achievable achievements.

  Args:
      possible_achievements: Binary array indicating which achievements are possible,
                           with length max_achievement + 1 where max_achievement is
                           the highest value in the Achievement enum
  """
  achievable = []
  for achievement in Achievement:
    # Check if this index is 1 in the binary array
    if possible_achievements[achievement.value] == 1:
      # Convert enum name from COLLECT_WOOD to "Collect Wood"
      achievement_name = achievement.name.replace("_", " ").title()
      achievable.append(achievement_name)
  if return_list:
    return achievable
  if achievable:
    print("\nCurrently achievable:")
    for name in sorted(achievable):
      print(f"- {name}")
  else:
    print("\nNo achievements currently achievable")


def generate_world(rng, params, static_params):
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
    player_strength=jnp.asarray(1, dtype=jnp.int32),
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
  )

  return state


def craftax_step(rng, state, action, params, static_params):
  init_achievements = state.achievements
  init_health = state.player_health

  # Interrupt action if sleeping or resting
  action = jax.lax.select(state.is_sleeping, Action.NOOP.value, action)
  action = jax.lax.select(state.is_resting, Action.NOOP.value, action)

  # Change floor
  state = change_floor(state, action, params, static_params)

  # Crafting
  state = do_crafting(state, action)

  # Interact (mining, melee attacking, eating plants, drinking water)
  rng, _rng = jax.random.split(rng)
  state = do_action(_rng, state, action, static_params)

  # Placing
  state = place_block(state, action, static_params)

  # Shooting
  state = shoot_projectile(state, action, static_params)

  # Casting
  state = cast_spell(state, action, static_params)

  # Potions
  state = drink_potion(state, action)

  # Read
  rng, _rng = jax.random.split(rng)
  state = read_book(_rng, state, action)

  # Enchant
  rng, _rng = jax.random.split(rng)
  state = enchant(_rng, state, action)

  # Boss
  state = boss_logic(state, static_params)

  # Attributes
  state = level_up_attributes(state, action, params)

  # Movement
  state = move_player(state, action, params)

  # Mobs
  rng, _rng = jax.random.split(rng)
  state = update_mobs(_rng, state, params, static_params)

  rng, _rng = jax.random.split(rng)
  state = spawn_mobs(state, _rng, params, static_params)

  # Plants
  state = update_plants(state, static_params)

  # Intrinsics
  state = update_player_intrinsics(state, action, static_params)

  # Cap inv
  state = clip_inventory_and_intrinsics(state, params)

  # Inventory achievements
  state = calculate_inventory_achievements(state)

  # NOTE: MAIN CHANGE is just exposing achievements
  # normally, just summed and provided to agent
  achievements = jnp.concatenate(
    (
      (state.achievements.astype(int) - init_achievements.astype(int)).astype(
        jnp.float32
      ),
      jnp.expand_dims(state.player_health - init_health, 0).astype(jnp.float32),
    )
  )
  achievement_coefficients = jnp.concatenate((ACHIEVEMENT_REWARD_MAP, jnp.array([0.1])))
  reward = (achievements * achievement_coefficients).sum()

  rng, _rng = jax.random.split(rng)

  state = state.replace(
    timestep=state.timestep + 1,
    light_level=calculate_light_level(state.timestep + 1, params),
    state_rng=_rng,
  )

  return state, reward, achievements, achievement_coefficients


class CraftaxSymbolicEnvNoAutoReset(EnvironmentNoAutoReset):
  """
  Made following changes:
  1. sample world set from pre-defined set of seeds
  2. structured obs with {image, achievable, achieved, task-vector}
  3. option to have task-vector double with zeros (to accomodate successor features)
  """

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

  def step_env(
    self, rng: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
  ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
    state, reward, achievements, achievement_coefficients = craftax_step(
      rng, state, action, params, self.static_env_params
    )

    done = self.is_terminal(state, params)
    info = log_achievements_to_info(state, done)
    info["discount"] = self.discount(state, params)

    return (
      lax.stop_gradient(
        self.get_obs(
          state=state,
          achievements=achievements,
          achievement_coefficients=achievement_coefficients,
          params=params,
          previous_action=action,
        )
      ),
      lax.stop_gradient(state),
      reward,
      done,
      info,
    )

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
    state = generate_world(world_rng, params, self.static_env_params)

    obs = self.get_obs(
      state=state,
      achievements=dummy_achievements,
      achievement_coefficients=dummy_achievements,
      params=params,
      previous_action=jnp.zeros((), dtype=jnp.int32),  # scalar
    )
    return obs, state

  def get_obs(
    self,
    state: EnvState,
    achievements: chex.Array,
    achievement_coefficients: chex.Array,
    params: EnvParams,
    previous_action: Optional[chex.Array] = None,
  ):
    del params
    achievable = get_possible_achievements(
      state=state, use_precondition=self.static_env_params.use_precondition
    )
    assert achievable.shape == achievements.shape, (
      "these should have the exact same shape"
    )

    task_w = jnp.concatenate(
      (
        achievement_coefficients,
        jnp.zeros(achievable.shape, dtype=achievement_coefficients.dtype),
      )
    )

    return Observation(
      image=render_craftax_symbolic(state),
      achievements=achievements,
      achievable=achievable,
      task_w=task_w,
      previous_action=previous_action,
    )

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
