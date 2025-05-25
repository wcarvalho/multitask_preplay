import jax
from functools import partial

from .craftax_fullmap_constants import *
from craftax.craftax.craftax_state import EnvState
from craftax.craftax.util.game_logic_utils import is_boss_vulnerable


@partial(
  jax.jit,
  static_argnums=(
    1,
    2,
    3,
    4,
  ),
)
def render_craftax_pixels(
  state,
  block_pixel_size,
  do_night_noise=True,
  show_inventory=False,
  show_agent=True,
  show_center_agent=False,
):
  textures = TEXTURES[block_pixel_size]
  obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)

  # RENDER MAP
  # Get view of map
  map = state.map[state.player_level]
  padded_grid = jnp.pad(
    map,
    (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
    constant_values=BlockType.OUT_OF_BOUNDS.value,
  )

  # Calculate view bounds to keep player within map edges
  half_obs = obs_dim_array // 2
  player_pos = state.player_position

  # Calculate top-left corner with bounds checking
  tl_corner = jnp.clip(
    player_pos - half_obs + MAX_OBS_DIM + 2,
    MAX_OBS_DIM + 2,  # Minimum allowed position after padding
    MAX_OBS_DIM + 2 + jnp.array([48, 48]) - obs_dim_array,  # Maximum allowed position
  )

  map_view = jax.lax.dynamic_slice(padded_grid, tl_corner, OBS_DIM)

  # Boss
  boss_block = jax.lax.select(
    is_boss_vulnerable(state),
    BlockType.NECROMANCER_VULNERABLE.value,
    BlockType.NECROMANCER.value,
  )

  map_view_boss = map_view == BlockType.NECROMANCER.value
  map_view = map_view_boss * boss_block + (1 - map_view_boss) * map_view

  # Render map tiles
  map_pixels_indexes = jnp.repeat(
    jnp.repeat(map_view, repeats=block_pixel_size, axis=0),
    repeats=block_pixel_size,
    axis=1,
  )
  map_pixels_indexes = jnp.expand_dims(map_pixels_indexes, axis=-1)
  map_pixels_indexes = jnp.repeat(map_pixels_indexes, repeats=3, axis=2)

  map_pixels = jnp.zeros(
    (OBS_DIM[0] * block_pixel_size, OBS_DIM[1] * block_pixel_size, 3),
    dtype=jnp.float32,
  )

  def _add_block_type_to_pixels(pixels, block_index):
    return (
      pixels
      + textures["full_map_block_textures"][block_index]
      * (map_pixels_indexes == block_index),
      None,
    )

  map_pixels, _ = jax.lax.scan(
    _add_block_type_to_pixels, map_pixels, jnp.arange(len(BlockType))
  )

  # Items
  padded_item_map = jnp.pad(
    state.item_map[state.player_level],
    (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
    constant_values=ItemType.NONE.value,
  )

  item_map_view = jax.lax.dynamic_slice(padded_item_map, tl_corner, OBS_DIM)

  # Insert blocked ladders
  is_ladder_down_open = (
    state.monsters_killed[state.player_level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL
  )
  ladder_down_item = jax.lax.select(
    is_ladder_down_open,
    ItemType.LADDER_DOWN.value,
    ItemType.LADDER_DOWN_BLOCKED.value,
  )

  item_map_view_is_ladder_down = item_map_view == ItemType.LADDER_DOWN.value
  item_map_view = (
    item_map_view_is_ladder_down * ladder_down_item
    + (1 - item_map_view_is_ladder_down) * item_map_view
  )

  map_pixels_item_indexes = jnp.repeat(
    jnp.repeat(item_map_view, repeats=block_pixel_size, axis=0),
    repeats=block_pixel_size,
    axis=1,
  )
  map_pixels_item_indexes = jnp.expand_dims(map_pixels_item_indexes, axis=-1)
  map_pixels_item_indexes = jnp.repeat(map_pixels_item_indexes, repeats=3, axis=2)

  def _add_item_type_to_pixels(pixels, item_index):
    full_map_texture = textures["full_map_item_textures"][item_index]
    mask = map_pixels_item_indexes == item_index

    pixels = pixels * (1 - full_map_texture[:, :, 3] * mask[:, :, 0])[:, :, None]
    pixels = (
      pixels + full_map_texture[:, :, :3] * mask * full_map_texture[:, :, 3][:, :, None]
    )

    return pixels, None

  map_pixels, _ = jax.lax.scan(
    _add_item_type_to_pixels, map_pixels, jnp.arange(1, len(ItemType))
  )

  if show_center_agent:
    # Render player
    player_texture_index = jax.lax.select(
      state.is_sleeping, 4, state.player_direction - 1
    )
    map_pixels = (
      map_pixels
      * (1 - textures["full_map_player_textures_alpha"][player_texture_index])
      + textures["full_map_player_textures"][player_texture_index]
      * textures["full_map_player_textures_alpha"][player_texture_index]
    )

  # Render mobs
  # Zombies

  def _add_mob_to_pixels(carry, mob_index):
    pixels, mobs, texture_name, alpha_texture_name = carry
    local_position = (
      mobs.position[state.player_level, mob_index]
      - state.player_position
      + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
    )
    on_screen = jnp.logical_and(
      local_position >= 0, local_position < obs_dim_array
    ).all()
    on_screen *= mobs.mask[state.player_level, mob_index]

    melee_mob_texture = texture_name[mobs.type_id[state.player_level, mob_index]]
    melee_mob_texture_alpha = alpha_texture_name[
      mobs.type_id[state.player_level, mob_index]
    ]

    melee_mob_texture = melee_mob_texture * on_screen

    melee_mob_texture_with_background = 1 - melee_mob_texture_alpha * on_screen

    melee_mob_texture_with_background = (
      melee_mob_texture_with_background
      * jax.lax.dynamic_slice(
        pixels,
        (
          local_position[0] * block_pixel_size,
          local_position[1] * block_pixel_size,
          0,
        ),
        (block_pixel_size, block_pixel_size, 3),
      )
    )

    melee_mob_texture_with_background = (
      melee_mob_texture_with_background + melee_mob_texture * melee_mob_texture_alpha
    )

    pixels = jax.lax.dynamic_update_slice(
      pixels,
      melee_mob_texture_with_background,
      (
        local_position[0] * block_pixel_size,
        local_position[1] * block_pixel_size,
        0,
      ),
    )

    return (pixels, mobs, texture_name, alpha_texture_name), None

  (map_pixels, _, _, _), _ = jax.lax.scan(
    _add_mob_to_pixels,
    (
      map_pixels,
      state.melee_mobs,
      textures["melee_mob_textures"],
      textures["melee_mob_texture_alphas"],
    ),
    jnp.arange(state.melee_mobs.mask.shape[1]),
  )

  (map_pixels, _, _, _), _ = jax.lax.scan(
    _add_mob_to_pixels,
    (
      map_pixels,
      state.passive_mobs,
      textures["passive_mob_textures"],
      textures["passive_mob_texture_alphas"],
    ),
    jnp.arange(state.passive_mobs.mask.shape[1]),
  )

  (map_pixels, _, _, _), _ = jax.lax.scan(
    _add_mob_to_pixels,
    (
      map_pixels,
      state.ranged_mobs,
      textures["ranged_mob_textures"],
      textures["ranged_mob_texture_alphas"],
    ),
    jnp.arange(state.ranged_mobs.mask.shape[1]),
  )

  def _add_projectile_to_pixels(carry, projectile_index):
    pixels, projectiles, projectile_directions = carry
    local_position = (
      projectiles.position[state.player_level, projectile_index]
      - state.player_position
      + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
    )
    on_screen = jnp.logical_and(
      local_position >= 0, local_position < obs_dim_array
    ).all()
    on_screen *= projectiles.mask[state.player_level, projectile_index]

    projectile_texture = textures["projectile_textures"][
      projectiles.type_id[state.player_level, projectile_index]
    ]
    projectile_texture_alpha = textures["projectile_texture_alphas"][
      projectiles.type_id[state.player_level, projectile_index]
    ]

    flipped_projectile_texture = jnp.flip(projectile_texture, axis=0)
    flipped_projectile_texture_alpha = jnp.flip(projectile_texture_alpha, axis=0)
    flip_projectile = jnp.logical_or(
      projectile_directions[state.player_level, projectile_index, 0] > 0,
      projectile_directions[state.player_level, projectile_index, 1] > 0,
    )

    projectile_texture = jax.lax.select(
      flip_projectile,
      flipped_projectile_texture,
      projectile_texture,
    )
    projectile_texture_alpha = jax.lax.select(
      flip_projectile,
      flipped_projectile_texture_alpha,
      projectile_texture_alpha,
    )

    transposed_projectile_texture = jnp.transpose(projectile_texture, (1, 0, 2))
    transposed_projectile_texture_alpha = jnp.transpose(
      projectile_texture_alpha, (1, 0, 2)
    )

    projectile_texture = jax.lax.select(
      projectile_directions[state.player_level, projectile_index, 1] != 0,
      transposed_projectile_texture,
      projectile_texture,
    )
    projectile_texture_alpha = jax.lax.select(
      projectile_directions[state.player_level, projectile_index, 1] != 0,
      transposed_projectile_texture_alpha,
      projectile_texture_alpha,
    )

    projectile_texture = projectile_texture * on_screen
    projectile_texture_with_background = 1 - projectile_texture_alpha * on_screen

    projectile_texture_with_background = (
      projectile_texture_with_background
      * jax.lax.dynamic_slice(
        pixels,
        (
          local_position[0] * block_pixel_size,
          local_position[1] * block_pixel_size,
          0,
        ),
        (block_pixel_size, block_pixel_size, 3),
      )
    )

    projectile_texture_with_background = (
      projectile_texture_with_background + projectile_texture * projectile_texture_alpha
    )

    pixels = jax.lax.dynamic_update_slice(
      pixels,
      projectile_texture_with_background,
      (
        local_position[0] * block_pixel_size,
        local_position[1] * block_pixel_size,
        0,
      ),
    )

    return (pixels, projectiles, projectile_directions), None

  (map_pixels, _, _), _ = jax.lax.scan(
    _add_projectile_to_pixels,
    (map_pixels, state.mob_projectiles, state.mob_projectile_directions),
    jnp.arange(state.mob_projectiles.mask.shape[1]),
  )

  (map_pixels, _, _), _ = jax.lax.scan(
    _add_projectile_to_pixels,
    (map_pixels, state.player_projectiles, state.player_projectile_directions),
    jnp.arange(state.player_projectiles.mask.shape[1]),
  )

  # Apply darkness (underground)
  light_map = state.light_map[state.player_level]
  padded_light_map = jnp.pad(
    light_map,
    (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
    constant_values=False,
  )

  light_map_view = jax.lax.dynamic_slice(padded_light_map, tl_corner, OBS_DIM)
  light_map_pixels = light_map_view.repeat(block_pixel_size, axis=0).repeat(
    block_pixel_size, axis=1
  )

  map_pixels = (light_map_pixels)[:, :, None] * map_pixels

  # Apply night
  night_pixels = textures["night_texture"]
  daylight = state.light_level
  daylight = jax.lax.select(state.player_level == 0, daylight, 1.0)

  if do_night_noise:
    night_noise = jax.random.uniform(state.state_rng, night_pixels.shape[:2]) * 95 + 32
    night_noise = jnp.expand_dims(night_noise, axis=-1).repeat(3, axis=-1)

    night_intensity = 2 * (0.5 - daylight)
    night_intensity = jnp.maximum(night_intensity, 0.0)
    night_mask = textures["night_noise_intensity_texture"] * night_intensity
    night = (1.0 - night_mask) * map_pixels + night_mask * night_noise

    night = night_pixels * 0.5 + 0.5 * night
    map_pixels = daylight * map_pixels + (1 - daylight) * night
  else:
    night_noise = jnp.ones(night_pixels.shape[:2]) * 64
    night_noise = jnp.expand_dims(night_noise, axis=-1).repeat(3, axis=-1)

    night_intensity = 2 * (0.5 - daylight)
    night_intensity = jnp.maximum(night_intensity, 0.0)
    night_mask = (
      jnp.ones_like(textures["night_noise_intensity_texture"]) * night_intensity * 0.5
    )
    night = (1.0 - night_mask) * map_pixels + night_mask * night_noise

    night = night_pixels * 0.5 + 0.5 * night
    map_pixels = daylight * map_pixels + (1 - daylight) * night
    # map_pixels = daylight * map_pixels
    # night_noise = jnp.ones(night_pixels.shape[:2]) * 64

  # Apply sleep
  sleep_pixels = jnp.zeros_like(map_pixels)
  sleep_level = 1.0 - state.is_sleeping * 0.5
  map_pixels = sleep_level * map_pixels + (1 - sleep_level) * sleep_pixels

  # Render mob map
  # mob_map_pixels = (
  #     jnp.array([[[128, 0, 0]]]).repeat(OBS_DIM[0], axis=0).repeat(OBS_DIM[1], axis=1)
  # )
  # padded_mob_map = jnp.pad(
  #     state.mob_map[state.player_level], MAX_OBS_DIM + 2, constant_values=False
  # )
  # mob_map_view = jax.lax.dynamic_slice(padded_mob_map, tl_corner, OBS_DIM)
  # mob_map_pixels = mob_map_pixels * jnp.expand_dims(mob_map_view, axis=-1)
  # mob_map_pixels = mob_map_pixels.repeat(block_pixel_size, axis=0).repeat(
  #     block_pixel_size, axis=1
  # )
  # map_pixels = map_pixels + mob_map_pixels

  # RENDER INVENTORY
  inv_pixel_left_space = (block_pixel_size - int(0.8 * block_pixel_size)) // 2
  inv_pixel_right_space = (
    block_pixel_size - int(0.8 * block_pixel_size) - inv_pixel_left_space
  )

  inv_pixels = jnp.zeros(
    (INVENTORY_OBS_HEIGHT * block_pixel_size, OBS_DIM[1] * block_pixel_size, 3),
    dtype=jnp.float32,
  )

  number_size = int(block_pixel_size * 0.4)
  number_offset = block_pixel_size - number_size
  number_double_offset = block_pixel_size - 2 * number_size

  def _render_digit(pixels, number, x, y):
    pixels = pixels.at[
      y * block_pixel_size + number_offset : (y + 1) * block_pixel_size,
      x * block_pixel_size + number_offset : (x + 1) * block_pixel_size,
    ].mul(1 - textures["number_textures_alpha"][number])

    pixels = pixels.at[
      y * block_pixel_size + number_offset : (y + 1) * block_pixel_size,
      x * block_pixel_size + number_offset : (x + 1) * block_pixel_size,
    ].add(textures["number_textures"][number])

    return pixels

  def _render_two_digit_number(pixels, number, x, y):
    tens = number // 10
    ones = number % 10

    ones_textures = jax.lax.select(
      number == 0,
      textures["number_textures"],
      textures["number_textures_with_zero"],
    )

    ones_textures_alpha = jax.lax.select(
      number == 0,
      textures["number_textures_alpha"],
      textures["number_textures_alpha_with_zero"],
    )

    pixels = pixels.at[
      y * block_pixel_size + number_offset : (y + 1) * block_pixel_size,
      x * block_pixel_size + number_offset : (x + 1) * block_pixel_size,
    ].mul(1 - ones_textures_alpha[ones])

    pixels = pixels.at[
      y * block_pixel_size + number_offset : (y + 1) * block_pixel_size,
      x * block_pixel_size + number_offset : (x + 1) * block_pixel_size,
    ].add(ones_textures[ones])

    pixels = pixels.at[
      y * block_pixel_size + number_offset : (y + 1) * block_pixel_size,
      x * block_pixel_size + number_double_offset : x * block_pixel_size
      + number_offset,
    ].mul(1 - textures["number_textures_alpha"][tens])

    pixels = pixels.at[
      y * block_pixel_size + number_offset : (y + 1) * block_pixel_size,
      x * block_pixel_size + number_double_offset : x * block_pixel_size
      + number_offset,
    ].add(textures["number_textures"][tens])

    return pixels

  def _render_icon(pixels, texture, x, y):
    return pixels.at[
      block_pixel_size * y + inv_pixel_left_space : block_pixel_size * (y + 1)
      - inv_pixel_right_space,
      block_pixel_size * x + inv_pixel_left_space : block_pixel_size * (x + 1)
      - inv_pixel_right_space,
    ].set(texture)

  def _render_icon_with_alpha(pixels, texture, x, y):
    existing_slice = pixels[
      block_pixel_size * y + inv_pixel_left_space : block_pixel_size * (y + 1)
      - inv_pixel_right_space,
      block_pixel_size * x + inv_pixel_left_space : block_pixel_size * (x + 1)
      - inv_pixel_right_space,
    ]

    new_slice = (
      existing_slice * (1 - texture[:, :, 3][:, :, None])
      + texture[:, :, :3] * texture[:, :, 3][:, :, None]
    )

    return pixels.at[
      block_pixel_size * y + inv_pixel_left_space : block_pixel_size * (y + 1)
      - inv_pixel_right_space,
      block_pixel_size * x + inv_pixel_left_space : block_pixel_size * (x + 1)
      - inv_pixel_right_space,
    ].set(new_slice)

  # Render player stats
  player_health = jnp.maximum(jnp.floor(state.player_health), 1).astype(int)
  health_texture = jax.lax.select(
    player_health > 0,
    textures["health_texture"],
    textures["smaller_empty_texture"],
  )
  inv_pixels = _render_icon(inv_pixels, health_texture, 0, 0)
  inv_pixels = _render_two_digit_number(inv_pixels, player_health, 0, 0)

  hunger_texture = jax.lax.select(
    state.player_food > 0,
    textures["hunger_texture"],
    textures["smaller_empty_texture"],
  )
  inv_pixels = _render_icon(inv_pixels, hunger_texture, 1, 0)
  inv_pixels = _render_two_digit_number(inv_pixels, state.player_food, 1, 0)

  thirst_texture = jax.lax.select(
    state.player_drink > 0,
    textures["thirst_texture"],
    textures["smaller_empty_texture"],
  )
  inv_pixels = _render_icon(inv_pixels, thirst_texture, 2, 0)
  inv_pixels = _render_two_digit_number(inv_pixels, state.player_drink, 2, 0)

  energy_texture = jax.lax.select(
    state.player_energy > 0,
    textures["energy_texture"],
    textures["smaller_empty_texture"],
  )
  inv_pixels = _render_icon(inv_pixels, energy_texture, 3, 0)
  inv_pixels = _render_two_digit_number(inv_pixels, state.player_energy, 3, 0)

  mana_texture = jax.lax.select(
    state.player_mana > 0,
    textures["mana_texture"],
    textures["smaller_empty_texture"],
  )
  inv_pixels = _render_icon(inv_pixels, mana_texture, 4, 0)
  inv_pixels = _render_two_digit_number(inv_pixels, state.player_mana, 4, 0)

  # Render inventory

  inv_wood_texture = jax.lax.select(
    state.inventory.wood > 0,
    textures["smaller_block_textures"][BlockType.WOOD.value],
    textures["smaller_empty_texture"],
  )
  inv_pixels = _render_icon(inv_pixels, inv_wood_texture, 0, 2)
  inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.wood, 0, 2)

  inv_stone_texture = jax.lax.select(
    state.inventory.stone > 0,
    textures["smaller_block_textures"][BlockType.STONE.value],
    textures["smaller_empty_texture"],
  )
  inv_pixels = _render_icon(inv_pixels, inv_stone_texture, 1, 2)
  inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.stone, 1, 2)

  inv_coal_texture = jax.lax.select(
    state.inventory.coal > 0,
    textures["smaller_block_textures"][BlockType.COAL.value],
    textures["smaller_empty_texture"],
  )
  inv_pixels = _render_icon(inv_pixels, inv_coal_texture, 0, 1)
  inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.coal, 0, 1)

  inv_iron_texture = jax.lax.select(
    state.inventory.iron > 0,
    textures["smaller_block_textures"][BlockType.IRON.value],
    textures["smaller_empty_texture"],
  )
  inv_pixels = _render_icon(inv_pixels, inv_iron_texture, 1, 1)
  inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.iron, 1, 1)

  inv_diamond_texture = jax.lax.select(
    state.inventory.diamond > 0,
    textures["smaller_block_textures"][BlockType.DIAMOND.value],
    textures["smaller_empty_texture"],
  )
  inv_pixels = _render_icon(inv_pixels, inv_diamond_texture, 2, 1)
  inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.diamond, 2, 1)

  inv_sapphire_texture = jax.lax.select(
    state.inventory.sapphire > 0,
    textures["smaller_block_textures"][BlockType.SAPPHIRE.value],
    textures["smaller_empty_texture"],
  )
  inv_pixels = _render_icon(inv_pixels, inv_sapphire_texture, 3, 1)
  inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.sapphire, 3, 1)

  inv_ruby_texture = jax.lax.select(
    state.inventory.ruby > 0,
    textures["smaller_block_textures"][BlockType.RUBY.value],
    textures["smaller_empty_texture"],
  )
  inv_pixels = _render_icon(inv_pixels, inv_ruby_texture, 4, 1)
  inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.ruby, 4, 1)

  inv_sapling_texture = jax.lax.select(
    state.inventory.sapling > 0,
    textures["sapling_texture"],
    textures["smaller_empty_texture"],
  )
  inv_pixels = _render_icon(inv_pixels, inv_sapling_texture, 5, 1)
  inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.sapling, 5, 1)

  # Render tools
  # Pickaxe
  pickaxe_texture = textures["pickaxe_textures"][state.inventory.pickaxe]
  inv_pixels = _render_icon(inv_pixels, pickaxe_texture, 8, 2)

  # Sword
  sword_texture = textures["sword_textures"][state.inventory.sword]
  inv_pixels = _render_icon(inv_pixels, sword_texture, 8, 1)

  # Bow and arrows
  bow_texture = textures["bow_textures"][state.inventory.bow]
  inv_pixels = _render_icon(inv_pixels, bow_texture, 6, 1)

  arrow_texture = jax.lax.select(
    state.inventory.arrows > 0,
    textures["player_projectile_textures"][0],
    textures["smaller_empty_texture"],
  )
  inv_pixels = _render_icon(inv_pixels, arrow_texture, 6, 2)
  inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.arrows, 6, 2)

  # Armour
  for i in range(4):
    armour_texture = textures["armour_textures"][state.inventory.armour[i], i]
    inv_pixels = _render_icon(inv_pixels, armour_texture, 7, i)

  # Torch
  torch_texture = jax.lax.select(
    state.inventory.torches > 0,
    textures["torch_inv_texture"],
    textures["smaller_empty_texture"],
  )
  inv_pixels = _render_icon(inv_pixels, torch_texture, 2, 2)
  inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.torches, 2, 2)

  # Potions
  potion_names = ["red", "green", "blue", "pink", "cyan", "yellow"]
  for potion_index, potion_name in enumerate(potion_names):
    potion_texture = jax.lax.select(
      state.inventory.potions[potion_index] > 0,
      textures["potion_textures"][potion_index],
      textures["smaller_empty_texture"],
    )
    inv_pixels = _render_icon(inv_pixels, potion_texture, potion_index, 3)
    inv_pixels = _render_two_digit_number(
      inv_pixels, state.inventory.potions[potion_index], potion_index, 3
    )

  # Books
  book_texture = jax.lax.select(
    state.inventory.books > 0,
    textures["book_texture"],
    textures["smaller_empty_texture"],
  )
  inv_pixels = _render_icon(inv_pixels, book_texture, 3, 2)
  inv_pixels = _render_two_digit_number(inv_pixels, state.inventory.books, 3, 2)

  # Learned spells
  fireball_texture = jax.lax.select(
    state.learned_spells[0],
    textures["fireball_inv_texture"],
    textures["smaller_empty_texture"],
  )
  inv_pixels = _render_icon(inv_pixels, fireball_texture, 4, 2)

  iceball_texture = jax.lax.select(
    state.learned_spells[1],
    textures["iceball_inv_texture"],
    textures["smaller_empty_texture"],
  )
  inv_pixels = _render_icon(inv_pixels, iceball_texture, 5, 2)

  # Enchantments
  sword_enchantment_texture = textures["sword_enchantment_textures"][
    state.sword_enchantment
  ]
  inv_pixels = _render_icon_with_alpha(inv_pixels, sword_enchantment_texture, 8, 1)

  arrow_enchantment_level = state.bow_enchantment * (state.inventory.arrows > 0)
  arrow_enchantment_texture = textures["arrow_enchantment_textures"][
    arrow_enchantment_level
  ]
  inv_pixels = _render_icon_with_alpha(inv_pixels, arrow_enchantment_texture, 6, 2)

  for i in range(4):
    armour_enchantment_texture = textures["armour_enchantment_textures"][
      state.armour_enchantments[i], i
    ]
    inv_pixels = _render_icon_with_alpha(inv_pixels, armour_enchantment_texture, 7, i)

  # Dungeon level
  inv_pixels = _render_digit(inv_pixels, state.player_level, 6, 0)

  # Attributes
  xp_texture = jax.lax.select(
    state.player_xp > 0, textures["xp_texture"], textures["smaller_empty_texture"]
  )
  inv_pixels = _render_icon(inv_pixels, xp_texture, 9, 0)
  inv_pixels = _render_digit(inv_pixels, state.player_xp, 9, 0)

  inv_pixels = _render_icon(inv_pixels, textures["dex_texture"], 9, 1)
  inv_pixels = _render_digit(inv_pixels, state.player_dexterity, 9, 1)

  inv_pixels = _render_icon(inv_pixels, textures["str_texture"], 9, 2)
  inv_pixels = _render_digit(inv_pixels, state.player_strength, 9, 2)

  inv_pixels = _render_icon(inv_pixels, textures["int_texture"], 9, 3)
  inv_pixels = _render_digit(inv_pixels, state.player_intelligence, 9, 3)

  # Render player
  if show_agent:
    # Create white square with black border for player
    n = 1 * block_pixel_size
    player_square = jnp.ones((n, n, 3))
    player_square = player_square.at[:, :, 2].set(
      0
    )  # Set blue channel to 0 to make it yellow
    # Calculate player position in pixels, centered on the original block
    player_pixel_pos = (
      state.player_position
      + 1
      - tl_corner
      + jnp.array([MAX_OBS_DIM + 2, MAX_OBS_DIM + 2])
    ) * block_pixel_size - block_pixel_size  # Offset to center the larger square

    # Update the map pixels with the player square
    map_pixels = map_pixels.at[
      player_pixel_pos[0] : player_pixel_pos[0] + n,
      player_pixel_pos[1] : player_pixel_pos[1] + n,
      :,
    ].set(player_square)

  # Combine map and inventory
  if show_inventory:
    pixels = jnp.concatenate([map_pixels, inv_pixels], axis=0)
  else:
    pixels = map_pixels

  # # Downscale by 2
  # pixels = pixels[::downscale, ::downscale]
  pixels = pixels[block_pixel_size:, block_pixel_size:]
  return pixels
