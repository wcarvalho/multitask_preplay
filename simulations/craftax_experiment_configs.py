import math
from typing import List, Tuple
from flax import struct
import numpy as np
import os.path
import matplotlib.pyplot as plt
from craftax.craftax.renderer import render_craftax_pixels as render_partial

try:
  from craftax_fullmap_renderer import render_craftax_pixels as render_full
except ModuleNotFoundError:
  pass
from craftax.craftax.constants import (
  BlockType,
  Achievement,
  TEXTURES,
  BLOCK_PIXEL_SIZE_IMG,
  BLOCK_PIXEL_SIZE_HUMAN,
)
import jax
import craftax_utils

BUCKET_NAME = "craftax-human-dyna"


class BlockConfig(struct.PyTreeNode):
  """Configuration for a single experimental block"""

  world_seed: int
  start_train_positions: List[Tuple[int, int]]
  start_eval_positions: List[Tuple[int, int]]
  train_objects: List[int] = None
  test_objects: List[int] = None
  start_eval2_positions: List[Tuple[int, int]] = None
  train_object_location: Tuple[int, int] = None
  test_object_location: Tuple[int, int] = None
  train_distractor_object_location: Tuple[int, int] = None
  # eval_goal_location: Tuple[int, int] = (-1, -1)
  type: str = ""


POSSIBLE_GOALS = (
  Achievement.COLLECT_SAPPHIRE.value,
  Achievement.COLLECT_RUBY.value,
  Achievement.COLLECT_DIAMOND.value,
  # Achievement.OPEN_CHEST.value,
)
GOAL_TO_BLOCK = {
  Achievement.COLLECT_SAPPHIRE.value: BlockType.SAPPHIRE.value,
  Achievement.COLLECT_RUBY.value: BlockType.RUBY.value,
  Achievement.COLLECT_DIAMOND.value: BlockType.DIAMOND.value,
  Achievement.OPEN_CHEST.value: BlockType.CHEST.value,
}
BLOCK_TO_GOAL = {v: k for k, v in GOAL_TO_BLOCK.items()}
BLOCK_TO_IDX = {
  BlockType.SAPPHIRE.value: 0,
  BlockType.RUBY.value: 1,
  BlockType.DIAMOND.value: 2,
  BlockType.CHEST.value: 3,
}
POSSIBLE_BLOCKS = [GOAL_TO_BLOCK[i] for i in POSSIBLE_GOALS]
POSSIBLE_BLOCKS_MOD = list(POSSIBLE_BLOCKS + POSSIBLE_BLOCKS + POSSIBLE_BLOCKS)


PRACTICE_BLOCK_CONFIG = BlockConfig(
  world_seed=101,
  start_train_positions=[
    (32, 30),
    (17, 28),
    (29, 15),
  ],
  start_eval_positions=[(28, 25)],
  train_object_location=(28, 30),
  test_object_location=(26, 28),
  train_distractor_object_location=(30, 20),
  train_objects=[BlockType.SAPPHIRE.value, BlockType.DIAMOND.value],
  test_objects=[BlockType.RUBY.value],
  type="practice",
)

########################################################
# Paths manipulation configs
########################################################

PATHS_CONFIGS = [
  BlockConfig(
    world_seed=3,
    start_train_positions=[
      (32, 30),
      (17, 28),
      (29, 15),
    ],
    start_eval_positions=[(28, 25)],
    train_object_location=(39, 44),
    train_distractor_object_location=(35, 13),
    test_object_location=(39, 46),
    type="paths",
  ),
  BlockConfig(
    world_seed=15,
    start_train_positions=[
      (46, 8),
      (45, 32),
      (35, 25),
    ],
    start_eval_positions=[(24, 24)],
    train_object_location=(40, 44),
    test_object_location=(38, 46),
    train_distractor_object_location=(37, 13),
    # train_objects=TRAIN_OBJECTS,
    # test_objects=TEST_OBJECTS,
    type="paths",
  ),
  BlockConfig(
    world_seed=20,
    start_train_positions=[
      (25, 22),
      (15, 15),
      (10, 21),
    ],
    start_eval_positions=[(18, 30)],
    train_object_location=(42, 30),
    train_distractor_object_location=(13, 13),
    test_object_location=(40, 32),
    # train_objects=TRAIN_OBJECTS,
    # test_objects=TEST_OBJECTS,
    type="paths",
  ),
  BlockConfig(
    world_seed=95,
    start_train_positions=[
      (9, 3),
      (3, 19),
      (2, 27),
    ],
    start_eval_positions=[(7, 16)],
    train_object_location=(3, 36),
    train_distractor_object_location=(17, 13),
    test_object_location=(6, 34),
    # train_objects=TRAIN_OBJECTS,
    # test_objects=TEST_OBJECTS,
    type="paths",
  ),
]

for i in range(len(PATHS_CONFIGS)):
  PATHS_CONFIGS[i] = PATHS_CONFIGS[i].replace(
    train_objects=POSSIBLE_BLOCKS_MOD[i : i + 2],
    test_objects=POSSIBLE_BLOCKS_MOD[i + 2 : i + 3],
  )


########################################################
# Juncture manipulation configs
########################################################

JUNCTURE_CONFIGS = [
  BlockConfig(
    world_seed=1,
    start_train_positions=[
      (11, 32),
      (25, 46),
      (14, 46),
    ],
    start_eval_positions=[(23, 40)],
    start_eval2_positions=[(19, 26)],
    train_object_location=(14, 24),
    test_object_location=(16, 26),
    type="juncture",
  ),
  BlockConfig(
    world_seed=2,
    start_train_positions=[
      (23, 18),
      (17, 34),
      (6, 35),
      # (2, 41),
      # (3, 40),
      # (21, 35),
      # (15, 16),
    ],
    start_eval_positions=[(15, 30)],
    start_eval2_positions=[(34, 25)],
    train_object_location=(37, 28),
    test_object_location=(34, 27),
    # eval_goal_location=(34, 27),
    type="juncture",
  ),
  BlockConfig(
    world_seed=16,
    start_train_positions=[
      (26, 20),
      (25, 33),
      (15, 26),
      # (10, 32),
      # (12, 31),
      # (38, 27),
      # (23, 7),
    ],
    start_eval_positions=[(24, 21)],
    start_eval2_positions=[(15, 18)],
    train_object_location=(14, 21),
    test_object_location=(18, 18),
    # eval_goal_location=(18, 18),
    type="juncture",
  ),
  BlockConfig(
    world_seed=21,
    start_train_positions=[
      (2, 38),
      (27, 37),
      (9, 40),
    ],
    start_eval_positions=[(5, 43)],
    start_eval2_positions=[(12, 33)],
    train_object_location=(15, 38),
    test_object_location=(12, 36),
    type="juncture",
  ),
]

for i in range(len(JUNCTURE_CONFIGS)):
  JUNCTURE_CONFIGS[i] = JUNCTURE_CONFIGS[i].replace(
    train_objects=POSSIBLE_BLOCKS_MOD[i : i + 1],
    test_objects=POSSIBLE_BLOCKS_MOD[i + 1 : i + 2],
  )


def get_fullmap_image(world_seed, type="paths"):
  # Use same cache directory as defined in craftax_utils
  subdir = "single" if type == "paths" else "juncture"
  cache_dir = os.path.join("craftax_cache", subdir)
  image_path = os.path.join(cache_dir, f"world_{world_seed}_paths.png")

  if not os.path.exists(image_path):
    raise FileNotFoundError(
      f"No cached map found for world seed {world_seed} at {image_path}"
    )

  # Read image using matplotlib to maintain consistency with how images were saved
  image = plt.imread(image_path)

  # Convert to uint8 if needed
  if image.dtype == np.float32:
    image = (image * 255).astype(np.uint8)

  # Ensure image has exactly 3 channels (RGB)
  if image.ndim == 3 and image.shape[2] == 4:  # RGBA image
    image = image[:, :, :3]  # Keep only RGB channels
  elif image.ndim == 2:  # Grayscale image
    image = np.stack([image] * 3, axis=-1)  # Convert to RGB

  assert image.ndim == 3 and image.shape[2] == 3, "Image must have exactly 3 channels"
  return image


def make_block_env_params(config: BlockConfig, default_params: struct.PyTreeNode):
  #########
  # Make sure both goal locations and objects have 3 values
  #########
  if config.train_distractor_object_location is None:
    # dummy value and location with tree
    train_distractor_object_location = (
      min(config.start_train_positions[0][0] + 15, 47),
      min(config.start_train_positions[0][1] + 15, 47),
    )
    goal_locations = (
      config.train_object_location,
      train_distractor_object_location,
      config.test_object_location,
    )
    goal_objects = np.concatenate(
      (
        config.train_objects,
        [BlockType.TREE.value],
        config.test_objects,
      )
    )
  else:
    goal_locations = (
      config.train_object_location,
      config.train_distractor_object_location,
      config.test_object_location,
    )
    goal_objects = np.concatenate((config.train_objects, config.test_objects))

  assert len(goal_objects) == len(goal_locations)

  env_params = default_params.replace(
    world_seeds=(config.world_seed,),
    goal_locations=goal_locations,
    placed_goals=goal_objects,
    placed_achievements=tuple(BLOCK_TO_GOAL[i] for i in goal_objects),
  )
  return env_params


def get_goal_image(
  achievement_idx: int, block_pixel_size: int = BLOCK_PIXEL_SIZE_HUMAN
):
  """Get the image for a goal Achievement."""
  # Map Achievement to corresponding BlockType/ItemType
  textures = TEXTURES[block_pixel_size]
  try:
    block = GOAL_TO_BLOCK[achievement_idx]
    texture = textures["full_map_block_textures"][block]
  except KeyError:
    # try:
    #  texture = textures["sword_textures"][achievement]
    # except KeyError:
    import ipdb

    ipdb.set_trace()
    achievement = Achievement(achievement_idx)
    raise ValueError(f"Unmapped achievement: {achievement}")
  return texture[:block_pixel_size, :block_pixel_size]


def visualize_block_config(config: BlockConfig, jax_env, **kwargs):
  """Visualizes a block configuration showing the full map and agent views from all starting positions.

  Args:
      config: BlockConfig instance containing world seed and start positions
      jax_env: The Craftax environment instance

  Returns:
      Tuple of (full_map_figure, agent_views_figure)
  """
  import matplotlib.pyplot as plt

  # Calculate number of start positions
  n_train = len(config.start_train_positions)
  n_eval = len(config.start_eval_positions)
  n_eval2 = len(config.start_eval2_positions) if config.start_eval2_positions else 0
  total_positions = n_train + n_eval + n_eval2 + 1

  # Create full map figure with two subplots side by side
  fig_map = plt.figure(figsize=(14, 7))

  # Left subplot - rendered environment
  goal_objects = np.concatenate((config.train_objects, config.test_objects))
  goal_locations = (
    config.train_object_location,
    config.test_object_location,
  )

  if config.train_distractor_object_location is not None:
    goal_locations = (
      config.train_object_location,
      config.train_distractor_object_location,
      config.test_object_location,
    )
    assert len(goal_objects) == len(goal_locations)

  env_params = jax_env.default_params.replace(
    world_seeds=(config.world_seed,),
    max_timesteps=100000,
    goal_locations=goal_locations,
    placed_goals=goal_objects,
  )
  plt.subplot(1, 2, 1)
  key = jax.random.PRNGKey(0)
  obs, state = jax_env.reset(key, env_params)
  with jax.disable_jit():
    full_map = render_full(
      state,
      show_agent=False,
      show_center_agent=True,
      block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN,
    ).astype(np.uint8)
  plt.imshow(full_map)
  title = f"Rendered Environment (World Seed: {config.world_seed})"
  title += f"\nTrain objects: {[BlockType(o).name for o in config.train_objects]}"
  title += f"\nTest objects: {[BlockType(o).name for o in config.test_objects]}"
  plt.title(title)

  plt.axis("off")

  # Right subplot - cached full map
  ax2 = plt.subplot(1, 2, 2)
  # Call train_test_paths to get the visualization
  craftax_utils.train_test_paths(
    jax_env=jax_env,
    params=env_params,
    world_seed=config.world_seed,
    start_position=config.start_eval_positions[0],
    train_object=BlockType(config.train_objects[0]),
    test_object=BlockType(config.test_objects[0]),
    train_object_location=config.train_object_location,
    test_object_location=config.test_object_location,
    train_distractor_object=BlockType(config.train_objects[1])
    if len(config.train_objects) > 1
    else None,
    train_distractor_object_location=config.train_distractor_object_location,
    extra_positions=config.start_train_positions,
    second_start_position=config.start_eval2_positions[0]
    if config.start_eval2_positions
    else None,
    ax=ax2,  # Pass the specific subplot axis
    **kwargs,
  )
  ax2.set_title(f"Path Visualization (World Seed: {config.world_seed})")
  ax2.axis("off")

  # Create agent views figure
  columns = 4
  rows = math.ceil((total_positions) / columns)  # Round up divisionn

  fig_views = plt.figure(figsize=(15, 4 * rows))

  # Function to render from a position
  def render_from_position(pos, idx, title_prefix):
    # Create environment params with this start position
    render_env_params = env_params.replace(
      start_positions=(pos,),
    )

    # Reset environment
    key = jax.random.PRNGKey(0)
    obs, state = jax_env.reset(key, render_env_params)
    import ipdb

    ipdb.set_trace()
    # Get partial observation using partial renderer
    obs = render_partial(state, block_pixel_size=BLOCK_PIXEL_SIZE_IMG).astype(np.uint8)

    # Plot
    plt.figure(fig_views.number)  # Ensure we're plotting on the views figure
    ax = plt.subplot(rows, columns, idx + 1)
    ax.imshow(obs)
    ax.set_title(f"{title_prefix} Start Pos: {pos}")
    ax.axis("off")

  # Plot train positions
  for idx, pos in enumerate(config.start_train_positions):
    render_from_position(pos, idx, "Train")

  # Plot eval positions
  offset = n_train
  for idx, pos in enumerate(config.start_eval_positions):
    render_from_position(pos, idx + offset, "Eval")

  # Plot eval2 positions if they exist
  if config.start_eval2_positions:
    offset = n_train + n_eval
    for idx, pos in enumerate(config.start_eval2_positions):
      render_from_position(pos, idx + offset, "Eval2")

  # Add view from near test object location
  test_view_pos = (config.train_object_location[0] + 1, config.train_object_location[1])
  render_from_position(test_view_pos, total_positions - 1, "Test Object View")

  # Remove any empty subplots
  for idx in range(total_positions + 1, rows * columns):  # +1 to account for new plot
    ax = plt.subplot(rows, columns, idx + 1)
    ax.remove()

  plt.figure(fig_views.number)  # Ensure we're adjusting the views figure
  plt.tight_layout()
  return fig_map, fig_views
