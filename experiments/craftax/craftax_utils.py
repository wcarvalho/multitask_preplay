from typing import Union, Tuple, List, Optional
import jax
import jax.numpy as jnp
from collections import deque
from craftax.craftax.constants import Action, BlockType, ItemType, BLOCK_PIXEL_SIZE_IMG
from craftax_fullmap_renderer import TEXTURES
from craftax_fullmap_renderer import render_craftax_pixels as render_craftax_pixels_full
from craftax.craftax.renderer import (
  render_craftax_pixels as render_craftax_pixels_partial,
)
import craftax_fullmap_constants as constants
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import namedtuple
from matplotlib.animation import FuncAnimation

try:
  from tqdm.notebook import tqdm
except ImportError:
  from tqdm import tqdm

CACHE_DIR = "craftax_cache"

TRAIN_COLOR = "red"
TEST_COLOR = "#679FE5"  # pretty blue
TEST_COLOR2 = "#FFB700"


def array_to_tuple(array):
  return tuple(int(i) for i in array)


def find_n_empty_positions(
  rng, state, n, static_params, radius=20, center_pos=(24, 24)
):
  """Find n empty positions within radius of center_pos.

  Args:
      rng: JAX random key
      current_map: 2D array representing the game map
      n: Number of empty positions to find
      center_pos: Reference position [y, x]
      radius: Maximum distance from center_pos
      static_params: Environment parameters containing map size

  Returns:
      positions: Array of shape [n, 2] containing empty positions
      rng: Updated random key
  """
  current_map = state.map[state.player_level]
  center_pos = jnp.asarray(center_pos)
  # Generate many candidate positions to increase success chance
  num_candidates = n * 10  # Try 10x more positions than needed

  rng, _rng = jax.random.split(rng)

  # Generate offsets from -radius to +radius
  offsets = jax.random.randint(
    _rng, shape=(num_candidates, 2), minval=-radius, maxval=radius + 1
  )

  # Add offsets to center position
  positions = center_pos + offsets

  # Clip to ensure within map bounds
  positions = jnp.clip(
    positions,
    a_min=2,
    a_max=jnp.array([static_params.map_size[0] - 2, static_params.map_size[1] - 2]),
  )

  # Check which positions match any walkable block (not just empty space)
  walkable_blocks = jnp.array(
    [
      BlockType.GRASS.value,
      BlockType.STONE.value,
      BlockType.PATH.value,
    ]
  )
  is_walkable = (
    current_map[positions[:, 0], positions[:, 1]][:, None] == walkable_blocks
  ).any(axis=1)

  # Get indices of empty positions
  empty_indices = jnp.where(is_walkable)[0]

  # Take the first n empty positions (or all if fewer than n exist)
  num_empty = jnp.minimum(n, empty_indices.shape[0])

  selected_indices = empty_indices[:num_empty]
  valid_positions = positions[selected_indices]

  # Pad with zeros if we found fewer than n positions
  padding = jnp.zeros((n - num_empty, 2), dtype=jnp.int32)
  valid_positions = jnp.concatenate([valid_positions, padding], axis=0)

  return valid_positions, rng


def bfs(
  state, goal: Union[int, Tuple[int, int]], key=None, walkable_blocks=None, budget=1e8
):
  """Performs Breadth-First Search to find a path from the player's position to a goal.

  Args:
      state: Game state object containing the map and player position.
      goal: Either a tuple of (row, col) coordinates or a BlockType value to search for.
      key: Optional JAX PRNG key for random direction shuffling. Defaults to key(42).
      walkable_blocks: List of block types that can be traversed. Defaults to [GRASS, STONE, PATH].
      budget: Maximum number of iterations before giving up. Defaults to 1e8.

  Returns:
      Tuple of:
          - JAX array of coordinates forming the path to the goal, or None if no path found
          - Number of iterations performed during search
  """
  map = state.map[state.player_level]  # Current level's map
  agent_pos = state.player_position

  if key is None:
    key = jax.random.PRNGKey(42)
  if walkable_blocks is None:
    walkable_blocks = [
      BlockType.GRASS.value,
      BlockType.STONE.value,
      BlockType.PATH.value,
    ]

  rows, cols = map.shape
  queue = deque([(agent_pos, [agent_pos])])
  visited = set()
  iterations = 0

  # Handle different goal types
  is_position_goal = isinstance(goal, (tuple, jnp.ndarray)) and len(goal) == 2
  if is_position_goal:
    goal_object_type = map[goal[0], goal[1]]
    walkable_blocks.append(goal_object_type)

  passible = jnp.array(walkable_blocks)

  # Create progress bar without total to show raw counts
  pbar = tqdm(desc="BFS Iterations")

  while queue:
    key, subkey = jax.random.split(key)
    iterations += 1
    pbar.update(1)  # Update progress bar

    if iterations >= budget:
      pbar.close()  # Close progress bar before returning
      return [], iterations

    current_pos, path = queue.popleft()
    # Check goal condition based on goal type
    if is_position_goal:
      if tuple(current_pos) == tuple(goal):  # Convert both to tuples for comparison
        pbar.close()  # Close progress bar before returning
        return jnp.array([p for p in path]), iterations
    else:  # integer (BlockType.value)
      if map[current_pos[0], current_pos[1]] == goal:
        pbar.close()  # Close progress bar before returning
        return jnp.array([p for p in path]), iterations
    visited.add(tuple([int(i) for i in current_pos]))

    # Shuffle the order of directions
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    key, subkey = jax.random.split(key)
    directions = jax.random.permutation(subkey, jnp.array(directions))

    for dx, dy in directions:
      new_x, new_y = int(current_pos[0] + dx), int(current_pos[1] + dy)
      if (
        0 <= new_x < rows
        and 0 <= new_y < cols
        and (new_x, new_y) not in visited
        and (map[new_x, new_y] == passible).any()
      ):
        new_path = path + [(new_x, new_y)]
        iterations += 1
        queue.append(((new_x, new_y), new_path))

  pbar.close()  # Close progress bar before returning
  return [], iterations


def manhattan_distance(pos1, pos2):
  """Calculate Manhattan distance between two positions."""
  return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def astar(
  state, goal: Union[int, Tuple[int, int]], key=None, walkable_blocks=None, budget=1e8
):
  """Performs A* Search to find a path from the player's position to a goal.

  Args: [same as before]
  """
  map = state.map[state.player_level]
  agent_pos = array_to_tuple(state.player_position)

  if key is None:
    key = jax.random.PRNGKey(42)
  if walkable_blocks is None:
    walkable_blocks = [
      BlockType.GRASS.value,
      BlockType.STONE.value,
      BlockType.PATH.value,
      BlockType.SAND.value,
    ]

  rows, cols = map.shape

  # Handle different goal types
  is_position_goal = isinstance(goal, (tuple, jnp.ndarray)) and len(goal) == 2
  if is_position_goal:
    goal_object_type = map[goal[0], goal[1]]
    walkable_blocks.append(goal_object_type)
    goal_pos = goal
  else:
    # For BlockType goals, find the closest matching block
    matches = jnp.where(map == goal)
    if len(matches[0]) == 0:
      return [], 0
    # Use Manhattan distance to find closest goal
    distances = [
      manhattan_distance(agent_pos, (y, x)) for y, x in zip(matches[0], matches[1])
    ]
    closest_idx = jnp.argmin(jnp.array(distances))
    goal_pos = (int(matches[0][closest_idx]), int(matches[1][closest_idx]))

  passible = jnp.array(walkable_blocks)

  # Priority queue implemented as a list of (priority, count, pos, path) tuples
  import heapq

  count = 0  # Tiebreaker for equal priorities
  open_set = [(0, count, agent_pos, [agent_pos])]
  closed_set = set()
  g_scores = {tuple(agent_pos): 0}  # Cost from start to node
  iterations = 0

  pbar = tqdm(desc="A* Iterations")

  while open_set:
    iterations += 1
    pbar.update(1)

    if iterations >= budget:
      pbar.close()
      return [], iterations

    # Get node with lowest f_score
    current = heapq.heappop(open_set)
    current_pos = current[2]
    current_path = current[3]

    if tuple(current_pos) == tuple(goal_pos):
      pbar.close()
      return jnp.array(current_path), iterations

    if tuple(current_pos) in closed_set:
      continue

    closed_set.add(tuple(current_pos))

    # Shuffle directions for randomness
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    key, subkey = jax.random.split(key)
    directions = jax.random.permutation(subkey, jnp.array(directions))

    for dx, dy in directions:
      new_x, new_y = int(current_pos[0] + dx), int(current_pos[1] + dy)
      new_pos = (new_x, new_y)

      if (
        0 <= new_x < rows
        and 0 <= new_y < cols
        and new_pos not in closed_set
        and (map[new_x, new_y] == passible).any()
      ):
        # g_score is distance from start
        tentative_g = g_scores[tuple(current_pos)] + 1

        if new_pos not in g_scores or tentative_g < g_scores[new_pos]:
          # This path is better than previous ones
          g_scores[new_pos] = tentative_g
          f_score = tentative_g + manhattan_distance(new_pos, goal_pos)
          count += 1
          heapq.heappush(open_set, (f_score, count, new_pos, current_path + [new_pos]))

  pbar.close()
  import ipdb

  ipdb.set_trace()
  return [], iterations


def actions_from_path(path):
  if path is None or len(path) < 2:
    return jnp.array([Action.NOOP.value])

  actions = []
  for i in range(1, len(path)):
    prev_pos = path[i - 1]
    curr_pos = path[i]

    dx = curr_pos[0] - prev_pos[0]
    dy = curr_pos[1] - prev_pos[1]

    if dx == 1:
      actions.append(Action.DOWN.value)
    elif dx == -1:
      actions.append(Action.UP.value)
    elif dy == 1:
      actions.append(Action.RIGHT.value)
    elif dy == -1:
      actions.append(Action.LEFT.value)
    else:
      actions.append(Action.NOOP.value)

  actions.append(Action.NOOP.value)
  return jnp.array(actions)


def place_arrows_on_image(
  image,
  positions,
  actions,
  maze_height,
  maze_width,
  arrow_scale=5,
  arrow_color="b",
  start_color="w",
  ax=None,
  display_image=True,
  show_path_length=True,
):
  # Get the dimensions of the image and the maze
  image_height, image_width, _ = image.shape

  # Calculate the scaling factors for mapping maze coordinates to image coordinates
  scale_y = image_height / maze_height
  scale_x = image_width / maze_width

  # No need for wall offsets anymore
  offset_y = 0
  offset_x = 0

  # Create a figure and axis
  if ax is None:
    fig, ax = plt.subplots(1, figsize=(5, 5))

  # Display the rendered image
  if display_image:
    ax.imshow(image)

  # Calculate the center coordinates near the end position for text placement
  if show_path_length and len(positions) > 3:
    # Position text 3 steps from the end
    text_idx = len(positions) - 3
    text_y = offset_y + (positions[text_idx][0] + 0.5) * scale_y
    text_x = offset_x + (positions[text_idx][1] + 0.5) * scale_x
    # Add text showing path length
    ax.text(
      text_x + scale_x,  # Slightly offset
      text_y - scale_y,
      f"{len(positions)}",
      color=arrow_color,
      fontsize=18,
      ha="left",
      va="bottom",
      weight="bold",
    )

  # Iterate over each position and action
  for (y, x), action in zip(positions, actions):
    # Calculate the center coordinates of the cell in the image
    center_y = offset_y + (y + 0.5) * scale_y
    center_x = offset_x + (x + 0.5) * scale_x

    # Define the arrow directions based on the action
    if action == Action.UP.value:
      dx, dy = 0, -scale_y / 2
    elif action == Action.DOWN.value:
      dx, dy = 0, scale_y / 2
    elif action == Action.LEFT.value:
      dx, dy = -scale_x / 2, 0
    elif action == Action.RIGHT.value:
      dx, dy = scale_x / 2, 0
    else:  # KeyboardActions.done
      continue  # Skip drawing an arrow for the 'done' action

    # Draw the arrow on the image with specified color
    ax.arrow(
      center_x,
      center_y,
      dx,
      dy,
      head_width=scale_x / (arrow_scale * 0.7),  # Increased head width by ~40%
      head_length=scale_y / (arrow_scale * 0.7),  # Increased head length by ~40%
      width=scale_x / (arrow_scale * 2),
      fc=arrow_color,
      ec=arrow_color,
    )

  # Remove the axis ticks and labels
  ax.set_xticks([])
  ax.set_yticks([])
  return ax


def get_object_positions(state, block_type):
  map = state.map
  # Find all positions where the block type matches
  matches = jnp.where(map[0] == block_type.value)
  # Stack the y and x coordinates into a single array of shape (N, 2)
  positions = jnp.stack([matches[0], matches[1]], axis=1)

  # Calculate distances to agent position
  agent_pos = state.player_position
  distances = jnp.sqrt(
    jnp.sum((positions - agent_pos) ** 2, axis=1)  # Euclidean distance
  )

  # Sort positions by distance
  sorted_indices = jnp.argsort(distances)
  positions = positions[sorted_indices]

  return positions


def render_fn(state, show_agent=True, block_pixel_size=constants.BLOCK_PIXEL_SIZE_IMG):
  image = render_craftax_pixels_full(
    state, block_pixel_size=block_pixel_size, show_agent=show_agent
  )
  return image.astype(jnp.uint8)


def get_cached_path(world, start_pos, goal_pos):
  """Load cached path if it exists."""
  start_pos = array_to_tuple(start_pos)
  goal_pos = array_to_tuple(goal_pos)

  cache_dir = os.path.join(CACHE_DIR, "paths")
  cache_file = os.path.join(cache_dir, f"world_{world}_path_{start_pos}_{goal_pos}.npy")
  if os.path.exists(cache_file):
    try:
      print(f"Loading path from cache: {cache_file}")
      return np.load(cache_file, allow_pickle=True)
    except Exception:
      return None
  return None


def save_path_to_cache(path, world, start_pos, goal_pos):
  """Save path to cache."""
  start_pos = array_to_tuple(start_pos)
  goal_pos = array_to_tuple(goal_pos)
  cache_dir = os.path.join(CACHE_DIR, "paths")
  os.makedirs(cache_dir, exist_ok=True)
  cache_file = os.path.join(cache_dir, f"world_{world}_path_{start_pos}_{goal_pos}.npy")
  print(f"Saving path to cache: {cache_file}")
  np.save(cache_file, path)


def display_map(
  state,
  params,
  block_pixel_size: int = constants.BLOCK_PIXEL_SIZE_IMG,
  goals: Optional[Union[BlockType, List[BlockType]]] = None,
  refresh_cache: bool = False,
  show_agent: bool = True,
  display_paths: bool = True,
  paths_nearby: Optional[Union[int, Tuple[int, int]]] = None,
  nearby_radius: int = 15,
  goal_idx: int = None,
):
  world = int(params.world_seeds[0])
  fig, ax = plt.subplots(1, figsize=(12, 12))

  image = render_fn(state, show_agent=show_agent, block_pixel_size=block_pixel_size)
  ax.imshow(image)

  ax.axis("off")  # This removes the axes and grid
  if goals is None or not display_paths:
    ax.set_title(f"World {world}")
    return image, fig, ax

  # colors = [
  #  "#FFB700",  # google orange
  #  "#679FE5",  # pretty blue
  #  "#D55E00",  # vermillion
  #  "#186CED",  # google blue
  #  "#CC79A7",  # reddish purple
  #  "#9B80E6",  # nice purple
  #  "#186CED",  # google blue
  #  (86 / 255, 180 / 255, 233 / 255),  # sky blue
  # ]
  colors = [TRAIN_COLOR, TEST_COLOR, TEST_COLOR2]
  color_idx = -1
  if isinstance(goals, BlockType):
    goals = [goals]

  path_lengths = []
  for goal in goals:
    print("=" * 30)
    print(f"Goal: {goal}")
    print("=" * 30)
    color_idx += 1
    goal_positions = get_object_positions(state, goal)
    # Filter goal positions if paths_nearby is specified
    if paths_nearby is not None:
      filtered_positions = []
      for pos in goal_positions:
        # Calculate Manhattan distance to paths_nearby point
        distance = jnp.sqrt(
          (pos[0] - paths_nearby[0]) ** 2 + (pos[1] - paths_nearby[1]) ** 2
        )
        if distance <= nearby_radius:
          filtered_positions.append(pos)
      goal_positions = jnp.array(filtered_positions)

    if goal_idx is not None:
      goal_positions = [goal_positions[goal_idx]]
    print("Goal positions: ", goal_positions)

    paths = []
    start_pos = tuple(state.player_position)
    for goal_position in goal_positions:
      goal_pos = tuple(goal_position)
      path = None
      if not refresh_cache:
        # Try to load from cache first
        path = get_cached_path(world, start_pos, goal_pos)

      if path is None:
        path, _ = astar(state, goal_position)
        save_path_to_cache(path, world, start_pos, goal_pos)

      if path is not None and len(path) > 0:
        # Check if path exists before getting actions
        actions = actions_from_path(path)
        paths.append((path, actions))

    # sort paths by length
    paths.sort(key=lambda x: len(x[0]))

    for path, actions in paths:
      path_lengths.append(int(len(path)))
      place_arrows_on_image(
        image,
        path,
        actions,
        state.map.shape[1],
        state.map.shape[2],
        ax=ax,
        display_image=False,
        arrow_color=colors[color_idx % len(colors)],
        show_path_length=True,
      )
  ax.set_title(f"World {world}\nPath lengths: {path_lengths}")
  return image, fig, ax


def place_start_marker(ax, position, state, image, start_color="w"):
  """Place a star marker at the starting position.

  Args:
      ax: matplotlib axis
      position: tuple of (y, x) coordinates in maze space
      image: rendered image array of shape (height, width, channels)
      start_color: color of the star marker
  """
  # Get image dimensions
  image_height, image_width, _ = image.shape

  # No need for wall offsets
  offset_y = 0
  offset_x = 0

  # Calculate the scaling factors for mapping maze coordinates to image coordinates
  scale_y = image_height / state.map.shape[1]
  scale_x = image_width / state.map.shape[2]

  # Calculate marker position
  start_y = offset_y + (position[0] + 0.5) * scale_y
  start_x = offset_x + (position[1] + 0.5) * scale_x

  ax.plot(
    start_x, start_y, "*", color=start_color, markersize=15, markeredgecolor="black"
  )


def draw_object_path(
  state,
  object_type,
  start_position,
  color,
  ax,
  image,
  world_seed,
  goal_idx: Optional[int] = None,
  nearby_goal: bool = False,
  show_path_length: bool = True,
  arrow_scale: int = 5,
):
  """Draw path to a specific object type from start position."""
  # Get goal position for the object
  goal_positions = get_object_positions(state, object_type)
  goal_position = goal_positions[goal_idx if goal_idx is not None else 0]

  # Get cached path or compute new one
  path = get_cached_path(world_seed, start_position, goal_position)
  if path is None:
    state = state.replace(player_position=start_position)
    path, _ = astar(state, goal_position)
    save_path_to_cache(path, world_seed, start_position, goal_position)

  # Draw the path
  actions = actions_from_path(path)
  place_arrows_on_image(
    image=image,
    positions=path,
    actions=actions,
    maze_height=state.map.shape[1],
    maze_width=state.map.shape[2],
    ax=ax,
    display_image=False,
    arrow_color=color,
    show_path_length=show_path_length,
    arrow_scale=arrow_scale,
    start_color=color,
  )
  return path


def train_test_paths(
  jax_env,
  params,
  world_seed,
  start_position,
  train_object,
  test_object,
  train_object_location,
  test_object_location,
  prefix: str = "",
  train_distractor_object=None,
  train_distractor_object_location=None,
  static_params=None,
  num_extra_start_positions: int = 0,
  extra_positions=None,
  extra_start_positions_rng=None,
  second_start_position: Optional[Tuple[int, int]] = None,
  extra_start_position_center: Optional[Tuple[int, int]] = None,
  nearby_goal: bool = True,
  goal_idx: Optional[int] = None,
  ax=None,
  show_path_length: bool = True,
  arrow_scale: int = 5,
  train_color="",
  eval_color="",
):
  #########################################
  # Create params
  #########################################
  start_position = start_position or (24, 24)

  goal_objects = (train_object, test_object)
  goal_locations = (train_object_location, test_object_location)

  if train_distractor_object_location is not None:
    goal_objects = goal_objects + (train_distractor_object,)
    goal_locations = goal_locations + (train_distractor_object_location,)

  params = params.replace(
    world_seeds=(world_seed,),
    always_diamond=False,
    start_positions=start_position,
    goal_locations=goal_locations,
    placed_goals=tuple(g.value for g in goal_objects),
  )

  #########################################
  # Reset env + display
  #########################################
  key = jax.random.PRNGKey(0)
  obs, state = jax_env.reset(key, params)

  if ax is None:
    fig, ax = plt.subplots(1, figsize=(8, 8))
  else:
    fig = ax.figure

  with jax.disable_jit():
    image = render_fn(
      state, show_agent=False, block_pixel_size=constants.BLOCK_PIXEL_SIZE_IMG
    )
    ax.imshow(image)
    ax.axis("off")  # This removes the axes and grid

  # Draw paths for each object
  if train_distractor_object is not None:
    draw_object_path(
      state,
      train_distractor_object,
      start_position,
      train_color or TRAIN_COLOR,
      ax,
      image,
      world_seed,
      show_path_length=show_path_length,
      arrow_scale=arrow_scale,
    )
  draw_object_path(
    state,
    test_object,
    start_position,
    eval_color or TEST_COLOR,
    ax,
    image,
    world_seed,
    goal_idx=goal_idx,
    show_path_length=show_path_length,
    arrow_scale=arrow_scale,
  )
  draw_object_path(
    state,
    train_object,
    start_position,
    train_color or TRAIN_COLOR,
    ax,
    image,
    world_seed,
    nearby_goal=nearby_goal,
    show_path_length=show_path_length,
    arrow_scale=arrow_scale,
  )

  # Place start marker for the first position
  place_start_marker(ax, start_position, state, image)

  if extra_positions is not None:
    for pos in extra_positions:
      place_start_marker(ax, pos, state, image, start_color="orange")

  # Sample and place extra start positions if requested
  if num_extra_start_positions > 0:
    if extra_start_positions_rng is None:
      extra_start_positions_rng = jax.random.PRNGKey(0)
    if extra_start_position_center is None:
      extra_start_position_center = start_position
    extra_positions, _ = find_n_empty_positions(
      extra_start_positions_rng,
      state,
      num_extra_start_positions,
      static_params or jax_env.default_static_params(),
      radius=15,
      center_pos=extra_start_position_center,
    )
    print("=" * 30)
    print("Extra start positions")
    print("=" * 30)
    print(f"{[(int(pos[0]), int(pos[1])) for pos in extra_positions]}")
    for pos in extra_positions:
      place_start_marker(ax, pos, state, image, start_color="orange")

  if second_start_position is not None:
    draw_object_path(
      state,
      test_object,
      second_start_position,
      TEST_COLOR2,
      ax,
      image,
      world_seed,
      goal_idx=goal_idx,
    )

  cache_dir = os.path.join(CACHE_DIR)
  if prefix:
    cache_dir = os.path.join(cache_dir, prefix)
  os.makedirs(cache_dir, exist_ok=True)
  output_path = os.path.join(cache_dir, f"world_{world_seed}_paths.png")
  plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
  print(f"Saved to {output_path}")
  return fig, ax


def render_goal_object(goal_object_idx: int, block_pixel_size: int):
  """Render just the goal object texture."""
  textures = TEXTURES[block_pixel_size]

  # For blocks
  if goal_object_idx < len(BlockType):
    return textures["full_map_block_textures"][goal_object_idx]

  # For items
  elif goal_object_idx < len(BlockType) + len(ItemType):
    item_idx = goal_object_idx - len(BlockType)
    item_texture = textures["full_map_item_textures"][item_idx]
    # Items have alpha channel, so we need to handle transparency
    rgb = item_texture[:, :, :3]
    alpha = item_texture[:, :, 3:4]
    return rgb * alpha + (1 - alpha) * np.ones_like(rgb)

  else:
    raise ValueError(f"Unknown goal object index: {goal_object_idx}")


def create_reaction_times_video(
  initial_map, first_state, images, path, actions, reaction_times, output_file, fps=3
):
  # Ensure the directory exists
  output_dir = os.path.dirname(output_file)
  if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

  n = len(images)
  width = 4
  fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3 * width, width), dpi=150)
  ax1.imshow(initial_map)
  place_arrows_on_image(
    image=initial_map,
    positions=path,
    actions=actions,
    maze_height=first_state.map.shape[1],
    maze_width=first_state.map.shape[2],
    ax=ax1,
    display_image=True,
    arrow_color="red",
    show_path_length=False,
    start_color="red",
  )

  def update(frame):
    # Clear previous content
    ax2.clear()
    ax3.clear()

    # Left plot: Image
    if images.size > 0:
      img = images[frame]
      ax2.imshow(img, cmap="viridis")
    else:
      ax2.text(0.5, 0.5, "No image data", ha="center", va="center")
    rt = reaction_times[frame]

    ax1.set_title(f"Step: {frame}, Reaction Time: {rt:.2f} s")
    ax1.axis("off")
    ax2.axis("off")

    # Right plot: Bar plot of reaction times
    bars = ax3.bar(range(len(reaction_times)), reaction_times, color="lightblue")
    bars[frame].set_color("red")  # Highlight current index
    ax3.set_xlabel("Time Index")
    ax3.set_title("Reaction Times")
    ax3.set_ylim(0, max(reaction_times) * 1.1)

    return ax3, ax2

  # Create the animation
  anim = FuncAnimation(fig, update, frames=n, interval=1000 / fps, blit=False)
  video = anim.to_html5_video()
  plt.close(fig)  # Close the figure after creating the video
  return video


def create_episode_reaction_times_video(
  episode_data,
  output_file="/tmp/housemaze_anlaysis_craftax/rt_video.mp4",
  fps=3,
  html: bool = True,
):
  def partial_render_fn(state):
    return render_craftax_pixels_partial(
      state, block_pixel_size=BLOCK_PIXEL_SIZE_IMG
    ).astype(jnp.uint8)

  def full_render_fn(state):
    return render_craftax_pixels_full(
      state, show_agent=False, block_pixel_size=BLOCK_PIXEL_SIZE_IMG
    ).astype(jnp.uint8)

  initial_map = full_render_fn(
    jax.tree_map(lambda x: x[0], episode_data.timesteps.state)
  )
  images = jax.vmap(partial_render_fn)(episode_data.timesteps.state)
  reaction_times = episode_data.reaction_times
  path = episode_data.timesteps.state.player_position
  actions = actions_from_path(path)
  assert len(path) == len(actions) == len(reaction_times), (
    f"lengths: {len(path)}, {len(actions)}, {len(reaction_times)}"
  )
  first_state = jax.tree_map(lambda s: s[0], episode_data.timesteps.state)
  video = create_reaction_times_video(
    initial_map=initial_map,
    first_state=first_state,
    images=images,
    path=path,
    actions=actions,
    reaction_times=reaction_times,
    output_file=output_file,
    fps=fps,
  )
  if html:
    from IPython.display import HTML, display

    return display(HTML(video))
  return video


def create_episode_video(
  episode_data,
  output_file="/tmp/housemaze_anlaysis_craftax/episode_video.mp4",
  fps=3,
  html: bool = True,
):
  """Creates a video of an episode without reaction time visualizations.

  Args:
    episode_data: Data from an episode containing states and paths
    output_file: Path to save the video file
    fps: Frames per second for the video
    html: Whether to return an HTML display object (for notebooks)

  Returns:
    Video as HTML if html=True, otherwise returns the video data
  """

  def partial_render_fn(state):
    return render_craftax_pixels_partial(
      state, block_pixel_size=BLOCK_PIXEL_SIZE_IMG
    ).astype(jnp.uint8)

  def full_render_fn(state):
    return render_craftax_pixels_full(
      state, show_agent=False, block_pixel_size=BLOCK_PIXEL_SIZE_IMG
    ).astype(jnp.uint8)

  initial_map = full_render_fn(
    jax.tree_map(lambda x: x[0], episode_data.timesteps.state)
  )
  images = jax.vmap(partial_render_fn)(episode_data.timesteps.state)
  path = episode_data.timesteps.state.player_position
  actions = actions_from_path(path)
  first_state = jax.tree_map(lambda s: s[0], episode_data.timesteps.state)

  # Ensure the directory exists
  output_dir = os.path.dirname(output_file)
  if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

  n = len(images)
  width = 4
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * width, width), dpi=150)
  ax1.imshow(initial_map)
  place_arrows_on_image(
    image=initial_map,
    positions=path,
    actions=actions,
    maze_height=first_state.map.shape[1],
    maze_width=first_state.map.shape[2],
    ax=ax1,
    display_image=True,
    arrow_color="red",
    show_path_length=False,
    start_color="red",
  )

  def update(frame):
    # Clear previous content
    ax2.clear()

    # Display current frame
    if images.size > 0:
      img = images[frame]
      ax2.imshow(img, cmap="viridis")
    else:
      ax2.text(0.5, 0.5, "No image data", ha="center", va="center")

    ax1.set_title(f"Step: {frame}")
    ax1.axis("off")
    ax2.axis("off")

    return (ax2,)

  # Create the animation
  anim = FuncAnimation(fig, update, frames=n, interval=1000 / fps, blit=False)
  video = anim.to_html5_video()
  plt.close(fig)  # Close the figure after creating the video

  if html:
    from IPython.display import HTML, display

    return display(HTML(video))
  return video


if __name__ == "__main__":
  import os
  from craftax.craftax.constants import BlockType
  from simulations.craftax_web_env import CraftaxSymbolicWebEnvNoAutoReset
  from tqdm import tqdm

  WorldConfig = namedtuple(
    "WorldConfig", ["world_seed", "start_position", "train_object", "test_object"]
  )
  # Example usage:
  # config = WorldConfig(world_seed=123,
  #                     start_position=(10,10),
  #                     train_object=BlockType.DIAMOND,
  #                     test_object=BlockType.CRAFTING_TABLE)

  #########################################
  # Create default environment
  #########################################
  MONSTERS = 1
  static_env_params = CraftaxSymbolicWebEnvNoAutoReset.default_static_params()
  static_env_params = static_env_params.replace(
    max_melee_mobs=MONSTERS,
    max_ranged_mobs=MONSTERS,
    max_passive_mobs=10,  # cows
    initial_crafting_tables=True,
    initial_strength=20,
    map_size=(48, 48),
  )
  jax_env = CraftaxSymbolicWebEnvNoAutoReset(
    static_env_params=static_env_params,
  )
  default_params = jax_env.default_params.replace(
    day_length=100000,
    mob_despawn_distance=100000,
  )

  #########################################
  # Create maps
  #########################################
  # Create cache directory if it doesn't exist
  cache_dir = os.path.join(CACHE_DIR, "maps")
  os.makedirs(cache_dir, exist_ok=True)
  # Generate and save maps for each world seed
  for world_seed in tqdm(range(200), desc="world"):
    # Check if file already exists
    output_path = os.path.join(cache_dir, f"world_{world_seed}.png")
    if not os.path.exists(output_path):
      # Set params for this world
      world_params = default_params.replace(
        world_seeds=(world_seed,),
      )

      # Generate world
      key = jax.random.PRNGKey(0)
      obs, state = jax_env.reset(key, world_params)

      # Save visualization
      with jax.disable_jit():
        image, fig, ax = display_map(
          state=state,
          params=world_params,
        )
      # Save figure
      plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
      plt.close()
  ###########################
  ### 2 paths (single goal)
  ##########################

  # configs = [
  #  WorldConfig(world_seed=3, start_position=(22,25), train_object=BlockType.DIAMOND, test_object=BlockType.CRAFTING_TABLE),
  #  WorldConfig(world_seed=15, start_position=(24,24), train_object=BlockType.DIAMOND, test_object=BlockType.CRAFTING_TABLE),
  #  #WorldConfig(world_seed=16, start_position=(40,30), train_object=BlockType.DIAMOND, test_object=BlockType.CRAFTING_TABLE),
  #  #WorldConfig(world_seed=21, start_position=(10,10), train_object=BlockType.DIAMOND, test_object=BlockType.CRAFTING_TABLE),
  # ]

  # for config in configs:
  #  fig, ax = train_test_paths(
  #    jax_env,
  #    params=default_params,
  #    world_seed=config.world_seed,
  #    start_position=config.start_position,
  #    train_object=config.train_object,
  #    test_object=config.test_object,
  #  )
