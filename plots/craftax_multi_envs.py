"""Render Craftax environment maps for all 4 PATHS_CONFIGS.

Produces PDFs in output/craftax_envs/:
  - craftax_multi_envs_fullmap.pdf: 2x2 grid of bare full world maps
  - craftax_multi_envs_fullmap_paths.pdf: 2x2 grid with train/eval paths overlaid
  - craftax_multi_envs_agentmap.pdf: 2x2 grid of agent first-person views
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
  0,
  os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulations"
  ),
)

import matplotlib.pyplot as plt
import numpy as np
import jax

from craftax.craftax.renderer import (
  render_craftax_pixels as render_craftax_pixels_partial,
)
from craftax.craftax.constants import BlockType, BLOCK_PIXEL_SIZE_HUMAN
import craftax_utils
from analysis.vis_utils import get_craftax_env_image, _get_craftax_env
import craftax_experiment_configs
import plot_configs

OUTPUT_DIR = os.path.join(
  os.path.dirname(os.path.abspath(__file__)), "output", "craftax_envs"
)


def save_figure(fig, filename):
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  path = os.path.join(OUTPUT_DIR, f"{filename}.pdf")
  plt.savefig(path, bbox_inches="tight", dpi=300)
  print(f"Saved {path}")
  plt.close()


def render_fullmap(config, ax):
  """Render bare full world map onto given axes."""
  cached_image, _, _ = get_craftax_env_image(config.world_seed)
  ax.imshow(cached_image)
  ax.axis("off")


def render_fullmap_with_paths(config, ax):
  """Render full world map with train/eval paths using cached image + cached A* paths.

  Draws blue (train) paths first, then red (eval) path last so it's visible on top.
  """
  cached_image, maze_height, maze_width = get_craftax_env_image(config.world_seed)
  ax.imshow(cached_image)

  train_color = plot_configs.TRAIN_COLOR
  eval_color = plot_configs.EVAL_COLOR
  start_pos = config.start_eval_positions[0]
  path_kwargs = dict(
    state=None,
    ax=ax,
    image=cached_image,
    world_seed=config.world_seed,
    show_path_length=False,
    arrow_scale=10,
    line_thickness=10.0,
    maze_height=maze_height,
    maze_width=maze_width,
  )

  # 1. Blue paths first (train + distractor)
  if config.train_distractor_object_location is not None:
    craftax_utils.draw_object_path(
      object_type=BlockType(config.train_objects[1]),
      start_position=start_pos,
      color=train_color,
      goal_position=config.train_distractor_object_location,
      **path_kwargs,
    )
  craftax_utils.draw_object_path(
    object_type=BlockType(config.train_objects[0]),
    start_position=start_pos,
    color=train_color,
    goal_position=config.train_object_location,
    **path_kwargs,
  )

  # 2. Red path last (eval/test) — drawn on top
  craftax_utils.draw_object_path(
    object_type=BlockType(config.test_objects[0]),
    start_position=start_pos,
    color=eval_color,
    goal_position=config.test_object_location,
    **path_kwargs,
  )

  # 3. Goal markers
  marker_kwargs = dict(
    state=None, image=cached_image, maze_height=maze_height, maze_width=maze_width
  )
  craftax_utils.place_goal_marker(
    ax,
    config.train_object_location,
    goal_color=train_color,
    marker_size=15,
    **marker_kwargs,
  )
  craftax_utils.place_goal_marker(
    ax,
    config.test_object_location,
    goal_color=eval_color,
    marker_size=15,
    **marker_kwargs,
  )
  if config.train_distractor_object_location is not None:
    craftax_utils.place_goal_marker(
      ax,
      config.train_distractor_object_location,
      goal_color=train_color,
      marker_size=15,
      **marker_kwargs,
    )

  # 4. Start markers — white star for eval, orange for train positions
  craftax_utils.place_start_marker(
    ax,
    start_pos,
    marker_size=30,
    **marker_kwargs,
  )
  for pos in config.start_train_positions:
    craftax_utils.place_start_marker(
      ax,
      pos,
      start_color="orange",
      marker_size=30,
      **marker_kwargs,
    )

  ax.axis("off")


def render_fullmap_with_markers(config, ax):
  """Render full world map with goal/start markers but no path lines."""
  cached_image, maze_height, maze_width = get_craftax_env_image(config.world_seed)
  ax.imshow(cached_image)

  train_color = plot_configs.TRAIN_COLOR
  eval_color = plot_configs.EVAL_COLOR
  marker_kwargs = dict(
    state=None, image=cached_image, maze_height=maze_height, maze_width=maze_width
  )

  # Goal markers
  craftax_utils.place_goal_marker(
    ax,
    config.train_object_location,
    goal_color=train_color,
    marker_size=15,
    **marker_kwargs,
  )
  craftax_utils.place_goal_marker(
    ax,
    config.test_object_location,
    goal_color=eval_color,
    marker_size=15,
    **marker_kwargs,
  )
  if config.train_distractor_object_location is not None:
    craftax_utils.place_goal_marker(
      ax,
      config.train_distractor_object_location,
      goal_color=train_color,
      marker_size=15,
      **marker_kwargs,
    )

  # Start markers — white star for eval, orange for train positions
  craftax_utils.place_start_marker(
    ax, config.start_eval_positions[0], marker_size=30, **marker_kwargs
  )
  for pos in config.start_train_positions:
    craftax_utils.place_start_marker(
      ax, pos, start_color="orange", marker_size=30, **marker_kwargs
    )

  ax.axis("off")


def render_agentmap(config, ax, jax_env, default_params):
  """Render agent first-person view onto given axes."""
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

  start_eval_pos = config.start_eval_positions[0]
  render_env_params = default_params.replace(
    world_seeds=(config.world_seed,),
    max_timesteps=100000,
    goal_locations=goal_locations,
    placed_goals=goal_objects,
    start_positions=(start_eval_pos,),
  )

  key = jax.random.PRNGKey(0)
  _obs, state = jax_env.reset(key, render_env_params)
  agent_view = render_craftax_pixels_partial(
    state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN
  ).astype(np.uint8)

  ax.imshow(agent_view)
  ax.axis("off")


def plot_2x2(configs, render_fn, filename, figsize=(14, 14), **render_kwargs):
  """Plot 4 configs in a 2x2 grid."""
  fig, axs = plt.subplots(2, 2, figsize=figsize)
  for i, config in enumerate(configs):
    row, col = divmod(i, 2)
    render_fn(config, axs[row, col], **render_kwargs)
    axs[row, col].set_title(f"World {config.world_seed}", fontsize=14)
  plt.tight_layout()
  save_figure(fig, filename)


def main():
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  configs = craftax_experiment_configs.PATHS_CONFIGS

  # Bare fullmaps
  plot_2x2(configs, render_fullmap, "craftax_multi_envs_fullmap")

  # Fullmaps with markers only (no path lines)
  plot_2x2(configs, render_fullmap_with_markers, "craftax_multi_envs_fullmap_markers")

  # Fullmaps with paths
  plot_2x2(configs, render_fullmap_with_paths, "craftax_multi_envs_fullmap_paths")

  # Agent views (needs env for partial rendering)
  jax_env, default_params = _get_craftax_env()
  plot_2x2(
    configs,
    render_agentmap,
    "craftax_multi_envs_agentmap",
    figsize=(8, 8),
    jax_env=jax_env,
    default_params=default_params,
  )


if __name__ == "__main__":
  main()
