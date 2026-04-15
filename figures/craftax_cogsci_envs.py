""" """

import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "simulations"))

import jax
import numpy as np
import matplotlib.pyplot as plt

from craftax.craftax.renderer import (
  render_craftax_pixels as render_craftax_pixels_partial,
)
from craftax.craftax.constants import BlockType, BLOCK_PIXEL_SIZE_HUMAN
import craftax_utils
from analysis.vis_utils import get_craftax_env_image, _get_craftax_env
from simulations.craftax_experiment_configs import BlockConfig
import plot_configs
import data_configs
import craftax_experiment_configs


def save_figure(fig, filename):
  directory = data_configs.CRAFTAX_ENV_FIGURES_DIR
  os.makedirs(directory, exist_ok=True)
  plt.savefig(os.path.join(directory, f"{filename}.pdf"), bbox_inches="tight", dpi=300)
  print(f"Saved figure to {directory}/{filename}.pdf")
  plt.close()


def visualize_simplified_block_config(
  config: BlockConfig,
  train_color=plot_configs.TRAIN_COLOR,
  eval_color=plot_configs.EVAL_COLOR,
):
  """Generates two simplified visualizations for a block configuration.

  Args:
      config: BlockConfig instance containing world seed and start positions.
      train_color: Color for the training path.
      eval_color: Color for the evaluation path.

  Returns:
      Tuple of (path_figure, agent_view_figure)
  """
  # --- Figure 1: Path Visualization (uses cached image + cached A* paths) ---
  fig_path, ax_path = plt.subplots(figsize=(7, 7))

  # Get cached env image instead of resetting env
  cached_image, maze_height, maze_width = get_craftax_env_image(config.world_seed)

  print(train_color, eval_color)
  fig_path, ax_path = craftax_utils.train_test_paths(
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
    ax=ax_path,
    train_color=train_color,
    eval_color=eval_color,
    show_path_length=False,
    arrow_scale=10,
    line_thickness=10.0,
    start_marker_size=30,
    image=cached_image,
    maze_height=maze_height,
    maze_width=maze_width,
  )
  ax_path.axis("off")
  plt.tight_layout()

  # --- Figure 2: Agent View from Eval Start (needs env for partial rendering) ---
  fig_view, ax_view = plt.subplots(figsize=(4, 4))

  # Lazy-load env only for agent partial view
  jax_env, default_params = _get_craftax_env()

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
  _obs_render, state_render = jax_env.reset(key, render_env_params)

  agent_view = render_craftax_pixels_partial(
    state_render, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN
  ).astype(np.uint8)

  ax_view.imshow(agent_view)
  ax_view.axis("off")
  plt.tight_layout()

  return fig_path, fig_view


if __name__ == "__main__":
  for i in range(4):
    fig_path, fig_view = visualize_simplified_block_config(
      craftax_experiment_configs.PATHS_CONFIGS[i]
    )
    save_figure(fig_path, f"{i}_fullmap")
    save_figure(fig_view, f"{i}_agentmap")
