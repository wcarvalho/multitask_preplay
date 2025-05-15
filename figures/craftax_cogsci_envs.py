""" """

import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "simulations"))

import jax
import numpy as np
import matplotlib.pyplot as plt

from simulations.craftax_web_env import CraftaxSymbolicWebEnvNoAutoReset
from craftax.craftax.renderer import (
  render_craftax_pixels as render_craftax_pixels_partial,
)
import craftax_utils
from craftax.craftax.constants import BlockType, BLOCK_PIXEL_SIZE_HUMAN
from simulations.craftax_experiment_configs import BlockConfig
import data_configs as plot_configs
import craftax_experiment_configs


def save_figure(fig, filename):
  directory = f"{plot_configs.DIRECTORY}/craftax_env_figures"
  os.makedirs(directory, exist_ok=True)
  # plt.savefig(os.path.join(directory, f"{filename}.png"), bbox_inches='tight', dpi=300)
  plt.savefig(os.path.join(directory, f"{filename}.pdf"), bbox_inches="tight", dpi=300)
  print(f"Saved figure to {directory}/{filename}.pdf")
  plt.close()


def make_craftax_env():
  MONSTERS = 1
  static_env_params = CraftaxSymbolicWebEnvNoAutoReset.default_static_params()
  static_env_params = static_env_params.replace(
    max_melee_mobs=MONSTERS,
    max_ranged_mobs=MONSTERS,
    max_passive_mobs=10,  # cows
    initial_crafting_tables=True,
    initial_strength=20,
    map_size=(48, 48),
    num_levels=1,
  )
  return CraftaxSymbolicWebEnvNoAutoReset(
    static_env_params=static_env_params,
  )


def visualize_simplified_block_config(
  config: BlockConfig,
  jax_env,
  train_color=plot_configs.TRAIN_COLOR,
  eval_color=plot_configs.EVAL_COLOR,
):
  """Generates two simplified visualizations for a block configuration.

  Args:
      config: BlockConfig instance containing world seed and start positions.
      jax_env: The Craftax environment instance.
      train_color: Color for the training path.
      eval_color: Color for the evaluation path.

  Returns:
      Tuple of (path_figure, agent_view_figure)
  """
  # --- Setup common environment parameters ---
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

  env_params = jax_env.default_params.replace(
    world_seeds=(config.world_seed,),
    max_timesteps=100000,
    goal_locations=goal_locations,
    placed_goals=goal_objects,
  )
  key = jax.random.PRNGKey(0)

  # --- Figure 1: Path Visualization ---
  fig_path, ax_path = plt.subplots(figsize=(7, 7))

  # Need to reset env to get the state for path finding
  # Using eval start position as the primary start for path finding context
  _obs, state_for_path = jax_env.reset(
    key, env_params.replace(start_positions=(config.start_eval_positions[0],))
  )

  fig_path, ax_path = craftax_utils.train_test_paths(
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
    ax=ax_path,
    train_color=train_color,
    eval_color=eval_color,
    show_path_length=False,
    arrow_scale=10,
  )
  ax_path.axis("off")
  plt.tight_layout()  # Adjust layout

  # --- Figure 2: Agent View from Eval Start ---
  fig_view, ax_view = plt.subplots(figsize=(4, 4))

  start_eval_pos = config.start_eval_positions[0]
  render_env_params = env_params.replace(
    start_positions=(start_eval_pos,),
  )

  # Reset environment to the specific start position
  _obs_render, state_render = jax_env.reset(key, render_env_params)

  # Get partial observation using partial renderer
  # Assuming BLOCK_PIXEL_SIZE_HUMAN is appropriate for the agent view
  # If BLOCK_PIXEL_SIZE_IMG is needed, ensure it's imported/defined
  agent_view = render_craftax_pixels_partial(
    state_render, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN
  ).astype(np.uint8)

  ax_view.imshow(agent_view)
  ax_view.axis("off")
  plt.tight_layout()  # Adjust layout

  return fig_path, fig_view


if __name__ == "__main__":
  jax_env = make_craftax_env()

  for i in range(4):
    fig_path, fig_view = visualize_simplified_block_config(
      craftax_experiment_configs.PATHS_CONFIGS[i], jax_env
    )
    save_figure(fig_path, f"{i}_fullmap")
    save_figure(fig_view, f"{i}_agentmap")
