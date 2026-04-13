"""Generate 1x10 grid showing full maps for Craftax worlds with seeds 1-10."""

import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "simulations"))

import jax
import numpy as np
import matplotlib.pyplot as plt

from simulations.craftax_web_env import CraftaxSymbolicWebEnvNoAutoReset
from experiments.craftax.craftax_fullmap_renderer import (
  render_craftax_pixels as render_craftax_pixels_full,
)
from experiments.craftax.craftax_fullmap_constants import BLOCK_PIXEL_SIZE_HUMAN
import data_configs


def save_figure(fig, filename):
  directory = data_configs.CRAFTAX_ENV_FIGURES_DIR
  os.makedirs(directory, exist_ok=True)
  plt.savefig(os.path.join(directory, f"{filename}.pdf"), bbox_inches="tight", dpi=300)
  print(f"Saved figure to {directory}/{filename}.pdf")
  plt.close()


def make_craftax_env():
  """Create Craftax environment with standard settings."""
  static_env_params = CraftaxSymbolicWebEnvNoAutoReset.default_static_params()
  return CraftaxSymbolicWebEnvNoAutoReset(static_env_params=static_env_params)


def render_full_map(jax_env, world_seed):
  """Render the full map for a given world seed."""
  # Get default params and set world seed
  env_params = jax_env.default_params.replace(world_seeds=(world_seed,))

  key = jax.random.PRNGKey(0)

  # Reset environment to generate the world
  _obs, state = jax_env.reset(key, env_params)

  # Use render_fn with jax.disable_jit to get the full image
  with jax.disable_jit():
    full_map_image = render_craftax_pixels_full(
      state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN
    ).astype(np.uint8)

  return full_map_image


def create_10_maps_figure():
  """Create a 1x10 figure showing full maps for seeds 1-10."""
  jax_env = make_craftax_env()

  # Create 1x10 subplot grid
  fig, axes = plt.subplots(1, 10, figsize=(50, 5))

  for i, seed in enumerate(range(1, 11)):
    print(f"Rendering map for seed {seed}...")

    # Render full map for this seed
    full_map = render_full_map(jax_env, seed)

    # Display in subplot
    axes[i].imshow(full_map)
    # axes[i].set_title(f'Seed {seed}', fontsize=10)
    axes[i].axis("off")

  plt.tight_layout()
  return fig


if __name__ == "__main__":
  # Generate the 1x10 figure with maps for seeds 1-10
  fig = create_10_maps_figure()
  save_figure(fig, "craftax_seeds_1_to_10_fullmaps")
