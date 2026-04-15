"""Generate 1x10 grid showing full maps for Craftax worlds with seeds 1-10."""

import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "simulations"))

import matplotlib.pyplot as plt

from analysis.vis_utils import get_craftax_env_image
import data_configs


def save_figure(fig, filename):
  directory = data_configs.CRAFTAX_ENV_FIGURES_DIR
  os.makedirs(directory, exist_ok=True)
  plt.savefig(os.path.join(directory, f"{filename}.pdf"), bbox_inches="tight", dpi=300)
  print(f"Saved figure to {directory}/{filename}.pdf")
  plt.close()


def render_full_map(world_seed):
  """Render the full map for a given world seed using cached images."""
  image, _h, _w = get_craftax_env_image(world_seed)
  return image


def create_10_maps_figure():
  """Create a 1x10 figure showing full maps for seeds 1-10."""
  # Create 1x10 subplot grid
  fig, axes = plt.subplots(1, 10, figsize=(50, 5))

  for i, seed in enumerate(range(1, 11)):
    print(f"Rendering map for seed {seed}...")

    # Render full map for this seed
    full_map = render_full_map(seed)

    # Display in subplot
    axes[i].imshow(full_map)
    axes[i].axis("off")

  plt.tight_layout()
  return fig


if __name__ == "__main__":
  # Generate the 1x10 figure with maps for seeds 1-10
  fig = create_10_maps_figure()
  save_figure(fig, "craftax_seeds_1_to_10_fullmaps")
