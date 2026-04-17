"""Render bare JaxMaze environments (no paths/arrows) for all rotations.

Produces 4 PDFs in output/jaxmaze_envs/:
  - jaxmaze_envs_raw.pdf / _no_agent.pdf: Practice, Two Paths, Shortcut, Intermediary Start
  - jaxmaze_envs_raw_juncture.pdf / _no_agent.pdf: Juncture (Near), Juncture (Far)

Each figure has 1 row per maze x 4 columns for rotations.
Objects rotate across columns via rotate_group() matching the experiment.
"""

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from jaxmaze import renderer, utils
from jaxmaze.human_dyna import mazes

OUTPUT_DIR = os.path.join(
  os.path.dirname(os.path.abspath(__file__)), "output", "jaxmaze_envs"
)

image_dict = utils.load_image_dict()
image_keys = image_dict["keys"]

# 2 groups matching experiment num_groups=2
INITIAL_GROUPS = np.array(
  [
    [image_keys.index("orange"), image_keys.index("potato")],
    [image_keys.index("lettuce"), image_keys.index("apple")],
  ],
  dtype=np.int32,
)

ROTATIONS = list(itertools.product([False, True], repeat=2))

# Main figure: row label -> maze variable
MAIN_MAZES = {
  "Practice": mazes.big_practice_maze,
  "Two Paths": mazes.big_m3_maze1,
  "Shortcut": mazes.big_m1_maze3,
  "Intermediary Start": mazes.big_m2_maze2,
}

# Juncture figure (separate)
JUNCTURE_MAZES = {
  "Juncture (Near)": mazes.big_m4_maze_short,
  "Juncture (Near, Blind)": mazes.big_m4_maze_short_blind,
  "Juncture (Far)": mazes.big_m4_maze_long,
}


def rotate_group(groups):
  """Circular left-shift of flattened groups, matching experiments.py."""
  flattened = groups.flatten()
  permuted_flat = np.concatenate([flattened[1:], [flattened[0]]])
  return permuted_flat.reshape(groups.shape)


def render_maze(maze_str, char2key, ax=None, show_agent=True):
  """Render a bare maze with objects, spawn locations, and optional agent."""
  level_init = utils.from_str(maze_str, char2key, return_map_init=False)
  grid, agent_pos, agent_dir = level_init
  spawn_locs = utils.from_str_spawning(maze_str)
  image = renderer.create_image_from_grid(
    grid, agent_pos, agent_dir, image_dict, spawn_locs=spawn_locs
  )

  if not show_agent:
    # Replace agent tile with spawn-colored floor (agent start is a spawn loc)
    images = image_dict["images"]
    tile_size = images.shape[-2]
    spawn_color = np.array([228, 130, 83], dtype=np.uint8)  # coral, matches renderer
    spawn_tile = np.full((tile_size, tile_size, 3), spawn_color, dtype=np.uint8)
    ay, ax_ = agent_pos
    # Account for 1-cell border added by create_image_from_grid
    py = (ay + 1) * tile_size
    px = (ax_ + 1) * tile_size
    image = np.array(image)
    image[py : py + tile_size, px : px + tile_size] = spawn_tile

  if ax is None:
    fig, ax = plt.subplots(1, figsize=(5, 5))
  ax.imshow(image)
  ax.set_xticks([])
  ax.set_yticks([])
  for spine in ax.spines.values():
    spine.set_visible(False)


def save_figure(fig, filename):
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  path = os.path.join(OUTPUT_DIR, f"{filename}.pdf")
  plt.savefig(path, bbox_inches="tight", dpi=300)
  print(f"Saved {path}")
  plt.close()


def plot_maze_grid(maze_dict, filename_base):
  """Plot a grid of mazes (rows) x rotations (cols) with object rotation."""
  n_rots = len(ROTATIONS)
  row_labels = list(maze_dict.keys())
  n_rows = len(row_labels)

  for show_agent in [True, False]:
    fig, axs = plt.subplots(n_rows, n_rots, figsize=(6 * n_rots, 5 * n_rows))
    if n_rows == 1:
      axs = axs[np.newaxis, :]

    for col, (h, v) in enumerate(ROTATIONS):
      axs[0, col].set_title(f"Rotation {col + 1}", fontsize=14)

    # Rotate groups once per column (matching experiment behavior)
    groups = INITIAL_GROUPS.copy()
    rotated_groups = []
    for _ in range(n_rots):
      groups = rotate_group(groups)
      rotated_groups.append(groups.copy())

    for row, label in enumerate(row_labels):
      maze_str = maze_dict[label]
      axs[row, 0].set_ylabel(label, fontsize=14, rotation=90, labelpad=15)
      for col, (h, v) in enumerate(ROTATIONS):
        char2key = mazes.groups_to_char2key(rotated_groups[col])
        rotated = utils.reverse(maze_str, h, v)
        render_maze(rotated, char2key, ax=axs[row, col], show_agent=show_agent)

    plt.tight_layout()
    suffix = "" if show_agent else "_no_agent"
    save_figure(fig, f"{filename_base}{suffix}")


def main():
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  plot_maze_grid(MAIN_MAZES, "jaxmaze_envs_raw")
  plot_maze_grid(JUNCTURE_MAZES, "jaxmaze_envs_raw_juncture")


if __name__ == "__main__":
  main()
