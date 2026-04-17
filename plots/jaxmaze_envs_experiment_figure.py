"""Plot JaxMaze environment figures showing manipulation paths.

Generates figures for all four manipulations:
  - Two Paths (train path only)
  - Juncture (train + eval paths, near/far, known/unknown)
  - Start Position (train path + eval start positions)
  - Shortcut (train path only)

Source: figures_supplemental/jaxmaze_envs_paper.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

import plot_configs
from jaxmaze import renderer, utils
from jaxmaze.human_dyna import mazes

TRAIN_COLOR = plot_configs.default_colors["sky blue"]
EVAL_COLOR = "red"
EVAL2_COLOR = "yellow"

OUTPUT_DIR = os.path.join(
  os.path.dirname(os.path.abspath(__file__)), "output", "jaxmaze_envs"
)

image_dict = utils.load_image_dict()

image_keys = image_dict["keys"]
groups = [
  # room 1
  [image_keys.index("orange"), image_keys.index("potato")],
  # room 2
  [image_keys.index("lettuce"), image_keys.index("apple")],
  ## room 3
  [image_keys.index("tomato"), image_keys.index("lettuce")],
]
groups = np.array(groups, dtype=np.int32)
task_objects = groups.reshape(-1)
char2key = mazes.groups_to_char2key(groups)


def render_path(
  maze_str,
  goal,
  ax=None,
  rng=None,
  plot_image=True,
  plot_path=True,
  include_spawn=True,
  use_title=False,
  arrow_color="g",
  star_color="",
  star_at_start=False,
  **kwargs,
):
  level_init = utils.from_str(maze_str, char2key, return_map_init=False)
  image = renderer.create_image_from_grid(
    *level_init,
    image_dict,
    spawn_locs=utils.from_str_spawning(maze_str) if include_spawn else None,
    **kwargs,
  )
  grid = level_init[0]
  path = utils.find_optimal_path(grid, level_init[1], np.array([goal]), rng=rng)
  actions = utils.actions_from_path(path)
  changes = utils.count_action_changes(actions[:-1])

  if ax is None:
    fig, ax = plt.subplots(1, figsize=(5, 5))
  if path is None:
    title = "NO PATH FOUND"
    ax.imshow(image)
    if use_title:
      ax.set_title(title)
  else:
    title = f"Path length: {len(path)}. Turns: {sum(changes)}"
    title += f"\n Goal = {image_dict['keys'][goal]}"
    if use_title:
      ax.set_title(title)
    if plot_path:
      renderer.place_arrows_on_image(
        image,
        path,
        actions,
        *level_init[0].shape[:2],
        ax=ax,
        arrow_color=arrow_color,
        plot_image=plot_image,
      )
    if star_at_start:
      image_height, image_width, _ = image.shape
      maze_height, maze_width = level_init[0].shape[:2]
      scale_y = image_height // (maze_height + 2)
      scale_x = image_width // (maze_width + 2)
      offset_y = (image_height - scale_y * maze_height) // 2
      offset_x = (image_width - scale_x * maze_width) // 2

      start_y, start_x = level_init[1]
      center_y = offset_y + (start_y + 0.5) * scale_y
      center_x = offset_x + (start_x + 0.5) * scale_x

      star_color = star_color or arrow_color
      ax.plot(
        center_x,
        center_y,
        marker="*",
        color=star_color,
        markersize=scale_x * (0.5),
        markeredgecolor=star_color,
        markeredgewidth=scale_x / 20,
      )
  ax.axis("off")
  ax.set_xticks([])
  ax.set_yticks([])


def save_figure(fig, filename, directory=None):
  directory = directory or OUTPUT_DIR
  os.makedirs(directory, exist_ok=True)
  plt.savefig(os.path.join(directory, f"{filename}.pdf"), bbox_inches="tight", dpi=300)
  print(f"Saved figure to {directory}/{filename}.pdf")
  plt.close()


def main():
  os.makedirs(OUTPUT_DIR, exist_ok=True)

  ########################################################
  # Two Paths Manipulation - train path only
  ########################################################
  fig, ax = plt.subplots(1, 1, figsize=(6, 5))
  render_path(
    mazes.big_m3_maze1,
    goal=task_objects[0],
    ax=ax,
    include_spawn=False,
    arrow_color=TRAIN_COLOR,
    star_at_start=True,
    star_color="white",
  )
  save_figure(fig, "jaxmaze_envs_two_paths")

  ########################################################
  # Juncture Manipulation - all paths (train + eval)
  ########################################################
  def plot_juncture_near_known(ax):
    render_path(
      mazes.big_m4_maze_short,
      goal=task_objects[0],
      ax=ax,
      include_spawn=False,
      arrow_color=TRAIN_COLOR,
      star_at_start=True,
      star_color="white",
    )
    render_path(
      mazes.big_m4_maze_short_eval_same,
      goal=task_objects[1],
      ax=ax,
      plot_image=False,
      arrow_color=EVAL_COLOR,
      star_at_start=True,
    )
    render_path(
      mazes.big_m4_maze_short_eval_diff,
      goal=task_objects[2],
      ax=ax,
      plot_image=False,
      arrow_color=EVAL2_COLOR,
      star_at_start=True,
    )

  def plot_juncture_near_unknown(ax):
    render_path(
      mazes.big_m4_maze_short_blind,
      goal=task_objects[0],
      ax=ax,
      include_spawn=False,
      arrow_color=TRAIN_COLOR,
      star_at_start=True,
      star_color="white",
    )
    render_path(
      mazes.big_m4_maze_short_eval_same_blind,
      goal=task_objects[1],
      ax=ax,
      plot_image=False,
      arrow_color=EVAL_COLOR,
      star_at_start=True,
    )
    render_path(
      mazes.big_m4_maze_short_eval_diff_blind,
      goal=task_objects[2],
      ax=ax,
      plot_image=False,
      arrow_color=EVAL2_COLOR,
      star_at_start=True,
    )

  def plot_juncture_far_known(ax):
    render_path(
      mazes.big_m4_maze_long,
      goal=task_objects[0],
      ax=ax,
      include_spawn=False,
      arrow_color=TRAIN_COLOR,
      star_at_start=True,
      star_color="white",
    )
    render_path(
      mazes.big_m4_maze_long_eval_same,
      goal=task_objects[1],
      ax=ax,
      plot_image=False,
      arrow_color=EVAL_COLOR,
      star_at_start=True,
    )
    render_path(
      mazes.big_m4_maze_long_eval_diff,
      goal=task_objects[2],
      ax=ax,
      plot_image=False,
      arrow_color=EVAL2_COLOR,
      star_at_start=True,
    )

  fig, axs = plt.subplots(1, 3, figsize=(15, 5))
  plot_juncture_near_known(axs[0])
  axs[0].set_title("Near, Known Test Goal")
  plot_juncture_near_unknown(axs[1])
  axs[1].set_title("Near, Unknown Test Goal")
  plot_juncture_far_known(axs[2])
  axs[2].set_title("Far, Known Test Goal")
  plt.tight_layout()
  save_figure(fig, "jaxmaze_envs_juncture")

  ########################################################
  # Start Manipulation - train path only, keep eval stars
  ########################################################
  fig, ax = plt.subplots(1, 1, figsize=(6, 5))
  render_path(
    mazes.big_m2_maze2,
    goal=task_objects[0],
    ax=ax,
    include_spawn=False,
    arrow_color=TRAIN_COLOR,
    star_at_start=True,
    star_color="white",
  )
  render_path(
    mazes.big_m2_maze2_onpath,
    goal=task_objects[0],
    ax=ax,
    plot_image=False,
    plot_path=False,
    arrow_color=EVAL_COLOR,
    star_at_start=True,
  )
  render_path(
    mazes.big_m2_maze2_offpath,
    goal=task_objects[0],
    ax=ax,
    plot_image=False,
    plot_path=False,
    arrow_color=EVAL2_COLOR,
    star_at_start=True,
  )
  save_figure(fig, "jaxmaze_envs_start")

  ########################################################
  # Shortcut Manipulation - train path only
  ########################################################
  fig, ax = plt.subplots(1, 1, figsize=(6, 5))
  render_path(
    mazes.big_m1_maze3,
    goal=task_objects[0],
    ax=ax,
    include_spawn=False,
    arrow_color=TRAIN_COLOR,
    star_at_start=True,
    star_color="white",
  )
  save_figure(fig, "jaxmaze_envs_shortcut")


if __name__ == "__main__":
  main()
