import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from housemaze import utils
from housemaze import renderer
from housemaze.human_dyna import mazes

from functools import partial
import os.path
import matplotlib.pyplot as plt
import numpy as np
import itertools
from configs import default_colors, DIRECTORY

TRAIN_COLOR = "red"
EVAL_COLOR = default_colors["sky blue"]
EVAL2_COLOR = "yellow"

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


def get_grid(maze_str):
  level_init = utils.from_str(maze_str, char2key)
  grid, agent_pos = level_init[:2]
  return grid, agent_pos


def render_maze(maze_str, goal=None, ax=None, include_spawn=True, **kwargs):
  level_init = utils.from_str(maze_str, char2key)
  image = renderer.create_image_from_grid(
    *level_init,
    image_dict,
    spawn_locs=utils.from_str_spawning(maze_str) if include_spawn else None,
    **kwargs,
  )
  if ax is None:
    fig, ax = plt.subplots(1, figsize=(5, 5))
  ax.imshow(image)
  # grid = level_init[0]
  # title = f"\n rows={grid.shape[0]}, cols={grid.shape[1]}"
  title = ""
  if goal is not None:
    title += f"Goal = {image_dict['keys'][goal]}"
  ax.set_title(title)
  # grid = level_init[0]
  # title = f"\n rows={grid.shape[0]}, cols={grid.shape[1]}"
  # ax.set_title(title)
  ax.axis("off")
  ax.set_xticks([])
  ax.set_yticks([])


def render_agent(maze_str, goal=None, ax=None, include_spawn=False, **kwargs):
  level_init = utils.from_str(maze_str, char2key)
  image = renderer.agent_position_in_grid(
    *level_init,
    image_dict,
    # spawn_locs=utils.from_str_spawning(maze_str) if include_spawn else None,
    # **kwargs
  )
  if ax is None:
    fig, ax = plt.subplots(1, figsize=(5, 5))
  ax.imshow(image)
  grid = level_init[0]
  # title = f"\n rows={grid.shape[0]}, cols={grid.shape[1]}"
  title = ""
  if goal is not None:
    title += f"Goal = {image_dict['keys'][goal]}"
  ax.set_title(title)
  grid = level_init[0]
  # title = f"\n rows={grid.shape[0]}, cols={grid.shape[1]}"
  # ax.set_title(title)
  ax.axis("off")
  ax.set_xticks([])
  ax.set_yticks([])


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
  level_init = utils.from_str(maze_str, char2key)
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
    # title += f"\n rows={grid.shape[0]}, cols={grid.shape[1]}"
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
      # Calculate the same scaling factors as used in place_arrows_on_image
      image_height, image_width, _ = image.shape
      maze_height, maze_width = level_init[0].shape[:2]
      scale_y = image_height // (maze_height + 2)
      scale_x = image_width // (maze_width + 2)
      offset_y = (image_height - scale_y * maze_height) // 2
      offset_x = (image_width - scale_x * maze_width) // 2

      # Get start position and convert to image coordinates
      start_y, start_x = level_init[1]
      center_y = offset_y + (start_y + 0.5) * scale_y
      center_x = offset_x + (start_x + 0.5) * scale_x

      # Plot star with size scaled relative to cell size
      star_color = star_color or arrow_color
      ax.plot(
        center_x,
        center_y,
        marker="*",
        color=star_color,
        markersize=scale_x * (0.5),  # Scale star size relative to cell size
        markeredgecolor=star_color,
        markeredgewidth=scale_x / 20,
      )
  ax.axis("off")
  ax.set_xticks([])
  ax.set_yticks([])


def save_figure(fig, filename, directory=None):
  directory = directory or DIRECTORY
  os.makedirs(directory, exist_ok=True)
  # plt.savefig(os.path.join(directory, f"{filename}.png"), bbox_inches='tight', dpi=300)
  plt.savefig(os.path.join(directory, f"{filename}.pdf"), bbox_inches="tight", dpi=300)
  print(f"Saved figure to {directory}/{filename}.pdf")
  plt.close()


def plot_single_and_rotations(base_name, plot_fn, save_figure_fn):
  """Plot both single first figure and all rotations for a given plot configuration.

  Args:
    base_name: Base name for the saved figures (e.g. "two_paths_manipulation")
    plot_fn: Function that takes (ax, reversal) and plots the figure
    save_figure_fn: Function to save the figure
  """
  # First plot - single figure
  fig_single, ax_single = plt.subplots(1, 1, figsize=(6, 5))
  plot_fn(ax_single, (False, False))
  save_figure_fn(fig_single, f"{base_name}_first")

  # All rotations
  fig_all, axs = plt.subplots(1, 4, figsize=(24, 5))
  for idx, reversal in enumerate(itertools.product([False, True], repeat=2)):
    plot_fn(axs[idx], reversal)
  save_figure_fn(fig_all, base_name)


if __name__ == "__main__":
  directory = f"{DIRECTORY}/jaxmaze_env_figures"
  os.makedirs(directory, exist_ok=True)
  save_figure = partial(save_figure, directory=directory)

  ########################################################
  # Two Paths Manipulation
  ########################################################
  def plot_two_paths(ax, reversal):
    render_path(
      utils.reverse(mazes.big_m3_maze1, *reversal),
      goal=task_objects[0],
      ax=ax,
      arrow_color=TRAIN_COLOR,
      star_at_start=True,
      star_color="white",
    )
    render_path(
      utils.reverse(mazes.big_m3_maze1, *reversal),
      goal=task_objects[1],
      ax=ax,
      arrow_color=EVAL_COLOR,
      star_at_start=True,
      star_color="white",
    )

  plot_single_and_rotations("1.two_paths_manipulation", plot_two_paths, save_figure)

  ########################################################
  # Juncture Manipulation
  ########################################################
  def plot_juncture_near_known(ax, reversal):
    render_path(
      utils.reverse(mazes.big_m4_maze_short, *reversal),
      goal=task_objects[0],
      ax=ax,
      arrow_color=TRAIN_COLOR,
      star_at_start=True,
    )
    render_path(
      utils.reverse(mazes.big_m4_maze_short_eval_same, *reversal),
      goal=task_objects[1],
      ax=ax,
      plot_image=False,
      arrow_color=EVAL_COLOR,
      star_at_start=True,
    )
    render_path(
      utils.reverse(mazes.big_m4_maze_short_eval_diff, *reversal),
      goal=task_objects[2],
      ax=ax,
      plot_image=False,
      arrow_color=EVAL2_COLOR,
      star_at_start=True,
    )

  plot_single_and_rotations(
    "2.juncture_manipulation_near_known", plot_juncture_near_known, save_figure
  )

  def plot_juncture_near_unknown(ax, reversal):
    render_path(
      utils.reverse(mazes.big_m4_maze_short_blind, *reversal),
      goal=task_objects[0],
      ax=ax,
      arrow_color=TRAIN_COLOR,
      star_at_start=True,
    )
    render_path(
      utils.reverse(mazes.big_m4_maze_short_eval_same_blind, *reversal),
      goal=task_objects[1],
      ax=ax,
      plot_image=False,
      arrow_color=EVAL_COLOR,
      star_at_start=True,
    )
    render_path(
      utils.reverse(mazes.big_m4_maze_short_eval_diff_blind, *reversal),
      goal=task_objects[2],
      ax=ax,
      plot_image=False,
      arrow_color=EVAL2_COLOR,
      star_at_start=True,
    )

  plot_single_and_rotations(
    "2.juncture_manipulation_near_unknown", plot_juncture_near_unknown, save_figure
  )

  def plot_juncture_far_known(ax, reversal):
    render_path(
      utils.reverse(mazes.big_m4_maze_long, *reversal),
      goal=task_objects[0],
      ax=ax,
      arrow_color=TRAIN_COLOR,
      star_at_start=True,
    )
    render_path(
      utils.reverse(mazes.big_m4_maze_long_eval_same, *reversal),
      goal=task_objects[1],
      ax=ax,
      plot_image=False,
      arrow_color=EVAL_COLOR,
      star_at_start=True,
    )
    render_path(
      utils.reverse(mazes.big_m4_maze_long_eval_diff, *reversal),
      goal=task_objects[2],
      ax=ax,
      plot_image=False,
      arrow_color=EVAL2_COLOR,
      star_at_start=True,
    )

  plot_single_and_rotations(
    "2.juncture_manipulation_far_known", plot_juncture_far_known, save_figure
  )

  # Plot all juncture manipulations together
  def plot_all_junctures():
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    plot_juncture_near_known(axs[0], (False, False))
    axs[0].set_title("Near, Known Test Goal")
    plot_juncture_near_unknown(axs[1], (False, False))
    axs[1].set_title("Near, Unknown Test Goal")
    plot_juncture_far_known(axs[2], (False, False))
    axs[2].set_title("Far, Known Test Goal")
    plt.tight_layout()
    save_figure(fig, "2_juncture_manipulation")

  plot_all_junctures()

  ########################################################
  # Start Manipulation
  ########################################################
  def plot_start_manipulation(ax, reversal):
    render_path(
      utils.reverse(mazes.big_m2_maze2, *reversal),
      goal=task_objects[0],
      ax=ax,
      arrow_color=TRAIN_COLOR,
      star_at_start=True,
    )
    render_path(
      utils.reverse(mazes.big_m2_maze2_onpath, *reversal),
      goal=task_objects[0],
      ax=ax,
      plot_image=False,
      plot_path=False,
      arrow_color=EVAL_COLOR,
      star_at_start=True,
    )
    render_path(
      utils.reverse(mazes.big_m2_maze2_offpath, *reversal),
      goal=task_objects[0],
      ax=ax,
      plot_image=False,
      plot_path=False,
      arrow_color=EVAL2_COLOR,
      star_at_start=True,
    )

  plot_single_and_rotations(
    "3.start_manipulation", plot_start_manipulation, save_figure
  )

  ########################################################
  # Shortcut Manipulation
  ########################################################
  def plot_shortcut_manipulation(ax, reversal):
    render_path(
      utils.reverse(mazes.big_m1_maze3_shortcut, *reversal),
      goal=task_objects[1],
      ax=ax,
      arrow_color=EVAL_COLOR,
      star_at_start=True,
      star_color="white",
    )
    render_path(
      utils.reverse(mazes.big_m1_maze3, *reversal),
      goal=task_objects[0],
      ax=ax,
      arrow_color=TRAIN_COLOR,
      star_at_start=True,
      star_color="white",
    )

  plot_single_and_rotations(
    "4.shortcut_manipulation", plot_shortcut_manipulation, save_figure
  )
