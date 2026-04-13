import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from housemaze import utils
from housemaze.human_dyna import mazes

from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from data_configs import JAXMAZE_ENV_FIGURES_DIR
from plot_configs import default_colors
from figures_supplemental.jaxmaze_envs import render_path, save_figure

TRAIN_COLOR = default_colors["sky blue"]
EVAL_COLOR = "red"
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


if __name__ == "__main__":
  directory = sys.argv[1] if len(sys.argv) > 1 else JAXMAZE_ENV_FIGURES_DIR
  os.makedirs(directory, exist_ok=True)
  save_figure = partial(save_figure, directory=directory)

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
  save_figure(fig, "1.two_paths_manipulation")

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
  save_figure(fig, "2.juncture_manipulation")

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
  save_figure(fig, "3.start_manipulation")

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
  save_figure(fig, "4.shortcut_manipulation")
