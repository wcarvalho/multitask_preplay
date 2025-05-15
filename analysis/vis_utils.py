import jax
import matplotlib.pyplot as plt

from housemaze import renderer
from housemaze.human_dyna import multitask_env
from data_processing.utils import get_in_episode
from housemaze.human_dyna import utils as housemaze_utils

image_dict = housemaze_utils.load_image_dict()


def housemaze_render_fn(state: multitask_env.EnvState):
  return renderer.create_image_from_grid(
    state.grid, state.agent_pos, state.agent_dir, image_dict
  )


def render_path(episode_data, from_model=True, ax=None):
  # get actions that are in episode
  timesteps = episode_data.timesteps
  actions = episode_data.actions
  if from_model:
    in_episode = get_in_episode(timesteps)
    actions = actions[in_episode][:-1]
    positions = jax.tree_map(lambda x: x[in_episode][:-1], timesteps.state.agent_pos)
  else:
    positions = timesteps.state.agent_pos[:-1]
  # positions in episode

  state_0 = jax.tree_map(lambda x: x[0], timesteps.state)

  # doesn't matter
  maze_height, maze_width, _ = timesteps.state.grid[0].shape

  if ax is None:
    fig, ax = plt.subplots(1, figsize=(5, 5))
  img = housemaze_render_fn(state_0)

  renderer.place_arrows_on_image(
    img, positions, actions, maze_height, maze_width, arrow_scale=5, ax=ax
  )
