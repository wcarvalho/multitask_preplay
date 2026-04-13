import os
import re

import jax
import matplotlib.pyplot as plt
import numpy as np

from housemaze import renderer
from housemaze import utils as housemaze_utils
from housemaze.human_dyna import multitask_env
from housemaze.human_dyna import mazes
from data_processing.utils import get_in_episode

import data_configs

image_dict = housemaze_utils.load_image_dict()

# JaxMaze setup (same pattern as jaxmaze_envs.py)
_char2key, _, _ = mazes.get_group_set()


######################################
# String parsing helpers
######################################


def parse_jax_array_string(s: str) -> np.ndarray:
  """Parse a jax/numpy 1D array string like '[0 1 3 2]' back to numpy array."""
  s = s.strip()
  # Remove brackets
  s = s.strip("[]")
  # Split on whitespace
  values = s.split()
  return np.array([float(v) for v in values])


def parse_positions_string(s: str) -> np.ndarray:
  """Parse a jax 2D array string like '[[5 3]\\n [5 4]\\n ...]' to numpy (N, 2)."""
  s = s.strip()
  # Remove outer brackets
  s = s.strip("[]")
  # Split into rows by ][
  rows = re.split(r"\]\s*\[", s)
  result = []
  for row in rows:
    row = row.strip().strip("[]")
    values = row.split()
    result.append([int(v) for v in values])
  return np.array(result)


def _parse_row_field(row, field: str, parser):
  """Get a field from a row (dict or polars row) and parse it."""
  try:
    val = row[field] if isinstance(row, dict) else row[field]
  except (KeyError, IndexError):
    return None
  if val is None or (isinstance(val, str) and val.lower() == "none"):
    return None
  if isinstance(val, str):
    return parser(val)
  return np.asarray(val)


######################################
# JaxMaze image caching
######################################

JAXMAZE_IMAGE_CACHE_DIR = os.path.join(data_configs.CACHE_DIR, "jaxmaze_env_images")


def _get_jaxmaze_rotation(block_name: str):
  """Parse 'reverse(Y=False,X=True)' to (horizontal_bool, vertical_bool)."""
  match = re.match(r"reverse\(Y=(True|False),X=(True|False)\)", block_name)
  if not match:
    return False, False
  vertical = match.group(1) == "True"
  horizontal = match.group(2) == "True"
  return horizontal, vertical


def _sanitize_block_name(block_name: str) -> str:
  """Make block_name safe for filenames."""
  return block_name.replace("(", "").replace(")", "").replace(",", "_").replace("=", "")


def get_jaxmaze_env_image(world: str, block_name: str):
  """Load cached JaxMaze env image, or generate and cache it.

  Returns:
      tuple: (image as numpy array, maze_height, maze_width)
  """
  os.makedirs(JAXMAZE_IMAGE_CACHE_DIR, exist_ok=True)
  sanitized = _sanitize_block_name(block_name)
  cache_path = os.path.join(JAXMAZE_IMAGE_CACHE_DIR, f"{world}_{sanitized}.npy")
  meta_path = os.path.join(JAXMAZE_IMAGE_CACHE_DIR, f"{world}_{sanitized}_meta.npy")

  if os.path.exists(cache_path) and os.path.exists(meta_path):
    image = np.load(cache_path)
    meta = np.load(meta_path)
    return image, int(meta[0]), int(meta[1])

  # Generate image
  maze_str = getattr(mazes, world)
  horizontal, vertical = _get_jaxmaze_rotation(block_name)
  maze_str = housemaze_utils.reverse(maze_str, horizontal, vertical)
  level_init = housemaze_utils.from_str(maze_str, _char2key, return_map_init=False)
  image = renderer.create_image_from_grid(*level_init, image_dict)
  image = np.asarray(image)
  grid = level_init[0]
  maze_height, maze_width = grid.shape[:2]

  np.save(cache_path, image)
  np.save(meta_path, np.array([maze_height, maze_width]))
  return image, maze_height, maze_width


def visualize_jaxmaze_row(
  row,
  ax_image=None,
  ax_rt=None,
  arrow_color="red",
  title=None,
  figsize_per_panel=(5, 5),
):
  """Visualize a JaxMaze DataFrame row: env image with path + optional RT bar chart.

  Args:
      row: A dict-like object (polars row, dict) with columns:
          world, block_name, positions (str), actions (str),
          and optionally reaction_times (str).
      ax_image: Optional matplotlib axis for the image panel.
      ax_rt: Optional matplotlib axis for the RT panel.
      arrow_color: Color for path arrows.
      title: Optional title for the image panel.
      figsize_per_panel: (width, height) per panel.

  Returns:
      (fig, axs) where axs is a list of axes.
  """
  positions = _parse_row_field(row, "positions", parse_positions_string)
  actions = _parse_row_field(row, "actions", parse_jax_array_string)
  if actions is not None:
    actions = actions.astype(int)
  reaction_times = _parse_row_field(row, "reaction_times", parse_jax_array_string)

  has_rt = reaction_times is not None and len(reaction_times) > 0

  # Create figure if axes not provided
  fig = None
  if ax_image is None:
    if has_rt:
      fig, (ax_image, ax_rt) = plt.subplots(
        1, 2, figsize=(figsize_per_panel[0] * 2, figsize_per_panel[1])
      )
    else:
      fig, ax_image = plt.subplots(1, 1, figsize=figsize_per_panel)

  # Get cached env image
  image, maze_height, maze_width = get_jaxmaze_env_image(
    row["world"], row["block_name"]
  )

  # Draw image + arrows
  if positions is not None and actions is not None:
    renderer.place_arrows_on_image(
      image,
      positions,
      actions,
      maze_height,
      maze_width,
      arrow_scale=5,
      ax=ax_image,
      arrow_color=arrow_color,
    )
  else:
    ax_image.imshow(image)
    ax_image.set_xticks([])
    ax_image.set_yticks([])

  ax_image.axis("off")
  if title:
    ax_image.set_title(title)

  # RT panel
  if has_rt and ax_rt is not None:
    ax_rt.bar(range(len(reaction_times)), reaction_times, color="lightblue")
    ax_rt.set_xlabel("Step")
    ax_rt.set_ylabel("Reaction Time (s)")
    ax_rt.set_title("Reaction Times")
    ax_rt.set_ylim(0, max(reaction_times) * 1.1)

  if fig is not None:
    fig.tight_layout()

  axs = [ax_image] + ([ax_rt] if has_rt and ax_rt is not None else [])
  return fig, axs


######################################
# Craftax image caching
######################################

CRAFTAX_IMAGE_CACHE_DIR = os.path.join(data_configs.CACHE_DIR, "craftax_env_images")

# Lazy-loaded craftax environment (expensive to initialize)
_craftax_env = None
_craftax_default_params = None


def _get_craftax_env():
  """Lazily initialize craftax environment (expensive first call)."""
  global _craftax_env, _craftax_default_params
  if _craftax_env is not None:
    return _craftax_env, _craftax_default_params

  from simulations.craftax_web_env import CraftaxSymbolicWebEnvNoAutoReset

  static_env_params = CraftaxSymbolicWebEnvNoAutoReset.default_static_params()
  static_env_params = static_env_params.replace(
    max_melee_mobs=1,
    max_ranged_mobs=1,
    max_passive_mobs=10,
    initial_crafting_tables=True,
    initial_strength=20,
    map_size=(48, 48),
    num_levels=1,
  )
  _craftax_env = CraftaxSymbolicWebEnvNoAutoReset(
    static_env_params=static_env_params,
  )
  _craftax_default_params = _craftax_env.default_params.replace(
    day_length=100000,
    mob_despawn_distance=100000,
  )
  return _craftax_env, _craftax_default_params


def get_craftax_env_image(world_seed: int):
  """Load cached Craftax env image, or generate and cache it.

  First call per world_seed may be slow (~30s) due to JAX compilation.

  Returns:
      tuple: (image as numpy array, maze_height, maze_width)
  """
  os.makedirs(CRAFTAX_IMAGE_CACHE_DIR, exist_ok=True)
  cache_path = os.path.join(CRAFTAX_IMAGE_CACHE_DIR, f"world_seed={world_seed}.npy")
  meta_path = os.path.join(CRAFTAX_IMAGE_CACHE_DIR, f"world_seed={world_seed}_meta.npy")

  if os.path.exists(cache_path) and os.path.exists(meta_path):
    image = np.load(cache_path)
    meta = np.load(meta_path)
    return image, int(meta[0]), int(meta[1])

  # Generate image
  from simulations import craftax_utils

  env, default_params = _get_craftax_env()
  params = default_params.replace(world_seeds=(world_seed,))
  key = jax.random.PRNGKey(0)
  _, state = env.reset(key, params)

  image = craftax_utils.render_fn(state, show_agent=False)
  image = np.asarray(image)
  maze_height = int(state.map.shape[1])
  maze_width = int(state.map.shape[2])

  np.save(cache_path, image)
  np.save(meta_path, np.array([maze_height, maze_width]))
  return image, maze_height, maze_width


def visualize_craftax_row(
  row,
  ax_image=None,
  ax_rt=None,
  arrow_color="red",
  title=None,
  figsize_per_panel=(5, 5),
):
  """Visualize a Craftax DataFrame row: env image with path + optional RT bar chart.

  Args:
      row: A dict-like object with columns:
          world_seed, positions (str), and optionally reaction_times (str).
          actions column is optional (computed from positions if missing).
      ax_image: Optional matplotlib axis for the image panel.
      ax_rt: Optional matplotlib axis for the RT panel.
      arrow_color: Color for path arrows.
      title: Optional title for the image panel.
      figsize_per_panel: (width, height) per panel.

  Returns:
      (fig, axs) where axs is a list of axes.
  """
  from simulations import craftax_utils

  positions = _parse_row_field(row, "positions", parse_positions_string)
  reaction_times = _parse_row_field(row, "reaction_times", parse_jax_array_string)

  has_rt = reaction_times is not None and len(reaction_times) > 0

  # Compute actions from positions (craftax stores positions, derives actions)
  actions = None
  if positions is not None:
    actions = craftax_utils.actions_from_path(positions)

  # Create figure if axes not provided
  fig = None
  if ax_image is None:
    if has_rt:
      fig, (ax_image, ax_rt) = plt.subplots(
        1, 2, figsize=(figsize_per_panel[0] * 2, figsize_per_panel[1])
      )
    else:
      fig, ax_image = plt.subplots(1, 1, figsize=figsize_per_panel)

  # Get cached env image
  world_seed = int(row["world_seed"])
  image, maze_height, maze_width = get_craftax_env_image(world_seed)

  # Draw image + arrows
  if positions is not None and actions is not None:
    craftax_utils.place_arrows_on_image(
      image=image,
      positions=positions,
      actions=actions,
      maze_height=maze_height,
      maze_width=maze_width,
      ax=ax_image,
      arrow_color=arrow_color,
      display_image=True,
      show_path_length=False,
    )
  else:
    ax_image.imshow(image)
    ax_image.set_xticks([])
    ax_image.set_yticks([])

  ax_image.axis("off")
  if title:
    ax_image.set_title(title)

  # RT panel
  if has_rt and ax_rt is not None:
    ax_rt.bar(range(len(reaction_times)), reaction_times, color="lightblue")
    ax_rt.set_xlabel("Step")
    ax_rt.set_ylabel("Reaction Time (s)")
    ax_rt.set_title("Reaction Times")
    ax_rt.set_ylim(0, max(reaction_times) * 1.1)

  if fig is not None:
    fig.tight_layout()

  axs = [ax_image] + ([ax_rt] if has_rt and ax_rt is not None else [])
  return fig, axs


######################################
# Legacy functions (kept for backward compatibility)
######################################


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
    positions = jax.tree_util.tree_map(
      lambda x: x[in_episode][:-1], timesteps.state.agent_pos
    )
  else:
    positions = timesteps.state.agent_pos[:-1]
  # positions in episode

  state_0 = jax.tree_util.tree_map(lambda x: x[0], timesteps.state)

  # doesn't matter
  maze_height, maze_width, _ = timesteps.state.grid[0].shape

  if ax is None:
    fig, ax = plt.subplots(1, figsize=(5, 5))
  img = housemaze_render_fn(state_0)

  renderer.place_arrows_on_image(
    img, positions, actions, maze_height, maze_width, arrow_scale=5, ax=ax
  )


if __name__ == "__main__":
  import polars as pl

  SAVE_DIR = "/tmp/multitask_preplay"
  os.makedirs(SAVE_DIR, exist_ok=True)

  # --- JaxMaze ---
  jm_human_path = os.path.join(
    data_configs.JAXMAZE_MODEL_DATA_DIR, "final", "human_data_episode_df.parquet"
  )
  if os.path.exists(jm_human_path):
    df = pl.read_parquet(jm_human_path)
    # Human sample: pick an eval success row
    sample = df.filter(pl.col("eval") & (pl.col("success") == 1))
    if len(sample) > 0:
      row = sample.row(0, named=True)
      fig, _ = visualize_jaxmaze_row(row, title="JaxMaze Human (eval, success)")
      path = os.path.join(SAVE_DIR, "jaxmaze_human_sample.png")
      fig.savefig(path, bbox_inches="tight", dpi=150)
      plt.close(fig)
      print(f"Saved {path}")

  # JaxMaze model
  for model in ["preplay"]:
    mpath = os.path.join(
      data_configs.JAXMAZE_MODEL_DATA_DIR, "final", f"{model}_episode_df.parquet"
    )
    if not os.path.exists(mpath):
      continue
    mdf = pl.read_parquet(mpath)
    if "positions" not in mdf.columns:
      print(
        f"JaxMaze {model} df missing positions column — re-run process_model_data.py --env jaxmaze --df"
      )
      continue
    sample = mdf.filter(pl.col("eval") & (pl.col("success") == 1))
    if len(sample) > 0:
      row = sample.row(0, named=True)
      fig, _ = visualize_jaxmaze_row(
        row, title=f"JaxMaze Model ({model}, eval, success)"
      )
      path = os.path.join(SAVE_DIR, "jaxmaze_model_sample.png")
      fig.savefig(path, bbox_inches="tight", dpi=150)
      plt.close(fig)
      print(f"Saved {path}")
      break

  # --- Craftax ---
  cx_human_path = os.path.join(
    data_configs.CRAFTAX_MODEL_DATA_DIR, "final", "human_data_episode_df.parquet"
  )
  if os.path.exists(cx_human_path):
    df = pl.read_parquet(cx_human_path)
    sample = df.filter(pl.col("eval") & (pl.col("success") == 1))
    if len(sample) > 0:
      row = sample.row(0, named=True)
      # Craftax uses 'world' column with string values like '3', '15', etc.
      # but visualize_craftax_row expects 'world_seed' (int)
      if "world_seed" not in row or row["world_seed"] is None:
        row = dict(row)
        row["world_seed"] = int(row["world"])
      fig, _ = visualize_craftax_row(row, title="Craftax Human (eval, success)")
      path = os.path.join(SAVE_DIR, "craftax_human_sample.png")
      fig.savefig(path, bbox_inches="tight", dpi=150)
      plt.close(fig)
      print(f"Saved {path}")

  # Craftax model
  for model in ["preplay"]:
    mpath = os.path.join(
      data_configs.CRAFTAX_MODEL_DATA_DIR, "final", f"{model}_episode_df.parquet"
    )
    if not os.path.exists(mpath):
      continue
    mdf = pl.read_parquet(mpath)
    if "positions" not in mdf.columns:
      print(
        f"Craftax {model} df missing positions column — re-run process_model_data.py --env craftax --df"
      )
      continue
    sample = mdf.filter(pl.col("eval") & (pl.col("success") == 1))
    if len(sample) > 0:
      row = sample.row(0, named=True)
      fig, _ = visualize_craftax_row(
        row, title=f"Craftax Model ({model}, eval, success)"
      )
      path = os.path.join(SAVE_DIR, "craftax_model_sample.png")
      fig.savefig(path, bbox_inches="tight", dpi=150)
      plt.close(fig)
      print(f"Saved {path}")
      break
