"""
Diagnostic script to find NaN overlap cases in processed model data
and generate plots showing the train path, test path, and overlap binary matrix.

Usage:
    python random/find_nan_overlap.py --env jaxmaze
    python random/find_nan_overlap.py --env craftax
    python random/find_nan_overlap.py  # both environments
"""

import argparse
import os
import sys

sys.path.insert(0, ".")
sys.path.append("simulations")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import nicewebrl

from data_processing.utils import compute_overlap, get_in_episode


def find_nan_episodes(df: pl.DataFrame) -> pl.DataFrame:
  """Filter to rows where overlap is NaN."""
  return df.filter(pl.col("overlap").is_nan() & pl.col("eval"))


def get_corresponding_train_episode(
  full_df: nicewebrl.DataFrame,
  nan_row: dict,
  env_name: str,
):
  """Find the corresponding train episode for a NaN test episode.

  First checks corresponding_train_episode_idx column.
  Falls back to matching by world, block_name, task_set=0, eval=False, success=1.
  """
  train_idx = nan_row.get("corresponding_train_episode_idx", -1)
  if train_idx is not None and train_idx >= 0:
    train_rows = full_df.filter(global_episode_idx=train_idx)
    if len(train_rows) > 0:
      return train_rows.episodes[0], "matched_by_idx"

  # Fallback: find a successful train episode with same world/block
  filter_kwargs = dict(
    world=nan_row["world"],
    block_name=nan_row["block_name"],
    task_set=0,
    eval=False,
    success=1,
  )
  # For model data, also match user_id (seed)
  if "user_id" in nan_row:
    filter_kwargs["user_id"] = nan_row["user_id"]

  train_rows = full_df.filter(**filter_kwargs)
  if len(train_rows) > 0:
    return train_rows.episodes[0], "matched_by_query"

  return None, "no_match"


def create_overlap_visualization(
  train_map, test_map, train_episode, test_episode, nan_row, env_name, save_path
):
  """Create 3-panel visualization: train path, test path, overlap matrix."""
  fig, axes = plt.subplots(1, 3, figsize=(15, 5))

  algo = nan_row.get("algo", "unknown")
  seed = nan_row.get("user_id", "?")
  world = nan_row.get("world", "?")
  episode_idx = nan_row.get("global_episode_idx", "?")

  # Panel 1: Train path
  ax = axes[0]
  if train_episode is not None:
    if env_name == "jaxmaze":
      _render_jaxmaze_path(train_episode, ax=ax, from_model=True)
    else:
      _render_craftax_path(train_episode, ax=ax)
    ax.set_title("Train path")
  else:
    ax.text(
      0.5,
      0.5,
      "No matching\ntrain episode",
      ha="center",
      va="center",
      fontsize=14,
      transform=ax.transAxes,
    )
    ax.set_title("Train path (missing)")

  # Panel 2: Test path
  ax = axes[1]
  if env_name == "jaxmaze":
    _render_jaxmaze_path(test_episode, ax=ax, from_model=True)
  else:
    _render_craftax_path(test_episode, ax=ax)
  ax.set_title("Test path")

  # Panel 3: Overlap binary matrix
  ax = axes[2]
  if train_map is not None and test_map is not None:
    # Combined map: 0=empty, 1=train-only, 2=test-only, 3=shared
    combined = np.zeros_like(train_map, dtype=np.int32)
    combined += np.array(train_map > 0, dtype=np.int32) * 1  # train = 1
    combined += np.array(test_map > 0, dtype=np.int32) * 2  # test = 2, shared = 3

    cmap = plt.cm.colors.ListedColormap(["white", "blue", "red", "purple"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(combined, cmap=cmap, norm=norm, interpolation="nearest")

    map2_length = int((np.array(test_map) > 0).sum())
    overlap_val = compute_overlap(np.array(train_map), np.array(test_map))
    ax.set_title(f"Overlap matrix\ntest_cells={map2_length}, overlap={overlap_val:.3f}")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
      Patch(facecolor="blue", label="Train only"),
      Patch(facecolor="red", label="Test only"),
      Patch(facecolor="purple", label="Shared"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
  elif test_map is not None:
    ax.imshow(np.array(test_map), cmap="Reds", interpolation="nearest")
    map2_length = int((np.array(test_map) > 0).sum())
    ax.set_title(f"Test map only\ntest_cells={map2_length}")
  else:
    ax.text(
      0.5,
      0.5,
      "No maps available",
      ha="center",
      va="center",
      fontsize=14,
      transform=ax.transAxes,
    )
    ax.set_title("Overlap matrix")

  fig.suptitle(
    f"{env_name} | algo={algo} | seed={seed} | world={world} | episode={episode_idx}",
    fontsize=12,
  )
  plt.tight_layout()
  plt.savefig(save_path, dpi=100, bbox_inches="tight")
  plt.close(fig)


def _render_jaxmaze_path(episode_data, ax, from_model=True):
  """Render a JaxMaze episode path on the given axes."""
  from analysis.vis_utils import render_path

  render_path(episode_data, from_model=from_model, ax=ax)


def _render_craftax_path(episode_data, ax):
  """Render a Craftax episode path on the given axes."""
  from simulations.craftax_utils import render_fn, place_arrows_on_image

  timesteps = episode_data.timesteps
  actions = episode_data.actions

  in_episode = get_in_episode(timesteps)
  actions_in_ep = actions[in_episode][:-1]
  positions = jax.tree_util.tree_map(
    lambda x: x[in_episode][:-1], timesteps.state.player_position
  )

  state_0 = jax.tree_util.tree_map(lambda x: x[0], timesteps.state)
  image = render_fn(state_0)
  image = np.array(image)

  maze_height, maze_width = timesteps.state.map.shape[2:]

  place_arrows_on_image(
    image, positions, actions_in_ep, maze_height, maze_width, arrow_scale=5, ax=ax
  )


def process_env(env_name: str, max_plots_per_algo: int = 50, algo: str = None):
  """Process NaN overlap cases for a given environment."""
  from data_processing import utils_jaxmaze, utils_craftax

  print(f"\n{'=' * 60}")
  print(f"Processing {env_name}")
  print(f"{'=' * 60}")

  # Load model data with episodes
  if env_name == "jaxmaze":
    from data_processing.process_model_data import get_jaxmaze_model_data

    model_df = get_jaxmaze_model_data(load_df_only=False)
    create_train_maps_fn = utils_jaxmaze.create_maps
    create_test_maps_fn = utils_jaxmaze.create_maps
  elif env_name == "craftax":
    from data_processing.process_model_data import get_craftax_model_data
    from functools import partial

    model_df = get_craftax_model_data(load_df_only=False)
    create_train_maps_fn = partial(utils_craftax.create_maps, add_error=True)
    create_test_maps_fn = partial(utils_craftax.create_maps, add_error=False)
  else:
    raise ValueError(f"Unknown env: {env_name}")

  # Find NaN overlap rows
  nan_df = find_nan_episodes(model_df._df)
  print(f"\nTotal NaN overlap episodes: {len(nan_df)}")

  if len(nan_df) == 0:
    print("No NaN overlap cases found!")
    return

  # Summary by algo and seed
  print("\nNaN counts by algo:")
  print(nan_df.group_by("algo").len().sort("algo"))
  print("\nNaN counts by algo and seed (user_id):")
  print(nan_df.group_by(["algo", "user_id"]).len().sort(["algo", "user_id"]))

  # Categorize NaN reasons
  has_train_match = nan_df.filter(pl.col("corresponding_train_episode_idx") >= 0)
  no_train_match = nan_df.filter(pl.col("corresponding_train_episode_idx") < 0)
  print(f"\nNaN with train match (empty test path): {len(has_train_match)}")
  print(f"NaN without train match (no successful train): {len(no_train_match)}")

  # Create output directory
  save_dir = os.path.join("random", "plots", "overlap_nans", env_name)
  os.makedirs(save_dir, exist_ok=True)

  # Process every algo separately, with max_plots_per_algo limit each
  all_algos = nan_df["algo"].unique().sort().to_list()
  if algo is not None:
    if algo not in all_algos:
      print(f"Algo '{algo}' has no NaN cases (available: {all_algos})")
      return
    all_algos = [algo]
  total_count = 0

  for algo in all_algos:
    algo_nan_df = nan_df.filter(pl.col("algo") == algo)
    print(f"\n--- {algo}: {len(algo_nan_df)} NaN cases ---")

    algo_dir = os.path.join(save_dir, algo)
    os.makedirs(algo_dir, exist_ok=True)

    count = 0
    for row in algo_nan_df.iter_rows(named=True):
      if count >= max_plots_per_algo:
        print(
          f"  Reached max_plots_per_algo={max_plots_per_algo} for {algo}, stopping."
        )
        break

      seed = row.get("user_id", "unknown")
      world = row.get("world", "unknown")
      episode_idx = row.get("global_episode_idx", 0)

      # Get the test episode
      test_rows = model_df.filter(global_episode_idx=episode_idx)
      if len(test_rows) == 0:
        print(f"  Warning: could not find episode {episode_idx}, skipping")
        continue
      test_episode = test_rows.episodes[0]

      # Get corresponding train episode
      train_episode, match_type = get_corresponding_train_episode(
        model_df, row, env_name
      )

      # Create maps
      test_map = create_test_maps_fn([test_episode])[0]
      train_map = None
      if train_episode is not None:
        train_map = create_train_maps_fn([train_episode])[0]

      # Save plot
      filename = f"{algo}_seed{seed}_{world}_{episode_idx}.png"
      save_path = os.path.join(algo_dir, filename)

      create_overlap_visualization(
        train_map=train_map,
        test_map=test_map,
        train_episode=train_episode,
        test_episode=test_episode,
        nan_row=row,
        env_name=env_name,
        save_path=save_path,
      )
      count += 1
      if count % 10 == 0:
        print(f"  Generated {count} plots for {algo}...")

    print(f"  Saved {count} plots to {algo_dir}/")
    total_count += count

  print(
    f"\nTotal: saved {total_count} plots across {len(all_algos)} algos to {save_dir}/"
  )


def main():
  parser = argparse.ArgumentParser(description="Find NaN overlap cases in model data")
  parser.add_argument(
    "--env",
    type=str,
    choices=["jaxmaze", "craftax"],
    default=None,
    help="Environment to process (default: both)",
  )
  parser.add_argument(
    "--max-plots-per-algo",
    type=int,
    default=50,
    help="Maximum number of plots per algo per environment",
  )
  parser.add_argument(
    "--algo",
    type=str,
    default=None,
    help="Only process this algo (e.g. qlearning, usfa, preplay, dyna, bfs, dfs)",
  )
  args = parser.parse_args()

  envs = [args.env] if args.env else ["jaxmaze", "craftax"]
  for env_name in envs:
    process_env(env_name, max_plots_per_algo=args.max_plots_per_algo, algo=args.algo)


if __name__ == "__main__":
  main()
