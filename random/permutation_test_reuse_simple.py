"""Simple permutation test for path reuse.

Shuffles train-test pairings within each block and recomputes overlap.
Uses the same user filtering as the main analysis scripts.

Usage:
  python analysis/permutation_test_reuse_simple.py --n_permutations 10000
"""

import argparse
import json
import os
import sys
import warnings

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_project_root)

import numpy as np
import polars as pl

import data_configs
from analysis.analysis_utils import positions_from_str
from data_processing.utils import compute_overlap


# ------------------------------------------------------------------
# Grid shape and map helpers (inlined to avoid importing
# permutation_test_reuse.py which uses Python 3.12+ f-string syntax)
# ------------------------------------------------------------------
def _setup_grid_shape_fn(env_name: str):
  """Return a function that gets (H, W) grid shape from the environment."""
  _cache = {}

  if env_name == "jaxmaze":
    try:
      from housemaze import utils
      from housemaze.human_dyna import mazes
    except ImportError:
      raise

    image_dict = utils.load_image_dict()
    image_keys = image_dict["keys"]
    groups = [
      [image_keys.index("orange"), image_keys.index("potato")],
      [image_keys.index("lettuce"), image_keys.index("apple")],
      [image_keys.index("tomato"), image_keys.index("lettuce")],
    ]
    groups = np.array(groups, dtype=np.int32)
    char2key = mazes.groups_to_char2key(groups)

    def get_grid_shape(world_name: str, block_name: str):
      key = (world_name, block_name)
      if key in _cache:
        return _cache[key]
      maze_str_base = getattr(mazes, world_name, None)
      if maze_str_base is None:
        return None
      horizontal = "Y=True" in block_name
      vertical = "X=True" in block_name
      maze_str = utils.reverse(maze_str_base, horizontal, vertical)
      level_init = utils.from_str(maze_str, char2key)
      shape = (level_init.grid.shape[0], level_init.grid.shape[1])
      _cache[key] = shape
      return shape

    return get_grid_shape

  elif env_name == "craftax":
    try:
      sys.path.append(os.path.join(_project_root, "simulations"))
      sys.path.append(os.path.join(_project_root, "experiments", "craftax"))
      import jax
      from simulations.craftax_experiment_configs import (
        PATHS_CONFIGS,
        make_block_env_params,
      )
      from simulations.craftax_web_env import (
        CraftaxSymbolicWebEnvNoAutoReset,
        EnvParams,
      )
    except ImportError:
      raise

    env = CraftaxSymbolicWebEnvNoAutoReset()
    seed_to_config = {}
    for i, cfg in enumerate(PATHS_CONFIGS):
      seed_to_config[str(cfg.world_seed)] = i

    def get_grid_shape(world_seed_str: str):
      if world_seed_str in _cache:
        return _cache[world_seed_str]
      idx = seed_to_config.get(world_seed_str)
      if idx is None:
        return None
      config = PATHS_CONFIGS[idx]
      env_params = make_block_env_params(config, EnvParams())
      _obs, state = env.reset(jax.random.PRNGKey(0), env_params)
      shape = (int(state.map.shape[1]), int(state.map.shape[2]))
      _cache[world_seed_str] = shape
      return shape

    return get_grid_shape

  return None


def positions_to_map(positions: np.ndarray, grid_shape, add_error: bool = False):
  """Convert Nx2 positions into a binary HxW occupancy grid.

  add_error=True expands each cell by +/-1 (used for Craftax training maps).
  """
  grid = np.zeros(grid_shape, dtype=np.int32)
  for pos in positions:
    y, x = int(pos[0]), int(pos[1])
    if add_error:
      for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
          ny, nx = y + dy, x + dx
          if 0 <= ny < grid_shape[0] and 0 <= nx < grid_shape[1]:
            grid[ny, nx] = 1
    else:
      if 0 <= y < grid_shape[0] and 0 <= x < grid_shape[1]:
        grid[y, x] = 1

  return grid


def parse_args():
  parser = argparse.ArgumentParser(description="Simple permutation test for path reuse")
  parser.add_argument(
    "--env",
    choices=["jaxmaze", "craftax", "all"],
    default="all",
    help="Which environment to run (default: all)",
  )
  parser.add_argument("--n_permutations", type=int, default=10_000)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--save_dir", type=str, default=None)
  return parser.parse_args()


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
def load_jaxmaze_df():
  path = os.path.join(
    data_configs.JAXMAZE_MODEL_RAW_DATA_DIR, "final", "human_data_episode_df.parquet"
  )
  return pl.read_parquet(path)


def load_craftax_df():
  path = os.path.join(
    data_configs.CRAFTAX_MODEL_RAW_DATA_DIR, "final", "human_data_episode_df.parquet"
  )
  return pl.read_parquet(path)


# ------------------------------------------------------------------
# Build matched (test_map, train_map) pairs for one filtered df
# ------------------------------------------------------------------
def build_block_pairs(
  full_df: pl.DataFrame,
  eval_df: pl.DataFrame,
  env_name: str,
  grid_shape_fn,
):
  """For each block in eval_df, build matched (test_map, train_map) pairs.

  Returns dict[block_name -> {test_maps, train_maps, observed_overlaps}].
  """
  is_craftax = env_name == "craftax"

  # Build a lookup: global_episode_idx -> positions string
  pos_lookup = dict(
    zip(
      full_df["global_episode_idx"].to_list(),
      full_df["positions"].to_list(),
    )
  )

  # Only successful eval rows with a valid corresponding train episode
  rows = eval_df.filter(
    (
      pl.col("corresponding_train_episode_idx") != -1
      # & (pl.col("success") == 1)
      # & pl.col("positions").is_not_null()
    )
  )

  blocks = {}
  for block_name in rows["block_name"].unique().sort().to_list():
    block_rows = rows.filter(pl.col("block_name") == block_name)

    # Grid shape
    if is_craftax:
      grid_shape = grid_shape_fn(block_name)
    else:
      worlds = block_rows["world"].unique().to_list()
      world_name = worlds[0] if worlds else None
      if world_name is None:
        continue
      grid_shape = grid_shape_fn(world_name, block_name)

    test_maps = []
    train_maps = []
    observed_overlaps = []

    for row in block_rows.iter_rows(named=True):
      test_pos = positions_from_str(row["positions"])
      train_idx = row["corresponding_train_episode_idx"]
      train_pos_str = pos_lookup.get(train_idx)
      if train_pos_str is None:
        continue
      train_pos = positions_from_str(train_pos_str)

      test_map = positions_to_map(test_pos, grid_shape, add_error=False)
      train_map = positions_to_map(train_pos, grid_shape, add_error=is_craftax)

      test_maps.append(test_map)
      train_maps.append(train_map)
      observed_overlaps.append(row["overlap"])

    if len(test_maps) == 0:
      continue

    blocks[block_name] = {
      "test_maps": np.array(test_maps),
      "train_maps": np.array(train_maps),
      "observed_overlaps": np.array(observed_overlaps, dtype=float),
    }

  return blocks


# ------------------------------------------------------------------
# Permutation test for one block
# ------------------------------------------------------------------
def permutation_test_block(test_maps, train_maps, n_permutations, rng):
  """Shuffle train maps across test maps, recompute mean overlap."""
  n = len(test_maps)
  null_means = np.empty(n_permutations)
  train_shuffled = train_maps.copy()
  for p in range(n_permutations):
    rng.shuffle(train_shuffled)
    total = 0.0
    for i in range(n):
      total += compute_overlap(train_shuffled[i], test_maps[i])
    null_means[p] = total / n
  return null_means


# ------------------------------------------------------------------
# Run one experiment
# ------------------------------------------------------------------
def run_experiment(name, block_pairs, n_permutations, rng):
  """Run permutation test across all blocks, return per-block null distributions."""
  print(f"\n{'=' * 60}")
  print(f"  {name}")
  print(f"{'=' * 60}")

  block_null_dists = []  # (n_pairs, n_permutations) per block
  block_n_pairs = []
  block_names = []

  for block_name, data in sorted(block_pairs.items()):
    null_dist = permutation_test_block(
      data["test_maps"],
      data["train_maps"],
      n_permutations,
      rng,
    )
    n = len(data["test_maps"])
    print(
      f"  block={block_name}: "
      f"null_mean={null_dist.mean():.4f}±{null_dist.std():.4f}, "
      f"n_pairs={n}"
    )
    block_null_dists.append(null_dist)
    block_n_pairs.append(n)
    block_names.append(block_name)

  # Weighted average across blocks (weight = n_pairs)
  weights = np.array(block_n_pairs, dtype=float)
  weights /= weights.sum()
  # (n_blocks, n_perms) -> weighted mean -> (n_perms,)
  global_null = np.average(np.stack(block_null_dists), axis=0, weights=weights)

  print(
    f"\n  GLOBAL (weighted): "
    f"null_mean={global_null.mean():.4f}±{global_null.std():.4f}, "
    f"n_total={sum(block_n_pairs)}"
  )

  return {
    "blocks": {
      bname: {"null_distribution": ndist, "n_pairs": np}
      for bname, ndist, np in zip(block_names, block_null_dists, block_n_pairs)
    },
    "global_null_distribution": global_null,
    "n_total": sum(block_n_pairs),
  }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
  args = parse_args()
  rng = np.random.default_rng(args.seed)

  save_dir = args.save_dir or os.path.join(
    data_configs.ANALYSIS_FIGURES_DIR, "permutation_test"
  )
  os.makedirs(save_dir, exist_ok=True)

  all_results = {}
  run_jaxmaze = args.env in ("jaxmaze", "all")
  run_craftax = args.env in ("craftax", "all")

  # ---- JaxMaze ----
  if run_jaxmaze:
    print("Loading JaxMaze data...")
    jm_df = load_jaxmaze_df()
    print(f"  {len(jm_df)} rows, {jm_df['user_id'].n_unique()} users")

    from analysis.jaxmaze_analysis import filter_users_by_success

    jm_grid_fn = _setup_grid_shape_fn("jaxmaze")

    # Paths manipulation
    paths_df = jm_df.filter(
      manipulation="paths",
      world="big_m3_maze1",
      eval_shares_start_pos=True,
    )
    paths_df, _ = filter_users_by_success(paths_df, analysis_name="path_reuse_results")
    paths_eval = paths_df.filter(eval=True)
    paths_blocks = build_block_pairs(jm_df, paths_eval, "jaxmaze", jm_grid_fn)
    all_results["jaxmaze_paths"] = run_experiment(
      "JaxMaze — Paths", paths_blocks, args.n_permutations, rng
    )

    # Shortcut manipulation
    short_df = jm_df.filter(
      manipulation="shortcut",
      eval_shares_start_pos=True,
    )
    short_df, _ = filter_users_by_success(short_df, analysis_name="shortcut_results")
    short_eval = short_df.filter(eval=True)
    short_blocks = build_block_pairs(jm_df, short_eval, "jaxmaze", jm_grid_fn)
    all_results["jaxmaze_shortcut"] = run_experiment(
      "JaxMaze — Shortcut", short_blocks, args.n_permutations, rng
    )

  # ---- Craftax ----
  if run_craftax:
    print("\nLoading Craftax data...")
    cx_df = load_craftax_df()
    print(f"  {len(cx_df)} rows, {cx_df['user_id'].n_unique()} users")

    from analysis.analysis_utils import filter_users_by_success_by_tell_reuse

    cx_grid_fn = _setup_grid_shape_fn("craftax")

    cx_filtered = cx_df.filter(eval_shares_start_pos=True)
    cx_filtered, _ = filter_users_by_success_by_tell_reuse(
      cx_filtered, analysis_name="permutation_test_simple"
    )

    for tell_reuse in [1, 0]:
      label = "known" if tell_reuse == 1 else "unknown"
      cx_eval = cx_filtered.filter(eval=True, tell_reuse=tell_reuse)
      cx_blocks = build_block_pairs(cx_df, cx_eval, "craftax", cx_grid_fn)
      all_results[f"craftax_{label}"] = run_experiment(
        f"Craftax — {label} eval goals", cx_blocks, args.n_permutations, rng
      )

  # ---- Save results (one file per experiment) ----
  for exp_name, res in all_results.items():
    global_null = res["global_null_distribution"]
    summary = {
      "global_null_mean": float(global_null.mean()),
      "global_null_std": float(global_null.std()),
      "n_total": res["n_total"],
      "blocks": {},
    }
    for block_name, bres in res["blocks"].items():
      null_dist = bres["null_distribution"]
      summary["blocks"][block_name] = {
        "null_mean": float(null_dist.mean()),
        "null_std": float(null_dist.std()),
        "n_pairs": bres["n_pairs"],
      }
    # summary["global_null_distribution"] = global_null.tolist()
    json_path = os.path.join(save_dir, f"results_simple_{exp_name}.json")
    with open(json_path, "w") as f:
      json.dump(summary, f, indent=2)
    print(f"Saved results → {json_path}")
