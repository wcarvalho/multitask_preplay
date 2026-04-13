"""
Pre-generate and cache base world states for all experiment world seeds.

Run this on the JAX version that produces the correct environments (currently 0.4.22
compatible). The cached states are then loaded at runtime, making experiments
reproducible regardless of JAX version.

Usage:
    python experiments/craftax/pregenerate_world_states.py [--verify] [--force]

Options:
    --verify   After saving, reload each state and verify it matches
    --force    Skip JAX version warning
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "simulations"))

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from simulations.craftax_web_env import EnvParams, StaticEnvParams, generate_world
from simulations.craftax_experiment_configs import (
  PRACTICE_BLOCK_CONFIG,
  PATHS_CONFIGS,
  JUNCTURE_CONFIGS,
)
from simulations.craftax_world_cache import (
  save_base_world_state,
  load_base_world_state,
  write_metadata,
  compute_file_hash,
  get_cache_dir,
  get_cache_path,
)


def collect_all_seeds():
  """Collect all unique world seeds from experiment configs."""
  seeds = set()
  seeds.add(PRACTICE_BLOCK_CONFIG.world_seed)
  for config in PATHS_CONFIGS:
    seeds.add(config.world_seed)
  for config in JUNCTURE_CONFIGS:
    seeds.add(config.world_seed)
  return sorted(seeds)


def generate_base_state(seed, static_params):
  """Generate a base world state for a given seed (no goals placed)."""
  params = EnvParams(
    world_seeds=(seed,),
    goal_locations=((-1, -1),),
    placed_goals=(-1,),
    placed_achievements=(-1,),
  )
  rng = jax.random.PRNGKey(seed)
  rng_select = jax.random.PRNGKey(0)
  return generate_world(rng, rng_select, params, static_params)


def verify_roundtrip(seed, static_params):
  """Verify a saved state matches a freshly generated one."""
  path = get_cache_path(seed)
  loaded = load_base_world_state(path, static_params)
  fresh = generate_base_state(seed, static_params)

  loaded_leaves = jtu.tree_leaves(loaded)
  fresh_leaves = jtu.tree_leaves(fresh)

  if len(loaded_leaves) != len(fresh_leaves):
    print(
      f"  FAIL seed {seed}: leaf count mismatch {len(loaded_leaves)} vs {len(fresh_leaves)}"
    )
    return False

  for i, (l, f) in enumerate(zip(loaded_leaves, fresh_leaves)):
    if not np.array_equal(np.asarray(l), np.asarray(f)):
      print(f"  FAIL seed {seed}: mismatch at leaf {i}")
      return False

  return True


def main():
  parser = argparse.ArgumentParser(description="Pre-generate Craftax world states")
  parser.add_argument("--verify", action="store_true", help="Verify after saving")
  parser.add_argument("--force", action="store_true", help="Skip JAX version warning")
  args = parser.parse_args()

  print(f"JAX version: {jax.__version__}")

  if not args.force:
    print(
      "\nWARNING: Make sure you are running this on the JAX version that "
      "produces the correct environments. Use --force to skip this warning."
    )
    response = input("Continue? [y/N] ")
    if response.lower() != "y":
      sys.exit(0)

  seeds = collect_all_seeds()
  print(f"\nWorld seeds to generate: {seeds}")

  static_params = StaticEnvParams()
  cache_dir = get_cache_dir()
  os.makedirs(cache_dir, exist_ok=True)

  file_hashes = {}
  for seed in seeds:
    print(f"  Generating seed {seed}...", end=" ", flush=True)
    state = generate_base_state(seed, static_params)
    path = get_cache_path(seed)
    save_base_world_state(state, path)
    file_hashes[str(seed)] = compute_file_hash(path)
    size_kb = os.path.getsize(path) / 1024
    print(f"saved ({size_kb:.1f} KB)")

  meta_path = write_metadata(cache_dir, seeds, file_hashes)
  print(f"\nMetadata written to {meta_path}")

  if args.verify:
    print("\nVerifying roundtrip...")
    all_ok = True
    for seed in seeds:
      ok = verify_roundtrip(seed, static_params)
      status = "OK" if ok else "FAIL"
      print(f"  seed {seed}: {status}")
      all_ok = all_ok and ok
    if all_ok:
      print("\nAll seeds verified successfully!")
    else:
      print("\nSOME SEEDS FAILED VERIFICATION")
      sys.exit(1)

  total_size = sum(os.path.getsize(get_cache_path(s)) for s in seeds)
  print(f"\nTotal cache size: {total_size / 1024:.1f} KB ({len(seeds)} seeds)")
  print("Done!")


if __name__ == "__main__":
  main()
