"""
Memory profiling: measure impact of reduce_timestep_size + stripping transitions.
"""
import os, sys
sys.path.insert(0, ".")
sys.path.append("simulations")

import jax
import jax.numpy as jnp
import numpy as np

from data_processing.process_model_data import (
    load_craftax_environment,
    load_algorithm_ckpt_params,
    load_qlearning_craftax_algorithm,
)
from data_processing.utils_craftax import (
    generate_algorithm_episodes,
    reduce_timestep_size,
)
from data_processing.utils import get_in_episode
from simulations import craftax_simulation_configs
import pickle

def tree_size_mb(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    total = sum(np.prod(l.shape) * l.dtype.itemsize for l in leaves if hasattr(l, 'shape'))
    return total / 1024 / 1024

# Load env + model
env, example_timestep, example_env_params = load_craftax_environment()
path = "/Users/wilka/git/research/preplay_results/data/craftax/qlearning/seed=1"
params = load_algorithm_ckpt_params(f"{path}/qlearning.safetensors")
with open(f"{path}/qlearning.config", "rb") as f:
    config = pickle.load(f)
algorithm = load_qlearning_craftax_algorithm(
    config=config, agent_params=params, env=env,
    example_env_params=example_env_params, path=path,
)

# Run eval on first config
first_config = jax.tree_util.tree_map(lambda x: x[0:1], craftax_simulation_configs.TRAIN_EVAL_CONFIGS)
env_params = craftax_simulation_configs.make_multigoal_env_params(first_config)
rng = jax.random.PRNGKey(1)
episodes = algorithm.eval_fn(rng, env_params)

# Trim one episode
episode = jax.tree_util.tree_map(lambda x: x[0], episodes)
in_episode = get_in_episode(episode.timesteps)
episode = jax.tree_util.tree_map(lambda x: x[in_episode], episode)

print("=== Single trimmed episode (in_episode only) ===")
print(f"  Full episode:        {tree_size_mb(episode):.2f} MB")
print(f"  .timesteps:          {tree_size_mb(episode.timesteps):.2f} MB")
print(f"  .transitions:        {tree_size_mb(episode.transitions):.2f} MB")
print(f"  .actions:            {tree_size_mb(episode.actions):.4f} MB")
print(f"  .positions:          {tree_size_mb(episode.positions):.4f} MB")

# Strip transitions
episode_no_trans = episode._replace(transitions=None)
print(f"\n  No transitions:      {tree_size_mb(episode_no_trans):.2f} MB")

# Apply reduce_timestep_size
reduced_timesteps = reduce_timestep_size(episode.timesteps)
episode_reduced = episode._replace(timesteps=reduced_timesteps, transitions=None)
print(f"  No trans + reduced:  {tree_size_mb(episode_reduced):.2f} MB")

# Breakdown of timestep components
ts = episode.timesteps
print(f"\n=== Timestep breakdown (before reduce) ===")
print(f"  .state.map:          {tree_size_mb(ts.state.map):.2f} MB  shape={ts.state.map.shape} dtype={ts.state.map.dtype}")
print(f"  .state.item_map:     {tree_size_mb(ts.state.item_map):.2f} MB  shape={ts.state.item_map.shape}")
print(f"  .state.mob_map:      {tree_size_mb(ts.state.mob_map):.2f} MB  shape={ts.state.mob_map.shape}")
print(f"  .state.light_map:    {tree_size_mb(ts.state.light_map):.2f} MB  shape={ts.state.light_map.shape}")
if ts.observation is not None:
    print(f"  .observation:        {tree_size_mb(ts.observation):.2f} MB")

rts = reduced_timesteps
print(f"\n=== Timestep breakdown (after reduce) ===")
print(f"  .state.map:          {tree_size_mb(rts.state.map):.2f} MB  shape={rts.state.map.shape} dtype={rts.state.map.dtype}")
print(f"  .state.item_map:     {tree_size_mb(rts.state.item_map):.2f} MB")
print(f"  .state.mob_map:      {tree_size_mb(rts.state.mob_map):.2f} MB")
print(f"  .state.light_map:    {tree_size_mb(rts.state.light_map):.2f} MB")

# Projections
print(f"\n=== Projections (no transitions + reduce_timestep_size) ===")
reduced_mb = tree_size_mb(episode_reduced)
print(f"  1 episode:           {reduced_mb:.2f} MB")
print(f"  1 config (10 eps):   {reduced_mb * 10:.1f} MB")
print(f"  8 configs (80 eps):  {reduced_mb * 80:.1f} MB")
print(f"  10 seeds (800 eps):  {reduced_mb * 800:.1f} MB")

# Also run full seed to get actual numbers
print(f"\n=== Running full seed generation ===")
rng = jax.random.PRNGKey(1)
extras = {"seed": 1}
all_episodes, all_configs = generate_algorithm_episodes(algorithm, rng, extras)
print(f"  Episodes generated: {len(all_episodes)}")
total_mb = sum(tree_size_mb(e) for e in all_episodes)
print(f"  Total size (transitions already stripped): {total_mb:.1f} MB")

# Now apply reduce_timestep_size to all episodes
reduced_episodes = [e._replace(timesteps=reduce_timestep_size(e.timesteps)) for e in all_episodes]
total_reduced_mb = sum(tree_size_mb(e) for e in reduced_episodes)
print(f"  Total size (+ reduce_timestep_size):       {total_reduced_mb:.1f} MB")
print(f"  10 seeds projected:                        {total_reduced_mb * 10:.1f} MB")
