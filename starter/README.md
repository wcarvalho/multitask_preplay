# starter/

Minimal, runnable starting points for working with the Craftax human experiment data and environment. Each script stands alone, runs from the repo root, and prints/plots enough to confirm it worked.

## Scripts

- `craftax_human_data.py` — Loads the Craftax human dataframe (`dataframes/craftax_human.parquet`), splits it into train (`eval=False`) and test (`eval=True`), prints the first 5 rows of each, and plots 3 matched train/test trajectory pairs (each column is the same person in the same world; train on top, test on bottom) with the start marked by a yellow star and the end by a yellow circle. Output: `starter/output/craftax_human_data.png`.
- `craftax_human_env.py` — Loads the train/test conditions, instantiates the actual Craftax environment in each condition, and steps it. `--policy=random` takes random actions; `--policy=human` picks a random participant and replays their recorded actions, verifying that the environment's rolled-out positions match the recorded path. It also renders a 2×5 figure per condition (top row = first 5 rollout states, bottom row = last 5) as the **first-person Craftax view the participant saw**, to `starter/output/craftax_human_env_{train,test}_{policy}.png` — a visual confirmation that the world loaded correctly. It builds the env via `make_human_experiment_env()` (in `simulations/craftax_web_env.py`) — the single canonical builder also used by the data-collection experiment (`experiments/craftax/craftax_experiment_structure.py`), so the starter and the real experiment construct the exact same env.
- `check_world_reproducibility.py` — Verifies the env's world is identical across JAX versions (see below).

## JAX-version reproducibility — why the env loads a cached world

`generate_world()` uses the JAX PRNG, whose output drifts across JAX versions, so resetting the env from a world seed under a future JAX would spawn a *different* map and silently break human-action replay. The env avoids this by loading a frozen world-state cache (`simulations/craftax_cache/world_states/seed_*.npz`, generated on JAX 0.4.22): `make_human_experiment_env(with_cache=True)` populates `static_env_params.cached_world_states`, and the env's `reset_env` then restores the cached world for any cached seed on **every** reset instead of regenerating it. This machinery pre-exists in the repo (`simulations/craftax_world_cache.py`); the builder just populates it.

Verified end-to-end with `check_world_reproducibility.py`: dumping every `EnvState` leaf after reset under **JAX 0.4.22 (pinned) vs JAX 0.5.3 and 0.7.0** (both have `threefry_partitionable=True`, so their PRNG genuinely differs from 0.4.22) gives, *with the cache*, byte-for-byte identical worlds (all 352 leaves across all 4 seeds); *without the cache*, the regenerated world drifts badly (the `.map` terrain alone differs by ~11k cells per seed). Run it yourself: `python starter/check_world_reproducibility.py dump --out a.npz` under each JAX version, then `... compare a.npz b.npz` (the script's docstring has the sandbox-venv recipe).

Note: JAX 0.7.0 is roughly the upper bound for this stack — `craftax 1.4.4` itself fails to import under the newest JAX (0.10.2) because its bundled texture-cache pickle was serialized with an old JAX (`ShapedArray` dropped the `named_shape` kwarg). That is a craftax limitation unrelated to the world cache; the cache reproducibility holds across every JAX version where the env can be built at all.

## Identifiers and privacy — read this

`user_id` is an experiment-internal identifier that is **randomly generated and completely independent of the participant's CloudResearch ID** (the `worker_id` column). You cannot infer, reverse, or link a participant's CloudResearch / worker identity from a `user_id` — the number carries no information about who the person is. It is assigned per experiment session, so the same CloudResearch worker who starts a new session receives a new, unrelated `user_id` (this is why the dataset has more distinct `user_id`s than `worker_id`s). Use `user_id` freely for grouping, plotting, and analysis; it is the safe, anonymous key for "which participant."
