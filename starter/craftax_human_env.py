"""Load Craftax human conditions, replay them in the real env, and render the views.

Loads a train (``eval=False``) and test (``eval=True``) condition from the human
dataframe, instantiates the experiment's Craftax env in each, steps it under the
chosen policy, and saves a 2x5 figure of the first/last 5 rollout states.

  * ``--policy=random`` : take random movement actions for up to ``--max-steps``.
  * ``--policy=human``  : replay a recorded human episode and check that the env
    positions match the recording step-for-step.

Run from the repo root::

    python starter/craftax_human_env.py --policy=random
    python starter/craftax_human_env.py --policy=human

The first env build compiles JAX kernels and may take ~30s.
"""

import argparse
import os
import subprocess
import sys

# Put repo root and simulations/ on sys.path so this runs from the repo root.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_project_root)
sys.path.append(os.path.join(_project_root, "simulations"))

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import matplotlib

matplotlib.use("Agg")  # save PNG, no display
import matplotlib.pyplot as plt

import data_configs
from analysis.download_dataframes import download_craftax_data

# make_human_experiment_env loads the frozen world-state cache, so reset()
# restores the cached world instead of regenerating it from the JAX PRNG.
from simulations.craftax_web_env import make_human_experiment_env
from analysis.vis_utils import parse_positions_string, parse_jax_array_string
from simulations import craftax_utils
from craftax.craftax.constants import Action

ACTION_NOOP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_UP = 3
ACTION_DOWN = 4
MOVEMENT_ACTIONS = np.array([ACTION_LEFT, ACTION_RIGHT, ACTION_UP, ACTION_DOWN])

# Recorded `actions` are web-app indices; remap to raw Craftax enum values
# before env.step. web idx -> enum: [RIGHT, DOWN, LEFT, UP, DO] = [2, 4, 1, 3, 5].
WEB_TO_ENV_ACTION = np.array(
  [
    Action.RIGHT.value,
    Action.DOWN.value,
    Action.LEFT.value,
    Action.UP.value,
    Action.DO.value,
  ],
  dtype=np.int32,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


# --------------------------------------------------------------------------- #
# Conditions
# --------------------------------------------------------------------------- #
def load_conditions(df: pl.DataFrame, seed: int):
  """Pick one random episode for each of the train and test conditions.

  Returns a list of (eval_flag, row_dict) in train-then-test order. Prefers
  successful episodes with parseable ``positions`` of length >= 2.
  """
  rng = np.random.default_rng(seed)
  conditions = []
  for eval_flag in (False, True):
    success_first = df.filter(eval=eval_flag, success=1.0)
    fallback = df.filter(eval=eval_flag)
    chosen = None
    for sub in (success_first, fallback):
      if len(sub) == 0:
        continue
      for idx in rng.permutation(len(sub)):
        row = sub.row(int(idx), named=True)
        positions = parse_positions_string(row["positions"])
        if positions is not None and len(positions) >= 2:
          chosen = row
          break
      if chosen is not None:
        break
    if chosen is None:
      raise RuntimeError(f"No usable episode found for eval={eval_flag}")
    conditions.append((eval_flag, chosen))
  return conditions


# --------------------------------------------------------------------------- #
# Env instantiation
# --------------------------------------------------------------------------- #
def make_env_state(env, default_params, world_seed: int, start_pos, key):
  """Reset the env in the given world and override the start position."""
  params = default_params.replace(world_seeds=(int(world_seed),))
  _, state = env.reset(key, params)
  state = state.replace(player_position=jnp.asarray(start_pos, dtype=jnp.int32))
  return params, state


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #
def run_random(env_step, state, params, key, max_steps: int):
  """Step the env with uniformly-random movement actions.

  Returns a dict summary: positions visited, total reward, steps taken,
  whether a terminal reward fired.
  """
  rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**30)))
  positions = [np.asarray(state.player_position)]
  states = [state]
  total_reward = 0.0
  goal_reached = False
  steps = 0
  step_key = key
  for _ in range(max_steps):
    action = int(rng.choice(MOVEMENT_ACTIONS))
    step_key, sub = jax.random.split(step_key)
    _, state, reward, done, _ = env_step(sub, state, jnp.int32(action), params)
    reward = float(reward)
    total_reward += reward
    positions.append(np.asarray(state.player_position))
    states.append(state)
    steps += 1
    if reward > 0.5:
      goal_reached = True
    if bool(done):
      break
  return {
    "steps": steps,
    "total_reward": total_reward,
    "goal_reached": goal_reached,
    "positions": np.asarray(positions),
    "states": states,
  }


def run_human(env_step, state, params, recorded_positions, recorded_actions=None):
  """Replay a human episode and check object collection.

  Uses ``recorded_actions`` (full behaviour incl. DO, so objects get collected)
  when given, else derives movement-only actions from ``positions``.
  """
  if recorded_actions is not None:
    actions = WEB_TO_ENV_ACTION[np.asarray(recorded_actions, dtype=np.int32)]
    action_source = "recorded"
  else:
    actions = np.asarray(craftax_utils.actions_from_path(recorded_positions))
    action_source = "positions"

  map_before = np.asarray(state.map[0])
  rolled_out = [np.asarray(state.player_position)]
  states = [state]
  total_reward = 0.0
  goal_reached = False
  step_key = jax.random.PRNGKey(0)
  for action in actions:
    step_key, sub = jax.random.split(step_key)
    _, state, reward, done, _ = env_step(sub, state, jnp.int32(int(action)), params)
    reward = float(reward)
    total_reward += reward
    rolled_out.append(np.asarray(state.player_position))
    states.append(state)
    if reward > 0.5:
      goal_reached = True
    if bool(done):
      break

  rolled = np.asarray(rolled_out)
  rec = np.asarray(recorded_positions)
  k = min(len(rolled), len(rec))
  pos_matches = int(np.all(rolled[:k] == rec[:k], axis=1).sum())

  # cells whose block changed (collected by the agent)
  map_after = np.asarray(state.map[0])
  yx = np.argwhere(map_before != map_after)
  removed = [
    (int(y), int(x), int(map_before[y, x]), int(map_after[y, x])) for y, x in yx
  ]
  return {
    "steps": len(actions),
    "total_reward": total_reward,
    "goal_reached": goal_reached,
    "action_source": action_source,
    "pos_matches": pos_matches,
    "pos_total": k,
    "n_positions": len(recorded_positions),
    "removed_blocks": removed,
    "actions": actions,
    "rolled_out": rolled,
    "states": states,
  }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _summarize(eval_flag, row, policy, result):
  cond = "TEST (eval=True)" if eval_flag else "TRAIN (eval=False)"
  start_pos = parse_positions_string(row["positions"])[0].tolist()
  print(f"\n=== {cond} | world seed {row['world']} ===")
  print(f"  start position : {start_pos}")
  print(f"  goal object id : {row['task_object_id']}")
  print(f"  policy         : {policy}")
  print(f"  steps taken    : {result['steps']}")
  print(f"  total reward   : {result['total_reward']:.3f}")
  print(f"  goal reached   : {result['goal_reached']}")
  if policy == "human":
    print(f"  recorded path  : {result['n_positions']} positions")
    print(f"  action source  : {result['action_source']}")
    print(
      f"  POSITION MATCH : {result['pos_matches']}/{result['pos_total']} steps"
      + (" (PASS)" if result["pos_matches"] == result["pos_total"] else " (partial)")
    )
    removed = result["removed_blocks"]
    if removed:
      print(f"  OBJECT REMOVED : True  ({len(removed)} block(s) collected)")
      for y, x, b0, b1 in removed[:6]:
        print(f"      map[{y},{x}]: block {b0} -> {b1}")
    else:
      print("  OBJECT REMOVED : False (no block changed -- movement-only replay)")


# --------------------------------------------------------------------------- #
# Visual verification
# --------------------------------------------------------------------------- #
def save_states_figure(states, title, out_path):
  """Render the first 5 and last 5 rollout states as the egocentric Craftax view.

  Top row = first 5 states, bottom row = last 5. Returns ``out_path``.
  """
  from craftax.craftax.renderer import render_craftax_pixels
  from craftax.craftax.constants import BLOCK_PIXEL_SIZE_IMG

  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  n = len(states)
  rows = [
    ("first", states[:5], 0),
    ("last", states[-5:], max(0, n - 5)),
  ]
  fig, axs = plt.subplots(2, 5, figsize=(18, 8))
  for r, (label, group, idx_base) in enumerate(rows):
    for col in range(5):
      ax = axs[r, col]
      ax.set_xticks([])
      ax.set_yticks([])
      if col >= len(group):
        ax.axis("off")
        continue
      image = render_craftax_pixels(group[col], block_pixel_size=BLOCK_PIXEL_SIZE_IMG)
      ax.imshow(np.asarray(image).astype(np.uint8))
      ax.set_title(f"{label} | step {idx_base + col}", fontsize=11)
  fig.suptitle(f"{title}  ({n} states)", fontsize=15)
  fig.tight_layout()
  fig.savefig(out_path, dpi=120, bbox_inches="tight")
  plt.close(fig)
  return out_path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--policy", choices=["random", "human"], default="random")
  parser.add_argument("--max-steps", type=int, default=60)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--data",
    default=data_configs.get_dataframe_path("craftax", "human"),
    help="human parquet to load (use the *_with_actions.parquet to replay real actions)",
  )
  parser.add_argument(
    "--actions",
    choices=["auto", "recorded", "positions"],
    default="auto",
    help="action source for --policy=human: 'recorded' uses the actions column "
    "(incl. DO, so objects get collected); 'positions' derives movement only; "
    "'auto' uses recorded when the df has an actions column.",
  )
  args = parser.parse_args()

  download_craftax_data()  # download parquets from HF if not already local
  df_path = args.data
  print(f"Loading conditions from: {df_path}")
  df = pl.read_parquet(df_path)

  use_recorded = args.actions == "recorded" or (
    args.actions == "auto" and "actions" in df.columns
  )
  if use_recorded and "actions" not in df.columns:
    raise SystemExit(
      f"--actions=recorded but {df_path} has no 'actions' column. "
      "Point --data at dataframes/craftax_human_with_actions.parquet."
    )
  print(
    f"Action source for human replay: {'recorded' if use_recorded else 'positions'}"
  )

  conditions = load_conditions(df, seed=args.seed)

  print("Building Craftax env with cached world states (first build ~30s)...")
  env, default_params = make_human_experiment_env()
  # seeds whose world is restored from cache on reset (vs regenerated)
  cached_seeds = {int(s) for s in (env.static_env_params.cached_world_states or {})}
  env_step = jax.jit(env.step)

  key = jax.random.PRNGKey(args.seed)
  any_removed = False
  figure_paths = []
  for eval_flag, row in conditions:
    key, reset_key, run_key = jax.random.split(key, 3)
    world_seed = int(row["world"])
    # reset() only uses the cache when the seed is present, else it regenerates
    if world_seed not in cached_seeds:
      raise RuntimeError(
        f"World seed {world_seed} is not in the cache {sorted(cached_seeds)}; "
        "reset() would regenerate it (version-sensitive). Re-run "
        "pregenerate_world_states.py to add it."
      )
    recorded_positions = parse_positions_string(row["positions"])
    start_pos = recorded_positions[0]
    params, state = make_env_state(
      env, default_params, world_seed, start_pos, reset_key
    )

    if args.policy == "random":
      result = run_random(env_step, state, params, run_key, args.max_steps)
    else:
      recorded_actions = (
        parse_jax_array_string(row["actions"]) if use_recorded else None
      )
      result = run_human(env_step, state, params, recorded_positions, recorded_actions)
      any_removed = any_removed or bool(result["removed_blocks"])

    _summarize(eval_flag, row, args.policy, result)

    cond = "test" if eval_flag else "train"
    fig_path = save_states_figure(
      result["states"],
      title=f"{cond.upper()} | world {world_seed} | policy={args.policy}",
      out_path=os.path.join(OUTPUT_DIR, f"craftax_human_env_{cond}_{args.policy}.png"),
    )
    figure_paths.append(fig_path)
    print(f"  saved states figure : {fig_path}")

  if args.policy == "human":
    print()
    if use_recorded:
      if any_removed:
        print("RECORDED-ACTION REPLAY: goal object(s) collected/removed. PASS.")
      else:
        print("RECORDED-ACTION REPLAY: no object removed in any condition (FAIL).")
        sys.exit(1)
    else:
      print(
        "MOVEMENT-ONLY REPLAY: objects are NOT collected (DO is lost). Use "
        "--data dataframes/craftax_human_with_actions.parquet to replay real actions."
      )

  for path in figure_paths:
    subprocess.run(["open", path])


if __name__ == "__main__":
  main()
