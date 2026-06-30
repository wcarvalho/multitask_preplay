"""Starter / onboarding script for the Craftax human experiment environment.

What this does
--------------
The human Craftax experiment presents people with two kinds of episodes per
world: a *train* condition (``eval=False``) and a *test* condition
(``eval=True``). This script:

  1. Loads BOTH a train and a test condition from the human dataframe
     (``dataframes/craftax_human.parquet``).
  2. Instantiates the real Craftax environment (the same
     ``CraftaxSymbolicWebEnvNoAutoReset`` the experiment used) in each
     condition, with the condition's world seed and start position.
  3. Steps the environment under a chosen policy.
  4. Renders a 2x5 figure per condition (top row = first 5 rollout states, bottom
     row = last 5) as the FIRST-PERSON Craftax view the human saw, saves it to
     ``starter/output/`` and opens it -- a visual check that the world loaded
     correctly and the agent stepped through it.

Two policies are supported:

  * ``--policy=random`` : take uniformly-random *movement* actions for up to
    ``--max-steps`` steps (or until the episode terminates), then print a
    summary of rewards / positions.

  * ``--policy=human``  : pick a RANDOM human episode for the condition, derive
    the human's action sequence from their recorded ``positions`` (via
    ``simulations.craftax_utils.actions_from_path``), replay those actions in
    the env, and VERIFY that the env's rolled-out agent positions match the
    human's recorded positions step-for-step. A successful match proves the env
    was correctly instantiated in the human's condition. This is the script's
    reason to exist.

Run from the repo root::

    python starter/craftax_human_env.py --policy=random
    python starter/craftax_human_env.py --policy=human

The first env build compiles JAX kernels and may take ~30s.

Verified facts this script relies on (file:line confirmed)
---------------------------------------------------------
  * ``CraftaxSymbolicWebEnvNoAutoReset.reset(key, params) -> (obs, state)``
        simulations/craftax_web_env.py:597-600
  * ``...step(key, state, action, params) -> (obs, state, reward, done, info)``
        simulations/craftax_web_env.py:538-548
  * Deterministic start position via ``state.replace(player_position=...)``
        simulations/craftax_utils.py:721
  * Action enum: NOOP=0, LEFT=1, RIGHT=2, UP=3, DOWN=4, DO=5
        craftax.craftax.constants.Action
  * ``actions_from_path(path)`` returns one action per recorded position:
    (N-1) movement actions + 1 trailing NOOP -> length N for N positions.
        simulations/craftax_utils.py:314-338
"""

import argparse
import os
import subprocess
import sys

# Put repo root (and simulations/) on sys.path so this runs as
# `python starter/craftax_human_env.py` from the repo root.
# (mirrors analysis/craftax_analysis.py lines 7-14)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_project_root)
sys.path.append(os.path.join(_project_root, "simulations"))

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import matplotlib

matplotlib.use("Agg")  # render figures to PNG; no display needed
import matplotlib.pyplot as plt

import data_configs

# Import the SINGLE canonical env builder (the exact env used to collect the
# human data) plus the position-string parser. make_human_experiment_env loads
# the frozen JAX-0.4.22 world-state cache, so reset() restores the cached world
# instead of regenerating it from the JAX PRNG -- this is what makes replay
# reproducible across JAX versions. It is the same builder the experiment uses.
from simulations.craftax_web_env import make_human_experiment_env
from analysis.vis_utils import parse_positions_string, parse_jax_array_string
from simulations import craftax_utils
from craftax.craftax.constants import Action

# Movement action indices (craftax.craftax.constants.Action). These are the
# only indices `actions_from_path` ever emits, so sampling from them for the
# random policy keeps the agent on the grid and comparable to a human path.
ACTION_NOOP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_UP = 3
ACTION_DOWN = 4
MOVEMENT_ACTIONS = np.array([ACTION_LEFT, ACTION_RIGHT, ACTION_UP, ACTION_DOWN])

# The recorded `actions` column stores WEB-experiment action indices, NOT raw
# Craftax `Action` enum values. The web app exposed only five actions, in this
# order, and stepped the env via `action_array[action_idx]`
# (experiments/craftax/craftax_experiment_structure.py:320-321,406):
#   web 0 -> RIGHT, 1 -> DOWN, 2 -> LEFT, 3 -> UP, 4 -> DO
# So a recorded `action_idx` must be mapped through this table before being
# passed to `env.step`, which expects raw enum values. Verified empirically:
# this remap reproduces every recorded position delta (13375/13375); passing the
# raw indices instead matches only ~12% and collects nothing.
# = [2, 4, 1, 3, 5].
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
  """Pick one random episode for each of the train (eval=False) and test
  (eval=True) conditions.

  Returns a list of (eval_flag, row_dict) in train-then-test order. Prefers
  SUCCESSFUL episodes (so a recorded-action replay actually collects the goal),
  and only rows whose recorded ``positions`` parse to length >= 2.
  """
  rng = np.random.default_rng(seed)
  conditions = []
  for eval_flag in (False, True):
    # Polars keyword-arg filtering (see CLAUDE.md): df.filter(eval=...) ==
    # df.filter(pl.col("eval") == ...). Prefer successful episodes first.
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
  """Reset the env in the given world and override the start position.

  Mirrors the canonical reset pattern in vis_utils.get_craftax_env_image:
  reset returns ``(obs, state)`` with state second; we then replace
  ``player_position`` deterministically (craftax_utils.py:721).
  """
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
  states = [state]  # full env states, for the visual sanity-check figure
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

  Two action sources:
    * ``recorded_actions`` given (the REAL per-step action indices from the
      ``actions`` column, incl. DO/interact): replay THOSE -> the human's full
      behaviour, so the goal object actually gets mined/collected.
    * else: derive movement-only actions from ``positions`` via
      ``actions_from_path`` (DO collapses to NOOP -> object is never collected).

  Records the world map before/after to report which blocks the agent removed
  (collected), and a step-wise position match against the recording.
  """
  if recorded_actions is not None:
    # Recorded indices are in WEB action space; remap to raw env enum values
    # before stepping (see WEB_TO_ENV_ACTION). The movement-only `actions_from_path`
    # branch below already emits raw enum actions, so it needs no remap.
    actions = WEB_TO_ENV_ACTION[np.asarray(recorded_actions, dtype=np.int32)]
    action_source = "recorded"
  else:
    actions = np.asarray(craftax_utils.actions_from_path(recorded_positions))
    action_source = "positions"

  map_before = np.asarray(state.map[0])  # goal objects are baked into the cached map
  rolled_out = [np.asarray(state.player_position)]
  states = [state]  # full env states, for the visual sanity-check figure
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

  # Position match: env positions vs the recording, aligned by index.
  rolled = np.asarray(rolled_out)
  rec = np.asarray(recorded_positions)
  k = min(len(rolled), len(rec))
  pos_matches = int(np.all(rolled[:k] == rec[:k], axis=1).sum())

  # Object removal: cells whose block changed (the agent mined/collected them).
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
  """Render the first 5 and last 5 rollout states into a 2x5 figure.

  Top row = first 5 states, bottom row = last 5 states, each rendered as the
  FIRST-PERSON (egocentric) Craftax view -- exactly what the human participant
  saw on screen (player centered). Watching this local view change is a quick
  visual confirmation that the env loaded the correct world and stepped through
  it as expected. Returns ``out_path``.
  """
  # The egocentric renderer the experiment displayed to humans
  # (craftax_experiment_structure.py render_fn). The player is centered, so no
  # agent marker is needed.
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
  # The single canonical builder (same env the experiment + data collection use).
  env, default_params = make_human_experiment_env()
  # Seeds whose WORLD is restored from cache on reset (vs regenerated). Read off
  # the env itself so the guard can't drift from what was actually loaded.
  cached_seeds = {int(s) for s in (env.static_env_params.cached_world_states or {})}
  # jit step once so per-step replay is fast after a single compile.
  env_step = jax.jit(env.step)

  key = jax.random.PRNGKey(args.seed)
  any_removed = False
  figure_paths = []
  for eval_flag, row in conditions:
    key, reset_key, run_key = jax.random.split(key, 3)
    world_seed = int(row["world"])
    # Guard against a silent fall-back to JAX-version-sensitive regeneration:
    # reset_env only uses the cache when the seed is present (web_env.py:618).
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

    # Visual sanity check: render the first 5 + last 5 states of this rollout.
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

  # Open the rendered figures so the loaded data can be eyeballed.
  for path in figure_paths:
    subprocess.run(["open", path])


if __name__ == "__main__":
  main()
