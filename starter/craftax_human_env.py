"""Step the Craftax env through a human episode (or random actions) and plot it.

python starter/craftax_human_env.py --policy human   # replay a participant
python starter/craftax_human_env.py --policy random  # random movement
"""

import argparse
import os
import subprocess
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_root)
sys.path.append(os.path.join(_root, "simulations"))

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import matplotlib

matplotlib.use("Agg")  # save PNG, no display
import matplotlib.pyplot as plt

import data_configs
from analysis.download_dataframes import download_craftax_data
from analysis.vis_utils import parse_positions_string, parse_jax_array_string
from simulations.craftax_web_env import make_human_experiment_env
from craftax.craftax.constants import Action, BLOCK_PIXEL_SIZE_IMG
from craftax.craftax.renderer import render_craftax_pixels

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Random policy: the four movement actions.
MOVEMENT_ACTIONS = np.array(
  [Action.LEFT.value, Action.RIGHT.value, Action.UP.value, Action.DOWN.value]
)
# The recorded `actions` are web-experiment indices [RIGHT, DOWN, LEFT, UP, DO];
# map them to raw Craftax enum values before stepping the env.
WEB_TO_ENV_ACTION = np.array(
  [
    Action.RIGHT.value,
    Action.DOWN.value,
    Action.LEFT.value,
    Action.UP.value,
    Action.DO.value,
  ]
)


def plot_states(states, title, out_path):
  """Save the first 5 and last 5 states as the first-person Craftax view (2x5 grid)."""
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  rows = [("first", states[:5], 0), ("last", states[-5:], max(0, len(states) - 5))]
  fig, axs = plt.subplots(2, 5, figsize=(18, 8))
  for r, (label, group, base) in enumerate(rows):
    for c in range(5):
      ax = axs[r, c]
      ax.axis("off")
      if c < len(group):
        image = render_craftax_pixels(group[c], block_pixel_size=BLOCK_PIXEL_SIZE_IMG)
        ax.imshow(np.asarray(image).astype(np.uint8))
        ax.set_title(f"{label} step {base + c}", fontsize=11)
  fig.suptitle(title)
  fig.tight_layout()
  fig.savefig(out_path, dpi=120, bbox_inches="tight")
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--policy", choices=["human", "random"], default="human")
  parser.add_argument("--max-steps", type=int, default=60)  # used by the random policy
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  download_craftax_data()  # download the human dataframe from HF if not already local
  df = pl.read_parquet(data_configs.get_dataframe_path("craftax", "human"))

  print("Building the Craftax env (loads the world-state cache; first build ~30s)...")
  env, default_params = make_human_experiment_env()
  step = jax.jit(env.step)

  rng = np.random.default_rng(args.seed)
  key = jax.random.PRNGKey(args.seed)
  rollouts = []

  # roll out one successful episode for each of the train / test conditions
  for eval_flag in (False, True):
    sub = df.filter(eval=eval_flag, success=1.0)
    row = sub.row(int(rng.integers(len(sub))), named=True)
    positions = parse_positions_string(row["positions"])

    # reset into this episode's world, at the human's start position
    params = default_params.replace(world_seeds=(int(row["world"]),))
    _, state = env.reset(key, params)
    state = state.replace(player_position=jnp.asarray(positions[0], dtype=jnp.int32))

    if args.policy == "human":
      actions = WEB_TO_ENV_ACTION[parse_jax_array_string(row["actions"]).astype(int)]
      n_steps = len(actions)
    else:
      n_steps = args.max_steps

    # ---- rollout loop ----
    states = [state]
    for t in range(n_steps):
      if args.policy == "human":
        action = int(actions[t])
      else:
        action = int(rng.choice(MOVEMENT_ACTIONS))
      key, subkey = jax.random.split(key)
      _, state, reward, done, _ = step(subkey, state, jnp.int32(action), params)
      states.append(state)
      if bool(done):
        break

    cond = "test" if eval_flag else "train"
    rollouts.append((cond, int(row["world"]), states))
    print(f"  {cond}: world {row['world']}, {len(states)} states")

  # ---- make the plots ----
  for cond, world, states in rollouts:
    out_path = os.path.join(OUTPUT_DIR, f"craftax_human_env_{cond}_{args.policy}.png")
    plot_states(states, f"{cond} | world {world} | policy={args.policy}", out_path)
    print(f"  saved {out_path}")
    subprocess.run(["open", out_path])


if __name__ == "__main__":
  main()
