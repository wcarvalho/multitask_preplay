"""Starter / onboarding script: load Craftax human experiment data and plot
example trajectories.

What it does:
  1. Loads the Craftax human dataframe (``dataframes/craftax_human.parquet``).
  2. Splits it into train (``eval=False``) and test (``eval=True``) dataframes
     and prints the first 5 rows of each.
  3. Plots 3 MATCHED train/test trajectory pairs (each column = same user_id +
     same world; top row = train, bottom row = test) into a single 2x3 figure,
     marks the start (yellow star) and end (yellow circle) of each path, saves
     it to ``starter/output/craftax_human_data.png``, and opens it (macOS
     ``open``).

Run from the repo root:
    python starter/craftax_human_data.py
"""

import os
import sys
import subprocess

# Put repo root (and simulations/) on sys.path so `import data_configs` and
# `from analysis import vis_utils` work when run as
# `python starter/craftax_human_data.py`.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_project_root)
sys.path.append(os.path.join(_project_root, "simulations"))

import numpy as np
import polars as pl
import matplotlib

matplotlib.use("Agg")  # no display needed; we save + `open` the PNG
import matplotlib.pyplot as plt

import data_configs
from analysis.vis_utils import (
  get_craftax_env_image,
  craftax_actions_from_path,
  craftax_place_arrows_on_image,
  parse_positions_string,
)


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_matched_train_row(test_row: dict, train_df: pl.DataFrame):
  """Return the train row matched to ``test_row``, or None.

  Preferred match: the train row whose ``global_episode_idx`` equals the test
  row's ``corresponding_train_episode_idx`` (guarantees same user_id + world).
  Fallback: any train row sharing the same (user_id, world).
  """
  corr_idx = test_row["corresponding_train_episode_idx"]
  if corr_idx is not None:
    match = train_df.filter(pl.col("global_episode_idx") == corr_idx)
    if match.height > 0:
      return match.row(0, named=True)

  # Fallback: same user_id + world.
  fallback = train_df.filter(
    (pl.col("user_id") == test_row["user_id"]) & (pl.col("world") == test_row["world"])
  )
  if fallback.height > 0:
    return fallback.row(0, named=True)

  return None


def sample_matched_pairs(test_df: pl.DataFrame, train_df: pl.DataFrame, n: int, rng):
  """Sample ``n`` (train_row, test_row) matched pairs.

  Each pair shares user_id + world. Test rows are preferred to be successful,
  non-trivial, and from DISTINCT (user_id, world) pairs so the columns don't
  all show the same world. Falls back gracefully if not enough qualify.
  """
  preferred = test_df.filter((pl.col("success") == 1.0) & (pl.col("path_length") > 2))
  pool = preferred if preferred.height >= n else test_df

  order = list(rng.permutation(pool.height))
  pairs = []
  seen_uw = set()
  for i in order:
    if len(pairs) >= n:
      break
    test_row = pool.row(i, named=True)
    uw = (test_row["user_id"], test_row["world"])
    if uw in seen_uw:
      continue
    train_row = find_matched_train_row(test_row, train_df)
    if train_row is None:
      continue  # no matching train row -> resample another test row
    seen_uw.add(uw)
    pairs.append((train_row, test_row))

  # If distinctness was too strict to reach n, relax it and fill remaining.
  if len(pairs) < n:
    for i in order:
      if len(pairs) >= n:
        break
      test_row = pool.row(i, named=True)
      train_row = find_matched_train_row(test_row, train_df)
      if train_row is None:
        continue
      if any(
        test_row["global_episode_idx"] == t["global_episode_idx"] for _, t in pairs
      ):
        continue
      pairs.append((train_row, test_row))

  return pairs


def draw_panel(row: dict, ax, title: str):
  """Render one trajectory panel: env image + red path arrows + start/end marks.

  Uses the same low-level helpers as ``plots/craftax_multi_overlap_examples.py``
  so the start (yellow star) / end (yellow circle) markers match the paper.
  """
  world_seed = int(row["world"])  # parquet stores world as a string
  image, maze_height, maze_width = get_craftax_env_image(world_seed)
  positions = parse_positions_string(row["positions"])  # (N, 2) as (row=y, col=x)
  actions = craftax_actions_from_path(positions)

  craftax_place_arrows_on_image(
    image=image,
    positions=positions,
    actions=actions,
    maze_height=maze_height,
    maze_width=maze_width,
    ax=ax,
    arrow_color="red",
    display_image=True,
    show_path_length=False,
  )

  if len(positions) > 0:
    scale_y = image.shape[0] / maze_height
    scale_x = image.shape[1] / maze_width
    # START -- yellow star (positions are (y, x); x/y swap is the repo convention)
    ax.plot(
      (positions[0][1] + 0.5) * scale_x,
      (positions[0][0] + 0.5) * scale_y,
      "*",
      color="yellow",
      markersize=18,
      markeredgecolor="black",
      markeredgewidth=1,
    )
    # END -- yellow circle
    ax.plot(
      (positions[-1][1] + 0.5) * scale_x,
      (positions[-1][0] + 0.5) * scale_y,
      "o",
      color="yellow",
      markersize=12,
      markeredgecolor="black",
      markeredgewidth=1,
    )

  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_title(title)


def main():
  # 1. Load the Craftax human dataframe.
  df_path = data_configs.get_dataframe_path("craftax", "human")
  df = pl.read_parquet(df_path)
  print(f"Loaded {df_path}")
  print(f"Full df shape: {df.shape}")

  # 2. Split into train / test and print the heads.
  train_df = df.filter(eval=False)
  test_df = df.filter(eval=True)
  print(f"\nTrain df shape: {train_df.shape}")
  print(f"Test df shape:  {test_df.shape}")

  print("\n=== TRAIN df.head(5) ===")
  print(train_df.head(5))
  print("\n=== TEST df.head(5) ===")
  print(test_df.head(5))

  # 3. Sample 3 MATCHED train/test pairs (reproducible) -- each column shares
  #    user_id + world.
  rng = np.random.default_rng(0)
  pairs = sample_matched_pairs(test_df, train_df, 3, rng)
  if len(pairs) < 3:
    raise RuntimeError(f"Only found {len(pairs)} matched pairs; expected 3.")

  fig, axes = plt.subplots(2, 3, figsize=(15, 10))

  print("\n=== Matched panels (eval, world, user_id, success, path_length) ===")
  for j, (train_row, test_row) in enumerate(pairs):
    # Each column is a matched pair: assert same user_id + world.
    assert train_row["user_id"] == test_row["user_id"], (
      f"user_id mismatch in column {j}: {train_row['user_id']} != {test_row['user_id']}"
    )
    assert train_row["world"] == test_row["world"], (
      f"world mismatch in column {j}: {train_row['world']} != {test_row['world']}"
    )

    for split_label, row, ax in [
      ("Train", train_row, axes[0, j]),
      ("Test", test_row, axes[1, j]),
    ]:
      world = row["world"]
      success = row["success"]
      path_length = row["path_length"]
      user_id = row["user_id"]
      print(
        f"  {split_label}: "
        f"(eval={row['eval']}, world={world}, user_id={user_id}, "
        f"success={success}, path_length={path_length})"
      )
      # user_id here is the experiment-internal random id, NOT the CloudResearch
      # / worker id (see starter/README.md).
      title = f"{split_label}. World={world}. user={user_id}."
      draw_panel(row, ax, title)

  fig.tight_layout()
  out_path = os.path.join(OUTPUT_DIR, "craftax_human_data.png")
  fig.savefig(out_path, dpi=150, bbox_inches="tight")
  plt.close(fig)
  size_kb = os.path.getsize(out_path) / 1024
  print(f"\nSaved figure to {out_path} ({size_kb:.1f} KB)")

  subprocess.run(["open", out_path])


if __name__ == "__main__":
  main()
