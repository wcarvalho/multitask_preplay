"""Load the Craftax human dataframe and plot example train/test trajectories.

Run from the repo root:
    python starter/craftax_human_data.py
"""

import os
import sys
import subprocess

# Put repo root and simulations/ on sys.path so the imports below resolve.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_project_root)
sys.path.append(os.path.join(_project_root, "simulations"))

import numpy as np
import polars as pl
import matplotlib

matplotlib.use("Agg")  # save PNG, no display
import matplotlib.pyplot as plt

import data_configs
from analysis.download_dataframes import download_craftax_data
from analysis.vis_utils import (
  get_craftax_env_image,
  craftax_actions_from_path,
  craftax_place_arrows_on_image,
  parse_positions_string,
)


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_matched_train_row(test_row: dict, train_df: pl.DataFrame):
  """Return the train row matched to ``test_row`` (same user_id + world), or None."""
  corr_idx = test_row["corresponding_train_episode_idx"]
  if corr_idx is not None:
    match = train_df.filter(pl.col("global_episode_idx") == corr_idx)
    if match.height > 0:
      return match.row(0, named=True)

  fallback = train_df.filter(
    (pl.col("user_id") == test_row["user_id"]) & (pl.col("world") == test_row["world"])
  )
  if fallback.height > 0:
    return fallback.row(0, named=True)

  return None


def sample_matched_pairs(test_df: pl.DataFrame, train_df: pl.DataFrame, n: int, rng):
  """Sample ``n`` (train_row, test_row) pairs, each sharing user_id + world.

  Prefers successful, non-trivial test rows from distinct (user_id, world)
  pairs, relaxing the distinctness constraint if too few qualify.
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
      continue
    seen_uw.add(uw)
    pairs.append((train_row, test_row))

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
  """Render one trajectory panel: env image + red path arrows + start/end marks."""
  world_seed = int(row["world"])  # parquet stores world as a string
  image, maze_height, maze_width = get_craftax_env_image(world_seed)
  positions = parse_positions_string(row["positions"])  # (N, 2) as (y, x)
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
    # start: yellow star
    ax.plot(
      (positions[0][1] + 0.5) * scale_x,
      (positions[0][0] + 0.5) * scale_y,
      "*",
      color="yellow",
      markersize=18,
      markeredgecolor="black",
      markeredgewidth=1,
    )
    # end: yellow circle
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
  download_craftax_data()  # download parquets from HF if not already local
  df_path = data_configs.get_dataframe_path("craftax", "human")
  df = pl.read_parquet(df_path)
  print(f"Loaded {df_path}")
  print(f"Full df shape: {df.shape}")

  train_df = df.filter(eval=False)
  test_df = df.filter(eval=True)
  print(f"\nTrain df shape: {train_df.shape}")
  print(f"Test df shape:  {test_df.shape}")

  print("\n=== TRAIN df.head(5) ===")
  print(train_df.head(5))
  print("\n=== TEST df.head(5) ===")
  print(test_df.head(5))

  rng = np.random.default_rng(0)
  pairs = sample_matched_pairs(test_df, train_df, 3, rng)
  if len(pairs) < 3:
    raise RuntimeError(f"Only found {len(pairs)} matched pairs; expected 3.")

  fig, axes = plt.subplots(2, 3, figsize=(15, 10))

  print("\n=== Matched panels (eval, world, user_id, success, path_length) ===")
  for j, (train_row, test_row) in enumerate(pairs):
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
