"""Add an `actions` column to the Craftax human dataframe.

Writes a NEW file (`dataframes/craftax_human_with_actions.parquet`); the original
`craftax_human.parquet` is read-only and never modified. Reads the small actions
arrays from the per-user episode caches one at a time (the aggregate safetensor
is too large to load), then joins them onto the dataframe.

Usage:
    python data_processing/add_craftax_actions.py            # writes the new file
    python data_processing/add_craftax_actions.py --dry-run  # verify, no write
"""

import argparse
import gc
import glob
import json
import os
import sys

import numpy as np
import polars as pl
from flax import serialization

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_configs

N_EXPECTED = 16887  # Craftax human episode count


def build_actions_by_user_episode():
  """Read per-user caches one at a time; return {(user_id, user_episode_idx): actions_str}."""
  cache_dir = os.path.join(data_configs.CRAFTAX_HUMAN_RAW_DATA_DIR, "cache")
  byte_files = sorted(glob.glob(os.path.join(cache_dir, "*_episode_data.bytes")))
  if not byte_files:
    raise FileNotFoundError(
      f"No per-user caches in {cache_dir}. Is the external drive mounted?"
    )
  print(f"Reading actions from {len(byte_files)} per-user caches in {cache_dir}")

  mapping = {}
  for n, bf in enumerate(byte_files):
    info_path = bf.replace("_episode_data.bytes", "_episode_info.json")
    if not os.path.exists(info_path):
      print(f"  [skip] no info json for {os.path.basename(bf)}")
      continue
    try:
      with open(bf, "rb") as fh:
        episodes = serialization.from_bytes(None, fh.read())
      with open(info_path) as fh:
        info = json.load(fh)
    except Exception as e:  # noqa: BLE001
      print(f"  [skip] failed to load {os.path.basename(bf)}: {e}")
      continue

    assert len(episodes) == len(info), (
      f"{os.path.basename(bf)}: {len(episodes)} episodes vs {len(info)} info entries"
    )
    for i in range(len(info)):
      meta = info[i]
      user_id = int(meta["user_storage"]["seed"])
      ue_idx = int(meta["user_episode_idx"])
      actions = np.asarray(episodes[str(i)]["actions"])
      mapping[(user_id, ue_idx)] = str(actions[:-1])

    del episodes, info
    gc.collect()
    if (n + 1) % 50 == 0 or n + 1 == len(byte_files):
      print(f"  processed {n + 1}/{len(byte_files)} caches ({len(mapping)} episodes)")
  return mapping


def build_actions_frame(mapping):
  """Build a polars frame [user_id, global_episode_idx, actions] via global metadata."""
  meta_path = os.path.join(
    data_configs.CRAFTAX_MODEL_RAW_DATA_DIR, "final", "human_data_episode_metadata.json"
  )
  with open(meta_path) as fh:
    meta = json.load(fh)
  print(f"Global metadata: {len(meta)} episodes (index == global_episode_idx)")

  uids, gidxs, acts, missing = [], [], [], 0
  for g, m in enumerate(meta):
    user_id = int(m["user_storage"]["seed"])
    ue_idx = int(m["user_episode_idx"])
    a = mapping.get((user_id, ue_idx))
    if a is None:
      missing += 1
      continue
    uids.append(user_id)
    gidxs.append(g)
    acts.append(a)
  if missing:
    print(f"  WARNING: {missing} global episodes had no matching cached actions")
  return pl.DataFrame(
    {"user_id": uids, "global_episode_idx": gidxs, "actions": acts},
    schema={"user_id": pl.Int64, "global_episode_idx": pl.Int64, "actions": pl.Utf8},
  )


def verify(enriched, human, orig_cols):
  """Check the enriched frame is the original plus an `actions` column."""
  assert enriched.height == human.height == N_EXPECTED, (
    f"row count changed: {enriched.height} vs {human.height}"
  )
  assert set(enriched.columns) == set(orig_cols) | {"actions"}, "unexpected columns"
  assert len(enriched.columns) == len(orig_cols) + 1, "more than one new column"
  assert enriched.drop("actions").equals(human), (
    "original columns are NOT byte-identical after the join"
  )
  assert enriched["actions"].null_count() == 0, (
    "actions has nulls (incomplete coverage)"
  )
  assert (
    enriched.unique(subset=["user_id", "global_episode_idx"]).height == enriched.height
  ), "row duplication detected"
  for c in ("worker_id", "hit_id", "assignment_id"):
    assert c in enriched.columns, (
      f"PII column {c} missing locally (must be present here)"
    )
  print(
    "  VERIFY: rows==16887, +1 col 'actions', original byte-identical, 0 nulls, PII present -> OK"
  )


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  default_out = os.path.join(
    data_configs.DATAFRAMES_DIR, "craftax_human_with_actions.parquet"
  )
  ap.add_argument("--out", default=default_out)
  ap.add_argument("--dry-run", action="store_true")
  ap.add_argument("--force", action="store_true", help="overwrite --out if it exists")
  args = ap.parse_args()

  human_path = data_configs.get_dataframe_path("craftax", "human")
  assert os.path.abspath(args.out) != os.path.abspath(human_path), (
    "--out must NOT be the original craftax_human.parquet (it stays untouched)"
  )

  mapping = build_actions_by_user_episode()
  actions_frame = build_actions_frame(mapping)
  assert actions_frame.height == N_EXPECTED, (
    f"built {actions_frame.height} action rows, expected {N_EXPECTED}"
  )

  human = pl.read_parquet(human_path)
  print(f"Loaded original {human_path}: {human.shape}")
  orig_cols = human.columns
  assert "actions" not in orig_cols, (
    "original parquet already has an 'actions' column?!"
  )

  enriched = (
    human.with_row_index("__orig_row")
    .join(
      actions_frame,
      on=["user_id", "global_episode_idx"],
      how="left",
      validate="1:1",
    )
    .sort("__orig_row")
    .drop("__orig_row")
    .select([*orig_cols, "actions"])
  )
  verify(enriched, human, orig_cols)
  print(f"  sample actions[0]: {enriched['actions'][0][:60]!r}...")

  if args.dry_run:
    print("--dry-run: not writing.")
    return

  if os.path.exists(args.out) and not args.force:
    raise FileExistsError(f"{args.out} exists; pass --force to overwrite the NEW file")
  enriched.write_parquet(args.out)
  print(f"Wrote {args.out}")
  verify(pl.read_parquet(args.out), pl.read_parquet(human_path), orig_cols)
  print(
    f"\nDone. Original {human_path} untouched. "
    "Next: python data_processing/upload_craftax_human_only.py"
  )


if __name__ == "__main__":
  main()
