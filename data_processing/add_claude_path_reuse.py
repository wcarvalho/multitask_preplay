"""Add `path_reuse_ai` to the published Craftax human dataframe.

Joins `plots/output/craftax_multi_claude_path_reuse_simple_full/classifications.parquet`
into `dataframes/craftax_human.parquet` on (user_id, global_episode_idx). Adds
exactly one column:

  - path_reuse_ai : float (0.0 / 1.0 / null) — Claude's per-eval-trial verdict.
                    Direct counterpart to the existing `reuse` column.

Non-eval rows (eval=False, manipulation!='paths', success=False, or worlds
outside the cached subset) get null. The audit fields (reasoning, same_side,
etc.) stay only in classifications.parquet under plots/output/.

After running this:
  python data_processing/upload_dataframes.py
will push the enriched parquet to `wcarvalho/Multitask_Preplay_Craftax_human` on HF.

Usage:
  python data_processing/add_claude_path_reuse.py            # in-place enrichment
  python data_processing/add_claude_path_reuse.py --dry-run  # report only, no write
"""

import argparse
import os
import shutil
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_configs

CLASSIFICATIONS_PARQUET = os.path.join(
  os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
  "plots",
  "output",
  "craftax_multi_claude_path_reuse_simple_full",
  "classifications.parquet",
)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Report what would change but don't overwrite the parquet.",
  )
  parser.add_argument(
    "--no-backup",
    action="store_true",
    help="Skip the .bak copy. Default writes a backup next to the original.",
  )
  args = parser.parse_args()

  human_path = data_configs.get_dataframe_path("craftax", "human")
  print(f"Reading {human_path}...")
  human = pl.read_parquet(human_path)
  print(f"  {human.shape[0]} rows, {human.shape[1]} cols")

  if not os.path.exists(CLASSIFICATIONS_PARQUET):
    raise FileNotFoundError(
      f"Missing {CLASSIFICATIONS_PARQUET}. Run "
      "`python plots/craftax_multi_claude_path_reuse.py --full --prompt simple "
      "--output-suffix simple_full` first."
    )
  print(f"Reading {CLASSIFICATIONS_PARQUET}...")
  ai = pl.read_parquet(CLASSIFICATIONS_PARQUET).filter(
    pl.col("claude_label").is_in(["path_reuse", "no_path_reuse"])
  )
  print(f"  {ai.shape[0]} successfully-classified rows")

  ai_join = ai.select(
    [
      "user_id",
      "global_episode_idx",
      (pl.col("claude_label") == "path_reuse").cast(pl.Float64).alias("path_reuse_ai"),
    ]
  )

  # Drop any pre-existing path_reuse_ai* columns so we always overwrite cleanly.
  drop_cols = [c for c in human.columns if c.startswith("path_reuse_ai")]
  if drop_cols:
    print(f"  Dropping pre-existing columns: {drop_cols}")
    human = human.drop(drop_cols)

  enriched = human.join(ai_join, on=["user_id", "global_episode_idx"], how="left")

  # Coverage report (eval, manipulation='paths' rows — what the analysis cares about).
  eligible = enriched.filter(eval=True, manipulation="paths")
  total_eligible = eligible.shape[0]
  covered = eligible.filter(pl.col("path_reuse_ai").is_not_null()).shape[0]
  print(
    f"\nCoverage on eval/paths rows: {covered} / {total_eligible} "
    f"({100 * covered / total_eligible:.1f}%) have path_reuse_ai populated."
  )
  for tr in (1, 0):
    sub = eligible.filter(tell_reuse=tr)
    sub_covered = sub.filter(pl.col("path_reuse_ai").is_not_null()).shape[0]
    print(
      f"  tell_reuse={tr}: {sub_covered}/{sub.shape[0]} rows covered, "
      f"path_reuse_ai mean = "
      f"{sub.filter(pl.col('path_reuse_ai').is_not_null())['path_reuse_ai'].mean():.3f}"
    )

  if args.dry_run:
    print("\n--dry-run set; not writing.")
    return

  if not args.no_backup:
    backup = human_path + ".bak"
    print(f"\nBacking up {human_path} → {backup}")
    shutil.copyfile(human_path, backup)

  print(f"Writing {human_path}...")
  enriched.write_parquet(human_path)
  print(f"  Now {enriched.shape[0]} rows, {enriched.shape[1]} cols")
  print("\nNext: `python data_processing/upload_dataframes.py` to push to HuggingFace.")


if __name__ == "__main__":
  main()
