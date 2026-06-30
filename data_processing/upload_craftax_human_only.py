"""Upload only the Craftax human parquet (with the `actions` column) to HF.

Touches exactly one repo (`wcarvalho/Multitask_Preplay_Craftax_human`); no model
or JaxMaze data. PII columns are dropped before upload.

Usage:
    python data_processing/upload_craftax_human_only.py            # real upload
    python data_processing/upload_craftax_human_only.py --dry-run  # checks, no commit
"""

import argparse
import os
import sys
import tempfile

import polars as pl
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_configs

api = HfApi()

PII_COLUMNS = ["worker_id", "hit_id", "assignment_id"]
GITHUB_URL = "https://github.com/wcarvalho/multitask_preplay"
N_EXPECTED = 16887


def make_readme(title: str, splits: list[str]) -> str:
  lines = ["---", "configs:", "- config_name: default", "  data_files:"]
  for split in splits:
    lines.append(f"  - split: {split}")
    lines.append(f"    path: data/{split}-*")
  lines += [
    "---",
    "",
    f"# {title}",
    "",
    'Data for the paper "Multitask Preplay" (PNAS).',
    f"Analysis code: {GITHUB_URL} (branch `pnas`).",
    "",
    "Splits: " + ", ".join(f"`{s}`" for s in splits) + ".",
    "Each split is also available as a top-level parquet file.",
    "",
  ]
  return "\n".join(lines)


def upload_splits(repo_id, title, split_files):
  api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
  operations = []
  for split, local_path in split_files.items():
    operations.append(
      CommitOperationAdd(
        path_in_repo=f"data/{split}-00000-of-00001.parquet", path_or_fileobj=local_path
      )
    )
    operations.append(
      CommitOperationAdd(path_in_repo=f"{split}.parquet", path_or_fileobj=local_path)
    )
  operations.append(
    CommitOperationAdd(
      path_in_repo="README.md",
      path_or_fileobj=make_readme(title, list(split_files)).encode(),
    )
  )
  info = api.create_commit(
    repo_id=repo_id,
    repo_type="dataset",
    operations=operations,
    commit_message="Add actions column to craftax human data",
  )
  print(f"  {repo_id}: committed {info.oid[:8]} ({len(split_files)} splits)")


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument(
    "--input",
    default=os.path.join(
      data_configs.DATAFRAMES_DIR, "craftax_human_with_actions.parquet"
    ),
  )
  ap.add_argument("--dry-run", action="store_true")
  args = ap.parse_args()

  repo_id = f"wcarvalho/{data_configs.HUGGINGFACE_CRAFTAX_DATASET_NAME}_human"
  df = pl.read_parquet(args.input)
  print(f"Input: {args.input}  ({df.shape})")
  print(f"Target repo (ONLY this one): {repo_id}")
  assert df.height == N_EXPECTED, f"expected {N_EXPECTED} rows, got {df.height}"
  assert "actions" in df.columns, "input is missing the 'actions' column"

  existing_pii = [c for c in PII_COLUMNS if c in df.columns]
  print(f"Dropping PII before upload: {existing_pii}")

  with tempfile.TemporaryDirectory() as tmp:
    human_path = os.path.join(tmp, "human_data.parquet")
    df.drop(existing_pii).write_parquet(human_path)

    out = pl.read_parquet(human_path)
    assert all(c not in out.columns for c in PII_COLUMNS), "PII not dropped!"
    assert "actions" in out.columns and out["actions"].null_count() == 0
    assert out.height == N_EXPECTED
    print(f"Upload artifact: {out.shape}, PII removed, actions present (0 nulls).")

    if args.dry_run:
      print("--dry-run: not committing. (No model/jaxmaze repos referenced.)")
      return

    upload_splits(
      repo_id, "Multitask Preplay — craftax human data", {"human_data": human_path}
    )

  # re-download the committed file and check it
  cached = hf_hub_download(
    repo_id=repo_id,
    filename="data/human_data-00000-of-00001.parquet",
    repo_type="dataset",
    force_download=True,
  )
  remote = pl.read_parquet(cached)
  assert "actions" in remote.columns and remote["actions"].null_count() == 0
  assert all(c not in remote.columns for c in PII_COLUMNS)
  assert remote.height == N_EXPECTED
  print(f"POST-UPLOAD OK: remote {remote.shape}, has actions (0 nulls), no PII.")
  print(
    "Note: collaborators must delete their local dataframes/craftax_human.parquet to "
    "re-download (download_dataframes.py skips existing files)."
  )


if __name__ == "__main__":
  main()
