"""Upload processed dataframes (parquets) to HuggingFace.

Each repo is updated in one atomic commit containing:
- ``data/{split}-00000-of-00001.parquet`` — the split files that
  ``analysis/download_dataframes.py`` (and ``datasets.load_dataset``) read.
- ``{split}.parquet`` — a top-level convenience copy for direct download.
- ``README.md`` — split config for the HF dataset viewer and modern
  ``datasets`` versions.

Human data has PII columns removed before upload.
"""

import os
import sys
import tempfile

import polars as pl
from huggingface_hub import CommitOperationAdd, HfApi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_configs

api = HfApi()

PII_COLUMNS = ["worker_id", "hit_id", "assignment_id"]

GITHUB_URL = "https://github.com/wcarvalho/multitask_preplay"


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


def upload_splits(repo_id: str, title: str, split_files: dict[str, str]):
  """Upload split parquets + README to *repo_id* in a single commit."""
  api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

  operations = []
  for split, local_path in split_files.items():
    operations.append(
      CommitOperationAdd(
        path_in_repo=f"data/{split}-00000-of-00001.parquet",
        path_or_fileobj=local_path,
      )
    )
    # Top-level convenience copy.
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
    commit_message="Upload dataframes",
  )
  print(f"  {repo_id}: committed {info.oid[:8]} ({len(split_files)} splits)")


def save_df(human_df, models, domain, dataset_name):
  with tempfile.TemporaryDirectory() as tmpdir:
    # Upload human data (remove PII). Split is named `human_data`;
    # download_dataframes.py maps it to `{domain}_human.parquet`.
    existing_cols = [c for c in PII_COLUMNS if c in human_df.columns]
    human_path = os.path.join(tmpdir, "human_data.parquet")
    human_df.drop(existing_cols).write_parquet(human_path)
    upload_splits(
      f"wcarvalho/{dataset_name}_human",
      f"Multitask Preplay — {domain} human data",
      {"human_data": human_path},
    )

  # Upload model data (local parquets, byte-for-byte).
  upload_splits(
    f"wcarvalho/{dataset_name}_models",
    f"Multitask Preplay — {domain} model data",
    {model: data_configs.get_dataframe_path(domain, model) for model in models},
  )


#######################
# JaxMaze
#######################
jaxmaze_models = ["qlearning", "usfa", "dyna", "preplay", "her", "bfs", "dfs"]

print("Uploading JaxMaze data...")
save_df(
  human_df=pl.read_parquet(data_configs.get_dataframe_path("jaxmaze", "human")),
  models=jaxmaze_models,
  domain="jaxmaze",
  dataset_name=data_configs.HUGGINGFACE_JAXMAZE_DATASET_NAME,
)

#######################
# Craftax
#######################
craftax_models = [
  "qlearning",
  "usfa",
  "dyna",
  "preplay",
  "her",
]

print("Uploading Craftax data...")
save_df(
  human_df=pl.read_parquet(data_configs.get_dataframe_path("craftax", "human")),
  models=craftax_models,
  domain="craftax",
  dataset_name=data_configs.HUGGINGFACE_CRAFTAX_DATASET_NAME,
)

print("Done!")
