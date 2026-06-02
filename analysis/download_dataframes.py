"""Download processed data (parquets) from HuggingFace.

Usage:
  python analysis/download_dataframes.py                # download both
  python analysis/download_dataframes.py --env jaxmaze  # jaxmaze only
  python analysis/download_dataframes.py --env craftax   # craftax only
"""

import argparse
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import HfApi, hf_hub_download

import data_configs

# One parquet per split, named data/{split}-00000-of-00001.parquet.
_SPLIT_FILE_RE = re.compile(r"data/(\w+)-\d{5}-of-\d{5}\.parquet")


def _has_local_data(domain: str, human: bool) -> bool:
  """Return True if local parquets already exist for this portion of the data.

  The human and model parquets come from separate HuggingFace repos, so check
  them separately: the human repo provides ``{domain}_human.parquet``; the
  models repo provides every other ``{domain}_*.parquet``.
  """
  import glob

  pattern = os.path.join(data_configs.DATAFRAMES_DIR, f"{domain}_*.parquet")
  files = glob.glob(pattern)
  human_file = data_configs.get_dataframe_path(domain, "human")
  if human:
    return human_file in files
  return any(f != human_file for f in files)


def download_data(
  dataset_name: str,
  domain: str,
):
  human = dataset_name.endswith("_human")
  if _has_local_data(domain, human=human):
    print(f"Local data found for {domain}, skipping download.")
    return

  # Copy the split parquets byte-for-byte. (Avoid `datasets.load_dataset`: its
  # pandas round-trip silently converts nulls to NaN, which polars treats
  # differently in filters and aggregations.)
  repo_id = f"wcarvalho/{dataset_name}"
  repo_files = HfApi().list_repo_files(repo_id, repo_type="dataset")

  os.makedirs(data_configs.DATAFRAMES_DIR, exist_ok=True)

  for path_in_repo in repo_files:
    match = _SPLIT_FILE_RE.fullmatch(path_in_repo)
    if match is None:
      continue
    # Map HuggingFace split names to our naming convention
    model_name = match.group(1).replace("human_data", "human")
    filename = data_configs.get_dataframe_path(domain, model_name)
    if os.path.exists(filename):
      print(f"Skipping {domain}_{model_name} because it already exists")
      continue
    cached = hf_hub_download(
      repo_id=repo_id, filename=path_in_repo, repo_type="dataset"
    )
    shutil.copyfile(cached, filename)
    print(f"Saved {domain}_{model_name} to {filename}")


def download_jaxmaze_data():
  download_data(
    dataset_name=f"{data_configs.HUGGINGFACE_JAXMAZE_DATASET_NAME}_human",
    domain="jaxmaze",
  )
  download_data(
    dataset_name=f"{data_configs.HUGGINGFACE_JAXMAZE_DATASET_NAME}_models",
    domain="jaxmaze",
  )


def download_craftax_data():
  download_data(
    dataset_name=f"{data_configs.HUGGINGFACE_CRAFTAX_DATASET_NAME}_human",
    domain="craftax",
  )
  download_data(
    dataset_name=f"{data_configs.HUGGINGFACE_CRAFTAX_DATASET_NAME}_models",
    domain="craftax",
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Download processed data from HuggingFace"
  )
  parser.add_argument(
    "--env",
    choices=["jaxmaze", "craftax", "both"],
    default="both",
    help="Which environment data to download",
  )
  args = parser.parse_args()

  if args.env in ["jaxmaze", "both"]:
    download_jaxmaze_data()

  if args.env in ["craftax", "both"]:
    download_craftax_data()
