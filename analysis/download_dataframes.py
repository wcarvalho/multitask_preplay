"""Download processed data (parquets) from HuggingFace.

Usage:
  python analysis/download_dataframes.py                # download both
  python analysis/download_dataframes.py --env jaxmaze  # jaxmaze only
  python analysis/download_dataframes.py --env craftax   # craftax only
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

import data_configs


def _has_local_data(domain: str) -> bool:
  """Return True if at least one parquet already exists for this domain."""
  import glob

  pattern = os.path.join(data_configs.DATAFRAMES_DIR, f"{domain}_*.parquet")
  return len(glob.glob(pattern)) > 0


def download_data(
  dataset_name: str,
  domain: str,
):
  if _has_local_data(domain):
    print(f"Local data found for {domain}, skipping download.")
    return

  dataset = load_dataset(f"wcarvalho/{dataset_name}")

  os.makedirs(data_configs.DATAFRAMES_DIR, exist_ok=True)

  for split_name, split_data in dataset.items():
    # Map HuggingFace split names to our naming convention
    model_name = split_name.replace("human_data", "human")
    filename = data_configs.get_dataframe_path(domain, model_name)
    if os.path.exists(filename):
      print(f"Skipping {domain}_{model_name} because it already exists")
      continue
    split_data.to_pandas().to_parquet(filename)
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
