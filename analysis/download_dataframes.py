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


def download_data(
  data_dir: str,
  dataset_name: str,
):
  dataset = load_dataset(f"wcarvalho/{dataset_name}")

  # Create directories
  os.makedirs(os.path.join(data_dir, "final"), exist_ok=True)

  # Save each split as CSV
  for split_name, split_data in dataset.items():
    filename = os.path.join(data_dir, "final", f"{split_name}_episode_df.parquet")
    if os.path.exists(filename):
      print(f"Skipping {split_name} data because it already exists")
      continue
    split_data.to_pandas().to_parquet(filename)
    print(f"Saved {split_name} data to {filename}")


def download_jaxmaze_data():
  print("Data will be downloaded or saved to", data_configs.JAXMAZE_MODEL_DATA_DIR)
  download_data(
    data_dir=data_configs.JAXMAZE_MODEL_DATA_DIR,
    dataset_name=f"{data_configs.HUGGINGFACE_JAXMAZE_DATASET_NAME}_human",
  )
  download_data(
    data_dir=data_configs.JAXMAZE_MODEL_DATA_DIR,
    dataset_name=f"{data_configs.HUGGINGFACE_JAXMAZE_DATASET_NAME}_models",
  )


def download_craftax_data():
  print("Data will be downloaded or saved to", data_configs.CRAFTAX_MODEL_DATA_DIR)
  download_data(
    data_dir=data_configs.CRAFTAX_MODEL_DATA_DIR,
    dataset_name=f"{data_configs.HUGGINGFACE_CRAFTAX_DATASET_NAME}_human",
  )
  download_data(
    data_dir=data_configs.CRAFTAX_MODEL_DATA_DIR,
    dataset_name=f"{data_configs.HUGGINGFACE_CRAFTAX_DATASET_NAME}_models",
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
