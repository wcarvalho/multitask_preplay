import os
import sys
import tempfile

import polars as pl
from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_configs

api = HfApi()


#######################
# Save dataframes
#######################
def upload_parquet(df, repo_id, filename):
  """Upload a polars DataFrame as a parquet file to HuggingFace."""
  with tempfile.NamedTemporaryFile(suffix=".parquet", delete=True) as f:
    df.write_parquet(f.name)
    api.upload_file(
      path_or_fileobj=f.name,
      path_in_repo=filename,
      repo_id=repo_id,
      repo_type="dataset",
    )
  print(f"  Uploaded {filename} to {repo_id}")


def save_df(human_df, models, domain, dataset_name):
  repo_human = f"wcarvalho/{dataset_name}_human"
  repo_models = f"wcarvalho/{dataset_name}_models"

  # Create repos if they don't exist
  api.create_repo(repo_human, repo_type="dataset", exist_ok=True)
  api.create_repo(repo_models, repo_type="dataset", exist_ok=True)

  # Upload human data (remove PII)
  columns_to_remove = ["worker_id", "hit_id", "assignment_id"]
  existing_cols = [c for c in columns_to_remove if c in human_df.columns]
  human_df = human_df.drop(existing_cols)
  upload_parquet(human_df, repo_human, "human_data.parquet")

  # Upload model data
  for model in models:
    model_df = pl.read_parquet(data_configs.get_dataframe_path(domain, model))
    upload_parquet(model_df, repo_models, f"{model}.parquet")


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
