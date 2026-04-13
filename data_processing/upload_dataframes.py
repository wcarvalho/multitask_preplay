from datasets import Dataset, DatasetDict
import os
import sys
from huggingface_hub import HfApi
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_configs
from data_processing.process_user_data import (
  get_jaxmaze_human_data,
  get_craftax_human_data,
)


#######################
# Save dataframes
#######################
def save_df(human_df, models, model_path, dataset_name, commit_message="Saving data"):
  # remove identifying information
  columns_to_remove = ["worker_id", "hit_id", "assignment_id"]
  human_df = human_df.drop(columns_to_remove)
  human_dataset = Dataset.from_dict(human_df.to_dict(as_series=False))

  # Create a DatasetDict with "human" as the key instead of pushing as a single dataset
  human_dataset_dict = DatasetDict({"human_data": human_dataset})
  human_dataset_dict.push_to_hub(
    f"wcarvalho/{dataset_name}_human", commit_message=commit_message
  )

  datasets = {}
  for model in models:
    model_df = pl.read_parquet(f"{model_path}/final/{model}_episode_df.parquet")
    datasets[model] = Dataset.from_dict(model_df.to_dict(as_series=False))

  # Push to hub
  dataset_dict = DatasetDict(datasets)
  dataset_dict.push_to_hub(
    f"wcarvalho/{dataset_name}_models", commit_message=commit_message
  )


#######################
# JaxMaze
#######################
human_df = get_jaxmaze_human_data()
jaxmaze_models = ["qlearning", "usfa", "dyna", "preplay", "bfs", "dfs"]

save_df(
  human_df=get_jaxmaze_human_data(),
  models=jaxmaze_models,
  model_path=data_configs.JAXMAZE_MODEL_DATA_DIR,
  dataset_name=data_configs.HUGGINGFACE_JAXMAZE_DATASET_NAME,
  commit_message="v1",
)

#######################
# Craftax
#######################
 human_df = get_craftax_human_data()
 craftax_models = [
  "qlearning",
  "usfa",
  "dyna",
  "preplay",
 ]

 save_df(
  human_df=get_craftax_human_data(),
  models=craftax_models,
  model_path=data_configs.CRAFTAX_DATA_DIR,
  dataset_name=data_configs.HUGGINGFACE_CRAFTAX_DATASET_NAME,
  commit_message="v1",
 )
