########################
## Save raw episode data
########################
## Initialize HuggingFace API
# api = HfApi()

# def save_model_data(
#  path,
#  models,
#  dataset_name_prefix="model_episodes",
#  commit_message="Model episode data",
# ):
#  """
#  Upload safetensor episode metadata files directly to HuggingFace Hub.

#  Args:
#    path: Base path containing the model files
#    models: List of model names to process
#    dataset_name_prefix: Prefix for the dataset name (e.g., "jaxmaze_episodes")
#    commit_message: Commit message for the upload
#  """
#  model_files = []
#  for model in models:
#    model_files.extend([
#      f"{path}/final/{model}_episode_metadata.safetensor",
#      f"{path}/final/{model}_episodes.safetensor",
#      f"{path}/final/{model}_episode_df.csv",
#    ])
#    assert os.path.exists(model_files[-1]), f"File {model_files[-1]} does not exist"

#  # Create single repository for all models
#  repo_id = f"wcarvalho/{dataset_name_prefix}"
#  print(f"Creating/accessing repository: {repo_id}")
#  api.create_repo(repo_id=repo_id, exist_ok=True)

#  # Upload all model files to the same repository
#  for model, model_file in zip(models, model_files):
#    print(f"Uploading {model} from {model_file}")

#    # Upload the file with model name in the filename
#    filename = model_file.split("/")[-1]
#    api.upload_file(
#      path_or_fileobj=model_file,
#      path_in_repo=filename,
#      repo_id=repo_id,
#      commit_message=f"{commit_message} - {filename}",
#    )
#    print(f"Successfully uploaded {filename} data to {repo_id}")

#  print(f"All models uploaded to {repo_id}")

# def save_human_data(
#  path,
#  models,
#  dataset_name_prefix="human_episodes",
#  commit_message="human episode data",
# ):
#  """
#  Upload safetensor episode metadata files directly to HuggingFace Hub.

#  Args:
#    path: Base path containing the model files
#    models: List of model names to process
#    dataset_name_prefix: Prefix for the dataset name (e.g., "jaxmaze_episodes")
#    commit_message: Commit message for the upload
#  """
#  model_files = [
#      f"{path}/final/human_data_episode_metadata.safetensor",
#      f"{path}/final/human_data_episodes.safetensor",
#      f"{path}/final/human_data_episode_df.csv",
#    ]
#  # Create single repository for all models
#  repo_id = f"wcarvalho/{dataset_name_prefix}"
#  print(f"Creating/accessing repository: {repo_id}")
#  api.create_repo(repo_id=repo_id, exist_ok=True)

#  # Upload all model files to the same repository
#  for model, model_file in zip(models, model_files):
#    print(f"Uploading {model} from {model_file}")

#    # Upload the file with model name in the filename
#    filename = model_file.split("/")[-1]
#    api.upload_file(
#      path_or_fileobj=model_file,
#      path_in_repo=filename,
#      repo_id=repo_id,
#      commit_message=f"{commit_message} - {filename}",
#    )
#    print(f"Successfully uploaded {filename} data to {repo_id}")

#  print(f"All models uploaded to {repo_id}")


# prefix = 'MultitaskPreplay'

########################
## Save human data
########################
# jaxmaze_human_data_df = get_jaxmaze_human_data()
# save_df(
#  df=jaxmaze_human_data_df,
#  dataset_name=f"{prefix}_jaxmaze_human",
#  commit_message="JaxMaze human data v1",
# )
##craftax_human_data_df = get_craftax_human_data()
##save_df(
##  df=craftax_human_data_df,
##  dataset_name=f"{prefix}_craftax_human",
##  commit_message="Craftax human data v1",
##)

#########################
### Save model data
#########################
### For JaxMaze models:
# jaxmaze_models = [
#  "qlearning",
#  #"usfa",
#  #"dyna",
#  #"preplay",
#  #"bfs",
#  #"dfs"
# ]
# save_model_data(
#  path=data_configs.JAXMAZE_DATA_DIR,
#  models=jaxmaze_models,
#  dataset_name_prefix=f"{prefix}_jaxmaze_model_data",
#  commit_message="JaxMaze model episode data v1"
# )

### For Craftax models:
##craftax_models = ["qlearning", "usfa", "dyna", "preplay"]  # example
##save_model_data(
##  path=data_configs.CRAFTAX_DATA_DIR,
##  models=craftax_models,
##  dataset_name_prefix=f"{prefix}_craftax_model_data",
##  commit_message="Craftax model episode data v1"
##)
