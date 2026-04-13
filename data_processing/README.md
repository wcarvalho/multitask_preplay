# Step-by-step instructions for full pipeline from downloading off server to uploading on huggingface

Base data directory: `$MULTITASK_PREPLAY_DATA_DIR` (default `../preplay_results/`)


### Human data

Download human data from Google Cloud Storage
* JaxMaze: from bucket `human-dyna` → `data/jaxmaze/human_data/*.json`
* Craftax: from bucket `craftax-human-dyna` → `data/craftax/human_data/*.json`
```
python data_processing/download_user_data_google.py
```

Process raw JSON into episodes and DataFrames
* `--episodes` output per env:
  * `{env}/final/human_data_episodes.safetensor` (e.g. `craftax/final/human_data_episodes.safetensor`)
  * `{env}/final/human_data_episode_metadata.json` (e.g. `craftax/final/human_data_episode_metadata.json`)
* `--df` output per env:
  * `{env}/final/human_data_episode_df.parquet` (e.g. `craftax/final/human_data_episode_df.parquet`)
```
python data_processing/process_user_data.py --env jaxmaze --episodes --df
python data_processing/process_user_data.py --env craftax --episodes --df
```


### Model data

Download model weights from Harvard Kempner SLURM cluster via rsync
* Output per model:
  * `data/{env}/{model_name}/{model_name}_seed={N}.safetensors` (e.g. `data/craftax/preplay/preplay_seed=0.safetensors`)
  * `data/{env}/{model_name}/{model_name}_seed={N}.config` (e.g. `data/craftax/preplay/preplay_seed=0.config`)
* JaxMaze models: qlearning, usfa, dyna, preplay
* Craftax models: qlearning, usfa, dyna, preplay
```
python data_processing/download_model_data_slurm.py --env jaxmaze
python data_processing/download_model_data_slurm.py --env craftax
```

Generate episodes from model parameters
* Output per model:
  * `{env}/final/{model}_episodes.safetensor` (e.g. `craftax/final/preplay_episodes.safetensor`)
  * `{env}/final/{model}_episode_metadata.safetensor` (e.g. `craftax/final/preplay_episode_metadata.safetensor`)
  * `{env}/final/{model}_episode_df.parquet` (e.g. `craftax/final/preplay_episode_df.parquet`)
* JaxMaze models: qlearning, usfa, dyna, preplay, bfs, dfs
* Craftax models: qlearning, usfa, dyna, preplay
```
python data_processing/process_model_data.py --env jaxmaze --episodes --df
python data_processing/process_model_data.py --env craftax --episodes --df
```
