# Preemptive Solving of Future Problems: Multitask Preplay in Humans and Machines
This repository is the official implementation of [Preemptive Solving of Future Problems: Multitask Preplay in Humans and Machines](link).

## Install
```
conda create -n multitask python=3.10 pip wheel -y
conda activate multitask
pip install -r requirements.txt
```

## Running web experiments

Note: before running a new experiment you want to delete `.nicegui`

**JaxMaze experiment**
```
# Two Paths Manipulation (prediction 1)
python experiments/jaxmaze/web_app.py MAN="paths"

# Juncture Manipulation (prediction 2)
python experiments/jaxmaze/web_app.py MAN="plan" SAY_REUSE=1  # known goals
python experiments/jaxmaze/web_app.py MAN="plan" SAY_REUSE=0  # unknown goals

# Start Manipulation (prediction 3)
python experiments/jaxmaze/web_app.py MAN="start"

# Shortcut Manipulation (prediction 4)
python experiments/jaxmaze/web_app.py MAN="shortcut"
```

**Craftax experiment**
Before running experiments, run `python experiments/craftax/load_caches.py` to load caches (this will take 20-40 minutes)
```
# known evaluation goals
python experiments/craftax/web_app.py SAY_REUSE=1

# unknown evaluation goals
python experiments/craftax/web_app.py SAY_REUSE=0
```



## Data Structure

The root directory for all results is set in `data_configs.py` with the `DIRECTORY` variable. Change this to somewhere on your local machine.
**Results**


**Model and Participant Data**
```
data_jaxmaze/
- human_data/
- human_data_episodes.safetensor
- human_data_episode_information.csv
- qlearning/
  - seed=1/
    - qlearning.config       # run settings
    - qlearning.safetensors  # parameters
  - seed=2/
  ...
- qlearning_episodes.safetensor
- qlearning_episode_information.csv
...

data_craftax/
- human_data/
- human_data_episodes.safetensor
- human_data_episode_information.csv
- qlearning/
  - seed=1/
    - qlearning.config       # run settings
    - qlearning.safetensors  # parameters
  - seed=2/
  ...
- qlearning_episodes.safetensor
- qlearning_episode_information.csv
...
```