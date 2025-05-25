"""
This script processes the user data and saves it to a parquet file.

Call from root directory with:
python data_processing/process_model_data.py --env jaxmaze --df
python data_processing/process_model_data.py --env craftax --df


"""

import os
import os.path
import sys

sys.path.append("simulations")

from glob import glob
from typing import List, Optional, Dict, Callable

# Third-party imports
from flax import serialization, struct
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import functools
import nicewebrl
import pickle
from flax.traverse_util import unflatten_dict
from safetensors.flax import load_file
from flax.core import FrozenDict

from jaxneurorl.agents import value_based_basics as vbb
from jaxneurorl.agents import qlearning


# Local application/library specific imports
from housemaze.human_dyna import sf_task_runner
from housemaze.human_dyna import utils
from housemaze.human_dyna import multitask_env
from housemaze.human_dyna import mazes
from housemaze.human_dyna import web_env
import data_configs
from data_processing import utils_jaxmaze
from data_processing import utils_craftax
from data_processing.utils import get_in_episode
from data_processing.utils import EpisodeData, load_episode_data
from simulations.craftax_web_env import CraftaxMultiGoalSymbolicWebEnvNoAutoReset
from simulations import craftax_simulation_configs
from simulations.networks import CategoricalHouzemazeObsEncoder
from data_processing.utils import add_reuse_dicts_to_df
from tqdm.auto import tqdm

################################################################
# Helper functions
################################################################


def success(e: EpisodeData):
  in_episode = get_in_episode(e.timesteps)
  rewards = e.timesteps.reward[in_episode]
  # return rewards
  assert rewards.ndim == 1, "this is only defined over vector, e.g. 1 episode"
  success = rewards > 0.5
  return success.any().astype(np.float32)


def path_length(e: EpisodeData):
  in_episode = get_in_episode(e.timesteps)
  return sum(in_episode)


def total_reward(e: EpisodeData):
  in_episode = get_in_episode(e.timesteps)
  return e.timesteps.reward[in_episode].sum()


def make_epsilon_greedy_actor(config, agent, rng, epsilon: float = 0.15):
  epsilons = jnp.full(config["NUM_ENVS"], epsilon)
  explorer = qlearning.FixedEpsilonGreedy(epsilons)

  def step(train_state, agent_state, timestep, rng):
    preds, agent_state = agent.apply(train_state.params, agent_state, timestep, rng)
    action = explorer.choose_actions(preds.q_vals, train_state.timesteps, rng)
    return preds, action, agent_state

  return vbb.Actor(train_step=step, eval_step=step)


def execute_trajectory(
  runner_state: vbb.RunnerState,
  num_steps: int,
  actor_step_fn: vbb.ActorStepFn,
  actions: jax.Array,
  env_step_fn: vbb.EnvStepFn,
  env_params: struct.PyTreeNode,
):
  def _env_step(state: vbb.RunnerState, action):
    """_summary_

    Buffer is updated with:
    - input agent state: s_{t-1}
    - agent obs input: x_t
    - agent prediction outputs: p_t
    - agent's action: a_t

    Args:
        rs (RunnerState): _description_
        unused (_type_): _description_

    Returns:
        _type_: _description_
    """
    # things that will be used/changed
    rng = state.rng
    prior_timestep = state.timestep
    prior_agent_state = state.agent_state

    # prepare rngs for actions and step
    rng, rng_a, rng_s = jax.random.split(rng, 3)

    preds, ingored_action, agent_state = actor_step_fn(
      state.train_state, prior_agent_state, prior_timestep, rng_a
    )

    transition = vbb.Transition(
      prior_timestep,
      action=action,
      extras=FrozenDict(preds=preds, agent_state=prior_agent_state),
    )

    # take step in env
    timestep = env_step_fn(rng_s, prior_timestep, action, env_params)

    state = state._replace(
      timestep=timestep,
      agent_state=agent_state,
      rng=rng,
    )

    return state, transition

  return jax.lax.scan(f=_env_step, init=runner_state, xs=actions, length=num_steps)


###################
# Search Algorithms
###################
def actions_from_search(rng, env_params, task, algo, budget):
  algo_fn = getattr(utils, algo)
  map_init = jax.tree_util.tree_map(lambda x: x[0], env_params.reset_params.map_init)
  grid = np.asarray(map_init.grid)
  agent_pos = tuple(int(o) for o in map_init.agent_pos)
  goal = np.array([task])
  path, _ = algo_fn(grid, agent_pos, goal, key=rng, budget=budget)
  actions = utils.actions_from_path(path)
  return actions


def collect_search_episodes(
  env, env_params, algorithm: str, rng, budget=None, n: int = 100
):
  budget = budget or 1e8
  default_init_timestep = env.reset(rng, env_params)
  task = default_init_timestep.state.task_object

  @jax.jit
  def concat_first_rest(first, rest):
    """concat first pytree with sequence of pytrees
    Args:
        first (struct.PyTree): [...]
        rest (struct.PyTree): [T, ...]

    Returns:
        struct.PyTree: [T+1, ...]
    """

    def concat_pytrees(tree1, tree2, **kwargs):
      return jax.tree_util.tree_map(
        lambda x, y: jnp.concatenate((x, y), **kwargs), tree1, tree2
      )

    def add_time(v):
      return jax.tree_util.tree_map(lambda x: x[None], v)

    return concat_pytrees(add_time(first), rest)

  @jax.jit
  def step_fn(carry, action):
    rng, timestep = carry
    rng, step_rng = jax.random.split(rng)
    next_timestep = env.step(step_rng, timestep, action, env_params)
    return (rng, next_timestep), next_timestep

  @jax.jit
  def collect_episode(actions, rng):
    init_timestep = env.reset(rng, env_params)
    initial_carry = (rng, init_timestep)
    (rng, _), timesteps = jax.lax.scan(step_fn, initial_carry, actions)
    init_timestep = jax.tree_util.tree_map(jnp.asarray, init_timestep)
    timesteps = jax.tree_util.tree_map(jnp.asarray, timesteps)
    return concat_first_rest(init_timestep, timesteps)

  #######################
  # first get actions from n different runs
  #######################
  all_actions = []
  rngs = jax.random.split(rng, n)

  # First, get all actions
  for idx in tqdm(range(n), f"{algorithm}: episodes"):
    actions = actions_from_search(
      rngs[idx], env_params, task, algo=algorithm, budget=budget
    )
    all_actions.append(actions)

  # Find the maximum length among all action sequences
  max_length = max(len(actions) for actions in all_actions)

  # Pad each action sequence to the maximum length
  padded_actions = []
  for actions in all_actions:
    padding = [0] * (max_length - len(actions))
    padded_actions.append(np.concatenate((actions, np.array(padding, dtype=np.int32))))

  # Convert to numpy array
  all_actions = np.array(padded_actions, dtype=np.int32)

  # Now compute all episodes
  # Vectorize collect_episode over batch dimension
  vmapped_collect_episode = jax.jit(jax.vmap(collect_episode))
  all_episodes = vmapped_collect_episode(all_actions[:, :-1], rngs)

  return EpisodeData(timesteps=all_episodes, actions=all_actions)


###################
# All algorithms
###################
@struct.dataclass
class Algorithm:
  config: dict = None
  train_state: Callable = None
  actor: Callable = None
  network: Callable = None
  vmap_reset_fn: Callable = None
  vmap_step_fn: Callable = None
  example_timestep: Optional[struct.PyTreeNode] = None
  eval_fn: Callable = None
  execute_fn: Callable = None
  path: str = None
  model_filename: str = None
  model_name: str = None

  def seed(self):
    return self.path.split("/")[-1]


def load_algorithm(
  config: dict,
  agent_params: dict,
  example_env_params: struct.PyTreeNode,
  env: Callable,
  make_agent: vbb.MakeAgentFn,
  make_optimizer: vbb.MakeOptimizerFn,
  make_actor: vbb.MakeActorFn,
  num_episodes: int = 1,
  max_steps: int = 600,
  path: Optional[str] = None,
  model_filename: Optional[str] = None,
  model_name: Optional[str] = None,
  overwrite: bool = True,
):
  """Loads and evaluates a trained algorithm using the same interface as make_train.

  Args:
      config (dict): Configuration dictionary
      env (environment.Environment): Environment class
      make_agent (MakeAgentFn): Function to create agent
      make_optimizer (MakeOptimizerFn): Function to create optimizer
      make_actor (MakeActorFn): Function to create actor
      env_params (Optional[environment.EnvParams]): Test environment parameters
      tasks (Optional[List[int]]): List of tasks to evaluate. Defaults to [0].
      n_episodes (int): Number of episodes per task. Defaults to 1.
      path (Optional[str]): Path to saved model
      name (Optional[str]): Name of model file

  Returns:
      List[List[EpisodeData]]: Evaluation episodes for each task
  """

  config["NUM_ENVS"] = num_episodes

  def vmap_reset(rng, env_params):
    return jax.vmap(env.reset, in_axes=(0, None))(
      jax.random.split(rng, num_episodes), env_params
    )

  def vmap_step(rng, env_state, action, env_params):
    return jax.vmap(env.step, in_axes=(0, 0, 0, None))(
      jax.random.split(rng, num_episodes), env_state, action, env_params
    )

  # Initialize environment and agent
  rng = jax.random.PRNGKey(config["SEED"])
  rng, rng_ = jax.random.split(rng)
  example_timestep = vmap_reset(rng_, example_env_params)

  # Create agent
  agent, init_params, reset_fn = make_agent(
    config=config,
    env=env,
    env_params=example_env_params,
    example_timestep=example_timestep,
    rng=rng_,
  )

  # Create actor
  rng, rng_ = jax.random.split(rng)
  actor = make_actor(config=config, agent=agent, rng=rng_)

  # Initialize train state
  train_state = vbb.CustomTrainState.create(
    apply_fn=agent.apply,
    params=agent_params if overwrite else init_params,
    target_network_params=agent_params,
    tx=make_optimizer(config),  # unnecessary
  )

  @jax.jit
  def eval_episode(rng, env_params):
    """Run a single evaluation episode"""
    rng, rng_ = jax.random.split(rng)
    init_timestep = vmap_reset(rng=rng_, env_params=env_params)

    # Set task and initialize agent state
    agent_state = reset_fn(train_state.params, init_timestep, rng_)

    # Create runner state and collect trajectory
    runner_state = vbb.RunnerState(
      train_state=train_state,
      timestep=init_timestep,
      agent_state=agent_state,
      rng=rng,
    )

    _, transitions = vbb.collect_trajectory(
      runner_state=runner_state,
      num_steps=max_steps,
      actor_step_fn=actor.eval_step,
      env_step_fn=vmap_step,
      env_params=env_params,
    )

    # [T, N, ....] --> # [N, T, ....]
    transitions = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 0), transitions)

    return EpisodeData(
      timesteps=transitions.timestep,
      actions=transitions.action,
      transitions=transitions,
      positions=None,
      reaction_times=None,
    )

  @jax.jit
  def execute_fn(init_timestep, env_params, actions):
    agent_state = reset_fn(train_state.params, init_timestep, rng_)

    # Create runner state and collect trajectory
    runner_state = vbb.RunnerState(
      train_state=train_state,
      timestep=init_timestep,
      agent_state=agent_state,
      rng=rng,
    )

    _, transitions = execute_trajectory(
      runner_state=runner_state,
      num_steps=None,
      actor_step_fn=actor.eval_step,
      actions=actions,
      env_step_fn=vmap_step,
      env_params=env_params,
    )

    # [T, N, ....] --> # [N, T, ....]
    transitions = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 0), transitions)
    return EpisodeData(
      timesteps=transitions.timestep,
      actions=transitions.action,
      transitions=transitions,
      positions=transitions.timestep.state.player_position,
      reaction_times=None,
    )

  example_timestep = env.reset(rng, example_env_params)
  return Algorithm(
    config=config,
    network=agent,
    actor=actor,
    vmap_reset_fn=vmap_reset,
    vmap_step_fn=vmap_step,
    example_timestep=example_timestep,
    train_state=train_state,
    eval_fn=jax.jit(eval_episode),
    execute_fn=jax.jit(execute_fn),
    path=path,
    model_filename=model_filename,
    model_name=model_name or model_filename,
  )


def get_jaxmaze_obs_encoder(config: dict):
  return functools.partial(
    CategoricalHouzemazeObsEncoder,
    num_categories=10000,
    embed_hidden_dim=config["EMBED_HIDDEN_DIM"],
    mlp_hidden_dim=config["MLP_HIDDEN_DIM"],
    num_embed_layers=config["NUM_EMBED_LAYERS"],
    num_mlp_layers=config["NUM_MLP_LAYERS"],
    activation=config["ACTIVATION"],
    norm_type=config.get("NORM_TYPE", "none"),
  )


def load_qlearning_jaxmaze_algorithm(
  config: dict,
  agent_params: dict,
  env: Callable,
  example_env_params: struct.PyTreeNode,
  path: str,
  num_episodes: int = 1,
  max_steps: int = 150,
):
  from simulations import qlearning_housemaze

  return load_algorithm(
    config=config,
    agent_params=agent_params,
    env=env,
    example_env_params=example_env_params,
    make_agent=functools.partial(
      qlearning_housemaze.make_housemaze_agent,
      ObsEncoderCls=get_jaxmaze_obs_encoder(config),
    ),
    num_episodes=num_episodes,
    max_steps=max_steps,
    make_optimizer=qlearning_housemaze.make_optimizer,
    make_actor=make_epsilon_greedy_actor,
    path=path,
    model_name="qlearning",
  )


def load_usfa_jaxmaze_algorithm(
  config: dict,
  agent_params: dict,
  env: Callable,
  example_env_params: struct.PyTreeNode,
  path: str,
  num_episodes: int = 1,
  max_steps: int = 150,
):
  from simulations import usfa_housemaze as usfa

  char2idx, groups, task_objects = mazes.get_group_set()
  train_objects = example_env_params.reset_params.train_objects[0]
  test_objects = example_env_params.reset_params.test_objects[0]
  eval_task_runner_sf = sf_task_runner.TaskRunner(
    task_objects=task_objects, vis_coeff=0.1, radius=5
  )
  train_tasks = jnp.array([eval_task_runner_sf.task_vector(o) for o in train_objects])
  test_tasks = jnp.array([eval_task_runner_sf.task_vector(o) for o in test_objects])
  all_tasks = jnp.concatenate((train_tasks, test_tasks), axis=0)

  return load_algorithm(
    config=config,
    agent_params=agent_params,
    env=env,
    example_env_params=example_env_params,
    make_agent=functools.partial(
      usfa.make_agent,
      train_tasks=train_tasks,
      ObsEncoderCls=get_jaxmaze_obs_encoder(config),
      all_tasks=all_tasks,
    ),
    num_episodes=num_episodes,
    max_steps=max_steps,
    make_optimizer=usfa.make_optimizer,
    make_actor=make_epsilon_greedy_actor,
    path=path,
    model_name="usfa",
  )


def load_dyna_jaxmaze_algorithm(
  config: dict,
  agent_params: dict,
  env: Callable,
  example_env_params: struct.PyTreeNode,
  path: str,
  num_episodes: int = 1,
  max_steps: int = 150,
):
  from simulations import multitask_preplay_craftax_v2 as dyna

  return load_algorithm(
    config=config,
    agent_params=agent_params,
    env=env,
    example_env_params=example_env_params,
    make_agent=functools.partial(
      dyna.make_jaxmaze_multigoal_agent,
      ObsEncoderCls=get_jaxmaze_obs_encoder(config),
    ),
    num_episodes=num_episodes,
    max_steps=max_steps,
    make_optimizer=dyna.make_optimizer,
    make_actor=make_epsilon_greedy_actor,
    path=path,
    model_name="dyna",
  )


def load_preplay_jaxmaze_algorithm(
  config: dict,
  agent_params: dict,
  env: Callable,
  example_env_params: struct.PyTreeNode,
  path: str,
  num_episodes: int = 1,
  max_steps: int = 150,
):
  from simulations import multitask_preplay_housemaze as offtask_dyna

  return load_algorithm(
    config=config,
    agent_params=agent_params,
    env=env,
    example_env_params=example_env_params,
    make_agent=functools.partial(
      offtask_dyna.make_agent,
      ObsEncoderCls=get_jaxmaze_obs_encoder(config),
    ),
    num_episodes=num_episodes,
    max_steps=max_steps,
    make_optimizer=offtask_dyna.make_optimizer,
    make_actor=make_epsilon_greedy_actor,
    path=path,
    model_name="dynaq_shared",
  )

def load_preplay_new_jaxmaze_algorithm(
  config: dict,
  agent_params: dict,
  env: Callable,
  example_env_params: struct.PyTreeNode,
  path: str,
  num_episodes: int = 1,
  max_steps: int = 150,
):
  from simulations import multitask_preplay_craftax_v2 as multitask_preplay

  return load_algorithm(
    config=config,
    agent_params=agent_params,
    env=env,
    example_env_params=example_env_params,
    make_agent=functools.partial(
      multitask_preplay.make_agent,
      ObsEncoderCls=get_jaxmaze_obs_encoder(config),
    ),
    num_episodes=num_episodes,
    max_steps=max_steps,
    make_optimizer=multitask_preplay.make_optimizer,
    make_actor=make_epsilon_greedy_actor,
    path=path,
    model_name="preplat",
  )



def load_jaxmaze_search_algorithm(
  config: dict,
  agent_params: dict,
  env: Callable,
  example_env_params: struct.PyTreeNode,
  path: str,
  algorithm: str = None,
  num_episodes: int = 100,
  max_steps: int = 150,
):
  del config, agent_params
  assert algorithm in ["bfs", "dfs"]

  def eval_fn(rng, env_params):
    return collect_search_episodes(
      env=env,
      env_params=env_params,
      algorithm=algorithm,
      rng=rng,
      # budget=budget,
      n=num_episodes,
    )

  return Algorithm(
    config={},
    model_name=algorithm,
    eval_fn=eval_fn,
    path=path,
  )


def load_qlearning_craftax_algorithm(
  config: dict,
  agent_params: dict,
  env: Callable,
  example_env_params: struct.PyTreeNode,
  path: str,
  num_episodes: int = 1,
  max_steps: int = 150,
):
  from simulations import qlearning_craftax

  return load_algorithm(
    config=config,
    agent_params=agent_params,
    env=env,
    example_env_params=example_env_params,
    make_agent=qlearning_craftax.make_multigoal_craftax_agent,
    num_episodes=num_episodes,
    max_steps=max_steps,
    make_optimizer=qlearning_craftax.make_optimizer,
    make_actor=make_epsilon_greedy_actor,
    path=path,
    model_name="qlearning",
  )


def load_usfa_craftax_algorithm(
  config: dict,
  agent_params: dict,
  env: Callable,
  example_env_params: struct.PyTreeNode,
  path: str,
  num_episodes: int = 1,
  max_steps: int = 150,
):
  from simulations import usfa_craftax
  from simulations.craftax_web_env import active_task_vectors

  all_tasks = jnp.concatenate(
    (active_task_vectors, jnp.zeros_like(active_task_vectors)), axis=-1
  )

  return load_algorithm(
    config=config,
    agent_params=agent_params,
    env=env,
    example_env_params=example_env_params,
    make_agent=functools.partial(
      usfa_craftax.make_multigoal_craftax_agent,
      all_tasks=all_tasks,
    ),
    num_episodes=num_episodes,
    max_steps=max_steps,
    make_optimizer=usfa_craftax.make_optimizer,
    make_actor=make_epsilon_greedy_actor,
    path=path,
    model_name="usfa",
  )


def load_dyna_craftax_algorithm(
  config: dict,
  agent_params: dict,
  env: Callable,
  example_env_params: struct.PyTreeNode,
  path: str,
  num_episodes: int = 1,
  max_steps: int = 150,
):
  from simulations import dyna_craftax as dyna

  return load_algorithm(
    config=config,
    agent_params=agent_params,
    env=env,
    example_env_params=example_env_params,
    make_agent=dyna.make_agent,
    num_episodes=num_episodes,
    max_steps=max_steps,
    make_optimizer=dyna.make_optimizer,
    make_actor=make_epsilon_greedy_actor,
    path=path,
    model_name="dyna",
  )


def load_preplay_craftax_algorithm(
  config: dict,
  agent_params: dict,
  env: Callable,
  example_env_params: struct.PyTreeNode,
  path: str,
  num_episodes: int = 1,
  max_steps: int = 150,
):
  from simulations import multitask_preplay_craftax_v2 as preplay

  return load_algorithm(
    config=config,
    agent_params=agent_params,
    env=env,
    example_env_params=example_env_params,
    make_agent=preplay.make_craftax_multigoal_agent,
    num_episodes=num_episodes,
    max_steps=max_steps,
    make_optimizer=preplay.make_optimizer,
    make_actor=make_epsilon_greedy_actor,
    path=path,
    model_name="preplay",
  )


def load_algorithm_ckpt_params(filename: str):
  flattened_dict = load_file(filename)
  params = unflatten_dict(flattened_dict, sep=",")

  return params


def generate_model_data(
  input_glob_pattern: str,
  output_data_path: str,
  example_timestep,
  env_name: str,
  model_name: str,
  overwrite_episode_data=False,
  overwrite_episode_df=False,
  load_df_only=True,
  debug=False,
  load_algorithm_fn: Callable = None,
  env: Callable = None,
  example_env_params: struct.PyTreeNode = None,
  extras: dict = None,
):
  """Process Model data for a specific environment.

  This function handles both loading pre-processed data and processing raw data.

  Difference from user data:
  - metadata for episodes is saved as safetensor, not json

  Args:
    data_path: Path to the raw data files
    human_data_filebase: Base path for saving/loading processed data
    example_timestep: Example timestep for deserialization
    env_name: Environment name ("jaxmaze" or "craftax")
    overwrite_episode_data: Whether to reprocess episode data
    overwrite_episode_df: Whether to reprocess episode info
    load_df_only: Whether to return only the DataFrame (not episodes)
    debug: Whether to run in debug mode (processing fewer files)
    parallel: Whether to process files in parallel

  Returns:
    Either a DataFrame of episode info or a DataFrame with episode data
  """
  ################################################################
  # Load data paths
  ################################################################

  if debug:
    all_episodes_data_filename = os.path.join(
      output_data_path, "debug", f"{model_name}_episodes.safetensor"
    )
    all_episode_metadata_filename = os.path.join(
      output_data_path, "debug", f"{model_name}_episode_metadata.safetensor"
    )
    all_episodes_df_filename = os.path.join(
      output_data_path, "debug", f"{model_name}_episode_df.csv"
    )
  else:
    all_episodes_data_filename = os.path.join(
      output_data_path, "final", f"{model_name}_episodes.safetensor"
    )
    all_episode_metadata_filename = os.path.join(
      output_data_path, "final", f"{model_name}_episode_metadata.safetensor"
    )
    all_episodes_df_filename = os.path.join(
      output_data_path, "final", f"{model_name}_episode_df.csv"
    )

  os.makedirs(os.path.dirname(all_episodes_data_filename), exist_ok=True)
  # --------------------------------
  # don't want to overwrite anything
  # --------------------------------
  data_file_exists = os.path.exists(all_episodes_data_filename)
  df_file_exists = os.path.exists(all_episodes_df_filename)
  overwrite_episode_df = (
    overwrite_episode_df or overwrite_episode_data
  )  # must redo df if data is overwritten
  if (
    not (overwrite_episode_data or overwrite_episode_df)
    and data_file_exists
    and df_file_exists
  ):
    if load_df_only:
      all_episodes_df = pl.read_csv(all_episodes_df_filename)
      return all_episodes_df
    else:
      all_episodes_df = pl.read_csv(all_episodes_df_filename)
      all_episode_data = load_episode_data(
        filename=all_episodes_data_filename, example_timestep=example_timestep
      )
      return nicewebrl.DataFrame(all_episodes_df, all_episode_data)
  # --------------------------------
  # don't want to overwrite episode data but want to overwrite episode info
  # --------------------------------
  elif (not overwrite_episode_data and data_file_exists) and (
    overwrite_episode_df or not df_file_exists
  ):
    # load all episode data from single safetensor file
    all_episode_data = load_episode_data(
      filename=all_episodes_data_filename, example_timestep=example_timestep
    )

    # load all episode metadata from single json file
    with open(all_episode_metadata_filename, "rb") as f:
      serialized_data = f.read()
      if env_name == "jaxmaze":
        example_config = utils_jaxmaze.dummy_config()
      else:
        example_config = utils_craftax.dummy_config()
      attempt1 = serialization.from_bytes(None, serialized_data)
      all_episode_metadata = serialization.from_bytes(
        [example_config] * len(attempt1), serialized_data
      )

    all_episodes_df = generate_all_episodes_df(
      all_episode_data,
      all_episode_metadata,
      env_name=env_name,
    )

    # Save updated episode info
    all_episodes_df.write_csv(all_episodes_df_filename)

    if load_df_only:
      return all_episodes_df
    else:
      return nicewebrl.DataFrame(all_episodes_df, all_episode_data)
  # --------------------------------
  # overwrite everything
  # --------------------------------
  elif (overwrite_episode_data and overwrite_episode_df) or not data_file_exists:
    paths = glob(input_glob_pattern)
    paths = sorted(paths)
    if debug:
      if isinstance(debug, bool):
        paths = paths[:1]
      elif isinstance(debug, int):
        paths = paths[debug : debug + 1]
      else:
        raise ValueError(
          f"Debug can only be an integer (specifying a seed) or bool: {debug}"
        )

    all_episode_data, all_episode_metadata = generate_all_episodes_data(
      paths=paths,
      example_timestep=example_timestep,
      env_name=env_name,
      model_name=model_name,
      debug=debug,
      parallel=True,
      env=env,
      example_env_params=example_env_params,
      extras=extras,
      load_algorithm_fn=load_algorithm_fn,
    )
    # Save using Flax serialization
    with open(all_episodes_data_filename, "wb") as f:
      serialized_data = serialization.to_bytes(all_episode_data)
      f.write(serialized_data)
      print(f"Saved {len(all_episode_data)} episodes to {all_episodes_data_filename}")
    # Save using Flax serialization
    with open(all_episode_metadata_filename, "wb") as f:
      serialized_data = serialization.to_bytes(all_episode_metadata)
      f.write(serialized_data)
      print(
        f"Saved {len(all_episode_metadata)} metadata to {all_episode_metadata_filename}"
      )

    all_episodes_df = generate_all_episodes_df(
      all_episode_data,
      all_episode_metadata,
      env_name=env_name,
    )

    # Save episode dataframe
    all_episodes_df.write_csv(all_episodes_df_filename)

    if load_df_only:
      return all_episodes_df
    else:
      return nicewebrl.DataFrame(all_episodes_df, all_episode_data)

  else:
    raise ValueError("Invalid overwrite_episode_data and overwrite_episode_info")


################################################################
# Main functions
################################################################
def generate_all_episodes_data(
  paths: List[str],
  example_timestep: struct.PyTreeNode,
  env_name: str,
  model_name: str,
  load_algorithm_fn: Callable,
  debug=False,
  parallel=True,
  env: Callable = None,
  example_env_params: struct.PyTreeNode = None,
  extras: dict = None,
):
  """Generate all episodes data for a given model. All paths belong to all seeds."""
  extras = extras or {}
  model_filename = extras.get("model_filename", model_name)
  search_algorithm = extras.get("search_algorithm", False)

  if env_name == "jaxmaze":
    generate_algorithm_episodes = utils_jaxmaze.generate_algorithm_episodes
  elif env_name == "craftax":
    generate_algorithm_episodes = utils_craftax.generate_algorithm_episodes
  else:
    raise ValueError(f"Unknown environment: {env_name}")

  if search_algorithm:
    algorithm = load_algorithm_fn(
      config=None,
      agent_params=None,
      env=env,
      example_env_params=example_env_params,
      path=None,
    )
    extras["seed"] = 42
    rng = jax.random.PRNGKey(42)
    all_episode_data, all_episode_metadata = generate_algorithm_episodes(
      algorithm, rng, extras
    )
    for episode_metadata in all_episode_metadata:
      # for search algorithms, each episode corresponds to a unique seed
      episode_metadata["seed"] = episode_metadata["episode_idx"]
    return all_episode_data, all_episode_metadata

  all_episodes = []
  all_episode_configs = []
  for path in paths:
    path = path.rstrip("/")
    params = load_algorithm_ckpt_params(f"{path}/{model_filename}.safetensors")
    with open(f"{path}/{model_filename}.config", "rb") as f:
      config = pickle.load(f)
    # remove trailing / from path
    algorithm = load_algorithm_fn(
      config=config,
      agent_params=params,
      env=env,
      example_env_params=example_env_params,
      path=path,
    )

    # update parameters for this ckpt
    # train_state = algorithm.train_state.replace(params=params)
    # algorithm = algorithm.replace(train_state=train_state)
    seed = int(path.split("seed=")[-1])
    rng = jax.random.PRNGKey(seed)
    extras["seed"] = seed
    episodes, episode_configs = generate_algorithm_episodes(algorithm, rng, extras)
    all_episodes.extend(episodes)
    all_episode_configs.extend(episode_configs)
  import ipdb; ipdb.set_trace()
  return all_episodes, all_episode_configs


def generate_all_episodes_df(all_episode_data, all_episode_metadata, env_name: str):
  assert env_name in ["jaxmaze", "craftax"], f"Unknown environment data: {env_name}"

  # Set up environment utils
  if env_name == "jaxmaze":
    env_utils = utils_jaxmaze
  elif env_name == "craftax":
    env_utils = utils_craftax
  else:
    raise ValueError(f"Unknown environment: {env_name}")

  all_episode_row_data = []
  for episode_data, episode_metadata in zip(all_episode_data, all_episode_metadata):
    row_data = env_utils.make_model_episode_row_data(
      episode=episode_data,
      metadata=episode_metadata,
    )
    row_data["global_episode_idx"] = len(all_episode_row_data)
    all_episode_row_data.append(row_data)

  all_episode_df = pl.DataFrame(all_episode_row_data)

  # Initialize empty dictionaries for reuse and overlap
  all_reuse_dicts = []
  all_overlap_dicts = []

  # Create a DataFrame for processing
  temp_df = nicewebrl.DataFrame(all_episode_df, all_episode_data)

  # Process each model and seed separately
  for model_name in all_episode_df["algo"].unique().to_list():
    model_df = temp_df.filter(algo=model_name)
    for seed in model_df["seed"].unique().to_list():
      seed_df = model_df.filter(seed=seed)
      reuse_dict, overlap_dict = env_utils.add_model_reuse_columns(seed_df)

      # Add to our collection (we'll combine them later)
      all_reuse_dicts.append(reuse_dict)
      all_overlap_dicts.append(overlap_dict)

  # Use our utility function to add the columns
  all_episode_df = add_reuse_dicts_to_df(
    all_episode_df, all_reuse_dicts, all_overlap_dicts
  )

  return all_episode_df


def generate_all_model_data(
  env_name: str,
  input_data_path: str,
  output_data_path: str,
  model_to_input_glob,
  model_to_env_objects,
  model_to_load_algorithm,
  model_to_extras: dict = None,
  overwrite_episode_data: bool = False,
  overwrite_episode_df: bool = False,
  load_df_only: bool = True,
  debug: bool = False,
  models: List[str] = None,
):
  models = models or model_to_load_algorithm.keys()
  model_dfs = []
  model_to_extras = model_to_extras or {}
  for model_name in models:
    env, example_timestep, example_env_params = model_to_env_objects[model_name]

    model_df = generate_model_data(
      input_glob_pattern=model_to_input_glob[model_name],
      output_data_path=output_data_path,
      example_timestep=example_timestep,
      env_name=env_name,
      model_name=model_name,
      load_algorithm_fn=model_to_load_algorithm[model_name],
      overwrite_episode_data=overwrite_episode_data,
      overwrite_episode_df=overwrite_episode_df,
      load_df_only=load_df_only,
      debug=debug,
      env=env,
      example_env_params=example_env_params,
      extras=model_to_extras.get(model_name, {}),
    )
    model_dfs.append(model_df)

  if load_df_only:
    return pl.concat(model_dfs)
  else:
    return nicewebrl.dataframe.concat_list(*model_dfs)


def load_jaxmaze_environment(
  num_categories: int = 200,
  load_sf_task_runner: bool = False,
  vis_coeff: float = 0.1,
):
  char2idx, groups, task_objects = mazes.get_group_set()
  example_env_params = mazes.get_maze_reset_params(
    groups=groups,
    char2key=char2idx,
    maze_str=mazes.big_practice_maze,
    randomize_agent=False,
    make_env_params=True,
  )

  if load_sf_task_runner:
    task_runner = sf_task_runner.TaskRunner(
      task_objects=task_objects, vis_coeff=vis_coeff, radius=5
    )
  else:
    task_runner = multitask_env.TaskRunner(task_objects=task_objects)

  env = web_env.HouseMaze(
    task_runner=task_runner,
    num_categories=num_categories,
  )

  dummy_rng = jax.random.PRNGKey(42)
  example_timestep = env.reset(dummy_rng, example_env_params)

  return env, example_timestep, example_env_params, task_runner


def get_jaxmaze_model_data(
  input_data_path: str = data_configs.JAXMAZE_DATA_DIR,
  output_data_path: str = data_configs.JAXMAZE_DATA_DIR,
  overwrite_episode_data=False,
  overwrite_episode_df=False,
  load_df_only: bool = True,
  model_to_input_glob: Optional[Dict[str, str]] = None,
  debug=False,
  models: List[str] = None,
):
  """Get human data for JaxMaze environment."""

  env, example_timestep, example_env_params, task_runner = load_jaxmaze_environment()
  sf_env, sf_example_timestep, sf_example_env_params, sf_task_runner = (
    load_jaxmaze_environment(load_sf_task_runner=True)
  )

  models = models or [
    'qlearning',
    'dyna',
    'usfa',
    #'preplay',
    'preplay_new',
    'bfs',
    'dfs',
  ]

  # Call the common human data function
  model_to_input_glob = model_to_input_glob or dict(
    qlearning=f"{input_data_path}/qlearning/seed=*/",
    dyna=f"{input_data_path}/dyna/seed=*/",
    usfa=f"{input_data_path}/usfa/seed=*/",
    preplay=f"{input_data_path}/preplay/seed=*/",
    preplay_new=f"{input_data_path}/preplay-new/seed=*/",
    bfs="",  # ignored
    dfs="",  # ignored
  )

  model_to_env_objects = dict(
    qlearning=(env, example_timestep, example_env_params),
    dyna=(env, example_timestep, example_env_params),
    usfa=(sf_env, sf_example_timestep, sf_example_env_params),
    preplay=(env, example_timestep, example_env_params),
    preplay_new=(env, example_timestep, example_env_params),
    bfs=(env, example_timestep, example_env_params),
    dfs=(env, example_timestep, example_env_params),
  )

  model_to_extras = dict(
    qlearning=dict(task_runner=task_runner),
    dyna=dict(task_runner=task_runner),
    usfa=dict(task_runner=sf_task_runner),
    preplay=dict(task_runner=task_runner, model_filename="dynaq_shared"),
    preplay_new=dict(task_runner=task_runner, model_filename="preplay"),
    bfs=dict(task_runner=task_runner, search_algorithm=True),
    dfs=dict(task_runner=task_runner, search_algorithm=True),
  )

  model_to_load_algorithm = dict(
    qlearning=load_qlearning_jaxmaze_algorithm,
    usfa=load_usfa_jaxmaze_algorithm,
    dyna=load_dyna_jaxmaze_algorithm,
    preplay=load_preplay_jaxmaze_algorithm,
    preplay_new=load_preplay_new_jaxmaze_algorithm,
    bfs=functools.partial(
      load_jaxmaze_search_algorithm,
      algorithm="bfs",
      max_steps=1 if debug else 100,
      num_episodes=1 if debug else 100,
    ),
    dfs=functools.partial(
      load_jaxmaze_search_algorithm,
      algorithm="dfs",
      max_steps=1 if debug else 100,
      num_episodes=1 if debug else 100,
    ),
  )

  return generate_all_model_data(
    env_name="jaxmaze",
    input_data_path=input_data_path,
    output_data_path=output_data_path,
    model_to_input_glob=model_to_input_glob,
    model_to_env_objects=model_to_env_objects,
    model_to_load_algorithm=model_to_load_algorithm,
    overwrite_episode_data=overwrite_episode_data,
    overwrite_episode_df=overwrite_episode_df,
    model_to_extras=model_to_extras,
    load_df_only=load_df_only,
    debug=debug,
    models=models,
  )


def load_craftax_environment(landmark_features=False):
  static_env_params = (
    CraftaxMultiGoalSymbolicWebEnvNoAutoReset.default_static_params().replace(
      landmark_features=landmark_features
    )
  )
  env = CraftaxMultiGoalSymbolicWebEnvNoAutoReset(static_env_params)
  env = nicewebrl.TimestepWrapper(env, autoreset=False)

  example_env_params = craftax_simulation_configs.default_params
  dummy_rng = jax.random.PRNGKey(42)
  example_timestep = env.reset(dummy_rng, example_env_params)

  return env, example_timestep, example_env_params


def get_craftax_model_data(
  input_data_path: str = data_configs.CRAFTAX_DATA_DIR,
  output_data_path: str = data_configs.CRAFTAX_DATA_DIR,
  overwrite_episode_data=False,
  overwrite_episode_df=False,
  load_df_only: bool = True,
  model_to_input_glob: Optional[Dict[str, str]] = None,
  debug=False,
  models: List[str] = None,
):
  """Get human data for Craftax environment."""

  env, example_timestep, example_env_params = load_craftax_environment()
  sf_env, sf_example_timestep, sf_example_env_params = load_craftax_environment(
    landmark_features=True
  )

  # Call the common human data function
  model_to_input_glob = model_to_input_glob or dict(
    qlearning=f"{input_data_path}/qlearning/seed=*/",
    dyna=f"{input_data_path}/dyna/seed=*/",
    usfa=f"{input_data_path}/usfa/seed=*/",
    preplay=f"{input_data_path}/preplay/seed=*/",
  )

  model_to_env_objects = dict(
    qlearning=(env, example_timestep, example_env_params),
    dyna=(env, example_timestep, example_env_params),
    usfa=(sf_env, sf_example_timestep, sf_example_env_params),
    preplay=(env, example_timestep, example_env_params),
  )

  model_to_load_algorithm = dict(
    qlearning=load_qlearning_craftax_algorithm,
    usfa=load_usfa_craftax_algorithm,
    dyna=load_dyna_craftax_algorithm,
    preplay=load_preplay_craftax_algorithm,
  )

  return generate_all_model_data(
    env_name="craftax",
    input_data_path=input_data_path,
    output_data_path=output_data_path,
    model_to_input_glob=model_to_input_glob,
    model_to_env_objects=model_to_env_objects,
    model_to_load_algorithm=model_to_load_algorithm,
    overwrite_episode_data=overwrite_episode_data,
    overwrite_episode_df=overwrite_episode_df,
    load_df_only=load_df_only,
    models=models,
    debug=debug,
  )


# Update the main function to handle the returned failed_files
if __name__ == "__main__":
  # Parse command line arguments
  import argparse

  parser = argparse.ArgumentParser(description="Process user data for experiments")
  parser.add_argument(
    "--env",
    choices=["jaxmaze", "craftax", "both"],
    default="both",
    help="Which environment to process (jaxmaze, craftax, or both)",
  )
  parser.add_argument("--episodes", action="store_true", help="Overwrite episode data")
  parser.add_argument("--df", action="store_true", help="Overwrite episode dataframe")
  parser.add_argument(
    "--debug", action="store_true", help="Run in debug mode with reduced data"
  )
  parser.add_argument(
    "--models",
    nargs="+",
    help="List of models to process (default: process all models)",
  )

  args = parser.parse_args()

  overwrite_episode_data = args.episodes
  overwrite_episode_df = args.df

  if args.env in ["jaxmaze", "both"]:
    jaxmaze_df = get_jaxmaze_model_data(
      overwrite_episode_data=overwrite_episode_data,
      overwrite_episode_df=overwrite_episode_df,
      load_df_only=True,
      debug=args.debug,
      models=args.models,
    )
    # Compute mean success rates for JaxMaze
    print("\nJaxMaze Success Rates:")
    success_rates = (
      jaxmaze_df.group_by(["algo", "eval"])
      .agg(pl.col("success").mean())
      .sort(["eval", "algo"])
    )
    print(success_rates)

  if args.env in ["craftax", "both"]:
    craftax_df = get_craftax_model_data(
      overwrite_episode_data=overwrite_episode_data,
      overwrite_episode_df=overwrite_episode_df,
      load_df_only=True,
      debug=args.debug,
      models=args.models,
    )
    # Compute mean success rates for Craftax
    print("\nCraftax Success Rates:")
    success_rates = (
      craftax_df.group_by(["algo", "eval"])
      .agg(pl.col("success").mean())
      .sort(["eval", "algo"])
    )
    print(success_rates)
  import ipdb

  ipdb.set_trace()
