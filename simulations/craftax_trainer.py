"""

TESTING:
JAX_DEBUG_NANS=True \
JAX_DISABLE_JIT=1 \
HYDRA_FULL_ERROR=1 JAX_TRACEBACK_FILTERING=off python -m ipdb -c continue craftax_trainer.py \
  app.parallel=none \
  app.debug=True \
  app.wandb=False \
  app.search=ql

RUNNING ON SLURM:
python craftax_trainer.py \
  app.parallel=slurm \
  app.search=usfa
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

MAX_SCORE = 226.0


import wandb
from functools import partial
from jaxneurorl import loggers
from jaxneurorl import utils
from jaxneurorl import launcher
from jaxneurorl.agents import value_based_pqn as vpq
from jaxneurorl.agents import value_based_basics as vbb
from typing import Any, Callable, Dict, Union, Optional
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.constants import Achievement

import jax
import chex
from flax import struct
import functools
import jax.numpy as jnp
import time


import hydra
from omegaconf import DictConfig


from safetensors.flax import save_file
from flax.traverse_util import flatten_dict

import numpy as np


from jaxneurorl.agents import value_based_pqn as pqn
from jaxneurorl.agents import value_based_basics as vbb
from jaxneurorl.wrappers import TimestepWrapper

import craftax_observer
import networks
import alphazero_craftax
import dyna_craftax
import multitask_preplay_craftax_v2
import qlearning_craftax
import qlearning_sf_aux_craftax
import usfa_craftax as usfa


@struct.dataclass
class AlgorithmConstructor:
  make_agent: Callable
  make_optimizer: Callable
  make_loss_fn_class: Callable
  make_actor: Callable


def get_pqn_fns(config):
  encoder = functools.partial(
    networks.CraftaxObsEncoder,
    # pqn.MLP,
    hidden_dim=config["MLP_HIDDEN_DIM"],
    num_layers=config["NUM_MLP_LAYERS"],
    activation=config["ACTIVATION"],
    norm_type=config.get("NORM_TYPE", "batch_norm"),
  )

  return AlgorithmConstructor(
    make_agent=functools.partial(pqn.make_agent, ObsEncoderCls=encoder),
    make_optimizer=pqn.make_optimizer,
    make_loss_fn_class=pqn.make_loss_fn_class,
    make_actor=pqn.make_actor,
  )


@struct.dataclass
class LogEnvState:
  env_state: Any
  episode_returns: float
  episode_lengths: int
  returned_episode_returns: float
  returned_episode_lengths: int
  timestep: int


class GymnaxWrapper(object):
  """Base class for Gymnax wrappers."""

  def __init__(self, env):
    self._env = env

  # provide proxy access to regular attributes of wrapped object
  def __getattr__(self, name):
    return getattr(self._env, name)


class LogWrapper(GymnaxWrapper):
  """Log the episode returns and lengths."""

  def __init__(self, env):
    super().__init__(env)

  @partial(jax.jit, static_argnums=(0, 2))
  def reset(self, key: chex.PRNGKey, params=None):
    obs, env_state = self._env.reset(key, params)
    # NOTE: change, technically episode length is 1 here, not 0
    state = LogEnvState(env_state, 0.0, 1, 0.0, 0, 0)
    return obs, state

  @partial(jax.jit, static_argnums=(0, 4))
  def step(
    self,
    key: chex.PRNGKey,
    state,
    action: Union[int, float],
    params=None,
  ):
    """
    # NOTE: MAJOR change from original code is that they assume that finishing an episode leads to the first state of the next episode in the same index.
    In this codebase, when you finish an episode, you see the first state of the next episode in the NEXT index.
    """

    obs, env_state, reward, done, info = self._env.step(
      key, state.env_state, action, params
    )
    new_episode_return = state.episode_returns + reward
    new_episode_length = state.episode_lengths + 1
    done_float = done.astype(jnp.float32)
    done_int = done.astype(jnp.int32)
    state = LogEnvState(
      env_state=env_state,
      episode_returns=new_episode_return * (1 - done_float),
      episode_lengths=new_episode_length * (1 - done_int),
      returned_episode_returns=(
        state.returned_episode_returns * (1 - done_float)
        + new_episode_return * done_float
      ),
      returned_episode_lengths=state.returned_episode_lengths * (1 - done_int)
      + new_episode_length * done_int,
      timestep=state.timestep + 1,
    )
    return obs, state, reward, done, info


class OptimisticResetVecEnvWrapper(object):
  """
  Provides efficient 'optimistic' resets.
  The wrapper also necessarily handles the batching of environment steps and resetting.
  reset_ratio: the number of environment workers per environment reset.  Higher means more efficient but a higher
  chance of duplicate resets.
  """

  def __getattr__(self, name):
    # provide proxy access to regular attributes of wrapped object
    return getattr(self._env, name)

  def __init__(self, env, num_envs: int, reset_ratio: int):
    self._env = env

    self.num_envs = num_envs
    self.reset_ratio = reset_ratio
    assert num_envs % reset_ratio == 0, "Reset ratio must perfectly divide num envs."
    self.num_resets = self.num_envs // reset_ratio

    self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
    self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

  @partial(jax.jit, static_argnums=(0, 2))
  def reset(self, rng, params=None):
    rng, _rng = jax.random.split(rng)
    rngs = jax.random.split(_rng, self.num_envs)
    timestep = self.reset_fn(rngs, params)
    return timestep

  @partial(jax.jit, static_argnums=(0, 4))
  def step(self, rng, prior_timestep, action, params=None):
    rng, _rng = jax.random.split(rng)
    rngs = jax.random.split(_rng, self.num_envs)
    timestep_st = self.step_fn(rngs, prior_timestep, action, params)

    rng, _rng = jax.random.split(rng)
    rngs = jax.random.split(_rng, self.num_resets)
    timestep_re = self.reset_fn(rngs, params)

    rng, _rng = jax.random.split(rng)
    reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)

    # NOTE: CHANGE: if PRIOR timestep is last, then we are resetting
    # before, if CURRENT timestep is last, then we are resetting
    done = prior_timestep.last()
    being_reset = jax.random.choice(
      _rng,
      jnp.arange(self.num_envs),
      shape=(self.num_resets,),
      p=done,
      replace=False,
    )
    reset_indexes = reset_indexes.at[being_reset].set(jnp.arange(self.num_resets))

    timestep_re = jax.tree.map(lambda x: x[reset_indexes], timestep_re)

    # Auto-reset environment based on termination
    def auto_reset(done, timestep_re, timestep_st):
      return jax.tree.map(
        lambda x, y: jax.lax.select(done, x, y), timestep_re, timestep_st
      )

    timestep = jax.vmap(auto_reset)(done, timestep_re, timestep_st)

    return timestep


def craftax_experience_logger(
  train_state,
  observer_state,
  key: str = "train",
  num_seeds: int = None,
  trajectory: Optional[struct.PyTreeNode] = None,
  **kwargs,
):
  ############################################################
  # LogWrapper logging
  ############################################################
  key_fn = lambda k: f"{k}-{num_seeds}" if num_seeds is not None else k
  main_key = key_fn(key)
  ach_key = key_fn(f"{key}-achievements")

  log_state = trajectory.timestep.state
  # [T, B, A]
  achievements = trajectory.timestep.state.env_state.achievements

  # [T, B]
  done = trajectory.timestep.last()

  achievements = achievements * done[:, :, None] * 100.0

  infos = {}
  infos[f"{main_key}/0.episode_return"] = log_state.returned_episode_returns
  infos[f"{main_key}/0.score"] = (
    log_state.returned_episode_returns / MAX_SCORE
  ) * 100.0
  infos[f"{main_key}/0.episode_length"] = log_state.returned_episode_lengths
  for achievement in Achievement:
    name = f"Achievements/{achievement.name.lower()}"
    infos[f"{ach_key}/{name}"] = achievements[:, :, achievement.value]

  metrics = jax.tree.map(
    lambda x: (x * done).sum() / (1e-5 + done.sum()),
    infos,
  )
  metrics[f"{main_key}/num_actor_steps"] = train_state.timesteps
  metrics[f"{main_key}/num_learner_updates"] = train_state.n_updates

  ############################################################
  # Observer logging
  ############################################################

  def callback(m, idx, lengths, returns):
    # NOTE: WRONG. observation logic is wrong
    # length_means = []
    # return_means = []
    # for env in range(lengths.shape[1]):
    #  end = min(idx[env] + 1, len(lengths[:, env]))
    #  length_means.append(lengths[:, env][:end].mean())
    #  return_means.append(returns[:, env][:end].mean())

    # key = key_fn('observer')

    # m.update({
    #    f'{key}/0.episode_length': np.mean(length_means),
    #    f'{key}/0.episode_return': np.mean(return_means),
    #    f'{key}/0.score': (np.mean(return_means)/MAX_SCORE)*100.0,
    # })
    if wandb.run is not None:
      wandb.log(m)

  jax.debug.callback(
    callback,
    metrics,
    observer_state.idx,
    observer_state.episode_lengths,
    observer_state.episode_returns,
  )


def make_logger(
  config: dict,
  env,
  env_params,
  learner_log_extra: Optional[Callable[[Any], Any]] = None,
):
  return loggers.Logger(
    gradient_logger=loggers.default_gradient_logger,
    learner_logger=loggers.default_learner_logger,
    experience_logger=partial(
      craftax_experience_logger, num_seeds=config.get("NUM_ENV_SEEDS", None)
    ),
    learner_log_extra=(
      partial(learner_log_extra, config=config)
      if learner_log_extra is not None
      else None
    ),
  )


def run_single(config: dict, save_path: str = None):
  rng = jax.random.PRNGKey(config["SEED"])
  config["TEST_NUM_ENVS"] = config.get("TEST_NUM_ENVS", None) or config["NUM_ENVS"]

  if config["ENV"] == "classic":
    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
    env_params = env.default_params
    test_env_params = env.default_params

  elif config["ENV"] == "craftax":
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    env_params = env.default_params
    test_env_params = env.default_params

  elif config["ENV"] == "craftax-gen":
    from craftax_env import CraftaxSymbolicEnvNoAutoReset

    config["STRUCTURED_INPUTS"] = True
    config["NUM_ENV_SEEDS"] = config.get("NUM_ENV_SEEDS", 10_000)

    static_env_params = CraftaxSymbolicEnvNoAutoReset.default_static_params().replace(
      use_precondition=config.get("USE_PRECONDITION", False)
    )

    env = CraftaxSymbolicEnvNoAutoReset(static_env_params=static_env_params)

    if config["NUM_ENV_SEEDS"]:
      env_params = env.default_params.replace(
        world_seeds=tuple(np.arange(config["NUM_ENV_SEEDS"]))
      )
      test_env_params = env.default_params.replace(
        world_seeds=tuple(
          np.arange(
            10_000,
            10_000 + config["TEST_NUM_ENVS"],
          )
        )
      )
    else:
      env_params = env.default_params
      test_env_params = env.default_params

  else:
    raise NotImplementedError(config["ENV"])

  if config["OPTIMISTIC_RESET_RATIO"] == 1:
    vec_env = env = TimestepWrapper(LogWrapper(env), autoreset=True)
    vmap_env = True
  else:
    env = TimestepWrapper(LogWrapper(env), autoreset=False)
    vec_env = OptimisticResetVecEnvWrapper(
      env,
      num_envs=config["NUM_ENVS"],
      reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
    )
    vmap_env = False

  if config["ALG"] == "qlearning":
    train_fn = vbb.make_train(
      config=config,
      env=vec_env,
      make_agent=qlearning_craftax.make_craftax_agent,
      make_optimizer=qlearning_craftax.make_optimizer,
      make_loss_fn_class=qlearning_craftax.make_loss_fn_class,
      make_actor=qlearning_craftax.make_actor,
      make_logger=partial(
        make_logger, learner_log_extra=qlearning_craftax.learner_log_extra
      ),
      train_env_params=env_params,
      test_env_params=test_env_params,
      ObserverCls=craftax_observer.Observer,
      vmap_env=vmap_env,
    )
  elif config["ALG"] == "qlearning_sf_aux":
    train_fn = vbb.make_train(
      config=config,
      env=vec_env,
      make_agent=qlearning_sf_aux_craftax.make_craftax_agent,
      make_optimizer=qlearning_sf_aux_craftax.make_optimizer,
      make_loss_fn_class=qlearning_sf_aux_craftax.make_loss_fn_class,
      make_actor=qlearning_sf_aux_craftax.make_actor,
      make_logger=partial(
        make_logger, learner_log_extra=qlearning_craftax.learner_log_extra
      ),
      train_env_params=env_params,
      test_env_params=test_env_params,
      ObserverCls=craftax_observer.Observer,
      vmap_env=vmap_env,
    )
  elif config["ALG"] in ["dyna"]:
    train_fn = dyna_craftax.make_train(
      config=config,
      env=vec_env,
      model_env=env,
      make_logger=partial(
        make_logger, learner_log_extra=dyna_craftax.learner_log_extra
      ),
      train_env_params=env_params,
      test_env_params=test_env_params,
      ObserverCls=craftax_observer.Observer,
      vmap_env=vmap_env,
    )
  elif config["ALG"] == "alphazero":
    train_fn = alphazero_craftax.make_train(
      config=config,
      env=vec_env,
      model_env=env,
      make_logger=partial(
        make_logger, learner_log_extra=alphazero_craftax.learner_log_extra
      ),
      train_env_params=env_params,
      test_env_params=test_env_params,
      ObserverCls=craftax_observer.Observer,
      vmap_env=vmap_env,
    )
  elif config["ALG"] in ["preplay"]:
    train_fn = multitask_preplay_craftax_v2.make_train_craftax_singlegoal(
      config=config,
      env=vec_env,
      model_env=env,
      make_logger=partial(
        make_logger,
        learner_log_extra=multitask_preplay_craftax_v2.learner_log_extra,
      ),
      train_env_params=env_params,
      test_env_params=test_env_params,
      ObserverCls=craftax_observer.Observer,
      vmap_env=vmap_env,
    )
  elif config["ALG"] == "pqn":
    constructor = get_pqn_fns(config)
    train_fn = vpq.make_train(
      config=config,
      env=vec_env,
      make_agent=constructor.make_agent,
      make_optimizer=constructor.make_optimizer,
      make_loss_fn_class=constructor.make_loss_fn_class,
      make_actor=constructor.make_actor,
      make_logger=make_logger,
      train_env_params=env_params,
      test_env_params=test_env_params,
      ObserverCls=craftax_observer.Observer,
      vmap_env=vmap_env,
    )
  elif config["ALG"] == "usfa":
    train_fn = vbb.make_train(
      config=config,
      env=vec_env,
      make_agent=usfa.make_craftax_agent,
      make_optimizer=usfa.make_optimizer,
      make_loss_fn_class=usfa.make_loss_fn_class,
      make_actor=usfa.make_actor,
      make_logger=partial(make_logger, learner_log_extra=usfa.learner_log_extra),
      train_env_params=env_params,
      test_env_params=test_env_params,
      ObserverCls=craftax_observer.Observer,
      vmap_env=vmap_env,
    )
  else:
    raise NotImplementedError(config["ALG"])

  start_time = time.time()
  train_vjit = jax.jit(jax.vmap(train_fn))

  rngs = jax.random.split(rng, config["NUM_SEEDS"])
  outs = jax.block_until_ready(train_vjit(rngs))
  elapsed_time = time.time() - start_time
  print("Elapsed time: {:.2f} seconds".format(elapsed_time))

  # ---------------
  # save model weights
  # ---------------
  alg_name = config["ALG"]
  if save_path is not None:

    def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
      flattened_dict = flatten_dict(params, sep=",")
      save_file(flattened_dict, filename)

    model_state = outs["runner_state"][0]
    # save only params of the firt run
    params = jax.tree.map(lambda x: x[0], model_state.params)
    os.makedirs(save_path, exist_ok=True)

    save_params(params, f"{save_path}/{alg_name}.safetensors")
    print(f"Parameters of first batch saved in {save_path}/{alg_name}.safetensors")

    config_filename = f"{save_path}/{alg_name}.config"
    import pickle

    # Save the dictionary as a pickle file
    with open(config_filename, "wb") as f:
      pickle.dump(config, f)
    print(f"Config saved in {config_filename}")


def sweep(search: str = ""):
  search = search or "ql"
  ############################################################
  # Testing
  ############################################################
  metric = {
    "name": "evaluator_performance-0/0.score",
    "goal": "maximize",
  }
  if search == "ql":
    sweep_config = {
      "metric": metric,
      "parameters": {
        # "ENV": {'values': ['craftax']},
        "NUM_ENV_SEEDS": {"values": [0]},
        "SEED": {"values": list(range(1, 2))},
      },
      "overrides": ["alg=ql", "rlenv=craftax-10m", "user=wilka"],
      "group": "ql-24-achievable",
    }
  elif search == "ql_sf":
    sweep_config = {
      "metric": metric,
      "parameters": {
        "ALG": {"values": ["qlearning_sf_aux"]},
        # "ENV": {'values': ['craftax']},
        "SEED": {"values": list(range(1, 2))},
        "NUM_ENV_SEEDS": {"values": [0]},
        "NUM_ENVS": {"values": [128]},
        "AUX_COEFF": {"values": [1e-6, 1e-7]},
        "USE_PRECONDITION": {"values": [True]},
      },
      "overrides": ["alg=ql", "rlenv=craftax-10m", "user=wilka"],
      "group": "ql-sf-16-achievable",
    }
  elif search == "usfa":
    sweep_config = {
      "metric": metric,
      "parameters": {
        "NUM_ENV_SEEDS": {"values": [0]},
        "NUM_ENVS": {"values": [128]},
        "USE_PRECONDITION": {"values": [True]},
        "AUX_COEFF": {"values": [1e-5, 1e-6]},
        "MAX_PRIORITY_WEIGHT": {"values": [0.0, 0.9]},
        "IMPORTANCE_SAMPLING_EXPONENT": {"values": [0.0, 0.6]},
      },
      "overrides": ["alg=usfa_craftax", "rlenv=craftax-10m", "user=wilka"],
      "group": "usfa-10-achievable",
    }

  elif search == "alphazero":
    sweep_config = {
      "metric": metric,
      "parameters": {
        "NUM_ENV_SEEDS": {"values": [0]},
        "NUM_SIMULATIONS": {"values": [2]},
        "GUMBEL_SCALE": {"values": [1.0, 0.1, 10.0]},
        "value_coef": {"values": [0.25, 1.0, 0.025]},
        "MAX_PRIORITY_WEIGHT": {"values": [0.0]},
        "IMPORTANCE_SAMPLING_EXPONENT": {"values": [0.0]},
        # "MAX_VALUE": {'values': [10, 20]},
        # "SAMPLE_LENGTH": {'values': [40, 60, 80]},
      },
      "overrides": ["alg=alphazero", "rlenv=craftax-1m-dyna", "user=wilka"],
      "group": "alphazero-4",
    }
  elif search == "dyna":
    sweep_config = {
      "metric": metric,
      "parameters": {
        "ALG": {"values": ["dyna"]},
        "SEED": {"values": list(range(1, 4))},
        "NUM_ENV_SEEDS": {"values": [128]},
        "NUM_SIMULATIONS": {"values": [4]},
        "FIXED_EPSILON": {"values": [2]},
      },
      "overrides": ["alg=dyna", "rlenv=craftax-1m-dyna", "user=wilka"],
      "group": "dyna-19-epsilon",
    }
  elif search == "preplay":
    sweep_config = {
      "metric": metric,
      "parameters": {
        "ALG": {"values": ["preplay"]},
        "SEED": {"values": list(range(1))},
        "NUM_ENV_SEEDS": {"values": [128, 512]},
        "COMBINE_REAL_SIM": {"values": [True, False]},
      },
      "overrides": ["alg=preplay", "rlenv=craftax-1m-dyna", "user=wilka"],
      "group": "preplay-combine-1",
    }
  elif search == "dyna":
    sweep_config = {
      "metric": metric,
      "parameters": {
        "ALG": {"values": ["dyna"]},
        "SEED": {"values": list(range(1))},
        "NUM_ENV_SEEDS": {"values": [128, 512]},
        "COMBINE_REAL_SIM": {"values": [True, False]},
      },
      "overrides": ["alg=dyna", "rlenv=craftax-1m-dyna", "user=wilka"],
      "group": "dyna-combine-1",
    }
  elif search == "pqn":
    sweep_config = {
      "metric": metric,
      "parameters": {
        "FIXED_EPSILON": {"values": [0, 2]},
        "LAMBDA": {"values": [0.95, 0.5]},
        "MAX_GRAD_NORM": {"values": [0.5, 10]},
        "LR_LINEAR_DECAY": {"values": [True, False]},
        "LR": {"values": [0.001, 0.0003]},
        "NUM_ENVS": {"values": [256]},
      },
      "overrides": ["alg=pqn-craftax", "rlenv=craftax-10m", "user=wilka"],
      "group": "pqn-7",
    }
  ############################################################
  # More "final" experiments
  ############################################################
  elif search == "ql-final":
    sweep_config = {
      "metric": metric,
      "parameters": {
        "NUM_ENV_SEEDS": {"values": [8, 16, 32, 64, 128, 256, 512]},
        "SEED": {"values": list(range(1, 6))},
      },
      "overrides": ["alg=ql", "rlenv=craftax-10m", "user=wilka"],
      "group": "ql-final-5",
    }
  elif search == "ql_sf-final":
    sweep_config = {
      "metric": metric,
      "parameters": {
        "ALG": {"values": ["qlearning_sf_aux"]},
        "NUM_ENV_SEEDS": {"values": [8, 16, 32, 64, 128, 256, 512]},
        "SEED": {"values": list(range(1, 6))},
      },
      "overrides": ["alg=ql-sf", "rlenv=craftax-10m", "user=wilka"],
      "group": "ql-sf-final-5",
    }
  elif search == "dyna-final":
    sweep_config = {
      "metric": metric,
      "parameters": {
        "ALG": {"values": ["dyna"]},
        "NUM_ENV_SEEDS": {"values": [8, 16, 32, 64, 128, 256, 512]},
        "SEED": {"values": list(range(1, 6))},
      },
      "overrides": ["alg=dyna", "rlenv=craftax-1m-dyna", "user=wilka"],
      "group": "dyna-final-5",
    }
  elif search == "preplay-final":
    sweep_config = {
      "metric": metric,
      "parameters": {
        "ALG": {"values": ["preplay"]},
        "NUM_ENV_SEEDS": {"values": [8, 16, 32, 64, 128, 256, 512]},
        "SEED": {"values": list(range(1, 6))},
      },
      "overrides": ["alg=preplay", "rlenv=craftax-1m-dyna", "user=wilka"],
      "group": "preplay-final-5",
    }
  ############################################################
  # Ablations
  ############################################################
  # elif search == "dyna-backtracking":
  #  sweep_config = {
  #    "metric": metric,
  #    "parameters": {
  #      "ALG": {"values": ["dyna"]},
  #      "NUM_ENV_SEEDS": {"values": [512]},
  #      "SEED": {"values": list(range(1, 5))},
  #      "BACKTRACKING": {"values": [False]},
  #    },
  #    "overrides": ["alg=dyna", "rlenv=craftax-1m-dyna", "user=wilka"],
  #    "group": "dyna-backtracking-1",
  #  }
  # elif search == "preplay-backtracking":
  #  sweep_config = {
  #    "metric": metric,
  #    "parameters": {
  #      "ALG": {"values": ["preplay"]},
  #      "NUM_ENV_SEEDS": {"values": [512]},
  #      "SEED": {"values": list(range(1, 5))},
  #      "BACKTRACKING": {"values": [False]},
  #    },
  #    "overrides": ["alg=preplay", "rlenv=craftax-1m-dyna", "user=wilka"],
  #    "group": "preplay-backtracking-1",
  #  }
  elif search == "preplay-main-loss-coeff":
    sweep_config = {
      "metric": metric,
      "parameters": {
        "ALG": {"values": ["preplay"]},
        "NUM_ENV_SEEDS": {"values": [512]},
        "SEED": {"values": list(range(1, 5))},
        "MAIN_COEFF": {"values": [0.0]},
        # "OFFTASK_COEFF": {"values": [1.]},
      },
      "overrides": ["alg=preplay", "rlenv=craftax-1m-dyna", "user=wilka"],
      "group": "preplay-main-loss-coeff-1",
    }
  elif search == "preplay-main-q-coeff":
    sweep_config = {
      "metric": metric,
      "parameters": {
        "ALG": {"values": ["preplay"]},
        "NUM_ENV_SEEDS": {"values": [512]},
        "SEED": {"values": list(range(1, 5))},
        "MAINQ_COEFF": {"values": [0.0]},
      },
      "overrides": ["alg=preplay", "rlenv=craftax-1m-dyna", "user=wilka"],
      "group": "preplay-main-q-coeff-1",
    }

  ############################################################
  # Preconditions
  ############################################################

  elif search == "ql-precondition":
    sweep_config = {
      "metric": metric,
      "parameters": {
        "NUM_ENV_SEEDS": {"values": [512]},
        "SEED": {"values": list(range(1, 5))},
        "USE_PRECONDITION": {"values": [False]},
      },
      "overrides": ["alg=ql", "rlenv=craftax-10m", "user=wilka"],
      "group": "ql-precondition-1",
    }
  elif search == "ql_sf-precondition":
    sweep_config = {
      "metric": metric,
      "parameters": {
        "ALG": {"values": ["qlearning_sf_aux"]},
        "NUM_ENV_SEEDS": {"values": [512]},
        "SEED": {"values": list(range(1, 5))},
        "USE_PRECONDITION": {"values": [False]},
      },
      "overrides": ["alg=ql-sf", "rlenv=craftax-10m", "user=wilka"],
      "group": "ql-sf-precondition-1",
    }
  elif search == "dyna-precondition":
    sweep_config = {
      "metric": metric,
      "parameters": {
        "ALG": {"values": ["dyna"]},
        "NUM_ENV_SEEDS": {"values": [512]},
        "SEED": {"values": list(range(1, 5))},
        "USE_PRECONDITION": {"values": [False]},
      },
      "overrides": ["alg=dyna", "rlenv=craftax-1m-dyna", "user=wilka"],
      "group": "dyna-precondition-1",
    }
  elif search == "preplay-precondition":
    sweep_config = {
      "metric": metric,
      "parameters": {
        "ALG": {"values": ["preplay"]},
        "NUM_ENV_SEEDS": {"values": [512]},
        "SEED": {"values": list(range(1, 5))},
        "USE_PRECONDITION": {"values": [False]},
      },
      "overrides": ["alg=preplay", "rlenv=craftax-1m-dyna", "user=wilka"],
      "group": "preplay-precondition-1",
    }
  else:
    raise NotImplementedError(search)

  return sweep_config


CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "configs"))


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config_craftax")
def main(config: DictConfig):
  launcher.run(
    config,
    trainer_filename=__file__,
    absolute_config_path=CONFIG_PATH,
    run_fn=run_single,
    sweep_fn=sweep,
    folder=os.environ.get("RL_RESULTS_DIR", "/tmp/rl_results_dir"),
  )


if __name__ == "__main__":
  main()
