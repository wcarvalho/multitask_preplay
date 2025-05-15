"""

TESTING:
JAX_DEBUG_NANS=True \
JAX_DISABLE_JIT=1 \
HYDRA_FULL_ERROR=1 JAX_TRACEBACK_FILTERING=off python -m ipdb -c continue housemaze_trainer.py \
  app.debug=True \
  app.wandb=False \
  app.search=preplay

RUNNING ON SLURM:
python housemaze_trainer.py \
  app.parallel=slurm_wandb \
  app.search=dynaq_shared
"""

from typing import Any, Callable, Dict, Union, Optional


import os
import jax
from flax import struct
import functools
import jax.numpy as jnp
import time


import hydra
from omegaconf import DictConfig


from safetensors.flax import save_file
from flax.traverse_util import flatten_dict

import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


from jaxneurorl.agents import value_based_basics as vbb
from jaxneurorl.agents import value_based_pqn as vpq
from jaxneurorl import launcher
from jaxneurorl import utils
from jaxneurorl import loggers

import alphazero
import qlearning_housemaze
import usfa_housemaze
import multitask_preplay_housemaze
import multitask_preplay_craftax_v2
import networks
import housemaze_observer as humansf_observers
import housemaze_experiments

from housemaze import renderer
from housemaze import utils as housemaze_utils
from housemaze.human_dyna import multitask_env


def make_logger(
  config: dict,
  env: multitask_env.HouseMaze,
  env_params: multitask_env.EnvParams,
  action_names: dict,
  render_fn: Callable = None,
  extract_task_info: Callable = None,
  get_task_name: Callable = None,
  learner_log_extra: Optional[Callable[[Any], Any]] = None,
):
  return loggers.Logger(
    gradient_logger=loggers.default_gradient_logger,
    learner_logger=loggers.default_learner_logger,
    experience_logger=functools.partial(
      humansf_observers.experience_logger,
      action_names=action_names,
      extract_task_info=extract_task_info,
      get_task_name=get_task_name,
      render_fn=render_fn,
      max_len=config["MAX_EPISODE_LOG_LEN"],
    ),
    learner_log_extra=learner_log_extra,
  )


@struct.dataclass
class AlgorithmConstructor:
  make_agent: Callable
  make_optimizer: Callable
  make_loss_fn_class: Callable
  make_actor: Callable


def get_qlearning_fns(
  config,
  num_categories=10_000,
):
  HouzemazeObsEncoder = functools.partial(
    networks.CategoricalHouzemazeObsEncoder,
    num_categories=num_categories,
    embed_hidden_dim=config["EMBED_HIDDEN_DIM"],
    mlp_hidden_dim=config["MLP_HIDDEN_DIM"],
    num_embed_layers=config["NUM_EMBED_LAYERS"],
    num_mlp_layers=config["NUM_MLP_LAYERS"],
    activation=config["ACTIVATION"],
    norm_type=config.get("NORM_TYPE", "none"),
  )

  return AlgorithmConstructor(
    make_agent=qlearning_housemaze.make_housemaze_agent,
    make_optimizer=qlearning_housemaze.make_optimizer,
    make_loss_fn_class=qlearning_housemaze.make_loss_fn_class,
    make_actor=qlearning_housemaze.make_actor,
  )


def get_sf_fns(
  config,
  env,
  env_params,
  num_categories=10_000,
):
  HouzemazeObsEncoder = functools.partial(
    networks.CategoricalHouzemazeObsEncoder,
    num_categories=num_categories,
    embed_hidden_dim=config["EMBED_HIDDEN_DIM"],
    mlp_hidden_dim=config["MLP_HIDDEN_DIM"],
    num_embed_layers=config["NUM_EMBED_LAYERS"],
    num_mlp_layers=config["NUM_MLP_LAYERS"],
    activation=config["ACTIVATION"],
    norm_type=config.get("NORM_TYPE", "none"),
  )
  train_objects = env_params.reset_params.train_objects[0]
  train_tasks = jnp.array([env.task_runner.task_vector(o) for o in train_objects])
  return AlgorithmConstructor(
    make_agent=functools.partial(
      usfa_housemaze.make_agent,
      train_tasks=train_tasks,
      ObsEncoderCls=HouzemazeObsEncoder,
    ),
    make_optimizer=usfa_housemaze.make_optimizer,
    make_loss_fn_class=usfa_housemaze.make_loss_fn_class,
    make_actor=functools.partial(usfa_housemaze.make_actor, remove_gpi_dim=False),
  )


def get_dynaq_fns(
  config, env, env_params, task_objects, num_categories=10_000, rng=None
):
  if rng is None:
    rng = jax.random.PRNGKey(42)
  import distrax

  HouzemazeObsEncoder = functools.partial(
    networks.CategoricalHouzemazeObsEncoder,
    num_categories=num_categories,
    embed_hidden_dim=config["EMBED_HIDDEN_DIM"],
    mlp_hidden_dim=config["MLP_HIDDEN_DIM"],
    num_embed_layers=config["NUM_EMBED_LAYERS"],
    num_mlp_layers=config["NUM_MLP_LAYERS"],
    activation=config["ACTIVATION"],
    norm_type=config.get("NORM_TYPE", "none"),
  )

  sim_policy = config["SIM_POLICY"]
  num_simulations = config["NUM_SIMULATIONS"]
  if sim_policy == "gamma":
    temp_dist = distrax.Gamma(
      concentration=config["TEMP_CONCENTRATION"], rate=config["TEMP_RATE"]
    )

    rng, rng_ = jax.random.split(rng)
    temperatures = temp_dist.sample(seed=rng_, sample_shape=(num_simulations - 1,))
    temperatures = jnp.concatenate((temperatures, jnp.array((1e-5,))))
    greedy_idx = int(temperatures.argmin())

    def simulation_policy(preds: struct.PyTreeNode, sim_rng: jax.Array):
      q_values = preds.q_vals
      assert q_values.shape[0] == temperatures.shape[0]
      logits = q_values / jnp.expand_dims(temperatures, -1)
      return distrax.Categorical(logits=logits).sample(seed=sim_rng)

  elif sim_policy == "epsilon":
    epsilon_setting = config["SIM_EPSILON_SETTING"]
    if epsilon_setting == 1:
      vals = np.logspace(num=256, start=1, stop=3, base=0.1)
    elif epsilon_setting == 2:
      vals = np.logspace(num=256, start=0.05, stop=0.9, base=0.1)
    epsilons = jax.random.choice(rng, vals, shape=(num_simulations - 1,))
    epsilons = jnp.concatenate((jnp.zeros(1), epsilons))
    greedy_idx = int(epsilons.argmin())

    def simulation_policy(preds: struct.PyTreeNode, sim_rng: jax.Array):
      q_values = preds.q_vals
      assert q_values.shape[0] == epsilons.shape[0]
      sim_rng = jax.random.split(sim_rng, q_values.shape[0])
      return jax.vmap(qlearning_housemaze.epsilon_greedy_act, in_axes=(0, 0, 0))(
        q_values, epsilons, sim_rng
      )

  else:
    raise NotImplementedError

  def make_init_offtask_timestep(x: multitask_env.TimeStep, offtask_w: jax.Array):
    task_object = (task_objects * offtask_w).sum(-1)
    task_object = task_object.astype(jnp.int32)
    new_state = x.state.replace(
      step_num=jnp.zeros_like(x.state.step_num),
      task_w=offtask_w,
      task_object=task_object,  # only used for logging
      is_train_task=jnp.full(x.reward.shape, False),
    )
    return x.replace(
      state=new_state,
      observation=jax.vmap(jax.vmap(env.make_observation))(
        new_state, x.observation.prev_action
      ),
      # reset reward, discount, step type
      reward=jnp.zeros_like(x.reward),
      discount=jnp.ones_like(x.discount),
      step_type=jnp.ones_like(x.step_type),
    )

  return AlgorithmConstructor(
    make_agent=functools.partial(
      multitask_preplay_housemaze.make_agent, ObsEncoderCls=HouzemazeObsEncoder
    ),
    make_optimizer=multitask_preplay_housemaze.make_optimizer,
    make_loss_fn_class=functools.partial(
      multitask_preplay_housemaze.make_loss_fn_class,
      make_init_offtask_timestep=make_init_offtask_timestep,
      simulation_policy=simulation_policy,
      online_coeff=config["ONLINE_COEFF"],
      dyna_coeff=config.get("DYNA_COEFF", 1.0),
    ),
    make_actor=multitask_preplay_housemaze.make_actor,
  )


def extract_task_info(timestep: multitask_env.TimeStep):
  state = timestep.state
  return {
    "map_idx": state.map_idx,
    "current_label": state.current_label,
    "is_train_task": state.is_train_task,
    "category": state.task_object,
  }


def task_from_variables(variables, keys, label2name):
  current_label = variables["current_label"]
  category = keys[variables["category"]]
  is_train_task = variables["is_train_task"]
  label = "1.train" if is_train_task else "0.TEST"
  setting = label2name.get(int(current_label))
  return f"{label} \n {setting} - {category}"


def run_single(config: dict, save_path: str = None):
  rng = jax.random.PRNGKey(config["SEED"])
  # config['save_path'] = save_path
  ###################
  # load data
  ###################
  exp = config["rlenv"]["ENV_KWARGS"].pop("exp")
  try:
    exp_fn = getattr(housemaze_experiments, exp, None)
  except Exception as e:
    raise RuntimeError(e)
  env_params, test_env_params, task_objects, idx2maze = exp_fn(config)

  image_dict = housemaze_utils.load_image_dict()
  # Reshape the images to separate the blocks
  images = image_dict["images"]
  reshaped = images.reshape(len(images), 8, 4, 8, 4, 3)

  # Take the mean across the block dimensions
  image_dict["images"] = reshaped.mean(axis=(2, 4)).astype(np.uint8)

  ###################
  # load env
  ###################
  if config["ALG"] == "usfa":
    from housemaze.human_dyna import sf_task_runner

    task_runner = sf_task_runner.TaskRunner(
      task_objects=task_objects,
      radius=config.get("VIS_RADIUS", 5),
      vis_coeff=config.get("VIS_COEFF", 0.0),
    )
    # def success_fn(timestep):
    #  features = timestep.observation.state_features
    #  task_w = timestep.observation.task_w
    #  # only first half count. 2nd half are about visibility.
    #  half = len(features)//2
    #  import ipdb; ipdb.set_trace()
    #  return (features[:half]*task_w[:half]).sum(-1)
  else:
    task_runner = multitask_env.TaskRunner(task_objects=task_objects)
    # success_fn = lambda timestep: timestep.rewards > .5
  keys = image_dict["keys"]
  env = multitask_env.HouseMaze(
    task_runner=task_runner,
    num_categories=200,
  )

  env = housemaze_utils.AutoResetWrapper(env)
  HouzemazeObsEncoder = functools.partial(
    networks.CategoricalHouzemazeObsEncoder,
    num_categories=max(10_000, env.total_categories(env_params)),
    embed_hidden_dim=config["EMBED_HIDDEN_DIM"],
    mlp_hidden_dim=config["MLP_HIDDEN_DIM"],
    num_embed_layers=config["NUM_EMBED_LAYERS"],
    num_mlp_layers=config["NUM_MLP_LAYERS"],
    activation=config["ACTIVATION"],
    norm_type=config.get("NORM_TYPE", "none"),
  )
  ###################
  ## custom observer
  ###################
  action_names = {action.value: action.name for action in env.action_enum()}

  def housemaze_render_fn(state: multitask_env.EnvState):
    return renderer.create_image_from_grid(
      state.grid, state.agent_pos, state.agent_dir, image_dict
    )

  observer_class = functools.partial(
    humansf_observers.TaskObserver,
    extract_task_info=extract_task_info,
    action_names=action_names,
    # success_fn=success_fn,
  )

  get_task_name = functools.partial(task_from_variables, keys=keys, label2name=idx2maze)
  ##################
  # algorithms
  ##################
  alg_name = config["ALG"]

  train_objects = env_params.reset_params.train_objects[0]
  test_objects = env_params.reset_params.test_objects[0]
  train_tasks = jnp.array([env.task_runner.task_vector(o) for o in train_objects])
  test_tasks = jnp.array([env.task_runner.task_vector(o) for o in test_objects])
  all_tasks = jnp.concatenate((train_tasks, test_tasks), axis=0)

  if alg_name == "qlearning":
    make_train = functools.partial(
      vbb.make_train,
      make_agent=functools.partial(
        qlearning_housemaze.make_housemaze_agent,
        ObsEncoderCls=HouzemazeObsEncoder,
      ),
      make_optimizer=qlearning_housemaze.make_optimizer,
      make_loss_fn_class=qlearning_housemaze.make_loss_fn_class,
      make_actor=qlearning_housemaze.make_actor,
      make_logger=functools.partial(
        make_logger,
        render_fn=housemaze_render_fn,
        extract_task_info=extract_task_info,
        get_task_name=get_task_name,
        action_names=action_names,
        learner_log_extra=functools.partial(
          qlearning_housemaze.learner_log_extra,
          config=config,
          action_names=action_names,
          extract_task_info=extract_task_info,
          get_task_name=get_task_name,
          render_fn=housemaze_render_fn,
        ),
      ),
    )
  elif alg_name == "usfa":
    make_train = functools.partial(
      usfa_housemaze.make_train,
      make_agent=functools.partial(
        usfa_housemaze.make_agent,
        train_tasks=train_tasks,
        all_tasks=all_tasks,
        ObsEncoderCls=HouzemazeObsEncoder,
      ),
      make_logger=functools.partial(
        make_logger,
        render_fn=housemaze_render_fn,
        extract_task_info=extract_task_info,
        get_task_name=get_task_name,
        action_names=action_names,
        learner_log_extra=functools.partial(
          usfa_housemaze.learner_log_extra,
          config=config,
          action_names=action_names,
          extract_task_info=extract_task_info,
          get_task_name=get_task_name,
          render_fn=housemaze_render_fn,
        ),
      ),
    )
  elif alg_name == "pqn":
    make_train = functools.partial(
      vpq.make_train,
      make_agent=functools.partial(
        vpq.make_agent,
        ObsEncoderCls=HouzemazeObsEncoder,
      ),
      make_logger=functools.partial(
        make_logger,
        render_fn=housemaze_render_fn,
        extract_task_info=extract_task_info,
        get_task_name=get_task_name,
        action_names=action_names,
        learner_log_extra=functools.partial(
          qlearning_housemaze.learner_log_extra,
          config=config,
          action_names=action_names,
          extract_task_info=extract_task_info,
          get_task_name=get_task_name,
          render_fn=housemaze_render_fn,
        ),
      ),
    )
  elif alg_name == "alphazero":
    import mctx

    max_value = config.get("MAX_VALUE", 10)
    num_bins = config["NUM_BINS"]

    discretizer = utils.Discretizer(
      max_value=max_value, num_bins=num_bins, min_value=-max_value
    )

    num_train_simulations = config.get("NUM_SIMULATIONS", 4)
    mcts_policy = functools.partial(
      mctx.gumbel_muzero_policy,
      max_depth=config.get("MAX_SIM_DEPTH", None),
      num_simulations=num_train_simulations,
      gumbel_scale=config.get("GUMBEL_SCALE", 1.0),
    )
    eval_mcts_policy = functools.partial(
      mctx.gumbel_muzero_policy,
      max_depth=config.get("MAX_SIM_DEPTH", None),
      num_simulations=config.get("NUM_EVAL_SIMULATIONS", num_train_simulations),
      gumbel_scale=config.get("GUMBEL_SCALE", 1.0),
    )

    make_train = functools.partial(
      vbb.make_train,
      make_agent=functools.partial(
        alphazero.make_agent,
        ObsEncoderCls=HouzemazeObsEncoder,
        test_env_params=test_env_params,
      ),
      make_optimizer=alphazero.make_optimizer,
      make_loss_fn_class=functools.partial(
        alphazero.make_loss_fn_class, discretizer=discretizer
      ),
      make_actor=functools.partial(
        alphazero.make_actor,
        discretizer=discretizer,
        mcts_policy=mcts_policy,
        eval_mcts_policy=eval_mcts_policy,
      ),
      make_logger=functools.partial(
        make_logger,
        render_fn=housemaze_render_fn,
        extract_task_info=extract_task_info,
        get_task_name=get_task_name,
        action_names=action_names,
      ),
    )

  elif alg_name in ("dynaq", "dynaq_shared"):
    import distrax

    # sim_policy = config["SIM_POLICY"]
    num_simulations = config["NUM_SIMULATIONS"]
    # if sim_policy == "gamma":
    #  temp_dist = distrax.Gamma(
    #    concentration=config["TEMP_CONCENTRATION"], rate=config["TEMP_RATE"]
    #  )

    #  rng, rng_ = jax.random.split(rng)
    #  temperatures = temp_dist.sample(seed=rng_, sample_shape=(num_simulations - 1,))
    #  temperatures = jnp.concatenate((temperatures, jnp.array((1e-5,))))
    #  greedy_idx = int(temperatures.argmin())

    #  def simulation_policy(preds: struct.PyTreeNode, sim_rng: jax.Array):
    #    q_values = preds.q_vals
    #    assert q_values.shape[0] == temperatures.shape[0]
    #    logits = q_values / jnp.expand_dims(temperatures, -1)
    #    return distrax.Categorical(logits=logits).sample(seed=sim_rng)

    # elif sim_policy == "epsilon":
    epsilon_setting = config["SIM_EPSILON_SETTING"]
    if epsilon_setting == 1:
      vals = np.logspace(num=256, start=1, stop=3, base=0.1)
    elif epsilon_setting == 2:
      vals = np.logspace(num=256, start=0.05, stop=0.9, base=0.1)
    elif epsilon_setting == 3:
      vals = np.ones(256) * 0.9
    epsilons = jax.random.choice(rng, vals, shape=(num_simulations - 1,))
    epsilons = jnp.concatenate((jnp.zeros(1), epsilons))
    greedy_idx = int(epsilons.argmin())

    def simulation_policy(preds: struct.PyTreeNode, sim_rng: jax.Array):
      q_values = preds.q_vals
      assert q_values.shape[0] == epsilons.shape[0]
      sim_rng = jax.random.split(sim_rng, q_values.shape[0])
      return jax.vmap(qlearning_housemaze.epsilon_greedy_act, in_axes=(0, 0, 0))(
        q_values, epsilons, sim_rng
      )

    # else:
    #  raise NotImplementedError

    def make_init_offtask_timestep(x: multitask_env.TimeStep, offtask_w: jax.Array):
      task_object = (task_objects * offtask_w).sum(-1)
      task_object = task_object.astype(jnp.int32)
      new_state = x.state.replace(
        step_num=jnp.zeros_like(x.state.step_num),
        task_w=offtask_w,
        task_object=task_object,  # only used for logging
        is_train_task=jnp.full(x.reward.shape, False),
      )
      return x.replace(
        state=new_state,
        observation=jax.vmap(jax.vmap(env.make_observation))(
          new_state, x.observation.prev_action
        ),
        # reset reward, discount, step type
        reward=jnp.zeros_like(x.reward),
        discount=jnp.ones_like(x.discount),
        step_type=jnp.ones_like(x.step_type),
      )

    make_train = functools.partial(
      vbb.make_train,
      make_agent=functools.partial(
        multitask_preplay_housemaze.make_agent,
        ObsEncoderCls=HouzemazeObsEncoder,
        model_env_params=test_env_params.replace(
          p_test_sample_train=jnp.array(0.5),
        ),
      ),
      make_loss_fn_class=functools.partial(
        multitask_preplay_housemaze.make_loss_fn_class,
        make_init_offtask_timestep=make_init_offtask_timestep,
        simulation_policy=simulation_policy,
        online_coeff=config["ONLINE_COEFF"],
        dyna_coeff=config.get("DYNA_COEFF", 1.0),
      ),
      make_optimizer=qlearning_housemaze.make_optimizer,
      make_actor=multitask_preplay_housemaze.make_actor,
      make_logger=functools.partial(
        make_logger,
        render_fn=housemaze_render_fn,
        extract_task_info=extract_task_info,
        get_task_name=get_task_name,
        action_names=action_names,
        learner_log_extra=functools.partial(
          multitask_preplay_housemaze.learner_log_extra,
          config=config,
          action_names=action_names,
          extract_task_info=extract_task_info,
          get_task_name=get_task_name,
          render_fn=housemaze_render_fn,
          sim_idx=greedy_idx,
        ),
      ),
    )

  elif alg_name in ("dyna", "preplay"):
    if alg_name in ("dyna"):
      config["SUBTASK_COEFF"] = 0.0  # Q-fn
      config["OFFTASK_COEFF"] = 0.0  # loss
      config["NUM_OFFTASK_GOALS"] = 0
      config["Q_COEFF"] = 1.0

    make_train = functools.partial(
      multitask_preplay_craftax_v2.make_train_jaxmaze_multigoal,
      config=config,
      env=env,
      model_env=env,
      make_logger=functools.partial(
        make_logger,
        render_fn=housemaze_render_fn,
        extract_task_info=extract_task_info,
        get_task_name=get_task_name,
        action_names=action_names,
        learner_log_extra=functools.partial(
          multitask_preplay_housemaze.learner_log_extra,
          config=config,
          action_names=action_names,
          extract_task_info=extract_task_info,
          get_task_name=get_task_name,
          render_fn=housemaze_render_fn,
          sim_idx=0,
        ),
      ),
      train_env_params=env_params,
      test_env_params=test_env_params,
      ObsEncoderCls=HouzemazeObsEncoder,
      task_objects=task_objects,
      all_tasks=all_tasks,
    )

  else:
    raise NotImplementedError(alg_name)

  start_time = time.time()
  train_fn = make_train(
    config=config,
    env=env,
    train_env_params=env_params,
    test_env_params=test_env_params,
    ObserverCls=observer_class,
  )
  train_vjit = jax.jit(jax.vmap(train_fn))

  rngs = jax.random.split(rng, config["NUM_SEEDS"])
  outs = jax.block_until_ready(train_vjit(rngs))
  elapsed_time = time.time() - start_time
  print("Elapsed time: {:.2f} seconds".format(elapsed_time))

  # ---------------
  # save model weights
  # ---------------
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
  if search == "ql":
    sweep_config = {
      "metric": {
        "name": "evaluator_performance/0.0 avg_episode_return",
        "goal": "maximize",
      },
      "parameters": {
        "SEED": {"values": list(range(1, 2))},
        "env.exp": {"values": ["exp2"]},
        "GAMMA": {"values": [0.99, 0.992]},
        "STEP_COST": {"values": [0.0001]},
        "NUM_Q_LAYERS": {"values": [1, 2, 3]},
        "Q_HIDDEN_DIM": {"values": [512, 1024]},
      },
      "overrides": ["alg=ql", "rlenv=housemaze", "user=wilka"],
      "group": "ql-big-6",
    }
  elif search == "usfa":
    sweep_config = {
      "metric": {
        "name": "evaluator_performance/0.0 avg_episode_return",
        "goal": "maximize",
      },
      "parameters": {
        "SEED": {"values": list(range(1, 2))},
        "env.exp": {"values": ["exp2"]},
        # "STEP_COST": {"values": [0.01, 0.005, 0.001]},
        "LEARN_VECTORS": {"values": ["TRAIN", "ALL_TASKS"]},
        # "VIS_COEFF": {"values": [0.0, 0.1]},
      },
      "overrides": ["alg=usfa", "rlenv=housemaze", "user=wilka"],
      "group": "usfa-landmark-3",
    }
  elif search == "dyna":
    sweep_config = {
      "metric": {
        "name": "evaluator_performance/0.0 avg_episode_return",
        "goal": "maximize",
      },
      "parameters": {
        "ALG": {"values": ["dyna"]},
        "SEED": {"values": list(range(1))},
        "env.exp": {"values": ["exp2"]},
        "COMBINE_REAL_SIM": {"values": [True]},
        # "SIM_EPSILON_SETTING": {"values": [3]},
        # "OFFTASK_SIMULATION": {"values": [False]},
        # "TOTAL_TIMESTEPS": {"values": [100_000_000]},
      },
      "overrides": ["alg=preplay_jaxmaze", "rlenv=housemaze", "user=wilka"],
      "group": "dyna-new-4",
    }
  elif search == "preplay":
    sweep_config = {
      "metric": {
        "name": "evaluator_performance/0.0 avg_episode_return",
        "goal": "maximize",
      },
      "parameters": {
        "ALG": {"values": ["preplay"]},
        "SEED": {"values": list(range(2))},
        "env.exp": {"values": ["exp2"]},
        "Q_COEFF": {"values": [0.1, 0.0]},
      },
      "overrides": ["alg=preplay_jaxmaze", "rlenv=housemaze", "user=wilka"],
      "group": "preplay-new-8-window-size-know-goal",
    }
  # elif search == "dynaq_shared":
  #  sweep_config = {
  #    "metric": {
  #      "name": "evaluator_performance/0.0 avg_episode_return",
  #      "goal": "maximize",
  #    },
  #    "parameters": {
  #      "ALG": {"values": ["dynaq_shared"]},
  #      "SEED": {"values": list(range(1, 2))},
  #      "env.exp": {"values": ["exp2"]},
  #      "GAMMA": {"values": [0.99, 0.992]},
  #      "STEP_COST": {"values": [0.0001]},
  #      "NUM_Q_LAYERS": {"values": [1, 2, 3]},
  #      "Q_HIDDEN_DIM": {"values": [512, 1024]},
  #    },
  #    "overrides": ["alg=preplay", "rlenv=housemaze", "user=wilka"],
  #    "group": "dynaq-big-6",
  #  }
  # =================================================================
  # final
  # =================================================================
  elif search == "ql-final":
    sweep_config = {
      "metric": {
        "name": "evaluator_performance/0.0 avg_episode_return",
        "goal": "maximize",
      },
      "parameters": {
        "SEED": {"values": list(range(1, 11))},
        "env.exp": {"values": ["exp2"]},
      },
      "overrides": ["alg=ql", "rlenv=housemaze", "user=wilka"],
      "group": "ql-final-run-6",
    }
  elif search == "usfa-final":
    sweep_config = {
      "metric": {
        "name": "evaluator_performance/0.0 avg_episode_return",
        "goal": "maximize",
      },
      "parameters": {
        "SEED": {"values": list(range(1, 11))},
        "env.exp": {"values": ["exp2"]},
      },
      "overrides": ["alg=usfa", "rlenv=housemaze", "user=wilka"],
      "group": "usfa-final-run-6",
    }
  elif search == "dyna-final":
    sweep_config = {
      "metric": {
        "name": "evaluator_performance/0.0 avg_episode_return",
        "goal": "maximize",
      },
      "parameters": {
        "ALG": {"values": ["dyna"]},
        "SEED": {"values": list(range(1, 11))},
        "env.exp": {"values": ["exp2"]},
      },
      "overrides": ["alg=preplay_jaxmaze", "rlenv=housemaze", "user=wilka"],
      "group": "dyna-final-run-6",
    }
  elif search == "preplay-old-final":
    sweep_config = {
      "metric": {
        "name": "evaluator_performance/0.0 avg_episode_return",
        "goal": "maximize",
      },
      "parameters": {
        "ALG": {"values": ["dynaq_shared"]},
        "SEED": {"values": list(range(1, 11))},
        "env.exp": {"values": ["exp2"]},
      },
      "overrides": ["alg=preplay", "rlenv=housemaze", "user=wilka"],
      "group": "preplay-old-final-run-8",
    }
  elif search == "preplay-final":
    sweep_config = {
      "metric": {
        "name": "evaluator_performance/0.0 avg_episode_return",
        "goal": "maximize",
      },
      "parameters": {
        "ALG": {"values": ["preplay"]},
        "SEED": {"values": list(range(1, 11))},
        "env.exp": {"values": ["exp2"]},
      },
      "overrides": ["alg=preplay_jaxmaze", "rlenv=housemaze", "user=wilka"],
      "group": "preplay-final-run-7",
    }

  else:
    raise NotImplementedError(search)

  return sweep_config


CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "configs"))


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
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
