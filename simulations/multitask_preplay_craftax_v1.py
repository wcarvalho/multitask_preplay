"""
Dyna with the ability to do off-task simulation.
"""

from typing import Tuple, Optional, Union, Callable
import functools
from functools import partial
import distrax
import jax
import jax.numpy as jnp
import flax
from flax import struct
import optax
import flax.linen as nn
from gymnax.environments import environment
import numpy as np
import rlax
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use("Agg")

import wandb

from jaxneurorl import losses
from jaxneurorl.agents.basics import TimeStep
from jaxneurorl.agents import value_based_basics as vbb
from jaxneurorl.agents import qlearning as base_agent
from networks import MLP, CraftaxObsEncoder

from visualizer import plot_frames

from housemaze import renderer

Agent = nn.Module
Params = flax.core.FrozenDict
Qvalues = jax.Array
RngKey = jax.Array
make_actor = base_agent.make_actor

RnnState = jax.Array
SimPolicy = Callable[[Qvalues, RngKey], int]


@struct.dataclass
class AgentState:
  timestep: jax.Array
  rnn_state: jax.Array


@struct.dataclass
class Predictions:
  q_vals: jax.Array
  state: struct.PyTreeNode


@struct.dataclass
class SimulationOutput:
  actions: jax.Array
  predictions: Optional[Predictions] = None


def make_float(x):
  return x.astype(jnp.float32)


def concat_pytrees(tree1, tree2, **kwargs):
  return jax.tree_map(lambda x, y: jnp.concatenate((x, y), **kwargs), tree1, tree2)


def add_time(v):
  return jax.tree_map(lambda x: x[None], v)


def concat_first_rest(first, rest):
  first = add_time(first)  # [N, ...] --> [1, N, ...]
  # rest: [T, N, ...]
  # output: [T+1, N, ...]
  return jax.vmap(concat_pytrees, 1, 1)(first, rest)


def make_optimizer(config: dict) -> optax.GradientTransformation:
  num_updates = int(config["NUM_UPDATES"] + config.get("NUM_EXTRA_REPLAY", 0))

  lr_scheduler = optax.linear_schedule(
    init_value=config["LR"], end_value=1e-10, transition_steps=num_updates
  )

  lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

  return optax.chain(
    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
    optax.adam(learning_rate=lr, eps=config["EPS_ADAM"]),
  )


def simulate_n_trajectories(
  h_tm1: RnnState,
  x_t: TimeStep,
  goal: jax.Array,
  rng: jax.random.PRNGKey,
  network: nn.Module,
  params: Params,
  policy_fn: SimPolicy = None,
  q_fn: Callable = None,
  num_steps: int = 5,
  num_simulations: int = 5,
):
  """

  return predictions and actions for every time-step including the current one.

  This first applies the model to the current time-step and then simulates T more time-steps.
  Output is num_steps+1.

  Args:
      x_t (TimeStep): [D]
      h_tm1 (RnnState): [D]
      rng (jax.random.PRNGKey): _description_
      network (nn.Module): _description_
      params (Params): _description_
      temperatures (jax.Array): _description_
      num_steps (int, optional): _description_. Defaults to 5.
      num_simulations (int, optional): _description_. Defaults to 5.

  Returns:
      _type_: _description_
  """

  def initial_predictions(x, prior_h, w, rng_):
    # roll through RNN
    # get q-values
    lstm_state, lstm_out = network.apply(
      params, prior_h, x, rng_, method=network.apply_rnn
    )
    preds = Predictions(
      q_vals=network.apply(params, lstm_out, w, method=q_fn), state=lstm_state
    )
    return x, lstm_state, preds

  # by giving state as input and returning, will
  # return copies. 1 for each sampled action.
  rng, rng_ = jax.random.split(rng)

  # one for each simulation
  # [N, ...]
  # replace (x_t, task) with N-copies
  x_t, h_t, preds_t = jax.vmap(
    initial_predictions, in_axes=(None, None, 0, 0), out_axes=0
  )(
    x_t,
    h_tm1,
    goal,
    jax.random.split(rng_, num_simulations),  # [D]  # [D]
  )
  a_t = policy_fn(preds_t, rng_)

  def _single_model_step(carry, inputs):
    del inputs  # unused
    (timestep, lstm_state, a, rng) = carry

    ###########################
    # 1. use state + action to predict next state
    ###########################
    rng, rng_ = jax.random.split(rng)
    # apply model to get next timestep
    next_timestep = network.apply(params, timestep, a, rng_, method=network.apply_model)

    ###########################
    # 2. get actions at next state
    ###########################
    # [N]
    next_lstm_state, next_rnn_out = network.apply(
      params, lstm_state, next_timestep, rng_, method=network.apply_rnn
    )
    next_preds = Predictions(
      q_vals=network.apply(params, next_rnn_out, goal, method=q_fn),
      state=lstm_state,
    )
    next_a = policy_fn(next_preds, rng_)
    carry = (next_timestep, next_lstm_state, next_a, rng)
    sim_output = SimulationOutput(
      predictions=next_preds,
      actions=next_a,
    )
    return carry, (next_timestep, sim_output)

  ################
  # get simulation ouputs
  ################
  initial_carry = (x_t, h_t, a_t, rng)
  _, (next_timesteps, sim_outputs) = jax.lax.scan(
    f=_single_model_step, init=initial_carry, xs=None, length=num_steps
  )

  # sim_outputs.predictions: [T, N, ...]
  # concat [1, ...] with [N, T, ...]
  sim_outputs = SimulationOutput(
    predictions=concat_first_rest(preds_t, sim_outputs.predictions),
    actions=concat_first_rest(a_t, sim_outputs.actions),
  )
  all_timesteps = concat_first_rest(x_t, next_timesteps)
  return all_timesteps, sim_outputs


################################
# function to copy something n times
################################
def repeat(x, N: int):
  def identity(y, n):
    return y

  return jax.vmap(identity, (None, 0), 0)(x, jnp.arange(N))


def simulate_n_trajectories_with_actions(
  h_tm1: RnnState,
  x_t: TimeStep,
  actions: jax.Array,
  task: jax.Array,
  rng: jax.random.PRNGKey,
  network: nn.Module,
  params: Params,
  q_fn: Callable = None,
):
  """

  return predictions and actions for every time-step including the current one.

  This first applies the model to the current time-step and then simulates T more time-steps.
  Output is num_steps+1.

  Args:
      x_t (TimeStep): [D]
      h_tm1 (RnnState): [D]
      rng (jax.random.PRNGKey): _description_
      network (nn.Module): _description_
      params (Params): _description_
      temperatures (jax.Array): _description_
      num_steps (int, optional): _description_. Defaults to 5.
      num_simulations (int, optional): _description_. Defaults to 5.

  Returns:
      _type_: _description_
  """
  num_simulations, T = actions.shape[:2]

  def initial_predictions(x, prior_h, w, rng_):
    # roll through RNN
    # get q-values
    lstm_state, lstm_out = network.apply(
      params, prior_h, x, rng_, method=network.apply_rnn
    )
    preds = Predictions(
      q_vals=network.apply(params, lstm_out, task, method=q_fn), state=lstm_state
    )
    return x, w, lstm_state, preds

  # by giving state as input and returning, will
  # return copies. 1 for each sampled action.
  rng, rng_ = jax.random.split(rng)

  # one for each simulation
  # [N, ...]
  # replace (x_t, task) with N-copies
  x_t, task, h_t, preds_t = jax.vmap(
    initial_predictions, in_axes=(None, None, None, 0), out_axes=0
  )(
    x_t,
    h_tm1,
    task,
    jax.random.split(rng_, num_simulations),  # [D]  # [D]
  )

  def _single_model_step(carry, a):
    # NOTE: main difference is action is input, not carry
    (timestep, lstm_state, rng) = carry

    ###########################
    # 1. use state + action to predict next state
    ###########################
    rng, rng_ = jax.random.split(rng)
    # apply model to get next timestep
    next_timestep = network.apply(params, timestep, a, rng_, method=network.apply_model)

    ###########################
    # 2. get actions at next state
    ###########################
    # [N]
    next_lstm_state, next_rnn_out = network.apply(
      params, lstm_state, next_timestep, rng_, method=network.apply_rnn
    )
    next_preds = Predictions(
      q_vals=network.apply(params, next_rnn_out, task, method=q_fn),
      state=lstm_state,
    )

    carry = (next_timestep, next_lstm_state, rng)
    return carry, next_preds

  ################
  # get simulation ouputs
  ################
  initial_carry = (x_t, h_t, rng)
  _, sim_preds = jax.lax.scan(
    f=_single_model_step, init=initial_carry, xs=actions.transpose(1, 0)
  )

  return SimulationOutput(
    predictions=concat_first_rest(preds_t, sim_preds),
    actions=None,
  )


def apply_rnn_and_q(
  h_tm1: RnnState,
  timesteps: TimeStep,
  task: jax.Array,
  rng: jax.random.PRNGKey,
  network: nn.Module,
  params: Params,
  q_fn: Callable = None,
):
  """
  Sequentially applies RNN and Q-function to a sequence of timesteps using scan.

  Args:
      h_tm1 (RnnState): Initial RNN state [N, D]
      timesteps (TimeStep): Sequence of timesteps [T, N, ...]
      task (jax.Array): Task specification [N, ...]
      rng (jax.random.PRNGKey): Random key
      network (nn.Module): Network module
      params (Params): Network parameters
      q_fn (Callable): Q-function to apply

  Returns:
      SimulationOutput containing predictions
  """

  def _single_step(carry, x_t):
    lstm_state, rng = carry

    # Get new RNG
    rng, rng_ = jax.random.split(rng)

    # Apply RNN
    next_lstm_state, next_rnn_out = network.apply(
      params, lstm_state, x_t, rng_, method=network.apply_rnn
    )

    # Apply Q-function
    next_preds = Predictions(
      q_vals=network.apply(params, next_rnn_out, task, method=q_fn),
      state=next_lstm_state,
    )

    return (next_lstm_state, rng), next_preds

  # Run scan over sequence
  initial_carry = (h_tm1, rng)
  _, preds = jax.lax.scan(f=_single_step, init=initial_carry, xs=timesteps)

  return preds


@struct.dataclass
class MultitaskPreplay(vbb.RecurrentLossFn):
  """Loss function for multitask preplay.

  Note: this assumes the agent uses the ground-truth environment as the environment model.
  """

  num_simulations: int = 15
  simulation_length: int = 5
  online_coeff: float = 1.0
  dyna_coeff: float = 1.0
  offtask_coeff: float = 1.0
  importance_sampling_exponent: float = 0.6

  simulation_policy: SimPolicy = None

  def loss_fn(
    self,
    timestep,
    online_preds,
    target_preds,
    actions,
    rewards,
    is_last,
    non_terminal,
    loss_mask,
  ):
    rewards = make_float(rewards)
    rewards = rewards - self.step_cost
    is_last = make_float(is_last)
    discounts = make_float(non_terminal) * self.discount
    lambda_ = jnp.ones_like(non_terminal) * self.lambda_

    # Get N-step transformed TD error and loss.
    batch_td_error_fn = jax.vmap(losses.q_learning_lambda_td, in_axes=1, out_axes=1)

    # [T, B]
    selector_actions = jnp.argmax(online_preds.q_vals, axis=-1)  # [T+1, B]
    q_t, target_q_t = batch_td_error_fn(
      online_preds.q_vals[:-1],  # [T+1] --> [T]
      actions[:-1],  # [T+1] --> [T]
      target_preds.q_vals[1:],  # [T+1] --> [T]
      selector_actions[1:],  # [T+1] --> [T]
      rewards[1:],  # [T+1] --> [T]
      discounts[1:],
      is_last[1:],
      lambda_[1:],
    )  # [T+1] --> [T]

    # ensure target = 0 when episode terminates
    target_q_t = target_q_t * non_terminal[:-1]
    batch_td_error = target_q_t - q_t
    batch_td_error = batch_td_error * loss_mask[:-1]

    # [T, B]
    batch_loss = 0.5 * jnp.square(batch_td_error)

    # [B]
    batch_loss_mean = (batch_loss * loss_mask[:-1]).mean(0)

    metrics = {
      "0.q_loss": batch_loss.mean(),
      "0.q_td": jnp.abs(batch_td_error).mean(),
      "1.reward": rewards[1:].mean(),
      "z.q_mean": online_preds.q_vals.mean(),
      "z.q_var": online_preds.q_vals.var(),
    }

    log_info = {
      "timesteps": timestep,
      "actions": actions,
      "td_errors": batch_td_error,  # [T]
      "loss_mask": loss_mask,  # [T]
      "q_values": online_preds.q_vals,  # [T, B]
      "q_loss": batch_loss,  # [ T, B]
      "q_target": target_q_t,
    }

    return batch_td_error, batch_loss_mean, metrics, log_info

  def error(
    self,
    data,
    online_preds: Predictions,
    online_state: AgentState,
    target_preds: Predictions,
    target_state: AgentState,
    params: Params,
    target_params: Params,
    steps: int,
    key_grad: jax.random.PRNGKey,
    **kwargs,
  ):
    ##################
    ## Q-learning loss on batch of data
    ##################

    # prepare data
    non_terminal = data.timestep.discount
    # either termination or truncation
    is_last = make_float(data.timestep.last())

    # truncated is discount on AND is last
    truncated = (non_terminal + is_last) > 1
    loss_mask = make_float(1 - truncated)

    all_metrics = {}
    all_log_info = {
      "n_updates": steps,
    }
    T, B = loss_mask.shape[:2]
    if self.online_coeff > 0.0:
      td_error, batch_loss, metrics, log_info = self.loss_fn(
        timestep=data.timestep,
        online_preds=online_preds,
        target_preds=target_preds,
        actions=data.action,
        rewards=data.reward,
        is_last=is_last,
        non_terminal=non_terminal,
        loss_mask=loss_mask,
      )
      # first label online loss with online
      all_metrics.update({f"online/{k}": v for k, v in metrics.items()})
      all_log_info["online"] = log_info
      td_error = jnp.concatenate((td_error, jnp.zeros(B)[None]), 0)
      td_error = jnp.abs(td_error)
    else:
      td_error = jnp.zeros_like(loss_mask)
      batch_loss = td_error.sum(0)  # time axis

    #################
    # Dyna Q-learning loss over simulated data
    #################
    if self.dyna_coeff > 0.0:
      # will use time-step + previous rnn-state to simulate
      # next state at each time-step and compute predictions
      remove_last = lambda x: jax.tree_map(lambda y: y[:-1], x)
      h_tm1_online = concat_first_rest(online_state, remove_last(online_preds.state))
      h_tm1_target = concat_first_rest(target_state, remove_last(target_preds.state))

      x_t = data.timestep

      T, B = h_tm1_online[0].shape[:2]
      rngs = jax.random.split(key_grad, T * B)
      rngs = rngs.reshape(T, B, -1)

      dyna_loss_fn = functools.partial(
        self.dyna_loss_fn, params=params, target_params=target_params
      )

      # vmap over batch + time
      dyna_loss_fn = jax.vmap(jax.vmap(dyna_loss_fn))
      dyna_td_error, dyna_batch_loss, dyna_metrics, dyna_log_info = dyna_loss_fn(
        x_t,
        h_tm1_online,
        h_tm1_target,
        rngs,
      )
      # [time, batch, num_sim, sim_length]
      # average over (num_sim, sim_length)
      dyna_td_error = dyna_td_error.mean(axis=(2, 3))
      # average over (time, num_sim)
      dyna_batch_loss = dyna_batch_loss.mean(axis=(0, 2))

      td_error += self.dyna_coeff * jnp.abs(dyna_td_error)

      batch_loss += self.dyna_coeff * dyna_batch_loss

      # update metrics with dyna metrics
      all_metrics.update({f"dyna/{k}": v for k, v in dyna_metrics.items()})

      all_log_info["dyna"] = dyna_log_info

    if self.logger.learner_log_extra is not None:
      self.logger.learner_log_extra(all_log_info)

    return td_error, batch_loss, all_metrics

  def dyna_loss_fn(
    self,
    x_t: TimeStep,
    h_tm1_online: jax.Array,
    h_tm1_target: jax.Array,
    rng: jax.random.PRNGKey,
    params,
    target_params,
  ):
    """

    Algorithm:
    -----------
    if off-task:
      - sample possible goal
      - use subtask q-fn to sample trajectories
      - off-task q-fn:
        - compute target predictions using target network
        - compute rewards for subtask
        - compute loss-fn for subtask reward
      - regular q-fn:
        - compute q-values for both {online, target} networks
        - compute loss-fn for regular reward
    else:
      - use regular q-fn to sample trajectories
      - compute target predictions using target network
      - compute loss-fn for regular reward

    Args:
        x_t (TimeStep): [D], timestep at t
        h_tm1 (jax.Array): [D], rnn-state at t-1
        h_tm1_target (jax.Array): [D], rnn-state at t-1 from target network
    """

    h_tm1_online_repeated = repeat(h_tm1_online, self.num_simulations)
    h_tm1_target_repeated = repeat(h_tm1_target, self.num_simulations)

    if self.offtask_coeff > 0.0:
      # sample possible goal
      # Sample 1-hot vector from achievable binary vector using categorical distribution
      rng, rng_ = jax.random.split(rng)
      # incase 0, add 1e-5
      achievable = x_t.observation.achievable.astype(jnp.float32) + 1e-5
      num_classes = x_t.observation.achievable.shape[-1]
      achievable = achievable / achievable.sum()
      goals = distrax.Categorical(probs=achievable).sample(
        seed=rng_, sample_shape=(self.num_simulations,)
      )
      goals = jax.nn.one_hot(goals, num_classes=num_classes)

      # ---------------------------
      # first use online params + off-task q-fn
      # ---------------------------
      # [self.simulation_length+1, ...]
      # from t=1, onwards including x_t
      rng, rng_ = jax.random.split(rng)
      # [T, K, ...]
      timesteps_t, sim_outputs_t = simulate_n_trajectories(
        h_tm1=h_tm1_online,
        x_t=x_t,
        rng=rng_,
        network=self.network,
        params=params,
        num_steps=self.simulation_length,
        num_simulations=self.num_simulations,
        policy_fn=self.simulation_policy,
        q_fn=self.network.subtask_q_fn,
        goal=goals,
      )
      offtask_online_preds = sim_outputs_t.predictions
      non_terminal = timesteps_t.discount
      # either termination or truncation
      is_last_t = make_float(timesteps_t.last())

      # time-step of termination and everything afterwards is masked out
      term_cumsum_t = jnp.cumsum(is_last_t, 0)
      loss_mask_t = make_float((term_cumsum_t + non_terminal) < 2)

      rng, rng_ = jax.random.split(rng)

      # [T, K, ...]
      offtask_target_preds = apply_rnn_and_q(
        h_tm1=h_tm1_target_repeated,
        timesteps=timesteps_t,
        task=goals,
        rng=rng_,
        network=self.network,
        params=target_params,
        q_fn=self.network.subtask_q_fn,
      )

      achievements = timesteps_t.observation.achievements
      # [T, K, D] * [1, K, D] --> [T, K]
      offtask_reward = (achievements * goals[None]).sum(-1)

      offtask_td_error, offtask_loss_mean, offtask_metrics, offtask_log_info = (
        self.loss_fn(
          timestep=timesteps_t,
          online_preds=offtask_online_preds,
          target_preds=offtask_target_preds,
          actions=sim_outputs_t.actions,
          rewards=offtask_reward,
          is_last=is_last_t,
          non_terminal=timesteps_t.discount,
          loss_mask=loss_mask_t,
        )
      )

      # ---------------------------
      # now regular q-fn
      # ---------------------------
      rng, rng_ = jax.random.split(rng)
      main_online_preds = apply_rnn_and_q(
        h_tm1=h_tm1_online_repeated,
        timesteps=timesteps_t,
        task=goals * 0,  # NOTE: goal will be ignored
        rng=rng_,
        network=self.network,
        params=params,
        q_fn=self.network.reg_q_fn,
      )
      rng, rng_ = jax.random.split(rng)
      main_target_preds = apply_rnn_and_q(
        h_tm1=h_tm1_target_repeated,
        timesteps=timesteps_t,
        task=goals * 0,  # NOTE: goal will be ignored
        rng=rng_,
        network=self.network,
        params=target_params,
        q_fn=self.network.reg_q_fn,
      )
      main_td_error, main_loss_mean, main_metrics, _ = self.loss_fn(
        timestep=timesteps_t,
        online_preds=main_online_preds,
        target_preds=main_target_preds,
        actions=sim_outputs_t.actions,
        rewards=timesteps_t.reward,
        is_last=is_last_t,
        non_terminal=timesteps_t.discount,
        loss_mask=loss_mask_t,
      )

      batch_td_error = self.offtask_coeff * jnp.abs(offtask_td_error) + jnp.abs(
        main_td_error
      )
      batch_loss_mean = self.offtask_coeff * offtask_loss_mean + main_loss_mean
      metrics = {
        **{f"offtask/{k}": v for k, v in offtask_metrics.items()},
        **main_metrics,
      }
      log_info = offtask_log_info

    else:
      # dummy goal that will be ignored
      goals = jnp.zeros_like(x_t.observation.achievable)
      goals = repeat(goals, self.num_simulations)

      rng, rng_ = jax.random.split(rng)
      timesteps_t, sim_outputs_t = simulate_n_trajectories(
        h_tm1=h_tm1_online,
        x_t=x_t,
        rng=rng_,
        network=self.network,
        params=params,
        num_steps=self.simulation_length,
        num_simulations=self.num_simulations,
        policy_fn=self.simulation_policy,
        q_fn=self.network.reg_q_fn,
        goal=goals,
      )
      online_preds = sim_outputs_t.predictions
      non_terminal = timesteps_t.discount
      # either termination or truncation
      is_last_t = make_float(timesteps_t.last())

      # time-step of termination and everything afterwards is masked out
      term_cumsum_t = jnp.cumsum(is_last_t, 0)
      loss_mask_t = make_float((term_cumsum_t + non_terminal) < 2)

      rng, rng_ = jax.random.split(rng)
      target_preds = apply_rnn_and_q(
        h_tm1=h_tm1_target_repeated,
        timesteps=timesteps_t,
        task=goals,
        rng=rng_,
        network=self.network,
        params=target_params,
        q_fn=self.network.reg_q_fn,
      )

      batch_td_error, batch_loss_mean, metrics, log_info = self.loss_fn(
        timestep=timesteps_t,
        online_preds=online_preds,
        target_preds=target_preds,
        actions=sim_outputs_t.actions,
        rewards=timesteps_t.reward,
        is_last=is_last_t,
        non_terminal=timesteps_t.discount,
        loss_mask=loss_mask_t,
      )

    return batch_td_error, batch_loss_mean, metrics, log_info


def make_loss_fn_class(config, **kwargs) -> MultitaskPreplay:
  return functools.partial(
    MultitaskPreplay,
    discount=config["GAMMA"],
    lambda_=config.get("TD_LAMBDA", 0.9),
    online_coeff=config.get("ONLINE_COEFF", 1.0),
    dyna_coeff=config.get("DYNA_COEFF", 1.0),
    offtask_coeff=config.get("OFFTASK_COEFF", 1.0),
    num_simulations=config.get("NUM_SIMULATIONS", 15),
    simulation_length=config.get("SIMULATION_LENGTH", 5),
    step_cost=config.get("STEP_COST", 0.0),
    **kwargs,
  )


def get_in_episode(timestep):
  # get mask for within episode
  non_terminal = timestep.discount
  is_last = timestep.last()
  term_cumsum = jnp.cumsum(is_last, -1)
  in_episode = (term_cumsum + non_terminal) < 2
  return in_episode


from craftax.craftax.constants import Action, BLOCK_PIXEL_SIZE_IMG
from craftax.craftax.renderer import render_craftax_pixels
from visualizer import plot_frames


def render_fn(state):
  image = render_craftax_pixels(state, block_pixel_size=BLOCK_PIXEL_SIZE_IMG)
  return image / 255.0


render_fn = jax.jit(render_fn)


def learner_log_extra(
  data: dict,
  config: dict,
  sim_idx: int = 0,
):
  def log_data(
    key: str,
    timesteps: TimeStep,
    actions: np.array,
    td_errors: np.array,
    loss_mask: np.array,
    q_values: np.array,
    q_loss: np.array,
    q_target: np.array,
  ):
    # Extract the relevant data
    # only use data from batch dim = 0
    # [T, B, ...] --> # [T, ...]

    discounts = timesteps.discount
    rewards = timesteps.reward
    q_values_taken = rlax.batched_index(q_values, actions)

    # Create a figure with three subplots
    width = 0.3
    nT = len(rewards)  # e.g. 20 --> 8
    width = max(int(width * nT), 10)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(width, 20))

    # Plot rewards and q-values in the top subplot
    def format(ax):
      ax.set_xlabel("Time")
      ax.grid(True)
      ax.set_xticks(range(0, len(rewards), 1))

    ax1.plot(rewards, label="Rewards")
    ax1.plot(q_values_taken, label="Q-Values")
    ax1.plot(q_target, label="Q-Targets")
    ax1.set_title("Rewards and Q-Values")
    format(ax1)
    ax1.legend()

    # Plot TD errors in the middle subplot
    ax2.plot(td_errors)
    format(ax2)
    ax2.set_title("TD Errors")

    # Plot Q-loss in the bottom subplot
    ax3.plot(q_loss)
    format(ax3)
    ax3.set_title("Q-Loss")

    # Plot episode quantities
    is_last = timesteps.last()
    ax4.plot(discounts, label="Discounts")
    ax4.plot(loss_mask, label="mask")
    ax4.plot(is_last, label="is_last")
    format(ax4)
    ax4.set_title("Episode markers")
    ax4.legend()

    if wandb.run is not None:
      wandb.log({f"learner_example/{key}/q-values": wandb.Image(fig)})
    plt.close(fig)

    ##############################
    # plot images of env
    ##############################
    # ------------
    # get images
    # ------------

    # state_images = []
    obs_images = []
    max_len = min(config.get("MAX_EPISODE_LOG_LEN", 40), len(rewards))
    for idx in range(max_len):
      index = lambda y: jax.tree_map(lambda x: x[idx], y)
      obs_image = render_fn(index(timesteps.state.env_state))
      obs_images.append(obs_image)

    # ------------
    # plot
    # ------------
    actions_taken = [Action(a).name for a in actions]

    def index(t, idx):
      return jax.tree_map(lambda x: x[idx], t)

    def panel_title_fn(timesteps, i):
      title = f"t={i}\n"
      title += f"{actions_taken[i]}\n"
      title += f"r={timesteps.reward[i]}, $\\gamma={timesteps.discount[i]}$"
      return title

    fig = plot_frames(
      timesteps=timesteps,
      frames=obs_images,
      panel_title_fn=panel_title_fn,
      ncols=6,
    )
    if wandb.run is not None:
      wandb.log({f"learner_example/{key}/trajectory": wandb.Image(fig)})
    plt.close(fig)

  def callback(d):
    log_data(**d, key="dyna")

  # this will be the value after update is applied
  n_updates = data["n_updates"] + 1
  is_log_time = n_updates % config["LEARNER_EXTRA_LOG_PERIOD"] == 0

  if "dyna" in data:
    # [T, B, K, N] --> [K]
    # K = the simulation length
    # get entire simulation, starting at:
    #   T=0 (1st time-point)
    #   B=0 (1st batch sample)
    #   N=index(t_min) (simulation with lowest temperaturee)
    dyna_data = jax.tree_map(lambda x: x[0, 0, :, sim_idx], data["dyna"])

    jax.lax.cond(
      is_log_time,
      lambda d: jax.debug.callback(callback, d),
      lambda d: None,
      dyna_data,
    )


class DynaAgentEnvModel(nn.Module):
  """

  Note: predictions contains rnn_state because when you use unroll, you only get the final rnn_state but predictions for all time-steps.

  """

  observation_encoder: nn.Module
  rnn: vbb.ScannedRNN
  q_fn: nn.Module
  q_fn_subtask: nn.Module
  env: environment.Environment
  env_params: environment.EnvParams

  def initialize(self, x: TimeStep):
    """Only used for initialization."""
    # [B, D]
    rng = jax.random.PRNGKey(0)
    batch_dims = (x.reward.shape[0],)
    rnn_state = self.initialize_carry(rng, batch_dims)
    predictions, rnn_state = self.__call__(rnn_state, x, rng)
    # dummy_action = jnp.zeros(batch_dims, dtype=jnp.int32)
    # self.apply_model(predictions.state, dummy_action, rng)
    task = jnp.zeros_like(x.observation.achievable)
    self.subtask_q_fn(rnn_state[1], task)

  def initialize_carry(self, *args, **kwargs):
    """Initializes the RNN state."""
    return self.rnn.initialize_carry(*args, **kwargs)

  def apply_rnn(self, rnn_state, x: TimeStep, rng: jax.random.PRNGKey):
    embedding = self.observation_encoder(x.observation)
    rnn_in = vbb.RNNInput(obs=embedding, reset=x.first())
    rng, _rng = jax.random.split(rng)
    return self.rnn(rnn_state, rnn_in, _rng)

  def __call__(
    self, rnn_state, x: TimeStep, rng: jax.random.PRNGKey
  ) -> Tuple[Predictions, RnnState]:
    new_rnn_state, rnn_out = self.apply_rnn(rnn_state, x, rng)
    q_vals = self.q_fn(rnn_out)
    predictions = Predictions(q_vals=q_vals, state=new_rnn_state)

    return predictions, new_rnn_state

  def reg_q_fn(self, rnn_out, task):
    # just so both have same signature
    del task
    return self.q_fn(rnn_out)

  def subtask_q_fn(self, rnn_out, task):
    inp = jnp.concatenate((rnn_out, task), axis=-1)
    return self.q_fn_subtask(inp)

  def unroll(
    self, rnn_state, xs: TimeStep, rng: jax.random.PRNGKey
  ) -> Tuple[Predictions, RnnState]:
    # rnn_state: [B]
    # xs: [T, B]

    embedding = jax.vmap(self.observation_encoder)(xs.observation)

    rnn_in = vbb.RNNInput(obs=embedding, reset=xs.first())
    rng, _rng = jax.random.split(rng)

    # [B, D], [T, B, D]
    new_rnn_state, new_rnn_states = self.rnn.unroll(rnn_state, rnn_in, _rng)

    rnn_out = self.rnn.output_from_state(new_rnn_states)
    q_vals = nn.BatchApply(self.q_fn)(rnn_out)
    predictions = Predictions(q_vals=q_vals, state=new_rnn_states)
    return predictions, new_rnn_state

  def apply_model(
    self,
    state: TimeStep,
    action: jnp.ndarray,
    rng: jax.random.PRNGKey,
  ) -> Tuple[Predictions, RnnState]:
    """This applies the model to each element in the state, action vectors.
    Args:
        state (State): states. [B, D]
        action (jnp.ndarray): actions to take on states. [B]
    Returns:
        Tuple[ModelOutput, State]: muzero outputs and new states for
          each state state action pair.
    """
    # take one step forward in the environment
    B = action.shape[0]

    def env_step(s, a, rng_):
      return self.env.step(rng_, s, a, self.env_params)

    return jax.vmap(env_step)(state, action, jax.random.split(rng, B))

  def compute_reward(self, timestep, task):
    return self.env.compute_reward(timestep, task, self.env_params)


def make_agent(
  config: dict,
  env: environment.Environment,
  env_params: environment.EnvParams,
  example_timestep: TimeStep,
  rng: jax.random.PRNGKey,
  model_env: Optional[environment.Environment] = None,
  model_env_params: Optional[environment.EnvParams] = None,
) -> Tuple[nn.Module, Params, vbb.AgentResetFn]:
  model_env_params = model_env_params or env_params
  cell_type = config.get("RNN_CELL_TYPE", "OptimizedLSTMCell")
  if cell_type.lower() == "none":
    rnn = vbb.DummyRNN()
  else:
    rnn = vbb.ScannedRNN(
      hidden_dim=config.get("AGENT_RNN_DIM", 256),
      cell_type=cell_type,
      unroll_output_state=True,
    )
  agent = DynaAgentEnvModel(
    observation_encoder=CraftaxObsEncoder(
      hidden_dim=config["MLP_HIDDEN_DIM"],
      num_layers=config["NUM_MLP_LAYERS"],
      activation=config["ACTIVATION"],
      norm_type=config.get("NORM_TYPE", "none"),
      structured_inputs=config.get("STRUCTURED_INPUTS", False),
      use_bias=config.get("USE_BIAS", True),
    ),
    rnn=rnn,
    q_fn=MLP(
      hidden_dim=config.get("Q_HIDDEN_DIM", 512),
      num_layers=config.get("NUM_Q_LAYERS", 1),
      out_dim=env.action_space(env_params).n,
      activation=config["ACTIVATION"],
      activate_final=False,
      use_bias=config.get("USE_BIAS", True),
    ),
    q_fn_subtask=MLP(
      hidden_dim=config.get("Q_HIDDEN_DIM", 512),
      num_layers=config.get("NUM_Q_LAYERS", 1),
      out_dim=env.action_space(env_params).n,
      activation=config["ACTIVATION"],
      activate_final=False,
      use_bias=config.get("USE_BIAS", True),
    ),
    env=model_env,
    env_params=model_env_params,
  )

  rng, _rng = jax.random.split(rng)
  network_params = agent.init(_rng, example_timestep, method=agent.initialize)

  def reset_fn(params, example_timestep, reset_rng):
    batch_dims = example_timestep.reward.shape
    return agent.apply(
      params, batch_dims=batch_dims, rng=reset_rng, method=agent.initialize_carry
    )

  return agent, network_params, reset_fn


def make_train(**kwargs):
  config = kwargs["config"]
  rng = jax.random.PRNGKey(config["SEED"])

  epsilon_setting = config["SIM_EPSILON_SETTING"]
  if epsilon_setting == 1:
    # ACME default
    # range of ~(0.001, .1)
    vals = np.logspace(num=256, start=1, stop=3, base=0.1)
  elif epsilon_setting == 2:
    # range of ~(.9,.1)
    vals = np.logspace(num=256, start=0.05, stop=0.9, base=0.1)
  elif epsilon_setting == 3:
    # very random
    vals = np.ones(256) * 0.9

  num_simulations = config["NUM_SIMULATIONS"]
  epsilons = jax.random.choice(rng, vals, shape=(num_simulations - 1,))
  epsilons = jnp.concatenate((jnp.zeros(1), epsilons))
  # greedy_idx = int(epsilons.argmin())

  def simulation_policy(preds: Predictions, sim_rng: jax.Array):
    q_values = preds.q_vals
    assert q_values.shape[0] == epsilons.shape[0]
    sim_rng = jax.random.split(sim_rng, q_values.shape[0])
    return jax.vmap(base_agent.epsilon_greedy_act, in_axes=(0, 0, 0))(
      q_values, epsilons, sim_rng
    )

  return vbb.make_train(
    make_agent=partial(make_agent, model_env=kwargs.pop("model_env")),
    make_loss_fn_class=functools.partial(
      make_loss_fn_class,
      simulation_policy=simulation_policy,
    ),
    make_optimizer=base_agent.make_optimizer,
    make_actor=make_actor,
    **kwargs,
  )
