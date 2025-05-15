"""
Dyna

# TODO: reuse RNN computation for Dyna more
# TODO: incorporate windowed overlapping Dyna TDs into TD error
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
import jax.tree_util as jtu

import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use("Agg")

import wandb

import craftax_env
import craftax_web_env
from craftax_web_env import Achiement_to_idx

from jaxneurorl import losses
from jaxneurorl.agents.basics import TimeStep
from jaxneurorl.agents import value_based_basics as vbb
from jaxneurorl.agents import qlearning as base_agent
from networks import MLP, CraftaxObsEncoder, CraftaxMultiGoalObsEncoder
from networks import CategoricalHouzemazeObsEncoder

from visualizer import plot_frames

from housemaze import renderer


def info(x):
  return jax.tree.map(lambda y: (y.shape, y.dtype), x)


Agent = nn.Module
Params = flax.core.FrozenDict
Qvalues = jax.Array
RngKey = jax.Array
make_actor = base_agent.make_actor

RnnState = jax.Array
SimPolicy = Callable[[Qvalues, RngKey], int]

MAX_REWARD = 1.0


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
  return jax.tree.map(lambda x, y: jnp.concatenate((x, y), **kwargs), tree1, tree2)


def add_time(v):
  return jax.tree.map(lambda x: x[None], v)


def concat_first_rest(first, rest):
  first = add_time(first)  # [N, ...] --> [1, N, ...]
  # rest: [T, N, ...]
  # output: [T+1, N, ...]
  return jax.vmap(concat_pytrees, 1, 1)(first, rest)


def concat_start_sims(start, simulations):
  # concat where vmap over simulation dimension
  # need this since have 1 start path, but multiple simulations
  concat_ = lambda a, b: jnp.concatenate((a, b))
  concat_ = jax.vmap(concat_, (None, 1), 1)
  return jax.tree.map(concat_, start, simulations)


def is_truncated(timestep):
  non_terminal = timestep.discount
  # either termination or truncation
  is_last = make_float(timestep.last())

  # truncated is discount=1 on AND is last
  truncated = (non_terminal + is_last) > 1
  return make_float(1 - truncated)


def simulation_finished_mask(initial_mask, next_timesteps):
  # get mask
  non_terminal = next_timesteps.discount[1:]
  # either termination or truncation
  is_last_t = make_float(next_timesteps.last()[1:])

  # time-step of termination and everything afterwards is masked out
  term_cumsum_t = jnp.cumsum(is_last_t, 0)
  loss_mask_t = make_float((term_cumsum_t + non_terminal) < 2)
  return concat_start_sims(initial_mask, loss_mask_t)


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


@partial(jax.jit, static_argnums=(1,))
def rolling_window(a, size: int):
  """Create rolling windows of a specified size from an input array.

  Rolls over the first dimension only, preserving other dimensions.

  Args:
      a (array-like): The input array of shape [T, ...]
      size (int): The size of the rolling window

  Returns:
      Array of shape [T-size+1, size, ...]
  """
  # Get shape info
  T = a.shape[0]  # length of first dimension
  remaining_dims = a.shape[1:]  # all other dimensions

  # Create start indices for the first dimension only
  starts = jnp.arange(T - size + 1)

  # Create slice for each start index, preserving other dimensions
  def slice_at(start):
    idx = (start,) + (0,) * len(remaining_dims)  # index tuple for all dims
    size_tuple = (size,) + remaining_dims  # size tuple for all dims
    return jax.lax.dynamic_slice(a, idx, size_tuple)

  return jax.vmap(slice_at)(starts)


def simulate_n_trajectories(
  h_tm1: RnnState,
  x_t: TimeStep,
  goal: jax.Array,
  rng: jax.random.PRNGKey,
  network: nn.Module,
  params: Params,
  policy_fn: SimPolicy = None,
  num_steps: int = 5,
  num_simulations: int = 5,
  use_offtask_policy: jax.Array = None,
  terminate_offtask: bool = False,
  mainq_coeff: float = 1.0,
  subtask_coeff: float = 1.0,
  make_init_goal_timestep=None,
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

  alpha = use_offtask_policy.astype(jnp.float32)

  def get_q_vals(beta, lstm_out, w):
    """
    Q-values that switch between subtask and regular Q-values.
    alpha is a global parameter, beta is updated per timestep.

    If alpha=1, use off-task Q-values, else use regular Q-values.
    If beta=0, use subtask Q-values, else use regular Q-values.
    beta can be interpeted as a termination parameter.

    Q_subtask(s, a) = alpha*Q_subtask(s,a) + (1-alpha)*Q_regular(s,a)
    Q(s, a) = (1 - beta)*Q_subtask(s, a) + beta*Q_regular(s, a)
    """
    beta = beta.astype(jnp.float32)

    subtask_q = network.apply(params, lstm_out, w, method=network.subtask_q_fn)
    reg_q = network.apply(params, lstm_out, w, method=network.reg_q_fn)

    # use subtask_q if use_offtask_policy=1 else use reg_q
    subtask_q = alpha * subtask_q + (1 - alpha) * reg_q
    if terminate_offtask:
      q_vals = (1 - beta) * subtask_q + beta * reg_q
    else:
      q_vals = mainq_coeff * reg_q + subtask_coeff * subtask_q
    return q_vals

  def initial_predictions(x, prior_h, w, rng_):
    # roll through RNN
    # get q-values
    lstm_state, lstm_out = network.apply(
      params, prior_h, x, rng_, method=network.apply_rnn
    )

    beta = jnp.zeros((), dtype=jnp.int32)  # scalar
    preds = Predictions(q_vals=get_q_vals(beta, lstm_out, w), state=lstm_state)

    if make_init_goal_timestep is not None:
      x = make_init_goal_timestep(x, w)
    return x, beta, lstm_state, preds

  # by giving state as input and returning, will
  # return copies. 1 for each sampled action.
  rng, rng_ = jax.random.split(rng)

  # one for each simulation
  # [N, ...]
  # replace (x_t, task) with N-copies
  x_t, beta_t, h_t, preds_t = jax.vmap(
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
    (timestep, beta, lstm_state, a, rng) = carry

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
    if terminate_offtask:
      # as soon as have first success, terminate off-task by setting beta to 1
      # beta will always be 1 afterwards switching to Q-fn
      coeff = goal.astype(jnp.float32)
      achievements = next_timestep.observation.achievements.astype(jnp.float32)
      offtask_reward = (achievements * coeff).sum(-1)
      beta = ((beta + offtask_reward) > 0).astype(jnp.int32)

    next_preds = Predictions(
      q_vals=jax.vmap(get_q_vals)(beta, next_rnn_out, goal), state=next_lstm_state
    )
    next_a = policy_fn(next_preds, rng_)
    carry = (next_timestep, beta, next_lstm_state, next_a, rng)
    sim_output = SimulationOutput(
      predictions=next_preds,
      actions=next_a,
    )
    return carry, (next_timestep, sim_output)

  ################
  # get simulation ouputs
  ################
  initial_carry = (x_t, beta_t, h_t, a_t, rng)
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


def reset_achievements(x_t):
  return x_t.replace(
    state=x_t.state.replace(
      env_state=x_t.state.env_state.replace(
        achievements=jnp.zeros_like(x_t.state.env_state.achievements),
      )
    )
  )


################################
# function to copy something n times
################################
def repeat(x, N: int):
  def identity(y, n):
    return y

  return jax.vmap(identity, (None, 0), 0)(x, jnp.arange(N))


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
class DynaLossFn(vbb.RecurrentLossFn):
  """Loss function for multitask preplay.

  Note: this assumes the agent uses the ground-truth environment as the environment model.
  """

  window_size: int = 5
  num_dyna_simulations: int = 1
  num_offtask_simulations: int = 1
  simulation_length: int = 5
  online_coeff: float = 1.0
  dyna_coeff: float = 1.0
  max_priority_weight: float = 0.9
  importance_sampling_exponent: float = 0.6
  num_offtask_goals: int = 5
  offtask_coeff: float = 1.0
  main_coeff: float = 1.0
  terminate_offtask: bool = False
  subtask_coeff: float = 1.0
  mainq_coeff: float = 1.0
  backtracking: bool = True
  combine_real_sim: bool = False

  dyna_policy: SimPolicy = None
  offtask_dyna_policy: SimPolicy = None
  sample_goals: Callable = None
  compute_rewards: Callable = None
  get_main_goal: Callable = None
  make_init_goal_timestep: Callable = None

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
    batch_loss_mean = (batch_loss * loss_mask[:-1]).sum(0) / loss_mask[:-1].sum(0)

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

    # truncated is discount on AND is last
    loss_mask = is_truncated(data.timestep)

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
        rewards=data.reward / MAX_REWARD,
        is_last=make_float(data.timestep.last()),
        non_terminal=data.timestep.discount,
        loss_mask=loss_mask,
      )
      # first label online loss with online
      all_metrics.update({f"{k}/online": v for k, v in metrics.items()})
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
      remove_last = lambda x: jax.tree.map(lambda y: y[:-1], x)
      h_tm1_online = concat_first_rest(online_state, remove_last(online_preds.state))
      h_tm1_target = concat_first_rest(target_state, remove_last(target_preds.state))
      x_t = data.timestep

      dyna_loss_fn = functools.partial(
        self.preplay_loss_fn, params=params, target_params=target_params
      )

      # vmap over batch
      dyna_loss_fn = jax.vmap(dyna_loss_fn, (1, 1, 1, 1, 1, 0), 0)
      _, dyna_batch_loss, dyna_metrics, dyna_log_info = dyna_loss_fn(
        x_t,
        data.action,
        h_tm1_online,
        h_tm1_target,
        loss_mask,
        jax.random.split(key_grad, B),
      )
      batch_loss += self.dyna_coeff * dyna_batch_loss

      # update metrics with dyna metrics
      all_metrics.update({f"{k}/dyna": v for k, v in dyna_metrics.items()})

      all_log_info["dyna"] = dyna_log_info

    if self.logger.learner_log_extra is not None:
      self.logger.learner_log_extra(all_log_info)

    return td_error, batch_loss, all_metrics

  def preplay_loss_fn(
    self,
    timesteps: TimeStep,
    actions: jax.Array,
    h_online: jax.Array,
    h_target: jax.Array,
    loss_mask: jax.Array,
    rng: jax.random.PRNGKey,
    params,
    target_params,
  ):
    """

    Algorithm:
    -----------

    Args:
        x_t (TimeStep): [D], timestep at t
        h_tm1 (jax.Array): [D], rnn-state at t-1
        h_tm1_target (jax.Array): [D], rnn-state at t-1 from target network
    """
    if not isinstance(self.window_size, (int, float)):
      raise TypeError(
        f"window_size must be an int or float, got {type(self.window_size)}"
      )

    if isinstance(self.window_size, int):
      window_size = self.window_size
    else:
      # float (i.e. percent)
      window_size = int(self.window_size * len(actions))
    window_size = min(window_size, len(actions))
    window_size = max(window_size, 1)

    roll = partial(rolling_window, size=window_size)
    simulate = partial(
      simulate_n_trajectories,
      network=self.network,
      params=params,
      num_steps=self.simulation_length,
      terminate_offtask=self.terminate_offtask,
      mainq_coeff=self.mainq_coeff,
      subtask_coeff=self.subtask_coeff,
      make_init_goal_timestep=self.make_init_goal_timestep,
    )

    # first do a rollowing window
    # T' = T-window_size+1
    # K = window_size
    # [T, ...] --> [T', K, ...]
    actions = jax.tree.map(roll, actions)
    timesteps = jax.tree.map(roll, timesteps)
    h_online = jax.tree.map(roll, h_online)
    h_target = jax.tree.map(roll, h_target)
    loss_mask = jax.tree.map(roll, loss_mask)

    def offtask_dyna_loss_fn_(
      t, a, h_on, h_tar, l_mask, achieve_poss, any_achievable, g, key
    ):
      """

      Args:
        t (jax.Array): [window_size, ...] timesteps leading up to simulation
        a (jax.Array): [window_size] actions leading up to simulation
        h_on (jax.Array): [window_size, D] rnn-states leading up to simulation
        h_tar (jax.Array): [window_size, D] rnn-states leading up to simulation
        l_mask (jax.Array): [window_size] loss mask
        any_achievable (jax.Array): [] boolean indicating if any achievable goal
        g (jax.Array): [num_offtask_goals, ...] off-task goals
        key (jax.random.PRNGKey): [2], random key for simulation
      """

      # if self.make_init_goal_timestep is not None:
      #  t = jax.vmap(self.make_init_goal_timestep)(t, g)

      # use same goal for all simulations
      # [G] --> [num_sims, G]
      num_simulations = self.num_offtask_simulations
      g = repeat(g, num_simulations)

      #########################################
      # get simulations starting from final timestep in window
      #########################################
      key, key_ = jax.random.split(key)
      # [sim_length, num_sim, ...]
      last = lambda y: jax.tree.map(lambda x: x[-1], y)
      x_t = last(t)

      # reset achievements at the beginning of the simulation
      # x_t = reset_achievements(x_t)

      next_t, sim_outputs_t = simulate(
        h_tm1=last(h_on),  # [D]
        x_t=x_t,  # [D, ...]
        rng=key_,
        goal=g,  # [num_sims, G]
        use_offtask_policy=jnp.ones((1,)),
        num_simulations=num_simulations,
        policy_fn=self.offtask_dyna_policy,
      )

      #########################################
      # run RNN + Off-task Q-network over simulation data
      #########################################
      if self.backtracking:
        # we replace last, because last action from data
        # is different than action from simulation
        # [window_size + sim_length, num_sims, ...]
        all_but_last = lambda y: jax.tree.map(lambda x: x[:-1], y)
        all_t = concat_start_sims(all_but_last(t), next_t)
        all_a = concat_start_sims(all_but_last(a), sim_outputs_t.actions)
        # start at beginning of experience data
        start_index = 0
      else:
        all_t = next_t
        all_a = sim_outputs_t.actions
        # start at last timestep before simulation
        start_index = -1
      # NOTE: we're recomputing RNN but easier to read this way...
      # TODO: reuse RNN online param computations for speed (probably not worth it)
      key, key_ = jax.random.split(key)
      h_tm1_online_repeated = jax.tree.map(lambda x: x[start_index], h_on)
      h_tm1_online_repeated = repeat(h_tm1_online_repeated, num_simulations)
      online_preds = apply_rnn_and_q(
        h_tm1=h_tm1_online_repeated,  # [num_sims, D]
        timesteps=all_t,  # [T', num_sims, ...]
        task=g,  # [num_sims, G]
        rng=key_,
        network=self.network,
        params=params,
        q_fn=self.network.subtask_q_fn,
      )

      key, key_ = jax.random.split(key)
      h_tm1_target_repeated = jax.tree.map(lambda x: x[start_index], h_tar)
      h_tm1_target_repeated = repeat(h_tm1_target_repeated, num_simulations)
      target_preds = apply_rnn_and_q(
        h_tm1=h_tm1_target_repeated,
        timesteps=all_t,
        task=g,
        rng=key_,
        network=self.network,
        params=target_params,
        q_fn=self.network.subtask_q_fn,
      )
      #########################################
      # Get Off-task loss
      #########################################
      all_t_mask = simulation_finished_mask(l_mask, next_t)
      if not self.backtracking:
        all_t_mask = all_t_mask[-self.simulation_length - 1 :]
      ## [T, K, D]
      # achievements = all_t.observation.achievements.astype(jnp.float32)
      ## [T, K, D]
      # achievement_coefficients = all_t.observation.task_w.astype(jnp.float32)
      ## [T, K, D]
      # achievement_coefficients = achievement_coefficients[..., : g.shape[-1]]

      # coeff_mask = g.astype(jnp.float32)
      ## coeff_time_mask = jnp.conj(
      ##   jnp.zeros_like(achievements)
      ## )
      ## [T, K, D] * [T, K, D] * [1, K, D] --> [T, K]
      # offtask_reward = (achievements * achievement_coefficients * coeff_mask[None]).sum(
      #  -1
      # )
      offtask_reward = self.compute_rewards(all_t, g)
      (
        offtask_batch_td_error,
        offtask_batch_loss_mean,
        offtask_metrics,
        offtask_log_info,
      ) = self.loss_fn(
        timestep=all_t,
        online_preds=online_preds,
        target_preds=target_preds,
        actions=all_a,
        rewards=offtask_reward / MAX_REWARD,
        is_last=make_float(all_t.last()),
        non_terminal=all_t.discount,
        loss_mask=all_t_mask,
      )

      # only use loss if any achievable goal
      any_achievable = any_achievable.astype(jnp.float32)
      offtask_batch_loss_mean = offtask_batch_loss_mean * any_achievable[None]
      offtask_batch_td_error = offtask_batch_td_error * any_achievable[None]

      # Compute normalized entropy of achievable probability mass function
      entropy = -jnp.sum(
        achieve_poss * jnp.log(achieve_poss + 1e-5), axis=-1
      ) / jnp.log(achieve_poss.shape[-1])
      # offtask_metrics['1.achievable'] = achievable
      offtask_metrics["2.achievable_entropy"] = entropy
      offtask_metrics["2.any_achievable"] = any_achievable
      # offtask_metrics["2.num_achievable"] = all_t.observation.achievable.sum(
      #  -1
      # )  # unnormalized

      offtask_log_info["goal"] = g
      offtask_log_info["any_achievable"] = any_achievable
      offtask_log_info["offtask_reward"] = offtask_reward
      #########################################
      # run RNN + main Q-network over simulation data
      #########################################

      # NOTE: we're recomputing RNN but easier to read this way...
      # TODO: reuse RNN online param computations for speed (probably not worth it)
      key, key_ = jax.random.split(key)
      online_preds = apply_rnn_and_q(
        h_tm1=h_tm1_online_repeated,
        timesteps=all_t,
        task=g * 0,  # NOTE: goal will be ignored
        rng=key_,
        network=self.network,
        params=params,
        q_fn=self.network.reg_q_fn,
      )

      key, key_ = jax.random.split(key)
      target_preds = apply_rnn_and_q(
        h_tm1=h_tm1_target_repeated,
        timesteps=all_t,
        task=g * 0,  # NOTE: goal will be ignored
        rng=key_,
        network=self.network,
        params=target_params,
        q_fn=self.network.reg_q_fn,
      )

      #########################################
      # Get Main task loss
      #########################################
      main_batch_td_error, main_batch_loss_mean, main_metrics, main_log_info = (
        self.loss_fn(
          timestep=all_t,
          online_preds=online_preds,
          target_preds=target_preds,
          actions=all_a,
          rewards=all_t.reward / MAX_REWARD,
          is_last=make_float(all_t.last()),
          non_terminal=all_t.discount,
          loss_mask=all_t_mask,
        )
      )
      offtask_log_info["main_q_values"] = main_log_info["q_values"]
      offtask_log_info["main_q_target"] = main_log_info["q_target"]

      batch_td_error = self.offtask_coeff * jnp.abs(
        offtask_batch_td_error
      ) + self.main_coeff * jnp.abs(main_batch_td_error)
      batch_loss_mean = (
        self.offtask_coeff * offtask_batch_loss_mean
        + self.main_coeff * main_batch_loss_mean
      )
      metrics = {
        **{f"{k}/offtask-sub": v for k, v in offtask_metrics.items()},
        **{f"{k}/offtask-reg": v for k, v in main_metrics.items()},
      }
      log_info = offtask_log_info
      return batch_td_error, batch_loss_mean, metrics, log_info

    def dyna_loss_fn_(t, a, h_on, h_tar, l_mask, key):
      """

      Args:
        t (jax.Array): [window_size, ...]
        h_on (jax.Array): [window_size, ...]
        h_tar (jax.Array): [window_size, ...]
        key (jax.random.PRNGKey): [2]
      """
      # get simulations starting from final timestep in window
      key, key_ = jax.random.split(key)
      # [sim_length, num_sim, ...]

      main_goal = self.get_main_goal(t)
      next_t, sim_outputs_t = simulate(
        h_tm1=jax.tree.map(lambda x: x[-1], h_on),
        x_t=jax.tree.map(lambda x: x[-1], t),
        rng=key_,
        goal=main_goal,
        use_offtask_policy=jnp.zeros((1,)),
        num_simulations=self.num_dyna_simulations,
        policy_fn=self.dyna_policy,
      )

      if self.backtracking:
        # we replace last, because last action from data
        # is different than action from simulation
        # [window_size + sim_length, num_sims, ...]
        all_but_last = lambda y: jax.tree.map(lambda x: x[:-1], y)
        all_t = concat_start_sims(all_but_last(t), next_t)
        all_a = concat_start_sims(all_but_last(a), sim_outputs_t.actions)
        # start at beginning of experience data
        start_index = 0
      else:
        all_t = next_t
        all_a = sim_outputs_t.actions
        # start at last timestep before simulation
        start_index = -1

      # NOTE: we're recomputing RNN but easier to read this way...
      # TODO: reuse RNN online param computations for speed (probably not worth it)
      key, key_ = jax.random.split(key)
      h_htm1 = jax.tree.map(lambda x: x[start_index], h_on)
      h_htm1 = repeat(h_htm1, self.num_dyna_simulations)
      online_preds = apply_rnn_and_q(
        h_tm1=h_htm1,
        timesteps=all_t,
        task=main_goal,
        rng=key_,
        network=self.network,
        params=params,
        q_fn=self.network.reg_q_fn,
      )

      key, key_ = jax.random.split(key)
      h_htm1 = jax.tree.map(lambda x: x[start_index], h_tar)
      h_htm1 = repeat(h_htm1, self.num_dyna_simulations)
      target_preds = apply_rnn_and_q(
        h_tm1=h_htm1,
        timesteps=all_t,
        task=main_goal,
        rng=key_,
        network=self.network,
        params=target_params,
        q_fn=self.network.reg_q_fn,
      )

      all_t_mask = simulation_finished_mask(l_mask, next_t)
      if not self.backtracking:
        all_t_mask = all_t_mask[-self.simulation_length - 1 :]

      batch_td_error, batch_loss_mean, metrics, log_info = self.loss_fn(
        timestep=all_t,
        online_preds=online_preds,
        target_preds=target_preds,
        actions=all_a,
        rewards=all_t.reward,
        is_last=make_float(all_t.last()),
        non_terminal=all_t.discount,
        loss_mask=all_t_mask,
      )

      return batch_td_error, batch_loss_mean, metrics, log_info

    def preplay_loss_fn_(t, a, h_on, h_tar, l_mask, key):
      """

      Args:
        t (jax.Array): [T', ...]
        h_on (jax.Array): [T', ...]
        h_tar (jax.Array): [T', ...]
        key (jax.random.PRNGKey): [2]
      """
      #########################################
      # First compute off-task loss
      #########################################
      goals, achievable, any_achievable = self.sample_goals(
        t, key, self.num_offtask_goals
      )
      ## sample goals available at last timestep in window
      # achievable = t.observation.achievable.astype(jnp.float32)
      # achievable = jax.tree.map(lambda x: x[-1], achievable)
      # any_achievable = achievable.sum() > 0.1
      # achieve_poss = achievable + 1e-5
      # achieve_poss = achieve_poss / achieve_poss.sum()

      ## sample a different goal for each timestep in window
      ## [num_offtask_goals]
      # key, key_ = jax.random.split(key)
      # goals = distrax.Categorical(probs=achieve_poss).sample(
      #  seed=key_, sample_shape=(self.num_offtask_goals)
      # )

      ## [num_offtask_goals, G]
      # num_classes = t.observation.achievable.shape[-1]
      # goals = jax.nn.one_hot(goals, num_classes=num_classes)

      key, key_ = jax.random.split(key)
      # [num_offtask_goals, T', num_sim], # [num_offtask_goals, num_sim], ...
      (
        offtask_batch_td_error,
        offtask_batch_loss_mean,
        offtask_metrics,
        offtask_log_info,
      ) = jax.vmap(
        offtask_dyna_loss_fn_,
        in_axes=(None, None, None, None, None, None, None, 0, 0),
      )(
        t,
        a,
        h_on,
        h_tar,
        l_mask,
        achievable,
        any_achievable,
        goals,
        jax.random.split(key_, self.num_offtask_goals),
      )

      #########################################
      # Afterward compute regular dyna loss
      #########################################
      key, key_ = jax.random.split(key)
      # [T', num_sim]
      dyna_batch_td_error, dyna_batch_loss_mean, dyna_metrics, _ = dyna_loss_fn_(
        t, a, h_on, h_tar, l_mask, key_
      )

      # merge num_sim and num_offtask_goals dimensions, them sum over those and divide by total number
      # don't reshape and concatenate to avoid unnecessary copies
      denominator = (
        offtask_batch_loss_mean.size  # [num_offtask_goals*num_sim]
        + self.main_coeff * dyna_batch_loss_mean.size  # [num_sim]
      )

      batch_td_error = (
        # sum over [num_offtask_goals, num_sim]
        offtask_batch_td_error.sum(axis=(0, 2))
        +
        # sum over [num_sim]
        self.main_coeff * dyna_batch_td_error.sum(axis=1)
      ) / denominator

      batch_loss_mean = (
        # sum over [num_offtask_goals, num_sim]
        offtask_batch_loss_mean.sum(axis=(0, 1))
        +
        # sum over [num_sim]
        self.main_coeff * dyna_batch_loss_mean.sum(axis=0)
      ) / denominator

      metrics = {
        **offtask_metrics,
        **dyna_metrics,
      }
      log_info = offtask_log_info

      return batch_td_error, batch_loss_mean, metrics, log_info

    # vmap over individual windows
    # TD ERROR: [T', T' + sim_length]
    # Loss: [T']
    loss_fn_ = preplay_loss_fn_ if self.num_offtask_goals > 0 else dyna_loss_fn_
    if self.combine_real_sim:
      batch_td_error, batch_loss_mean, metrics, log_info = jax.vmap(loss_fn_)(
        timesteps,  # [T', W, ...]
        actions,  # [T', W]
        h_online,  # [T', W, D]
        h_target,  # [T', W, D]
        loss_mask,  # [T', W]
        jax.random.split(rng, len(actions)),  # [W, 2]
      )
    else:
      batch_td_error, batch_loss_mean, metrics, log_info = jax.vmap(
        loss_fn_, (1, 1, 1, 1, 1, 0), 0
      )(
        timesteps,  # [T', W, ...]
        actions,  # [T', W]
        h_online,  # [T', W, D]
        h_target,  # [T', W, D]
        loss_mask,  # [T', W]
        jax.random.split(rng, window_size),  # [W, 2]
      )

    # figuring out how to incorporate windows into TD error is annoying so punting
    # TODO: incorporate windowed overlappping TDs into TD error
    batch_td_error = batch_td_error.mean()  # [num_sim]
    batch_loss_mean = batch_loss_mean.mean()  # []

    return batch_td_error, batch_loss_mean, metrics, log_info


def make_loss_fn_class(config, **kwargs) -> DynaLossFn:
  return functools.partial(
    DynaLossFn,
    discount=config["GAMMA"],
    lambda_=config.get("TD_LAMBDA", 0.9),
    online_coeff=config.get("ONLINE_COEFF", 1.0),
    dyna_coeff=config.get("DYNA_COEFF", 1.0),
    num_dyna_simulations=config.get("NUM_DYNA_SIMULATIONS", 2),
    num_offtask_simulations=config.get("NUM_OFFTASK_SIMULATIONS", 2),
    simulation_length=config.get("SIMULATION_LENGTH", 5),
    importance_sampling_exponent=config.get("IMPORTANCE_SAMPLING_EXPONENT", 0.6),
    max_priority_weight=config.get("MAX_PRIORITY_WEIGHT", 0.9),
    step_cost=config.get("STEP_COST", 0.001),
    window_size=config.get("WINDOW_SIZE", 1.0),
    backtracking=config.get("BACKTRACKING", True),
    offtask_coeff=config.get("OFFTASK_COEFF", 1.0),
    main_coeff=config.get("MAIN_COEFF", 1.0),
    num_offtask_goals=config.get("NUM_OFFTASK_GOALS", 5),
    terminate_offtask=config.get("TERMINATE_OFFTASK", False),
    subtask_coeff=config.get("SUBTASK_COEFF", 1.0),
    mainq_coeff=config.get("MAINQ_COEFF", 1.0),
    combine_real_sim=config.get("COMBINE_REAL_SIM", False),
    **kwargs,
  )


def get_in_episode(timestep):
  # get mask for within episode
  non_terminal = timestep.discount
  is_last = timestep.last()
  term_cumsum = jnp.cumsum(is_last, -1)
  in_episode = (term_cumsum + non_terminal) < 2
  return in_episode


from craftax.craftax.constants import Action, BLOCK_PIXEL_SIZE_IMG, Achievement
from craftax.craftax.renderer import render_craftax_pixels
from visualizer import plot_frames


def render_fn(state):
  image = render_craftax_pixels(state, block_pixel_size=BLOCK_PIXEL_SIZE_IMG)
  return image / 255.0


render_fn = jax.jit(render_fn)


def learner_log_extra(
  data: dict,
  config: dict,
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
    main_q_values: np.array,
    main_q_target: np.array,
    goal: np.array,
    any_achievable: np.array,
    offtask_reward: np.array,
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

    ax1.plot(offtask_reward, label="Subtask Reward")
    # ax1.plot(rewards, label='Main task reward')
    ax1.plot(q_values_taken, label="Q-Values")
    ax1.plot(q_target, label="Q-Targets")
    ax1.set_title("Subtask Rewards and Q-Values")
    format(ax1)
    ax1.legend()

    # Plot TD errors in the middle subplot
    # ax2.plot(offtask_reward, label='Subtask Reward')
    main_q_values_taken = rlax.batched_index(main_q_values, actions)
    ax2.plot(rewards, label="Main task reward")
    ax2.plot(main_q_values_taken, label="Main Q-Values")
    ax2.plot(main_q_target, label="Main Q-Targets")
    ax2.set_title("Main Rewards and Q-Values")
    format(ax2)
    ax2.legend()
    # ax2.plot(td_errors)
    # format(ax2)
    # ax2.set_title('TD Errors')

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
    max_len = min(config.get("MAX_EPISODE_LOG_LEN", 60), len(rewards))
    for idx in range(max_len):
      index = lambda y: jax.tree.map(lambda x: x[idx], y)
      obs_image = render_fn(index(timesteps.state.env_state))
      obs_images.append(obs_image)

    # ------------
    # plot
    # ------------
    actions_taken = [Action(a).name for a in actions]

    def index(t, idx):
      return jax.tree.map(lambda x: x[idx], t)

    def panel_title_fn(timesteps, i):
      if hasattr(timesteps.observation, "achievements"):
        goal_idx = goal.argmax(-1)
        achievement = Achievement(goal_idx).name
        title = f"{achievement}. t={i}"
        title += f"\nA={actions_taken[i]}"
        if i >= len(timesteps.reward) - 1:
          return title
        title += (
          f"\nr={timesteps.reward[i + 1]:.2f}, $\\gamma={timesteps.discount[i + 1]}$"
        )
        title += f"\nr_off={offtask_reward[i + 1]:.2f}"
        achieved = timesteps.observation.achievements[i + 1]
        if achieved.sum() > 1e-5:
          achievement_idx = achieved.argmax()
          try:
            achievement = Achievement(achievement_idx).name
            title += f"\n{achievement}"
          except ValueError:
            title += f"\nHealth?"
        achievable_list = craftax_env.print_possible_achievements(
          timesteps.observation.achievable[i], return_list=True
        )

        if achievable_list:
          title += "\nAchievable:"
          for name in achievable_list:
            title += f"\n- {name}"
      elif hasattr(timesteps.state.env_state, "current_goal"):
        start_location = timesteps.state.env_state.start_position[i]
        idx_to_achievement = {v: k for k, v in Achiement_to_idx.items()}
        import ipdb

        ipdb.set_trace()
        env_goal = timesteps.state.env_state.current_goal[i]
        env_goal_name = Achievement(int(env_goal)).name
        ngoals = len(Achiement_to_idx)
        obs_goal = timesteps.observation.task_w[i]
        obs_goal = obs_goal[:ngoals]
        goal = goal[:ngoals]

        obs_goal_idx = int(obs_goal.argmax(-1))
        obs_goal_name = Achievement(idx_to_achievement[obs_goal_idx]).name

        sim_goal_idx = int(goal.argmax(-1))
        sim_goal_name = Achievement(idx_to_achievement[sim_goal_idx]).name

        title = f"t={i}"
        title += f"\nA={actions_taken[i]}"
        if i >= len(timesteps.reward) - 1:
          return title
        title += (
          f"\nr={timesteps.reward[i + 1]:.2f}, $\\gamma={timesteps.discount[i + 1]}$"
        )
        title += f"\nstart={start_location}\ng={env_goal_name}\ng'={sim_goal_name}"

        if obs_goal_name != env_goal_name:
          title += f"\no={obs_goal_name}!= e={env_goal_name}"

      return title

    fig = plot_frames(
      timesteps=timesteps,
      frames=obs_images,
      panel_title_fn=panel_title_fn,
      row_height=2.5,
      ncols=6,
    )
    if wandb.run is not None:
      wandb.log({f"learner_example/{key}/trajectory": wandb.Image(fig)})
    plt.close(fig)

  def callback(d, g, any_achievable):
    log_data(**d, goal=g, any_achievable=any_achievable, key="dyna")

  # this will be the value after update is applied
  n_updates = data["n_updates"] + 1
  is_log_time = n_updates % config["LEARNER_EXTRA_LOG_PERIOD"] == 0

  if "dyna" in data:
    # [Batch, Env Time, Num Goals, Num Simuations]
    goal = data["dyna"].pop("goal")
    goal = goal[0, 0, 0, 0]

    # [Batch, Env Time, Num Goals]
    any_achievable = data["dyna"].pop("any_achievable")
    any_achievable = any_achievable[0, 0, 0]

    # [Batch, Env Time, Num Goals, T+Sim, Num Simuations]
    dyna_data = jax.tree.map(lambda x: x[0, 0, 0, :, 0], data["dyna"])

    jax.lax.cond(
      is_log_time,
      lambda d: jax.debug.callback(callback, d, goal, any_achievable),
      lambda d: None,
      dyna_data,
    )


class DuellingMLP(nn.Module):
  hidden_dim: int
  out_dim: int = 0
  num_layers: int = 1
  norm_type: str = "none"
  activation: str = "relu"
  activate_final: bool = True
  use_bias: bool = False

  @nn.compact
  def __call__(self, x, train: bool = False):
    value_mlp = MLP(
      hidden_dim=self.hidden_dim,
      num_layers=self.num_layers,
      norm_type=self.norm_type,
      activation=self.activation,
      activate_final=False,
      use_bias=self.use_bias,
      out_dim=1,
    )
    advantage_mlp = MLP(
      hidden_dim=self.hidden_dim,
      num_layers=self.num_layers,
      norm_type=self.norm_type,
      activation=self.activation,
      activate_final=False,
      use_bias=self.use_bias,
      out_dim=self.out_dim,
    )
    assert self.out_dim > 0, "must have at least one action"

    value = value_mlp(x, train)  # [B, 1]
    advantages = advantage_mlp(x, train)  # [B, A]

    # Advantages have zero mean.
    advantages -= jnp.mean(advantages, axis=-1, keepdims=True)  # [B, A]

    q_values = value + advantages  # [B, A]

    return q_values


class DuellingDotMLP(nn.Module):
  hidden_dim: int
  out_dim: int = 0
  num_layers: int = 1
  norm_type: str = "none"
  activation: str = "relu"
  activate_final: bool = True
  use_bias: bool = False

  @nn.compact
  def __call__(self, x, task, train: bool = False):
    task_dim = task.shape[-1]
    value_mlp = MLP(
      hidden_dim=self.hidden_dim,
      num_layers=self.num_layers,
      norm_type=self.norm_type,
      activation=self.activation,
      activate_final=False,
      use_bias=self.use_bias,
      out_dim=task_dim,
    )
    advantage_mlp = MLP(
      hidden_dim=self.hidden_dim,
      num_layers=self.num_layers,
      norm_type=self.norm_type,
      activation=self.activation,
      activate_final=False,
      use_bias=self.use_bias,
      out_dim=self.out_dim * task_dim,
    )
    assert self.out_dim > 0, "must have at least one action"

    # Compute value & advantage
    value = value_mlp(x, train)  # [B, C]
    advantages = advantage_mlp(x, train)  # [B, A*C]

    # Reshape value and advantages
    value = jnp.expand_dims(value, axis=-2)  # [B, 1, C]
    # Reshape advantages to [B, (T), A, C] where T dimension is optional
    advantages_shape = list(advantages.shape[:-1]) + [self.out_dim, task_dim]
    advantages = jnp.reshape(advantages, advantages_shape)  # [B, A, C]

    # Advantages have zero mean across actions
    advantages -= jnp.mean(advantages, axis=1, keepdims=True)  # [B, A, C]

    # Combine value and advantages
    sf = value + advantages  # [B, A, C]

    # Dot product with task vector to get Q-values
    q_values = jnp.sum(sf * jnp.expand_dims(task, axis=-2), axis=-1)  # [B, A]

    return q_values


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
  q_head_type: str

  def setup(self):
    kernel_init = nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)
    self.task_fn = nn.Dense(128, kernel_init=kernel_init)

  def initialize(self, x: TimeStep):
    """Only used for initialization."""
    # [B, D]
    rng = jax.random.PRNGKey(0)
    batch_dims = (x.reward.shape[0],)
    rnn_state = self.initialize_carry(rng, batch_dims)
    _, rnn_state = self.__call__(rnn_state, x, rng)
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
    task = jnp.zeros_like(x.observation.achievements)
    q_vals = self.reg_q_fn(rnn_out, task)
    predictions = Predictions(q_vals=q_vals, state=new_rnn_state)

    return predictions, new_rnn_state

  def reg_q_fn(self, rnn_out, task):
    # just so both have same signature
    if self.q_head_type == "dot":
      return self.q_fn(rnn_out, task)
    else:
      del task
      return self.q_fn(rnn_out)

  def subtask_q_fn(self, rnn_out, task):
    task = self.task_fn(task)
    inp = jnp.concatenate((rnn_out, task), axis=-1)
    if self.q_head_type == "dot":
      return self.q_fn_subtask(inp, task)
    else:
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

    task = jnp.zeros_like(xs.observation.achievements)
    rnn_out = self.rnn.output_from_state(new_rnn_states)
    q_vals = nn.BatchApply(self.reg_q_fn)(rnn_out, task)
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


class DynaAgentEnvModelSharedHead(nn.Module):
  """

  Note: predictions contains rnn_state because when you use unroll, you only get the final rnn_state but predictions for all time-steps.

  """

  observation_encoder: nn.Module
  rnn: vbb.ScannedRNN
  q_fn: nn.Module
  q_fn_subtask: nn.Module  # ignored
  env: environment.Environment
  env_params: environment.EnvParams
  q_head_type: str

  def setup(self):
    kernel_init = nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)
    self.task_fn = nn.Dense(128, kernel_init=kernel_init)

  def initialize(self, x: TimeStep):
    """Only used for initialization."""
    # [B, D]
    rng = jax.random.PRNGKey(0)
    batch_dims = (x.reward.shape[0],)
    rnn_state = self.initialize_carry(rng, batch_dims)
    _, rnn_state = self.__call__(rnn_state, x, rng)
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
    task = jnp.zeros_like(x.observation.achievements)
    q_vals = self.reg_q_fn(rnn_out, task)
    predictions = Predictions(q_vals=q_vals, state=new_rnn_state)

    return predictions, new_rnn_state

  def reg_q_fn(self, rnn_out, task):
    task = self.task_fn(task)
    inp = jnp.concatenate((rnn_out, task), axis=-1)
    if self.q_head_type == "dot":
      # CHECKED
      return self.q_fn(inp, task)
    else:
      return self.q_fn(inp)

  def subtask_q_fn(self, rnn_out, task):
    return self.reg_q_fn(rnn_out, task)

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

    task = jnp.zeros_like(xs.observation.achievements)
    rnn_out = self.rnn.output_from_state(new_rnn_states)
    q_vals = nn.BatchApply(self.reg_q_fn)(rnn_out, task)
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


############################################################
# Craftax Single Goal
############################################################
def make_craftax_singlegoal_agent(
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
  if config.get("SHARE_HEADS", True):
    AgentCls = DynaAgentEnvModelSharedHead
  else:
    AgentCls = DynaAgentEnvModel

  if config.get("QHEAD_TYPE", "dot") == "dot":
    QFnCls = DuellingDotMLP
  elif config.get("QHEAD_TYPE", "dot") == "duelling":
    QFnCls = DuellingMLP
  else:
    QFnCls = MLP

  agent = AgentCls(
    observation_encoder=CraftaxObsEncoder(
      hidden_dim=config["MLP_HIDDEN_DIM"],
      num_layers=config["NUM_MLP_LAYERS"],
      activation=config["ACTIVATION"],
      norm_type=config.get("NORM_TYPE", "none"),
      structured_inputs=config.get("STRUCTURED_INPUTS", False),
      use_bias=config.get("USE_BIAS", True),
      include_achievable=config.get("INCLUDE_ACHIEVABLE", True),
      action_dim=env.action_space(env_params).n,
    ),
    rnn=rnn,
    q_fn=QFnCls(
      hidden_dim=config.get("Q_HIDDEN_DIM", 512),
      num_layers=config.get("NUM_Q_LAYERS", 1),
      out_dim=env.action_space(env_params).n,
      activation=config["ACTIVATION"],
      activate_final=False,
      use_bias=config.get("USE_BIAS", True),
    ),
    q_fn_subtask=QFnCls(
      hidden_dim=config.get("Q_HIDDEN_DIM", 512),
      num_layers=config.get("NUM_AUX_LAYERS", 0),
      out_dim=env.action_space(env_params).n,
      activation=config["ACTIVATION"],
      activate_final=False,
      use_bias=config.get("USE_BIAS", True),
    ),
    env=model_env,
    env_params=model_env_params,
    q_head_type=config.get("QHEAD_TYPE", "dot"),
  )

  rng, _rng = jax.random.split(rng)
  network_params = agent.init(_rng, example_timestep, method=agent.initialize)

  def reset_fn(params, example_timestep, reset_rng):
    batch_dims = example_timestep.reward.shape
    return agent.apply(
      params, batch_dims=batch_dims, rng=reset_rng, method=agent.initialize_carry
    )

  return agent, network_params, reset_fn


def make_train_craftax_singlegoal(**kwargs):
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

  num_offtask_simulations = config["NUM_OFFTASK_SIMULATIONS"]

  assert config["NUM_DYNA_SIMULATIONS"] == 1

  def dyna_policy(preds: Predictions, sim_rng: jax.Array):
    del sim_rng
    return jnp.argmax(preds.q_vals, axis=-1)

  def offtask_dyna_policy(preds: Predictions, sim_rng: jax.Array):
    q_values = preds.q_vals
    sim_rng, sim_rng_ = jax.random.split(sim_rng)
    epsilons = jax.random.choice(sim_rng_, vals, shape=(num_offtask_simulations - 1,))
    epsilons = jnp.concatenate((jnp.array([0.0]), epsilons))
    eps = epsilons[: q_values.shape[0]]
    assert q_values.shape[0] == eps.shape[0]
    sim_rng = jax.random.split(sim_rng, q_values.shape[0])
    return jax.vmap(base_agent.epsilon_greedy_act, in_axes=(0, 0, 0))(
      q_values, eps, sim_rng
    )

  def sample_goals(timestep, key, num_offtask_goals):
    # sample goals available at last timestep in window
    # [T, G]
    achievable = timestep.observation.achievable.astype(jnp.float32)

    # [G]
    achievable = jax.tree.map(lambda x: x[-1], achievable)
    any_achievable = achievable.sum() > 0.1
    achieve_poss = achievable + 1e-5
    achieve_poss = achieve_poss / achieve_poss.sum()

    # sample a different goal for each timestep in window
    # [num_offtask_goals]
    key, key_ = jax.random.split(key)
    goals = distrax.Categorical(probs=achieve_poss).sample(
      seed=key_, sample_shape=(num_offtask_goals)
    )

    # [num_offtask_goals, G]
    num_classes = timestep.observation.achievable.shape[-1]
    goals = jax.nn.one_hot(goals, num_classes=num_classes)
    return goals, achievable, any_achievable

  def compute_rewards(timesteps, goal_onehot):
    # [T, K, D]
    achievements = timesteps.observation.achievements.astype(jnp.float32)
    # [T, K, D]
    achievement_coefficients = timesteps.observation.task_w.astype(jnp.float32)
    # [T, K, D]
    achievement_coefficients = achievement_coefficients[..., : goal_onehot.shape[-1]]

    coeff_mask = goal_onehot.astype(jnp.float32)
    # coeff_time_mask = jnp.conj(
    #   jnp.zeros_like(achievements)
    # )
    # [T, K, D] * [T, K, D] * [1, K, D] --> [T, K]
    offtask_reward = (achievements * achievement_coefficients * coeff_mask[None]).sum(
      -1
    )
    return offtask_reward

  num_dyna_simulations = config.get("NUM_DYNA_SIMULATIONS", 2)

  def get_main_goal(timestep):
    return jnp.zeros(
      (num_dyna_simulations, timestep.observation.achievements[0].shape[-1])
    )

  return vbb.make_train(
    make_agent=partial(
      make_craftax_singlegoal_agent, model_env=kwargs.pop("model_env")
    ),
    make_loss_fn_class=functools.partial(
      make_loss_fn_class,
      dyna_policy=dyna_policy,
      offtask_dyna_policy=offtask_dyna_policy,
      sample_goals=sample_goals,
      compute_rewards=compute_rewards,
      get_main_goal=get_main_goal,
    ),
    make_optimizer=base_agent.make_optimizer,
    make_actor=make_actor,
    **kwargs,
  )


############################################################
# Craftax Multigoal
############################################################
class DynaAgentEnvModelMultigoal(nn.Module):
  """

  Note: predictions contains rnn_state because when you use unroll, you only get the final rnn_state but predictions for all time-steps.

  """

  observation_encoder: nn.Module
  rnn: vbb.ScannedRNN
  q_fn: nn.Module
  q_fn_subtask: nn.Module
  env: environment.Environment
  env_params: environment.EnvParams
  share_q_fn: bool = True

  def setup(self):
    kernel_init = nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)
    self.task_fn = nn.Dense(128, kernel_init=kernel_init)

  def initialize(self, x: TimeStep):
    """Only used for initialization."""
    # [B, D]
    rng = jax.random.PRNGKey(0)
    batch_dims = (x.reward.shape[0],)
    rnn_state = self.initialize_carry(rng, batch_dims)
    _, rnn_state = self.__call__(rnn_state, x, rng)
    # dummy_action = jnp.zeros(batch_dims, dtype=jnp.int32)
    # self.apply_model(predictions.state, dummy_action, rng)
    dummy_task = jnp.zeros_like(x.observation.state_features)
    self.subtask_q_fn(rnn_state[1], dummy_task)

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

    dummy_task = jnp.zeros_like(x.observation.state_features)
    q_vals = self.reg_q_fn(rnn_out, dummy_task)
    predictions = Predictions(q_vals=q_vals, state=new_rnn_state)

    return predictions, new_rnn_state

  def reg_q_fn(self, rnn_out, task):
    task = self.task_fn(task)
    inp = jnp.concatenate((rnn_out, task), axis=-1)
    return self.q_fn(inp)

  def subtask_q_fn(self, rnn_out, task):
    if self.share_q_fn:
      return self.reg_q_fn(rnn_out, task)
    else:
      task = self.task_fn(task)
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

    dummy_task = jnp.zeros_like(xs.observation.state_features)
    rnn_out = self.rnn.output_from_state(new_rnn_states)
    q_vals = nn.BatchApply(self.reg_q_fn)(rnn_out, dummy_task)
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


def make_craftax_multigoal_agent(
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

  if config.get("QHEAD_TYPE", "dot") == "dot":
    QFnCls = DuellingDotMLP
  elif config.get("QHEAD_TYPE", "dot") == "duelling":
    QFnCls = DuellingMLP
  else:
    QFnCls = MLP

  agent = DynaAgentEnvModelMultigoal(
    observation_encoder=CraftaxMultiGoalObsEncoder(
      hidden_dim=config["MLP_HIDDEN_DIM"],
      num_layers=config["NUM_MLP_LAYERS"],
      activation=config["ACTIVATION"],
      norm_type=config.get("NORM_TYPE", "none"),
      use_bias=config.get("USE_BIAS", True),
      include_goal=config.get("OBS_INCLUDE_GOAL", False),
      action_dim=env.action_space(env_params).n,
    ),
    rnn=rnn,
    q_fn=QFnCls(
      hidden_dim=config.get("Q_HIDDEN_DIM", 512),
      num_layers=config.get("NUM_PRED_LAYERS", 1),
      out_dim=env.action_space(env_params).n,
      activation=config["ACTIVATION"],
      activate_final=False,
      use_bias=config.get("USE_BIAS", True),
    ),
    q_fn_subtask=QFnCls(
      hidden_dim=config.get("Q_HIDDEN_DIM", 512),
      num_layers=config.get("NUM_PRED_LAYERS", 0),
      out_dim=env.action_space(env_params).n,
      activation=config["ACTIVATION"],
      activate_final=False,
      use_bias=config.get("USE_BIAS", True),
    ),
    env=model_env,
    env_params=model_env_params,
    share_q_fn=config.get("SHARE_Q_FN", True),
    # q_head_type=config.get("QHEAD_TYPE", "dot"),
  )

  rng, _rng = jax.random.split(rng)
  network_params = agent.init(_rng, example_timestep, method=agent.initialize)

  def reset_fn(params, example_timestep, reset_rng):
    batch_dims = example_timestep.reward.shape
    return agent.apply(
      params, batch_dims=batch_dims, rng=reset_rng, method=agent.initialize_carry
    )

  return agent, network_params, reset_fn


def make_train_craftax_multigoal(**kwargs):
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

  num_offtask_simulations = config["NUM_OFFTASK_SIMULATIONS"]

  assert config["NUM_DYNA_SIMULATIONS"] == 1

  def dyna_policy(preds: Predictions, sim_rng: jax.Array):
    del sim_rng
    return jnp.argmax(preds.q_vals, axis=-1)

  def offtask_dyna_policy(preds: Predictions, sim_rng: jax.Array):
    q_values = preds.q_vals
    sim_rng, sim_rng_ = jax.random.split(sim_rng)
    epsilons = jax.random.choice(sim_rng_, vals, shape=(num_offtask_simulations - 1,))
    epsilons = jnp.concatenate((jnp.array([0.0]), epsilons))
    eps = epsilons[: q_values.shape[0]]
    assert q_values.shape[0] == eps.shape[0]
    sim_rng = jax.random.split(sim_rng, q_values.shape[0])
    return jax.vmap(base_agent.epsilon_greedy_act, in_axes=(0, 0, 0))(
      q_values, eps, sim_rng
    )

  def sample_goals(timestep, key, num_offtask_goals):
    # [T, G] --> [G]
    current_goal = timestep.observation.task_w[0]
    other_goals = 1.0 - current_goal
    other_goal_probs = other_goals / other_goals.sum()

    achievable = jnp.ones_like(other_goal_probs)
    any_achievable = achievable.sum() > 0.1

    key, key_ = jax.random.split(key)
    goals = distrax.Categorical(probs=other_goal_probs).sample(
      seed=key_, sample_shape=(num_offtask_goals)
    )

    # [num_offtask_goals, G]
    num_goals = timestep.observation.task_w.shape[-1]
    goals = jax.nn.one_hot(goals, num_classes=num_goals)
    return goals, achievable, any_achievable

  def compute_rewards(timesteps, goal_onehot):
    # [T, K, D]
    state_features = timesteps.observation.state_features.astype(jnp.float32)
    # [T, K, D] * [1, K, D] --> [T, K]
    offtask_reward = (state_features * goal_onehot.astype(jnp.float32)[None]).sum(-1)
    return offtask_reward

  num_dyna_simulations = config.get("NUM_DYNA_SIMULATIONS", 2)

  def get_main_goal(timestep):
    goal = timestep.observation.task_w[:1]
    return jnp.tile(goal, (num_dyna_simulations, 1))

  def make_init_goal_timestep(timestep, goal):
    """
    goal is 1-hot vector
    """
    achievement_goal = jax.lax.dynamic_index_in_dim(
      craftax_web_env.IDX_to_Achievement, goal.argmax(-1), keepdims=False
    )
    new_state = timestep.state.replace(
      env_state=timestep.state.env_state.replace(
        current_goal=achievement_goal,
      ),
    )
    new_observation = timestep.observation.replace(
      task_w=goal,
    )
    return timestep.replace(state=new_state, observation=new_observation)

  return vbb.make_train(
    make_agent=partial(make_craftax_multigoal_agent, model_env=kwargs.pop("model_env")),
    make_loss_fn_class=functools.partial(
      make_loss_fn_class,
      dyna_policy=dyna_policy,
      offtask_dyna_policy=offtask_dyna_policy,
      sample_goals=sample_goals,
      compute_rewards=compute_rewards,
      get_main_goal=get_main_goal,
      make_init_goal_timestep=make_init_goal_timestep,
    ),
    make_optimizer=base_agent.make_optimizer,
    make_actor=make_actor,
    **kwargs,
  )


############################################################
# JaxMaze Multigoal
############################################################
class DynaAgentEnvModelMultigoalJaxMaze(nn.Module):
  """

  Note: predictions contains rnn_state because when you use unroll, you only get the final rnn_state but predictions for all time-steps.

  """

  observation_encoder: nn.Module
  rnn: vbb.ScannedRNN
  q_fn: nn.Module
  q_fn_subtask: nn.Module
  env: environment.Environment
  env_params: environment.EnvParams
  share_q_fn: bool = True

  def setup(self):
    kernel_init = nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)
    self.task_fn = nn.Dense(128, kernel_init=kernel_init)

  def initialize(self, x: TimeStep):
    """Only used for initialization."""
    # [B, D]
    rng = jax.random.PRNGKey(0)
    batch_dims = (x.reward.shape[0],)
    rnn_state = self.initialize_carry(rng, batch_dims)
    _, rnn_state = self.__call__(rnn_state, x, rng)
    # dummy_action = jnp.zeros(batch_dims, dtype=jnp.int32)
    # self.apply_model(predictions.state, dummy_action, rng)
    dummy_task = jnp.zeros_like(x.state.task_state.features)
    self.subtask_q_fn(rnn_state[1], dummy_task)

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

    dummy_task = jnp.zeros_like(x.state.task_state.features)
    q_vals = self.reg_q_fn(rnn_out, dummy_task)
    predictions = Predictions(q_vals=q_vals, state=new_rnn_state)

    return predictions, new_rnn_state

  def reg_q_fn(self, rnn_out, task):
    task = self.task_fn(task)
    inp = jnp.concatenate((rnn_out, task), axis=-1)
    return self.q_fn(inp)

  def subtask_q_fn(self, rnn_out, task):
    if self.share_q_fn:
      return self.reg_q_fn(rnn_out, task)
    else:
      task = self.task_fn(task)
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

    dummy_task = jnp.zeros_like(xs.state.task_state.features)
    rnn_out = self.rnn.output_from_state(new_rnn_states)
    q_vals = nn.BatchApply(self.reg_q_fn)(rnn_out, dummy_task)
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


def make_jaxmaze_multigoal_agent(
  config: dict,
  env: environment.Environment,
  env_params: environment.EnvParams,
  example_timestep: TimeStep,
  rng: jax.random.PRNGKey,
  model_env: Optional[environment.Environment] = None,
  model_env_params: Optional[environment.EnvParams] = None,
  ObsEncoderCls: nn.Module = CategoricalHouzemazeObsEncoder,
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

  if config.get("QHEAD_TYPE", "dot") == "dot":
    QFnCls = DuellingDotMLP
  elif config.get("QHEAD_TYPE", "dot") == "duelling":
    QFnCls = DuellingMLP
  else:
    QFnCls = MLP

  agent = DynaAgentEnvModelMultigoalJaxMaze(
    observation_encoder=ObsEncoderCls(),
    rnn=rnn,
    q_fn=QFnCls(
      hidden_dim=config.get("Q_HIDDEN_DIM", 512),
      num_layers=config.get("NUM_PRED_LAYERS", 1),
      out_dim=env.num_actions(env_params),
      activation=config["ACTIVATION"],
      activate_final=False,
      use_bias=config.get("USE_BIAS", False),
    ),
    q_fn_subtask=QFnCls(
      hidden_dim=config.get("Q_HIDDEN_DIM", 512),
      num_layers=config.get("NUM_PRED_LAYERS", 0),
      out_dim=env.num_actions(env_params),
      activation=config["ACTIVATION"],
      activate_final=False,
      use_bias=config.get("USE_BIAS", False),
    ),
    env=model_env,
    env_params=model_env_params,
    share_q_fn=config.get("SHARE_Q_FN", True),
    # q_head_type=config.get("QHEAD_TYPE", "dot"),
  )

  rng, _rng = jax.random.split(rng)
  network_params = agent.init(_rng, example_timestep, method=agent.initialize)

  def reset_fn(params, example_timestep, reset_rng):
    batch_dims = example_timestep.reward.shape
    return agent.apply(
      params, batch_dims=batch_dims, rng=reset_rng, method=agent.initialize_carry
    )

  return agent, network_params, reset_fn


def make_train_jaxmaze_multigoal(**kwargs):
  config = kwargs["config"]
  rng = jax.random.PRNGKey(config["SEED"])

  epsilon_setting = config["SIM_EPSILON_SETTING"]
  all_tasks = jnp.array(kwargs.pop("all_tasks"))  # [G, D]
  known_offtask_goal = config.get("KNOWN_OFFTASK_GOAL", True)
  num_offtask_goals = 1 if known_offtask_goal else all_tasks.shape[0]
  config["NUM_OFFTASK_GOALS"] = num_offtask_goals

  num_dyna_simulations = 1
  config["NUM_DYNA_SIMULATIONS"] = num_dyna_simulations

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

  num_offtask_simulations = config["NUM_OFFTASK_SIMULATIONS"]

  assert config["NUM_DYNA_SIMULATIONS"] == 1

  def dyna_policy(preds: Predictions, sim_rng: jax.Array):
    del sim_rng
    return jnp.argmax(preds.q_vals, axis=-1)

  def offtask_dyna_policy(preds: Predictions, sim_rng: jax.Array):
    q_values = preds.q_vals
    sim_rng, sim_rng_ = jax.random.split(sim_rng)
    epsilons = jax.random.choice(sim_rng_, vals, shape=(num_offtask_simulations - 1,))
    epsilons = jnp.concatenate((jnp.array([0.0]), epsilons))
    eps = epsilons[: q_values.shape[0]]
    assert q_values.shape[0] == eps.shape[0]
    sim_rng = jax.random.split(sim_rng, q_values.shape[0])
    return jax.vmap(base_agent.epsilon_greedy_act, in_axes=(0, 0, 0))(
      q_values, eps, sim_rng
    )

  def sample_goals(timestep, key, num_offtask_goals):
    # [T, G] --> [G]
    # goal_probs = jnp.ones(len(all_tasks)) / len(all_tasks)
    achievable = jnp.ones(len(all_tasks))
    any_achievable = achievable.sum() > 0.1

    # assume you know offtask goal
    if known_offtask_goal:
      # first time-step, 1 goal
      goals = timestep.state.offtask_w[0][None]
    else:
      goals = all_tasks

    # key, key_ = jax.random.split(key)

    ## [N]
    # goals = distrax.Categorical(probs=goal_probs).sample(seed=key_, sample_shape=(num_offtask_goals))

    ## output should be [N, D]
    # index = lambda x, i: jax.lax.dynamic_index_in_dim(x, i, keepdims=False)
    # index = jax.vmap(index, (None, 0), 0)
    # goals = index(all_tasks, goals)
    return goals, achievable, any_achievable

  def compute_rewards(timesteps, goal_onehot):
    # [T, K, D]
    state_features = timesteps.state.task_state.features.astype(jnp.float32)
    # [T, K, D] * [1, K, D] --> [T, K]
    offtask_reward = (state_features * goal_onehot.astype(jnp.float32)[None]).sum(-1)
    return offtask_reward

  def get_main_goal(timestep):
    goal = timestep.state.task_w[:1]
    return jnp.tile(goal, (num_dyna_simulations, 1))

  env = kwargs["model_env"]
  task_objects = kwargs.pop("task_objects")

  def make_init_goal_timestep(timestep, goal):
    """
    goal is 1-hot vector
    """
    task_object = (task_objects * goal).sum(-1)
    task_object = task_object.astype(jnp.int32)
    new_state = timestep.state.replace(
      step_num=jnp.zeros_like(timestep.state.step_num),
      task_w=goal,
      task_object=task_object,  # only used for logging
    )
    new_observation = env.make_observation(new_state, timestep.observation.prev_action)
    return timestep.replace(
      state=new_state,
      observation=new_observation,
      # reset reward, discount, step type
      reward=jnp.zeros_like(timestep.reward),
      discount=jnp.ones_like(timestep.discount),
      step_type=jnp.ones_like(timestep.step_type),
    )

  return vbb.make_train(
    make_agent=partial(
      make_jaxmaze_multigoal_agent,
      model_env=kwargs.pop("model_env"),
      ObsEncoderCls=kwargs.pop("ObsEncoderCls"),
    ),
    make_loss_fn_class=functools.partial(
      make_loss_fn_class,
      dyna_policy=dyna_policy,
      offtask_dyna_policy=offtask_dyna_policy,
      sample_goals=sample_goals,
      compute_rewards=compute_rewards,
      get_main_goal=get_main_goal,
      make_init_goal_timestep=make_init_goal_timestep,
    ),
    make_optimizer=base_agent.make_optimizer,
    make_actor=make_actor,
    **kwargs,
  )
