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

from jaxneurorl import losses
from jaxneurorl.agents.basics import TimeStep
from jaxneurorl.agents import value_based_basics as vbb
from jaxneurorl.agents import qlearning as base_agent
from networks import MLP, CraftaxObsEncoder, CraftaxMultiGoalObsEncoder

from visualizer import plot_frames

from housemaze import renderer

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
      state=next_lstm_state,
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
  num_simulations: int = 2
  simulation_length: int = 5
  online_coeff: float = 1.0
  dyna_coeff: float = 1.0
  max_priority_weight: float = 0.9
  importance_sampling_exponent: float = 0.6
  backtracking: bool = True
  combine_real_sim: bool = False

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
        self.dyna_loss_fn, params=params, target_params=target_params
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

  def dyna_loss_fn(
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
    window_size = (
      self.window_size
      if isinstance(self.window_size, int)
      else int(self.window_size * len(actions))
    )
    window_size = min(window_size, len(actions))
    window_size = max(window_size, 1)
    roll = partial(rolling_window, size=window_size)
    simulate = partial(
      simulate_n_trajectories,
      network=self.network,
      params=params,
      num_steps=self.simulation_length,
      num_simulations=self.num_simulations,
      policy_fn=self.simulation_policy,
      q_fn=self.network.reg_q_fn,
      goal=None,
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
      next_t, sim_outputs_t = simulate(
        h_tm1=jax.tree.map(lambda x: x[-1], h_on),
        x_t=jax.tree.map(lambda x: x[-1], t),
        rng=key_,
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
      h_htm1 = repeat(h_htm1, self.num_simulations)
      online_preds = apply_rnn_and_q(
        h_tm1=h_htm1,
        timesteps=all_t,
        task=None,
        rng=key_,
        network=self.network,
        params=params,
        q_fn=self.network.reg_q_fn,
      )

      key, key_ = jax.random.split(key)
      h_htm1 = jax.tree.map(lambda x: x[start_index], h_tar)
      h_htm1 = repeat(h_htm1, self.num_simulations)
      target_preds = apply_rnn_and_q(
        h_tm1=h_htm1,
        timesteps=all_t,
        task=None,
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
        rewards=all_t.reward / MAX_REWARD,
        is_last=make_float(all_t.last()),
        non_terminal=all_t.discount,
        loss_mask=all_t_mask,
      )
      return batch_td_error, batch_loss_mean, metrics, log_info

    # vmap over individual windows
    # TD ERROR: [window_size, T + sim_length, num_sim]
    # Loss: [window_size, num_sim, ...]
    if self.combine_real_sim:
      batch_td_error, batch_loss_mean, metrics, log_info = jax.vmap(dyna_loss_fn_)(
        timesteps,  # [T, W, ...]
        actions,  # [T, W]
        h_online,  # [T, W, D]
        h_target,  # [T, W, D]
        loss_mask,  # [T, W]
        jax.random.split(rng, len(actions)),  # [W, 2]
      )
    else:
      batch_td_error, batch_loss_mean, metrics, log_info = jax.vmap(
        dyna_loss_fn_, (1, 1, 1, 1, 1, 0), 0
      )(
        timesteps,  # [T, W, ...]
        actions,  # [T, W]
        h_online,  # [T, W, D]
        h_target,  # [T, W, D]
        loss_mask,  # [T, W]
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
    num_simulations=config.get("NUM_SIMULATIONS", 2),
    simulation_length=config.get("SIMULATION_LENGTH", 5),
    importance_sampling_exponent=config.get("IMPORTANCE_SAMPLING_EXPONENT", 0.6),
    max_priority_weight=config.get("MAX_PRIORITY_WEIGHT", 0.9),
    step_cost=config.get("STEP_COST", 0.0),
    window_size=config.get("WINDOW_SIZE", 20),
    backtracking=config.get("BACKTRACKING", True),
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
      title = f"t={i}"
      title += f"\n{actions_taken[i]}"
      if i >= len(timesteps.reward) - 1:
        return title
      title += (
        f"\nr={timesteps.reward[i + 1]:.2f}, $\\gamma={timesteps.discount[i + 1]}$"
      )

      if hasattr(timesteps.observation, "achievements"):
        achieved = timesteps.observation.achievements[i + 1]
        if achieved.sum() > 1e-5:
          achievement_idx = achieved.argmax()
          try:
            achievement = Achievement(achievement_idx).name
            title += f"\n{achievement}"
          except ValueError:
            title += f"\nHealth?"
      elif hasattr(timesteps.state.env_state, "current_goal"):
        start_location = timesteps.state.env_state.start_position
        goal = timesteps.state.env_state.current_goal[i]
        goal_name = Achievement(int(goal)).name
        title += f"\nstart={start_location}\ngoal={goal}\ngoal={goal_name}"

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
    # [Batch, Env Time, Sim Time, Num Simuations]
    dyna_data = jax.tree.map(lambda x: x[0, 0, :, 0], data["dyna"])

    jax.lax.cond(
      is_log_time,
      lambda d: jax.debug.callback(callback, d),
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


class DynaAgentEnvModel(nn.Module):
  """

  Note: predictions contains rnn_state because when you use unroll, you only get the final rnn_state but predictions for all time-steps.

  """

  observation_encoder: nn.Module
  rnn: vbb.ScannedRNN
  q_fn: nn.Module
  env: environment.Environment
  env_params: environment.EnvParams

  def initialize(self, x: TimeStep):
    """Only used for initialization."""
    # [B, D]
    rng = jax.random.PRNGKey(0)
    batch_dims = (x.reward.shape[0],)
    rnn_state = self.initialize_carry(rng, batch_dims)
    predictions, rnn_state = self.__call__(rnn_state, x, rng)

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
  if config.get("MULTIGOAL", False):
    observation_encoder = CraftaxMultiGoalObsEncoder(
      hidden_dim=config["MLP_HIDDEN_DIM"],
      num_layers=config["NUM_MLP_LAYERS"],
      activation=config["ACTIVATION"],
      norm_type=config.get("NORM_TYPE", "none"),
    )
  else:
    observation_encoder = CraftaxObsEncoder(
      hidden_dim=config["MLP_HIDDEN_DIM"],
      num_layers=config["NUM_MLP_LAYERS"],
      activation=config["ACTIVATION"],
      norm_type=config.get("NORM_TYPE", "none"),
      structured_inputs=config.get("STRUCTURED_INPUTS", False),
      use_bias=config.get("USE_BIAS", True),
      action_dim=env.action_space(env_params).n,
    )
  agent = DynaAgentEnvModel(
    observation_encoder=observation_encoder,
    rnn=rnn,
    q_fn=DuellingMLP(
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


def make_multigoal_agent(
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
    observation_encoder=CraftaxMultiGoalObsEncoder(
      hidden_dim=config["MLP_HIDDEN_DIM"],
      num_layers=config["NUM_MLP_LAYERS"],
      activation=config["ACTIVATION"],
      norm_type=config.get("NORM_TYPE", "none"),
      use_bias=config.get("USE_BIAS", True),
      action_dim=env.action_space(env_params).n,
    ),
    rnn=rnn,
    q_fn=DuellingMLP(
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
  num_simulations = config["NUM_SIMULATIONS"]
  sim_epsilon = config.get("SIM_EPSILON", 0)
  if epsilon_setting == 1:
    # ACME default
    # range of ~(0.001, .1)
    vals = np.logspace(num=256, start=1, stop=3, base=0.1)
  elif epsilon_setting == 2:
    # range of ~(.9,.1)
    vals = np.logspace(num=256, start=0.05, stop=0.9, base=0.1)
  elif epsilon_setting == 3:
    # very random
    vals = np.ones(256) * sim_epsilon

  epsilons = jax.random.choice(rng, vals, shape=(num_simulations - 1,))
  epsilons = jnp.concatenate((jnp.array((0,)), epsilons))
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
