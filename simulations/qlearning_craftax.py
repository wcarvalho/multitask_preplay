"""
Recurrent Q-learning for Craftax.
Add an auxiliary task for predicting achievements. This predicts both which has been achieved at this timestep and which is achievable.
"""

import jax
from typing import Callable, NamedTuple


import flax.linen as nn

import flax
from flax import struct
import jax.numpy as jnp
from gymnax.environments import environment
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use("Agg")


from jaxneurorl.agents.basics import TimeStep
from jaxneurorl.agents import value_based_basics as vbb
from jaxneurorl.agents.qlearning import *
from jaxneurorl import losses
from networks import CraftaxObsEncoder, CraftaxMultiGoalObsEncoder

MAX_REWARD = 1.0

Agent = nn.Module
Params = flax.core.FrozenDict
AgentState = flax.struct.PyTreeNode
RNNInput = vbb.RNNInput


class Predictions(NamedTuple):
  q_vals: jax.Array
  rnn_states: jax.Array
  achievements: jax.Array = None


class RnnAgent(nn.Module):
  """_summary_

  - observation encoder: CNN
  Args:
      nn (_type_): _description_

  Returns:
      _type_: _description_
  """

  observation_encoder: nn.Module
  rnn: vbb.ScannedRNN
  q_fn: nn.Module
  achieve_fn: nn.Module = None

  def initialize(self, x: TimeStep):
    """Only used for initialization."""

    rng = jax.random.PRNGKey(0)
    batch_dims = (x.reward.shape[0],)
    rnn_state = self.initialize_carry(rng, batch_dims)

    return self(rnn_state, x, rng)

  def __call__(self, rnn_state, x: TimeStep, rng: jax.random.PRNGKey):
    embedding = self.observation_encoder(x.observation)

    rnn_in = RNNInput(obs=embedding, reset=x.first())
    rng, _rng = jax.random.split(rng)
    new_rnn_state, rnn_out = self.rnn(rnn_state, rnn_in, _rng)

    q_vals = self.q_fn(rnn_out)
    if self.achieve_fn is not None:
      achieve_vals = self.achieve_fn(rnn_out)
      return Predictions(
        q_vals=q_vals,
        achievements=achieve_vals,
        rnn_states=rnn_out,
      ), new_rnn_state
    else:
      return Predictions(
        q_vals=q_vals,
        rnn_states=rnn_out,
      ), new_rnn_state

  def unroll(self, rnn_state, xs: TimeStep, rng: jax.random.PRNGKey):
    # rnn_state: [B]
    # xs: [T, B]

    embedding = nn.BatchApply(self.observation_encoder)(xs.observation)

    rnn_in = RNNInput(obs=embedding, reset=xs.first())
    rng, _rng = jax.random.split(rng)
    new_rnn_state, rnn_out = self.rnn.unroll(rnn_state, rnn_in, _rng)

    q_vals = nn.BatchApply(self.q_fn)(rnn_out)
    if self.achieve_fn is not None:
      achieve_vals = nn.BatchApply(self.achieve_fn)(rnn_out)
      return Predictions(
        q_vals=q_vals,
        achievements=achieve_vals,
        rnn_states=rnn_out,
      ), new_rnn_state
    else:
      return Predictions(
        q_vals=q_vals,
        rnn_states=rnn_out,
      ), new_rnn_state

  def initialize_carry(self, *args, **kwargs):
    """Initializes the RNN state."""
    return self.rnn.initialize_carry(*args, **kwargs)


@struct.dataclass
class R2D2LossFn(vbb.RecurrentLossFn):
  """Loss function of R2D2.

  https://openreview.net/forum?id=r1lyTjAqYX
  """

  extract_q: Callable[[jax.Array], jax.Array] = lambda preds: preds.q_vals
  aux_coeff: float = 1.0
  max_priority_weight: float = 0.9
  importance_sampling_exponent: float = 0.6

  def error(
    self,
    data,
    online_preds,
    online_state,
    target_preds,
    target_state,
    steps,
    **kwargs,
  ):
    """R2D2 learning."""

    float = lambda x: x.astype(jnp.float32)
    # Get value-selector actions from online Q-values for double Q-learning.
    selector_actions = jnp.argmax(self.extract_q(online_preds), axis=-1)  # [T+1, B]

    # Preprocess discounts & rewards.
    discounts = float(data.discount) * self.discount
    lambda_ = jnp.ones_like(data.discount) * self.lambda_
    rewards = float(data.reward) / MAX_REWARD
    rewards = rewards - self.step_cost
    is_last = float(data.is_last)

    # Get N-step transformed TD error and loss.
    batch_td_error_fn = jax.vmap(
      partial(losses.q_learning_lambda_td, tx_pair=self.tx_pair),
      in_axes=1,
      out_axes=1,
    )

    # [T, B]
    q_t, target_q_t = batch_td_error_fn(
      self.extract_q(online_preds)[:-1],  # [T+1] --> [T]
      data.action[:-1],  # [T+1] --> [T]
      self.extract_q(target_preds)[1:],  # [T+1] --> [T]
      selector_actions[1:],  # [T+1] --> [T]
      rewards[1:],  # [T+1] --> [T]
      discounts[1:],
      is_last[1:],
      lambda_[1:],
    )  # [T+1] --> [T]

    # ensure target = 0 when episode terminates
    target_q_t = target_q_t * data.discount[:-1]
    batch_td_error = target_q_t - q_t

    # ensure loss = 0 when episode truncates
    # truncated if FINAL time-step but data.discount = 1.0, something like [1,1,2,1,1]
    truncated = (data.discount + is_last) > 1  # truncated is discount on AND is last
    loss_mask = (1 - truncated).astype(batch_td_error.dtype)[:-1]
    batch_td_error = batch_td_error * loss_mask

    # [T, B]
    batch_loss = 0.5 * jnp.square(batch_td_error)

    # [B]
    batch_loss_mean = (batch_loss * loss_mask).mean(0)

    metrics = {
      "0.q_loss": batch_loss.mean(),
      "0.q_td": jnp.abs(batch_td_error).mean(),
      "1.reward": rewards[1:].mean(),
      "1.reward_min": rewards[1:].min(),
      "1.reward_max": rewards[1:].max(),
      "z.q_max": self.extract_q(online_preds).max(),
      "z.q_min": self.extract_q(online_preds).min(),
      "z.q_mean": self.extract_q(online_preds).mean(),
      "z.q_var": self.extract_q(online_preds).var(),
    }

    ###########
    # Add auxiliary for achievements and achievable
    ###########
    if self.aux_coeff > 0:
      # [T, B, D]
      achievements = data.timestep.observation.achievements.astype(jnp.float32)
      achievable = data.timestep.observation.achievable.astype(jnp.float32)

      # current prediction is for next timestep
      achieve_predictions = online_preds.achievements[:-1]
      achieve = jnp.concatenate((achievements, achievable), axis=-1)[1:]

      # [T, B, D], sum over achievements
      aux_error = achieve - achieve_predictions
      achieve_loss = 0.5 * jnp.square(aux_error).mean(-1)

      achieve_loss = achieve_loss.mean(axis=0)  # [B]
      metrics["0.achieve_loss"] = achieve_loss
      metrics["0.full_td"] = batch_td_error

      batch_loss_mean = batch_loss_mean + self.aux_coeff * achieve_loss

    if self.logger.learner_log_extra is not None:
      self.logger.learner_log_extra(
        {
          "data": data,
          "td_errors": jnp.abs(batch_td_error),  # [T]
          "mask": loss_mask,  # [T]
          "q_values": self.extract_q(online_preds),  # [T, B]
          "q_loss": batch_loss,  # [ T, B]
          "q_target": target_q_t,
          "n_updates": steps,
        }
      )

    return batch_td_error, batch_loss_mean, metrics  # [T-1, B], [B]#


def make_loss_fn_class(config) -> vbb.RecurrentLossFn:
  return functools.partial(
    R2D2LossFn,
    discount=config["GAMMA"],
    importance_sampling_exponent=config.get("IMPORTANCE_SAMPLING_EXPONENT", 0.6),
    max_priority_weight=config.get("MAX_PRIORITY_WEIGHT", 0.9),
    tx_pair=(
      rlax.SIGNED_HYPERBOLIC_PAIR
      if config.get("TX_PAIR", "none") == "hyperbolic"
      else rlax.IDENTITY_PAIR
    ),
    step_cost=config.get("STEP_COST", 0.0),
    aux_coeff=config.get("AUX_COEFF", 0.001),
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
      use_bias=self.use_bias,
      out_dim=1,
    )
    advantage_mlp = MLP(
      hidden_dim=self.hidden_dim,
      num_layers=self.num_layers,
      use_bias=self.use_bias,
      out_dim=self.out_dim,
    )
    assert self.out_dim > 0, "must have at least one action"

    value = value_mlp(x)  # [B, 1]
    advantages = advantage_mlp(x)  # [B, A]

    # Advantages have zero mean.
    advantages -= jnp.mean(advantages, axis=-1, keepdims=True)  # [B, A]

    q_values = value + advantages  # [B, A]

    return q_values


def make_craftax_agent(
  config: dict,
  env: environment.Environment,
  env_params: environment.EnvParams,
  example_timestep: TimeStep,
  rng: jax.random.PRNGKey,
):
  rnn = vbb.ScannedRNN(hidden_dim=config["AGENT_RNN_DIM"])
  achievable = example_timestep.observation.achievable
  achievements = example_timestep.observation.achievements
  n_achieve = achievable.shape[-1] + achievements.shape[-1]

  agent = RnnAgent(
    observation_encoder=CraftaxObsEncoder(
      hidden_dim=config["MLP_HIDDEN_DIM"],
      num_layers=config["NUM_MLP_LAYERS"],
      activation=config["ACTIVATION"],
      norm_type=config.get("NORM_TYPE", "none"),
      structured_inputs=config.get("STRUCTURED_INPUTS", False),
      use_bias=config.get("USE_BIAS", True),
      action_dim=env.action_space(env_params).n,
    ),
    rnn=rnn,
    q_fn=DuellingMLP(
      hidden_dim=config.get("Q_HIDDEN_DIM", 512),
      num_layers=config.get("NUM_Q_LAYERS", 2),
      out_dim=env.action_space(env_params).n,
      use_bias=config.get("USE_BIAS", True),
    ),
    achieve_fn=MLP(
      hidden_dim=config.get("Q_HIDDEN_DIM", 512),
      num_layers=config.get("NUM_AUX_LAYERS", 0),
      out_dim=n_achieve,
      use_bias=config.get("USE_BIAS", True),
    ),
  )

  rng, _rng = jax.random.split(rng)
  network_params = agent.init(_rng, example_timestep, method=agent.initialize)

  def reset_fn(params, example_timestep, reset_rng):
    batch_dims = (example_timestep.reward.shape[0],)
    return agent.apply(
      params, batch_dims=batch_dims, rng=reset_rng, method=agent.initialize_carry
    )

  return agent, network_params, reset_fn


def make_multigoal_craftax_agent(
  config: dict,
  env: environment.Environment,
  env_params: environment.EnvParams,
  example_timestep: TimeStep,
  rng: jax.random.PRNGKey,
):
  rnn = vbb.ScannedRNN(hidden_dim=config["AGENT_RNN_DIM"])

  agent = RnnAgent(
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
      num_layers=config.get("NUM_Q_LAYERS", 2),
      out_dim=env.action_space(env_params).n,
      use_bias=config.get("USE_BIAS", True),
    ),
  )
  rng, _rng = jax.random.split(rng)
  network_params = agent.init(_rng, example_timestep, method=agent.initialize)

  def reset_fn(params, example_timestep, reset_rng):
    batch_dims = (example_timestep.reward.shape[0],)
    return agent.apply(
      params, batch_dims=batch_dims, rng=reset_rng, method=agent.initialize_carry
    )

  return agent, network_params, reset_fn


from craftax.craftax.constants import Action, BLOCK_PIXEL_SIZE_IMG, Achievement
from craftax.craftax.renderer import render_craftax_pixels
from visualizer import plot_frames


def render_fn(state):
  image = render_craftax_pixels(state, block_pixel_size=BLOCK_PIXEL_SIZE_IMG)
  return image / 255.0


render_fn = jax.jit(render_fn)


def learner_log_extra(data: dict, config: dict):
  def callback(d):
    n_updates = d.pop("n_updates")

    # Extract the relevant data
    # only use data from batch dim = 0
    # [T, B, ...] --> # [T, ...]
    d_ = jax.tree_map(lambda x: x[:, 0], d)

    mask = d_["mask"]
    discounts = d_["data"].timestep.discount
    rewards = d_["data"].timestep.reward
    actions = d_["data"].action
    q_values = d_["q_values"]
    q_target = d_["q_target"]
    q_values_taken = rlax.batched_index(q_values, actions)
    td_errors = d_["td_errors"]
    q_loss = d_["q_loss"]
    # Create a figure with three subplots
    width = 0.3
    nT = len(rewards)  # e.g. 20 --> 8
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(int(width * nT), 16))

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
    is_last = d_["data"].timestep.last()
    ax4.plot(discounts, label="Discounts")
    ax4.plot(mask, label="mask")
    ax4.plot(is_last, label="is_last")
    format(ax4)
    ax4.set_title("Episode markers")
    ax4.legend()

    # Adjust the spacing between subplots
    # plt.tight_layout()
    # log
    if wandb.run is not wandb.sdk.lib.disabled.RunDisabled:
      wandb.log({f"learner_example/q-values": wandb.Image(fig)})
    plt.close(fig)

    ##############################
    # plot images of env
    ##############################
    # timestep = jax.tree_map(lambda x: jnp.array(x), d_['data'].timestep)
    timesteps: TimeStep = d_["data"].timestep

    # ------------
    # get images
    # ------------

    obs_images = []
    max_len = min(config.get("MAX_EPISODE_LOG_LEN", 40), len(rewards))
    for idx in range(max_len):
      index = lambda y: jax.tree_map(lambda x: x[idx], y)
      obs_image = render_fn(index(d_["data"].timestep.state.env_state))
      obs_images.append(obs_image)
    # ------------
    # plot
    # ------------
    actions_taken = [Action(a).name for a in actions]

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
      elif hasattr(timesteps.state, "current_goal"):
        start_location = timesteps.state.start_position
        goal = timesteps.state.current_goal
        goal_name = Achievement(goal).name
        title += f"\nstart={start_location}\ngoal={goal}\ngoal={goal_name}"
      return title

    fig = plot_frames(
      timesteps=timesteps,
      frames=obs_images,
      panel_title_fn=panel_title_fn,
      ncols=6,
    )
    if wandb.run is not wandb.sdk.lib.disabled.RunDisabled:
      wandb.log({"learner_example/trajectory": wandb.Image(fig)})
    plt.close(fig)

  # this will be the value after update is applied
  n_updates = data["n_updates"] + 1
  is_log_time = n_updates % config["LEARNER_EXTRA_LOG_PERIOD"] == 0

  jax.lax.cond(
    is_log_time, lambda d: jax.debug.callback(callback, d), lambda d: None, data
  )
