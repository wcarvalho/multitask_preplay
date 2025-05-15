from typing import Optional

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp
from flax import struct

from jaxneurorl.agents.value_based_pqn import MLP, BatchRenorm, get_activation_fn

Observation = struct.PyTreeNode


class CategoricalHouzemazeObsEncoder(nn.Module):
  """_summary_

  - observation encoder: CNN over binary inputs
  - MLP with truncated-normal-initialized Linear layer as initial layer for other inputs
  """

  num_categories: int = 10_000
  include_task: bool = True
  embed_hidden_dim: int = 64
  mlp_hidden_dim: int = 256
  num_mlp_layers: int = 0
  num_embed_layers: int = 0
  norm_type: str = "none"
  activation: str = "relu"

  @nn.compact
  def __call__(self, obs: Observation, train: bool = False):
    has_batch = obs.image.ndim == 3
    assert obs.image.ndim in (2, 3), "either [B, H, W] or [H, W]"
    if has_batch:
      flatten = lambda x: x.reshape(x.shape[0], -1)
      expand = lambda x: x[:, None]
    else:
      flatten = lambda x: x.reshape(-1)
      expand = lambda x: x[None]

    act = get_activation_fn(self.activation)
    if self.norm_type == "layer_norm":
      norm = lambda x: act(nn.LayerNorm()(x))
    elif self.norm_type == "batch_norm":
      norm = lambda x: act(BatchRenorm(use_running_average=not train)(x))
    elif self.norm_type == "none":
      norm = lambda x: x
    else:
      raise NotImplementedError(self.norm_type)

    all_flattened = jnp.concatenate(
      (
        flatten(obs.image),
        obs.position,
        expand(obs.direction),
        expand(obs.prev_action),
      ),
      axis=-1,
    ).astype(jnp.int32)
    embedding = nn.Embed(
      num_embeddings=self.num_categories,
      features=self.embed_hidden_dim,
    )(all_flattened)
    embedding = flatten(embedding)
    embedding = norm(embedding)
    embedding = MLP(
      self.mlp_hidden_dim,
      self.num_embed_layers,
      norm_type=self.norm_type,
      activation=self.activation,
    )(embedding)

    if self.include_task:
      kernel_init = nn.initializers.variance_scaling(
        1.0, "fan_in", "normal", out_axis=0
      )
      task_w = nn.Dense(
        128,
        kernel_init=kernel_init,
      )(obs.task_w.astype(jnp.float32))
      task_w = norm(task_w)
      outputs = (embedding, task_w)
      outputs = jnp.concatenate(outputs, axis=-1)
    else:
      outputs = embedding

    outputs = MLP(
      self.mlp_hidden_dim,
      self.num_mlp_layers,
      norm_type=self.norm_type,
      activation=self.activation,
    )(outputs)

    return outputs


class CraftaxObsEncoder(nn.Module):
  """
  Follows: https://github.com/mttga/purejaxql/blob/2205ae5308134d2cedccd749074bff2871832dc8/purejaxql/pqn_rnn_craftax.py#L199

  - observation encoder: CNN over binary inputs
  - MLP with truncated-normal-initialized Linear layer as initial layer for other inputs
  """

  action_dim: int = None
  hidden_dim: int = 512
  num_layers: int = 0
  norm_type: str = "batch_norm"
  activation: str = "relu"
  structured_inputs: bool = False
  use_bias: bool = True
  include_achievable: bool = True

  @nn.compact
  def __call__(self, obs: Observation, train: bool = False):
    act = get_activation_fn(self.activation)
    if self.norm_type == "layer_norm":
      norm = lambda x: act(nn.LayerNorm()(x))
    elif self.norm_type == "batch_norm":
      norm = lambda x: act(BatchRenorm(use_running_average=not train)(x))
    elif self.norm_type == "none":
      norm = lambda x: x
    else:
      raise NotImplementedError(self.norm_type)

    if self.structured_inputs:
      image = norm(obs.image)
    else:
      image = norm(obs)

    if self.num_layers == 1:
      outputs = nn.Dense(self.hidden_dim, use_bias=self.use_bias)(image)
    else:
      outputs = MLP(
        hidden_dim=self.hidden_dim,
        num_layers=self.num_layers,
        norm_type=self.norm_type,
        use_bias=self.use_bias,
        activation=self.activation,
      )(image, train)

    if self.structured_inputs:
      kernel_init = nn.initializers.variance_scaling(
        1.0, "fan_in", "normal", out_axis=0
      )
      achievable = nn.Dense(
        # binary vector
        128,
        kernel_init=kernel_init,
        use_bias=False,
      )(obs.achievable.astype(jnp.float32))
      if self.include_achievable:
        to_concat = (outputs, achievable)
      else:
        to_concat = (outputs,)
      if obs.previous_action is not None:
        action = jax.nn.one_hot(obs.previous_action, self.action_dim or 50)
        # common trick for one-hot encodings, same as nn.Embed
        # main benefit comes from adding action
        action = nn.Dense(128, kernel_init=kernel_init, use_bias=False)(action)
        to_concat = to_concat + (action,)
      outputs = jnp.concatenate(to_concat, axis=-1)

    return outputs


class CraftaxMultiGoalObsEncoder(nn.Module):
  """
  Follows: https://github.com/mttga/purejaxql/blob/2205ae5308134d2cedccd749074bff2871832dc8/purejaxql/pqn_rnn_craftax.py#L199

  - observation encoder: CNN over binary inputs
  - MLP with truncated-normal-initialized Linear layer as initial layer for other inputs
  """

  action_dim: int = None
  hidden_dim: int = 512
  num_layers: int = 0
  norm_type: str = "batch_norm"
  activation: str = "relu"
  use_bias: bool = True
  include_goal: bool = True

  @nn.compact
  def __call__(self, obs: Observation, train: bool = False):
    act = get_activation_fn(self.activation)
    if self.norm_type == "layer_norm":
      norm = lambda x: act(nn.LayerNorm()(x))
    elif self.norm_type == "batch_norm":
      norm = lambda x: act(BatchRenorm(use_running_average=not train)(x))
    elif self.norm_type == "none":
      norm = lambda x: x
    else:
      raise NotImplementedError(self.norm_type)

    image = norm(obs.image)

    if self.num_layers == 1:
      outputs = nn.Dense(self.hidden_dim, use_bias=self.use_bias)(image)
    else:
      outputs = MLP(
        hidden_dim=self.hidden_dim,
        num_layers=self.num_layers,
        norm_type=self.norm_type,
        use_bias=self.use_bias,
        activation=self.activation,
      )(image, train)

    kernel_init = nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)
    if self.include_goal:
      goal = nn.Dense(
        # binary vector
        128,
        kernel_init=kernel_init,
        use_bias=False,
      )(obs.task_w.astype(jnp.float32))
      to_concat = (outputs, goal)
    else:
      to_concat = (outputs,)
    if hasattr(obs, "previous_action") and obs.previous_action is not None:
      action = jax.nn.one_hot(obs.previous_action, self.action_dim or 50)
      # common trick for one-hot encodings, same as nn.Embed
      # main benefit comes from adding action
      action = nn.Dense(128, kernel_init=kernel_init, use_bias=False)(action)
      to_concat = to_concat + (action,)
    outputs = jnp.concatenate(to_concat, axis=-1)

    return outputs
