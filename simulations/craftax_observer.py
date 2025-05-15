import abc
import collections
from typing import Dict, Union, Optional, Callable, Optional

import jax
import jax.numpy as jnp
import flashbax as fbx
from flax import struct
from flax.training.train_state import TrainState
import numpy as np
import wandb
import matplotlib.pyplot as plt


from visualizer import plot_frames
from jaxneurorl.agents.basics import TimeStep

Number = Union[int, float, np.float32, jnp.float32]


@struct.dataclass
class BasicObserverState:
  episode_returns: jax.Array
  episode_lengths: jax.Array
  finished: jax.Array
  idx: jax.Array = jnp.array(0, dtype=jnp.int32)


def get_first(b):
  return jax.tree_map(lambda x: x[0], b)


def add_first_to_buffer(
  buffer: fbx.trajectory_buffer.TrajectoryBufferState,
  buffer_state: struct.PyTreeNode,
  x: struct.PyTreeNode,
):
  """
  x: [num_envs, ...]
  get first env data and dummy dummy time dim.
  """
  x = jax.tree_map(lambda y: y[:1, np.newaxis], x)
  return buffer.add(buffer_state, x)


class Observer:
  """This is an observer that keeps track of timesteps, actions, and predictions.

  It only uses information from the first envionment. Annoying to track each env.

  """

  def __init__(
    self,
    log_period: int = 50_000,
    max_episode_length: int = 200,
    max_num_episodes: int = 200,
    **kwargs,
  ):
    self.log_period = log_period
    self.max_episode_length = max_episode_length
    self.buffer = fbx.make_trajectory_buffer(
      max_length_time_axis=max_episode_length * max_num_episodes,
      min_length_time_axis=1,
      sample_batch_size=1,  # unused
      add_batch_size=1,
      sample_sequence_length=1,  # unused
      period=1,
    )

  def init(self, *args, example_timestep, example_action, **kwargs):
    shape = (self.log_period,) + example_action.shape
    observer_state = BasicObserverState(
      episode_returns=jnp.zeros(shape, dtype=jnp.float32),
      episode_lengths=jnp.zeros(shape, dtype=jnp.int32),
      finished=jnp.zeros(shape, dtype=jnp.int32),
      idx=jnp.zeros(example_action.shape, dtype=jnp.int32),
    )
    return observer_state

  def observe_first(
    self,
    observer_state: BasicObserverState,
    first_timestep: TimeStep,
    agent_state: Optional[struct.PyTreeNode] = None,
  ) -> BasicObserverState:
    del first_timestep, agent_state
    return observer_state

  def observe(
    self,
    observer_state: BasicObserverState,
    predictions: struct.PyTreeNode,
    action: jax.Array,
    next_timestep: TimeStep,
    agent_state: Optional[struct.PyTreeNode] = None,
    **kwargs,
  ) -> None:
    """Update log and flush if terminal + log period hit.


    Args:
        agent_state (struct.PyTreeNode): _description_
        predictions (struct.PyTreeNode): _description_
        action (jax.Array): _description_
        next_timestep (TimeStep): _description_
    """
    del agent_state
    # NOTE: THIS IS WRONG!!!!!! NEED TO RETHINK THROUGH THE LOGIC!!!
    return observer_state
    # next_timestep_last = (next_timestep.last()).astype(jnp.int32)
    # idx = observer_state.idx
    # next_idx = observer_state.idx + next_timestep_last
    # import pdb; pdb.set_trace()
    # observer_state = observer_state.replace(
    #    idx=next_idx,
    #    finished=observer_state.finished.at[idx].add(next_timestep_last),
    #    episode_lengths=observer_state.episode_lengths.at[idx].add(1),
    #    episode_returns=observer_state.episode_returns.at[idx].add(next_timestep.reward),
    # )

    return observer_state
