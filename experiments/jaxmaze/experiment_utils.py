from nicegui import app
import jax
import jax.numpy as jnp
import numpy as np
from housemaze.human_dyna import multitask_env
from housemaze.human_dyna import utils
from housemaze.env import StepType


def task_object_mask(objects, possible_task_objects):
  isin = jax.vmap(lambda a, b: a == b, in_axes=(0, None))(
    objects, possible_task_objects
  )
  isin = isin.sum(0)
  return isin


def update_timestep(timestep, successes):
  return timestep.replace(
    state=timestep.state.replace(
      successes=successes,
    )
  )


class SuccessTrackingAutoResetWrapper(utils.AutoResetWrapper):
  def __init__(
    self,
    env: multitask_env.HouseMaze,
    num_success: int = 5,
  ):
    self._env = env
    self.num_success = num_success

  def reset(self, key, params):
    objects_to_mask = jax.lax.cond(
      params.training,
      lambda: params.reset_params.train_objects[0],
      lambda: params.reset_params.test_objects[0],
    )
    mask = task_object_mask(objects_to_mask, self.task_runner.task_objects)
    probs = mask / (mask.sum())

    params = params.replace(task_probs=probs)

    init_timestep = self._env.reset(key, params)
    return update_timestep(
      timestep=init_timestep,
      successes=jnp.zeros_like(init_timestep.state.task_state.features),
    )

  def compute_successes_remaining(self, timestep, params):
    successes = timestep.state.successes
    features = timestep.state.task_state.features
    task_w = timestep.state.task_w
    successes = successes + features * task_w
    maximum = jnp.ones_like(successes) * self.num_success
    successes = jnp.minimum(successes, maximum)
    remaining = jax.nn.relu(self.num_success - successes)

    objects_to_mask = jax.lax.cond(
      params.training,
      lambda: params.reset_params.train_objects[0],
      lambda: params.reset_params.test_objects[0],
    )
    mask = task_object_mask(objects_to_mask, self.task_runner.task_objects)
    remaining = (remaining * mask).astype(jnp.int32)

    return successes, remaining, mask

  def __auto_reset(self, key, params, timestep):
    key, key_ = jax.random.split(key)

    successes, remaining, mask = self.compute_successes_remaining(timestep, params)
    assert len(params.reset_params.train_objects) == 1, "edge case"
    probs = jax.lax.cond(
      remaining.sum() < 1,
      lambda: mask / mask.sum(),
      lambda: remaining / remaining.sum(),
    )

    params = params.replace(task_probs=probs)
    init_timestep = self._env.reset(key_, params)
    init_timestep = update_timestep(timestep=init_timestep, successes=successes)

    return init_timestep

  def step(self, key: jax.Array, prior_timestep, action, params):
    new_timestep = jax.lax.cond(
      prior_timestep.last(),
      lambda: self.__auto_reset(key, params, prior_timestep),
      lambda: self._env.step(key, prior_timestep, action, params),
    )

    _, remaining, _ = self.compute_successes_remaining(new_timestep, params)
    none_left = remaining.sum() < 1
    new_timestep = jax.lax.cond(
      jnp.logical_or(none_left, prior_timestep.finished),
      lambda: new_timestep.replace(finished=jnp.array(True)),
      lambda: new_timestep,
    )
    return new_timestep
