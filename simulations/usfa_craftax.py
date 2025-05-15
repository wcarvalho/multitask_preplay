import wandb
import matplotlib.pyplot as plt
from visualizer import plot_frames
from craftax.craftax.renderer import render_craftax_pixels
from craftax.craftax import constants
from craftax.craftax.constants import Action

import rlax
import usfa_landmark
from usfa_landmark import *
from networks import CraftaxObsEncoder, CraftaxMultiGoalObsEncoder


def render_fn(state):
  image = render_craftax_pixels(state, block_pixel_size=constants.BLOCK_PIXEL_SIZE_IMG)
  return image / 255.0


render_fn = jax.jit(render_fn)


@struct.dataclass
class UsfaR2D2LossFn(vbb.RecurrentLossFn):
  extract_cumulants: Callable = lambda data: data.timestep.observation.state_features
  extract_task: Callable = lambda data: data.timestep.observation.task_w
  aux_coeff: float = 1.0
  q_coeff: float = 1.0

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
    # Prepare Data
    online_sf = online_preds.sf  # [T+1, B, N, A, C]
    online_w = data.timestep.observation.task_w  # [T+1, B, C]
    target_sf = target_preds.sf  # [T+1, B, N, A, C]

    # NOTE: main change was to concat these
    if hasattr(data.timestep.observation, "achievements"):
      achievements = data.timestep.observation.achievements.astype(online_sf.dtype)
      achievable = data.timestep.observation.achievable.astype(online_sf.dtype)
      cumulants = jnp.concatenate((achievements, achievable), axis=-1)
    else:
      cumulants = self.extract_cumulants(data)
    cumulants = cumulants - self.step_cost / cumulants.shape[-1]

    # Get selector actions from online Q-values for double Q-learning
    dot = lambda x, y: (x * y).sum(axis=-1)
    # vdot = jax.vmap(jax.vmap(dot, (2, None), 2), (2, None), 2)
    online_w = jnp.expand_dims(online_preds.gpi_tasks, axis=-2)  # [T+1, B, N, 1, C]
    online_q = dot(online_sf, online_w)  # [T+1, B, N, A]
    selector_actions = jnp.argmax(online_q, axis=-1)  # [T+1, B, N]

    # Preprocess discounts & rewards
    def float(x):
      return x.astype(jnp.float32)

    discounts = float(data.discount) * self.discount
    lambda_ = jnp.ones_like(data.discount) * self.lambda_
    is_last = float(data.is_last)

    # Prepare loss (via vmaps)
    # vmap over batch dimension (B), return B in dim=1
    td_error_fn = jax.vmap(
      partial(losses.q_learning_lambda_td, tx_pair=self.tx_pair),
      in_axes=1,
      out_axes=1,
    )
    # vmap over policy dimension (N), return N in dim=2
    td_error_fn = jax.vmap(
      td_error_fn, in_axes=(2, None, 2, 2, None, None, None, None), out_axes=2
    )

    # vmap over cumulant dimension (C), return in dim=3
    td_error_fn = jax.vmap(
      td_error_fn, in_axes=(4, None, 4, None, 2, None, None, None), out_axes=3
    )

    # [T, B, N, C]
    sf_t, target_sf_t = td_error_fn(
      online_sf[:-1],  # [T, B, N, A, C]
      data.action[:-1],  # [T, B]
      target_sf[1:],  # [T, B, N, A, C]
      selector_actions[1:],  # [T, B, N]
      cumulants[1:],  # [T, B, C]
      discounts[1:],
      is_last[1:],
      lambda_[1:],
    )

    # Ensure target = 0 when episode terminates
    target_sf_t = target_sf_t * data.discount[:-1, :, None, None]
    batch_td_error = target_sf_t - sf_t

    # Ensure loss = 0 when episode truncates
    truncated = (data.discount + is_last) > 1
    loss_mask = (1 - truncated).astype(batch_td_error.dtype)

    loss_mask_td = loss_mask[:-1, :, None, None]
    batch_td_error = batch_td_error * loss_mask_td

    batch_sq_error = jnp.square(batch_td_error)

    ################
    # Compute SF loss
    ################
    # [T, B, N, C]
    batch_sf_loss = 0.5 * batch_sq_error

    # [T, B, N, C] --> [B]
    batch_sf_loss_mean = batch_sf_loss.sum(axis=3).mean(axis=(0, 2))

    # mean over policy and cumulant dimensions
    sf_td_error = jnp.abs(batch_td_error).mean(axis=(2, 3))
    ################
    # Compute Q-loss
    ################
    # NOTE: CHANGE is to weight TD errors by task weights
    task_w = data.timestep.observation.task_w[:-1]
    task_w = task_w[:, :, None]  # add policy dimension (N)

    # [T, B, N, C] --> [T, B, N]
    q_td_error = (task_w * batch_td_error).sum(axis=-1)

    batch_q_loss = 0.5 * jnp.square(q_td_error)
    # [T, B, N] --> [B]
    batch_q_loss_mean = batch_q_loss.mean(axis=(0, 2))

    batch_loss_mean = (
      self.q_coeff * batch_q_loss_mean + self.aux_coeff * batch_sf_loss_mean
    )

    # mean over policy dimension
    batch_td_error = self.q_coeff * jnp.abs(q_td_error).mean(
      2
    ) + self.aux_coeff * jnp.abs(sf_td_error)

    metrics = {
      "0.sf_loss": batch_sf_loss_mean.mean(),
      "0.q_loss": batch_q_loss_mean.mean(),
      "0.sf_td": jnp.abs(batch_td_error).mean(),
      "0.q_td": jnp.abs(q_td_error).mean(),
      "1.cumulants": cumulants.mean(),
      "z.sf_mean": online_sf.mean(),
      "z.sf_var": online_sf.var(),
      "z.cumulants_min": cumulants[1:].min(),
      "z.cumulants_max": cumulants[1:].max(),
    }

    if self.logger.learner_log_extra is not None:
      self.logger.learner_log_extra(
        {
          "data": data,
          "cumulants": cumulants[1:],
          "td_errors": jnp.abs(batch_td_error),  # [T, B, N, C]
          "mask": loss_mask,  # [T, B]
          "sf_values": online_sf[:-1],  # [T+1, B, N, A, C]
          "sf_loss": batch_sf_loss,  # [T, B, N, C]
          "sf_target": target_sf_t,
          "n_updates": steps,
        }
      )

    return batch_td_error, batch_loss_mean, metrics  # [T, B, N, C], [B]


def make_loss_fn_class(config) -> vbb.RecurrentLossFn:
  return functools.partial(
    UsfaR2D2LossFn,
    discount=config["GAMMA"],
    step_cost=config.get("STEP_COST", 0.0),
    aux_coeff=config.get("AUX_COEFF", 1.0),
    q_coeff=config.get("Q_COEFF", 1.0),
    tx_pair=(
      rlax.SIGNED_HYPERBOLIC_PAIR
      if config.get("TX_PAIR", "none") == "hyperbolic"
      else rlax.IDENTITY_PAIR
    ),
  )


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
      out_dim=task_dim,
    )
    advantage_mlp = MLP(
      hidden_dim=self.hidden_dim,
      num_layers=self.num_layers,
      out_dim=self.out_dim * task_dim,
    )
    assert self.out_dim > 0, "must have at least one action"

    # Compute value & advantage
    value = value_mlp(x)  # [B, C]
    advantages = advantage_mlp(x)  # [B, A*C]

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

    return sf, q_values


class SfGpiHead(nn.Module):
  num_actions: int
  state_features_dim: int
  all_policy_tasks: jnp.ndarray
  num_layers: int = 2
  hidden_dim: int = 512
  nsamples: int = 10
  variance: float = 0.5
  eval_task_support: str = "train"

  def setup(self):
    self.policy_net = lambda x: x
    self.sf_net = DuellingDotMLP(
      hidden_dim=self.hidden_dim,
      num_layers=self.num_layers,
      out_dim=self.num_actions,
    )

  @nn.compact
  def __call__(
    self,
    usfa_input: jnp.ndarray,
    task: jnp.ndarray,
    train_tasks: jnp.ndarray,
    rng: jax.random.PRNGKey,
  ) -> USFAPreds:
    return self.evaluate(
      usfa_input=usfa_input, task=task, train_tasks=train_tasks, support="eval"
    )

  def evaluate(
    self,
    usfa_input: jnp.ndarray,
    task: jnp.ndarray,
    train_tasks: jnp.ndarray,
    support: str = "",
  ) -> USFAPreds:
    support = support or self.eval_task_support

    if support == "train":
      policies = jnp.array(
        [get_task_vector(task, self.all_policy_tasks) for task in train_tasks]
      )
      gpi_tasks = train_tasks
    elif support == "eval":
      policies = jnp.expand_dims(get_task_vector(task, self.all_policy_tasks), axis=-2)
      gpi_tasks = jnp.expand_dims(task, axis=-2)
    elif support == "train_eval":
      task_expand = jnp.expand_dims(
        get_task_vector(task, self.all_policy_tasks), axis=-2
      )
      train_policies = jnp.array(
        [get_task_vector(task, self.all_policy_tasks) for task in train_tasks]
      )
      policies = jnp.concatenate((train_policies, task_expand), axis=-2)
      gpi_tasks = jnp.concatenate((train_tasks, task), axis=-1)
    else:
      raise RuntimeError(self.eval_task_support)

    return self.sfgpi(
      usfa_input=usfa_input, policies=policies, gpi_tasks=gpi_tasks, task=task
    )

  def sfgpi(
    self,
    usfa_input: jnp.ndarray,
    policies: jnp.ndarray,
    gpi_tasks: jnp.ndarray,
    task: jnp.ndarray,
  ) -> USFAPreds:
    def compute_sf_q(sf_input: jnp.ndarray, policy: jnp.ndarray, task: jnp.ndarray):
      sf_input = jnp.concatenate((sf_input, policy), axis=-1)  # 2D
      return self.sf_net(sf_input, task)

    # []
    policy_embeddings = self.policy_net(policies)
    sfs, q_values = jax.vmap(compute_sf_q, in_axes=(None, 0, None), out_axes=0)(
      usfa_input, policy_embeddings, task
    )

    q_values = jnp.max(q_values, axis=-2)
    policies = jnp.expand_dims(policies, axis=-2)
    policies = jnp.tile(policies, (1, self.num_actions, 1))

    return USFAPreds(
      sf=sfs, policy=policies, gpi_tasks=gpi_tasks, q_vals=q_values, task=task
    )


def make_craftax_agent(
  config: dict,
  env: environment.Environment,
  env_params: environment.EnvParams,
  example_timestep: TimeStep,
  rng: jax.random.PRNGKey,
) -> Tuple[nn.Module, Params, vbb.AgentResetFn]:
  # just 1 task
  train_tasks = jnp.expand_dims(example_timestep.observation.task_w, axis=-2)

  sf_head = SfGpiHead(
    num_actions=env.action_space(env_params).n,
    state_features_dim=example_timestep.observation.task_w.shape[-1],
    nsamples=config.get("NSAMPLES", 1),
    eval_task_support=config.get("EVAL_TASK_SUPPORT", "eval"),
    train_tasks=train_tasks,
    num_layers=config.get("NUM_SF_LAYERS", 2),
    hidden_dim=config.get("SF_HIDDEN_DIM", 512),
  )

  agent = UsfaAgent(
    observation_encoder=CraftaxObsEncoder(
      hidden_dim=config["MLP_HIDDEN_DIM"],
      num_layers=config["NUM_MLP_LAYERS"],
      activation=config["ACTIVATION"],
      norm_type=config.get("NORM_TYPE", "none"),
      structured_inputs=config.get("STRUCTURED_INPUTS", False),
    ),
    rnn=vbb.ScannedRNN(hidden_dim=config["AGENT_RNN_DIM"]),
    sf_head=sf_head,
  )

  rng, _rng = jax.random.split(rng)
  network_params = agent.init(_rng, example_timestep, method=agent.initialize)

  def reset_fn(params, example_timestep, reset_rng):
    batch_dims = (example_timestep.reward.shape[0],)
    return agent.apply(
      params, batch_dims=batch_dims, rng=reset_rng, method=agent.initialize_carry
    )

  return agent, network_params, reset_fn


class UsfaAgent(nn.Module):
  """Note: only change is that this assumes train tasks come from observation"""

  observation_encoder: nn.Module
  rnn: vbb.ScannedRNN
  sf_head: SfGpiHead
  learn_tasks: jnp.ndarray
  learn_z_vectors: str

  def initialize(self, x: TimeStep):
    rng = jax.random.PRNGKey(0)
    batch_dims = (x.reward.shape[0],)
    rnn_state = self.initialize_carry(rng, batch_dims)
    return self.__call__(rnn_state, x, rng)

  def initialize_carry(self, *args, **kwargs):
    return self.rnn.initialize_carry(*args, **kwargs)

  def __call__(
    self, rnn_state, x: TimeStep, rng: jax.random.PRNGKey, evaluate: bool = False
  ):
    embedding = self.observation_encoder(x.observation)
    rnn_in = vbb.RNNInput(obs=embedding, reset=x.first())
    rng, _rng = jax.random.split(rng)
    new_rnn_state, rnn_out = self.rnn(rnn_state, rnn_in, _rng)

    if evaluate:
      predictions = jax.vmap(self.sf_head.evaluate)(
        rnn_out,
        x.observation.task_w,
        x.observation.train_tasks,
      )
    else:
      B = rnn_out.shape[0]
      predictions = jax.vmap(self.sf_head)(
        # [B, D], [B, D]
        rnn_out,
        x.observation.task_w,
        x.observation.train_tasks,
        jax.random.split(rng, B),
      )

    return predictions, new_rnn_state

  def unroll(self, rnn_state, xs: TimeStep, rng: jax.random.PRNGKey):
    embedding = nn.BatchApply(self.observation_encoder)(xs.observation)
    rnn_in = vbb.RNNInput(obs=embedding, reset=xs.first())
    rng, _rng = jax.random.split(rng)
    new_rnn_state, rnn_out = self.rnn.unroll(rnn_state, rnn_in, _rng)

    if self.learn_z_vectors == "ALL_TASKS":
      pred_fn = jax.vmap(self.sf_head.sfgpi, in_axes=(0, None, None, 0), out_axes=0)
      pred_fn = jax.vmap(pred_fn, in_axes=(0, None, None, 0), out_axes=0)
      # [N, D]
      learn_z_vectors = jnp.array(
        [get_task_vector(task, self.learn_tasks) for task in self.learn_tasks]
      )
      learn_w_vectors = self.learn_tasks
    elif self.learn_z_vectors == "TRAIN":
      pred_fn = jax.vmap(jax.vmap(self.sf_head.sfgpi))
      # [T, B, 1, D]
      v_map_get_task_vector = jax.vmap(get_task_vector, (0, None), 0)
      v_map_get_task_vector = jax.vmap(v_map_get_task_vector, (0, None), 0)
      learn_z_vectors = jnp.expand_dims(
        v_map_get_task_vector(xs.observation.task_w, self.learn_tasks), axis=-2
      )
      learn_w_vectors = jnp.expand_dims(xs.observation.task_w, axis=-2)
    else:
      raise ValueError(self.learn_z_vectors)
    predictions = pred_fn(
      # usfa_input (T, B, D), learn_z_vectors (N, D), task (C)
      # NOTE: task isn't used in lsos
      rnn_out,
      learn_z_vectors,
      learn_w_vectors,
      xs.observation.task_w,
    )

    return predictions, new_rnn_state


def make_multigoal_craftax_agent(
  config: dict,
  env: environment.Environment,
  env_params: environment.EnvParams,
  example_timestep: TimeStep,
  rng: jax.random.PRNGKey,
  all_tasks: jnp.ndarray,
) -> Tuple[nn.Module, Params, vbb.AgentResetFn]:
  sf_head = SfGpiHead(
    num_actions=env.action_space(env_params).n,
    state_features_dim=example_timestep.observation.task_w.shape[-1],
    nsamples=config.get("NSAMPLES", 1),
    all_policy_tasks=all_tasks,
    eval_task_support=config.get("EVAL_TASK_SUPPORT", "train"),
    num_layers=config.get("NUM_SF_LAYERS", 2),
    hidden_dim=config.get("SF_HIDDEN_DIM", 512),
  )

  agent = UsfaAgent(
    observation_encoder=CraftaxMultiGoalObsEncoder(
      hidden_dim=config["MLP_HIDDEN_DIM"],
      num_layers=config["NUM_MLP_LAYERS"],
      activation=config["ACTIVATION"],
      norm_type=config.get("NORM_TYPE", "none"),
    ),
    rnn=vbb.ScannedRNN(hidden_dim=config["AGENT_RNN_DIM"]),
    sf_head=sf_head,
    learn_tasks=all_tasks,
    learn_z_vectors=config.get("LEARN_VECTORS", "ALL_TASKS"),
  )

  rng, _rng = jax.random.split(rng)
  network_params = agent.init(_rng, example_timestep, method=agent.initialize)

  def reset_fn(params, example_timestep, reset_rng):
    batch_dims = (example_timestep.reward.shape[0],)
    return agent.apply(
      params, batch_dims=batch_dims, rng=reset_rng, method=agent.initialize_carry
    )

  return agent, network_params, reset_fn


# only redoing this
def learner_log_extra(
  data: dict,
  config: dict,
):
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
    cumulants = d_["cumulants"]  # [T, C]
    sf_values = d_["sf_values"][:, 0]  # [T, A, C]
    sf_target = d_["sf_target"][:, 0]  # [T, C]
    # [T, C]
    # vmap over cumulant dimension
    sf_values_taken = jax.vmap(rlax.batched_index, in_axes=(2, None), out_axes=1)(
      sf_values, actions[:-1]
    )
    # sf_td_errors = d_['td_errors']  # [T, C]
    # sf_loss = d_['sf_loss']  # [T, C]

    ##############################
    # SF-learning plots
    ##############################
    # Create a figure with subplots for each cumulant
    num_cumulants = cumulants.shape[-1]
    num_cols = min(6, num_cumulants)
    num_rows = (num_cumulants + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    axes = axes.flatten()

    for i in range(num_cumulants):
      ax = axes[i]
      time_steps = range(len(cumulants))

      ax.plot(time_steps, cumulants[:, i], label="Cumulants")
      ax.plot(time_steps, sf_values_taken[:, i], label="SF Values Taken")
      ax.plot(time_steps, sf_target[:, i], label="SF Target")
      ax.set_title(f"Cumulant {i + 1}")
      ax.set_xlabel("Time")
      ax.set_ylabel("Value")
      ax.legend()
      ax.grid(True)

    # Remove any unused subplots
    for i in range(num_cumulants, len(axes)):
      fig.delaxes(axes[i])

    # Log the Q-learning figure
    if wandb.run is not wandb.sdk.lib.disabled.RunDisabled:
      wandb.log({f"learner_example/sf-learning": wandb.Image(fig)})
    plt.close(fig)

    ##############################
    # Q-learning plots
    ##############################
    # Plot rewards and q-values in the top subplot
    width = 0.3
    nT = len(rewards)  # e.g. 20 --> 8

    task = d_["data"].timestep.observation.task_w[:-1]
    q_values_taken = (sf_values_taken * task).sum(-1)
    q_target = (sf_target * task).sum(-1)
    td_errors = jnp.abs(q_target - q_values_taken)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(int(width * nT), 16))

    def format(ax):
      ax.set_xlabel("Time")
      ax.grid(True)
      ax.set_xticks(range(0, len(rewards), 1))

    # Set the same x-limit for all subplots
    x_max = len(rewards)

    ax1.plot(rewards, label="Rewards")
    ax1.plot(q_values_taken, label="Q-Values")
    ax1.plot(q_target, label="Q-Targets")
    ax1.set_title("Rewards and Q-Values")
    ax1.set_xlim(0, x_max)
    format(ax1)
    ax1.legend()

    # Plot TD errors in the middle subplot
    ax2.plot(td_errors)
    ax2.set_xlim(0, x_max)
    format(ax2)
    ax2.set_title("TD Errors")

    # Plot episode quantities
    is_last = d_["data"].timestep.last()
    ax3.plot(discounts, label="Discounts")
    ax3.plot(mask, label="mask")
    ax3.plot(is_last, label="is_last")
    ax3.set_xlim(0, x_max)
    format(ax3)
    ax3.set_title("Episode markers")
    ax3.legend()

    # Ensure all subplots have the same x-axis range
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

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

    # state_images = []
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

    def index(t, idx):
      return jax.tree_map(lambda x: x[idx], t)

    def panel_title_fn(timesteps, i):
      title = f"t={i}\n"
      title += f"{actions_taken[i]}\n"
      title += f"r={timesteps.reward[i]}, $\\gamma={timesteps.discount[i]}$"
      if hasattr(timesteps.state, "current_goal"):
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
