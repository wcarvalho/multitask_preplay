import os
import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import asyncio
from functools import partial
from typing import Callable, List, Tuple, Optional, Union
from dotenv import load_dotenv
from flax import struct, serialization
import jax
import jax.tree_util as jtu
import distrax
from jax.experimental import io_callback
import jax.numpy as jnp
import numpy as np
import os
from skimage.transform import resize

import matplotlib.pyplot as plt
from nicegui import ui, app
import nicewebrl
from nicewebrl import JaxWebEnv, base64_npimage, TimestepWrapper
from nicewebrl import Stage, EnvStage
from nicewebrl import get_logger
from craftax.craftax.renderer import render_craftax_pixels
from simulations.craftax_web_env import EnvParams as OriginalEnvParams
from craftax.craftax.constants import (
  Action,
  BlockType,
  # BLOCK_PIXEL_SIZE_HUMAN,
  BLOCK_PIXEL_SIZE_IMG,
  Achievement,
)

# import craftax_experiment_configs
from simulations.craftax_experiment_configs import (
  PRACTICE_BLOCK_CONFIG,
  PATHS_CONFIGS,
  JUNCTURE_CONFIGS,
  BlockConfig,
  make_block_env_params,
  POSSIBLE_GOALS,
  BLOCK_TO_GOAL,
  GOAL_TO_BLOCK,
  BLOCK_TO_IDX,
  get_goal_image,
)


load_dotenv()

logger = get_logger(__name__)
VERBOSITY = int(os.environ.get("VERBOSITY", 0))
DEBUG = int(os.environ.get("DEBUG", 0))
MANIPULATION = os.environ.get("MANIPULATION", "paths")
SAY_REUSE = int(os.environ.get("SAY_REUSE", 0))
NUM_BLOCKS = int(os.environ.get("NUM_BLOCKS", 100))
EVAL_SHOW_MAP = int(os.environ.get("EVAL_SHOW_MAP", 1))
GIVE_INSTRUCTIONS = int(os.environ.get("GIVE_INSTRUCTIONS", 0))

PRECOMPILE = int(os.environ.get("PRECOMPILE", 1))

DUMMY_ENV = int(os.environ.get("DUMMY_ENV", 0))
MONSTERS = int(os.environ.get("MONSTERS", 1))
NAME = os.environ.get("NAME", "exp")
DATA_DIR = os.environ.get("DATA_DIR", "data")
NTRAIN = int(os.environ.get("NTRAIN", 8))

MAX_STAGE_EPISODES = 50 if DEBUG == 0 else 2
MIN_SUCCESS_TASK = NTRAIN if DEBUG == 0 else 1
MAX_START_POSITIONS = 10


@struct.dataclass
class EnvParams(OriginalEnvParams):
  active_goals: Tuple[int, ...] = tuple()
  num_success: int = 5
  min_samples_per_location: int = 1
  num_start_locations: int = 10


class BlockStageConfig(struct.PyTreeNode):
  world_seed: int
  start_positions: List[Tuple[int, int]]
  goals: List[int]


if DUMMY_ENV:
  # from craftax_dummy_env import CraftaxSymbolicWebEnvNoAutoReset
  from simulations.craftax_web_env import (
    CraftaxSymbolicWebEnvNoAutoResetDummy as CraftaxSymbolicWebEnvNoAutoReset,
  )

  def fullmap_render(timestep, world_seed):
    return np.zeros((96, 96, 3), dtype=np.uint8)

else:
  from simulations.craftax_web_env import CraftaxSymbolicWebEnvNoAutoReset
  from craftax_fullmap_renderer import render_craftax_pixels as render_fullmap_pixels

  def fullmap_render(timestep, world_seed):
    with jax.disable_jit():
      return render_fullmap_pixels(
        timestep.state, show_agent=True, block_pixel_size=BLOCK_PIXEL_SIZE_IMG
      ).astype(jnp.uint8)


def get_user_save_file_fn():
  return (
    f"{DATA_DIR}/user={app.storage.user.get('seed')}_name={NAME}_debug={DEBUG}.json"
  )


########################################
# Utility functions
########################################
EvaluateSuccessFn = Callable[[nicewebrl.TimeStep, EnvParams], jnp.bool_]


def sample_from_remaining(
  key: str,
  possible_values: list,
  min_samples: Union[list, int],
  num_dimensions: int = 1e10,
) -> tuple:
  """Sample from remaining counts while maintaining storage state.

  Args:
      storage_key: Key to identify the type of sampling (e.g. 'goal', 'start_position')
      possible_values: List of possible values to sample from
      min_samples: List of minimum required samples for each value

  Returns:
      tuple: (sampled value, dict of remaining counts)
  """
  # get subset of dims
  num_dimensions = int(num_dimensions)
  possible_values = possible_values[:num_dimensions]

  # Convert values to strings for storage
  possible_values_str = [str(i) for i in possible_values]

  if isinstance(min_samples, list):
    num_required = [int(i) for i in min_samples]
  elif isinstance(min_samples, (np.ndarray, jnp.ndarray)):
    if min_samples.shape == possible_values.shape:
      num_required = [int(i) for i in min_samples]
    elif min_samples.shape in ((1,), ()):
      num_required = [int(min_samples)] * len(possible_values)
    else:
      raise RuntimeError(f"{min_samples.shape}, {min_samples.dtype}")
  else:
    num_required = [int(min_samples)] * len(possible_values)

  num_dimensions = min(num_dimensions, len(possible_values))
  possible_values_str = possible_values_str[:num_dimensions]  # redundant
  num_required = num_required[:num_dimensions]

  # Get or initialize remaining counts
  default = {g: n for g, n in zip(possible_values_str, num_required)}
  remaining_counts = app.storage.user.get(key, default)
  app.storage.user[key] = remaining_counts

  # Maintain order of values
  output_counts = jnp.array(
    [
      remaining_counts.get(str(g), n) for g, n in zip(possible_values_str, num_required)
    ],
    dtype=jnp.int32,
  )

  # Sample based on remaining counts
  output_counts = output_counts + 1e-6
  probs = output_counts / (output_counts.sum())
  sampler = distrax.Categorical(probs=probs)
  rng = nicewebrl.new_rng()
  idx = sampler.sample(seed=rng)
  sampled_value = possible_values[idx]

  return sampled_value, remaining_counts


def sample_goal_and_position(
  possible_goals,
  success_per_goal,
  start_positions,
  min_samples_per_location: int = 1,
  num_start_locations: int = 1,
):
  try:
    stage_idx = app.storage.user.get("stage_idx", 0)
  except Exception:
    # no page setup yet
    logger.info("no page setup yet")
    dummy_goal = possible_goals[0]
    dummy_position = start_positions[0]
    return dummy_goal, dummy_position

  ###################
  # sample goal in proportion to remaining successes
  ####################
  stage_idx = app.storage.user.get("stage_idx", 0)
  key = f"{stage_idx}_remaining_goal_success"
  goal, remaining_counts = sample_from_remaining(
    key=key,
    possible_values=possible_goals,
    min_samples=success_per_goal,
  )
  app.storage.user[key] = remaining_counts
  logger.info(f"sampled: {goal}. remaining goal counts: {remaining_counts}")

  ###################
  # sample start position in proportion to how often not sampled
  ####################
  key = f"{stage_idx}_goal_{goal}_remaining_start_positions"
  start_position, remaining_counts = sample_from_remaining(
    key=key,
    possible_values=start_positions,
    min_samples=min_samples_per_location,
    num_dimensions=num_start_locations,
  )
  logger.info(
    f"sampled: {start_position} for goal {goal}. prior start position counts: {remaining_counts}"
  )
  app.storage.user[key] = remaining_counts

  return goal, start_position


def on_episode_finish_updates(current_goal, success, start_position):
  try:
    stage_idx = app.storage.user["stage_idx"]
  except Exception:
    # no page setup yet
    return
  ################################
  # Update number of successes
  ################################
  key = f"{stage_idx}_remaining_goal_success"
  remaining = app.storage.user.get(key)
  if remaining:
    current_goal = str(int(current_goal))
    remaining[current_goal] -= int(success)
    remaining[current_goal] = max(remaining[current_goal], 0)
    app.storage.user[key] = remaining
    logger.info(f"remaining {key} counts: {remaining}")
  else:
    logger.info(f"{key} not found. storage: {app.storage.user}")

  ################################
  # Update number of times position seen
  ################################
  key = f"{stage_idx}_goal_{int(current_goal)}_remaining_start_positions"
  remaining = app.storage.user.get(key)
  if remaining:
    start_position = str(start_position)
    remaining[start_position] -= 1
    remaining[start_position] = max(remaining[start_position], 0)
    app.storage.user[key] = remaining
    logger.info(f"remaining {key} counts: {remaining}")
  else:
    logger.info(f"{key} not found. storage: {app.storage.user}")


class StatefulResetEnvWrapper(TimestepWrapper):
  """
  Wraps an environment to (1) sample new goals and (2) track the number of successful episodes.

  This is in tracker because goals are sampled until enough successful episodes are completed.

  NOTE: this intercepts step in TimestepWrapper and replaces its reset with this reset. This is how you get goal-control automatically while you step and automatically reset.

  NOTE: can't autoreset because no longer jittable. reset in stage.
  """

  def __init__(
    self,
    env,
    possible_goals: jnp.ndarray = None,
    **kwargs,
  ):
    super().__init__(env, autoreset=False, **kwargs)
    self.possible_goals = possible_goals

  def reset(self, key, params: EnvParams):
    """Sample goals according to user successes"""
    #########################################
    # Compute how many successful episodes must be completed
    # for each goal
    #########################################
    goal, start_position = io_callback(
      sample_goal_and_position,
      (
        jax.ShapeDtypeStruct(shape=(), dtype=jnp.int32),  # goal
        jax.ShapeDtypeStruct(shape=(2,), dtype=jnp.int32),  # start_position
      ),
      self.possible_goals,
      params.num_success * params.active_goals,
      params.start_positions,
      params.min_samples_per_location,
      params.num_start_locations,
    )

    params = params.replace(
      current_goal=goal.astype(jnp.int32),
      start_positions=start_position.astype(jnp.int32),
    )
    assert params.current_goal.ndim == 0, "multiple goals?"
    assert params.start_positions.shape == (2,), "should be (y, z)"

    timestep = super().reset(key, params)
    return timestep


########################################
# Define actions and corresponding keys
########################################
actions = [Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP, Action.DO]
action_array = jnp.array([a.value for a in actions])
action_keys = ["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp", " "]
action_to_name = [a.name for a in actions]

########################################
# Define goals
########################################
possible_goals = jnp.array(POSSIBLE_GOALS)


def blocks_to_goals(blocks: BlockType, default: int = None) -> int:
  """get all possible goals from a list of blocks.

  if don't provide default, will remove unknown blocks.
  """
  goals = [BLOCK_TO_GOAL.get(b, default) for b in blocks]
  goals = [g for g in goals if g is not None]
  return goals


def blocks_to_active_goals(blocks: List[BlockType]) -> jnp.ndarray:
  """Creates a binary vector indicating which goals are active based on the provided Achievements."""
  ngoals = len(POSSIBLE_GOALS)
  goals = jnp.zeros(ngoals, dtype=jnp.float32)
  for block in blocks:
    if hasattr(block, "value"):
      block = block.value
    idx = BLOCK_TO_IDX[block]
    goals = goals.at[idx].set(1)
  return goals


# random default
all_goals_active = jnp.ones(len(possible_goals), dtype=jnp.float32)

if MANIPULATION == "paths":
  dummy_block_config = PATHS_CONFIGS[0]
elif MANIPULATION == "juncture":
  dummy_block_config = JUNCTURE_CONFIGS[0]
else:
  raise ValueError(f"Invalid manipulation: {MANIPULATION}")
########################################
# Define Craftax environment
########################################

static_env_params = CraftaxSymbolicWebEnvNoAutoReset.default_static_params()
static_env_params = static_env_params.replace(
  max_melee_mobs=MONSTERS,
  max_ranged_mobs=MONSTERS,
  max_passive_mobs=10,  # cows
  initial_crafting_tables=True,
  initial_strength=20,
  map_size=(48, 48),
  num_levels=1,
)
jax_env = CraftaxSymbolicWebEnvNoAutoReset(
  static_env_params=static_env_params,
)


def make_start_position(start_positions):
  start_position = jnp.zeros((MAX_START_POSITIONS, 2), dtype=jnp.int32)
  return start_position.at[: len(start_positions)].set(jnp.asarray(start_positions))


dummy_start_position = make_start_position((24, 24))

default_params = EnvParams(
  day_length=100000,
  max_timesteps=300 if DEBUG == 0 else 2,
  mob_despawn_distance=100000,
  # possible_goals=possible_goals,
  active_goals=all_goals_active,
  world_seeds=(0,),
  start_positions=dummy_start_position,
  num_start_locations=1,
)


dummy_params = make_block_env_params(dummy_block_config, default_params).replace(
  # to have compilation use valid current_goal value
  current_goal=dummy_block_config.train_objects[0],
)

jax_env = StatefulResetEnvWrapper(
  env=jax_env,
  possible_goals=possible_goals,
)

# create web environment wrapper
jax_web_env = JaxWebEnv(env=jax_env, actions=action_array)


# Define rendering function
if DUMMY_ENV:

  def render_fn(timestep: nicewebrl.TimeStep):
    return timestep.observation.astype(jnp.uint8)

else:

  def render_fn(timestep: nicewebrl.TimeStep):
    return render_craftax_pixels(
      timestep.state, block_pixel_size=BLOCK_PIXEL_SIZE_IMG
    ).astype(jnp.uint8)


# precompile vmapped render fn that will vmap over all actions
vmap_render_fn = None
# pre-compile jax functions before experiment starts.
if PRECOMPILE:
  jax_web_env.precompile(dummy_env_params=dummy_params)
  vmap_render_fn = jax_web_env.precompile_vmap_render_fn(render_fn, dummy_params)
  render_fn = (
    jax.jit(render_fn)
    .lower(jax_web_env.reset(jax.random.PRNGKey(0), dummy_params))
    .compile()
  )


def evaluate_success_fn(timestep: nicewebrl.TimeStep, params: EnvParams):
  success = timestep.reward > 0.5 and timestep.last() > 0
  on_episode_finish_updates(
    timestep.state.current_goal, success, timestep.state.start_position
  )
  return success


########################################
# Utils for defining stages of experiment
########################################
def remove_extra_spaces(text):
  """For each line, remove extra space."""
  return "\n".join([i.strip() for i in text.strip().split("\n")])


# ------------------
# Environment stage
# ------------------
def make_image_html(src, id="stateImage", percent=100):
  html = f"""
  <div id="{id}Container" style="display: flex; justify-content: center; align-items: center;">
      <img id="{id}" src="{src}" style="width: {percent}%; height: {percent}%; object-fit: contain;">
  </div>
  """
  return html


def get_remaining_goals():
  stage_idx = app.storage.user.get("stage_idx", 0)
  key = f"{stage_idx}_remaining_goal_success"
  remaining = app.storage.user.get(key, {})
  name = lambda i: Achievement(int(i)).name.replace("_", " ").title()
  remaining = {name(i): r for i, r in remaining.items()}
  return remaining


def debug_info(stage):
  stage_state = stage.get_user_data("stage_state")
  debug_info = (
    f"**Manipulation**: {stage.metadata['block_metadata'].get('manipulation')}. "
  )
  if stage_state is not None:
    state = stage_state.timestep.state
    # ------------
    # stage, world information
    # ------------
    debug_info += f"**Eval**: {stage.metadata['eval']}. "
    # debug_info += f"**World**: {stage.metadata['world_seed']}. "
    # debug_info += f"**Episode** idx: {stage_state.nepisodes}. "
    debug_info += f"**Step**: {state.timestep}/{stage.env_params.max_timesteps}. "
    # ui.markdown(debug_info)
    # ------------
    # position information
    # ------------
    start_positions = stage.env_params.start_positions
    debug_info += f"**start_positions**: {start_positions}. "
    debug_info += f"**position**: {state.player_position}. "
    ui.markdown(debug_info)
    # ------------
    # remaining goals information
    # ------------
    debug_info = f"**remaining**: {get_remaining_goals()}. "
    ui.markdown(debug_info)

  return debug_info


# ------------------
# Instruction stage
# ------------------
async def experiment_instructions_display_fn(stage, container):
  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)

    ui.markdown(f"## {stage.title}")
    ui.markdown(f"{remove_extra_spaces(stage.body)}", extras=["cuddled-lists"])
    ui.markdown(
      "**You can control the agent using the arrow keys. Press the space bar to 'interact', i.e. to collect objects**."
    )
    if EVAL_SHOW_MAP == 0:
      ui.markdown("**You will only get the full map in phase 1.**")
    ui.markdown("Below are the stones you will need to mine.")

    # Get all possible goals and create images for each
    goals = [int(g) for g in possible_goals]
    width = 1.5
    figsize = (len(goals) * width, width)

    with ui.matplotlib(figsize=figsize).figure as fig:
      axs = fig.subplots(1, len(goals))
      for i, goal_idx in enumerate(goals):
        # Get image for this goal
        image = get_goal_image(goal_idx)
        # Plot in matplotlib
        axs[i].imshow(image)
        block_idx = GOAL_TO_BLOCK[goal_idx]
        category = BlockType(block_idx).name.title()
        axs[i].set_title(f"{category}")
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].axis("off")

      # Adjust layout
      fig.tight_layout()


async def wait_period():
  await asyncio.sleep(1)
  button = ui.button("next")
  await nicewebrl.wait_for_button_or_keypress(button, ignore_recent_press=True)
  button.delete()


async def practice_stage_instructions_display_fn(
  stage, container, eval=False, **kwargs
):
  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)

    ui.markdown(f"## {stage.title}")
    if DEBUG:
      debug_info(stage)
    ui.markdown(f"{remove_extra_spaces(stage.body)}", extras=["cuddled-lists"])

    ui.markdown("Below are the stones you will need to mine.")
    if SAY_REUSE and not eval:
      await wait_period()
      ui.markdown(
        "**We note the stone relevant to phase 2 in <span style='color: green'>GREEN</span> below**"
      )
      await wait_period()

    # Get all possible goals
    goals = [int(g) for g in possible_goals]

    # Create random display order
    key = nicewebrl.new_rng()
    order = jax.random.permutation(key, jnp.arange(len(goals)))

    test_objects = stage.metadata["block_metadata"]["test_objects"]
    test_objects = blocks_to_goals(test_objects)

    width = 1.5
    figsize = (len(goals) * width, width)
    with ui.matplotlib(figsize=figsize).figure as fig:
      axs = fig.subplots(1, len(goals))
      for i, idx in enumerate(order):
        goal_idx = goals[idx]
        image = get_goal_image(goal_idx)
        block_idx = GOAL_TO_BLOCK[goal_idx]
        category = BlockType(block_idx).name.title()

        axs[i].imshow(image)
        if SAY_REUSE:
          is_test_object = goal_idx in test_objects
          reward = 1 if is_test_object else 0
          axs[i].set_title(
            f"{category}: {reward}",
            fontsize=10,
            color="green" if is_test_object else "black",
            weight="bold" if is_test_object else "normal",
          )
        else:
          axs[i].set_title(f"{category}")

        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].axis("off")

      fig.tight_layout()

    if eval:
      if EVAL_SHOW_MAP == 0:
        await wait_period()
        ui.markdown("**Note that you will NOT get the full map phase 2.**")
    else:
      await wait_period()
      ui.markdown("**Note that the agent is located at the black box in the full map**")
    await asyncio.sleep(1)


async def stage_instructions_display_fn(stage, container, new_world=False):
  if new_world:
    ########################################
    # First tell participant that entering a new world
    ########################################
    with container.style("align-items: center;"):
      nicewebrl.clear_element(container)
      ui.markdown("# You are entering a new mining world.")
      await asyncio.sleep(1)
      button = ui.button("click to start")
      await nicewebrl.wait_for_button_or_keypress(button, ignore_recent_press=True)

  ########################################
  # Then tell participant the task
  ########################################
  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)

    ui.markdown(f"## {stage.title}")
    if DEBUG:
      debug_info(stage)
    ui.markdown(f"{remove_extra_spaces(stage.body)}", extras=["cuddled-lists"])

    ui.markdown("Below are the stones you will need to mine.")
    if SAY_REUSE:
      ui.markdown("**We note the stone relevant to phase 2**")

    # Get all possible goals
    goals = [int(g) for g in possible_goals]

    # Create random display order
    key = nicewebrl.new_rng()
    order = jax.random.permutation(key, jnp.arange(len(goals)))

    test_objects = stage.metadata["block_metadata"]["test_objects"]
    test_objects = blocks_to_goals(test_objects)

    width = 1.5
    figsize = (len(goals) * width, width)
    with ui.matplotlib(figsize=figsize).figure as fig:
      axs = fig.subplots(1, len(goals))
      for i, idx in enumerate(order):
        goal_idx = goals[idx]
        image = get_goal_image(goal_idx)
        block_idx = GOAL_TO_BLOCK[goal_idx]
        category = BlockType(block_idx).name.title()

        axs[i].imshow(image)
        if SAY_REUSE:
          is_test_object = goal_idx in test_objects
          reward = 1 if is_test_object else 0
          axs[i].set_title(
            f"{category}: {reward}",
            fontsize=10,
            color="green" if is_test_object else "black",
            weight="bold" if is_test_object else "normal",
          )
        else:
          axs[i].set_title(f"{category}")

        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].axis("off")

      fig.tight_layout()

    ui.markdown("Please wait 3 seconds before continuing.")
    await asyncio.sleep(3)


async def env_reset_display_fn(
  stage: EnvStage,
  container: ui.element,
  timestep: nicewebrl.TimeStep,
):
  goal_object_idx = int(timestep.state.current_goal)
  image = get_goal_image(goal_object_idx)
  image = resize(image, (64, 64, 3), anti_aliasing=True, preserve_range=True).astype(
    np.uint8
  )
  image = base64_npimage(image)

  category = Achievement(goal_object_idx).name.replace("_", " ").title()

  remaining = get_remaining_goals()
  logger.info(f"remaining: {remaining}")

  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)
    ui.markdown(f"## {stage.title}")
    ui.markdown(f"#### Goal task: {category}")
    ui.html(make_image_html(src=image))
    button = ui.button("click to start")
    await nicewebrl.wait_for_button_or_keypress(button, ignore_recent_press=True)


def distance(x1, x2):
  # Euclidean distance
  return jnp.sqrt(jnp.sum((x1 - x2) ** 2, axis=0))


async def set_initial_timestep_from_training(
  stage: EnvStage,
) -> Optional[nicewebrl.TimeStep]:
  """Sets the initial timestep for an evaluation stage based on training data.

  This function:
  1. Loads training stage states
  2. Finds timesteps where agent started in target position
  3. Picks the timestep closest to the goal
  4. Updates stage params and state with this position

  Args:
      stage: The evaluation stage to set initial timestep for

  Returns:
      Optional[TimeStep]: The new timestep if successful, None if no training data found
  """
  current_name = stage.name  # e.g. "juncture_0_eval1"
  training_name = current_name.replace("_eval1", "_training")

  # Load training stage states
  logger.info(f"Loading data from {training_name}")
  training_stage_states = await nicewebrl.StageStateModel.filter(
    session_id=app.storage.browser["id"],
    name=training_name,
  ).all()
  logger.info(f"num training_stage_states: {len(training_stage_states)}")

  if len(training_stage_states) == 0:
    return None

  # Deserialize training stage states
  current_stage_state = stage.get_user_data("stage_state")
  training_stage_states = [
    serialization.from_bytes(current_stage_state, s.data) for s in training_stage_states
  ]
  timesteps = [s.timestep for s in training_stage_states]

  # Combine all timesteps from all stages
  all_timesteps = jtu.tree_map(lambda *v: jnp.stack(v), *timesteps)

  # Get timesteps where agent started in position we care about
  goal_start_position = stage.env_params.start_positions[0]  # [2]
  timestep_start_position = all_timesteps.state.start_position  # [N, 2]
  match = (timestep_start_position == goal_start_position[None]).sum(axis=-1) == 2
  relevant_timesteps = jax.tree_map(lambda t: t[match], all_timesteps)

  # For each timestep, compute distance to goal
  current_goal = current_stage_state.timestep.state.current_goal
  placed_blocks = stage.env_params.placed_goals
  placed_goals = jnp.array(blocks_to_goals(placed_blocks, default=-1), dtype=jnp.int32)
  goal_idx = (current_goal == placed_goals).argmax()
  goal_location = stage.env_params.goal_locations[goal_idx]

  distances = jax.vmap(distance, in_axes=(None, 0), out_axes=(0))(
    jnp.asarray(goal_location), relevant_timesteps.state.player_position
  )
  logger.info(f"current_goal: {current_goal}")
  logger.info(f"goal_location: {goal_location}")
  logger.info(f"player_locations: {relevant_timesteps.state.player_position}")
  logger.info(f"distances: {distances}")

  # Pick closest timestep as starting point
  sorted_indices = jnp.argsort(distances)
  closest_idx = sorted_indices[0]
  relevant_timestep_agent_pos = relevant_timesteps.state.player_position[closest_idx]
  logger.info(f"will spawn at position: {relevant_timestep_agent_pos}")

  # Set the initial params for the stage to have this as a start position
  stage.env_params = stage.env_params.replace(
    start_positions=make_start_position(relevant_timestep_agent_pos[None])
  )
  rng = nicewebrl.new_rng()
  new_timestep = stage.web_env.reset(rng, stage.env_params)
  await stage.set_user_data(stage_state=stage.state_cls(timestep=new_timestep))

  return new_timestep


async def env_reset_juncture_display_fn(
  stage: EnvStage,
  container: ui.element,
  timestep: nicewebrl.TimeStep,
):
  juncture_stage = "juncture" in stage.name
  eval1 = stage.metadata["condition"] == 1
  if juncture_stage and eval1:
    ui.markdown("# Loading task. One moment please")
    new_timestep = await set_initial_timestep_from_training(stage)
    if new_timestep is not None:
      timestep = new_timestep

  ############################
  # Display goal object image
  ############################
  goal_object_idx = int(timestep.state.current_goal)
  image = get_goal_image(goal_object_idx)
  image = resize(image, (64, 64, 3), anti_aliasing=True, preserve_range=True).astype(
    np.uint8
  )
  image = base64_npimage(image)

  category = Achievement(goal_object_idx).name.replace("_", " ").title()

  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)
    ui.markdown(f"## {stage.title}")
    ui.markdown(f"#### Goal task: {category}")
    ui.html(make_image_html(src=image))
    button = ui.button("click to start")
    await nicewebrl.wait_for_button_or_keypress(button, ignore_recent_press=True)


async def env_stage_display_fn(
  stage: EnvStage,
  container: ui.element,
  timestep: nicewebrl.TimeStep,
  display_full_map: bool = True,
):
  # Get partial observation image
  partial_obs_image = stage.render_fn(timestep)
  partial_obs_image = base64_npimage(partial_obs_image)

  # Get full map image from metadata
  world_seed = stage.env_params.world_seeds[0]
  full_map_image = fullmap_render(timestep, world_seed)
  full_map_image = base64_npimage(full_map_image)

  # Get goal object image
  goal_object_idx = int(timestep.state.current_goal)
  goal_image = get_goal_image(goal_object_idx)
  current_goal_name = Achievement(int(goal_object_idx)).name
  current_goal_name = current_goal_name.replace("_", " ").title()

  stage_state = stage.get_user_data("stage_state")

  with container.style("align-items: center;"):
    nicewebrl.clear_element(container)
    # ui.markdown(f"## {stage.title}")
    # ui.markdown(f"#### Goal task: {current_goal_name}")
    # Display goal object using matplotlib
    with ui.matplotlib(figsize=(1, 1)).figure as fig:
      ax = fig.subplots(1, 1)
      ax.set_title("Goal")
      ax.imshow(goal_image)
      ax.axis("off")
      fig.tight_layout()

    # Debug info if enabled
    if DEBUG:
      debug_info(stage)

    # Progress tracking row
    with ui.row():
      with ui.element("div").classes("p-2 bg-blue-100"):
        ui.label(
          f"Number of successful episodes: {stage_state.nsuccesses}/{int(stage.min_success)}"
        )
      with ui.element("div").classes("p-2 bg-green-100"):
        ui.label().bind_text_from(
          stage_state, "nepisodes", lambda n: f"Try: {n}/{stage.max_episodes}"
        )
    # Required episodes text
    text = f"You must complete at least {int(stage.min_success)} episodes. You have {stage.max_episodes} tries."
    ui.html(text).style("align-items: center;")
    # ui.html(make_image_html(src=partial_obs_image, id="stateImage"))
    # Side by side images using direct HTML with optimized sizing
    if display_full_map:
      ui.html(f"""
      <div id="stateImageContainer" style="display: flex; width: 100%; gap: 10px; justify-content: center; align-items: center; margin-top: 10px;">
        <div style="flex: 3; max-width: 58%;">
            <div style="text-align: center; margin-bottom: 5px;">Full Map</div>
            <img src="{full_map_image}" id="fullMapImage" style="width: 100%; height: auto; max-height: 60vh; object-fit: contain;">
        </div>
        <div style="flex: 2; max-width: 38%;">
            <div style="text-align: center; margin-bottom: 5px;">Current View</div>
            <img src="{partial_obs_image}" id="stateImage" style="width: 100%; height: auto; max-height: 60vh; object-fit: contain;">
        </div>
      </div>
      """)
    else:
      ui.html(f"""
      <div id="stateImageContainer" style="display: flex; width: 200%; gap: 10px; justify-content: center; align-items: center; margin-top: 10px;">
          <img src="{partial_obs_image}" id="stateImage" style="width: 200%; height: auto; max-height: 90vh; object-fit: contain;">
      </div>
      """)


def reduce_timestep_size(t: nicewebrl.TimeStep):
  state = t.state

  def make_uint(x):
    return x.astype(jnp.uint8)
    # return x

  new_state = state.replace(
    # remove all levels after 1st
    map=jax.tree_map(lambda t: make_uint(t[:1]), state.map),
    item_map=jax.tree_map(lambda t: make_uint(t[:1]), state.item_map),
    mob_map=jax.tree_map(lambda t: make_uint(t[:1]), state.mob_map),
    light_map=jax.tree_map(lambda t: make_uint(t[:1]), state.light_map),
  )
  return t.replace(
    observation=None,  # remove observation
    state=new_state,
  )


def make_env_stage(
  name: str,
  title: str,
  block_config: BlockConfig,
  stage_config: BlockStageConfig,
  metadata: dict,
  min_success: Optional[int] = None,
  max_episodes: Optional[int] = None,
):
  active_goals = blocks_to_active_goals(stage_config.goals)
  env_params = make_block_env_params(block_config, default_params)
  min_success = min_success or MIN_SUCCESS_TASK
  max_episodes = max_episodes or MAX_STAGE_EPISODES
  env_params = env_params.replace(
    active_goals=active_goals.astype(jnp.float32),
    start_positions=make_start_position(stage_config.start_positions),
    num_start_locations=len(stage_config.start_positions),
    num_success=min_success,
  )

  eval_stage = metadata.get("eval", False)
  if eval_stage:
    display_fn = partial(
      env_stage_display_fn,
      display_full_map=EVAL_SHOW_MAP,
    )
  else:
    display_fn = partial(
      env_stage_display_fn,
      display_full_map=True,
    )

  juncture_display = "juncture" in name
  if juncture_display and eval_stage:
    reset_display_fn = env_reset_juncture_display_fn
  else:
    reset_display_fn = env_reset_display_fn

  if VERBOSITY:
    print("=" * 30)
    print(f"Made stage {name} with config")
    print(stage_config)
    print("=" * 30)

  return EnvStage(
    name=name,
    title=title,
    web_env=jax_web_env,
    action_keys=action_keys,
    action_to_name=action_to_name,
    env_params=env_params,
    render_fn=render_fn,
    vmap_render_fn=vmap_render_fn,
    reset_display_fn=reset_display_fn,
    display_fn=display_fn,
    evaluate_success_fn=evaluate_success_fn,
    min_success=min_success * sum(active_goals),
    max_episodes=max_episodes,
    verbosity=VERBOSITY,
    user_save_file_fn=get_user_save_file_fn,
    autoreset_on_done=True,
    msg_display_time=100,
    metadata=metadata,
    preprocess_timestep=reduce_timestep_size,
    precompile=DEBUG == 0,
  )


#########################################################################
# Define stages of experiment
#########################################################################


instruct_text = """
  In this experiment, you will play a game where you are a traveling miner in a crafting world. In different episodes, you will need to obtain different stones. 

  There will be different worlds you can mine in. In each world, there will be two phases where you try to retrieve different objects.

  Be weary of monsters.
"""


def train_phase_text():
  phase_1_text = f"""
    Please learn to obtain the specified stone. You need to succeed {MIN_SUCCESS_TASK} times per specified stone.

    If you retrieve the wrong stone, the episode ends early.
    """
  return phase_1_text


def eval_phase_text(time=30):
  threshold = int(time * 2 / 3)
  phase_2_text = f"""
    You will get a <span style="color: green; font-weight: bold;">bonus</span> if you complete the task in less than <span style="color: green; font-weight: bold;">{int(threshold)}</span> seconds.
    """
  return phase_2_text


def make_block(
  block_config: BlockConfig,
  train_text: str,
  eval_text: str,
  train_config: BlockStageConfig,
  eval_config: BlockStageConfig,
  metadata: dict,
  eval2_config: Optional[BlockStageConfig] = None,
  name: Optional[str] = None,
  min_success: Optional[int] = None,
):
  """
  A block is defined by
  1. training stage instructions
  2. training stage environment
  3. evaluation stage instructions
  4. evaluation stage environment
  5. (optional) second evaluation stage environment
  """
  if VERBOSITY:
    print("=" * 50)
    print(f"Block: {name}")
    print("=" * 50)
  is_practice = "practice" in name

  if is_practice:
    train_display_fn = practice_stage_instructions_display_fn
    eval_display_fn = partial(practice_stage_instructions_display_fn, eval=True)

    def make_title(t):
      return f"(Practice) {t}"
  else:
    train_display_fn = eval_display_fn = stage_instructions_display_fn

    def make_title(t):
      return t

  train_stage_instructions = Stage(
    name=f"{name}_train_instructions",
    title=make_title("Phase 1 instructions"),
    body=train_text,
    display_fn=partial(train_display_fn, new_world=True),
  )

  train_stage_env = make_env_stage(
    name=f"{name}_training",
    title=make_title("Phase 1"),
    block_config=block_config,
    stage_config=train_config,
    min_success=min_success,
    metadata=dict(
      world_seed=train_config.world_seed,
      condition=0,
      eval=False,
    ),
  )

  eval_stage_instructions = Stage(
    name=f"{name}_eval_instructions",
    title=make_title("Phase 2 instructions"),
    body=eval_text,
    display_fn=eval_display_fn,
  )

  eval_stage_env = make_env_stage(
    name=f"{name}_eval1",
    title=make_title("Phase 2"),
    block_config=block_config,
    stage_config=eval_config,
    min_success=1,
    max_episodes=1,
    metadata=dict(
      world_seed=eval_config.world_seed,
      condition=1,
      eval=True,
    ),
  )

  stages = [
    train_stage_instructions,
    train_stage_env,
    eval_stage_instructions,
    eval_stage_env,
  ]
  randomize = []
  if eval2_config is not None:
    # If have 2 evals, randomize them
    randomize = [False, False, False, True, True]
    eval2_stage = make_env_stage(
      name=f"{name}_eval2",
      title=make_title("Phase 2"),
      block_config=block_config,
      stage_config=eval2_config,
      min_success=1,
      max_episodes=1,
      metadata=dict(
        world_seed=eval2_config.world_seed,
        condition=2,
        eval=True,
      ),
    )
    stages.append(eval2_stage)

  block = nicewebrl.Block(
    name=name,
    metadata=metadata,
    stages=stages,
    randomize=randomize,
  )
  return block


def metadata_from_config(config: BlockConfig):
  keys = [
    "world_seed",
    "start_train_positions",
    "start_eval_positions",
    "train_objects",
    "test_objects",
    "start_eval2_positions",
    "train_object_location",
    "test_object_location",
    "train_distractor_object_location",
  ]
  return {k: getattr(config, k) for k in keys}


####################
# practice block
####################
def make_practice_block():
  config = PRACTICE_BLOCK_CONFIG
  train_config = BlockStageConfig(
    world_seed=config.world_seed,
    start_positions=config.start_eval_positions,
    goals=config.train_objects,
  )

  eval_config = BlockStageConfig(
    world_seed=config.world_seed,
    start_positions=config.start_eval_positions,
    goals=config.test_objects,
  )
  return make_block(
    block_config=config,
    train_config=train_config,
    eval_config=eval_config,
    train_text=train_phase_text(),
    eval_text=eval_phase_text(10),
    min_success=1,
    name="practice",
    metadata=dict(
      manipulation="practice",
      desc="practice",
      long="practice",
      **metadata_from_config(config),
    ),
  )


def make_manipulation_block(
  config,
  manipulation: str,
  name: str,
  desc: str,
  long: str,
  train_text: str,
  eval_text: str,
):
  """Creates a manipulation block for either paths or juncture experiments.

  Args:
      config: Block configuration object containing world_seed, positions, objects etc.
      manipulation: String indicating experiment type
      name: Name of the block
      desc: Short description of the block
      long: Long description of the block
      train_text: Text to display during training phase
      eval_text: Text to display during evaluation phase
  """

  start_positions = config.start_train_positions + config.start_eval_positions

  train_config = BlockStageConfig(
    world_seed=config.world_seed,
    start_positions=start_positions,
    goals=config.train_objects,
  )

  eval_config = BlockStageConfig(
    world_seed=config.world_seed,
    start_positions=config.start_eval_positions,
    goals=config.test_objects,
  )

  eval2_config = None
  if config.start_eval2_positions:
    eval2_config = BlockStageConfig(
      world_seed=config.world_seed,
      start_positions=config.start_eval2_positions,
      goals=config.test_objects,
    )

  metadata = dict(
    manipulation=manipulation,
    desc=desc,
    long=long,
    name=name,
    **metadata_from_config(config),
  )

  return make_block(
    block_config=config,
    train_config=train_config,
    eval_config=eval_config,
    eval2_config=eval2_config,
    train_text=train_text,
    eval_text=eval_text,
    metadata=metadata,
    name=name,
  )


# Create experiment blocks with descriptions inline
if MANIPULATION == "paths":
  experiment_blocks = [
    make_manipulation_block(
      config=config,
      manipulation="paths",
      name=f"paths_{idx}",
      desc="reusing longer of two paths which matches training path",
      long="Here there are two paths to the test object. We predict that people will take the path that was used to get to the training object.",
      train_text=train_phase_text(),
      eval_text=eval_phase_text(),
    )
    for idx, config in enumerate(PATHS_CONFIGS)
  ]
elif MANIPULATION == "juncture":
  experiment_blocks = [
    make_manipulation_block(
      config=config,
      manipulation="juncture",
      name=f"juncture_{idx}",
      desc="probe behavior at juncture",
      long="Here there is a juncture to the test object. We predict that people will be faster at a juncture than at another point on the map.",
      train_text=train_phase_text(),
      eval_text=eval_phase_text(10),
    )
    for idx, config in enumerate(JUNCTURE_CONFIGS)
  ]

NUM_BLOCKS = min(NUM_BLOCKS, len(experiment_blocks))
experiment_blocks = experiment_blocks[:NUM_BLOCKS]

instruct_block = nicewebrl.Block(
  name="instructions",
  stages=[
    Stage(
      name="Experiment instructions",
      title="Experiment instructions",
      body=instruct_text,
      display_fn=experiment_instructions_display_fn,
    ),
  ],
  metadata=dict(desc="instructions", long="instructions"),
)

all_blocks = []
randomize = []
if GIVE_INSTRUCTIONS:
  all_blocks.extend([instruct_block, make_practice_block()])
  randomize.extend([False, False])

all_blocks.extend(experiment_blocks)
randomize.extend([True] * len(experiment_blocks))

experiment = nicewebrl.Experiment(
  blocks=all_blocks,
  randomize=randomize,
  name=f"craftax_experiment_{NAME}",
)

# all_stages = [stage for block in all_blocks for stage in block.stages]

###########################
## generating block order
###########################


# def generate_block_order(rng_key):
#  """Take blocks defined above and generate a random order"""
#  fixed_blocks = []
#  offset = 0
#  if GIVE_INSTRUCTIONS:
#    offset = 2
#  # fix ordering of instruct_block, practice_block
#  fixed_blocks.extend(list(range(offset)))
#  fixed_blocks = jnp.array(fixed_blocks)

#  # blocks afterward are randomized
#  randomized_blocks = list(all_blocks[offset:])
#  random_order = jax.random.permutation(rng_key, len(randomized_blocks)) + offset

#  block_order = jnp.concatenate(
#    [
#      fixed_blocks,  # instruction blocks
#      random_order,  # experiment blocks
#    ]
#  ).astype(jnp.int32)
#  block_order = block_order.tolist()
#  return [int(i) for i in block_order]
