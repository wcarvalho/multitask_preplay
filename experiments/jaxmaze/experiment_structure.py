import os
import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import List, Callable, Tuple
from functools import partial

from skimage.transform import resize
import matplotlib.pyplot as plt
import asyncio
from housemaze import renderer
from housemaze.env import KeyboardActions
from housemaze.human_dyna import utils
from housemaze.human_dyna import multitask_env as maze
from housemaze.human_dyna import multitask_env
from housemaze.human_dyna import web_env
from housemaze.human_dyna import mazes
import numpy as np

import jax
import jax.numpy as jnp
from flax import struct
from dotenv import load_dotenv
import os
from experiment_utils import SuccessTrackingAutoResetWrapper


from nicegui import ui, app
import nicewebrl
from nicewebrl import stages
from nicewebrl.stages import Stage, EnvStage, Block, FeedbackStage
from nicewebrl.nicejax import JaxWebEnv, base64_npimage, make_serializable
from nicewebrl.utils import wait_for_button_or_keypress, clear_element
from nicewebrl import nicejax

load_dotenv()

GIVE_INSTRUCTIONS = int(os.environ.get("INST", 1))
DEBUG = int(os.environ.get("DEBUG", 0))
NAME = os.environ.get("NAME", "exp")
MAN = os.environ.get("MAN", "paths")  # which manipulation
DATA_DIR = os.environ.get("DATA_DIR", "data")
DATA_DIR = os.path.join(os.path.dirname(__file__), DATA_DIR)

# USE_REVERSALS = int(os.environ.get('REV', 0))
# EVAL_OBJECTS = int(os.environ.get('EVAL_OBJECTS', 1))
FEEDBACK = int(os.environ.get("FEEDBACK", 0))
SAY_REUSE = int(os.environ.get("SAY_REUSE", 1))
COND2_TRAIN = int(os.environ.get("COND2_TRAIN", 1))
TIMER = int(os.environ.get("TIMER", 0))
VERBOSITY = int(os.environ.get("VERBOSITY", 0))
NTRAIN = int(os.environ.get("NTRAIN", 8))
TIME_LIMIT = int(os.environ.get("TIME_LIMIT", 10_000_000))
USE_DONE = DEBUG > 0


# number of rooms to user for tasks (1st n)
num_rooms = 2


min_success_task = NTRAIN
min_success_train = min_success_task * num_rooms
max_episodes_task = 50
if DEBUG == 0:
  pass
elif DEBUG == 1:
  max_episodes_train = NTRAIN * num_rooms
  max_episodes_task = NTRAIN
elif DEBUG == 2:
  min_success_task = 2
  min_success_train = 2
  max_episodes_task = 2
  max_episodes_train = 2

max_episodes_train = max_episodes_task * num_rooms


def reversal_label(reversal):
  reversal = tuple(reversal)
  if reversal == (False, False):
    return "F,F"
  elif reversal == (True, False):
    return "T,F"
  elif reversal == (False, True):
    return "F,T"
  elif reversal == (True, True):
    return "T,T"
  else:
    raise ValueError(f"reversal: {reversal}")


def get_user_save_file_fn():
  return (
    f"{DATA_DIR}/user={app.storage.user.get('seed')}_name={NAME}_debug={DEBUG}.json"
  )


##############################################
# Creating environment stuff
##############################################
image_data = utils.load_image_dict()


def create_env_params(
  maze_str,
  groups,
  char2idx,
  randomize_agent=False,
  use_done=False,
  training=True,
  force_room=False,
  label=0,
  time_limit=TIME_LIMIT,
  default_room=0,
  p_test_sample_train=1.0,
):
  env_params = mazes.get_maze_reset_params(
    groups=groups,
    char2key=char2idx,
    maze_str=maze_str,
    label=jnp.array(label),
    make_env_params=True,
    randomize_agent=randomize_agent,
  ).replace(
    time_limit=time_limit,
    terminate_with_done=jnp.array(2) if use_done else jnp.array(0),
    randomize_agent=randomize_agent,
    training=training,
    force_room=jnp.array(force_room or not training),
    default_room=jnp.array(default_room),
    p_test_sample_train=p_test_sample_train,
  )
  return env_params


def permute_groups(groups):
  # Flatten the groups
  flattened = groups.flatten()

  # Create a random permutation
  permutation = np.random.permutation(len(flattened))

  # Apply the permutation
  permuted_flat = flattened[permutation]

  # Reshape back to the original shape
  new_groups = permuted_flat.reshape(groups.shape)

  # Create a new char2idx mapping
  new_char2idx = mazes.groups_to_char2key(new_groups)

  return new_groups, new_char2idx


def housemaze_render_fn(
  timestep: maze.TimeStep, include_objects: bool = True
) -> jnp.ndarray:
  image = renderer.create_image_from_grid(
    timestep.state.grid,
    timestep.state.agent_pos,
    timestep.state.agent_dir,
    image_data,
    include_objects=include_objects,
  )
  return image


image_keys = image_data["keys"]
groups = [
  # room 1
  [image_keys.index("orange"), image_keys.index("potato")],
  # room 2
  [image_keys.index("lettuce"), image_keys.index("apple")],
  # room 3
  # [image_keys.index('tomato'), image_keys.index('lettuce')],
]
groups = np.array(groups, dtype=np.int32)
task_objects = groups.reshape(-1)

# can auto-generate this from group_set
block_char2idx = mazes.groups_to_char2key(groups)

# shared across all tasks
task_runner = multitask_env.TaskRunner(task_objects=task_objects)
keys = image_data["keys"]

jax_env = web_env.HouseMaze(
  task_runner=task_runner,
  num_categories=len(keys),
  use_done=USE_DONE,
)
jax_env = SuccessTrackingAutoResetWrapper(jax_env, num_success=min_success_task)

actions = [
  KeyboardActions.right,
  KeyboardActions.down,
  KeyboardActions.left,
  KeyboardActions.up,
  KeyboardActions.done,
]
action_array = jnp.array([a.value for a in actions])
action_keys = ["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp", "d"]
action_to_name = [a.name for a in actions]

jax_web_env = JaxWebEnv(jax_env)
# Call this function to pre-compile jax functions before experiment starts.
dummy_env_params = create_env_params(
  maze_str=mazes.big_practice_maze, groups=groups, char2idx=block_char2idx
)
dummy_timestep = jax_web_env.reset(jax.random.PRNGKey(42), dummy_env_params)
jax_web_env.precompile(dummy_env_params=dummy_env_params)


render_fn = jax.jit(housemaze_render_fn)
vmap_render_fn = jax_web_env.precompile_vmap_render_fn(render_fn, dummy_env_params)


def went_to_junction(timestep, junction):
  position = timestep.state.agent_pos
  match = np.array(junction) == position
  match = match.sum(-1) == 2  # both x and y matches
  return match.any()


def manip1_data_fn(timestep):
  old_path = went_to_junction(timestep, junction=(2, 14))
  return {
    "old_path": old_path,
  }


def manip3_data_fn(timestep):
  return {
    "old_path": went_to_junction(timestep, junction=(14, 25)),
    "new_path": went_to_junction(timestep, junction=(3, 11)),
  }


##############################################
# Block/Stage Utility functions
##############################################
def remove_extra_spaces(text):
  """For each line, remove extra space."""
  return "\n".join([i.strip() for i in text.strip().split("\n")])


def debug_info(stage):
  stage_state = stage.get_user_data("stage_state")
  debug_info = (
    f"**Manipulation**: {stage.metadata['block_metadata'].get('manipulation')}. "
  )
  if stage_state is not None:
    debug_info += f"**Eval**: {stage.metadata['eval']}. "
    debug_info += f"**Episode** idx: {stage_state.nepisodes}. "
    debug_info += f"**Step**: {stage_state.nsteps}/{stage.env_params.time_limit}. "
  return debug_info


async def experiment_instructions_display_fn(stage, container):
  with container.style("align-items: center;"):
    clear_element(container)
    ui.markdown(f"## {stage.name}")
    ui.markdown(f"{remove_extra_spaces(stage.body)}", extras=["cuddled-lists"])
    ui.markdown("Task objects will be selected from the set below.")

    cats = [int(i) for i in block_char2idx.values()]
    width = 1
    figsize = (len(cats) * width, width)
    with ui.matplotlib(figsize=figsize).figure as fig:
      axs = fig.subplots(1, len(cats))
      for i, cat in enumerate(cats):
        axs[i].imshow(image_data["images"][cat])
        axs[i].set_title(f"{keys[cat]}")
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].axis("off")
      # Adjust layout
      fig.tight_layout()


async def stage_instructions_display_fn(stage, container):
  with container.style("align-items: center;"):
    clear_element(container)
    ui.markdown(f"## {stage.title}")
    if DEBUG:
      ui.markdown(debug_info(stage))
    ui.markdown(f"{remove_extra_spaces(stage.body)}", extras=["cuddled-lists"])

    await asyncio.sleep(1)
    ui.markdown("Task objects will be selected from the set below.")
    if SAY_REUSE:
      ui.markdown("**We note objects relevant to phase 2**")
    # phase 2 reward
    # idxs = char2idx.values()
    groups = stage.metadata["block_metadata"].get("groups", None)
    cats = groups[0] + groups[1]
    eval_prices = [0, 1, 0, 0]
    key = nicejax.new_rng()
    order = jax.random.permutation(key, jnp.arange(len(cats)))

    width = 1
    figsize = (len(cats) * width, width)
    with ui.matplotlib(figsize=figsize).figure as fig:
      axs = fig.subplots(1, len(order))
      for i, idx in enumerate(order):
        cat = cats[idx]
        axs[i].imshow(image_data["images"][cat])
        if SAY_REUSE:
          axs[i].set_title(
            (
              f"{keys[cat]}: {eval_prices[idx]}"
              if eval_prices[idx] == 0
              else f"{keys[cat]}: {eval_prices[idx]}"
            ),
            fontsize=10,
            color="green" if eval_prices[idx] != 0 else "black",
            weight="bold" if eval_prices[idx] != 0 else "normal",
          )
        else:
          axs[i].set_title(f"{keys[cat]}")
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].axis("off")
      # Adjust layout
      fig.tight_layout()
  # ui.markdown(f"Please wait {3} seconds before starting.")
  await asyncio.sleep(3)


def make_image_html(src):
  html = """
    <div id="stateImageContainer" style="display: flex; justify-content: center; align-items: center; height: 100%;">
        <img id="stateImage" src="{src}" style="max-width: 900px; max-height: 450px; object-fit: contain;">
    </div>
    """.format(src=src)
  return html


async def env_reset_display_fn(
  stage,
  container,
  timestep,
  pause: int = 0,
):
  category = keys[timestep.state.task_object]
  image = image_data["images"][timestep.state.task_object]
  image = resize(image, (64, 64, 3), anti_aliasing=True, preserve_range=True).astype(
    np.uint8
  )
  image = base64_npimage(image)

  with container.style("align-items: center;"):
    clear_element(container)
    ui.markdown(f"#### Goal object: {category}")
    if pause > 0:
      ui.markdown(f"Please wait {pause} seconds before starting.")
      await asyncio.sleep(pause)
      clear_element(container)
      ui.markdown(f"#### Goal object: {category}")
    if DEBUG:
      ui.markdown(debug_info(stage))
    ui.html(make_image_html(src=image))
    button = ui.button("click to start")
    await wait_for_button_or_keypress(button, ignore_recent_press=True)


async def env_stage_display_fn(stage, container, timestep):
  state_image = stage.render_fn(timestep)
  state_image = base64_npimage(state_image)
  # category = keys[timestep.state.task_object]

  object_image = image_data["images"][timestep.state.task_object]

  stage_state = stage.get_user_data("stage_state")
  with container.style("align-items: center;"):
    clear_element(container)
    # ui.markdown(f"#### Goal object: {category}")
    with ui.matplotlib(figsize=(1, 1)).figure as fig:
      ax = fig.subplots(1, 1)
      ax.set_title("Goal")
      ax.imshow(object_image)
      ax.axis("off")
      fig.tight_layout()
    if DEBUG:
      ui.markdown(debug_info(stage))
    with ui.row():
      with ui.element("div").classes("p-2 bg-blue-100"):
        n = timestep.state.successes.sum()
        ui.label(f"Number of successful episodes: {n}/{stage.min_success}")
      with ui.element("div").classes("p-2 bg-green-100"):
        ui.label().bind_text_from(
          stage_state, "nepisodes", lambda n: f"Try: {n}/{stage.max_episodes}"
        )

    text = f"You must complete at least {stage.min_success} episodes. You have {stage.max_episodes} tries."
    ui.html(text).style("align-items: center;")
    ui.html(make_image_html(src=state_image))


async def train_eval_feedback_display_fn(
  stage,
  container,
  stages_for_feedback: List[Tuple[Stage, Stage]],
  question: str,
):
  del stage  # won't use
  output = {}
  with container.style("align-items: center;"):
    rng = nicejax.new_rng()
    for train_stage, eval_stage in stages_for_feedback:
      clear_element(container)
      train_timestep = train_stage.web_env.reset(rng, train_stage.env_params)
      train_image = train_stage.render_fn(train_timestep)

      eval_timestep = eval_stage.web_env.reset(rng, eval_stage.env_params)
      eval_image = eval_stage.render_fn(eval_timestep)
      eval_category = keys[eval_timestep.state.task_object]

      # Calculate aspect ratio and set figure size
      fig_width = 12
      fig_height = 4
      with ui.matplotlib(figsize=(int(fig_width), int(fig_height))).figure as fig:
        axs = fig.subplots(1, 2)
        axs[0].set_title("Phase 1 map")
        axs[0].imshow(train_image)
        axs[0].axis("off")
        axs[1].set_title(f"Phase 2 object: {eval_category}")
        axs[1].imshow(eval_image)
        axs[1].axis("off")
      ui.html(question)
      radio = ui.radio({1: "Yes", 2: "No"}).props("inline")

      async def submit():
        if radio.value is None:
          ui.notify("Please select an option before submitting.", type="warning")
          return

        noticed_difference = "Yes" if radio.value == 1 else "No"
        reversal = train_stage.metadata["block_metadata"]["reversal"]
        output[f"noticed_cond={reversal}"] = noticed_difference

      button = ui.button("Submit", on_click=submit)
      await button.clicked()
  return output


def make_env_stage(
  maze_str,
  groups,
  char2idx,
  max_episodes=1,
  min_success=1,
  training=True,
  default_room=0,
  force_room=False,
  metadata=None,
  randomize_agent: bool = True,
  custom_data_fn=None,
  duration=None,
  name="stage",
  use_done=False,
  force_random_room: bool = False,
  pause: int = 0,
  **kwargs,
):
  metadata = metadata or {}
  metadata["eval"] = not training

  randomize_agent = randomize_agent and training

  env_params = create_env_params(
    groups=groups,
    char2idx=char2idx,
    maze_str=maze_str,
    randomize_agent=randomize_agent,
    use_done=use_done,
    training=training,
    force_room=force_room,
    default_room=default_room,
  )

  if force_random_room:
    env_params = env_params.replace(force_room=jnp.array(False))

  return EnvStage(
    name=name,
    web_env=jax_web_env,
    action_keys=action_keys,
    action_to_name=action_to_name,
    env_params=env_params,
    render_fn=render_fn,
    vmap_render_fn=vmap_render_fn,
    reset_display_fn=partial(env_reset_display_fn, pause=pause),
    display_fn=env_stage_display_fn,
    evaluate_success_fn=lambda t, params: int(t.reward > 0.5),
    check_finished=lambda t: t.finished,
    # state_cls=EnvStageState,
    max_episodes=max_episodes,
    min_success=min_success,
    metadata=metadata,
    custom_data_fn=custom_data_fn,
    duration=duration if not training else None,
    notify_success=True,
    verbosity=DEBUG > 1,
    **kwargs,
  )


def make_block(
  phase_1_text: str,
  phase_1_maze_name: str,
  phase_2_text: str,
  phase_2_cond1_maze_name: str,
  block_groups: np.ndarray,
  block_char2idx: dict,
  eval_duration: int,
  metadata: dict,
  min_success: int = None,
  max_episodes: int = None,
  phase_2_cond2_maze_name: str = None,
  phase_2_cond1_name: str = None,
  phase_2_cond2_name: str = None,
  make_env_kwargs: dict = None,
  phase2_cond1_env_kwargs: dict = None,
  phase2_cond2_env_kwargs: dict = None,
  str_transform: Callable[[str], str] = lambda s: s,
  appendix: str = "",
):
  block_name = metadata.get("desc", "block")

  def create_stage(name, title, body):
    return Stage(
      name=name, title=title, body=body, display_fn=stage_instructions_display_fn
    )

  make_env_kwargs = make_env_kwargs or {}
  phase2_cond1_env_kwargs = phase2_cond1_env_kwargs or {}
  phase2_cond2_env_kwargs = phase2_cond2_env_kwargs or {}

  def create_env_stage(
    name,
    maze_name,
    training,
    min_success,
    max_episodes,
    duration=None,
    extra_apppendix="",
    **kwargs,
  ):
    all_kwargs = dict(
      name=f"{name}_{appendix}_{extra_apppendix}",
      maze_str=str_transform(getattr(mazes, maze_name)),
      min_success=min_success,
      max_episodes=max_episodes,
      duration=duration,
      groups=block_groups,
      training=training,
      user_save_file_fn=get_user_save_file_fn,
    )
    all_kwargs.update(make_env_kwargs)
    all_kwargs.update(kwargs)
    return make_env_stage(**all_kwargs)

  phase2_cond1_kwargs = dict(
    name=phase_2_cond1_name or phase_2_cond1_maze_name,
    extra_apppendix="train",
    maze_name=phase_2_cond1_maze_name,
    metadata=dict(maze=phase_2_cond1_maze_name + appendix, condition=1),
    training=False,
    min_success=1,
    max_episodes=1,
    duration=eval_duration if TIMER else None,
    # end_on_final_timestep=True,
    pause=3,
    char2idx=block_char2idx,
  )
  phase2_cond1_kwargs.update(phase2_cond1_env_kwargs)
  stages = [
    create_stage(name=f"{block_name} Phase 1", title="Phase 1", body=phase_1_text),
    create_env_stage(
      name=phase_1_maze_name,
      extra_apppendix="eval1",
      maze_name=phase_1_maze_name,
      metadata=dict(maze=phase_1_maze_name + appendix, condition=0),
      training=True,
      char2idx=block_char2idx,
      min_success=min_success or min_success_train,
      max_episodes=max_episodes or max_episodes_train,
    ),
    create_stage(name=f"{block_name} Phase 2", title="Phase 2", body=phase_2_text),
    create_env_stage(**phase2_cond1_kwargs),
  ]
  randomize = []
  if phase_2_cond2_maze_name is not None:
    randomize = [False, False, False, True, True]
    phase2_cond2_kwargs = dict(
      name=phase_2_cond2_name or phase_2_cond2_maze_name,
      extra_apppendix="eval2",
      maze_name=phase_2_cond2_maze_name,
      metadata=dict(maze=phase_2_cond2_maze_name + appendix, condition=2),
      training=False,
      min_success=1,
      max_episodes=1,
      duration=eval_duration if TIMER else None,
      # end_on_final_timestep=True,
      char2idx=block_char2idx,
      pause=3,
    )
    phase2_cond2_kwargs.update(phase2_cond2_env_kwargs)
    stages.append(create_env_stage(**phase2_cond2_kwargs))

  block = Block(
    metadata=dict(
      **metadata,
      groups=make_serializable(block_groups),
      char2idx=jax.tree_map(int, block_char2idx),
    ),
    stages=stages,
    randomize=randomize,
  )
  return block


##############################################
# Create blocks
##############################################

if SAY_REUSE:
  instruct_text = """
    This experiment tests how effectively people can learn about goals before direct experience on them.

    It will consist of blocks with two phases each: **one** where you navigate to objects, and **another** where you navigate to other objects that you could have learned about previously.
  """

  def make_phase_2_text(time=30, include_time=True):
    time_str = (
      f' of <span style="color: red; font-weight: bold;">{time}</span> seconds'
      if include_time
      else ""
    )
    threshold = int(time * 2 / 3)
    phase_2_text = f"""
      You will get a <span style="color: green; font-weight: bold;">bonus</span> if you complete the task in less than <span style="color: green; font-weight: bold;">{int(threshold)}</span> seconds. 
      """
    phase_2_text += (
      f"""
      You have a <span style="color: red; font-weight: bold;">time-limit</span>{time_str}.
      """
      if TIMER
      else ""
    )
    if MAN == "shortcut":
      phase_2_text += """
        <p style="color: red;"><strong>Note that some parts of the maze have changed</strong>.</p>
        """
    phase_2_text += """
      If you retrieve the wrong object, the episode ends early. You have 1 try.

      """
    return phase_2_text

else:
  instruct_text = """
          This experiment tests how people learn to navigate maps.

          It will consist of blocks with two phases each: **one** where you navigate to objects, and **another** where you navigate to other objects.
  """

  def make_phase_2_text(time=30, include_time=True):
    time_str = (
      f' of <span style="color: red; font-weight: bold;">{time}</span> seconds'
      if include_time
      else ""
    )
    threshold = int(time * 2 / 3)
    phase_2_text = f"""
      You will get a <span style="color: green; font-weight: bold;">bonus</span> if you complete the task in less than <span style="color: green; font-weight: bold;">{int(threshold)}</span> seconds.
      """
    phase_2_text += (
      f"""
      You have a <span style="color: red; font-weight: bold;">time-limit</span>{time_str}.
      """
      if TIMER
      else ""
    )
    if MAN == "shortcut":
      phase_2_text += """
        <p style="color: red;"><strong>Note that some parts of the maze have changed</strong>.</p>
        """
    phase_2_text += """
      If you retrieve the wrong object, the episode ends early. You have 1 try.
      """
    return phase_2_text


def make_phase_1_text():
  phase_1_text = f"""
    Please learn to obtain these objects. You need to succeed {min_success_task} times per object.

    If you retrieve the wrong object, the episode ends early.
    """
  return phase_1_text


####################
# practice block
####################
def create_practice_block(reversal: Tuple[bool, bool] = [False, False]):
  str_transform = partial(mazes.reverse, horizontal=reversal[0], vertical=reversal[1])
  block_groups, block_char2idx = permute_groups(groups)
  return make_block(
    eval_duration=TIMER,
    min_success=2 if not DEBUG else 2,
    max_episodes=10 if not DEBUG else 2,
    make_env_kwargs=dict(force_room=True),
    phase_1_text=make_phase_1_text(),
    phase_1_maze_name="big_practice_maze",
    phase_2_text=make_phase_2_text(10),
    phase_2_cond1_maze_name="big_practice_maze",
    block_groups=block_groups,
    block_char2idx=block_char2idx,
    metadata=dict(manipulation=-1, desc="practice", long="practice"),
    str_transform=str_transform,
    appendix=f"_({reversal_label(reversal)})",
  )


####################
# (1) Shortcut manipulation
####################
def create_shortcut_manipulation_block(reversal: Tuple[bool, bool] = [False, False]):
  str_transform = partial(mazes.reverse, horizontal=reversal[0], vertical=reversal[1])
  block_groups, block_char2idx = permute_groups(groups)
  return make_block(
    phase_1_text=make_phase_1_text(),
    phase_1_maze_name="big_m1_maze3",
    phase_2_text=make_phase_2_text(),
    phase_2_cond1_maze_name="big_m1_maze3_shortcut",
    block_groups=block_groups,
    block_char2idx=block_char2idx,
    eval_duration=TIMER,
    make_env_kwargs=dict(custom_data_fn=manip1_data_fn),
    metadata=dict(
      manipulation=1,
      reversal=reversal,
      desc="shortcut",
      long="A shortcut is introduced",
      short=f"shortcut_{reversal_label(reversal)}",
    ),
    str_transform=str_transform,
    appendix=f"_({reversal_label(reversal)})",
  )


####################
# (3) paths manipulation: reusing longer of two paths matching training path
####################
def create_path_manipulation_block(reversal: Tuple[bool, bool] = [False, False]):
  str_transform = partial(mazes.reverse, horizontal=reversal[0], vertical=reversal[1])
  block_groups, block_char2idx = permute_groups(groups)
  return make_block(
    phase_1_text=make_phase_1_text(),
    phase_1_maze_name="big_m3_maze1",
    phase_2_text=make_phase_2_text(),
    phase_2_cond1_maze_name="big_m3_maze1",
    phase_2_cond1_name="big_m3_maze1_eval",
    block_groups=block_groups,
    block_char2idx=block_char2idx,
    eval_duration=TIMER,
    make_env_kwargs=dict(custom_data_fn=manip3_data_fn),
    metadata=dict(
      manipulation=3,
      reversal=reversal,
      desc="reusing longer of two paths which matches training path",
      long="Here there are two paths to the test object. We predict that people will take the path that was used to get to the training object.",
      short=f"paths_{reversal_label(reversal)}",
    ),
    str_transform=str_transform,
    appendix=f"_({reversal_label(reversal)})",
  )


####################
# (2) Start manipulation: Faster when on-path but further than off-path but closer
####################
def create_start_manipulation_block(reversal: Tuple[bool, bool] = [False, False]):
  str_transform = partial(mazes.reverse, horizontal=reversal[0], vertical=reversal[1])
  block_groups, block_char2idx = permute_groups(groups)

  kwargs = dict()
  if COND2_TRAIN:
    kwargs["phase2_cond2_env_kwargs"] = dict(
      # force to focus on train object from "1st room"
      force_room=True,
      default_room=0,
      training=True,
    )
  return make_block(
    phase_1_text=make_phase_1_text(),
    phase_1_maze_name="big_m2_maze2",
    phase_2_text=make_phase_2_text(),
    phase_2_cond1_maze_name="big_m2_maze2_onpath",
    phase_2_cond2_maze_name="big_m2_maze2_offpath",
    block_groups=block_groups,
    block_char2idx=block_char2idx,
    eval_duration=TIMER,
    metadata=dict(
      manipulation=2,
      reversal=reversal,
      desc="faster when on-path but further than off-path but closer",
      long="In the first, the agent is tested with starting in a familiar location. In the second, the agent is started from a different, but closer part of the path.",
      short=f"start_{reversal_label(reversal)}",
    ),
    str_transform=str_transform,
    appendix=f"_({reversal_label(reversal)})",
    **kwargs,
  )


####################
# (4) planning manipulation (short)
####################
def create_plan_manipulation_block(
  reversal: Tuple[bool, bool] = [False, False],
  setting: str = "short",
):
  str_transform = partial(mazes.reverse, horizontal=reversal[0], vertical=reversal[1])
  block_groups, block_char2idx = permute_groups(groups)
  return make_block(
    # special case for short planning maze
    min_success=min_success_task,
    max_episodes=max_episodes_task,
    eval_duration=5 if setting == "short" else 15,
    make_env_kwargs=dict(default_room=0, force_room=True),
    phase2_cond1_env_kwargs={} if SAY_REUSE else dict(force_random_room=True),
    phase2_cond2_env_kwargs=dict(
      # force to focus on train object from "2nd room"
      force_room=True,
      default_room=1,
      training=True,
    ),
    # regular commands
    phase_1_text=make_phase_1_text(),
    phase_1_maze_name=(
      f"big_m4_maze_{setting}" if SAY_REUSE else f"big_m4_maze_{setting}_blind"
    ),
    phase_2_text=make_phase_2_text(time=5 if setting == "short" else 15),
    phase_2_cond1_maze_name=(
      f"big_m4_maze_{setting}_eval_same"
      if SAY_REUSE
      else f"big_m4_maze_{setting}_eval_same_blind"
    ),
    phase_2_cond2_maze_name=(
      f"big_m4_maze_{setting}_eval_diff"
      if SAY_REUSE
      else f"big_m4_maze_{setting}_eval_diff_blind"
    ),
    block_groups=block_groups,
    block_char2idx=block_char2idx,
    metadata=dict(
      manipulation=4,
      reversal=reversal,
      desc=f"See if faster off train path than planning ({setting})",
      long="Here there are two branches from a training path. We predict that people will have a shorter response time when an object is in the same location it was in phase 1.",
      short=f"plan_{setting}_{reversal_label(reversal)}",
    ),
    str_transform=str_transform,
    appendix=f"_({reversal_label(reversal)})",
  )


##########################
# Combining all together
##########################

reversals = [(False, False), (True, False), (False, True), (True, True)]
if DEBUG > 1:
  reversals = [(False, False)]

feedback_block = []
if MAN == "start":  # start manipulation (2)
  manipulations = [create_start_manipulation_block(r) for r in reversals]
elif MAN == "paths":  # paths manipulation (3)
  manipulations = [create_path_manipulation_block(r) for r in reversals]
  stages_for_feedback = []
  for block in manipulations:
    train_stage = block.stages[1]
    eval_stage = block.stages[3]
    stages_for_feedback.append((train_stage, eval_stage))

  if FEEDBACK:
    display_fn = partial(
      train_eval_feedback_display_fn,
      stages_for_feedback=stages_for_feedback,
      question="Did you notice both paths to the phase 2 object?",
    )
    feedback_block.append(
      Block(
        stages=[
          FeedbackStage(
            user_save_file_fn=get_user_save_file_fn,
            name="paths_manipulation_feedback",
            display_fn=display_fn,
          )
        ],
        metadata=dict(desc="paths_manipulation_feedback"),
      )
    )
elif MAN == "plan":  # planning manipulation (4)
  manipulations = [create_plan_manipulation_block(r, "short") for r in reversals] + [
    create_plan_manipulation_block(r, "long") for r in reversals
  ]
elif MAN == "shortcut":  # shortcut manipulation (1)
  manipulations = [create_shortcut_manipulation_block(r) for r in reversals]
  stages_for_feedback = []
  for block in manipulations:
    train_stage = block.stages[1]
    eval_stage = block.stages[3]
    stages_for_feedback.append((train_stage, eval_stage))

  if FEEDBACK:
    display_fn = partial(
      train_eval_feedback_display_fn,
      stages_for_feedback=stages_for_feedback,
      question="Did you notice that the map from phase 2 was different from the map in phase 1?",
    )
    feedback_block.append(
      Block(
        stages=[
          FeedbackStage(
            user_save_file_fn=get_user_save_file_fn,
            name="shortcut_manipulation_feedback",
            display_fn=display_fn,
          )
        ],
        metadata=dict(desc="shortcut_manipulation_feedback"),
      )
    )
else:
  raise NotImplementedError


instruct_block = Block(
  stages=[
    Stage(
      name="Experiment instructions",
      body=instruct_text,
      display_fn=experiment_instructions_display_fn,
    ),
  ],
  metadata=dict(desc="instructions", long="instructions"),
)

all_blocks = []
# randomize = []
if GIVE_INSTRUCTIONS:
  all_blocks.extend([instruct_block, create_practice_block()])
  # randomize.extend([False, False])

all_blocks.extend(manipulations + feedback_block)
# randomize.extend([True] * len(manipulations + feedback_block))

# experiment = nicewebrl.Experiment(
#  blocks=all_blocks,
#  randomize=randomize,
#  name=f'jaxmaze_experiment_{NAME}',
# )
all_stages = stages.prepare_blocks(all_blocks)

##########################
# generating stage order
##########################


def generate_block_stage_order(rng_key):
  """Take blocks defined above, flatten all their stages, and generate an order where the (1) blocks are randomized, and (2) stages within blocks are randomized if they're consecutive eval stages."""
  fixed_blocks = []
  offset = 0
  nfeedback = len(feedback_block)
  if GIVE_INSTRUCTIONS:
    offset = 2
  # fix ordering of instruct_block, practice_block
  fixed_blocks.extend(list(range(offset)))
  fixed_blocks = jnp.array(fixed_blocks)

  # blocks afterward are randomized
  if nfeedback > 0:
    randomized_blocks = list(all_blocks[offset:-nfeedback])
  else:
    randomized_blocks = list(all_blocks[offset:])
  random_order = jax.random.permutation(rng_key, len(randomized_blocks)) + offset

  if nfeedback > 0:
    n = len(fixed_blocks) + len(randomized_blocks)
    block_order = jnp.concatenate(
      [
        fixed_blocks,  # instruction blocks
        random_order,  # experiment blocks
        np.arange(n, n + nfeedback, dtype=np.int32),
      ]
    ).astype(jnp.int32)
  else:
    block_order = jnp.concatenate(
      [
        fixed_blocks,  # instruction blocks
        random_order,  # experiment blocks
      ]
    ).astype(jnp.int32)
  block_order = block_order.tolist()
  stage_order = stages.generate_stage_order(all_blocks, block_order, rng_key)
  stage_order = [int(i) for i in stage_order]
  return block_order, stage_order
