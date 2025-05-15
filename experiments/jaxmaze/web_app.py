import inspect
import collections
import asyncio
import aiofiles
import subprocess

from dotenv import load_dotenv
import jax.numpy as jnp
from nicegui import app, ui
from fastapi import Request
from tortoise import Tortoise
import os


from experiments.gcs import save_data_to_gcs
from experiments.gcs import save_to_gcs_with_retries
import nicewebrl
import nicewebrl.nicejax
import nicewebrl.stages
import nicewebrl.utils
from nicewebrl.stages import EnvStage
from nicewebrl.utils import wait_for_button_or_keypress, clear_element
from nicewebrl.logging import setup_logging, get_logger

from asyncio import Lock

load_dotenv()

DATABASE_FILE = os.environ.get("DB_FILE", "db.sqlite")
DATA_DIR = os.environ.get("DATA_DIR", "jaxmaze_data")
NAME = os.environ.get("NAME", "exp")

DEBUG = int(os.environ.get("DEBUG", 0))
DEBUG_SEED = int(os.environ.get("SEED", 0))
EXPERIMENT = int(os.environ.get("EXP", 4))
UPLOAD_DATA = int(os.environ.get("UPLOAD_DATA", 1))
os.makedirs(DATA_DIR, exist_ok=True)


def log_filename_fn(log_dir, user_id):
  return os.path.join(log_dir, f"log_{user_id}.log")


def get_date_filename(data_dir, user_id):
  return os.path.join(data_dir, f"log_{user_id}.log")


_user_locks = {}


def get_user_lock():
  user_seed = app.storage.user["seed"]
  if user_seed not in _user_locks:
    _user_locks[user_seed] = Lock()
  return _user_locks[user_seed]


def blob_user_filename():
  """filename structure for user data in GCS (cloud)"""
  seed = app.storage.user["seed"]
  worker = app.storage.user.get("worker_id", None)
  if worker is not None:
    return f"user={seed}_worker={worker}_name={NAME}_debug={DEBUG}"
  else:
    return f"user={seed}_name={NAME}_debug={DEBUG}"


setup_logging(
  DATA_DIR, log_filename_fn=log_filename_fn, nicegui_storage_user_key="user_id"
)
logger = get_logger("main")

import experiment_structure as experiment

APP_TITLE = "Dyna 4"
all_stages = experiment.all_stages

DATABASE_FILE = f"{DATABASE_FILE}_name={NAME}_debug={DEBUG}"

#####################################
# Consent Form
#####################################


async def make_consent_form(container):
  consent_given = asyncio.Event()
  with container:
    ui.markdown("## Consent Form")
    consent_file = os.path.join(os.path.dirname(__file__), "consent.md")
    with open(consent_file, "r") as consent_file:
      consent_text = consent_file.read()
    ui.markdown(consent_text)
    ui.checkbox("I agree to participate.", on_change=lambda: consent_given.set())

  await consent_given.wait()


async def collect_demographic_info(container):
  # Create a markdown title for the section
  clear_element(container)
  collected_demographic_info_event = asyncio.Event()
  with container:
    ui.markdown("## Demographic Info")
    ui.markdown("Please fill out the following information.")

    with ui.column():
      with ui.column():
        ui.label("Biological Sex")
        sex_input = ui.radio(["Male", "Female"], value="Male").props("inline")

      # Collect age with a textbox input
      age_input = ui.input("Age")

    # Button to submit and store the data
    async def submit():
      age = age_input.value
      sex = sex_input.value

      # Validation for age input
      if not age.isdigit() or not (0 < int(age) < 100):
        ui.notify("Please enter a valid age between 1 and 99.", type="warning")
        return
      app.storage.user["age"] = int(age)
      app.storage.user["sex"] = sex
      logger.info(f"age: {int(age)}, sex: {sex}")
      collected_demographic_info_event.set()

    ui.button("Submit", on_click=submit)
    await collected_demographic_info_event.wait()


########################
# Utility functions
########################
def stage_name(stage):
  block_idx = get_block_idx(stage)
  block_name = stage.metadata["block_metadata"].get("short", "generic")
  return f"block {block_idx}: {block_name}. Stage: {stage.name}"


def get_stage(raw_stage_idx):
  stage_order = app.storage.user["stage_order"]
  ordered_stage_idx = stage_order[raw_stage_idx]
  return all_stages[ordered_stage_idx]


async def update_stage():
  # -------------------
  # get who called this
  # -------------------
  # Get the current frame and the caller's frame
  current_frame = inspect.currentframe()
  caller_frame = current_frame.f_back
  # Extract the name of the calling function
  fn_name = caller_frame.f_code.co_name if caller_frame else "Unknown"

  # -------------------
  # check if leaving stage without saving
  # -------------------
  stage_idx = app.storage.user["stage_idx"]
  stage = get_stage(stage_idx)
  saved_data = stage.get_user_data("saved_data", False)
  if isinstance(stage, EnvStage) and not saved_data:
    logger.error(f"\n{fn_name}: leaving stage {stage.name} without saved data")
    stage_idx = len(all_stages)
    app.storage.user["stage_idx"] = stage_idx
    ui.notification("Error: Experiment unexpectedly ended early", type="negative")
    if DEBUG:
      import os

      os._exit(1)
    else:
      return stage_idx

  # -------------------
  # Update stage index
  # -------------------
  stage_idx += 1
  app.storage.user["stage_idx"] = stage_idx
  return stage_idx


async def finish_experiment(meta_container, stage_container, button_container):
  clear_element(meta_container)
  clear_element(stage_container)
  clear_element(button_container)
  logger.info("Finishing experiment")
  experiment_finished = app.storage.user.get("experiment_finished", False)

  if experiment_finished and not DEBUG:
    # in case called multiple times
    return

  #########################
  # Save data
  #########################
  async def submit(feedback):
    app.storage.user["experiment_finished"] = True
    with meta_container:
      clear_element(meta_container)
      ui.markdown("## Saving data. Please wait")
      ui.markdown(
        "**Once the data is uploaded, this app will automatically move to the next screen**"
      )

    # when over, delete user data.
    await save_data(final_save=True, feedback=feedback)
    app.storage.user["data_saved"] = True

  app.storage.user["data_saved"] = app.storage.user.get("data_saved", False)
  if not app.storage.user["data_saved"]:
    with meta_container:
      clear_element(meta_container)
      await nicewebrl.prevent_default_spacebar_behavior(False)
      ui.markdown(
        "Please provide feedback on the experiment here. For example, please describe if anything went wrong or if you have any suggestions for the experiment."
      )
      text = ui.textarea().style("width: 80%;")  # Set width to 80% of the container
      button = ui.button("Submit")
      await button.clicked()
      await submit(text.value)

  #########################
  # Final screen
  #########################
  with meta_container:
    clear_element(meta_container)
    ui.markdown("# Experiment over")
    ui.markdown("## Data saved")
    ui.markdown(
      "### Please record the following code which you will need to provide for compensation"
    )
    ui.markdown("### 'gershman.dyna2'")
    ui.markdown("#### You may close the browser")


async def save_data(final_save=True, feedback=None, **kwargs):
  user_data_file = experiment.get_user_save_file_fn()

  if final_save:
    user_storage = nicewebrl.nicejax.make_serializable(dict(app.storage.user))
    last_line = dict(
      finished=True,
      feedback=feedback,
      user_storage=user_storage,
      **kwargs,
    )
    async with aiofiles.open(user_data_file, "ab") as f:
      await nicewebrl.write_msgpack_record(f, last_line)

  if UPLOAD_DATA:
    files_to_save = [
      (user_data_file, f"data/{blob_user_filename()}.json"),
      (
        log_filename_fn(DATA_DIR, app.storage.user.get("user_id")),
        f"logs/{blob_user_filename()}.log",
      ),
    ]
    await save_to_gcs_with_retries(
      files_to_save,
      max_retries=5 if final_save else 1,
    )


async def check_if_over(*args, episode_limit=60, **kwargs):
  minutes_passed = nicewebrl.get_user_session_minutes()
  minutes_passed = app.storage.user["session_duration"]
  if minutes_passed > episode_limit:
    logger.info(f"experiment timed out after {minutes_passed} minutes")
    app.storage.user["stage_idx"] = len(all_stages)
    await finish_experiment(*args, **kwargs)


def get_git_version():
  try:
    # Get the current commit hash
    git_hash = (
      subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    )
    # Get any uncommitted changes
    git_diff = (
      subprocess.check_output(["git", "status", "--porcelain"]).decode("ascii").strip()
    )
    is_dirty = bool(git_diff)
    return f"{git_hash}{'_dirty' if is_dirty else ''}"
  except (subprocess.CalledProcessError, FileNotFoundError):
    return "git_version_unknown"


def initalize_user(user_info):
  #########
  # User settings
  #########
  nicewebrl.initialize_user(seed=DEBUG_SEED)

  app.storage.user["user_id"] = user_info["worker_id"] or app.storage.user["seed"]

  #########
  # Stage settings
  #########
  app.storage.user["stage_idx"] = app.storage.user.get("stage_idx", 0)
  app.storage.user["block_idx"] = app.storage.user.get("block_idx", 0)
  app.storage.user["block_progress"] = app.storage.user.get("block_progress", 0.0)

  stage_order = app.storage.user.get("stage_order", None)
  block_order_to_idx = app.storage.user.get("block_order_to_idx", None)

  if not stage_order:
    init_rng_key = jnp.array(app.storage.user["init_rng_key"], dtype=jnp.uint32)

    # example block order
    # [e.g., 0, 1, 3, 2]
    block_order, stage_order = experiment.generate_block_stage_order(init_rng_key)
    block_order_to_idx = {str(i): int(idx) for idx, i in enumerate(block_order)}

  app.storage.user["stage_order"] = stage_order
  # this will be used to track which block you're currently in

  app.storage.user["block_order_to_idx"] = block_order_to_idx

  #########
  # Logging
  #########

  logger.info(f"Initialized user: {app.storage.user['seed']}")
  logger.info(f"Loaded block: {app.storage.user['block_idx']}")
  logger.info(f"Loaded stage order: {stage_order}")
  stage_names_in_order = [all_stages[i].name for i in stage_order]
  logger.info(f"Loaded stages: {stage_names_in_order}")
  logger.info(f"Loaded stage: {app.storage.user['stage_idx']}")
  stage_names = collections.OrderedDict()
  stage_to_block_idx = {}
  for i, stage_idx in enumerate(stage_order):
    stage = all_stages[stage_idx]
    block = stage.metadata.get("block_metadata", {}).get("idx", -1)
    stage_names[block] = stage_names.get(block, {})
    stage_names[block].update({i: (stage_idx, stage.name)})

  block, block_pieces = next(iter(stage_names.items()))
  for block, block_pieces in stage_names.items():
    for gloabl_idx, (idx_in_block, name) in block_pieces.items():
      stage_to_block_idx[gloabl_idx] = (idx_in_block % len(block_pieces), name)

  app.storage.user["stage_to_block_idx"] = stage_to_block_idx
  app.storage.user["stage_names"] = stage_names
  app.storage.user["nstages"] = len(all_stages)
  logger.info(f"Total stages: {len(all_stages)}")


def get_block_idx(stage):
  # says which current block we're in
  # e.g. 3. from [0, 1, 3, 2]
  block_order = stage.metadata["block_metadata"]["idx"]

  # for 3, I'd want to get but 2.
  # how do we get that?
  block_idx = app.storage.user["block_order_to_idx"][str(block_order)]
  return block_idx


def block_progress():
  """Return a 2-digit rounded decimal of the progress."""
  return float(
    f"{(app.storage.user.get('block_idx') + 1) / len(experiment.all_blocks):.2f}"
  )


#####################################
# Setup database
#####################################
if not os.path.exists(DATA_DIR):
  os.mkdir(DATA_DIR)


async def init_db() -> None:
  await Tortoise.init(
    db_url=f"sqlite://{DATA_DIR}/{DATABASE_FILE}",
    modules={"models": ["nicewebrl.stages"]},
  )
  await Tortoise.generate_schemas()


async def close_db() -> None:
  await Tortoise.close_connections()


app.on_startup(init_db)
app.on_shutdown(close_db)


########################
# Run experiment
########################
async def start_experiment(meta_container, stage_container, button_container):
  if not (app.storage.user.get("experiment_started", False) or DEBUG):
    await make_consent_form(stage_container)
    await collect_demographic_info(stage_container)
    app.storage.user["experiment_started"] = True

  if DEBUG == 0:
    ui.run_javascript("window.require_fullscreen = true")
  else:
    ui.run_javascript("window.require_fullscreen = false")

  async def experiment_not_finished():
    async with get_user_lock():
      not_finished = not app.storage.user.get("experiment_finished", False)
      not_finished &= app.storage.user["stage_idx"] < len(all_stages)
    return not_finished

  async def global_handle_key_press(e):
    if DEBUG == 0 and not await nicewebrl.utils.check_fullscreen():
      ui.notify("Please enter fullscreen mode to continue experiment", type="negative")
      return
    stage_idx = app.storage.user["stage_idx"]
    if app.storage.user["stage_idx"] >= len(all_stages):
      return

    stage = get_stage(stage_idx)
    if stage.get_user_data("finished", False):
      return

    await stage.handle_key_press(e, stage_container)
    local_handle_key_press = stage.get_user_data("local_handle_key_press")
    if local_handle_key_press is not None:
      await local_handle_key_press()

  ui.on("key_pressed", global_handle_key_press)

  logger.info("Starting experiment")
  while True and await experiment_not_finished():
    stage_idx = app.storage.user["stage_idx"]
    stage = get_stage(stage_idx)
    app.storage.user["block_idx"] = get_block_idx(stage)
    app.storage.user["block_progress"] = block_progress()
    logger.info("=" * 30)
    logger.info(f"Began {stage_name(stage)}")
    await nicewebrl.prevent_default_spacebar_behavior(True)
    await run_stage(stage, stage_container, button_container)
    await nicewebrl.prevent_default_spacebar_behavior(False)

    # wait for any saves to finish before updating stage
    if isinstance(stage, EnvStage):
      await stage.finish_saving_user_data()
    async with get_user_lock():
      await update_stage()
    if app.storage.user["stage_idx"] >= len(all_stages):
      break

  await finish_experiment(meta_container, stage_container, button_container)


async def try_to_make_fullscreen():
  if DEBUG > 0:
    return True
  if not await nicewebrl.utils.check_fullscreen():
    ui.run_javascript("await document.documentElement.requestFullscreen()")
    await asyncio.sleep(1)
    ui.notify(
      "Please stay in fullscreen mode for the experiment",
      position="bottom",
      type="negative",
    )
  return await nicewebrl.utils.check_fullscreen()


async def run_stage(stage, stage_container, button_container):
  #########
  # create functions for handling key and button presses
  # Create an event to signal when the stage is over
  #########
  stage_over_event = asyncio.Event()

  async def local_handle_key_press():
    if stage.get_user_data("finished", False):
      # Signal that the stage is over
      logger.info(
        f"Finished {stage_name(stage)} via key press: {stage.get_user_data('finished', False)}"
      )
      stage_over_event.set()

  async def handle_button_press():
    if not await try_to_make_fullscreen():
      logger.info("Button press but not fullscreen")
      ui.notify("Please enter fullscreen mode to continue experiment", type="negative")
      return
    if stage.get_user_data("finished", False):
      return
    # nicewebrl.clear_element(button_container)
    await stage.handle_button_press(stage_container)
    if stage.get_user_data("finished", False):
      # Signal that the stage is over
      logger.info(f"Finished {stage_name(stage)} via button press")
      stage_over_event.set()

  #############################################
  # Activate new stage
  #############################################
  with stage_container.style("align-items: center;"):
    await stage.activate(stage_container)

  if stage.get_user_data("finished", False):
    # over as soon as stage activation was complete
    logger.info(f"Finished {stage_name(stage)} immediately after activation")
    stage_over_event.set()

  await stage.set_user_data(local_handle_key_press=local_handle_key_press)

  with button_container.style("align-items: center;"):
    nicewebrl.clear_element(button_container)
    ####################
    # Button to go to next page
    ####################
    checking_fullscreen = DEBUG == 0
    next_button_container = ui.row()

    async def create_button_and_wait():
      with next_button_container:
        nicewebrl.clear_element(next_button_container)
        button = ui.button("Next page").bind_visibility_from(stage, "next_button")
        # await button.clicked()
        await wait_for_button_or_keypress(button)
        logger.info("Button or key pressed")
        await handle_button_press()

    if stage.next_button:
      if checking_fullscreen:
        await create_button_and_wait()
        while not await nicewebrl.utils.check_fullscreen():
          if await stage_over_event.wait():
            break
          logger.info("Waiting for fullscreen")
          await asyncio.sleep(0.1)
          await create_button_and_wait()
      else:
        await create_button_and_wait()

  await stage_over_event.wait()
  nicewebrl.clear_element(button_container)


#####################################
# Home page
#####################################


def footer(footer_container):
  with footer_container:
    with ui.row():
      ui.label().bind_text_from(app.storage.user, "seed", lambda v: f"user id: {v}.")
      ui.label()
      ui.label().bind_text_from(app.storage.user, "stage_idx", lambda v: f"stage: {v}.")
      ui.label()
      ui.label().bind_text_from(
        app.storage.user,
        "session_duration",
        lambda v: f"minutes passed: {int(v)}.",
      )
      ui.label()
      ui.label().bind_text_from(
        app.storage.user,
        "block_idx",
        lambda v: f"block: {int(v) + 1}/{len(experiment.all_blocks)}.",
      )

    ui.linear_progress(value=block_progress()).bind_value_from(
      app.storage.user, "block_progress"
    )
    ui.button(
      "Toggle fullscreen",
      icon="fullscreen",
      on_click=nicewebrl.utils.toggle_fullscreen,
    ).props("flat")


@ui.page("/")
async def index(request: Request):
  logger.info("Starting webpage")
  user_info = dict(
    worker_id=request.query_params.get("workerId", None),
    hit_id=request.query_params.get("hitId", None),
    assignment_id=request.query_params.get("assignmentId", None),
    git_version=get_git_version(),
  )
  initalize_user(user_info)
  env_vars = {
    k: v
    for k, v in dict(os.environ).items()
    if not (k.startswith("/") or v.startswith("/"))
  }
  app.storage.user["user_info"] = user_info
  app.storage.user["env_vars"] = env_vars

  def print_ping(e):
    logger.info(str(e.args))

  ui.on("ping", print_ping)

  ui.run_javascript(f"window.debug = {DEBUG}")
  ################
  # Get user data and save to GCS
  ################
  if DEBUG == 0:
    await save_data_to_gcs(
      data=user_info, blob_filename=f"info/{blob_user_filename()}.json"
    )

  ################
  # Start experiment
  ################
  basic_javascript_file = nicewebrl.basic_javascript_file()
  with open(basic_javascript_file) as f:
    ui.add_body_html("<script>" + f.read() + "</script>")

  card = (
    ui.card(align_items=["center"])
    .classes("fixed-center")
    .style(
      "max-width: 90vw;"  # Set the max width of the card
      "max-height: 90vh;"  # Ensure the max height is 90% of the viewport height
      "overflow: auto;"  # Allow scrolling inside the card if content overflows
      "display: flex;"  # Use flexbox for centering
      "flex-direction: column;"  # Stack content vertically
      "justify-content: flex-start;"
      "align-items: center;"
    )
  )
  with card:
    episode_limit = 200
    ui.timer(
      1,  # check every minute
      lambda: check_if_over(
        episode_limit=episode_limit,
        meta_container=meta_container,
        stage_container=stage_container,
        button_container=button_container,
      ),
    )
    meta_container = ui.column()
    with meta_container.style("align-items: center;"):
      stage_container = ui.column()
      button_container = ui.column()
      footer_container = ui.row()
      footer(footer_container)
    with meta_container.style("align-items: center;"):
      await start_experiment(meta_container, stage_container, button_container)


ui.run(
  storage_secret="private key to secure the browser session cookie",
  reload="FLY_ALLOC_ID" not in os.environ,
  # reload=False,
  title=APP_TITLE,
)
