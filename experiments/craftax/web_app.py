import aiofiles
import subprocess
import os.path
import asyncio
from asyncio import Lock
from nicegui import app, ui
from fastapi import Request, APIRouter
from tortoise import Tortoise
import time
import traceback
from datetime import datetime

from experiments.gcs import save_to_gcs_with_retries
import nicewebrl
from nicewebrl.logging import setup_logging, get_logger
from nicewebrl.utils import wait_for_button_or_keypress
from nicewebrl import stages
from importlib.util import find_spec
import shutil


DATABASE_FILE = os.environ.get("DB_FILE", "db.sqlite")
DATA_DIR = os.environ.get("DATA_DIR", "craftax_data")
DATA_DIR = os.path.join(os.path.dirname(__file__), DATA_DIR)

LOGGER_DISPLAY_TIME = int(os.environ.get("LOGGER_DISPLAY_TIME", 0))
DEBUG = int(os.environ.get("DEBUG", 0))
CONSENT = int(os.environ.get("CONSENT", 1))
DEBUG_SEED = int(os.environ.get("SEED", 0))
NAME = os.environ.get("NAME", "exp")
DATABASE_FILE = f"{DATABASE_FILE}_name={NAME}_debug={DEBUG}"
DUMMY_ENV = int(os.environ.get("DUMMY_ENV", 1))
VERBOSITY = int(os.environ.get("VERBOSITY", 0))

os.makedirs(DATA_DIR, exist_ok=True)

_user_locks = {}

# Add module loading management
experiment_structure = None
config = None
craftax_loaded = asyncio.Event()


# Enhanced logging setup for craftax loader
#####################################
# Setup logger
#####################################
def log_filename_fn(log_dir, user_id):
  return os.path.join(log_dir, f"log_{user_id}.log")


setup_logging(
  DATA_DIR,
  # each user has a unique seed
  # can use this to identify users
  display_time=LOGGER_DISPLAY_TIME,
  log_filename_fn=log_filename_fn,
  nicegui_storage_user_key="user_id",
)
logger = get_logger("main")
loader_logger = get_logger("craftax_loader")

# Global variables for tracking load state
load_start_time = None
load_error = None


def restore_texture_cache_if_needed():
  """Restore texture cache files from local cache if they don't exist in the package directory."""
  # Get paths for texture cache files
  original_constants_directory = os.path.join(
    os.path.dirname(find_spec("craftax.craftax.constants").origin), "assets"
  )
  TEXTURE_CACHE_FILE = os.path.join(original_constants_directory, "texture_cache.pbz2")
  FULLMAP_TEXTURE_CACHE_FILE = os.path.join(
    original_constants_directory, "fullmap_texture_cache_48.pbz2"
  )

  # Local cache paths
  cache_dir = "craftax_cache"
  source_cache = os.path.join(cache_dir, "texture_cache.pbz2")
  source_fullmap_cache = os.path.join(cache_dir, "fullmap_texture_cache_48.pbz2")

  # Create the destination directories if they don't exist
  os.makedirs(os.path.dirname(TEXTURE_CACHE_FILE), exist_ok=True)
  os.makedirs(os.path.dirname(FULLMAP_TEXTURE_CACHE_FILE), exist_ok=True)

  # Copy texture cache files if needed
  if not os.path.exists(TEXTURE_CACHE_FILE) and os.path.exists(source_cache):
    loader_logger.info(
      f"Restoring texture cache from {source_cache} to {TEXTURE_CACHE_FILE}"
    )
    shutil.copy2(source_cache, TEXTURE_CACHE_FILE)
    loader_logger.info("Regular cache file restored successfully!")
  else:
    loader_logger.info(f"{TEXTURE_CACHE_FILE} already exists.")

  if not os.path.exists(FULLMAP_TEXTURE_CACHE_FILE) and os.path.exists(
    source_fullmap_cache
  ):
    loader_logger.info(
      f"Restoring fullmap texture cache from {source_fullmap_cache} to {FULLMAP_TEXTURE_CACHE_FILE}"
    )
    shutil.copy2(source_fullmap_cache, FULLMAP_TEXTURE_CACHE_FILE)
    loader_logger.info("Fullmap cache file restored successfully!")
  else:
    loader_logger.info(f"{FULLMAP_TEXTURE_CACHE_FILE} already exists.")


async def load_craftax_module():
  global experiment_structure, load_start_time, load_error, config
  load_start_time = datetime.now()
  loop = asyncio.get_event_loop()
  loader_logger.info("Starting craftax module load attempt")

  # Restore texture cache if needed
  restore_texture_cache_if_needed()

  try:
    loader_logger.info("Attempting to import craftax_experiment_structure")

    def import_with_logging():
      try:
        import craftax_experiment_structure
        from simulations import craftax_experiment_configs

        loader_logger.info("Import successful")
        return craftax_experiment_structure, craftax_experiment_configs
      except Exception as e:
        error_msg = f"Import failed: {str(e)}\n{traceback.format_exc()}"
        loader_logger.error(error_msg)
        raise

    experiment_structure, config = await loop.run_in_executor(None, import_with_logging)

    loader_logger.info("Craftax module loaded successfully")
    load_duration = (datetime.now() - load_start_time).total_seconds()
    loader_logger.info(f"Total load time: {load_duration} seconds")

  except Exception as e:
    load_error = str(e)
    error_msg = f"Failed to load craftax module: {str(e)}\n{traceback.format_exc()}"
    loader_logger.error(error_msg)
    raise
  finally:
    craftax_loaded.set()


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


#####################################
# Helper functions
#####################################
def stage_name(stage):
  return stage.name


# This is used to ensure that each user has a unique lock
def get_user_lock():
  """A function that returns a lock for the current user using their unique seed"""
  user_seed = app.storage.user["seed"]
  if user_seed not in _user_locks:
    _user_locks[user_seed] = Lock()
  return _user_locks[user_seed]


def experiment_not_finished():
  """Check if the experiment is not finished"""
  # async with get_user_lock():
  global experiment_structure
  not_finished = not app.storage.user.get("experiment_finished", False)
  not_finished &= experiment_structure.experiment.not_finished()
  return not_finished


def blob_user_filename():
  """filename structure for user data in GCS (cloud)"""
  seed = app.storage.user["seed"]
  worker = app.storage.user.get("worker_id", None)
  if worker is not None:
    return f"user={seed}_worker={worker}_name={NAME}_debug={DEBUG}"
  else:
    return f"user={seed}_name={NAME}_debug={DEBUG}"


async def global_handle_key_press(e, container):
  """Define global key press handler

  We can get stage-specific key handling by using this and having this function
  call the stage-specific key handler. When the experiment begins, we'll register
  a key listener to call this function
  """
  global experiment_structure
  if DEBUG == 0 and not await nicewebrl.utils.check_fullscreen():
    ui.notify("Please enter fullscreen mode to continue experiment", type="negative")
    return

  if not craftax_loaded.is_set():
    logger.info("craftax not loaded")
    return

  experiment = experiment_structure.experiment
  stage = await experiment.get_stage()

  if stage.get_user_data("finished", False):
    return

  await stage.handle_key_press(e, container)
  local_handle_key_press = stage.get_user_data("local_handle_key_press")
  if local_handle_key_press is not None:
    await local_handle_key_press()


async def save_data(final_save=True, feedback=None, **kwargs):
  global experiment_structure, config
  user_data_file = experiment_structure.get_user_save_file_fn()

  if final_save:
    # --------------------------------
    # save user data to final line of file
    # --------------------------------
    user_storage = nicewebrl.make_serializable(dict(app.storage.user))
    last_line = dict(
      finished=True,
      feedback=feedback,
      user_storage=user_storage,
      **kwargs,
    )
    async with aiofiles.open(user_data_file, "ab") as f:
      await nicewebrl.write_msgpack_record(f, last_line)

  files_to_save = [
    (user_data_file, f"data/{blob_user_filename()}.json"),
    (
      log_filename_fn(DATA_DIR, app.storage.user.get("user_id")),
      f"logs/{blob_user_filename()}.log",
    ),
  ]
  if not DEBUG:
    logger.info(f"Saving to bucket: {config.BUCKET_NAME}")
    await save_to_gcs_with_retries(
      files_to_save,
      max_retries=5 if final_save else 1,
      bucket_name=config.BUCKET_NAME,
    )

  if final_save:
    # Try to delete local files after successful upload
    from nicewebrl.stages import StageStateModel

    logger.info(f"Deleting data for user {app.storage.browser['id']}")
    await StageStateModel.filter(session_id=app.storage.browser["id"]).delete()
    logger.info(
      f"Successfully deleted stage inforation for user {app.storage.browser['id']}"
    )
    for local_file, _ in files_to_save:
      try:
        os.remove(local_file)
        logger.info(f"Successfully deleted local file: {local_file}")
      except Exception as e:
        logger.warning(f"Failed to delete local file {local_file}: {str(e)}")


#####################################
# Setup database for storing experiment data
#####################################
if not os.path.exists(DATA_DIR):
  os.mkdir(DATA_DIR)


async def init_db() -> None:
  await Tortoise.init(
    db_url=f"sqlite://{DATA_DIR}/{DATABASE_FILE}",
    # this will look in models.py,
    # models.py uses defaults from nicewebrl
    modules={"models": ["nicewebrl.stages"]},
  )
  await Tortoise.generate_schemas()


async def close_db() -> None:
  await Tortoise.close_connections()


# Modify startup handler
@app.on_startup
async def startup():
  asyncio.create_task(load_craftax_module())
  await init_db()


#####################################
# Consent Form and demographic info
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
  nicewebrl.clear_element(container)
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

    button = ui.button("Submit", on_click=submit)
    await button.clicked()


########################
# Run experiment
########################


def get_experiment_progress():
  global experiment_structure
  experiment: nicewebrl.Experiment = experiment_structure.experiment
  stage_progress = float(
    f"{(app.storage.user['stage_idx'] + 1) / experiment.num_stages:.2f}"
  )
  app.storage.user["stage_progress"] = stage_progress
  return stage_progress


async def start_experiment(meta_container, stage_container, button_container):
  global experiment_structure
  # ========================================
  # Consent form and demographic info
  # ========================================
  if not (app.storage.user.get("experiment_started", False) or DEBUG) and CONSENT:
    await make_consent_form(stage_container)
    await collect_demographic_info(stage_container)
    app.storage.user["experiment_started"] = True

  # ========================================
  # Force fullscreen
  # ========================================
  if DEBUG == 0:
    ui.run_javascript("window.require_fullscreen = true")
  else:
    ui.run_javascript("window.require_fullscreen = false")

  # ========================================
  # Register global key press handler
  # ========================================
  ui.on("key_pressed", lambda e: global_handle_key_press(e, stage_container))

  experiment: nicewebrl.Experiment = experiment_structure.experiment

  # ========================================
  # Run experiment
  # ========================================
  logger.info("Starting experiment")
  block_order = await experiment.get_block_order()
  block_names_in_order = [experiment.blocks[i].name for i in block_order]
  logger.info(f"Block order: {block_names_in_order}")

  while experiment_not_finished():
    # get current block
    block = await experiment.get_block()

    while await block.not_finished():
      stage = await block.get_stage()

      # activate stage
      await nicewebrl.prevent_default_spacebar_behavior(False)
      await run_stage(stage, stage_container, button_container)
      await nicewebrl.prevent_default_spacebar_behavior(True)
      logger.info(f"Finished stage '{stage.name}'")
      # wait for any saves to finish before updating stage
      # very important, otherwise may lose data
      if isinstance(stage, stages.EnvStage):
        await stage.finish_saving_user_data()
        logger.info(f"Saved data for stage '{stage.name}'")

      # next stage
      await experiment.advance_stage()
      get_experiment_progress()

    # next block
    logger.info(f"Finished block '{block.name}'")
    await experiment.advance_block()

    # see if over
    blocks_over = experiment.get_block_idx() >= experiment.num_blocks
    stages_over = experiment.get_experiment_stage_idx() >= experiment.num_stages
    if blocks_over or stages_over:
      break

  await finish_experiment(meta_container, stage_container, button_container)


async def finish_experiment(meta_container, stage_container, button_container):
  nicewebrl.clear_element(meta_container)
  nicewebrl.clear_element(stage_container)
  nicewebrl.clear_element(button_container)
  logger.info("Finishing experiment")

  # if DEBUG > 0:
  #  # in case called multiple times
  #  return

  #########################
  # Save data
  #########################
  async def submit(feedback):
    app.storage.user["experiment_finished"] = True
    status_container = None
    with meta_container:
      nicewebrl.clear_element(meta_container)
      ui.markdown(
        "## Your data is being saved. Please do not close or refresh the page."
      )
      status_container = ui.markdown("Saving local files...")

    try:
      # Create a task for the save operation with a timeout
      save_task = asyncio.create_task(save_data(final_save=True, feedback=feedback))
      start_time = time.time()

      # Update status every 2 seconds while waiting for save
      while not save_task.done():
        elapsed_seconds = int(time.time() - start_time)
        status_container.content = (
          f"Still saving... ({elapsed_seconds}s elapsed). This may take 5-10 minutes."
        )
        try:
          # Wait for either task completion or timeout
          await asyncio.wait_for(asyncio.shield(save_task), timeout=2.0)
        except asyncio.TimeoutError:
          # This is expected - we use timeout to update status
          continue
        except Exception as e:
          logger.error(f"Error during save: {e}")
          status_container.content = (
            "⚠️ Error saving data. Please contact the experimenter."
          )
          raise

      # If we get here, save was successful
      elapsed_seconds = int(time.time() - start_time)
      status_container.content = (
        f"✅ Save complete in {elapsed_seconds}s! Moving to next screen..."
      )
      app.storage.user["data_saved"] = True

    except Exception as e:
      logger.error(f"Save failed: {e}")
      status_container.content = "⚠️ Error saving data. Please contact the experimenter."
      raise

  app.storage.user["data_saved"] = app.storage.user.get("data_saved", False)
  if not app.storage.user["data_saved"]:
    with meta_container:
      nicewebrl.clear_element(meta_container)
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
    nicewebrl.clear_element(meta_container)
    ui.markdown("# Experiment over")
    ui.markdown("## Data saved")
    ui.markdown(
      "### Please record the following code which you will need to provide for compensation"
    )
    ui.markdown("### 'gershman.craftax'")
    ui.markdown("#### You may close the browser")


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
# Root page
#####################################
async def initalize_user(request: Request):
  nicewebrl.initialize_user(seed=DEBUG_SEED)
  app.storage.user["worker_id"] = request.query_params.get("workerId", None)
  app.storage.user["hit_id"] = request.query_params.get("hitId", None)
  app.storage.user["assignment_id"] = request.query_params.get("assignmentId", None)

  app.storage.user["user_id"] = (
    app.storage.user["worker_id"] or app.storage.user["seed"]
  )
  global experiment_structure
  experiment: nicewebrl.Experiment = experiment_structure.experiment
  await experiment.initialize()


@ui.page("/")
async def index(request: Request):
  # set up callback to log all pings
  def print_ping(e):
    if VERBOSITY:
      print(str(e.args))

  ui.on("ping", print_ping)

  #########################################
  # Add javascript file (responsible for pinging)
  #########################################
  with open(nicewebrl.basic_javascript_file()) as f:
    ui.add_body_html("<script>" + f.read() + "</script>")

  # Show loading screen if module not ready
  if not craftax_loaded.is_set():
    with ui.card().classes("fixed-center") as card:
      card.style("width: 80vw; max-height: 90vh;")

      # Main loading message
      ui.label("Loading experiment... This will take up to 5 minutes.").classes(
        "text-h4"
      )
      elapsed_time = ui.label("Please don't close or refresh the page")
      # Progress information
      elapsed_time = ui.label("Time elapsed: 0 seconds")
      load_status = ui.label("Current status: Initializing...")
      error_display = ui.label().classes("text-red")

      start_time = time.time()

      async def update_loading_info():
        if not craftax_loaded.is_set():
          seconds = int(time.time() - start_time)
          elapsed_time.text = f"Time elapsed: {seconds} seconds"
          if load_error:
            error_display.text = f"Error: {load_error}"
            load_status.text = "Status: Failed to load"

          # Log periodic updates
          if seconds % 10 == 0:  # Log every 10 seconds
            ui.run_javascript(f"console.log('loading for {seconds} seconds')")
            loader_logger.info(f"Still loading after {seconds} seconds")

          return not craftax_loaded.is_set()  # Continue until loaded

      ui.timer(1.0, update_loading_info)

      # Add detailed JavaScript monitoring
      ui.add_body_html("""
          <script>
          let lastPingTime = Date.now();
          
          async function checkStatus() {
              try {
                  const response = await fetch('/status');
                  const data = await response.json();
                  console.log('Status check:', data);
                  
                  if (data.loaded) {
                      console.log('Module loaded, reloading page');
                      window.location.reload();
                  } else if (data.load_error) {
                      console.error('Loading error:', data.load_error);
                  }
                  
                  // Calculate time since last ping
                  const currentTime = Date.now();
                  const timeSinceLastPing = currentTime - lastPingTime;
                  console.log('Time since last ping:', timeSinceLastPing, 'ms');
                  lastPingTime = currentTime;
                  
                  if (!data.loaded) {
                      setTimeout(checkStatus, 1000);
                  }
              } catch (error) {
                  console.error('Status check failed:', error);
                  setTimeout(checkStatus, 1000);
              }
          }
          
          // Start checking status
          checkStatus();
          
          // Monitor for any JavaScript errors
          window.onerror = function(msg, url, line) {
              console.error('JavaScript error:', msg, 'at', url, 'line', line);
              return false;
          };
          </script>
      """)
    return

  ################
  # Start experiment
  ################
  await initalize_user(request)
  user_info = dict(
    worker_id=request.query_params.get("workerId", None),
    hit_id=request.query_params.get("hitId", None),
    assignment_id=request.query_params.get("assignmentId", None),
    git_version=get_git_version(),
  )
  env_vars = {
    k: v
    for k, v in dict(os.environ).items()
    if not (k.startswith("/") or v.startswith("/"))
  }
  app.storage.user["user_info"] = user_info
  app.storage.user["env_vars"] = env_vars

  ui.run_javascript(f"window.debug = {DEBUG}")

  card = (
    ui.card(align_items=["center"])
    .classes("fixed-center")
    .style(
      "width: 80vw;"  # Set width to 90% of viewport width
      "max-height: 90vh;"  # Keep the same max height
      "overflow: auto;"
      "display: flex;"
      "flex-direction: column;"
      "justify-content: flex-start;"
      "align-items: center;"
      "padding: 1rem;"
    )
  )
  with card:
    episode_limit = 200
    meta_container = ui.column()
    with meta_container.style("align-items: center;"):
      #########################################
      # Run experiment
      #########################################
      stage_container = ui.column()
      button_container = ui.column()
      ui.timer(
        interval=1,
        callback=lambda: check_if_over(
          episode_limit=episode_limit,
          meta_container=meta_container,
          stage_container=stage_container,
          button_container=button_container,
        ),
      )
      footer_container = ui.row()
      await footer(footer_container)
    with meta_container.style("align-items: center;"):
      await start_experiment(meta_container, stage_container, button_container)


async def check_if_over(*args, episode_limit=60, **kwargs):
  """If past time limit, finish experiment"""
  global experiment_structure
  experiment = experiment_structure.experiment
  minutes_passed = nicewebrl.get_user_session_minutes()
  minutes_passed = app.storage.user["session_duration"]
  if minutes_passed > episode_limit:
    logger.info(f"experiment timed out after {minutes_passed} minutes")
    experiment.force_finish()
    await finish_experiment(*args, **kwargs)


async def footer(footer_container):
  """Add user information and progress bar to the footer"""
  global experiment_structure
  experiment = experiment_structure.experiment
  with footer_container:
    with ui.row():
      user_id = app.storage.user.get("seed", None)
      if user_id is None:
        return

      ui.label().bind_text_from(app.storage.user, "seed", lambda v: f"user id: {v}.")
      ui.label()

      def get_stage_idx(v):
        return f"stage: {int(v) + 1}/{experiment.num_stages}."

      ui.label().bind_text_from(app.storage.user, "stage_idx", get_stage_idx)
      ui.label()
      ui.label().bind_text_from(
        app.storage.user, "session_duration", lambda v: f"minutes passed: {int(v)}."
      )

    ui.linear_progress(value=get_experiment_progress()).bind_value_from(
      app.storage.user, "stage_progress"
    )

    ui.button(
      "Toggle fullscreen",
      icon="fullscreen",
      on_click=nicewebrl.utils.toggle_fullscreen,
    ).props("flat")


# Add status endpoint
router = APIRouter()


@router.get("/status")
async def get_status():
  """Enhanced status endpoint with detailed loading information"""
  global load_start_time, load_error

  current_time = datetime.now()
  load_duration = (
    None
    if load_start_time is None
    else (current_time - load_start_time).total_seconds()
  )

  return {
    "loaded": craftax_loaded.is_set(),
    "load_duration": load_duration,
    "load_error": load_error,
    "load_start_time": load_start_time.isoformat() if load_start_time else None,
  }


app.include_router(router)


ui.run(
  storage_secret="private key to secure the browser session cookie",
  reload="FLY_ALLOC_ID" not in os.environ,
  title="Crafter Web App",
  port=8080,
)
