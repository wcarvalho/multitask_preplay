import os
import os.path
from dotenv import load_dotenv

load_dotenv()

GIVE_INSTRUCTIONS = int(os.environ.get("INST", 1))
DEBUG = int(os.environ.get("DEBUG", 0))
DEBUG_SEED = int(os.environ.get("SEED", 0))
NAME = os.environ.get("NAME", "exp")
MAN = os.environ.get("MAN", "paths")
DATA_DIR = os.environ.get("DATA_DIR", "experiment_data")
DATA_DIR = os.path.join(os.path.dirname(__file__), DATA_DIR)

NICEGUI_DIR = os.environ.get("NICEGUI_DIR", "nicegui_data")
NICEGUI_DIR = os.path.join(os.path.dirname(__file__), NICEGUI_DIR)

FEEDBACK = int(os.environ.get("FEEDBACK", 0))
SAY_REUSE = int(os.environ.get("SAY_REUSE", 1))
COND2_TRAIN = int(os.environ.get("COND2_TRAIN", 1))
TIMER = int(os.environ.get("TIMER", 0))
VERBOSITY = int(os.environ.get("VERBOSITY", 0))
NTRAIN = int(os.environ.get("NTRAIN", 8))
TIME_LIMIT = int(os.environ.get("TIME_LIMIT", 10_000_000))

EXPERIMENT = int(os.environ.get("EXP", 4))
UPLOAD_DATA = int(os.environ.get("UPLOAD_DATA", 1))
DATABASE_FILE = os.environ.get("DB_FILE", "db.sqlite")

USE_DONE = DEBUG > 0

# Set NiceGUI storage path to DATA_DIR
os.environ["NICEGUI_STORAGE_PATH"] = NICEGUI_DIR

CLEAR_CACHE = int(os.environ.get("CLEAR_CACHE", 1))
if CLEAR_CACHE:
  import shutil

  shutil.rmtree(NICEGUI_DIR, ignore_errors=True)
