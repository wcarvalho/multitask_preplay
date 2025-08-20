import os
import os.path
from dotenv import load_dotenv

load_dotenv()

# Core experiment settings
GIVE_INSTRUCTIONS = int(os.environ.get("INST", 1))
DEBUG = int(os.environ.get("DEBUG", 0))
DEBUG_SEED = int(os.environ.get("SEED", 0))
NAME = os.environ.get("NAME", "exp")
MAN = os.environ.get("MAN", "paths")
MANIPULATION = os.environ.get("MANIPULATION", "paths")

# Directory settings
DATA_DIR = os.environ.get("DATA_DIR", "experiment_data")
DATA_DIR = os.path.join(os.path.dirname(__file__), DATA_DIR)
NICEGUI_DIR = os.environ.get("NICEGUI_DIR", "nicegui_data")
NICEGUI_DIR = os.path.join(os.path.dirname(__file__), NICEGUI_DIR)

# Experiment parameters
FEEDBACK = int(os.environ.get("FEEDBACK", 0))
SAY_REUSE = int(os.environ.get("SAY_REUSE", 1))
VERBOSITY = int(os.environ.get("VERBOSITY", 0))
NTRAIN = int(os.environ.get("NTRAIN", 8))
NUM_BLOCKS = int(os.environ.get("NUM_BLOCKS", 100))
EVAL_SHOW_MAP = int(os.environ.get("EVAL_SHOW_MAP", 0))

# System settings
DATABASE_FILE = os.environ.get("DB_FILE", "db.sqlite")
PRECOMPILE = int(os.environ.get("PRECOMPILE", 1))
DUMMY_ENV = int(os.environ.get("DUMMY_ENV", 0))
MONSTERS = int(os.environ.get("MONSTERS", 1))

# UI/Web settings
CONSENT = int(os.environ.get("CONSENT", 1))

# Set NiceGUI storage path to DATA_DIR
os.environ["NICEGUI_STORAGE_PATH"] = NICEGUI_DIR

CLEAR_CACHE = int(os.environ.get("CLEAR_CACHE", 1))
if CLEAR_CACHE:
    import shutil
    shutil.rmtree(NICEGUI_DIR, ignore_errors=True)