import os
import os.path
import shutil
from importlib.util import find_spec
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

_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_CANONICAL_CACHE_DIR = os.path.join(_THIS_DIR, "craftax_cache")
_ROOT_CACHE_DIR = os.path.join(_PROJECT_ROOT, "craftax_cache")
_CACHE_FILENAMES = (
  "texture_cache.pbz2",
  "fullmap_texture_cache_48.pbz2",
)


def _unique_paths(paths):
  seen = set()
  for path in paths:
    if not path:
      continue
    normalized = os.path.abspath(path)
    if normalized in seen:
      continue
    seen.add(normalized)
    yield normalized


def check_cache(verbose: bool = True) -> str:
  """Ensure Craftax cache files exist in experiments/craftax/craftax_cache and package assets."""
  os.makedirs(CRAFTAX_CACHE_DIR, exist_ok=True)

  cache_candidates = list(
    _unique_paths(
      [
        CRAFTAX_CACHE_DIR,
        _ROOT_CACHE_DIR,
        os.environ.get("CRAFTAX_CACHE_DIR"),
      ]
    )
  )

  for filename in _CACHE_FILENAMES:
    canonical_path = os.path.join(CRAFTAX_CACHE_DIR, filename)
    if os.path.exists(canonical_path):
      continue

    for candidate_dir in cache_candidates:
      candidate_path = os.path.join(candidate_dir, filename)
      if candidate_path == canonical_path or not os.path.exists(candidate_path):
        continue
      if verbose:
        print(f"Copying {filename} from {candidate_path} to {canonical_path}")
      shutil.copy2(candidate_path, canonical_path)
      break

  spec = find_spec("craftax.craftax.constants")
  if spec is None or spec.origin is None:
    return CRAFTAX_CACHE_DIR

  assets_dir = os.path.join(os.path.dirname(spec.origin), "assets")
  os.makedirs(assets_dir, exist_ok=True)

  for filename in _CACHE_FILENAMES:
    source_path = os.path.join(CRAFTAX_CACHE_DIR, filename)
    dest_path = os.path.join(assets_dir, filename)
    if os.path.exists(dest_path) or not os.path.exists(source_path):
      continue
    if verbose:
      print(f"Restoring {filename} from {source_path} to {dest_path}")
    shutil.copy2(source_path, dest_path)

  return CRAFTAX_CACHE_DIR

# Set NiceGUI storage path to DATA_DIR
os.environ["NICEGUI_STORAGE_PATH"] = NICEGUI_DIR
CRAFTAX_CACHE_DIR = _CANONICAL_CACHE_DIR
os.makedirs(CRAFTAX_CACHE_DIR, exist_ok=True)

os.environ["CRAFTAX_CACHE_DIR"] = CRAFTAX_CACHE_DIR
check_cache()
CLEAR_CACHE = int(os.environ.get("CLEAR_CACHE", 1))
if CLEAR_CACHE:
  shutil.rmtree(NICEGUI_DIR, ignore_errors=True)
