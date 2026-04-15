import os
import socket
from pathlib import Path


GOOGLE_CREDENTIALS = "datastore-key.json"

JAXMAZE_BUCKET = "human-dyna"
CRAFTAX_BUCKET = "craftax-human-dyna"

JAXMAZE_HUMAN_DATA_PATTERN = "*final*v2*debug=0*.json"
CRAFTAX_HUMAN_DATA_PATTERN = "*final*v2*debug=0*.json"

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECTS_ROOT = PROJECT_ROOT.parent
HOSTNAME = socket.getfqdn().lower()
IS_SERVER = str(PROJECT_ROOT).startswith("/n/") or "rc.fas.harvard.edu" in HOSTNAME


def _choose_existing_dir(candidates: list[Path], fallback: Path) -> str:
  for candidate in candidates:
    if candidate.exists():
      return str(candidate)
  return str(fallback)


def _resolve_directory(
  *,
  primary_env_var: str,
  legacy_env_var: str | None,
  candidates: list[Path],
  fallback: Path,
) -> str:
  configured = os.environ.get(primary_env_var)
  if configured:
    return str(Path(configured).expanduser())

  if legacy_env_var:
    legacy_value = os.environ.get(legacy_env_var)
    if legacy_value:
      return str(Path(legacy_value).expanduser())

  return _choose_existing_dir(candidates, fallback)


def _result_candidates() -> tuple[list[Path], Path]:
  if IS_SERVER:
    candidates = [
      PROJECT_ROOT / "preplay_results",
      PROJECT_ROOT / "plots" / "output",
    ]
    return candidates, candidates[0]

  candidates = [
    Path("/Volumes/Crucial X9/data-backup/git/research/preplay_results"),
    PROJECT_ROOT / "preplay_results",
    PROJECT_ROOT / "plots" / "output",
  ]
  return candidates, candidates[0]

def _raw_data_candidates() -> tuple[list[Path], Path]:
  if IS_SERVER:
    candidates = [
      PROJECTS_ROOT / "raw_preplay_data",
      Path("/n/holylfs06/LABS/kempner_fellow_wcarvalho/raw_preplay_data"),
      PROJECT_ROOT / "preplay_results",
    ]
    return candidates, candidates[0]

  candidates = [
    Path("/Volumes/Crucial X9/data-backup/git/research/preplay_results"),
    PROJECTS_ROOT / "raw_preplay_data",
    PROJECT_ROOT / "preplay_results",
  ]
  return candidates, candidates[0]


_RESULT_CANDIDATES, _RESULT_FALLBACK = _result_candidates()
_RAW_DATA_CANDIDATES, _RAW_DATA_FALLBACK = _raw_data_candidates()

RESULTS_DIRECTORY = _resolve_directory(
  primary_env_var="MULTITASK_PREPLAY_RESULTS_DIR",
  legacy_env_var="MULTITASK_PREPLAY_DATA_DIR",
  candidates=_RESULT_CANDIDATES,
  fallback=_RESULT_FALLBACK,
)

DATA_DIRECTORY = _resolve_directory(
  primary_env_var="MULTITASK_PREPLAY_RAW_DATA_DIR",
  legacy_env_var="MULTITASK_PREPLAY_DATA_DIR",
  candidates=_RAW_DATA_CANDIDATES,
  fallback=_RAW_DATA_FALLBACK,
)


CACHE_DIR = os.path.join(DATA_DIRECTORY, "cache")
current_directory = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_CACHE_DIR = os.path.join(current_directory, "analysis", "paper_results_cache")

# Raw data directories
JAXMAZE_MODEL_RAW_DATA_DIR = os.path.join(DATA_DIRECTORY, "data", "jaxmaze")
CRAFTAX_MODEL_RAW_DATA_DIR = os.path.join(DATA_DIRECTORY, "data", "craftax")
JAXMAZE_HUMAN_RAW_DATA_DIR = os.path.join(DATA_DIRECTORY, "data", "jaxmaze", "human_data")
CRAFTAX_HUMAN_RAW_DATA_DIR = os.path.join(DATA_DIRECTORY, "data", "craftax", "human_data")

# Processed data directories
JAXMAZE_MODEL_DATA_DIR = os.path.join(RESULTS_DIRECTORY, "data", "jaxmaze")
CRAFTAX_MODEL_DATA_DIR = os.path.join(RESULTS_DIRECTORY, "data", "craftax")
JAXMAZE_HUMAN_DATA_DIR = os.path.join(
    RESULTS_DIRECTORY, "data", "jaxmaze", "human_data")
CRAFTAX_HUMAN_DATA_DIR = os.path.join(
    RESULTS_DIRECTORY, "data", "craftax", "human_data")


# Results directories
JAXMAZE_RESULTS_DIR = os.path.join(RESULTS_DIRECTORY, "results/jaxmaze")
CRAFTAX_RESULTS_DIR = os.path.join(RESULTS_DIRECTORY, "results/craftax")
CRAFTAX_AI_DIR = os.path.join(RESULTS_DIRECTORY, "results/craftax_ai")

# Env figure directory
ENV_FIGURES_DIR = os.path.join(RESULTS_DIRECTORY, "env_figures")
JAXMAZE_ENV_FIGURES_DIR = os.path.join(ENV_FIGURES_DIR, "jaxmaze")
CRAFTAX_ENV_FIGURES_DIR = os.path.join(ENV_FIGURES_DIR, "craftax")

# Analysis figure directory
ANALYSIS_FIGURES_DIR = os.path.join(RESULTS_DIRECTORY, "analysis_figures")
JAXMAZE_INDIVIDUAL_RTS_DIR = os.path.join(
  ANALYSIS_FIGURES_DIR, "jaxmaze_individual_rts"
)
JAXMAZE_SF_DIR = os.path.join(ANALYSIS_FIGURES_DIR, "jaxmaze_sf_analysis")
JAXMAZE_OVERLAP_ANALYSIS_DIR = os.path.join(
  ANALYSIS_FIGURES_DIR, "jaxmaze_overlap_analysis"
)
CRAFTAX_OVERLAP_ANALYSIS_DIR = os.path.join(
  ANALYSIS_FIGURES_DIR, "craftax_overlap_analysis"
)

# File for paper stats
PAPER_STATS_FILE = os.path.join(RESULTS_DIRECTORY, "paper_stats.yaml")

# Huggingface dataset names
HUGGINGFACE_JAXMAZE_DATASET_NAME = "Multitask_Preplay_JaxMaze"
HUGGINGFACE_CRAFTAX_DATASET_NAME = "Multitask_Preplay_Craftax"


# Overlap thresholds
TWO_PATHS_OVERLAP_THRESHOLD = 0.5
SHORTCUT_OVERLAP_THRESHOLD = 0.7
CRAFTAX_OVERLAP_THRESHOLD = 0.25
COSINE_THRESHOLD = 0.5
