import os.path


GOOGLE_CREDENTIALS = "datastore-key.json"

JAXMAZE_BUCKET = "human-dyna"
CRAFTAX_BUCKET = "craftax-human-dyna"

JAXMAZE_HUMAN_DATA_PATTERN = "*final*v2*debug=0*.json"
CRAFTAX_HUMAN_DATA_PATTERN = "*final*v2*debug=0*.json"

RESULTS_DIRECTORY = os.environ.get(
  "MULTITASK_PREPLAY_DATA_DIR",
  os.path.join(os.path.dirname(os.path.abspath(__file__)), "preplay_results"),
)

DATA_DIRECTORY = os.environ.get(
  "MULTITASK_PREPLAY_DATA_DIR",
  os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw_preplay_data"
  ),
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
