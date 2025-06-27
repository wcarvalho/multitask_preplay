import os.path


GOOGLE_CREDENTIALS = "datastore-key.json"

JAXMAZE_BUCKET = "human-dyna"
CRAFTAX_BUCKET = "craftax-human-dyna"

JAXMAZE_HUMAN_DATA_PATTERN = "*final*v2*debug=0*.json"
CRAFTAX_HUMAN_DATA_PATTERN = "*final*v2*debug=0*.json"


DIRECTORY = os.environ.get(
  "MULTITASK_PREPLAY_DATA_DIR", "../preplay_results"
)

CACHE_DIR = os.path.join(DIRECTORY, "cache")

# Data directories
JAXMAZE_DATA_DIR = os.path.join(DIRECTORY, "data", "jaxmaze")
CRAFTAX_DATA_DIR = os.path.join(DIRECTORY, "data", "craftax")
JAXMAZE_USER_DIR = os.path.join(DIRECTORY, "data", "jaxmaze", "human_data")
CRAFTAX_USER_DIR = os.path.join(DIRECTORY, "data", "craftax", "human_data")


# Results directories
JAXMAZE_RESULTS_DIR = os.path.join(DIRECTORY, "results/jaxmaze")
CRAFTAX_RESULTS_DIR = os.path.join(DIRECTORY, "results/craftax")
CRAFTAX_AI_DIR = os.path.join(DIRECTORY, "results/craftax_ai")

# Env figure directory
ENV_FIGURES_DIR = os.path.join(DIRECTORY, "env_figures")
JAXMAZE_ENV_FIGURES_DIR = os.path.join(ENV_FIGURES_DIR, "jaxmaze")
CRAFTAX_ENV_FIGURES_DIR = os.path.join(ENV_FIGURES_DIR, "craftax")

# Analysis figure directory
ANALYSIS_FIGURES_DIR = os.path.join(DIRECTORY, "figures", "analysis_figures")
JAXMAZE_INDIVIDUAL_RTS_DIR = os.path.join(ANALYSIS_FIGURES_DIR, "jaxmaze_individual_rts")
JAXMAZE_SF_DIR = os.path.join(ANALYSIS_FIGURES_DIR, "jaxmaze_sf_analysis")
JAXMAZE_OVERLAP_ANALYSIS_DIR = os.path.join(ANALYSIS_FIGURES_DIR, "jaxmaze_overlap_analysis")
CRAFTAX_OVERLAP_ANALYSIS_DIR = os.path.join(ANALYSIS_FIGURES_DIR, "craftax_overlap_analysis")

# File for paper stats
PAPER_STATS_FILE = os.path.join(DIRECTORY, "paper_stats.yaml")

# Huggingface dataset names
HUGGINGFACE_JAXMAZE_DATASET_NAME = "Multitask_Preplay_JaxMaze"
HUGGINGFACE_CRAFTAX_DATASET_NAME = "Multitask_Preplay_Craftax"
