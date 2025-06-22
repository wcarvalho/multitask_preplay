import os.path


GOOGLE_CREDENTIALS = "datastore-key.json"

JAXMAZE_BUCKET = "human-dyna"
CRAFTAX_BUCKET = "craftax-human-dyna"

JAXMAZE_HUMAN_DATA_PATTERN = "*final*v2*debug=0*.json"
CRAFTAX_HUMAN_DATA_PATTERN = "*final*v2*debug=0*.json"


DIRECTORY = os.environ.get(
  "MULTITASK_PREPLAY_DATA_DIR", "/Users/wilka/git/research/preplay_results"
)

CACHE_DIR = os.path.join(DIRECTORY, "cache")

JAXMAZE_DATA_DIR = os.path.join(DIRECTORY, "data_jaxmaze")
CRAFTAX_DATA_DIR = os.path.join(DIRECTORY, "data_craftax")

JAXMAZE_USER_DIR = os.path.join(DIRECTORY, "data_jaxmaze", "human_data")
CRAFTAX_USER_DIR = os.path.join(DIRECTORY, "data_craftax", "human_data")

JAXMAZE_RESULTS_DIR = os.path.join(DIRECTORY, "results/jaxmaze")
CRAFTAX_RESULTS_DIR = os.path.join(DIRECTORY, "results/craftax")

CRAFTAX_AI_DIR = os.path.join(DIRECTORY, "results/craftax_ai")

PAPER_STATS_FILE = os.path.join(DIRECTORY, "paper_stats.json")
PAPER_STATS_MODEL_FILE = os.path.join(DIRECTORY, "paper_stats_model.yaml")

HUGGINGFACE_JAXMAZE_DATASET_NAME = "Multitask_Preplay_JaxMaze"
HUGGINGFACE_CRAFTAX_DATASET_NAME = "Multitask_Preplay_Craftax"
