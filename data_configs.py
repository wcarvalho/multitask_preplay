import os.path


GOOGLE_CREDENTIALS = "datastore-key.json"

JAXMAZE_BUCKET = "human-dyna"
CRAFTAX_BUCKET = "craftax-human-dyna"

JAXMAZE_HUMAN_DATA_PATTERN = "*final*v2*debug=0*.json"
CRAFTAX_HUMAN_DATA_PATTERN = "*final*v2*debug=0*.json"

DIRECTORY = "/Users/wilka/git/research/preplay_results"

JAXMAZE_DATA_DIR = os.path.join(DIRECTORY, "data_jaxmaze")
CRAFTAX_DATA_DIR = os.path.join(DIRECTORY, "data_craftax")

JAXMAZE_USER_DIR = os.path.join(DIRECTORY, "data_jaxmaze", "human_data")
CRAFTAX_USER_DIR = os.path.join(DIRECTORY, "data_craftax", "human_data")

JAXMAZE_RESULTS_DIR = os.path.join(DIRECTORY, "jaxmaze_results")
CRAFTAX_RESULTS_DIR = os.path.join(DIRECTORY, "craftax_results")

PAPER_STATS_FILE = os.path.join(DIRECTORY, "paper_stats.json")
PAPER_STATS_MODEL_FILE = os.path.join(DIRECTORY, "paper_stats_model.yaml")
