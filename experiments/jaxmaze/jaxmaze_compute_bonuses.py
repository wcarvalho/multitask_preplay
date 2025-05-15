"""
Compute bonuses for people.

1. get files
2. load files
3. for each file, look at the evaluation episode
  - maybe easiest to directly make a dataframe?
  - then filter based on eval=True
  - then see if time of full episode is < some metric (this will be based on maze)
  - based on different values, compute different bonuses
  - will pay between {0 and 3}
4. store the results in a csv file
"""

import os.path
import glob
import polars as pl
from analysis import housemaze_user_data


def main(files, outfile, debug=False):
  # files = 'jaxemaze_data/*.json'
  valid_files = list(set(glob.glob(files)))
  print(f"Found {len(valid_files)} files")

  user_df = housemaze_user_data.get_human_data(
    valid_files,
    overwrite_episode_data=False,
    overwrite_episode_info=True,
    load_df_only=True,
    require_finished=False,
    debug=debug,
  )

  bonus_values = {}

  # Handle manipulation bonuses
  for manipulation in range(0, 4):
    eval_df = user_df.filter(eval=True, manipulation=manipulation)
    eval_df = eval_df.with_columns(
      pl.col("total_rt").lt(20).cast(pl.Int64).alias("bonus")
    )

    # Group by user_id and sum bonuses
    bonus_sum = eval_df.group_by("worker_id").agg(pl.sum("bonus"))

    for row in bonus_sum.iter_rows(named=True):
      user_id = row["worker_id"]
      bonus = row["bonus"]
      bonus_values[user_id] = bonus_values.get(user_id, 0) + bonus

  # Handle juncture mazes
  settings = ["short", "long"]
  blind_options = [True, False]

  def maze_names(setting, blind):
    condition1 = (
      f"big_m4_maze_{setting}_eval_same_blind"
      if blind
      else f"big_m4_maze_{setting}_eval_same"
    )
    condition2 = (
      f"big_m4_maze_{setting}_eval_diff_blind"
      if blind
      else f"big_m4_maze_{setting}_eval_diff"
    )
    return condition1, condition2

  # Define thresholds for each setting
  time_thresholds = {"short": 6, "long": 10}

  for setting in settings:
    for blind in blind_options:
      condition1, condition2 = maze_names(setting, blind)

      # Process each suffix
      for suffix in ["F,F", "T,F", "F,T", "T,T"]:
        # Get the threshold based on setting
        threshold = time_thresholds[setting]

        # Filter for each maze pattern
        for condition in [condition1, condition2]:
          maze_pattern = f"{condition}_{suffix}"

          # Filter dataframe for this specific maze pattern using polars syntax
          maze_df = user_df.filter(pl.col("maze").str.contains(maze_pattern))

          if maze_df.height > 0:
            # Add bonus column based on threshold
            maze_df = maze_df.with_columns(
              pl.col("total_rt").lt(threshold).cast(pl.Int64).alias("bonus")
            )

            # Group by user_id and sum bonuses
            maze_bonus_sum = maze_df.group_by("worker_id").agg(pl.sum("bonus"))

            for row in maze_bonus_sum.iter_rows(named=True):
              user_id = row["worker_id"]
              bonus = row["bonus"]
              bonus_values[user_id] = bonus_values.get(user_id, 0) + bonus

  bonus_values = {k: 3.0 / 4.0 * v for k, v in bonus_values.items()}
  # Convert the bonus values dictionary to a dataframe and save to CSV
  bonus_df = pl.DataFrame(
    {"worker_id": list(bonus_values.keys()), "total_bonus": list(bonus_values.values())}
  )

  # Write to CSV
  bonus_df.write_csv(outfile, include_header=False)
  print(f"Wrote bonus data to {outfile}")


if __name__ == "__main__":
  # command line arg which specifies a pattern for files

  DEBUG = False
  RESULTS_DIR = "/Users/wilka/git/research/"
  USER_RESULTS_DIR = os.path.join(
    RESULTS_DIR, "preplay_results/jaxmaze_user_data/user_data/exps"
  )

  data_pattern = "final*v2*"
  files = f"{USER_RESULTS_DIR}/*{data_pattern}*.json"

  if DEBUG:
    outfile = "final_v2_bonuses_debug.csv"
  else:
    outfile = "final_v2_bonuses.csv"
  main(files, outfile, debug=DEBUG)
