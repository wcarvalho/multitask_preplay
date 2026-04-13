import os

from data_processing import process_user_data
from analysis import analysis_utils
from analysis.jaxmaze_analysis import filter_users_by_success
import numpy as np
import matplotlib.pyplot as plt
from analysis import vis_utils
from plot_configs import default_colors
import polars as pl

# Configuration
N_ROWS = 5  # Number of episodes to show per condition

global_df = process_user_data.get_jaxmaze_human_data(
  load_df_only=False,
)

df, _ = filter_users_by_success(
  global_df.filter(
    manipulation="paths",
    world="big_m3_maze1",
    eval=True,
    tell_reuse=1,
    eval_shares_start_pos=True,
  ),
  analysis_name="path_reuse_results",
)

df = analysis_utils.add_reuse_column(df, "reuse", 0.5)

reuse_df = df.filter(reuse=1)
no_reuse_df = df.filter(reuse=0)

# Sort by first_rt and get top N for each condition
reuse_top = reuse_df.sort("first_rt").head(N_ROWS)
no_reuse_top = no_reuse_df.sort("first_rt").head(N_ROWS)


# Function to parse reaction times string
def parse_reaction_times(rt_string):
  # Remove brackets and split by whitespace
  rt_string = rt_string.strip("[]")
  return np.array([float(x) for x in rt_string.split()])


# Create figure with N_ROWS x 6 subplots
fig, axes = plt.subplots(N_ROWS, 6, figsize=(20, 3 * N_ROWS))
fig.suptitle(
  "Training/Test Episodes and Reaction Times for Fastest First RT Episodes", fontsize=16
)

# Process no_reuse (columns 0-2)
for i, row in enumerate(no_reuse_top.iter_rows(named=True)):
  # Get test episode using global_episode_idx
  test_global_idx = row["global_episode_idx"]
  test_episode = df.filter(global_episode_idx=test_global_idx).episodes[0]

  # Get corresponding training episode
  train_idx = row["corresponding_train_episode_idx"]
  episodes = global_df.filter(global_episode_idx=train_idx).episodes

  if episodes:
    # Plot training episode (column 0)
    ax_train = axes[i, 0]
    vis_utils.render_path(episodes[0], ax=ax_train)
    ax_train.set_title("Train Episode", fontsize=12)

  # Plot test episode (column 1)
  ax_test = axes[i, 1]
  vis_utils.render_path(test_episode, ax=ax_test)
  ax_test.set_title("Test Episode", fontsize=12)

  # Plot reaction times (column 2)
  reaction_times = parse_reaction_times(row["reaction_times"])
  first_rt = row["first_rt"]

  ax_rt = axes[i, 2]
  analysis_utils.plot_reaction_times(
    reaction_times,
    ax=ax_rt,
    color=default_colors["nice purple"],
    title=f"RTs (first={first_rt:.2f}s)",
    ylabel=(i == N_ROWS // 2),  # Only show ylabel for middle row
    show_xlabel=(i == N_ROWS - 1),  # Only show xlabel for bottom row
    ylim=[0, 2],
  )

# Process reuse (columns 3-5)
for i, row in enumerate(reuse_top.iter_rows(named=True)):
  # Get test episode using global_episode_idx
  test_global_idx = row["global_episode_idx"]
  test_episode = df.filter(global_episode_idx=test_global_idx).episodes[0]

  # Get corresponding training episode
  train_idx = row["corresponding_train_episode_idx"]
  episodes = global_df.filter(global_episode_idx=train_idx).episodes

  if episodes:
    # Plot training episode (column 3)
    ax_train = axes[i, 3]
    vis_utils.render_path(episodes[0], ax=ax_train)
    ax_train.set_title("Train Episode", fontsize=12)

  # Plot test episode (column 4)
  ax_test = axes[i, 4]
  vis_utils.render_path(test_episode, ax=ax_test)
  ax_test.set_title("Test Episode", fontsize=12)

  # Plot reaction times (column 5)
  reaction_times = parse_reaction_times(row["reaction_times"])
  first_rt = row["first_rt"]

  ax_rt = axes[i, 5]
  analysis_utils.plot_reaction_times(
    reaction_times,
    ax=ax_rt,
    color=default_colors["bluish green"],
    title=f"RTs (first={first_rt:.2f}s)",
    ylabel=False,
    show_xlabel=(i == N_ROWS - 1),  # Only show xlabel for bottom row
    ylim=[0, 2],
  )

# Add column group labels
fig.text(
  0.25,
  0.98,
  "New Path",
  ha="center",
  fontsize=14,
  weight="bold",
  transform=fig.transFigure,
)
fig.text(
  0.75,
  0.98,
  "Partial Reuse",
  ha="center",
  fontsize=14,
  weight="bold",
  transform=fig.transFigure,
)

# Add vertical lines to separate the two groups
fig.add_artist(
  plt.Line2D(
    [0.5, 0.5], [0.05, 0.95], transform=fig.transFigure, color="gray", linewidth=2
  )
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
os.makedirs("random/plots", exist_ok=True)
plt.savefig("random/plots/min_first_rts.png", dpi=300, bbox_inches="tight")
