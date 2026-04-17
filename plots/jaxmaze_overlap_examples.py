"""Plot JaxMaze overlap analysis examples showing train/test path comparisons.

Generates above/below threshold overlap visualizations for human and model data
across two-paths and shortcut manipulations at various overlap thresholds.

NOTE on "closest to threshold" sorting: When selecting N samples closest to a
threshold, sort DESCENDING (reverse=True) for below-threshold and ASCENDING for
above-threshold before slicing [:N]. A plain ascending sort picks the FURTHEST
samples from the threshold, not the closest.

python plots/jaxmaze_overlap_examples.py --manipulations two_paths --models human --thresholds .5
python plots/jaxmaze_overlap_examples.py --manipulations shortcut --models dfs human --thresholds .7
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import shutil

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from analysis import vis_utils

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def _save_figure(fig, filename, directory=None):
  directory = directory or OUTPUT_DIR
  os.makedirs(directory, exist_ok=True)
  filepath = os.path.join(directory, f"{filename}.pdf")
  plt.savefig(filepath, bbox_inches="tight", dpi=300)
  print(f"Saved figure to {filepath}")
  plt.close(fig)


def _plot_overlap_histogram(df, manipulation_id, test_maze, model):
  """Plot overlap distribution for a manipulation+model combo. Uses all block names."""
  all_eval = df.filter(eval=True, manipulation=manipulation_id, world=test_maze)
  overlaps = all_eval["overlap"].drop_nulls().to_list()
  if not overlaps:
    return
  output_dir = os.path.join(OUTPUT_DIR, "jaxmaze_overlap_examples")
  fig, ax = plt.subplots(figsize=(6, 4))
  ax.hist(overlaps, bins=30, edgecolor="black", alpha=0.7)
  ax.set_xticks(np.arange(0, 1.1, 0.1))
  ax.set_xlabel("Overlap")
  ax.set_ylabel("Count")
  ax.set_title(f"{model} overlap ({manipulation_id})")
  fig.tight_layout()
  _save_figure(fig, f"{manipulation_id}_{model}_overlap", directory=output_dir)


def _split_path_by_overlap(train_row, test_row):
  """Split train and test paths into overlap vs non-overlap segments.

  Returns dict with keys:
      train_only_pos/act, test_only_pos/act, overlap_pos/act,
      test_start, train_end, test_end (goal positions).
  """
  train_pos_all = vis_utils.parse_positions_string(train_row["positions"])
  test_pos_all = vis_utils.parse_positions_string(test_row["positions"])
  train_act = vis_utils.parse_jax_array_string(train_row["actions"]).astype(int)
  test_act = vis_utils.parse_jax_array_string(test_row["actions"]).astype(int)

  # Positions may have one more entry than actions (final pos has no action).
  # Truncate to match for arrow drawing.
  train_pos = train_pos_all[: len(train_act)]
  test_pos = test_pos_all[: len(test_act)]

  test_visited = set(map(tuple, test_pos_all))

  # Split train path: overlap (cell also in test) vs train-only
  train_only_mask = np.array([tuple(p) not in test_visited for p in train_pos])
  overlap_mask = ~train_only_mask

  # Test-only: cells not in train
  train_visited = set(map(tuple, train_pos_all))
  test_only_mask = np.array([tuple(p) not in train_visited for p in test_pos])

  return dict(
    train_only_pos=train_pos[train_only_mask],
    train_only_act=train_act[train_only_mask],
    test_only_pos=test_pos[test_only_mask],
    test_only_act=test_act[test_only_mask],
    overlap_pos=train_pos[overlap_mask],
    overlap_act=train_act[overlap_mask],
    test_start=tuple(test_pos_all[0]),
    train_end=tuple(train_pos_all[-1]),
    test_end=tuple(test_pos_all[-1]),
  )


def _make_summary_pdf(below_samples, above_samples, threshold, output_dir, n_summary=5):
  """Create a summary PDF with maze images and colored arrows.

  Layout: 2 rows x n_summary columns.
    Row 0 (below threshold): overlap increases left-to-right.
    Row 1 (above threshold): overlap increases left-to-right.

  Arrow colors: blue=train-only, red=test-only, white=overlap.
  Markers: white star=start, blue square=train goal, red square=test goal.
  """
  from housemaze import renderer
  from matplotlib.lines import Line2D
  from matplotlib.patches import FancyArrowPatch

  # Pick n closest to threshold: descending for below (highest first), ascending for above
  below = sorted(below_samples, key=lambda s: s["overlap"], reverse=True)[:n_summary]
  below.sort(key=lambda s: s["overlap"])  # re-sort ascending for L→R display
  above = sorted(above_samples, key=lambda s: s["overlap"])[:n_summary]

  ncols = max(len(below), len(above))
  if ncols == 0:
    return

  fig, axs = plt.subplots(2, ncols, figsize=(2.5 * ncols, 5))
  if ncols == 1:
    axs = axs[:, np.newaxis]

  for r in range(2):
    for c in range(ncols):
      axs[r, c].axis("off")

  def _plot_overlap_arrows(ax, sample):
    train_row = sample["train"].row(0, named=True)
    test_row = sample["test"].row(0, named=True)
    image, maze_h, maze_w = vis_utils.get_jaxmaze_env_image(
      train_row["world"], train_row["block_name"]
    )
    paths = _split_path_by_overlap(train_row, test_row)

    # Draw maze image
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])

    # Overlay arrows: blue=train-only, red=test-only, white=overlap
    for pos_key, act_key, color in [
      ("train_only_pos", "train_only_act", "#4484CE"),
      ("test_only_pos", "test_only_act", "#E8524A"),
      ("overlap_pos", "overlap_act", "white"),
    ]:
      if len(paths[pos_key]) > 0:
        renderer.place_arrows_on_image(
          image,
          paths[pos_key],
          paths[act_key],
          maze_h,
          maze_w,
          arrow_scale=5,
          arrow_color=color,
          ax=ax,
          plot_image=False,
        )

    # Compute pixel coords for markers
    image_h, image_w = image.shape[:2]
    scale_y = image_h // (maze_h + 2)
    scale_x = image_w // (maze_w + 2)
    offset_y = (image_h - scale_y * maze_h) // 2
    offset_x = (image_w - scale_x * maze_w) // 2

    def _to_pixel(rc):
      return offset_x + (rc[1] + 0.5) * scale_x, offset_y + (rc[0] + 0.5) * scale_y

    # White star at test start
    sx, sy = _to_pixel(paths["test_start"])
    ax.plot(
      sx,
      sy,
      marker="*",
      color="white",
      markersize=10,
      markeredgecolor="black",
      markeredgewidth=0.5,
      zorder=4,
    )

    # Blue square at train goal, red square at test goal
    tx, ty = _to_pixel(paths["train_end"])
    ax.plot(
      tx,
      ty,
      marker="s",
      color="#4484CE",
      markersize=8,
      markeredgecolor="black",
      markeredgewidth=0.5,
      zorder=4,
    )
    ex, ey = _to_pixel(paths["test_end"])
    ax.plot(
      ex,
      ey,
      marker="s",
      color="#E8524A",
      markersize=8,
      markeredgecolor="black",
      markeredgewidth=0.5,
      zorder=4,
    )

    ax.set_title(f"{sample['overlap']:.2f}", fontsize=9)
    ax.axis("off")

  # Row 0: below threshold
  for i, sample in enumerate(below):
    _plot_overlap_arrows(axs[0, i], sample)

  # Row 1: above threshold
  for i, sample in enumerate(above):
    _plot_overlap_arrows(axs[1, i], sample)

  # Row labels
  axs[0, 0].text(
    -0.15,
    0.5,
    f"Below {threshold}",
    transform=axs[0, 0].transAxes,
    ha="right",
    va="center",
    fontsize=11,
    fontweight="bold",
    rotation=90,
  )
  axs[1, 0].text(
    -0.15,
    0.5,
    f"Above {threshold}",
    transform=axs[1, 0].transAxes,
    ha="right",
    va="center",
    fontsize=11,
    fontweight="bold",
    rotation=90,
  )

  # Legend
  legend_elements = [
    Line2D(
      [0],
      [0],
      marker="*",
      color="w",
      markerfacecolor="white",
      markeredgecolor="black",
      markersize=10,
      label="Start",
      linestyle="None",
    ),
    FancyArrowPatch(
      (0, 0), (1, 0), arrowstyle="->", color="#4484CE", label="Train path"
    ),
    FancyArrowPatch(
      (0, 0), (1, 0), arrowstyle="->", color="#E8524A", label="Test path"
    ),
    FancyArrowPatch((0, 0), (1, 0), arrowstyle="->", color="white", label="Overlap"),
    Line2D(
      [0],
      [0],
      marker="s",
      color="w",
      markerfacecolor="#4484CE",
      markeredgecolor="black",
      markersize=8,
      label="Train goal",
      linestyle="None",
    ),
    Line2D(
      [0],
      [0],
      marker="s",
      color="w",
      markerfacecolor="#E8524A",
      markeredgecolor="black",
      markersize=8,
      label="Test goal",
      linestyle="None",
    ),
  ]
  fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=6,
    fontsize=8,
    frameon=True,
    bbox_to_anchor=(0.5, -0.02),
  )

  fig.tight_layout(rect=[0, 0.05, 1, 1])
  _save_figure(fig, "summary", directory=output_dir)


def visualize_examples_by_reuse(
  df,
  manipulation_id,
  threshold,
  model,
  n_search=10,
  n_examples=10,
  train_maze=None,
  test_maze=None,
  reverse_above=False,
  reverse_below=False,
  n_summary=5,
  summary_only=False,
):
  """Visualize examples classified by reuse (1 or 0) based on threshold."""
  manipulation_name = manipulation_id

  output_dir = os.path.join(
    OUTPUT_DIR,
    "jaxmaze_overlap_examples",
    f"{manipulation_name}_{threshold}",
    model,
  )

  if not summary_only and os.path.exists(output_dir):
    print(f"Removing old data from {output_dir}...")
    shutil.rmtree(output_dir)

  os.makedirs(output_dir, exist_ok=True)

  samples = []

  user_ids = df["user_id"].unique().to_list()
  block_names = df["block_name"].unique().to_list()
  for user_id in user_ids:
    for block_name in block_names:
      test_initial = df.filter(
        eval=True,
        user_id=user_id,
        manipulation=manipulation_id,
        block_name=block_name,
        world=test_maze,
      ).sort("overlap", descending=False)

      if len(test_initial) == 0:
        continue

      ntest = len(test_initial)
      for test_episode_idx in range(min(n_search, ntest)):
        start_pos_val = test_initial["start_pos"].to_list()[test_episode_idx]
        block_name_val = test_initial["block_name"].to_list()[test_episode_idx]
        task_set_val = test_initial["task_set"].to_list()[test_episode_idx]

        filters = dict(
          start_pos=start_pos_val,
          block_name=block_name_val,
          user_id=user_id,
          task_set=task_set_val,
        )

        test = df.filter(
          eval=True,
          manipulation=manipulation_id,
          world=test_maze,
          **filters,
        )
        corresponding_train_episode_idx = test[
          "corresponding_train_episode_idx"
        ].to_list()[test_episode_idx]

        train = df.filter(
          global_episode_idx=corresponding_train_episode_idx,
        )

        if len(train) == 0 or len(test) == 0:
          continue

        overlap_value = test["overlap"].to_list()[0]
        reuse = int(overlap_value >= threshold)

        samples.append(
          {
            "key": user_id,
            "train": train,
            "test": test,
            "overlap": overlap_value,
            "reuse": reuse,
            "filters": filters,
            "block_name": block_name_val,
          }
        )

  above_threshold_samples = [s for s in samples if s["overlap"] >= threshold]
  below_threshold_samples = [s for s in samples if s["overlap"] < threshold]

  if len(above_threshold_samples) == 0 and len(below_threshold_samples) == 0:
    raise RuntimeError("No examples found")

  print(f"{model} - {manipulation_name} - threshold {threshold}:")
  print(f"  Found {len(above_threshold_samples)} examples with overlap >= {threshold}")
  print(f"  Found {len(below_threshold_samples)} examples with overlap < {threshold}")

  # Select n closest to threshold on each side
  above_threshold_samples.sort(key=lambda x: x["overlap"])
  above_threshold_samples = above_threshold_samples[:n_examples]

  below_threshold_samples.sort(key=lambda x: x["overlap"], reverse=True)
  below_threshold_samples = below_threshold_samples[:n_examples]

  # Sort for display (default: both ascending)
  above_threshold_samples.sort(key=lambda x: x["overlap"], reverse=reverse_above)
  below_threshold_samples.sort(key=lambda x: x["overlap"], reverse=reverse_below)

  # Combine: below first (ascending), then above (ascending) — continuous overlap order
  all_ordered = [(s, "below") for s in below_threshold_samples] + [
    (s, "above") for s in above_threshold_samples
  ]

  if not summary_only:
    is_human = model == "human"
    if is_human:
      output_dir_rt = output_dir + "_rt"
      if os.path.exists(output_dir_rt):
        shutil.rmtree(output_dir_rt)
      os.makedirs(output_dir_rt, exist_ok=True)

    for idx, (sample, label) in enumerate(all_ordered):
      k = sample["key"]
      train_filtered = sample["train"]
      test_filtered = sample["test"]
      overlap_value = sample["overlap"]
      reuse_value = sample["reuse"]

      train_row = train_filtered.row(0, named=True)
      test_row = test_filtered.row(0, named=True)
      train_success = train_filtered["success"].to_list()[0]
      test_success = test_filtered["success"].to_list()[0]

      if is_human:
        train_title = f"Train (user: {k}, success: {train_success})"
      else:
        train_title = f"Train (seed: {k}, success: {train_success})"
      test_title = f"Test (success: {test_success}, overlap: {overlap_value:.3f}, reuse: {reuse_value})"
      filename = f"{idx:02d}_{label}_{overlap_value:.3f}"

      # 2-column version (train + test)
      fig, axs = plt.subplots(1, 2, figsize=(10, 5))
      vis_utils.visualize_jaxmaze_row(train_row, ax_image=axs[0])
      vis_utils.visualize_jaxmaze_row(test_row, ax_image=axs[1])
      axs[0].set_title(train_title)
      axs[1].set_title(test_title)
      fig.tight_layout()
      _save_figure(fig, filename, directory=output_dir)

      # 3-column version (train + test + RT) — human only
      if is_human:
        fig_rt, axs_rt = plt.subplots(1, 3, figsize=(15, 5))
        vis_utils.visualize_jaxmaze_row(train_row, ax_image=axs_rt[0])
        vis_utils.visualize_jaxmaze_row(test_row, ax_image=axs_rt[1], ax_rt=axs_rt[2])
        axs_rt[0].set_title(train_title)
        axs_rt[1].set_title(test_title)
        axs_rt[2].set_title("Test RT")
        fig_rt.tight_layout()
        _save_figure(fig_rt, filename, directory=output_dir_rt)

  # Summary PDF with path-overlap heatmaps
  _make_summary_pdf(
    below_threshold_samples,
    above_threshold_samples,
    threshold,
    output_dir,
    n_summary=n_summary,
  )


def main():
  import data_configs

  parser = argparse.ArgumentParser(
    description="Generate JaxMaze overlap analysis visualizations"
  )
  parser.add_argument(
    "--models",
    nargs="+",
    default=["preplay", "dfs", "human"],
    choices=["human", "preplay", "usfa", "dyna", "qlearning", "bfs", "dfs"],
    help="Models to analyze (can specify multiple). Use 'human' for human data.",
  )
  parser.add_argument(
    "--manipulations",
    nargs="+",
    default=["two_paths", "shortcut"],
    choices=["two_paths", "shortcut"],
    help="Manipulations to analyze (can specify multiple).",
  )
  parser.add_argument(
    "--thresholds",
    nargs="+",
    type=float,
    default=[0.3, 0.5, 0.6, 0.7],
    help="Overlap thresholds to analyze (default: 0.3 0.5 0.6 0.7)",
  )
  parser.add_argument(
    "--n",
    type=int,
    default=20,
    help="Number of examples to generate for each reuse category (default: 10)",
  )
  parser.add_argument(
    "--reverse",
    action="store_true",
    help="Reverse sort order for both above and below threshold (default: ascending)",
  )
  parser.add_argument(
    "--n_summary",
    type=int,
    default=5,
    help="Number of examples per side in the summary heatmap PDF (default: 5)",
  )
  parser.add_argument(
    "--summary_only",
    action="store_true",
    help="Only generate summary heatmap PDFs, skip individual example plots",
  )
  parser.add_argument(
    "--dist",
    action="store_true",
    help="Only generate overlap distribution histograms, skip example visualizations",
  )

  args = parser.parse_args()

  thresholds = args.thresholds
  reverse = args.reverse

  manipulations = [
    {
      "name": "two_paths",
      "manipulation_id": "paths",
      "train_maze": "big_m3_maze1",
      "test_maze": "big_m3_maze1",
    },
    {
      "name": "shortcut",
      "manipulation_id": "shortcut",
      "train_maze": "big_m1_maze3",
      "test_maze": "big_m1_maze3_shortcut",
    },
  ]
  if args.manipulations:
    manipulations = [m for m in manipulations if m["name"] in args.manipulations]

  requested_models = [m for m in args.models if m != "human"]

  model2algo = dict(
    preplay_new="preplay",
  )

  if requested_models:
    print("Loading model data...")
    model_df = data_configs.load_dataframes("jaxmaze", models=requested_models)

    for manipulation in manipulations:
      manip_name = manipulation["name"]
      manip_id = manipulation["manipulation_id"]
      train_maze = manipulation["train_maze"]
      test_maze = manipulation["test_maze"]

      for model in requested_models:
        print(f"\nProcessing {model} model for {manip_name}")
        model_df_filtered = model_df.filter(algo=model2algo.get(model, model))
        _plot_overlap_histogram(model_df_filtered, manip_id, test_maze, model)

        if args.dist:
          continue

        for threshold in thresholds:
          print(f"Processing {manip_name} with threshold {threshold} for {model}")
          visualize_examples_by_reuse(
            df=model_df_filtered,
            manipulation_id=manip_id,
            threshold=threshold,
            model=model,
            train_maze=train_maze,
            test_maze=test_maze,
            n_examples=args.n,
            reverse_above=reverse,
            reverse_below=reverse,
            n_summary=args.n_summary,
            summary_only=args.summary_only,
          )

  if "human" in args.models:
    print("Loading human data...")
    user_df = pl.read_parquet(data_configs.get_dataframe_path("jaxmaze", "human"))

    for manipulation in manipulations:
      manip_name = manipulation["name"]
      manip_id = manipulation["manipulation_id"]
      train_maze = manipulation["train_maze"]
      test_maze = manipulation["test_maze"]

      _plot_overlap_histogram(user_df, manip_id, test_maze, "human")

      if args.dist:
        continue

      for threshold in thresholds:
        print(f"\nProcessing {manip_name} with threshold {threshold} for humans")
        visualize_examples_by_reuse(
          df=user_df,
          manipulation_id=manip_id,
          threshold=threshold,
          model="human",
          train_maze=train_maze,
          test_maze=test_maze,
          n_examples=args.n,
          reverse_above=reverse,
          reverse_below=reverse,
          n_summary=args.n_summary,
          summary_only=args.summary_only,
        )

  print("Analysis complete!")


if __name__ == "__main__":
  main()
