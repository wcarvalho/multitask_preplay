"""
This script generates figures for the JaxMaze overlap analysis.

Call from root directory with:
python figures_supplemental/craftax_overlap_analysis.py

The figures are saved in the {data_configs.CRAFTAX_OVERLAP_ANALYSIS_DIR} directory."""

import sys
import os

# add this directory to the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import matplotlib.pyplot as plt
import numpy as np
import jax
from simulations import craftax_utils
from tqdm import tqdm
import numpy as np

# from data_processing import utils_craftax as utils
from data_processing.utils_craftax import create_maps, OPTIMAL_TEST_PATHS
from data_processing.utils import compute_overlap, compute_average_vector_from_end
from figures import figure_utils
import data_configs
import nicewebrl


# For Jupyter notebook display
try:
  from IPython.display import display

  JUPYTER_AVAILABLE = True
except ImportError:
  JUPYTER_AVAILABLE = False


def make_image_path_panel(episode, ax, title):
  first_state = jax.tree_util.tree_map(lambda x: x[0], episode.timesteps.state)
  image = craftax_utils.render_fn(first_state, show_agent=False)
  path = episode.positions
  actions = craftax_utils.actions_from_path(path)
  craftax_utils.place_arrows_on_image(
    image=image,
    positions=path,
    actions=actions,
    maze_height=first_state.map.shape[1],
    maze_width=first_state.map.shape[2],
    ax=ax,
    display_image=True,
    arrow_color="red",
    show_path_length=False,
    start_color="red",
  )
  ax.set_title(title)
  ax.axis("off")


def parse_and_compute_overlaps(
  df: nicewebrl.DataFrame,
  manipulation: str,
  model: str,
  world_seed: int = None,
  debug: int = -1,
  manual_user_id: int = None,
  max_tries: int = 1000000,
  **kwargs,
):
  """
  Parse dataframe and compute all overlap values once.

  Returns:
    List of sample dictionaries with computed overlap values
  """
  samples = []

  debug = debug if manual_user_id is None else 0
  user_ids = df["user_id"].unique().to_list()
  block_names = df["block_name"].unique().to_list()

  if debug != -1:
    if manual_user_id is not None:
      user_ids = [manual_user_id]
    else:
      user_ids = [user_ids[debug]]

  for block_name in tqdm(block_names, desc="Block Names"):
    user_ids_to_process = user_ids[:max_tries]

    for user_id in tqdm(user_ids_to_process, desc="User IDs"):
      # Get test examples for this user
      test_initial = df.filter(
        eval=True,
        user_id=user_id,
        manipulation=manipulation,
        block_name=block_name,
      ).sort("overlap", descending=True)

      if len(test_initial.episodes) == 0:
        continue

      # Process each test episode
      for test_episode_idx in range(min(len(test_initial.episodes), max_tries)):
        # Get values for filtering
        start_pos_val = test_initial["start_pos"].to_list()[test_episode_idx]
        block_name_val = test_initial["block_name"].to_list()[test_episode_idx]
        task_set_val = test_initial["task_set"].to_list()[test_episode_idx]
        world_seed_val = test_initial["world"].to_list()[test_episode_idx]

        # Skip if we're filtering by world seed and this doesn't match
        if world_seed is not None and str(world_seed_val) != str(world_seed):
          continue

        filters = dict(
          start_pos=start_pos_val,
          block_name=block_name_val,
          user_id=user_id,
          task_set=task_set_val,
        )

        # Get properly filtered test and train
        test = df.filter(
          eval=True,
          manipulation=manipulation,
          **filters,
        )
        corresponding_train_episode_idx = test[
          "corresponding_train_episode_idx"
        ].to_list()[test_episode_idx]
        train = df.filter(
          global_episode_idx=corresponding_train_episode_idx,
        )

        if len(train.episodes) == 0 or len(test.episodes) == 0:
          continue

        assert len(train.episodes) == 1

        # Calculate map overlap
        train_map = create_maps([train.episodes[0]], add_error=True)
        test_map = create_maps([test.episodes[0]], add_error=False)
        map_overlap_value = compute_overlap(train_map, test_map).mean()

        # calculate cosine similarity between train and test average vectors
        train_avg_vector, train_vectors = compute_average_vector_from_end(
          train.episodes[0].positions
        )
        test_avg_vector, test_vectors = compute_average_vector_from_end(
          test.episodes[0].positions
        )
        cosine = np.dot(train_avg_vector, test_avg_vector) / (
          np.linalg.norm(train_avg_vector) * np.linalg.norm(test_avg_vector) + 1e-5
        )

        optimal_test_path = OPTIMAL_TEST_PATHS[int(world_seed_val)]
        ratio_optimal_test_path = len(test.episodes[0].positions) / len(
          optimal_test_path
        )

        df_overlap_value = test["overlap"].to_list()[0]
        assert np.abs(df_overlap_value - map_overlap_value) < 1e-5

        samples.append(
          {
            "key": user_id,
            "train": train,
            "test": test,
            "train_map": train_map,
            "test_map": test_map,
            "cosine": cosine,
            "df_overlap": df_overlap_value,
            "best_idx": 0,
            "filters": filters,
            "world_seed": world_seed_val,
            "ratio_optimal_test_path": ratio_optimal_test_path,
          }
        )

        if debug != -1 or manual_user_id is not None:
          return samples

  return samples


def visualize_computations(data, model=None, threshold=None, cosine_threshold=0.5):
  train = data["train"]
  test = data["test"]
  train_map = data["train_map"]
  test_map = data["test_map"]
  # overlap_value = data["df_overlap"]
  overlap_value = data["df_overlap"]
  ratio_optimal_test_path = data["ratio_optimal_test_path"]
  if train_map.ndim > 2:
    train_map = train_map[0]
  if test_map.ndim > 2:
    test_map = test_map[0]

  # Extract information for titles
  k = data["key"]
  sample_world_seed = data.get("world_seed", None)

  # Get success information
  train_success = (
    train["success"].to_list()[0] if len(train["success"].to_list()) > 0 else "N/A"
  )
  test_success = (
    test["success"].to_list()[0] if len(test["success"].to_list()) > 0 else "N/A"
  )

  # Calculate reuse if threshold is provided
  reuse_value = int(overlap_value >= threshold) if threshold is not None else "N/A"

  # Create figure with 3x2 subplots (added one more row)
  fig, axes = plt.subplots(3, 2, figsize=(12, 15))

  # Generate detailed titles like the original
  if model == "human":
    train_title = f"Train (user: {k}, success: {train_success})"
  else:
    model_key = "seed"
    train_title = f"Train ({model_key}: {k}, success: {train_success})"

  test_title = f"Test success: {test_success}\n reuse: {reuse_value}, path_length: {len(test.episodes[0].positions)}, ratio: {ratio_optimal_test_path:.3f},"

  if sample_world_seed is not None:
    test_title = f"World {sample_world_seed} - " + test_title

  # Row 0: Path visualizations
  # Column 0: Best train path
  with jax.disable_jit():
    make_image_path_panel(train.episodes[0], axes[0, 0], train_title)

    # Column 1: Test path
    make_image_path_panel(test.episodes[0], axes[0, 1], test_title)

  # Row 1: Heat maps
  # Column 0: Best train map
  im1 = axes[1, 0].imshow(train_map, cmap="viridis")
  axes[1, 0].set_title("Best Train Map")
  axes[1, 0].axis("off")
  plt.colorbar(im1, ax=axes[1, 0], shrink=0.8)

  # Column 1: Test map
  test_map = test_map + train_map * test_map  # to show overlap
  im2 = axes[1, 1].imshow(test_map, cmap="viridis")
  axes[1, 1].set_title("Test Map")
  axes[1, 1].axis("off")
  plt.colorbar(im2, ax=axes[1, 1], shrink=0.8)

  # Row 2: Average vectors from last half of trajectory
  # Get positions for train and test
  train_positions = train.episodes[0].positions[:-1]
  test_positions = test.episodes[0].positions[:-1]

  # Compute average vectors
  train_avg_vector, train_vectors = compute_average_vector_from_end(train_positions)
  test_avg_vector, test_vectors = compute_average_vector_from_end(test_positions)

  # Calculate dot product
  cosine = np.dot(train_avg_vector, test_avg_vector) / (
    np.linalg.norm(train_avg_vector) * np.linalg.norm(test_avg_vector)
  )

  # Visualize train average vector
  axes[2, 0].quiver(
    0,
    0,
    train_avg_vector[0],
    train_avg_vector[1],
    scale=1,
    scale_units="xy",
    angles="xy",
    color="blue",
    width=0.01,
  )
  axes[2, 0].set_xlim(-2, 2)
  axes[2, 0].set_ylim(-2, 2)
  axes[2, 0].set_aspect("equal")
  axes[2, 0].grid(True, alpha=0.3)
  axes[2, 0].axhline(y=0, color="k", linewidth=0.5)
  axes[2, 0].axvline(x=0, color="k", linewidth=0.5)
  axes[2, 0].set_title(
    f"Train Avg Vector (last half): ({train_avg_vector[0]:.3f}, {train_avg_vector[1]:.3f})"
  )

  # Visualize test average vector
  axes[2, 1].quiver(
    0,
    0,
    test_avg_vector[0],
    test_avg_vector[1],
    scale=1,
    scale_units="xy",
    angles="xy",
    color="red",
    width=0.01,
  )
  axes[2, 1].set_xlim(-2, 2)
  axes[2, 1].set_ylim(-2, 2)
  axes[2, 1].set_aspect("equal")
  axes[2, 1].grid(True, alpha=0.3)
  axes[2, 1].axhline(y=0, color="k", linewidth=0.5)
  axes[2, 1].axvline(x=0, color="k", linewidth=0.5)
  axes[2, 1].set_title(
    f"Test Avg Vector (last half): ({test_avg_vector[0]:.3f}, {test_avg_vector[1]:.3f})"
  )

  # Display overlap info as main title
  title = f"user: {k}"
  overlap_reuse_value = int(overlap_value >= threshold)
  cosine_reuse_value = int(cosine >= cosine_threshold)
  title += f"\noverlap: {overlap_value:.3f}, reuse: {overlap_reuse_value}"
  title += f"\ncosine: {cosine:.3f}, reuse: {cosine_reuse_value}"
  title += f"\njoint reuse: {(cosine_reuse_value + overlap_reuse_value) > 1}"

  fig.suptitle(
    title,
    fontsize=14,
  )

  # Apply tight layout for better spacing
  plt.tight_layout()

  return fig


def visualize_from_samples(
  samples: list,
  threshold: float,
  model: str,
  manipulation: str,
  cosine_threshold: float = 0.5,
  world_seed: int = None,
  n_examples: int = 10,
  display_image: bool = False,
  clear_output_dir: bool = True,
  **kwargs,
):
  """
  Visualize examples from pre-computed samples based on threshold.

  Args:
    samples: Pre-computed list of sample dictionaries with overlap values
    threshold: Threshold value for determining reuse
    model: The model or 'human'
    manipulation: The manipulation type (e.g., "paths")
    world_seed: Specific world seed to analyze (optional)
    n_examples: Number of examples to generate for each reuse value
    display_image: Whether to display images
    clear_output_dir: Whether to clear the output directory
  """
  # Create directory structure
  output_dir = f"{data_configs.CRAFTAX_OVERLAP_ANALYSIS_DIR}/{manipulation}_{threshold}_{cosine_threshold}/{model}"
  if world_seed is not None:
    output_dir = f"{output_dir}_world_{world_seed}"

  # Delete existing directory to remove old data
  import shutil

  if os.path.exists(output_dir) and clear_output_dir:
    print(f"Removing old data from {output_dir}...")
    shutil.rmtree(output_dir)

  # Create fresh directory
  os.makedirs(output_dir, exist_ok=True)

  # Update reuse values based on current threshold
  for sample in samples:
    sample["reuse"] = int(
      sample["df_overlap"] >= threshold and sample["cosine"] >= cosine_threshold
    )
  # Separate into above and below threshold examples
  above_threshold_samples = [s for s in samples if s["reuse"]]
  below_threshold_samples = [s for s in samples if not s["reuse"]]

  # Sort by overlap for better visualization
  above_threshold_samples.sort(
    key=lambda x: x["df_overlap"], reverse=False
  )  # Lower overlap first
  below_threshold_samples.sort(
    key=lambda x: x["df_overlap"], reverse=True
  )  # Highest overlap first

  print(f"{model} - {manipulation} - threshold {threshold}:")
  if world_seed is not None:
    print(f"  World seed: {world_seed}")
  print(f"  Found {len(above_threshold_samples)} examples with overlap >= {threshold}")
  print(f"  Found {len(below_threshold_samples)} examples with overlap < {threshold}")

  # Limit to n_examples
  above_threshold_samples = above_threshold_samples[:n_examples]
  below_threshold_samples = below_threshold_samples[:n_examples]

  # Create individual figures for above threshold examples
  for idx, sample in enumerate(above_threshold_samples):
    overlap_value = sample["df_overlap"]
    reuse_value = sample["reuse"]
    sample_world_seed = sample["world_seed"]
    ratio_optimal_test_path = sample["ratio_optimal_test_path"]
    above_ratio = ratio_optimal_test_path > 2.0

    # Use visualize_computations to create the comprehensive figure
    fig = visualize_computations(sample, model=model, threshold=threshold)

    # Save with numbered filename including overlap value
    filename = f"above_threshold_example_{idx + 1:02d}"
    if above_ratio:
      filename += "_above_ratio"

    if sample_world_seed is not None:
      filename += f"_world_{sample_world_seed}"
    filename += f"_overlap_{overlap_value:.3f}_reuse_{reuse_value}"

    figure_utils.save_figure(
      fig,
      filename,
      directory=output_dir if not above_ratio else f"{output_dir}_above_ratio",
    )

    if display_image:
      if JUPYTER_AVAILABLE:
        display(fig)
      else:
        plt.show()

    plt.close(fig)  # Close the figure to free memory

  # Create individual figures for below threshold examples
  for idx, sample in enumerate(below_threshold_samples):
    overlap_value = sample["df_overlap"]
    reuse_value = sample["reuse"]
    sample_world_seed = sample["world_seed"]
    ratio_optimal_test_path = sample["ratio_optimal_test_path"]
    above_ratio = ratio_optimal_test_path > 2.0

    # Use visualize_computations to create the comprehensive figure
    fig = visualize_computations(sample, model=model, threshold=threshold)

    # Save with numbered filename including overlap value
    filename = f"below_threshold_example_{idx + 1:02d}"
    if sample_world_seed is not None:
      filename += f"_world_{sample_world_seed}"
    filename += f"_overlap_{overlap_value:.3f}_reuse_{reuse_value}"

    figure_utils.save_figure(
      fig,
      filename,
      directory=output_dir if not above_ratio else f"{output_dir}_above_ratio",
    )

    if display_image:
      if JUPYTER_AVAILABLE:
        display(fig)
      else:
        plt.show()

    plt.close(fig)  # Close the figure to free memory


def visualize_examples_by_reuse(
  df: nicewebrl.DataFrame,
  manipulation: str,
  threshold: float,
  model: str,
  n_examples: int = 10,
  world_seed: int = None,
  display_image: bool = False,
  debug: int = -1,
  manual_user_id: int = None,
  clear_output_dir: bool = True,
  max_tries: int = 1000000,
):
  """
  Visualize examples classified by reuse (1 or 0) based on threshold.

  Args:
    df: Full dataframe (for humans) or filtered dataframe (for models)
    manipulation: The manipulation type (e.g., "paths")
    threshold: Threshold value for determining reuse
    model: The model or 'human'
    n_examples: Number of examples to generate for each reuse value
    world_seed: Specific world seed to analyze (optional)
  """

  # Create directory structure
  output_dir = (
    f"{data_configs.CRAFTAX_OVERLAP_ANALYSIS_DIR}/{manipulation}_{threshold}/{model}"
  )
  if world_seed is not None:
    output_dir = f"{output_dir}_world_{world_seed}"

  # Delete existing directory to remove old data
  import shutil

  if os.path.exists(output_dir) and clear_output_dir:
    print(f"Removing old data from {output_dir}...")
    shutil.rmtree(output_dir)

  # Create fresh directory
  os.makedirs(output_dir, exist_ok=True)

  # Different handling for human data vs model data
  samples = []

  debug = debug if manual_user_id is None else 0
  # if model == "human":
  # For humans, we need to carefully match train and test examples
  user_ids = df["user_id"].unique().to_list()
  if debug != -1:
    if manual_user_id is not None:
      user_ids = [manual_user_id]
    else:
      user_ids = [user_ids[debug]]

  user_ids = user_ids[:max_tries]
  for user_id in tqdm(user_ids, desc="User IDs"):
    # Get test examples for this user
    test_initial = df.filter(
      eval=True,
      user_id=user_id,
      manipulation=manipulation,
    ).sort("overlap", descending=True)

    if len(test_initial.episodes) == 0:
      continue

    # Process each test episode one by one
    for test_episode_idx in range(min(len(test_initial.episodes), max_tries)):
      # Get values for filtering
      start_pos_val = test_initial["start_pos"].to_list()[test_episode_idx]
      block_name_val = test_initial["block_name"].to_list()[test_episode_idx]
      task_set_val = test_initial["task_set"].to_list()[test_episode_idx]
      world_seed_val = test_initial["world"].to_list()[test_episode_idx]

      filters = dict(
        start_pos=start_pos_val,
        block_name=block_name_val,
        user_id=user_id,
        task_set=task_set_val,
      )

      # Get properly filtered test and train
      test = df.filter(
        eval=True,
        manipulation=manipulation,
        **filters,
      )
      corresponding_train_episode_idx = test[
        "corresponding_train_episode_idx"
      ].to_list()[test_episode_idx]
      train = df.filter(
        global_episode_idx=corresponding_train_episode_idx,
      )
      if len(train.episodes) == 0 or len(test.episodes) == 0:
        continue

      assert len(train.episodes) == 1
      # Calculate overlap
      train_map = create_maps([train.episodes[0]], add_error=True)
      test_map = create_maps([test.episodes[0]], add_error=False)

      overlap_value = compute_overlap(train_map, test_map).mean()

      optimal_test_path = OPTIMAL_TEST_PATHS[int(world_seed_val)]
      ratio_optimal_test_path = len(test.episodes[0].positions) / len(optimal_test_path)
      reuse_value = int(overlap_value >= threshold)
      df_overlap_value = test["df_overlap"].to_list()[0]
      samples.append(
        {
          "key": user_id,
          "train": train,
          "test": test,
          "train_map": train_map,
          "test_map": test_map,
          "df_overlap": overlap_value,
          "df_overlap": df_overlap_value,
          "best_idx": 0,
          "filters": filters,  # Store filters for reference
          "world_seed": world_seed_val,
          "ratio_optimal_test_path": ratio_optimal_test_path,
          "reuse": reuse_value,
        }
      )
      if debug != -1 or manual_user_id is not None:
        return samples[-1]

  # Separate into above and below threshold examples
  above_threshold_samples = [s for s in samples if s["df_overlap"] >= threshold]
  below_threshold_samples = [s for s in samples if s["df_overlap"] < threshold]

  # Sort by overlap for better visualization
  above_threshold_samples.sort(
    key=lambda x: x["df_overlap"], reverse=False
  )  # Lower overlap first
  below_threshold_samples.sort(
    key=lambda x: x["overlap"], reverse=True
  )  # Highest overlap first

  print(f"{model} - {manipulation} - threshold {threshold}:")
  if world_seed is not None:
    print(f"  World seed: {world_seed}")
  print(f"  Found {len(above_threshold_samples)} examples with overlap >= {threshold}")
  print(f"  Found {len(below_threshold_samples)} examples with overlap < {threshold}")

  # Limit to n_examples
  above_threshold_samples = above_threshold_samples[:n_examples]
  below_threshold_samples = below_threshold_samples[:n_examples]

  # Create individual figures for above threshold examples
  for idx, sample in enumerate(above_threshold_samples):
    overlap_value = sample["overlap"]
    reuse_value = sample["reuse"]
    sample_world_seed = sample["world_seed"]
    ratio_optimal_test_path = sample["ratio_optimal_test_path"]
    above_ratio = ratio_optimal_test_path > 2.0

    # Use visualize_computations to create the comprehensive figure
    fig = visualize_computations(sample, model=model, threshold=threshold)

    # Save with numbered filename including overlap value
    filename = f"above_threshold_example_{idx + 1:02d}"
    if above_ratio:
      filename += "_above_ratio"

    if sample_world_seed is not None:
      filename += f"_world_{sample_world_seed}"
    filename += f"_overlap_{overlap_value:.3f}_reuse_{reuse_value}"

    figure_utils.save_figure(
      fig,
      filename,
      directory=output_dir if not above_ratio else f"{output_dir}_above_ratio",
    )

    if display_image:
      if JUPYTER_AVAILABLE:
        display(fig)
      else:
        plt.show()

    plt.close(fig)  # Close the figure to free memory

  # Create individual figures for below threshold examples
  for idx, sample in enumerate(below_threshold_samples):
    overlap_value = sample["overlap"]
    reuse_value = sample["reuse"]
    sample_world_seed = sample["world_seed"]
    ratio_optimal_test_path = sample["ratio_optimal_test_path"]
    above_ratio = ratio_optimal_test_path > 2.0

    # Use visualize_computations to create the comprehensive figure
    fig = visualize_computations(sample, model=model, threshold=threshold)

    # Save with numbered filename including overlap value
    filename = f"below_threshold_example_{idx + 1:02d}"
    if sample_world_seed is not None:
      filename += f"_world_{sample_world_seed}"
    filename += f"_overlap_{overlap_value:.3f}_reuse_{reuse_value}"

    figure_utils.save_figure(
      fig,
      filename,
      directory=output_dir if not above_ratio else f"{output_dir}_above_ratio",
    )

    if display_image:
      if JUPYTER_AVAILABLE:
        display(fig)
      else:
        plt.show()

    plt.close(fig)  # Close the figure to free memory


if __name__ == "__main__":
  from data_processing import process_model_data
  from data_processing import process_user_data

  # Parse command line arguments
  parser = argparse.ArgumentParser(
    description="Generate Craftax overlap analysis visualizations"
  )
  parser.add_argument(
    "--models",
    nargs="+",
    default=["human", "preplay", "usfa", "dyna", "qlearning"],
    choices=["human", "preplay", "usfa", "dyna", "qlearning"],
    help="Models to analyze (can specify multiple). Use 'human' for human data.",
  )
  parser.add_argument(
    "--thresholds",
    nargs="+",
    type=float,
    default=[0.15, 0.2, 0.25, 0.3],
    help="Overlap thresholds to analyze (default: 0.15 0.2)",
  )
  parser.add_argument(
    "--worlds",
    nargs="+",
    type=int,
    default=[3, 15, 20, 95],
    help="World seeds to analyze (default: 3 15 20 95)",
  )
  parser.add_argument(
    "--n-examples",
    type=int,
    default=10,
    help="Number of examples to generate for each reuse category (default: 10)",
  )

  args = parser.parse_args()

  # Define thresholds to analyze
  thresholds = args.thresholds

  # Define models available for Craftax (excluding human which is handled separately)
  available_models = ["preplay", "usfa", "dyna", "qlearning"]
  requested_models = [m for m in args.models if m != "human"]

  # Craftax world seeds
  world_seeds = args.worlds
  manipulation = "paths"

  # Load and process human data if requested
  if "human" in args.models:
    print("\n=== Processing Human Data ===")
    user_df = process_user_data.get_craftax_human_data(
      load_df_only=False,
    )

    # Parse data once for each world seed
    for world_seed in world_seeds:
      print(f"\nParsing human data for world {world_seed}...")
      samples = parse_and_compute_overlaps(
        df=user_df,
        manipulation=manipulation,
        model="human",
        world_seed=world_seed,
      )

      print(f"Found {len(samples)} samples for world {world_seed}")

      # Apply each threshold to the same parsed data
      for threshold in thresholds:
        print(f"Visualizing with threshold {threshold}...")
        visualize_from_samples(
          samples=samples,
          threshold=threshold,
          model="human",
          manipulation=manipulation,
          world_seed=world_seed,
          n_examples=args.n_examples,
          clear_output_dir=True,
        )

  # Load and process model data if any models were requested
  if requested_models:
    print("\n=== Processing Model Data ===")
    model_df = process_model_data.get_craftax_model_data(
      load_df_only=False,
      models=requested_models,
    )

    # Process each model
    for model in requested_models:
      print(f"\n=== Processing {model} model ===")

      # Parse data once for each world seed
      for world_seed in world_seeds:
        print(f"\nParsing {model} data for world {world_seed}...")
        samples = parse_and_compute_overlaps(
          df=model_df.filter(algo=model),
          manipulation=manipulation,
          model=model,
          world_seed=world_seed,
        )

        print(f"Found {len(samples)} samples for world {world_seed}")

        # Apply each threshold to the same parsed data
        for threshold in thresholds:
          print(f"Visualizing with threshold {threshold}...")
          visualize_from_samples(
            samples=samples,
            threshold=threshold,
            model=model,
            manipulation=manipulation,
            world_seed=world_seed,
            n_examples=args.n_examples,
            clear_output_dir=True,
          )

  print("\nAnalysis complete!")
