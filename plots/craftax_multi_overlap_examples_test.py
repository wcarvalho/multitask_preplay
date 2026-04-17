"""Plot Craftax overlap analysis examples showing train/test path and map comparisons.

Generates comprehensive visualizations with path images, heat maps, and direction
vectors for above/below threshold overlap examples across world seeds.

python plots/craftax_multi_overlap_examples.py --models human
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
  os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulations"
  )
)

import argparse
import pickle
import shutil
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from analysis.vis_utils import (
  get_craftax_env_image,
  craftax_actions_from_path,
  craftax_place_arrows_on_image,
)
from data_processing.utils import compute_average_vector_from_end, compute_overlap
from data_processing.utils_craftax import (
  OPTIMAL_TEST_PATHS,
  OPTIMAL_TRAIN_PATHS,
  create_maps,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
CACHE_DIR = os.path.join(OUTPUT_DIR, "craftax_multi_overlap_examples", "cache")


# ---------------------------------------------------------------------------
# Lightweight stand-ins so cached samples work with visualize_computations
# ---------------------------------------------------------------------------
@dataclass
class CachedEpisode:
  positions: np.ndarray


@dataclass
class CachedDataFrame:
  """Minimal nicewebrl.DataFrame stand-in for visualization access patterns."""

  _episodes: list = field(default_factory=list)
  _columns: dict = field(default_factory=dict)

  @property
  def episodes(self):
    return self._episodes

  def __getitem__(self, key):
    class _Column:
      def __init__(self, values):
        self._values = values

      def to_list(self):
        return self._values

    return _Column(self._columns[key])


def _sample_to_cacheable(sample):
  """Convert a sample dict to a pickle-friendly dict."""
  train = sample["train"]
  test = sample["test"]
  return {
    "key": sample["key"],
    "train_positions": [np.asarray(ep.positions) for ep in train.episodes],
    "train_columns": {"success": train["success"].to_list()},
    "test_positions": [np.asarray(ep.positions) for ep in test.episodes],
    "test_columns": {"success": test["success"].to_list()},
    "train_map": np.asarray(sample["train_map"]),
    "test_map": np.asarray(sample["test_map"]),
    "cosine": float(sample["cosine"]),
    "df_overlap": float(sample["df_overlap"]),
    "best_idx": sample["best_idx"],
    "filters": sample["filters"],
    "world_seed": sample["world_seed"],
    "ratio_optimal_test_path": float(sample["ratio_optimal_test_path"]),
    "ratio_optimal_train_path": float(sample["ratio_optimal_train_path"]),
  }


def _sample_from_cache(cached):
  """Reconstruct a sample dict from cached data."""
  return {
    "key": cached["key"],
    "train": CachedDataFrame(
      [CachedEpisode(p) for p in cached["train_positions"]],
      cached["train_columns"],
    ),
    "test": CachedDataFrame(
      [CachedEpisode(p) for p in cached["test_positions"]],
      cached["test_columns"],
    ),
    "train_map": cached["train_map"],
    "test_map": cached["test_map"],
    "cosine": cached["cosine"],
    "df_overlap": cached["df_overlap"],
    "best_idx": cached["best_idx"],
    "filters": cached["filters"],
    "world_seed": cached["world_seed"],
    "ratio_optimal_test_path": cached["ratio_optimal_test_path"],
    "ratio_optimal_train_path": cached["ratio_optimal_train_path"],
  }


def _cache_path(model, manipulation, world_seed):
  return os.path.join(CACHE_DIR, f"{model}_{manipulation}_world{world_seed}.pkl")


def save_samples_cache(samples, model, manipulation, world_seed):
  os.makedirs(CACHE_DIR, exist_ok=True)
  path = _cache_path(model, manipulation, world_seed)
  cacheable = [_sample_to_cacheable(s) for s in samples]
  with open(path, "wb") as f:
    pickle.dump(cacheable, f)
  print(f"Cached {len(samples)} samples to {path}")


def load_samples_cache(model, manipulation, world_seed):
  path = _cache_path(model, manipulation, world_seed)
  if not os.path.exists(path):
    return None
  with open(path, "rb") as f:
    cacheable = pickle.load(f)
  samples = [_sample_from_cache(c) for c in cacheable]
  print(f"Loaded {len(samples)} cached samples from {path}")
  return samples


def _save_figure(fig, filename, directory=None):
  directory = directory or OUTPUT_DIR
  os.makedirs(directory, exist_ok=True)
  filepath = os.path.join(directory, f"{filename}.pdf")
  plt.savefig(filepath, bbox_inches="tight", dpi=300)
  print(f"Saved figure to {filepath}")
  plt.close(fig)


def make_image_path_panel(episode, ax, title, world_seed):
  image, maze_height, maze_width = get_craftax_env_image(int(world_seed))
  path = episode.positions
  actions = craftax_actions_from_path(path)
  craftax_place_arrows_on_image(
    image=image,
    positions=path,
    actions=actions,
    maze_height=maze_height,
    maze_width=maze_width,
    ax=ax,
    display_image=True,
    arrow_color="red",
    show_path_length=False,
    start_color="red",
  )
  ax.set_title(title)
  ax.axis("off")


def _make_sample_from_episode(df, test_df, test_idx, user_id, world_seed_val):
  """Create a single sample dict from a specific test episode index.

  Uses test_df.episodes[test_idx] and test_df columns at test_idx consistently.
  """
  test_episode = test_df.episodes[test_idx]
  corresponding_train_idx = test_df["corresponding_train_episode_idx"].to_list()[
    test_idx
  ]
  train = df.filter(global_episode_idx=corresponding_train_idx)

  if len(train.episodes) == 0:
    return None
  assert len(train.episodes) == 1

  train_success = train["success"].to_list()[0]
  test_success = test_df["success"].to_list()[test_idx]
  if not train_success or not test_success:
    return None

  train_map = create_maps([train.episodes[0]], add_error=True)
  test_map = create_maps([test_episode], add_error=False)

  train_avg_vector, _ = compute_average_vector_from_end(train.episodes[0].positions)
  test_avg_vector, _ = compute_average_vector_from_end(test_episode.positions)
  cosine = np.dot(train_avg_vector, test_avg_vector) / (
    np.linalg.norm(train_avg_vector) * np.linalg.norm(test_avg_vector) + 1e-5
  )

  optimal_test_path = OPTIMAL_TEST_PATHS[int(world_seed_val)]
  ratio_optimal_test_path = len(test_episode.positions) / len(optimal_test_path)

  train_start = tuple(int(x) for x in train.episodes[0].positions[0])
  train_goal = train["task_object_id"].to_list()[0]
  optimal_train_key = (int(world_seed_val), train_start, train_goal)
  optimal_train_path = OPTIMAL_TRAIN_PATHS.get(optimal_train_key)
  ratio_optimal_train_path = (
    len(train.episodes[0].positions) / len(optimal_train_path)
    if optimal_train_path is not None
    else 0.0
  )

  df_overlap_value = compute_overlap(train_map, test_map).mean()

  # Build single-episode test DataFrame stand-in for caching/visualization
  single_test = test_df.filter(
    global_episode_idx=test_df["global_episode_idx"].to_list()[test_idx],
  )

  filters = dict(
    start_pos=test_df["start_pos"].to_list()[test_idx],
    block_name=test_df["block_name"].to_list()[test_idx],
    user_id=user_id,
    task_set=test_df["task_set"].to_list()[test_idx],
  )

  return {
    "key": user_id,
    "train": train,
    "test": single_test,
    "train_map": train_map,
    "test_map": test_map,
    "cosine": cosine,
    "df_overlap": df_overlap_value,
    "best_idx": 0,
    "filters": filters,
    "world_seed": world_seed_val,
    "ratio_optimal_test_path": ratio_optimal_test_path,
    "ratio_optimal_train_path": ratio_optimal_train_path,
  }


def parse_and_compute_overlaps(
  df,
  manipulation,
  model,
  world_seed=None,
  debug=-1,
  manual_user_id=None,
  max_tries=1000000,
  one_per_seed=False,
  **kwargs,
):
  """Parse dataframe and compute all overlap values once.

  Args:
    one_per_seed: If True, pick 1 random goal-reaching test episode per seed.
      Filters out episodes with ratio_optimal_test_path > 2.0.
  """
  import random

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
      test_initial = df.filter(
        eval=True,
        user_id=user_id,
        manipulation=manipulation,
        block_name=block_name,
      ).sort("overlap", descending=True)

      if len(test_initial.episodes) == 0:
        continue

      world_seed_val = test_initial["world"].to_list()[0]
      if world_seed is not None and str(world_seed_val) != str(world_seed):
        continue

      if one_per_seed:
        # Pick 1 random goal-reaching test episode per seed
        optimal_test_path = OPTIMAL_TEST_PATHS[int(world_seed_val)]
        candidate_indices = []
        for idx in range(len(test_initial.episodes)):
          ratio = len(test_initial.episodes[idx].positions) / len(optimal_test_path)
          if ratio <= 2.0:
            candidate_indices.append(idx)
        if not candidate_indices:
          continue
        chosen_idx = random.choice(candidate_indices)
        sample = _make_sample_from_episode(
          df, test_initial, chosen_idx, user_id, world_seed_val
        )
        if sample is not None:
          samples.append(sample)
      else:
        for test_episode_idx in range(min(len(test_initial.episodes), max_tries)):
          sample = _make_sample_from_episode(
            df, test_initial, test_episode_idx, user_id, world_seed_val
          )
          if sample is not None:
            samples.append(sample)

          if debug != -1 or manual_user_id is not None:
            return samples

  return samples


def visualize_computations(data, model=None, threshold=None, cosine_threshold=0.5):
  train = data["train"]
  test = data["test"]
  train_map = data["train_map"]
  test_map = data["test_map"]
  overlap_value = data["df_overlap"]
  ratio_optimal_test_path = data["ratio_optimal_test_path"]
  if train_map.ndim > 2:
    train_map = train_map[0]
  if test_map.ndim > 2:
    test_map = test_map[0]

  k = data["key"]
  sample_world_seed = data.get("world_seed", None)

  train_success = (
    train["success"].to_list()[0] if len(train["success"].to_list()) > 0 else "N/A"
  )
  test_success = (
    test["success"].to_list()[0] if len(test["success"].to_list()) > 0 else "N/A"
  )

  fig, axes = plt.subplots(3, 2, figsize=(12, 15))

  if model == "human":
    train_title = f"Train (user: {k}, success: {train_success})"
  else:
    model_key = "seed"
    train_title = f"Train ({model_key}: {k}, success: {train_success})"

  test_title = (
    f"Test success: {test_success}, "
    f"path_length: {len(test.episodes[0].positions)}, "
    f"ratio: {ratio_optimal_test_path:.3f}"
  )

  if sample_world_seed is not None:
    test_title = f"World {sample_world_seed} - " + test_title

  # Row 0: Path visualizations
  make_image_path_panel(train.episodes[0], axes[0, 0], train_title, sample_world_seed)
  make_image_path_panel(test.episodes[0], axes[0, 1], test_title, sample_world_seed)

  # Row 1: Heat maps
  im1 = axes[1, 0].imshow(train_map, cmap="viridis")
  axes[1, 0].set_title("Best Train Map")
  axes[1, 0].axis("off")
  plt.colorbar(im1, ax=axes[1, 0], shrink=0.8)

  test_map = test_map + train_map * test_map  # to show overlap
  im2 = axes[1, 1].imshow(test_map, cmap="viridis")
  axes[1, 1].set_title("Test Map")
  axes[1, 1].axis("off")
  plt.colorbar(im2, ax=axes[1, 1], shrink=0.8)

  # Row 2: Average vectors from last half of trajectory
  train_positions = train.episodes[0].positions[:-1]
  test_positions = test.episodes[0].positions[:-1]

  train_avg_vector, _train_vectors = compute_average_vector_from_end(train_positions)
  test_avg_vector, _test_vectors = compute_average_vector_from_end(test_positions)

  cosine = np.dot(train_avg_vector, test_avg_vector) / (
    np.linalg.norm(train_avg_vector) * np.linalg.norm(test_avg_vector)
  )

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

  cond1 = overlap_value > threshold and cosine > cosine_threshold
  cond2 = overlap_value > 2 * threshold
  is_reuse = cond1 or cond2

  title = f"user: {k}"
  title += f"\ncondition 1: overlap ({overlap_value:.3f}) > {threshold} AND cosine ({cosine:.3f}) > {cosine_threshold}: {cond1}"
  title += f"\ncondition 2: overlap ({overlap_value:.3f}) > {2 * threshold}: {cond2}"
  title += f"\nreuse: condition 1 OR condition 2: {is_reuse}"

  fig.suptitle(
    title,
    fontsize=14,
  )

  plt.tight_layout()
  return fig


def _is_reuse(sample, threshold, cosine_threshold):
  """Match the reuse logic from analysis_utils.add_reuse_column (line 115)."""
  cond1 = sample["df_overlap"] > threshold and sample["cosine"] > cosine_threshold
  cond2 = sample["df_overlap"] > 2 * threshold
  return cond1 or cond2


def visualize_from_samples(
  samples,
  threshold,
  model,
  manipulation,
  cosine_threshold=0.5,
  world_seed=None,
  n_examples=10,
  display_image=False,
  clear_output_dir=True,
  **kwargs,
):
  """Visualize examples from pre-computed samples based on threshold."""
  world_dir = f"{model}_world_{world_seed}" if world_seed is not None else model
  output_dir = os.path.join(
    OUTPUT_DIR,
    "craftax_multi_overlap_examples",
    world_dir,
    f"{manipulation}_{threshold}_{cosine_threshold}",
  )

  if os.path.exists(output_dir) and clear_output_dir:
    print(f"Removing old data from {output_dir}...")
    shutil.rmtree(output_dir)

  os.makedirs(output_dir, exist_ok=True)

  reuse_samples = [s for s in samples if _is_reuse(s, threshold, cosine_threshold)]
  no_reuse_samples = [
    s for s in samples if not _is_reuse(s, threshold, cosine_threshold)
  ]

  print(f"{model} - {manipulation} - threshold {threshold}:")
  if world_seed is not None:
    print(f"  World seed: {world_seed}")
  print(f"  Found {len(reuse_samples)} reuse")
  print(f"  Found {len(no_reuse_samples)} no_reuse")

  # Select N closest to boundary on each side
  reuse_samples.sort(key=lambda x: x["df_overlap"])
  reuse_samples = reuse_samples[:n_examples]

  no_reuse_samples.sort(key=lambda x: x["df_overlap"], reverse=True)
  no_reuse_samples = no_reuse_samples[:n_examples]

  # Sort for display: both ascending for continuous overlap order
  no_reuse_samples.sort(key=lambda x: x["df_overlap"])

  # Combine: no_reuse first (ascending), then reuse (ascending)
  all_ordered = [(s, "no_reuse") for s in no_reuse_samples] + [
    (s, "reuse") for s in reuse_samples
  ]

  for idx, (sample, label) in enumerate(all_ordered):
    overlap_value = sample["df_overlap"]
    ratio_optimal_test_path = sample["ratio_optimal_test_path"]
    ratio_optimal_train_path = sample["ratio_optimal_train_path"]
    above_ratio = ratio_optimal_test_path > 2.0 or ratio_optimal_train_path > 2.0

    fig = visualize_computations(
      sample, model=model, threshold=threshold, cosine_threshold=cosine_threshold
    )

    filename = f"{idx:02d}_{label}_{overlap_value:.3f}"

    _save_figure(
      fig,
      filename,
      directory=output_dir if not above_ratio else f"{output_dir}_above_ratio",
    )


def main():
  from data_processing import process_model_data
  from data_processing import process_user_data

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
    default=[0.25],
    help="Overlap thresholds to analyze (default: 0.15 0.2 0.25 0.3)",
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
  parser.add_argument(
    "--no-cache",
    action="store_true",
    help="Force recomputation, ignoring cached samples",
  )

  args = parser.parse_args()

  thresholds = args.thresholds
  requested_models = [m for m in args.models if m != "human"]
  world_seeds = args.worlds
  manipulation = "paths"
  use_cache = not args.no_cache

  if "human" in args.models:
    print("\n=== Processing Human Data ===")
    user_df = None  # lazy-load only if cache misses

    for world_seed in world_seeds:
      samples = None
      if use_cache:
        samples = load_samples_cache("human", manipulation, world_seed)

      if samples is None:
        if user_df is None:
          user_df = process_user_data.get_craftax_human_data(load_df_only=False)
        print(f"\nParsing human data for world {world_seed}...")
        samples = parse_and_compute_overlaps(
          df=user_df,
          manipulation=manipulation,
          model="human",
          world_seed=world_seed,
        )
        save_samples_cache(samples, "human", manipulation, world_seed)

      print(f"Found {len(samples)} samples for world {world_seed}")

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

  if requested_models:
    print("\n=== Processing Model Data ===")
    model_df = None  # lazy-load only if cache misses

    for model in requested_models:
      print(f"\n=== Processing {model} model ===")

      for world_seed in world_seeds:
        samples = None
        if use_cache:
          samples = load_samples_cache(model, manipulation, world_seed)

        if samples is None:
          if model_df is None:
            model_df = process_model_data.get_craftax_model_data(
              load_df_only=False,
              models=requested_models,
            )
          print(f"\nParsing {model} data for world {world_seed}...")
          samples = parse_and_compute_overlaps(
            df=model_df.filter(algo=model),
            manipulation=manipulation,
            model=model,
            world_seed=world_seed,
            one_per_seed=True,
          )
          save_samples_cache(samples, model, manipulation, world_seed)

        print(f"Found {len(samples)} samples for world {world_seed}")

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


if __name__ == "__main__":
  main()
