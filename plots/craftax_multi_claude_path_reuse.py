"""Claude-classified Craftax path reuse for human participants.

Renders train/test path images for every (user, eval episode) in the canonical
Craftax path-reuse cohort, then asks `claude -p` to classify each pair as
path_reuse / no_path_reuse by visual judgment.

The deployed prompt is in `PROMPT_TEMPLATE` below — a 2-question rubric
(same side of water; same route/approach). It is also written to
`<output_dir>/prompt.txt` (paper-ready) on every run.

  python plots/craftax_multi_claude_path_reuse.py            # 20-pair pilot
  python plots/craftax_multi_claude_path_reuse.py --full     # all pairs
"""

import argparse
import atexit
import concurrent.futures
import glob
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import only from analysis.vis_utils — pulling make_image_path_panel from
# craftax_multi_overlap_examples drags in data_processing.utils_craftax, which
# uses a stale nicewebrl symbol (TimeStep vs Timestep) and won't import.
from analysis.vis_utils import (  # noqa: E402
  craftax_actions_from_path,
  craftax_place_arrows_on_image,
  get_craftax_env_image,
)

UPSTREAM_CACHE_DIR = os.path.join(
  os.path.dirname(os.path.abspath(__file__)),
  "output",
  "craftax_multi_overlap_examples",
  "cache",
)
ARROW_THICKNESS = 3.0


@dataclass
class _PositionsEpisode:
  positions: np.ndarray


def _draw_path_panel(positions, ax, world_seed):
  image, maze_height, maze_width = get_craftax_env_image(int(world_seed))
  actions = craftax_actions_from_path(positions)
  craftax_place_arrows_on_image(
    image=image,
    positions=positions,
    actions=actions,
    maze_height=maze_height,
    maze_width=maze_width,
    ax=ax,
    display_image=True,
    arrow_color="red",
    show_path_length=False,
    start_color="red",
    line_thickness=ARROW_THICKNESS,
  )
  if len(positions) > 0:
    image_height, image_width, _ = image.shape
    scale_y = image_height / maze_height
    scale_x = image_width / maze_width
    sy = (positions[0][0] + 0.5) * scale_y
    sx = (positions[0][1] + 0.5) * scale_x
    ax.plot(
      sx,
      sy,
      "*",
      color="yellow",
      markersize=22,
      markeredgecolor="black",
      markeredgewidth=1.2,
    )
    ey = (positions[-1][0] + 0.5) * scale_y
    ex = (positions[-1][1] + 0.5) * scale_x
    ax.plot(
      ex,
      ey,
      "o",
      color="yellow",
      markersize=14,
      markeredgecolor="black",
      markeredgewidth=1.2,
    )
  ax.set_xticks([])
  ax.set_yticks([])


_BASE_OUTPUT_DIR = os.path.join(
  os.path.dirname(os.path.abspath(__file__)),
  "output",
)
# These are populated in main() once we know the optional --output-suffix.
OUTPUT_DIR = os.path.join(_BASE_OUTPUT_DIR, "craftax_multi_claude_path_reuse")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
CLASSIFICATIONS_CACHE = os.path.join(CACHE_DIR, "classifications.json")
PILOT_KEYS_CACHE = os.path.join(CACHE_DIR, "pilot_keys.json")
PROMPT_AUDIT_PATH = os.path.join(CACHE_DIR, "prompt_template.txt")


def _set_output_dir(suffix=""):
  global OUTPUT_DIR, IMAGES_DIR, CACHE_DIR
  global CLASSIFICATIONS_CACHE, PILOT_KEYS_CACHE, PROMPT_AUDIT_PATH
  name = "craftax_multi_claude_path_reuse"
  if suffix:
    name = f"{name}_{suffix}"
  OUTPUT_DIR = os.path.join(_BASE_OUTPUT_DIR, name)
  IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
  CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
  CLASSIFICATIONS_CACHE = os.path.join(CACHE_DIR, "classifications.json")
  PILOT_KEYS_CACHE = os.path.join(CACHE_DIR, "pilot_keys.json")
  PROMPT_AUDIT_PATH = os.path.join(CACHE_DIR, "prompt_template.txt")


LABEL_FOLDERS = ("path_reuse", "no_path_reuse", "ambiguous")
VALID_LABELS = set(LABEL_FOLDERS)

OVERLAP_THRESHOLD = 0.25
COSINE_THRESHOLD = 0.5

# The prompt sent to `claude -p` for every (train, test) image pair.
# This is also written to `<output_dir>/prompt.txt` (paper-ready, placeholders
# expanded) and `<output_dir>/cache/prompt_template.txt` (raw) on every run.
PROMPT_TEMPLATE = """\
You will see two images of the same Craftax world. In each, a red line shows
the path a participant took from the yellow star (start) to the yellow circle
(goal). Both paths are from the same person on different trials — first
reaching a training goal, then reaching a test goal placed nearby. They
started in the same position both times.

Read both images first:
1. Read {train_path}
2. Read {test_path}

Did the participant reuse the training path on the test trial? Answer two
questions:

1. SAME SIDE OF WATER: For each meaningful body of water (blue), did both
   paths go on the same side of it?

2. SAME ROUTE: Do both paths follow roughly the same route to the goal and
   approach from roughly the same direction? (A brief detour where the test
   path wanders off and then rejoins the training route is fine.)

Output exactly ONE JSON object as the LAST line of your response, no code
fences. Label "path_reuse" only if both answers are yes; otherwise
"no_path_reuse". Include a brief reasoning (1-2 sentences) describing what
you saw on each side of each path:
{{"same_side": "yes|no", "same_approach": "yes|no", "reasoning": "<1-2 sentences>", "label": "path_reuse|no_path_reuse"}}
"""


def _ensure_dirs():
  for d in (OUTPUT_DIR, IMAGES_DIR, CACHE_DIR):
    os.makedirs(d, exist_ok=True)


def _load_json(path, default):
  if not os.path.exists(path):
    return default
  with open(path) as f:
    return json.load(f)


def _atomic_write_json(path, obj):
  tmp = path + ".tmp"
  with open(tmp, "w") as f:
    json.dump(obj, f, indent=2)
  os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Build the row table from the upstream overlap-examples cache
# ---------------------------------------------------------------------------
# We read pickles produced by `craftax_multi_overlap_examples.py`. They contain
# per-(user, world) train/test position arrays + df_overlap + cosine, which is
# everything we need to render images and stratify. The full canonical
# first-100-per-tell_reuse cohort lives behind data we can't reach right now
# (the env-construction path in process_user_data fails on the current nicewebrl
# version, and the parquet on the Crucial X9 SSD isn't mounted), so the cache
# is the practical source for the pilot.
def build_pairs(cache_glob=None):
  cache_glob = cache_glob or os.path.join(UPSTREAM_CACHE_DIR, "human_paths_world*.pkl")
  cache_files = sorted(glob.glob(cache_glob))
  if not cache_files:
    raise FileNotFoundError(
      f"No cached samples at {cache_glob}. Run craftax_multi_overlap_examples.py "
      "first, or wire up a real data loader."
    )

  print(f"Loading {len(cache_files)} cache file(s)...")
  pairs = []
  for path in cache_files:
    with open(path, "rb") as f:
      cached = pickle.load(f)
    print(f"  {os.path.basename(path)}: {len(cached)} samples")

    for sample in cached:
      train_positions = sample.get("train_positions") or []
      test_positions = sample.get("test_positions") or []
      if not train_positions or not test_positions:
        continue

      train_success = sample.get("train_columns", {}).get("success", [True])[0]
      test_success = sample.get("test_columns", {}).get("success", [True])[0]
      if not train_success or not test_success:
        continue

      df_overlap = float(sample["df_overlap"])
      cosine = float(sample["cosine"])
      cond1 = df_overlap > OVERLAP_THRESHOLD and cosine > COSINE_THRESHOLD
      cond2 = df_overlap > 2 * OVERLAP_THRESHOLD
      rule_reuse = bool(cond1 or cond2)

      user_id = sample["key"]
      world = str(sample["world_seed"])
      filters = sample.get("filters", {})
      best_idx = sample.get("best_idx", 0)
      key = f"{user_id}_world{world}_idx{best_idx}"

      pairs.append(
        {
          "key": key,
          "user_id": user_id,
          "world": world,
          "block_name": str(filters.get("block_name", world)),
          "task_set": filters.get("task_set"),
          "start_pos": str(filters.get("start_pos", "")),
          "best_idx": int(best_idx),
          "manipulation": "paths",
          "df_overlap": df_overlap,
          "cosine": cosine,
          "rule_reuse": rule_reuse,
          "ratio_optimal_train_path": float(
            sample.get("ratio_optimal_train_path", 0.0)
          ),
          "ratio_optimal_test_path": float(sample.get("ratio_optimal_test_path", 0.0)),
          "train_positions": np.asarray(train_positions[0]),
          "test_positions": np.asarray(test_positions[0]),
          "train_image": os.path.join(IMAGES_DIR, f"train_{key}.png"),
          "test_image": os.path.join(IMAGES_DIR, f"test_{key}.png"),
          "pair_image": os.path.join(IMAGES_DIR, f"pair_{key}.png"),
        }
      )

  pairs = _annotate_with_tell_reuse_and_cohort(pairs)

  print(f"Built {len(pairs)} (train, test) pairs.")
  return pairs


HUMAN_DF_PATH = os.path.join(
  os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
  "dataframes",
  "craftax_human.parquet",
)
COHORT_PICKLE_PATH = os.path.join(
  os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
  "analysis",
  "paper_results_cache",
  "path_reuse_manipulation_analysis_tell_reuse_user_ids.pkl",
)


def _annotate_with_tell_reuse_and_cohort(pairs):
  """Join `tell_reuse`, `global_episode_idx`, `corresponding_train_episode_idx`,
  and `in_canonical_cohort` from the public human parquet + cached cohort pickle.

  Returns a new pair list with annotations; drops pairs that don't match any
  parquet row (shouldn't happen given the cached pickles were generated from
  the same upstream data).
  """
  if not os.path.exists(HUMAN_DF_PATH):
    print(f"WARNING: {HUMAN_DF_PATH} missing — skipping tell_reuse annotation.")
    for p in pairs:
      p["tell_reuse"] = None
      p["global_episode_idx"] = None
      p["corresponding_train_episode_idx"] = None
      p["in_canonical_cohort"] = False
    return pairs

  print(f"Joining tell_reuse / cohort metadata from {HUMAN_DF_PATH}...")
  human = pl.read_parquet(HUMAN_DF_PATH).filter(
    eval=True,
    manipulation="paths",
    success=True,
    eval_shares_start_pos=True,
  )

  # Build lookup keyed by (user_id, world) → first matching row's metadata.
  # The pickled samples have one entry per (user, world); the parquet should
  # match one-to-one for those filters.
  lookup = {}
  for r in human.iter_rows(named=True):
    key = (int(r["user_id"]), str(r["world"]))
    if key in lookup:
      continue  # take the first matching row
    lookup[key] = {
      "tell_reuse": int(r["tell_reuse"]) if r["tell_reuse"] is not None else None,
      "global_episode_idx": int(r["global_episode_idx"]),
      "corresponding_train_episode_idx": (
        int(r["corresponding_train_episode_idx"])
        if r["corresponding_train_episode_idx"] is not None
        else None
      ),
    }

  cohort = set()
  if os.path.exists(COHORT_PICKLE_PATH):
    with open(COHORT_PICKLE_PATH, "rb") as f:
      cohort = {int(u) for u in pickle.load(f)}
    print(f"  Loaded canonical cohort: {len(cohort)} users.")
  else:
    print(f"WARNING: {COHORT_PICKLE_PATH} missing — in_canonical_cohort will be False.")

  annotated = []
  missing = 0
  for p in pairs:
    k = (int(p["user_id"]), str(p["world"]))
    meta = lookup.get(k)
    if meta is None:
      missing += 1
      continue
    p = dict(p)
    p["tell_reuse"] = meta["tell_reuse"]
    p["global_episode_idx"] = meta["global_episode_idx"]
    p["corresponding_train_episode_idx"] = meta["corresponding_train_episode_idx"]
    p["in_canonical_cohort"] = int(p["user_id"]) in cohort
    annotated.append(p)

  if missing:
    print(f"  Dropped {missing} pair(s) with no matching parquet row.")
  in_cohort = sum(1 for p in annotated if p["in_canonical_cohort"])
  print(
    f"  Annotated {len(annotated)} pairs; "
    f"{in_cohort} in canonical cohort, {len(annotated) - in_cohort} outside."
  )
  return annotated


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------
def _render_single_panel(positions, world_seed, title, out_path):
  fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
  _draw_path_panel(positions, ax, world_seed)
  ax.set_title(title, fontsize=20)
  fig.savefig(out_path, bbox_inches="tight", dpi=150)
  plt.close(fig)


def _render_pair_panel(train_positions, test_positions, world_seed, out_path):
  fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(14, 7), dpi=150)
  _draw_path_panel(train_positions, ax_train, world_seed)
  _draw_path_panel(test_positions, ax_test, world_seed)
  ax_train.set_title("TRAIN", fontsize=18)
  ax_test.set_title("TEST", fontsize=18)
  fig.savefig(out_path, bbox_inches="tight", dpi=150)
  plt.close(fig)


def render_images(pairs, regenerate=False):
  todo = []
  for p in pairs:
    if regenerate or not all(
      os.path.exists(p[k]) for k in ("train_image", "test_image", "pair_image")
    ):
      todo.append(p)

  if not todo:
    print(f"All {len(pairs)} pair images already exist; skipping render.")
    return

  print(f"Rendering {len(todo)} pair(s) (of {len(pairs)} total)...")
  for p in tqdm(todo, desc="Rendering"):
    train_pos = p["train_positions"]
    test_pos = p["test_positions"]
    world_seed = p["world"]

    if regenerate or not os.path.exists(p["train_image"]):
      _render_single_panel(train_pos, world_seed, "TRAIN", p["train_image"])
    if regenerate or not os.path.exists(p["test_image"]):
      _render_single_panel(test_pos, world_seed, "TEST", p["test_image"])
    if regenerate or not os.path.exists(p["pair_image"]):
      _render_pair_panel(train_pos, test_pos, world_seed, p["pair_image"])


# ---------------------------------------------------------------------------
# Pilot selection
# ---------------------------------------------------------------------------
def select_pilot(pairs, n=20):
  """Stratify the pair list by df_overlap into 4 buckets and take ~n/4 from each."""
  buckets = {
    "clear_below": [p for p in pairs if p["df_overlap"] < 0.20],
    "borderline_below": [
      p for p in pairs if 0.20 <= p["df_overlap"] < OVERLAP_THRESHOLD
    ],
    "borderline_above": [
      p for p in pairs if OVERLAP_THRESHOLD <= p["df_overlap"] < 2 * OVERLAP_THRESHOLD
    ],
    "clear_above": [p for p in pairs if p["df_overlap"] >= 2 * OVERLAP_THRESHOLD],
  }

  per_bucket = max(1, n // len(buckets))

  picked = []
  picked_keys = set()
  for name, group in buckets.items():
    group.sort(key=lambda p: p["df_overlap"])
    if len(group) <= per_bucket:
      sample = group
    else:
      idxs = np.linspace(0, len(group) - 1, per_bucket).round().astype(int)
      sample = [group[i] for i in idxs]
    for s in sample:
      if s["key"] not in picked_keys:
        picked.append(s)
        picked_keys.add(s["key"])
    print(f"  [{name}] pool={len(group):4d} picked={len(sample)}")

  if len(picked) < n:
    extras = sorted(
      (p for p in pairs if p["key"] not in picked_keys),
      key=lambda p: abs(p["df_overlap"] - OVERLAP_THRESHOLD),
    )
    for s in extras:
      if len(picked) >= n:
        break
      picked.append(s)
      picked_keys.add(s["key"])

  picked = picked[:n]
  return picked


def load_or_choose_pilot(pairs, n):
  if os.path.exists(PILOT_KEYS_CACHE):
    keys = _load_json(PILOT_KEYS_CACHE, [])
    by_key = {p["key"]: p for p in pairs}
    pilot = [by_key[k] for k in keys if k in by_key]
    if len(pilot) == len(keys) and len(pilot) > 0:
      print(f"Loaded {len(pilot)} cached pilot keys from {PILOT_KEYS_CACHE}.")
      return pilot
    print("Cached pilot keys stale (some keys missing); reselecting.")

  print("Selecting stratified pilot...")
  pilot = select_pilot(pairs, n=n)
  _atomic_write_json(PILOT_KEYS_CACHE, [p["key"] for p in pilot])
  print(f"Saved pilot keys ({len(pilot)}) to {PILOT_KEYS_CACHE}.")
  return pilot


# ---------------------------------------------------------------------------
# claude -p classifier
# ---------------------------------------------------------------------------
def build_prompt(train_path, test_path):
  return PROMPT_TEMPLATE.format(train_path=train_path, test_path=test_path)


def _decision_table_label(parsed):
  """Derive the label from the structured sub-fields ({same_side, same_approach}).

  path_reuse iff BOTH "yes". Returns None if the fields are missing or invalid;
  callers fall back to Claude's own self-reported `label`.
  """
  if parsed is None:
    return None
  ss = (parsed.get("same_side") or "").strip().lower()
  sa = (parsed.get("same_approach") or "").strip().lower()
  if ss not in ("yes", "no") or sa not in ("yes", "no"):
    return None
  return "path_reuse" if (ss == "yes" and sa == "yes") else "no_path_reuse"


def _parse_label(stdout):
  """Find the last JSON line. Return (label, parsed_obj, raw_line)."""
  for line in reversed(stdout.splitlines()):
    line = line.strip()
    if not line or not line.startswith("{"):
      continue
    try:
      obj = json.loads(line)
    except json.JSONDecodeError:
      continue
    label = str(obj.get("label", "")).strip().lower()
    if label in VALID_LABELS:
      return label, obj, line
  # fallback: scan for bare token
  text = stdout.lower()
  for tok in VALID_LABELS:
    if tok in text:
      return tok, {"label": tok, "_fallback": True}, stdout.strip()
  return None, None, stdout.strip()


def classify_one(pair, model, timeout=600):
  """Run claude -p on a single pair. Return dict with label/raw/error."""
  prompt = build_prompt(pair["train_image"], pair["test_image"])

  env = {
    k: v
    for k, v in os.environ.items()
    if k not in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")
  }

  with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    f.write(prompt)
    prompt_path = f.name

  cmd = (
    f'cat "{prompt_path}" | claude -p --no-session-persistence '
    f"--allowedTools Read --output-format text --model {model}"
  )

  result = {
    "label": None,
    "reasoning": None,
    "raw": None,
    "error": None,
    "model": model,
    "ts": time.time(),
  }
  try:
    proc = subprocess.run(
      cmd,
      shell=True,
      capture_output=True,
      text=True,
      timeout=timeout,
      env=env,
    )
    if proc.returncode != 0:
      result["error"] = f"exit={proc.returncode}: {proc.stderr[:500]}"
      result["raw"] = proc.stdout
      return result

    label, parsed, raw = _parse_label(proc.stdout)
    result["raw"] = raw
    result["label"] = label or "parse_error"
    if parsed:
      # Both schemas: store every sub-field that's present.
      for k in ("same_side", "same_approach", "reasoning"):
        if k in parsed:
          result[k] = parsed.get(k)
      derived = _decision_table_label(parsed)
      result["derived_label"] = derived
      if derived and label and label != "ambiguous" and derived != label:
        result["derivation_mismatch"] = True
    if label is None:
      result["error"] = "no valid label parsed"
    return result
  except subprocess.TimeoutExpired:
    result["error"] = f"timeout after {timeout}s"
    return result
  except Exception as exc:  # pragma: no cover
    result["error"] = f"{type(exc).__name__}: {exc}"
    return result
  finally:
    Path(prompt_path).unlink(missing_ok=True)


def classify_batch(targets, cache, model, workers, save_every=5):
  """Classify all targets that aren't already cached. Mutates `cache` in place."""
  todo = [p for p in targets if p["key"] not in cache]
  if not todo:
    print(f"All {len(targets)} target(s) already classified.")
    return

  print(
    f"Classifying {len(todo)} pair(s) with claude -p (model={model}, workers={workers})..."
  )
  done_since_flush = 0

  def _flush():
    _atomic_write_json(CLASSIFICATIONS_CACHE, cache)

  atexit.register(_flush)

  try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
      futures = {pool.submit(classify_one, p, model): p for p in todo}
      for fut in tqdm(
        concurrent.futures.as_completed(futures),
        total=len(futures),
        desc="claude -p",
      ):
        pair = futures[fut]
        try:
          result = fut.result()
        except Exception as exc:  # pragma: no cover
          result = {"label": "parse_error", "error": str(exc), "raw": ""}
        cache[pair["key"]] = result
        done_since_flush += 1
        if done_since_flush >= save_every:
          _flush()
          done_since_flush = 0
  finally:
    _flush()


# ---------------------------------------------------------------------------
# Output: dataframe + folder mirroring + summary
# ---------------------------------------------------------------------------
_DROP_COLS = ("train_positions", "test_positions")


def write_classifications_df(pairs, cache):
  rows = []
  for p in pairs:
    c = cache.get(p["key"], {})
    row = {k: v for k, v in p.items() if k not in _DROP_COLS}
    row["claude_label"] = c.get("label")
    row["claude_reasoning"] = c.get("reasoning")
    row["claude_same_side"] = c.get("same_side")
    row["claude_same_approach"] = c.get("same_approach")
    row["claude_derived_label"] = c.get("derived_label")
    row["claude_derivation_mismatch"] = bool(c.get("derivation_mismatch", False))
    row["claude_raw"] = c.get("raw")
    row["claude_error"] = c.get("error")
    row["claude_model"] = c.get("model")
    rows.append(row)

  # `infer_schema_length=None` scans all rows, avoiding the "first row None →
  # later rows have str → ComputeError" failure when some Claude calls errored
  # and others returned valid output.
  df = pl.DataFrame(rows, infer_schema_length=None)
  parquet_path = os.path.join(OUTPUT_DIR, "classifications.parquet")
  csv_path = os.path.join(OUTPUT_DIR, "classifications.csv")
  df.write_parquet(parquet_path)
  df.write_csv(csv_path)
  print(f"Wrote {parquet_path}")
  print(f"Wrote {csv_path}")
  return df


def mirror_label_folders(pairs, cache):
  for label in LABEL_FOLDERS:
    folder = os.path.join(OUTPUT_DIR, label)
    if os.path.exists(folder):
      shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

  count = 0
  for p in pairs:
    c = cache.get(p["key"], {})
    label = c.get("label")
    if label not in VALID_LABELS:
      continue
    src = os.path.abspath(p["pair_image"])
    if not os.path.exists(src):
      continue
    dst = os.path.join(OUTPUT_DIR, label, f"{p['key']}.png")
    try:
      os.symlink(src, dst)
      count += 1
    except FileExistsError:
      pass
  print(f"Mirrored {count} symlinks across {LABEL_FOLDERS}.")


def _html_escape(s):
  if s is None:
    return ""
  return (
    str(s)
    .replace("&", "&amp;")
    .replace("<", "&lt;")
    .replace(">", "&gt;")
    .replace('"', "&quot;")
  )


def generate_html_review(pairs, cache):
  """Write a self-contained review.html.

  Sections are ordered by:
    1. agree/disagree with the rule-based label (agree first, disagree last)
    2. Claude's label (path_reuse first, no_path_reuse second)
  Within each section, rows are sorted by df_overlap.
  """
  buckets = {
    ("agree", "path_reuse"): [],
    ("agree", "no_path_reuse"): [],
    ("disagree", "path_reuse"): [],
    ("disagree", "no_path_reuse"): [],
  }
  for p in pairs:
    c = cache.get(p["key"], {})
    label = c.get("label")
    if label not in ("path_reuse", "no_path_reuse"):
      continue
    rule_label = "path_reuse" if p["rule_reuse"] else "no_path_reuse"
    status = "agree" if label == rule_label else "disagree"
    buckets[(status, label)].append((p, c, rule_label))

  for k in buckets:
    buckets[k].sort(key=lambda triple: triple[0]["df_overlap"])

  html_path = os.path.join(OUTPUT_DIR, "review.html")
  parts = [
    "<!doctype html><html><head><meta charset='utf-8'>",
    f"<title>Path-reuse review — {os.path.basename(OUTPUT_DIR)}</title>",
    "<style>",
    "body{font-family:-apple-system,system-ui,sans-serif;margin:2rem;color:#222;}",
    "h1{margin:0 0 .25rem 0;}",
    "h2{margin-top:2.5rem;padding:.5rem .75rem;border-radius:6px;}",
    "h2.agree{background:#e8f4ea;color:#2c5a32;}",
    "h2.disagree{background:#fdecec;color:#7a2222;}",
    "h3{margin:1.5rem 0 .5rem 0;color:#333;}",
    "table{border-collapse:collapse;width:100%;}",
    "td{vertical-align:top;padding:.75rem .5rem;border-bottom:1px solid #eee;}",
    ".img{width:520px;}",
    ".img img{width:520px;display:block;border:1px solid #ccc;}",
    ".meta{font-family:ui-monospace,Menlo,monospace;font-size:.85rem;color:#444;}",
    ".meta b{color:#000;}",
    ".reason{font-size:.95rem;line-height:1.4;max-width:48ch;}",
    ".key{font-family:ui-monospace,Menlo,monospace;font-size:.8rem;color:#666;}",
    ".tag{display:inline-block;padding:.1rem .4rem;border-radius:3px;font-weight:600;}",
    ".tag-reuse{background:#e8f4ea;color:#2c5a32;}",
    ".tag-no{background:#fdecec;color:#7a2222;}",
    ".tag-agree{color:#2c7a2c;font-weight:600;}",
    ".tag-disagree{color:#a92020;font-weight:600;}",
    "</style></head><body>",
    f"<h1>Path-reuse review — {_html_escape(os.path.basename(OUTPUT_DIR))}</h1>",
    "<p class='meta'>Each row: train+test paths, Claude's verdict + reasoning, "
    "the rule-based verdict (overlap > 0.25 ∧ cosine > 0.5) ∨ (overlap > 0.5). "
    "Sorted by agree/disagree (top), then by Claude's label.</p>",
  ]

  # Per-tell_reuse summary box (cohort + all-pairs)
  cohort_props = _compute_tell_reuse_proportions(pairs, cache, only_cohort=True)
  all_props = _compute_tell_reuse_proportions(pairs, cache, only_cohort=False)

  def _props_table_html(props, title):
    rows_html = []
    for tr in (1, 0):
      s = props[tr]
      m = f"{s['mean'] * 100:.1f}%" if s["mean"] is not None else "—"
      md = f"{s['median'] * 100:.1f}%" if s["median"] is not None else "—"
      rows_html.append(
        f"<tr>"
        f"<td>{_html_escape(_TELL_REUSE_LABELS[tr])}</td>"
        f"<td style='text-align:right;'>{s['n_classified']}</td>"
        f"<td style='text-align:right;'>{s['n_users']}</td>"
        f"<td style='text-align:right;'>{s['n_path_reuse']}</td>"
        f"<td style='text-align:right;font-weight:700;'>{m}</td>"
        f"<td style='text-align:right;color:#555;'>{md}</td>"
        f"</tr>"
      )
    return (
      f"<h3 style='margin:.5rem 0;'>{_html_escape(title)}</h3>"
      "<table style='width:auto;border:1px solid #ddd;'>"
      "<tr style='background:#f0f0f0;'>"
      "<th style='text-align:left;padding:.4rem .75rem;'>Condition</th>"
      "<th style='padding:.4rem .75rem;'>n pairs</th>"
      "<th style='padding:.4rem .75rem;'>n users</th>"
      "<th style='padding:.4rem .75rem;'>n path_reuse</th>"
      "<th style='padding:.4rem .75rem;'>mean %</th>"
      "<th style='padding:.4rem .75rem;'>median %</th>"
      "</tr>" + "".join(rows_html) + "</table>"
      "<p class='meta' style='margin-top:.4rem;'>Mean and median computed "
      "the same way as <code>analysis_utils.compute_binary_measure_statistics</code> "
      "(group by user_id → per-user rate; mean weighted by n_trials, median over users). "
      "<b>Mean is the headline statistic</b> reported in the paper revision.</p>"
    )

  parts.append(
    "<div style='margin:1.5rem 0;padding:1rem;background:#fafaf2;"
    "border:1px solid #d6d6b8;border-radius:6px;'>"
    "<h2 style='margin:0 0 .5rem 0;background:none;padding:0;color:#444;'>"
    "Path-reuse proportions by tell_reuse</h2>"
    + _props_table_html(cohort_props, "Canonical cohort (matches paper)")
    + _props_table_html(all_props, "All available pairs")
    + "</div>"
  )

  parts += [
    "<details open style='margin:1.5rem 0;'>",
    "<summary style='font-weight:600;cursor:pointer;'>"
    "Prompt sent to claude -p (click to collapse)</summary>",
    "<pre style='background:#f6f6f6;border:1px solid #ddd;padding:1rem;"
    "white-space:pre-wrap;font-family:ui-monospace,Menlo,monospace;"
    "font-size:.85rem;line-height:1.45;border-radius:4px;'>",
    _html_escape(PROMPT_TEMPLATE),
    "</pre></details>",
  ]

  total_classified = sum(len(b) for b in buckets.values())
  total_agree = len(buckets[("agree", "path_reuse")]) + len(
    buckets[("agree", "no_path_reuse")]
  )
  total_disagree = len(buckets[("disagree", "path_reuse")]) + len(
    buckets[("disagree", "no_path_reuse")]
  )
  parts.append(
    f"<p class='meta'>Total classified: <b>{total_classified}</b> "
    f"&nbsp; agree with rule: <b>{total_agree}</b> "
    f"&nbsp; disagree with rule: <b>{total_disagree}</b></p>"
  )

  for status in ("agree", "disagree"):
    section_count = len(buckets[(status, "path_reuse")]) + len(
      buckets[(status, "no_path_reuse")]
    )
    if section_count == 0:
      continue
    parts.append(
      f"<h2 class='{status}'>{status.upper()} with rule ({section_count})</h2>"
    )
    for lbl in ("path_reuse", "no_path_reuse"):
      rows = buckets[(status, lbl)]
      if not rows:
        continue
      parts.append(
        f"<h3>Claude says <span class='tag "
        f"{'tag-reuse' if lbl == 'path_reuse' else 'tag-no'}'>"
        f"{_html_escape(lbl)}</span> &nbsp; ({len(rows)})</h3>"
      )
      parts.append("<table>")
      for p, c, rule_label in rows:
        img_rel = os.path.relpath(
          os.path.abspath(p["pair_image"]), os.path.dirname(html_path)
        )

        reasoning = c.get("reasoning") or ""
        same_side = c.get("same_side")
        same_approach = c.get("same_approach")

        meta_lines = []
        if same_side is not None or same_approach is not None:
          meta_lines.append(
            f"<b>same_side:</b> {_html_escape(same_side)} &nbsp; "
            f"<b>same_approach:</b> {_html_escape(same_approach)}"
          )
        meta_html = "<br>".join(meta_lines)

        parts.append("<tr>")
        parts.append(
          f"<td class='img'>"
          f"<a href='{_html_escape(img_rel)}'>"
          f"<img src='{_html_escape(img_rel)}' loading='lazy'>"
          f"</a>"
          f"<div class='key'>{_html_escape(p['key'])}</div>"
          f"</td>"
        )
        parts.append(
          f"<td class='reason'>"
          f"<p>{_html_escape(reasoning) or '<i>(no reasoning)</i>'}</p>"
          f"<div class='meta'>{meta_html}</div>"
          f"<div class='meta'>"
          f"df_overlap = {p['df_overlap']:.3f} &nbsp; "
          f"cosine = {p['cosine']:.3f}"
          f"</div>"
          f"<div class='meta'>"
          f"rule says: <b>{rule_label}</b> &nbsp; "
          f"<span class='tag-{status}'>"
          f"({'agree' if status == 'agree' else 'DISAGREE'})</span>"
          f"</div>"
          f"</td>"
        )
        parts.append("</tr>")
      parts.append("</table>")

  parts.append("</body></html>")
  with open(html_path, "w") as f:
    f.write("\n".join(parts))
  print(f"Wrote review page → {html_path}")


def _compute_tell_reuse_proportions(pairs, cache, only_cohort):
  """Return a dict keyed by tell_reuse value with classification stats.

  Aggregation matches `analysis_utils.compute_binary_measure_statistics` so the
  numbers are directly comparable to the paper's mean/median proportions:
    - Group by user_id; compute per-user reuse rate (n_path_reuse / n_trials).
    - **mean** = sum(per_user_rate * n_trials) / total_trials.
        (This collapses to total_path_reuse / total_pairs — the per-pair %.)
    - **median** = median over users of per-user rates.
  """
  result = {}
  for tr in (1, 0):
    rows = []
    for p in pairs:
      if only_cohort and not p.get("in_canonical_cohort"):
        continue
      if p.get("tell_reuse") != tr:
        continue
      c = cache.get(p["key"], {})
      label = c.get("label")
      if label not in ("path_reuse", "no_path_reuse"):
        continue
      rows.append((p, label))

    n_classified = len(rows)
    n_path_reuse = sum(1 for _, lbl in rows if lbl == "path_reuse")
    n_no_path_reuse = n_classified - n_path_reuse

    by_user = {}
    for p, lbl in rows:
      uid = int(p["user_id"])
      by_user.setdefault(uid, []).append(1 if lbl == "path_reuse" else 0)

    if by_user:
      per_user_rates = np.array([sum(vs) / len(vs) for vs in by_user.values()])
      per_user_n = np.array([len(vs) for vs in by_user.values()])
      total_trials = per_user_n.sum()
      mean_proportion = float(
        (per_user_rates * per_user_n).sum() / total_trials
      )  # paper's mean (weighted by n_trials, == n_path_reuse / total_trials)
      median_proportion = float(np.median(per_user_rates))  # paper's median
    else:
      per_user_rates = np.array([])
      mean_proportion = None
      median_proportion = None

    result[tr] = {
      "n_classified": n_classified,
      "n_path_reuse": n_path_reuse,
      "n_no_path_reuse": n_no_path_reuse,
      "n_users": len(by_user),
      "mean": mean_proportion,
      "median": median_proportion,
      "per_user_rates": per_user_rates.tolist(),
    }
  return result


_TELL_REUSE_LABELS = {1: "tell_reuse=1 (known eval goal)", 0: "tell_reuse=0 (unknown)"}


def print_tell_reuse_summary(pairs, cache):
  """Compute and persist per-tell_reuse proportions in two views.

  - canonical_cohort: matches the paper's first-100-per-tell_reuse filter.
  - all_pairs: every classified pair regardless of cohort membership.
  """
  cohort = _compute_tell_reuse_proportions(pairs, cache, only_cohort=True)
  allpairs = _compute_tell_reuse_proportions(pairs, cache, only_cohort=False)

  out_json = {"canonical_cohort": cohort, "all_pairs": allpairs}
  json_path = os.path.join(OUTPUT_DIR, "tell_reuse_summary.json")
  with open(json_path, "w") as f:
    json.dump(out_json, f, indent=2)

  lines = []

  def _row(stats):
    m = stats["mean"]
    md = stats["median"]
    m_s = f"{m * 100:.1f}%" if m is not None else "—"
    md_s = f"{md * 100:.1f}%" if md is not None else "—"
    return (
      f"  n={stats['n_classified']:>4d}  users={stats['n_users']:>3d}  "
      f"path_reuse={stats['n_path_reuse']:>4d}  "
      f"mean={m_s:>6s}  median={md_s:>6s}"
    )

  for view_name, view_data in (
    ("CANONICAL COHORT (first-100 per tell_reuse)", cohort),
    ("ALL PAIRS (sensitivity check)", allpairs),
  ):
    lines.append("")
    lines.append(f"=== {view_name} ===")
    for tr in (1, 0):
      lines.append(f"{_TELL_REUSE_LABELS[tr]}")
      lines.append(_row(view_data[tr]))

  text = "\n".join(lines)
  txt_path = os.path.join(OUTPUT_DIR, "tell_reuse_summary.txt")
  with open(txt_path, "w") as f:
    f.write(text + "\n")

  print(text)
  print(f"\nSaved tell_reuse summary → {txt_path}")
  print(f"Saved tell_reuse summary → {json_path}")


def plot_method_comparison(pairs, cache):
  """Bar plot: (rule-based, Claude) × (known goal, unknown goal).

  Mean + 95% CI computed by `analysis_utils.compute_binary_measure_statistics`
  (the same function the paper plots use): per-user reuse rate, mean weighted
  by n_trials, parametric or bootstrap CI depending on Shapiro-Wilk normality.
  Restricted to the canonical 200-user cohort.
  """
  import plot_configs
  from analysis import analysis_utils

  def _stats_for(measure_name, df, tr):
    sub = df.filter(
      (pl.col("in_canonical_cohort"))
      & (pl.col("tell_reuse") == tr)
      & (pl.col(measure_name).is_not_null())
    )
    if sub.height == 0:
      return None
    return analysis_utils.compute_binary_measure_statistics(
      sub, group_col="user_id", measure=measure_name, center="mean"
    )

  # Build long DataFrame from pairs+cache so compute_binary_measure_statistics
  # gets per-trial rows.
  rows = []
  for p in pairs:
    if p.get("tell_reuse") not in (0, 1):
      continue
    c = cache.get(p["key"], {})
    label = c.get("label")
    claude_bin = (
      1.0 if label == "path_reuse" else (0.0 if label == "no_path_reuse" else None)
    )
    rows.append(
      {
        "user_id": int(p["user_id"]),
        "tell_reuse": int(p["tell_reuse"]),
        "in_canonical_cohort": bool(p["in_canonical_cohort"]),
        "rule_reuse_int": 1.0 if p["rule_reuse"] else 0.0,
        "path_reuse_ai": claude_bin,
      }
    )
  df = pl.DataFrame(rows)

  data = {}
  for tr in (1, 0):
    data[tr] = {
      "rule": _stats_for("rule_reuse_int", df, tr),
      "claude": _stats_for("path_reuse_ai", df, tr),
    }

  fig, ax = plt.subplots(figsize=(6, 4.5))
  group_labels = ["Known eval goal", "Unknown eval goal"]
  x = np.arange(len(group_labels))
  width = 0.36

  rule_color = plot_configs.default_colors["bluish green"]
  claude_color = plot_configs.default_colors["nice purple"]

  def _vals(method_key):
    means = []
    err_low = []
    err_high = []
    for tr in (1, 0):
      s = data[tr][method_key]
      if s is None:
        means.append(0)
        err_low.append(0)
        err_high.append(0)
        continue
      m = s["mean"]
      lo, hi = s["ci"]
      means.append(m * 100)
      err_low.append((m - lo) * 100)
      err_high.append((hi - m) * 100)
    return means, [err_low, err_high]

  rule_means, rule_err = _vals("rule")
  claude_means, claude_err = _vals("claude")

  bars_rule = ax.bar(
    x - width / 2,
    rule_means,
    width,
    yerr=rule_err,
    color=rule_color,
    edgecolor="black",
    linewidth=0.8,
    capsize=4,
    error_kw={"elinewidth": 1.2, "ecolor": "#222"},
    label="Path-reuse algorithm",
  )
  bars_claude = ax.bar(
    x + width / 2,
    claude_means,
    width,
    yerr=claude_err,
    color=claude_color,
    edgecolor="black",
    linewidth=0.8,
    capsize=4,
    error_kw={"elinewidth": 1.2, "ecolor": "#222"},
    label="Claude (vision-based)",
  )

  # Place value labels above the upper error bar.
  for bars, means, err in (
    (bars_rule, rule_means, rule_err),
    (bars_claude, claude_means, claude_err),
  ):
    for bar, mean, hi in zip(bars, means, err[1]):
      ax.text(
        bar.get_x() + bar.get_width() / 2,
        mean + hi + 1.5,
        f"{mean:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
      )

  ax.set_xticks(x)
  ax.set_xticklabels(group_labels, fontsize=11)
  ax.set_ylabel("Path Reuse (%)", fontsize=11)
  ax.set_ylim(0, 100)
  ax.set_title("Comparing path reuse algorithm to Claude", fontsize=12)
  ax.legend(loc="upper right", fontsize=9, frameon=False)
  ax.grid(axis="y", alpha=0.3, linestyle="--")
  ax.set_axisbelow(True)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  fig.tight_layout()

  # Print CIs for the chat / paper text.
  for tr in (1, 0):
    cond = "Known" if tr == 1 else "Unknown"
    for method in ("rule", "claude"):
      s = data[tr][method]
      if s is None:
        continue
      lo, hi = s["ci"]
      print(
        f"  {cond:>7} eval goal — {method:>6}: "
        f"mean={s['mean'] * 100:.1f}% "
        f"[95% CI: {lo * 100:.1f}%, {hi * 100:.1f}%], "
        f"is_normal={s['normality']['is_normal']}, "
        f"n_users={s['n_groups']}"
      )

  pdf_path = os.path.join(OUTPUT_DIR, "method_comparison.pdf")
  png_path = os.path.join(OUTPUT_DIR, "method_comparison.png")
  fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
  fig.savefig(png_path, bbox_inches="tight", dpi=200)
  plt.close(fig)
  print(f"Wrote method comparison plot → {pdf_path}")
  print(f"Wrote method comparison plot → {png_path}")


def print_summary(pairs, cache):
  classified = [p for p in pairs if p["key"] in cache]
  labels = [cache[p["key"]].get("label") for p in classified]
  errors = [
    cache[p["key"]].get("error") for p in classified if cache[p["key"]].get("error")
  ]

  mismatches = [
    p["key"] for p in classified if cache[p["key"]].get("derivation_mismatch")
  ]

  print("\n=== Summary ===")
  print(f"Total pairs:        {len(pairs)}")
  print(f"Classified:         {len(classified)}")
  print(f"Errors:             {len(errors)}")
  print(f"Derivation mismatches (Claude's label != decision table): {len(mismatches)}")
  if mismatches:
    for k in mismatches:
      c = cache[k]
      print(
        f"  {k}: label={c.get('label')} derived={c.get('derived_label')} "
        f"same_side={c.get('same_side')} same_approach={c.get('same_approach')}"
      )
  counts = Counter(labels)
  for lbl in (*LABEL_FOLDERS, "parse_error", None):
    print(f"  {str(lbl):>15}: {counts[lbl]}")

  print("\nAgreement vs rule-based reuse (rows = rule, cols = claude):")
  print("                  path_reuse  no_path_reuse  ambiguous  parse_error")
  for rule_val in (True, False):
    name = "rule_reuse" if rule_val else "rule_no_reuse"
    row = [
      sum(
        1
        for p in classified
        if p["rule_reuse"] is rule_val and cache[p["key"]].get("label") == lbl
      )
      for lbl in ("path_reuse", "no_path_reuse", "ambiguous", "parse_error")
    ]
    print(f"  {name:>15}: " + "  ".join(f"{v:>10}" for v in row))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--pilot", type=int, default=20, help="Pilot batch size (default 20)."
  )
  parser.add_argument(
    "--full", action="store_true", help="Classify all pairs, not just the pilot."
  )
  parser.add_argument("--workers", type=int, default=6)
  parser.add_argument("--model", default="sonnet", choices=["sonnet", "opus", "haiku"])
  parser.add_argument(
    "--no-cache", action="store_true", help="Ignore cached classifications."
  )
  parser.add_argument(
    "--regenerate-images", action="store_true", help="Force re-render of all PNGs."
  )
  parser.add_argument(
    "--dry-run-classify",
    action="store_true",
    help="Build pairs and render images, but skip claude -p.",
  )
  parser.add_argument(
    "--output-suffix",
    default="",
    help="Optional suffix; output goes to craftax_multi_claude_path_reuse_<suffix>/.",
  )
  args = parser.parse_args()

  _set_output_dir(args.output_suffix)
  _ensure_dirs()
  print(f"Output dir: {OUTPUT_DIR}")
  with open(PROMPT_AUDIT_PATH, "w") as f:
    f.write(PROMPT_TEMPLATE)
  # Paper-ready copy (placeholders replaced, JSON braces unescaped, at top level)
  paper_prompt = (
    PROMPT_TEMPLATE.replace("{train_path}", "<path to TRAIN image>")
    .replace("{test_path}", "<path to TEST image>")
    .replace("{{", "{")
    .replace("}}", "}")
  )
  with open(os.path.join(OUTPUT_DIR, "prompt.txt"), "w") as f:
    f.write(paper_prompt)

  pairs = build_pairs()
  if not pairs:
    print("No pairs found. Aborting.")
    return

  cache = {} if args.no_cache else _load_json(CLASSIFICATIONS_CACHE, {})

  if args.full:
    targets = pairs
  else:
    targets = load_or_choose_pilot(pairs, n=args.pilot)

  # Render only what we'll classify — full-cohort rendering is ~30 min and
  # not needed up front. Already-classified pair images get re-rendered for
  # the folder-mirroring step, but that's already cheap (cache hits).
  render_images(targets, regenerate=args.regenerate_images)

  if args.dry_run_classify:
    print("--dry-run-classify set; skipping claude -p.")
  else:
    classify_batch(targets, cache, model=args.model, workers=args.workers)
    _atomic_write_json(CLASSIFICATIONS_CACHE, cache)

  write_classifications_df(pairs, cache)
  mirror_label_folders(pairs, cache)
  generate_html_review(pairs, cache)
  print_summary(targets, cache)
  print_tell_reuse_summary(pairs, cache)
  plot_method_comparison(pairs, cache)

  if not args.full:
    print(
      f"\nPilot complete. Spot-check `{OUTPUT_DIR}` (path_reuse/, no_path_reuse/, "
      f"ambiguous/, classifications.csv). Re-run with `--full` once happy."
    )


if __name__ == "__main__":
  main()
