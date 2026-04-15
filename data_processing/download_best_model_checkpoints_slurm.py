"""
Download the best saved checkpoint for the latest JaxMaze and Craftax multigoal
runs on the Kempner SLURM cluster.

This keeps the existing download_model_data_slurm.py workflow intact and adds a
new flow that:
1. Uses the plot files as the source of truth for WandB groups and test metrics.
2. Resolves the matching run directories on the cluster over SSH.
3. Scores checkpoints locally from WandB history.
4. Downloads only the best checkpoint per run into the existing
   model/seed={seed} directory layout used by process_model_data.py.

Examples:
  python data_processing/download_best_model_checkpoints_slurm.py
  python data_processing/download_best_model_checkpoints_slurm.py --env jaxmaze
  python data_processing/download_best_model_checkpoints_slurm.py --env craftax_multi
  python data_processing/download_best_model_checkpoints_slurm.py --models preplay her
  python data_processing/download_best_model_checkpoints_slurm.py --dry-run
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import wandb

REPO_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = REPO_ROOT / "plots"
for import_path in (REPO_ROOT, PLOTS_DIR):
  import_path_str = str(import_path)
  if import_path_str not in sys.path:
    sys.path.insert(0, import_path_str)

data_configs = importlib.import_module("data_configs")

ENTITY = "wcarvalho92"
SSH_HOST = "rcfas_login"

REMOTE_TRAINER_ROOTS = {
  "jaxmaze": (
    "/n/holylfs06/LABS/kempner_fellow_wcarvalho/"
    "jax_rl_results/jaxmaze_trainer"
  ),
  "craftax": (
    "/n/holylfs06/LABS/kempner_fellow_wcarvalho/"
    "jax_rl_results/craftax_multigoal_trainer"
  ),
}

COMMON_SEARCH_NAMES = {
  "qlearning": "ql-final",
  "usfa": "usfa-final",
  "dyna": "dyna-final",
  "her": "her-final",
  "preplay": "preplay-final",
}

MODEL_ALIASES = {
  "ql": "qlearning",
  "qlearning": "qlearning",
  "sf": "usfa",
  "usfa": "usfa",
  "dyna": "dyna",
  "her": "her",
  "preplay": "preplay",
  "preplay-new": "preplay",
}

REMOTE_LISTING_SCRIPT = """
import json
import sys
from pathlib import Path

trainer_root = Path(sys.argv[1])
group = sys.argv[2]
run_name = sys.argv[3]
seed = sys.argv[4]
preferred_search = [s for s in sys.argv[5].split(",") if s]

candidates = []
seen = set()

def add_candidate(path: Path) -> None:
  if not path.exists():
    return
  resolved = path.resolve()
  resolved_str = str(resolved)
  if resolved_str in seen:
    return
  seen.add(resolved_str)
  files = []
  for child in sorted(resolved.iterdir()):
    if child.is_file():
      stat = child.stat()
      files.append(
        {
          "name": child.name,
          "mtime": stat.st_mtime,
          "size": stat.st_size,
        }
      )
  wandb_run_id_path = resolved / "wandb_run_id"
  wandb_run_id = None
  if wandb_run_id_path.exists():
    wandb_run_id = wandb_run_id_path.read_text().strip() or None
  candidates.append(
    {
      "path": resolved_str,
      "search_name": resolved.parents[3].name,
      "wandb_run_id": wandb_run_id,
      "files": files,
    }
  )

for search_name in preferred_search:
  preferred_path = trainer_root / search_name / "save_data" / group / run_name / f"seed={seed}"
  add_candidate(preferred_path)

for path in sorted(trainer_root.glob(f"*/save_data/{group}/{run_name}/seed={seed}")):
  add_candidate(path)

print(json.dumps({"candidates": candidates}))
""".strip()


@dataclass(frozen=True)
class ModelSpec:
  env_name: str
  project: str
  model_key: str
  group: str
  search_name: str
  score_keys: List[str]
  step_key: str
  checkpoint_prefix: str
  local_model_dir: str
  base_local_dir: Path
  trainer_root: str


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Download best model checkpoints from the SLURM cluster."
  )
  parser.add_argument(
    "--env",
    choices=["jaxmaze", "craftax", "craftax_multi", "both"],
    default="both",
    help="Which environment to download. 'craftax' means craftax multigoal.",
  )
  parser.add_argument(
    "--models",
    nargs="*",
    help="Optional subset of models: qlearning usfa dyna her preplay.",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Resolve and score checkpoints without downloading files.",
  )
  return parser.parse_args()


def load_module(module_name: str, module_path: Path):
  spec = importlib.util.spec_from_file_location(module_name, module_path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load module from {module_path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def normalize_group_label(raw_group: str) -> str:
  return raw_group.split(": ", 1)[1] if ": " in raw_group else raw_group


def normalize_requested_models(models: Sequence[str] | None) -> set[str] | None:
  if models is None:
    return None
  normalized = set()
  for model in models:
    key = model.strip().lower()
    if key not in MODEL_ALIASES:
      raise ValueError(f"Unknown model: {model}")
    normalized.add(MODEL_ALIASES[key])
  return normalized


def build_model_specs() -> Dict[str, List[ModelSpec]]:
  plot_modules = {
    "jaxmaze": load_module("jaxmaze_train_test", PLOTS_DIR / "jaxmaze_train_test.py"),
    "craftax": load_module("craftax_train_test", PLOTS_DIR / "craftax_train_test.py"),
  }

  env_config = {
    "jaxmaze": {
      "trainer_root": REMOTE_TRAINER_ROOTS["jaxmaze"],
      "base_local_dir": Path(data_configs.JAXMAZE_MODEL_RAW_DATA_DIR),
      "local_model_dirs": {"preplay": "preplay-new"},
    },
    "craftax": {
      "trainer_root": REMOTE_TRAINER_ROOTS["craftax"],
      "base_local_dir": Path(data_configs.CRAFTAX_MODEL_RAW_DATA_DIR),
      "local_model_dirs": {},
    },
  }

  specs: Dict[str, List[ModelSpec]] = {}
  for env_name, module in plot_modules.items():
    groups = module.GROUPS
    keys = module.KEYS
    if "test_keys" in keys:
      score_keys = [metric_key for metric_key, _ in keys["test_keys"]]
    elif "test_key" in keys:
      score_keys = [keys["test_key"]]
    else:
      raise RuntimeError(f"No test metric keys found for {env_name}")

    env_specs: List[ModelSpec] = []
    for model_key, raw_group in groups.items():
      group = normalize_group_label(raw_group)
      env_specs.append(
        ModelSpec(
          env_name=env_name,
          project=module.PROJECT,
          model_key=model_key,
          group=group,
          search_name=COMMON_SEARCH_NAMES[model_key],
          score_keys=score_keys,
          step_key=keys["test_step_key"],
          checkpoint_prefix=model_key,
          local_model_dir=env_config[env_name]["local_model_dirs"].get(
            model_key, model_key
          ),
          base_local_dir=env_config[env_name]["base_local_dir"],
          trainer_root=env_config[env_name]["trainer_root"],
        )
      )
    specs[env_name] = env_specs
  return specs


def extract_seed(run: wandb.apis.public.Run) -> int:
  if "SEED" in run.config:
    return int(run.config["SEED"])
  alg_config = run.config.get("alg")
  if isinstance(alg_config, dict) and "SEED" in alg_config:
    return int(alg_config["SEED"])
  raise KeyError(f"Could not determine SEED for run {run.id}")


def is_missing_number(value: Any) -> bool:
  return value is None or (isinstance(value, float) and math.isnan(value))


def fetch_runs_for_spec(
  api: wandb.Api,
  spec: ModelSpec,
) -> List[wandb.apis.public.Run]:
  filters = {"group": spec.group, "state": {"$in": ["finished", "crashed"]}}
  runs = list(api.runs(f"{ENTITY}/{spec.project}", filters=filters))
  if not runs:
    raise RuntimeError(
      f"No WandB runs found for {spec.project} group={spec.group} ({spec.model_key})"
    )
  runs.sort(key=lambda run: (extract_seed(run), run.id))
  return runs


def fetch_eval_rows(
  run: wandb.apis.public.Run,
  score_keys: Sequence[str],
  step_key: str,
) -> List[Dict[str, Any]]:
  history_keys = ["_timestamp", step_key, *score_keys]
  rows = []
  for row in run.scan_history(keys=history_keys):
    metric_values = []
    skip_row = False
    for metric_key in score_keys:
      metric_value = row.get(metric_key)
      if is_missing_number(metric_value):
        skip_row = True
        break
      metric_values.append(float(metric_value))
    timestamp = row.get("_timestamp")
    step = row.get(step_key)
    if skip_row or is_missing_number(timestamp) or is_missing_number(step):
      continue
    rows.append(
      {
        "timestamp": float(timestamp),
        "step": float(step),
        "score": sum(metric_values) / len(metric_values),
        "metrics": {
          metric_key: float(row[metric_key])
          for metric_key in score_keys
        },
      }
    )
  if not rows:
    raise RuntimeError(
      f"No evaluation rows found for run {run.id} with keys {list(score_keys)}"
    )
  return rows


def run_ssh_listing(
  trainer_root: str,
  group: str,
  run_name: str,
  seed: int,
  preferred_search_names: Sequence[str],
) -> Dict[str, Any]:
  result = subprocess.run(
    [
      "ssh",
      SSH_HOST,
      "python",
      "-",
      trainer_root,
      group,
      run_name,
      str(seed),
      ",".join(preferred_search_names),
    ],
    input=REMOTE_LISTING_SCRIPT,
    text=True,
    capture_output=True,
    check=True,
  )
  return json.loads(result.stdout)


def resolve_remote_run(
  spec: ModelSpec,
  run: wandb.apis.public.Run,
) -> Dict[str, Any]:
  candidate_names = [run.display_name]
  if run.name and run.name not in candidate_names:
    candidate_names.append(run.name)

  all_candidates = []
  for candidate_name in candidate_names:
    listing = run_ssh_listing(
      trainer_root=spec.trainer_root,
      group=spec.group,
      run_name=candidate_name,
      seed=extract_seed(run),
      preferred_search_names=[spec.search_name],
    )
    if listing["candidates"]:
      all_candidates.extend(listing["candidates"])

  if not all_candidates:
    raise RuntimeError(
      f"No remote directory found for run {run.id} "
      f"({spec.group}/{run.display_name}/seed={extract_seed(run)})"
    )

  unique_candidates = {
    candidate["path"]: candidate for candidate in all_candidates
  }
  candidates = list(unique_candidates.values())

  exact_id_matches = [
    candidate for candidate in candidates if candidate.get("wandb_run_id") == run.id
  ]
  if len(exact_id_matches) == 1:
    return exact_id_matches[0]
  if len(exact_id_matches) > 1:
    preferred = [
      candidate
      for candidate in exact_id_matches
      if candidate["search_name"] == spec.search_name
    ]
    if len(preferred) == 1:
      return preferred[0]
    raise RuntimeError(
      f"Multiple remote directories matched WandB run id {run.id}: "
      f"{[candidate['path'] for candidate in exact_id_matches]}"
    )

  preferred_search_matches = [
    candidate
    for candidate in candidates
    if candidate["search_name"] == spec.search_name
  ]
  if len(preferred_search_matches) == 1:
    return preferred_search_matches[0]
  if len(candidates) == 1:
    return candidates[0]

  raise RuntimeError(
    f"Ambiguous remote directories for run {run.id}: "
    f"{[candidate['path'] for candidate in candidates]}"
  )


def parse_checkpoint_index(filename: str, prefix: str) -> int:
  stem = Path(filename).stem
  if stem == prefix:
    return 10**9
  suffix = stem.removeprefix(f"{prefix}_")
  if stem.startswith(f"{prefix}_") and suffix.isdigit():
    return int(suffix)
  raise ValueError(f"Invalid checkpoint filename {filename} for prefix {prefix}")


def list_checkpoint_candidates(
  remote_run: Dict[str, Any],
  prefix: str,
) -> List[Dict[str, Any]]:
  checkpoints = []
  for file_info in remote_run["files"]:
    name = file_info["name"]
    if not name.endswith(".safetensors"):
      continue
    stem = Path(name).stem
    if stem != prefix and not (
      stem.startswith(f"{prefix}_")
      and stem.removeprefix(f"{prefix}_").isdigit()
    ):
      continue
    checkpoints.append(
      {
        "name": name,
        "mtime": float(file_info["mtime"]),
        "size": int(file_info["size"]),
        "index": parse_checkpoint_index(name, prefix),
        "is_final": stem == prefix,
      }
    )
  if not checkpoints:
    raise RuntimeError(
      f"No checkpoint files found in {remote_run['path']} for prefix {prefix}"
    )
  checkpoints.sort(key=lambda checkpoint: checkpoint["index"])
  return checkpoints


def match_checkpoints_to_history(
  checkpoints: Sequence[Dict[str, Any]],
  eval_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
  matches = []
  for checkpoint in checkpoints:
    nearest_row = min(
      eval_rows,
      key=lambda row: abs(row["timestamp"] - checkpoint["mtime"]),
    )
    matches.append(
      {
        **checkpoint,
        "matched_timestamp": nearest_row["timestamp"],
        "matched_step": nearest_row["step"],
        "matched_score": nearest_row["score"],
        "matched_metrics": nearest_row["metrics"],
        "delta_seconds": checkpoint["mtime"] - nearest_row["timestamp"],
      }
    )
  return matches


def choose_best_checkpoint(
  checkpoint_matches: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
  return max(
    checkpoint_matches,
    key=lambda match: (
      match["matched_score"],
      match["matched_step"],
      match["is_final"],
      match["mtime"],
    ),
  )


def matching_state_filename(checkpoint_name: str, prefix: str) -> str:
  stem = Path(checkpoint_name).stem
  if stem == prefix:
    return f"{prefix}_state.pkl"
  return f"{stem}_state.pkl"


def download_file(
  remote_dir: str,
  filename: str,
  destination_dir: Path,
  local_name: str | None = None,
) -> None:
  destination_dir.mkdir(parents=True, exist_ok=True)
  remote_file = f"{remote_dir}/{filename}"
  destination_file = destination_dir / (local_name or filename)
  subprocess.run(
    [
      "scp",
      f"{SSH_HOST}:{remote_file}",
      str(destination_file),
    ],
    check=True,
  )


def replace_model_directories(
  temp_root: Path,
  base_local_dir: Path,
  model_dirs: Iterable[str],
) -> None:
  for model_dir in sorted(set(model_dirs)):
    source_dir = temp_root / model_dir
    if not source_dir.exists():
      continue
    target_dir = base_local_dir / model_dir
    if target_dir.exists():
      if target_dir.is_symlink():
        target_dir.unlink()
      else:
        shutil.rmtree(target_dir)
    shutil.move(str(source_dir), str(target_dir))


def collect_selected_runs(
  api: wandb.Api,
  specs: Sequence[ModelSpec],
) -> List[Dict[str, Any]]:
  selected_runs = []
  for spec in specs:
    print("=" * 72)
    print(
      f"Resolving {spec.env_name} {spec.model_key}: "
      f"{spec.project} group={spec.group}"
    )
    runs = fetch_runs_for_spec(api, spec)
    for run in runs:
      remote_run = resolve_remote_run(spec, run)
      eval_rows = fetch_eval_rows(run, spec.score_keys, spec.step_key)
      checkpoints = list_checkpoint_candidates(remote_run, spec.checkpoint_prefix)
      checkpoint_matches = match_checkpoints_to_history(checkpoints, eval_rows)
      best_checkpoint = choose_best_checkpoint(checkpoint_matches)
      file_names = {file_info["name"] for file_info in remote_run["files"]}
      state_name = matching_state_filename(
        best_checkpoint["name"], spec.checkpoint_prefix
      )
      selected_runs.append(
        {
          "env": spec.env_name,
          "project": spec.project,
          "model": spec.model_key,
          "local_model_dir": spec.local_model_dir,
          "group": spec.group,
          "search_name": remote_run["search_name"],
          "run_id": run.id,
          "run_url": run.url,
          "display_name": run.display_name,
          "seed": extract_seed(run),
          "remote_dir": remote_run["path"],
          "remote_wandb_run_id": remote_run.get("wandb_run_id"),
          "checkpoint_name": best_checkpoint["name"],
          "checkpoint_mtime": best_checkpoint["mtime"],
          "matched_timestamp": best_checkpoint["matched_timestamp"],
          "matched_step": best_checkpoint["matched_step"],
          "matched_score": best_checkpoint["matched_score"],
          "matched_metrics": best_checkpoint["matched_metrics"],
          "delta_seconds": best_checkpoint["delta_seconds"],
          "is_final_checkpoint": best_checkpoint["is_final"],
          "config_name": (
            f"{spec.checkpoint_prefix}.config"
            if f"{spec.checkpoint_prefix}.config" in file_names
            else None
          ),
          "state_name": state_name if state_name in file_names else None,
          "has_wandb_run_id_file": "wandb_run_id" in file_names,
          "candidate_matches": checkpoint_matches,
        }
      )
      print(
        f"  seed={extract_seed(run):>2} run_id={run.id} -> "
        f"{best_checkpoint['name']} "
        f"(score={best_checkpoint['matched_score']:.4f}, "
        f"delta={best_checkpoint['delta_seconds']:.1f}s)"
      )
  return selected_runs


def write_manifest(
  env_name: str,
  base_local_dir: Path,
  selected_runs: Sequence[Dict[str, Any]],
) -> Path:
  manifest = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "env": env_name,
    "runs": list(selected_runs),
  }
  manifest_path = base_local_dir / f"best_checkpoints_{env_name}.json"
  with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
  return manifest_path


def download_selected_runs(
  env_name: str,
  base_local_dir: Path,
  selected_runs: Sequence[Dict[str, Any]],
  dry_run: bool,
) -> None:
  if dry_run:
    print(f"[dry-run] Skipping downloads for {env_name}")
    return

  base_local_dir.mkdir(parents=True, exist_ok=True)
  temp_root = Path(tempfile.mkdtemp(prefix=f".{env_name}_best_", dir=base_local_dir))
  try:
    for run_info in selected_runs:
      destination_dir = (
        temp_root
        / run_info["local_model_dir"]
        / f"seed={run_info['seed']}"
      )
      filenames = [run_info["checkpoint_name"]]
      if run_info["config_name"] is not None:
        filenames.append(run_info["config_name"])
      if run_info["state_name"] is not None:
        filenames.append(run_info["state_name"])
      if run_info["has_wandb_run_id_file"]:
        filenames.append("wandb_run_id")

      print(
        f"Downloading {run_info['model']} seed={run_info['seed']} "
        f"{run_info['checkpoint_name']}"
      )
      for filename in filenames:
        local_name = None
        canonical_checkpoint = f"{run_info['model']}.safetensors"
        if filename == run_info["checkpoint_name"]:
          local_name = canonical_checkpoint
        download_file(
          run_info["remote_dir"],
          filename,
          destination_dir,
          local_name=local_name,
        )

    replace_model_directories(
      temp_root=temp_root,
      base_local_dir=base_local_dir,
      model_dirs=[run_info["local_model_dir"] for run_info in selected_runs],
    )
  finally:
    shutil.rmtree(temp_root, ignore_errors=True)


def summarize_runs(env_name: str, selected_runs: Sequence[Dict[str, Any]]) -> None:
  if not selected_runs:
    print(f"No runs selected for {env_name}")
    return
  best_intermediates = sum(
    not run_info["is_final_checkpoint"] for run_info in selected_runs
  )
  print(
    f"{env_name}: selected {len(selected_runs)} runs, "
    f"{best_intermediates} best checkpoints are intermediate."
  )


def main() -> None:
  args = parse_args()
  requested_models = normalize_requested_models(args.models)
  env_choice = "craftax" if args.env == "craftax_multi" else args.env

  all_specs = build_model_specs()
  selected_envs = ["jaxmaze", "craftax"] if env_choice == "both" else [env_choice]
  api = wandb.Api()

  for env_name in selected_envs:
    env_specs = all_specs[env_name]
    if requested_models is not None:
      env_specs = [
        spec for spec in env_specs if spec.model_key in requested_models
      ]
      if not env_specs:
        raise RuntimeError(
          f"No models left to process for {env_name} after filtering "
          f"{sorted(requested_models)}"
        )

    selected_runs = collect_selected_runs(api, env_specs)
    summarize_runs(env_name, selected_runs)
    manifest_path = write_manifest(
      env_name=env_name,
      base_local_dir=env_specs[0].base_local_dir,
      selected_runs=selected_runs,
    )
    print(f"Wrote manifest: {manifest_path}")
    download_selected_runs(
      env_name=env_name,
      base_local_dir=env_specs[0].base_local_dir,
      selected_runs=selected_runs,
      dry_run=args.dry_run,
    )


if __name__ == "__main__":
  main()
