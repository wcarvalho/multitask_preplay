"""Download representative run configs for each model group in the final
train/test plots, so we can write the hyperparameters section of the paper.

Writes one YAML per (project, model) to ./final_configs/.

Usage:
    python final_configs/download_configs.py
    python final_configs/download_configs.py --refresh
"""

import argparse
import os
import sys

import wandb
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, os.pardir, "plots"))

ENTITY = "wcarvalho92"

# Mirrors plots/craftax_train_test.py
CRAFTAX = {
  "project": "craftax-multigoal",
  "groups": {
    "qlearning": "Group: ql-pnas-revision-3",
    "usfa": "Group: usfa-pnas-revision-3",
    "dyna": "Group: dyna-pnas-revision-2",
    "her": "Group: her-pnas-revision-1",
    "preplay": "Group: preplay-pnas-revision-6",
  },
}

# Mirrors plots/jaxmaze_train_test.py
JAXMAZE = {
  "project": "housemaze",
  "groups": {
    "qlearning": "ql-final-epsilon-1",
    "usfa": "Group: usfa-pnas-revision-2",
    "dyna": "Group: dyna-pnas-revision-2",
    "her": "her-pnas-revision-2",
    "preplay": "preplay-pnas-revision-5",
  },
}

# Mirrors plots/craftax_ai_results.py
CRAFTAX_AI = {
  "project": "craftax",
  "groups": {
    "ql": "ql-final-5",
    "ql-sf": "ql-sf-final-5",
    "dyna": "dyna-final-5",
    "her": "her-2",
    "preplay": "preplay-pnas-revision-3",
  },
}


def _group_name(group_str):
  """Plot scripts strip 'Group: ' / 'Name: ' prefixes and filter by group."""
  return group_str.split(": ", 1)[1] if ": " in group_str else group_str


def fetch_one_config(api, project, group_str):
  group = _group_name(group_str)
  filters = {"group": group, "state": {"$in": ["finished", "crashed"]}}
  runs = api.runs(f"{ENTITY}/{project}", filters=filters)
  runs = list(runs)
  if not runs:
    print(f"  [!] no runs for {project} / {group_str}")
    return None, None
  # Pick first run as representative (all share group/model hyperparams).
  run = runs[0]
  config = {k: v for k, v in run.config.items() if not k.startswith("_")}
  meta = {
    "project": project,
    "group_query": group_str,
    "run_id": run.id,
    "run_name": run.name,
    "n_runs_in_group": len(runs),
  }
  return config, meta


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--refresh", action="store_true")
  args = parser.parse_args()

  api = wandb.Api()

  for spec in (CRAFTAX, JAXMAZE, CRAFTAX_AI):
    project = spec["project"]
    out_dir = os.path.join(SCRIPT_DIR, project)
    os.makedirs(out_dir, exist_ok=True)
    for model, group_str in spec["groups"].items():
      out_path = os.path.join(out_dir, f"{model}.yaml")
      if os.path.exists(out_path) and not args.refresh:
        print(f"[skip] {out_path}")
        continue
      print(f"[fetch] {project} / {model} <- {group_str}")
      config, meta = fetch_one_config(api, project, group_str)
      if config is None:
        continue
      payload = {"_meta": meta, "config": config}
      with open(out_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=True, default_flow_style=False)
      print(f"        -> {out_path}")


if __name__ == "__main__":
  main()
