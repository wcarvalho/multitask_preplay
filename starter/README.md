# starter/

Minimal, runnable entry points for the Craftax human data and environment. Run from the repo root; the scripts read `dataframes/craftax_human.parquet` (fetch it with `python analysis/download_dataframes.py --env craftax`) and write figures to `starter/output/`.

## Scripts

- `craftax_human_data.py` — Loads the human dataframe, splits it into train/test, prints the first rows of each, and plots matched train/test trajectory pairs.
- `craftax_human_env.py` — Instantiates the Craftax env in each train/test condition and steps it under `--policy=random` or `--policy=human` (replays a participant's recorded actions), saving a first-person rollout figure.

## Identifiers

`user_id` is a random, per-session identifier independent of the participant's CloudResearch / `worker_id` — it carries no identifying information, so use it freely as the anonymous "which participant" key.
