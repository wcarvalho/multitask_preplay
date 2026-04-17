# Plots

All figures for the PNAS paper "Multitask Preplay" are generated from this directory.

## Quick Start

```bash
# Regenerate all script-based figures
uv run python plots/generate_all.py

# Run a single script
uv run python plots/generate_all.py --only train_test

# Re-fetch WandB data (ignores cache)
uv run python plots/generate_all.py --refresh

# List available scripts
uv run python plots/generate_all.py --list
```

## Figure Map

### JaxMaze (Human + Model DataFrame)

| Script | Paper Figure | Description | Data Source |
|--------|-------------|-------------|-------------|
| `jaxmaze_rts.py` | SI Fig: two_paths_first, juncture_first | Reaction time differences for min/median/max users in two-paths and juncture manipulations | JaxMaze human DataFrame |
| `jaxmaze_envs_experiment_figure.py` | SI Fig: environment visualizations | Maze layouts with colored paths for each experimental manipulation (two paths, juncture, start, shortcut) | Hardcoded maze definitions |
| `jaxmaze_envs_raw.py` | SI Fig: bare maze layouts | All 6 base maze layouts × 4 rotations, with and without agent, in a single grid figure | Hardcoded maze definitions |
| `jaxmaze_sf_analysis.py` | SI Fig: jaxmaze-sf-analysis | Successor feature values during training and evaluation episodes, showing how SF representations evolve | JaxMaze model data (parquet) |
| `jaxmaze_overlap_examples.py` | SI Fig: overlap threshold examples | Example episodes showing above/below overlap threshold for human and model path reuse | JaxMaze human + model DataFrame |
| `jaxmaze_rt_by_action.py` | SI Fig: rt_by_action | Reaction times split by whether the action repeated or changed from the previous timestep | JaxMaze human DataFrame |
| `jaxmaze_top_rt_analysis.py` | SI Fig: top10_relative | Analysis of the 10 slowest timesteps per episode — where deliberation happens during navigation | JaxMaze human DataFrame |
| `jaxmaze_preplay_ablation.py` | SI Fig: preplay all-goals ablation | Bar plots comparing Multitask Preplay vs ablations (no Peng's Q(λ), no CQL, no all-goals) on JaxMaze train/test | WandB: `housemaze` |
| `jaxmaze_her_ablation.py` | SI Fig: HER all-goals ablation | Bar plots comparing HER with vs without all-goals learning on JaxMaze train/test | WandB: `housemaze` |

### Craftax

| Script | Paper Figure | Description | Data Source |
|--------|-------------|-------------|-------------|
| `craftax_ai_results.py` | Main Fig 5, SI Fig: AI_train_eval, AI_achievement_bars | Training curves and achievement scores for all RL algorithms on Craftax | WandB: `craftax` |
| `craftax_ai_ablations.py` | SI Fig: AI_preplay_ablation, AI_randomization_ablation | Sim policy/precondition ablation curves and randomized-object-location bar plot | WandB: `craftax` |
| `craftax_ai_baselines.py` | SI Fig: AI_baseline_comparison | Bar plot comparing Multitask Preplay against external baselines (TWM, PPO-RNN) | WandB: `craftax` + hardcoded baselines |
| `craftax_ai_dyna_multigoal.py` | SI Fig: dyna multigoal scores | Score comparison across Dyna multigoal variants on Craftax | WandB: `craftax` |
| `craftax_multi_overlap_examples.py` | SI Fig: overlap threshold examples | Example episodes with path visualizations, heatmaps, and direction vectors for Craftax | Craftax human + model DataFrame |

### Cross-Environment

| Script | Paper Figure | Description | Data Source |
|--------|-------------|-------------|-------------|
| `overlap_distribution.py` | SI Fig: overlap distributions | 2×2 histogram panel showing path overlap distributions across JaxMaze and Craftax experiments | JaxMaze + Craftax human DataFrame |
| `optimal_length.py` | SI Fig: optimal_length_deviation | Deviation from optimal (BFS/A*) path length for humans and models across experiments | JaxMaze + Craftax human + model DataFrame |
| `train_test.py` | SI Fig: train_test curves/bars | Train/test performance bar plots (or learning curves with --no-bar) for configurable WandB experiments | WandB: `craftax-multigoal`, `housemaze` |

### Notebooks (run manually)

| Notebook | Paper Figure | Description |
|----------|-------------|-------------|
| `jaxmaze_results.ipynb` | Main Fig 3 + SI distributions/success rates | JaxMaze path reuse and shortcut analysis with statistical tests |
| `craftax_cogsci_results.ipynb` | Main Fig 4 | Craftax cognitive science results — human vs model path reuse |

## Data Sources

1. **JaxMaze DataFrame** — Processed human + model parquet files. Generate with: `python data_processing/process_user_data.py --env jaxmaze --df`
2. **Craftax DataFrame** — Processed human + model parquet files. Generate with: `python data_processing/process_user_data.py --env craftax --df`
3. **WandB** — Training curves fetched from wandb.ai and cached locally in `_wandb_cache/`. Projects:
   - `wcarvalho92/housemaze` — JaxMaze RL training runs
   - `wcarvalho92/craftax-multigoal` — Craftax multigoal training runs
   - `wcarvalho92/craftax` — Craftax AI training runs

## Shared Modules

| File | Purpose |
|------|---------|
| `plot_configs.py` | Model colors, names, ordering, measure labels |
| `wandb_config.py` | WandB project/group definitions and metric keys |
| `wandb_utils.py` | Shared WandB data fetching with caching and smoothing |
| `figure_utils.py` | Plotting utilities used by notebooks |
