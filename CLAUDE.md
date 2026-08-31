# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This project uses `uv` for Python dependency management:

```bash
# Create environment and install dependencies
uv sync --python 3.11
source .venv/bin/activate

# For development with submodules (simulations)
git submodule init
git submodule update
```

## Development Commands

**Data Processing:**
```bash
# Process model data for JaxMaze experiments  
python -m ipdb -c continue data_processing/process_model_data.py --env jaxmaze --df --models qlearning usfa preplay_new dfs bfs
python -m ipdb -c continue data_processing/process_model_data.py --env jaxmaze --df --episodes --models bfs dfs
```

**Running Analysis:**
```bash
# Start Jupyter for analysis notebooks
jupyter lab

# Key analysis notebooks:
# - figures/jaxmaze_results.ipynb (JaxMaze analysis)  
# - figures/craftax_cogsci_results.ipynb (Craftax analysis)
```

**Web Experiments:**
```bash
# JaxMaze experiments
python experiments/jaxmaze/web_app.py MAN="paths"     # Two Paths Manipulation
python experiments/jaxmaze/web_app.py MAN="shortcut" # Shortcut Manipulation  
python experiments/jaxmaze/web_app.py MAN="start"    # Start Manipulation
python experiments/jaxmaze/web_app.py MAN="plan" SAY_REUSE=1  # Known goals
python experiments/jaxmaze/web_app.py MAN="plan" SAY_REUSE=0  # Unknown goals

# Craftax experiments (run load_caches.py first - takes 20-40 min)
python experiments/craftax/load_caches.py
python experiments/craftax/web_app.py SAY_REUSE=1  # Known evaluation goals
python experiments/craftax/web_app.py SAY_REUSE=0  # Unknown evaluation goals
```

**Code Quality:**
```bash
ruff check .     # Lint code  
ruff format .    # Format code
```

## Project Architecture

**Core Structure:**
- `plots/` - **All paper figure scripts** (see `plots/README.md` for the figure map)
- `experiments/` - Web-based human experiments (JaxMaze, Craftax)
- `simulations/` - AI model simulations and training scripts  
- `analysis/` - Analysis utilities and result processing
- `figures/` - Legacy figure notebooks (being migrated to `plots/`)
- `data_processing/` - Scripts for processing raw experimental data
- `craftax_cache/` - Cached game assets and precomputed paths

**Key Configuration:**
- `data_configs.py` - Central data directory configuration
- `plots/plot_configs.py` - Visualization settings and styling  
- `plots/wandb_config.py` - WandB project/group definitions and metric keys
- `pyproject.toml` - Python package configuration with uv
- `ruff.toml` - Code linting/formatting configuration

**Data Organization:**
Data directory is configurable via `MULTITASK_PREPLAY_DATA_DIR` environment variable, defaults to `../preplay_results`. Contains:
- `data/jaxmaze/` and `data/craftax/` - Raw experimental data
- `results/` - Processed analysis results  
- `analysis_figures/` - Generated analysis figures
- `env_figures/` - Environment visualization figures

**Simulation Framework:**
- Uses JAX for high-performance computing
- Hydra for experiment configuration management
- Custom RL algorithms: Q-learning, USFA, Dyna, Preplay
- Web interface built with NiceGUI for human experiments

**Dependencies:**
- JAX ecosystem for ML (jax, flax, optax)
- Data processing: polars, pandas, numpy
- Visualization: matplotlib, seaborn
- Web experiments: nicegui, fastapi
- External packages: craftax, housemaze (JaxMaze), jaxneurorl, nicewebrl

## Analysis Framework

The analysis code in `analysis/` provides statistical analysis and visualization for comparing human behavior with model predictions.

### JaxMaze Analysis (`analysis/jaxmaze_analysis.py`)

This module contains experiment-specific analysis functions for four experimental manipulations:

1. **`path_reuse_results()`** - Two Paths Manipulation
   - Tests if humans reuse previously learned paths when navigating to new goals
   - Compares success rates and path overlap between human and model data
   - Analyzes response times (RT) for "new path" vs "partial reuse" conditions

2. **`shortcut_results()`** - Shortcut Manipulation
   - Tests if humans discover and use shortcuts when available
   - Similar metrics: success rate, path reuse, comparison with models

3. **`start_results()`** - Start Position Manipulation
   - Tests RT differences when starting from familiar vs novel positions
   - Uses within-subject condition comparisons (condition 1 vs condition 2)

4. **`juncture_results()`** - Juncture Manipulation
   - Tests RT at decision points under different conditions
   - Compares across distance (short/long) and goal knowledge (known/unknown)
   - Produces combined bar plots with multiple conditions

**Common Workflow:**
- Filter users by minimum training success (first 100 qualifying users)
- Cache user IDs to ensure reproducible analysis
- Compute statistics and generate figures saved to `save_dir`
- Write statistical results to text files for paper reporting

### Craftax Analysis (`analysis/craftax_analysis.py`)

This module contains analysis functions for the Craftax environment, a more complex game-like setting:

**Precomputed Data:**
- `OPTIMAL_TEST_PATHS` / `OPTIMAL_TEST_LENGTHS` - A* optimal paths for each world configuration, cached to disk for efficiency comparison

**Main Analysis Function:**
- **`path_reuse_manipulation_analysis()`** - Primary analysis entry point
  - Filters to first 100 users per tell_reuse condition
  - Creates success rate vs path reuse scatter plots
  - Compares human data (known vs unknown eval goals) with model predictions
  - Saves figures and statistics to `save_dir`

**Plotting Functions:**
1. **`plot_success_rate_path_reuse_metrics()`** - 2D scatter plot
   - Separates human data by `tell_reuse` (known=1 vs unknown=0 eval goals)
   - Uses different markers: circle for known, X for unknown
   - Overlays model predictions if provided

2. **`plot_success_rate_path_reuse_metrics_efficiency()`** - Extended version
   - Adds `suboptimal_path` dimension (efficient vs inefficient)
   - Creates 4 human data points (2 tell_reuse × 2 efficiency levels)
   - Lighter colors for suboptimal/inefficient paths

3. **`plot_efficiency()`** - Two-panel bar plot
   - Shows proportion of efficient vs inefficient paths
   - Separate panels for known and unknown eval goal conditions

4. **`plot_non_reuse_frequency_by_world_seed()`** - Grouped bar plot
   - Compares reuse frequency by world seed
   - User data vs preplay model side-by-side

**Visualization:**
- **`visualize_user_path_reuse()`** - Detailed per-user visualization
  - Shows train/test maze images with path overlays (red arrows)
  - Displays reaction times as bar plot
  - Reports path overlap, success, and optimal path length comparison

**Key Differences from JaxMaze:**
- Lower overlap threshold (typically 0.15 vs 0.6-0.7 for JaxMaze)
- Optional `cosine_threshold` for direction-based path similarity
- Path efficiency analysis (suboptimal_path dimension)
- A* pathfinding for optimal path comparison

### Analysis Utilities (`analysis/analysis_utils.py`)

Core utility functions used across all analyses:

**Data Processing:**
- `get_polars_df()` - Converts DataFrame/nicewebrl DataFrame to polars
- `add_reuse_column()` - Creates boolean reuse column based on path overlap threshold
- `filter_users_by_success_by_tell_reuse()` - Filters to first N users per condition
- `compute_condition_difference_df()` - Computes within-subject differences between conditions

**Statistical Analysis:**
- `compute_binary_measure_statistics()` - Analyzes binary proportions (success, reuse)
  - Tests normality via Shapiro-Wilk
  - Uses t-test for normal data, Wilcoxon signed-rank for non-normal
  - Computes effect sizes (Cohen's d or r)
  - Bootstrap confidence intervals for medians

- `power_analysis_rt_across_groups()` - Between-groups RT comparison
  - Uses linear mixed effects models (`statsmodels.mixedlm`)
  - Accounts for repeated measures within subjects
  - Computes power via simulation

- `power_analysis_rt_differences()` - Within-subject RT differences
  - Similar mixed effects approach for paired comparisons

- `mixed_effects_compute_power()` - Simulates statistical power
  - Parallel simulation support for efficiency

**Plotting Functions:**
- `plot_success_rate_path_reuse_metrics()` - 2D scatter plot (reuse vs success) with error bars
- `plot_bar_rt_comparison()` - Bar/box plots comparing RT between conditions
- `plot_rt_differences()` - Bar plots of RT differences with CIs
- `bar_plot_error()` - Human vs model comparison bar charts

**Stats Reporting:**
- `add_to_file()` - Appends results to YAML file for paper statistics
- Statistics are formatted as paper-ready text (e.g., "Mean=X, Median=Y [95% CI: a, b], t(df)=Z, p=P")

### Key Statistical Concepts

- **Path Overlap:** Proportion of test path that overlaps with training paths (threshold typically 0.6-0.7)
- **Response Time Measures:** `first_log_rt` (first action), `max_log_rt` (slowest action), `avg_log_rt` (average)
- **Mixed Effects Models:** Account for non-independence of trials within subjects
- **Bootstrap CIs:** 10,000 resamples for robust confidence intervals on medians
- **Hodges-Lehmann Estimator:** Robust measure of central tendency for Wilcoxon tests

## Data Processing

The `data_processing/` directory contains scripts for converting raw experimental data into structured DataFrames.

### User Data Processing (`data_processing/process_user_data.py`)

Processes raw JSON files from web experiments into polars DataFrames.

**CLI Usage:**
```bash
# Process human data
python data_processing/process_user_data.py --env jaxmaze --df --episodes
python data_processing/process_user_data.py --env craftax --df
```

**Entry Points:**
- `get_jaxmaze_human_data()` - Process JaxMaze user data
- `get_craftax_human_data()` - Process Craftax user data

**Processing Pipeline:**

1. **Read Raw Data** - `nicewebrl.read_all_records_sync()`
   - One JSON file per user session
   - Filters incomplete sessions and practice trials

2. **Separate into Episodes** - `seperate_data_into_episodes()`
   - Groups timesteps by block/stage using environment-specific parsers
   - Creates unique `user_episode_idx` per episode

3. **Generate Episode Data** - `generate_file_episodes_data()`
   - Creates `EpisodeData` objects containing:
     - `actions` - action indices taken
     - `positions` - agent positions over time
     - `timesteps` - full environment state
     - `reaction_times` - time between image shown and action taken
   - Validates and fixes step ordering if needed
   - Caches per-user processed data to `cache/` directory

4. **Generate DataFrame** - `generate_all_episodes_df()`
   - Creates polars DataFrame with episode metadata
   - Computes derived measures (see below)
   - Adds path reuse columns via environment-specific utilities
   - Saves to parquet file

**Derived Measures:**
| Measure | Description |
|---------|-------------|
| `success` | Whether goal was reached |
| `path_length` | Number of actions (excluding final) |
| `first_rt` / `first_log_rt` | First action response time |
| `avg_rt` / `avg_log_rt` | Average response time |
| `max_rt` / `max_log_rt` | Maximum response time |
| `total_rt` / `total_log_rt` | Sum of response times |
| `eval_shares_start_pos` | Whether eval shares start position with training |
| `min_train_success` | Whether user passed training success threshold |
| `overlap` | Path overlap with corresponding training episode |
| `reuse` | Boolean reuse indicator (overlap > threshold) |
| `train_test_cosine` | Cosine similarity of train/test path directions |
| `corresponding_train_episode_idx` | Index of matched training episode |

**Output Files:**
- `human_data_episodes.safetensor` - Serialized episode data (Flax)
- `human_data_episode_metadata.json` - Episode metadata
- `human_data_episode_df.parquet` - Final DataFrame

### Environment-Specific Utilities

Each environment has utility modules that provide specialized functions:

**`data_processing/utils_jaxmaze.py`:**
- `get_block_stage_description()` - Parse block/stage info from metadata
- `deserialize_timestep()` - Convert raw data to HouseMaze timestep
- `make_human_episode_row_data()` - Create DataFrame row from episode
- `add_reuse_columns()` - Compute path overlap with training episodes
- `success()` - Check if correct goal was reached
- `compute_if_block_passed()` - Check if user passed training threshold (16+ successes)

**`data_processing/utils_craftax.py`:**
- Similar functions adapted for Craftax environment
- Different training threshold logic
- A* pathfinding integration for optimal path comparison

### Refreshing model parquets from the Kempner cluster

Model `_episode_df.parquet` generation has moved to the Kempner cluster — for the models that are processed there, the cluster is now authoritative. Don't re-run `process_model_data.py --df` for those models locally; pull the parquet down instead.

- **Remote layout:** `/n/holylfs06/LABS/kempner_fellow_wcarvalho/jax_rl_results/model_analysis/pnas_eps0.1/{env}/final/{model}_episode_df.parquet`, where `{env} ∈ {jaxmaze, craftax}` and `{model}` is one of `qlearning`, `usfa`, `dyna`, `her`, `preplay`. The `pnas_eps0.1/` parent encodes the eval-policy epsilon — bump this when other configurations come online.
- **Local target:** `dataframes/{env}_{model}.parquet`. This is what `data_configs.get_dataframe_path(env, model)` returns and what every paper plot reads via `data_configs.load_dataframes(env)`. Note the filename rewrite — drop `_episode_df`, prepend the env.
- **SSH host alias:** `rcfas_login` (already used by `data_processing/download_model_data_slurm.py`; resolves to `holylogin07.rc.fas.harvard.edu` per `~/.ssh/config`).

One-shot copy for the models you want to refresh:
```bash
REMOTE=rcfas_login:/n/holylfs06/LABS/kempner_fellow_wcarvalho/jax_rl_results/model_analysis/pnas_eps0.1
LOCAL=/Users/wilka/git/projects/multitask_preplay/dataframes
for ENV in jaxmaze craftax; do
  for MODEL in usfa her; do  # add/remove models as needed
    scp "$REMOTE/$ENV/final/${MODEL}_episode_df.parquet" "$LOCAL/${ENV}_${MODEL}.parquet"
  done
done
```

After refreshing, regenerate any plot that consumes those parquets — at minimum `plots/jaxmaze_train_test_df.py` and `plots/craftax_train_test_df.py`. See `plots/README.md` for the full figure map. The plot scripts read via `load_dataframes`, so no code edits are needed when only the parquets change.

## Code Conventions

**Always use the canonical filtered data for analysis.** When working with human experiment data (JaxMaze or Craftax), NEVER load the raw parquet and filter manually. Instead, use the existing filter functions which apply the correct cohort selection (first 100 qualifying users, training success thresholds, correct world/manipulation filters, cached user IDs for reproducibility):
- JaxMaze: `jaxmaze_analysis.get_path_reuse_eval_data(user_df)`, `jaxmaze_analysis.get_shortcut_eval_data(user_df)`
- Craftax: `craftax_analysis.get_path_reuse_eval_data(user_df)`

**Polars filtering:** This codebase uses polars keyword-arg filtering: `df.filter(eval=True)` is equivalent to `df.filter(pl.col("eval") == True)`.

**"Closest to threshold" sorting:** When selecting N samples closest to a threshold, sort **descending** (`reverse=True`) for below-threshold and **ascending** for above-threshold before slicing `[:N]`. A plain ascending sort on below-threshold samples picks the **furthest** from the threshold, not the closest. This mistake has been made multiple times.

## Plots Directory Conventions (`plots/`)

**Naming:** Every script in `plots/` MUST have an environment prefix:
- `jaxmaze_` — JaxMaze/HouseMaze scripts
- `craftax_multi_` — Craftax multigoal scripts
- `craftax_ai_` — Craftax AI scripts
- No prefix only for cross-environment scripts (e.g., `overlap_distribution.py`, `optimal_length.py`, `train_test.py`)

**Output naming:** Every output file MUST start with the script name (minus `.py`). When `jaxmaze_rts.py` produces files, they are `jaxmaze_rts.pdf` or `jaxmaze_rts_{suffix}.pdf`. Never use unrelated output names.

**Save location — DO NOT use `data_configs.*_RESULTS_DIR` in `plots/` scripts.** Those paths resolve to external drives or directories outside the repo. Scripts in `plots/` MUST define their own `OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")` and pass it as `save_dir` to any analysis function that accepts one. When calling analysis functions that default to `data_configs.*_RESULTS_DIR`, always pass `save_dir=OUTPUT_DIR` explicitly.

**WandB data — inspect before editing.** When a plot shows wrong values, DO NOT guess at the fix. First inspect the cached data (`python -c "import json, pandas as pd; ..."`) to see the actual setting names, metric keys, and values. Show the user what you found before proposing changes. The `final-5` groups (e.g., `preplay-final-5`, `dyna-final-5`) log metrics for every training env count (`evaluator_performance-8`, `-16`, ..., `-512`). Forgetting to filter to the right setting silently averages across all of them.

**WandB metric keys — DO NOT guess key names.** When you need metric key names for a WandB project, query the API to find them:
```python
import wandb
api = wandb.Api()
runs = api.runs('wcarvalho92/<project>', filters={'group': '<group>'})
run = list(runs)[0]
keys = [k for k in run.summary.keys() if '<search_term>' in k.lower()]
for k in sorted(keys): print(repr(k))
```
Key names often contain `\n` and other surprising characters. Never guess — always verify against the actual run.

JaxMaze and Craftax log different metric names:
- Craftax (`craftax-multigoal`, `craftax` projects): `actor_performance/0.episode_return`
- JaxMaze/HouseMaze (`housemaze` project): `actor_performance/0.0 avg_episode_return`

**Output format:** All figures are PDF (for PNAS paper). Save to `plots/output/`.

**After adding/modifying plot scripts**, run `/update-plots-readme` to keep `plots/README.md` current.

## Updating the paper's OmniGraffle composite figures programmatically

The PNAS paper's main-text Figures 3/4 are OmniGraffle composites (`~/Library/CloudStorage/Dropbox/personal/omni/2024-preplay-v2/preplay-figure-{3-jaxmaze,4-crafter-cogsci}.{graffle,pdf}`) that embed scatter panels generated from `plots/output/` (e.g. `jaxmaze_results/1.path_reuse_tell_reuse=1/success_rate_path_reuse_mean.pdf`, `4.shortcut_tell_reuse=1/success_rate_path_reuse_mean.pdf`, `craftax_multi_results/5.craftax_path_reuse_manipulation/success_path_reuse_mean.pdf` — the paper uses the `_mean` variants). OmniGraffle AppleScript export is Pro-gated on this machine, so after regenerating a panel, update the composites without the app:

1. Back up the `.graffle` + exported `.pdf`.
2. The `.graffle` is a flat zip: replace the matching embedded `imageN.pdf` (identify by rendering each) with the new panel under the same internal name, same page size (393.794×288.738 pt for these scatters), and re-zip.
3. Patch the exported composite `.pdf` with PyMuPDF: locate the panel rect via OpenCV template match of the old panel (cross-check against `data.plist` bounds), stamp white underlay + new panel via `show_pdf_page` at that rect, and re-overlay any OmniGraffle panel labels ("(B)", "(C)") clipped from the original if they sat on the panel corner.
4. Verify by pixel-diffing old vs new (changes confined to the panel rects) and visual PNG inspection; then copy into the paper repo via its `move_figures.sh` and rebuild.

Full step-by-step recipe with caveats lives in `~/git/papers/preplay-writing/CLAUDE.md` ("Updating OmniGraffle composite figures programmatically").