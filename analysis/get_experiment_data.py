"""Unified data loading and filtering for all experiments.

Mirrors the exact preprocessing from jaxmaze_analysis.py and craftax_analysis.py
so that downstream scripts get identically filtered DataFrames.
"""

from analysis.jaxmaze_analysis import filter_users_by_success
from analysis import analysis_utils
from data_processing import process_model_data, process_user_data


# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------
EXPERIMENT_CONFIGS = {
  "jaxmaze_path_reuse": dict(
    env="jaxmaze",
    human_eval_filters=dict(
      manipulation="paths",
      world="big_m3_maze1",
      eval=True,
      tell_reuse=1,
      eval_shares_start_pos=True,
    ),
    user_filter_fn="filter_users_by_success",
    analysis_name="path_reuse_results",
    model_eval_filters=dict(
      manipulation="paths",
      world="big_m3_maze1",
      eval=True,
    ),
  ),
  "jaxmaze_shortcut": dict(
    env="jaxmaze",
    human_eval_filters=dict(
      manipulation="shortcut",
      world="big_m1_maze3_shortcut",
      tell_reuse=1,
      eval_shares_start_pos=True,
    ),
    user_filter_fn="filter_users_by_success",
    analysis_name="shortcut_results",
    model_eval_filters=dict(
      world="big_m1_maze3_shortcut",
      eval=True,
    ),
  ),
  "jaxmaze_start": dict(
    env="jaxmaze",
    human_eval_filters=dict(
      manipulation="start",
      eval=True,
      tell_reuse=1,
    ),
    user_filter_fn="filter_users_by_success",
    analysis_name="start_results",
    model_eval_filters=None,
  ),
  "jaxmaze_juncture": dict(
    env="jaxmaze",
    human_eval_filters=dict(
      manipulation="juncture",
    ),
    user_filter_fn="filter_users_by_success_by_tell_reuse",
    analysis_name="juncture_results",
    model_eval_filters=None,
  ),
  "craftax_path_reuse": dict(
    env="craftax",
    human_eval_filters=dict(
      eval_shares_start_pos=True,
      eval=True,
    ),
    user_filter_fn="filter_users_by_success_by_tell_reuse",
    analysis_name=None,  # uses default from filter fn
    model_eval_filters=dict(
      eval=True,
    ),
  ),
}

# ---------------------------------------------------------------------------
# Data loaders by environment
# ---------------------------------------------------------------------------
_LOADERS = {
  "jaxmaze": {
    "user": lambda: process_user_data.get_jaxmaze_human_data(load_df_only=True),
    "model": lambda: process_model_data.get_jaxmaze_model_data(),
  },
  "craftax": {
    "user": lambda: process_user_data.get_craftax_human_data(load_df_only=True),
    "model": lambda: process_model_data.get_craftax_model_data(),
  },
}


def get_experiment_data(
  experiment: str,
  user_df=None,
  model_df=None,
) -> dict:
  """Load and filter data for a named experiment.

  Parameters
  ----------
  experiment : str
      One of the keys in ``EXPERIMENT_CONFIGS``.
  user_df : polars/nicewebrl DataFrame, optional
      Pre-loaded human DataFrame. Loaded automatically if *None*.
  model_df : polars/nicewebrl DataFrame, optional
      Pre-loaded model DataFrame. Loaded automatically if *None*.

  Returns
  -------
  dict with keys ``"user_df"``, ``"model_df"``, ``"user_ids"``, ``"env"``.
  ``"model_df"`` is *None* when the experiment has no model filters.
  """
  if experiment not in EXPERIMENT_CONFIGS:
    raise ValueError(
      f"Unknown experiment: {experiment!r}. Choose from {list(EXPERIMENT_CONFIGS)}"
    )

  cfg = EXPERIMENT_CONFIGS[experiment]
  env = cfg["env"]

  # --- load raw data if not provided ---
  if user_df is None:
    user_df = _LOADERS[env]["user"]()
  if model_df is None and cfg["model_eval_filters"] is not None:
    model_df = _LOADERS[env]["model"]()

  # --- filter human data ---
  filtered_user_df = user_df.filter(**cfg["human_eval_filters"])

  if cfg["user_filter_fn"] == "filter_users_by_success":
    kwargs = {}
    if cfg["analysis_name"] is not None:
      kwargs["analysis_name"] = cfg["analysis_name"]
    filtered_user_df, user_ids = filter_users_by_success(filtered_user_df, **kwargs)
  elif cfg["user_filter_fn"] == "filter_users_by_success_by_tell_reuse":
    kwargs = {}
    if cfg["analysis_name"] is not None:
      kwargs["analysis_name"] = cfg["analysis_name"]
    filtered_user_df, user_ids = analysis_utils.filter_users_by_success_by_tell_reuse(
      filtered_user_df, **kwargs
    )
  else:
    raise ValueError(f"Unknown filter function: {cfg['user_filter_fn']}")

  # --- filter model data ---
  filtered_model_df = None
  if model_df is not None and cfg["model_eval_filters"] is not None:
    filtered_model_df = model_df.filter(**cfg["model_eval_filters"])

  return {
    "user_df": filtered_user_df,
    "model_df": filtered_model_df,
    "user_ids": user_ids,
    "env": env,
  }
