"""Bar-plot train/test performance from local model parquet dataframes.

Sources data from data_configs.load_dataframes(env) instead of WandB.
Parquets contain only final evaluation episodes per (algo, seed), so only
bar plots are supported (no learning curves).

Test panels reuse the main paper's compute path: each test panel calls
`analysis_utils.get_model_success_rate_path_reuse_data` directly — the same
function jaxmaze_analysis.path_reuse_results / shortcut_results and
craftax_analysis.path_reuse_manipulation_analysis use to populate
paper_stats.yaml. So test bars agree with paper_stats.yaml to the digit.

Train panels can't go through that function because training rows have
NaN overlap by construction (no training path to overlap against), and
add_reuse_column drops NaN-overlap rows. Train panels therefore call
`compute_binary_measure_statistics` directly with the same center="mean"
convention.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import data_configs
import plot_configs
from analysis import analysis_utils


TITLE_SIZE = 24
LABEL_SIZE = 20
TICK_SIZE = 18
LEGEND_SIZE = 18
LINE_WIDTH = 2


def _stats_from_compute(df: pl.DataFrame, metric: str) -> dict:
  """Train-side: per-algo {center, err} via compute_binary_measure_statistics."""
  stats_by_algo = {}
  for algo in df["algo"].unique().to_list():
    algo_df = df.filter(algo=algo)
    if len(algo_df) == 0:
      continue
    s = analysis_utils.compute_binary_measure_statistics(
      algo_df, "user_id", metric, center="mean"
    )
    center = s["center_value"]
    if s["normality"]["is_normal"]:
      err = s["se"]
    else:
      ci_low, ci_high = s["ci"]
      err = np.array([[center - ci_low], [ci_high - center]])
    stats_by_algo[algo] = {"center": center, "err": err}
  return stats_by_algo


def _stats_entry_from_normality(
  center: float, error, is_normal: bool, scale: float = 1.0
) -> dict:
  """Pack a (center, err) entry from normality-aware stats output.

  scale=1/100 is used when the upstream function returns percentages.
  """
  if is_normal:
    err = error * scale
  else:
    ci_low, ci_high = error[0] * scale, error[1] * scale
    err = np.array([[center - ci_low], [ci_high - center]])
  return {"center": center, "err": err}


def _human_test_stats(
  test_df: pl.DataFrame,
  overlap_threshold: float,
  cosine_threshold: float | None,
  split: str,
) -> dict:
  """Test-side human stats. Mirrors the main-paper compute path."""
  if split == "by_tell_reuse":
    out = {}
    for tell_reuse, key in [(1, "human_known"), (0, "human_unknown")]:
      sub_df = test_df.filter(tell_reuse=tell_reuse)
      if len(sub_df) == 0:
        continue
      d = analysis_utils.get_human_success_rate_path_reuse_data(
        df=sub_df,
        overlap_threshold=overlap_threshold,
        cosine_threshold=cosine_threshold,
        experiment_name=None,
        center="mean",
      )["human"]
      center = d["success"] / 100
      out[key] = _stats_entry_from_normality(
        center, d["success_error"], d["success_is_normal"], scale=1 / 100
      )
    return out

  d = analysis_utils.get_human_success_rate_path_reuse_data(
    df=test_df,
    overlap_threshold=overlap_threshold,
    cosine_threshold=cosine_threshold,
    experiment_name=None,
    center="mean",
  )["human"]
  center = d["success"] / 100
  return {
    "human": _stats_entry_from_normality(
      center, d["success_error"], d["success_is_normal"], scale=1 / 100
    )
  }


def _human_train_stats(train_df: pl.DataFrame, metric: str, split: str) -> dict:
  """Train-side human stats: per-cohort success on eval=False rows."""
  if metric not in train_df.columns:
    return {}

  def _one(sub_df):
    s = analysis_utils.compute_binary_measure_statistics(
      sub_df, "user_id", metric, center="mean"
    )
    center = s["center_value"]
    error = s["se"] if s["normality"]["is_normal"] else np.array(s["ci"])
    return _stats_entry_from_normality(center, error, s["normality"]["is_normal"])

  if split == "by_tell_reuse":
    out = {}
    for tell_reuse, key in [(1, "human_known"), (0, "human_unknown")]:
      sub_df = train_df.filter(tell_reuse=tell_reuse)
      if len(sub_df) == 0:
        continue
      out[key] = _one(sub_df)
    return out

  return {"human": _one(train_df)}


def _stats_from_main_paper(
  df: pl.DataFrame, overlap_threshold: float, cosine_threshold: float | None
) -> dict:
  """Test-side: per-algo {center, err} via main paper's
  get_model_success_rate_path_reuse_data. Returns success% (rescaled to 0-1
  to match _stats_from_compute output)."""
  model_data = analysis_utils.get_model_success_rate_path_reuse_data(
    model_df=df,
    overlap_threshold=overlap_threshold,
    cosine_threshold=cosine_threshold,
    center="mean",
    experiment_name=None,  # don't clobber paper_stats.yaml
  )
  out = {}
  for algo, d in model_data.items():
    center = d["success"] / 100  # back to 0-1
    out[algo] = _stats_entry_from_normality(
      center, d["success_error"], d["success_is_normal"], scale=1 / 100
    )
  return out


def _plot_bar_panel(
  ax,
  stats_map: dict,
  title: str,
  ylabel: str,
  ylim: tuple[float, float] | None,
  show_legend: bool,
  show_ylabel: bool,
  legend_loc: str,
):
  algos_present = list(stats_map.keys())
  ordered = [a for a in plot_configs.model_order if a in algos_present]
  ordered += [a for a in algos_present if a not in ordered]

  labels, means, errs, colors = [], [], [], []
  for algo in ordered:
    s = stats_map[algo]
    labels.append(plot_configs.model_names.get(algo, algo))
    means.append(s["center"])
    errs.append(s["err"])
    colors.append(plot_configs.model_colors.get(algo, "gray"))

  if not labels:
    return

  x = np.arange(len(labels))
  yerr = np.zeros((2, len(labels)))
  for i, e in enumerate(errs):
    if isinstance(e, np.ndarray) and e.shape == (2, 1):
      yerr[0, i] = e[0, 0]
      yerr[1, i] = e[1, 0]
    else:
      yerr[0, i] = e
      yerr[1, i] = e

  bars = ax.bar(
    x,
    means,
    yerr=yerr,
    capsize=5,
    color=colors,
    edgecolor="black",
    linewidth=LINE_WIDTH,
    width=0.6,
  )
  for bar, label in zip(bars, labels):
    bar.set_label(label)

  ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
  ax.set_xticks(x)
  ax.set_xticklabels([""] * len(labels))
  if show_legend and len(labels) > 1:
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9, loc=legend_loc)
  if show_ylabel:
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
  ax.tick_params(axis="y", labelsize=TICK_SIZE)
  ax.grid(True, alpha=0.3, linewidth=0.5, axis="y")
  if ylim is not None:
    ax.set_ylim(*ylim)
  ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))


def plot_train_test_from_df(
  env: str,
  save_dir: str,
  output_name: str,
  metric: str = "success",
  ylabel: str = "Success Rate",
  ylim: tuple[float, float] | None = (0, 1.1),
  splits: list | None = None,
  algos: list[str] | None = None,
  legend_ax: int = 0,
  legend_loc: str = "lower left",
  overlap_threshold: float | None = None,
  cosine_threshold: float | None = None,
  human_train_df: pl.DataFrame | None = None,
  human_test_df: pl.DataFrame | None = None,
  human_split: str = "by_tell_reuse",
):
  """Bar plot of final train/test performance per algorithm.

  When `overlap_threshold` is supplied, the test panel calls
  `analysis_utils.get_model_success_rate_path_reuse_data` directly — the
  exact function the main paper's scatter plots use. The train panel falls
  back to `compute_binary_measure_statistics` because training rows have
  NaN overlap (no training path to overlap against) and
  `add_reuse_column` would drop every row.

  `splits` accepts either 2-tuples (manipulation, title) — falling back to
  the scalar `overlap_threshold` / `cosine_threshold` — or 3-tuples
  (manipulation, title, overlap_threshold) / 4-tuples
  (manipulation, title, overlap_threshold, cosine_threshold) when
  manipulations use different thresholds (e.g. JaxMaze paths=0.5 vs
  shortcut=0.6).

  The metric arg is only honored on the train panel; the test panel uses
  "success" because that's what the main-paper function computes.
  """
  df = data_configs.load_dataframes(env)
  if algos is not None:
    df = df.filter(pl.col("algo").is_in(algos))

  if splits is None:
    split_specs: list[tuple[pl.DataFrame, str, float | None, float | None]] = [
      (df, "", overlap_threshold, cosine_threshold)
    ]
  else:
    split_specs = []
    for entry in splits:
      if len(entry) == 2:
        val, name = entry
        ot, ct = overlap_threshold, cosine_threshold
      elif len(entry) == 3:
        val, name, ot = entry
        ct = cosine_threshold
      elif len(entry) == 4:
        val, name, ot, ct = entry
      else:
        raise ValueError(f"splits entry must have 2-4 elements, got {len(entry)}")
      sdf = df.filter(pl.col("manipulation") == val)
      split_specs.append((sdf, name, ot, ct))

  def _filter_by_split(human_df, val):
    if human_df is None or val is None or "manipulation" not in human_df.columns:
      return human_df
    return human_df.filter(pl.col("manipulation") == val)

  panels: list[tuple[dict, str]] = []
  for entry_idx, (sdf, sname, ot, ct) in enumerate(split_specs):
    train_title = f"{sname} (Train)" if sname else "Train"
    test_title = f"{sname} (Test)" if sname else "Test"
    train_df = sdf.filter(pl.col("eval") == False)  # noqa: E712
    test_df = sdf.filter(pl.col("eval") == True)  # noqa: E712

    train_stats = _stats_from_compute(train_df, metric)
    if ot is not None:
      test_stats = _stats_from_main_paper(test_df, ot, ct)
    else:
      test_stats = _stats_from_compute(test_df, metric)

    split_val = splits[entry_idx][0] if splits is not None else None
    h_train = _filter_by_split(human_train_df, split_val)
    h_test = _filter_by_split(human_test_df, split_val)
    if h_train is not None and len(h_train) > 0:
      train_stats = {**_human_train_stats(h_train, metric, human_split), **train_stats}
    if h_test is not None and len(h_test) > 0 and ot is not None:
      test_stats = {**_human_test_stats(h_test, ot, ct, human_split), **test_stats}

    panels.append((train_stats, train_title))
    panels.append((test_stats, test_title))

  n_rows = len(split_specs)
  n_cols = 2
  fig, axs = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows), sharey=True)
  flat_axs = np.atleast_2d(axs).flatten()

  for i, (stats_map, title) in enumerate(panels):
    _plot_bar_panel(
      flat_axs[i],
      stats_map,
      title=title,
      ylabel=ylabel,
      ylim=ylim,
      show_legend=(i == legend_ax),
      show_ylabel=(i % n_cols == 0),
      legend_loc=legend_loc,
    )

  fig.tight_layout()
  os.makedirs(save_dir, exist_ok=True)
  path = os.path.join(save_dir, output_name)
  fig.savefig(path, bbox_inches="tight", dpi=300)
  print(f"Saved {path}")
  plt.close(fig)
