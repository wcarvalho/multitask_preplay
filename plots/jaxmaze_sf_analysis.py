"""Plot successor feature (SF) analysis for JaxMaze USFA model.

Generates SF value plots showing how successor features evolve over time
during training and evaluation episodes for different seeds.

Source: figures_supplemental/sf_analysis.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
  os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulations"
  )
)

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

import data_configs
from analysis.housemaze_model_data import get_usfa_data
from analysis import housemaze_utils
from analysis import experiment_analysis
from nicewebrl.dataframe import concat_list

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def get_sf_df(
  in_path=None,
  out_path=None,
  support_set=None,
  overwrite_episodes=True,
  overwrite_df=True,
  **kwargs,
):
  if support_set is None:
    support_set = ["train", "eval", "train_eval"]
  if in_path is None:
    data_dir = os.path.join(data_configs.DATA_DIRECTORY, "jaxmaze_model_data")
    in_path = f"{data_dir}/final/sf/seed=*"
  out_path = out_path or os.path.join(data_configs.DATA_DIRECTORY, "sf_analysis")

  sf_dfs = []
  for eval_task_support in support_set:
    sf_dfs.append(
      get_usfa_data(
        in_path,
        overwrite_episodes=overwrite_episodes,
        overwrite_df=overwrite_df,
        config_updates=dict(EVAL_TASK_SUPPORT=eval_task_support),
        num_episodes=1,
        vis_coeff=0.01,
        path=out_path,
        **kwargs,
      )
    )

  sf_dfs = concat_list(*sf_dfs)
  return sf_dfs


def episode_sf_value(e, idx=None):
  actions = e.actions
  preds = e.transitions.extras["preds"]
  sf_values = preds.sf  # [T, N, A, W]

  sf_values = jnp.take_along_axis(sf_values, actions[:, None, None, None], axis=-2)
  sf_values = jnp.squeeze(sf_values, axis=-2)  # [T, N, W]

  in_episode = experiment_analysis.get_in_episode(e.timesteps)
  sf_values = sf_values[in_episode]
  if idx is not None:
    sf_values = sf_values[:, idx]
  return sf_values


def save_figure(fig, filename):
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}.pdf"), bbox_inches="tight", dpi=300)
  print(f"Saved figure to {OUTPUT_DIR}/{filename}.pdf")


def make_sf_plot(sf_df, seed=1):
  train_settings = dict(
    eval=False, seed=seed, manipulation=3, EVAL_TASK_SUPPORT="eval", task=26
  )
  train_df = sf_df.filter(**train_settings)
  eval_settings = dict(
    eval=True, seed=seed, manipulation=3, EVAL_TASK_SUPPORT="eval", task=40
  )
  eval_df = sf_df.filter(**eval_settings)
  fig, axs = plt.subplots(2, 2, figsize=(10, 10))

  housemaze_utils.render_path(train_df.episodes[0], ax=axs[0, 0])
  axs[0, 0].set_title("Training episode", fontsize=15)
  housemaze_utils.render_path(eval_df.episodes[0], ax=axs[1, 0])
  axs[1, 0].set_title("Evaluation episode", fontsize=15)

  train_sf_values = episode_sf_value(train_df.episodes[0], idx=0)
  eval_sf_values = episode_sf_value(eval_df.episodes[0], idx=0)

  line_names = [
    "$g$ completion feature",
    "$g'$ completion feature",
    "",
    "",
    "$g$ landmark feature",
    "$g'$ landmark feature",
    "",
    "",
  ]

  line_mask = [True, True, False, False, True, True, False, False]

  colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
  styles = ["-", "--"]

  # Plot successor feature values for training episode
  ax = axs[0, 1]
  time_steps = np.arange(train_sf_values.shape[0])
  n_total = train_sf_values.shape[1]
  n_half = n_total // 2

  max_t = train_sf_values.shape[0]
  for i in range(train_sf_values.shape[1]):
    if line_mask is not None and not line_mask[i]:
      continue

    color_idx = i % n_half
    style_idx = i // n_half

    label = line_names[i] if i < len(line_names) else f"Feature {i}"
    ax.plot(
      time_steps,
      train_sf_values[:, i],
      label=label,
      color=colors[color_idx % len(colors)],
      linestyle=styles[style_idx % len(styles)],
    )

  ax.set_title("Main task SF Values $\\psi(s, a, w_{g})$", fontsize=15)
  ax.set_xlabel("Time Step", fontsize=15)
  ax.set_ylabel("SF Value", fontsize=15)
  ax.set_ylim(0, 1.1)
  ax.legend(loc="upper left", fontsize=12)

  # Plot successor feature values for evaluation episode
  ax = axs[1, 1]
  time_steps = np.arange(eval_sf_values.shape[0])

  for i in range(eval_sf_values.shape[1]):
    if line_mask is not None and not line_mask[i]:
      continue

    color_idx = i % n_half
    style_idx = i // n_half

    ax.plot(
      time_steps[:max_t],
      eval_sf_values[:max_t, i],
      color=colors[color_idx % len(colors)],
      linestyle=styles[style_idx % len(styles)],
    )

  ax.set_title("Counterfactual task SF Values $\\psi(s, a, w_{g'})$", fontsize=15)
  ax.set_xlabel("Time Step", fontsize=15)
  ax.set_ylabel("SF Value", fontsize=15)
  ax.set_ylim(0, 1.1)

  plt.tight_layout()
  save_figure(fig, f"jaxmaze_sf_analysis_seed_{seed}")
  return fig, axs


def main():
  sf_df = get_sf_df(
    overwrite_episodes=True,
    overwrite_df=True,
  )

  for seed in [1, 2, 4]:
    make_sf_plot(sf_df, seed=seed)


if __name__ == "__main__":
  main()
