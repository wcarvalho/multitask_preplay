"""Helpers for building the preplay paper-launch promo MP4 (Craftax).

Driven from `plots/make_promo_video.ipynb` because Craftax texture loading and
JAX compilation are expensive — we load once per kernel and iterate on visuals.

The three scene rollouts take an explicit RNG key and are cached to disk, so
re-rendering frames never re-rolls the rollouts (prevents preview/encode drift).
"""

from __future__ import annotations

import functools
import hashlib
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax.traverse_util import unflatten_dict
from matplotlib.patches import Circle, FancyArrowPatch
from safetensors.flax import load_file

# Repo layout: .../multitask_preplay/plots/make_promo_video.py
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _extra in (
  _REPO_ROOT / "simulations",
  _REPO_ROOT / "experiments" / "craftax",
  _REPO_ROOT / "plots",
):
  if str(_extra) not in sys.path:
    sys.path.insert(0, str(_extra))

from promo_video_design import (
  DESIGN,
  ENCODE_FFMPEG_PARAMS,
  THEMES,
  add_caption_gradient,
  add_grain,
  add_vignette,
  bridge_scenes,
  interp_trail,
  ken_burns_imshow,
  load_fonts,
  title_card_frames,
  use_theme,
)

# ────────── Constants ──────────

# Index into the 3-slot one-hot task_w used by the Craftax multi-goal env.
# Confirmed against simulations/craftax_web_env.py:676-688.
ACHIEVEMENT_IDX = {"diamond": 0, "sapphire": 1, "ruby": 2}

# Theme-aware color accessors. Resolve from DESIGN at call time, so swapping
# themes via `use_theme(name)` (or passing `theme=...` to a builder) changes
# trail/chrome colors mid-session without re-importing. Use the functions
# inside helpers that draw — they hand a fresh hex string to matplotlib.


def train_color() -> str:
  """Hex color for on-task trails / landmark A under the active theme."""
  return DESIGN["blue"]


def test_color() -> str:
  """Hex color for off-task trails / landmark B under the active theme."""
  return DESIGN["amber"]


# Title-card copy. Eyebrow is uppercased + tracked-out by the helper. The
# subtitle is intentionally None — the paper develops AI generalization AND
# explains hippocampal preplay, but the launch thread (not the title card)
# carries that story; one bold title reads cleaner.
TITLE_EYEBROW = "MULTITASK PREPLAY · 2026"
TITLE_TEXT = "One mechanism. Two puzzles."
TITLE_SUBTITLE = (
  "Preemptive Solving of Future Problems:\nMultitask Preplay in Humans and Machines"
)

# Single editable knob-set for the storyboard. Caption / duration / Ken-Burns
# pan changes happen here, not inside the builder code. Per-block builders read
# their entry by name (e.g. STORYBOARD["panel1"]).
#
# Panel sequencing model: visuals are introduced one at a time so the viewer
# isn't overwhelmed.
#   panel0 — `phases` reveal (map → +blue ball → +red ball), continuous Ken Burns.
#   panel1 — animation, then `label_phases` hold the last frame with labels
#            appearing one at a time.
#   panel2 — `intro_phases` reveal labels on frame 0 BEFORE rollout, then
#            animation with no labels, then `final_pause_s` hold with both
#            labels back. `drop_pickup=True` removes the pickup step.
#   panel3 — animation, then `final_pause_s` hold with the "previous preplay
#            goal" label. `drop_pickup=True`.
STORYBOARD: dict = {
  "size": (1280, 720),
  "crossfade_n": 10,
  # encode_fps is the FINAL MP4 container framerate. Each panel builds at its
  # own perceived `fps` (unique frames per wall-clock second); the orchestrator
  # repeats each panel's frames so the joined storyboard plays at one rate.
  "encode_fps": 20,
  "title": {"duration_s": 1.4, "fade_s": 0.25, "fps": 20},
  "title_reprise": {"duration_s": 1.4, "fade_s": 0.30, "fps": 20},
  "panel0": {
    "phases": [
      {"goals": (), "labels": {}, "duration_s": 2.0},
      {
        "goals": ("sapphire",),
        "labels": {"sapphire": ("goal 1", "top-left")},
        "duration_s": 2.0,
      },
      {
        "goals": ("sapphire", "diamond"),
        "labels": {
          "sapphire": ("goal 1", "top-left"),
          "diamond": ("goal 2", "top-right"),
        },
        "duration_s": 2.0,
      },
    ],
    "caption": "A large world where goals are clustered.",
    # Static panel — no Ken Burns / pan. Kept for backwards-compat readers.
    "fps": 4,
  },
  "panel1": {
    "duration_s": 1.0,
    "caption": "On-task experience.",
    "fps": 2,  # deliberate / slow — was 5
    # Pre-intro reveals before the rollout: minimap (no star/trail) → minimap
    # with start star → animation. The earlier "partial-only" beat was dropped
    # because the partial view by itself wasn't telling the viewer anything
    # (and rendered in a different axes box that made it visibly larger than
    # subsequent dual-layout frames).
    #   minimap_no_star_s — + minimap (no star, no trail) — "here's the world"
    #   minimap_star_s    — + star at start cell — "agent starts here"
    "minimap_no_star_s": 2.0,
    "minimap_star_s": 2.0,
    "label_phases": [
      {"labels": ("current",), "duration_s": 2.0},
      {"labels": ("current", "accessible"), "duration_s": 2.0},
    ],
  },
  "panel2": {
    "duration_s": 2.5,
    "caption": "**Preplay**: Reuse experience to simulate pursuit of another goal",
    "title": 'What the agent "imagines" during replay',
    "fps": 2,
    # Pre-intro reveals on frame 0, BEFORE the label phases. Each beat
    # progressively adds one element so the viewer can absorb them in turn:
    #   image_only_s  — partial view alone (no caption, no title, no labels)
    #   caption_only_s — caption + title appear (still no labels)
    "image_only_s": 1.0,
    "caption_only_s": 5.0,
    "intro_phases": [
      {"labels": ("on_task",), "duration_s": 2.0},
      {"labels": ("on_task", "counterfactual"), "duration_s": 2.0},
    ],
    "final_pause_s": 3.0,
    "drop_pickup": True,
  },
  "panel3": {
    "duration_s": 2.5,
    "caption": "When previously accessible goals are tasks,\nno more planning is needed.",
    "fps": 2,
    "final_pause_s": 5.0,
    "drop_pickup": True,
  },
}

# Loaded lazily so importing the module is cheap; first call also prints the
# fallback warning if Inter isn't installed.
_FONTS_CACHE: dict | None = None


def _fonts() -> dict:
  global _FONTS_CACHE
  if _FONTS_CACHE is None:
    _FONTS_CACHE = load_fonts()
  return _FONTS_CACHE


def _draw_partial_view_labels(
  ax,
  labels: list[tuple[str, str, str]],
  *,
  arrow_targets: list[tuple[float, float] | None] | None = None,
) -> None:
  """Paint colored corner badges on top of the partial view.

  Args:
    labels: list of (text, color_hex, position). Position ∈ {top-left,
      top-right, bottom-left, bottom-right}.
    arrow_targets: optional, same length as labels. Each entry is a (y, x)
      pixel coordinate in the imshow'd image (data coords) — the badge will
      annotate with a leader-line arrow to that point. None = no arrow.
  """
  positions = {
    "top-left": (0.04, 0.96, "left", "top"),
    "top-right": (0.96, 0.96, "right", "top"),
    "bottom-left": (0.04, 0.04, "left", "bottom"),
    "bottom-right": (0.96, 0.04, "right", "bottom"),
  }
  for i, (text, color_hex, pos) in enumerate(labels):
    x, y, ha, va = positions[pos]
    target = arrow_targets[i] if arrow_targets is not None else None
    # Opaque pill — alpha removed per design feedback (don't see image behind).
    bbox = dict(
      facecolor=DESIGN["bg"],
      edgecolor=color_hex,
      linewidth=2.0,
      boxstyle="round,pad=0.4",
    )
    if target is None:
      ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        color=color_hex,
        fontproperties=_font("semibold", 18),
        bbox=bbox,
        zorder=30,
      )
    else:
      ty, tx = target
      # Curve from corner badge toward goal — opposite curl per side keeps
      # the leader from crossing back over its own badge.
      curve = 0.18 if pos.endswith("right") else -0.18
      ax.annotate(
        text,
        xy=(tx, ty),
        xycoords="data",
        xytext=(x, y),
        textcoords="axes fraction",
        ha=ha,
        va=va,
        color=color_hex,
        fontproperties=_font("semibold", 18),
        bbox=bbox,
        arrowprops=dict(
          arrowstyle="-|>",
          color=color_hex,
          lw=2.5,
          shrinkA=4,
          shrinkB=8,
          connectionstyle=f"arc3,rad={curve}",
        ),
        zorder=30,
      )


def _goal_pixel_in_partial(
  agent_pos: tuple[int, int],
  goal_pos: tuple[int, int],
  *,
  upsample_size: tuple[int, int] = (720, 720),
) -> tuple[float, float] | None:
  """Pixel position (y, x) of `goal_pos` in the upsampled partial view.

  Returns None when the goal is outside the agent's view. The native partial
  view is OBS_DIM[0] × OBS_DIM[1] tiles in the game area + INVENTORY_OBS_HEIGHT
  rows of inventory below; the agent sits at tile (OBS_DIM[0]//2, OBS_DIM[1]//2).
  After `_upsample` to upsample_size (passed as (W, H)), tile pixels stretch
  non-uniformly — we account for that here.
  """
  from craftax.craftax.constants import (
    OBS_DIM,
    BLOCK_PIXEL_SIZE_IMG,
    INVENTORY_OBS_HEIGHT,
  )

  agent_tile_y, agent_tile_x = OBS_DIM[0] // 2, OBS_DIM[1] // 2
  dy_tile = goal_pos[0] - agent_pos[0]
  dx_tile = goal_pos[1] - agent_pos[1]
  goal_tile_y = agent_tile_y + dy_tile
  goal_tile_x = agent_tile_x + dx_tile
  # In-view check: only the game area, not the inventory bar at bottom.
  if not (0 <= goal_tile_y < OBS_DIM[0]):
    return None
  if not (0 <= goal_tile_x < OBS_DIM[1]):
    return None
  # Native pixel position (center of the tile in the rendered view).
  native_w = OBS_DIM[1] * BLOCK_PIXEL_SIZE_IMG
  native_h = (OBS_DIM[0] + INVENTORY_OBS_HEIGHT) * BLOCK_PIXEL_SIZE_IMG
  y_native = (goal_tile_y + 0.5) * BLOCK_PIXEL_SIZE_IMG
  x_native = (goal_tile_x + 0.5) * BLOCK_PIXEL_SIZE_IMG
  # Upsample is (W, H) → scale factors per axis.
  W, H = upsample_size
  return (y_native * H / native_h, x_native * W / native_w)


def _resolve_goal_arrow_targets(
  bundle: "CraftaxBundle", state, env_params
) -> list[tuple[float, float] | None]:
  """Compute (sapphire_pix, diamond_pix) arrow targets for the labeled-frame
  overlay. Returns [None, None] when goals fall outside the agent's view.

  Recomputed per frame so arrow heads track the moving partial view as the
  agent steps toward the goal cluster."""
  raw_state = _unwrap_state(state)
  agent_pos = (
    int(raw_state.player_position[0]),
    int(raw_state.player_position[1]),
  )
  try:
    sapphire_cell = _goal_cell_xy(env_params, ACHIEVEMENT_IDX["sapphire"])
    sapphire_pix = _goal_pixel_in_partial(agent_pos, sapphire_cell)
  except Exception:
    sapphire_pix = None
  try:
    diamond_cell = _goal_cell_xy(env_params, ACHIEVEMENT_IDX["diamond"])
    diamond_pix = _goal_pixel_in_partial(agent_pos, diamond_cell)
  except Exception:
    diamond_pix = None
  return [sapphire_pix, diamond_pix]


def _font(weight: str, size: int):
  """Return a FontProperties for `weight` (regular/semibold/bold) at `size` pt.

  Necessary because matplotlib's `set_title(fontproperties=fp, fontsize=N)`
  uses fp's size (default 10pt) and SILENTLY ignores fontsize. Always pass
  `fontproperties=_font(...)` to set_title and skip fontsize."""
  fp = _fonts()[weight].copy()
  fp.set_size(size)
  return fp


OUTPUT_DIR = _REPO_ROOT / "plots" / "output" / "make_promo_video"
CACHE_DIR = OUTPUT_DIR / "_cache"


# ────────── Bundle ──────────


@dataclass
class CraftaxBundle:
  """Everything a rollout/render needs, loaded once per kernel."""

  env: Any
  env_params_train: Any
  env_params_test: Any
  agent: Any
  params: Any
  config: dict
  checkpoint_path: Path
  signature: str  # sha1(checkpoint + config); used for rollout cache keying


def _signature(checkpoint_path: Path, config: dict) -> str:
  """Stable signature so rollout cache invalidates when checkpoint/config changes."""
  h = hashlib.sha1()
  h.update(str(checkpoint_path.resolve()).encode())
  h.update(str(checkpoint_path.stat().st_mtime_ns).encode())
  # Rough config fingerprint — just the keys that would change rollout behavior
  for k in sorted(config.keys()):
    h.update(f"{k}={config.get(k)!r}".encode())
  return h.hexdigest()[:12]


def load_craftax(checkpoint_local_dir: Path) -> CraftaxBundle:
  """Expensive: import craftax modules, build env, instantiate agent, load params.

  `checkpoint_local_dir` must contain `preplay.safetensors` and `preplay.config`.
  """
  checkpoint_local_dir = Path(checkpoint_local_dir).expanduser()
  sf_path = checkpoint_local_dir / "preplay.safetensors"
  cfg_path = checkpoint_local_dir / "preplay.config"
  if not sf_path.exists() or not cfg_path.exists():
    raise FileNotFoundError(
      f"Missing checkpoint files in {checkpoint_local_dir}. "
      f"scp preplay.safetensors and preplay.config from the server first."
    )

  import craftax_simulation_configs
  from craftax_web_env import CraftaxMultiGoalSymbolicWebEnvNoAutoReset
  from multitask_preplay import make_craftax_multigoal_agent
  from jaxneurorl.wrappers import TimestepWrapper

  # LogWrapper is defined inside craftax_multigoal_trainer (not the craftax package).
  from craftax_multigoal_trainer import LogWrapper

  with open(cfg_path, "rb") as f:
    config = pickle.load(f)

  # Env setup mirrors craftax_multigoal_trainer.run_single (lines 447-455),
  # minus vmapping — we want a single-env rollout interface.
  use_landmark_features = config.get("ALG") == "usfa" or config.get(
    "LANDMARK_FEATURES", False
  )
  static_env_params = (
    CraftaxMultiGoalSymbolicWebEnvNoAutoReset.default_static_params().replace(
      landmark_features=use_landmark_features
    )
  )
  base_env = CraftaxMultiGoalSymbolicWebEnvNoAutoReset(
    static_env_params=static_env_params
  )
  env = TimestepWrapper(LogWrapper(base_env), autoreset=False)

  default_params = craftax_simulation_configs.default_params
  env_params_train = default_params.replace(
    task_configs=craftax_simulation_configs.TRAIN_CONFIGS
  )
  env_params_test = default_params.replace(
    task_configs=craftax_simulation_configs.TEST_CONFIGS, training=False
  )

  # Build an example timestep so the agent can .init()
  example_rng = jax.random.PRNGKey(0)
  example_ts = env.reset(example_rng, env_params_train)
  # Add batch dim. Use jnp.asarray first because some TimeStep leaves are python
  # scalars (bool/float) that don't support `x[None]`.
  example_ts_batched = jax.tree_util.tree_map(
    lambda x: jnp.asarray(x)[None], example_ts
  )

  agent, _init_params, _reset_fn = make_craftax_multigoal_agent(
    config=config,
    env=env,
    env_params=env_params_train,
    example_timestep=example_ts_batched,
    rng=example_rng,
    model_env=base_env,
    model_env_params=env_params_train,
  )

  flat = load_file(str(sf_path))
  params = unflatten_dict(flat, sep=",")

  sig = _signature(sf_path, config)
  return CraftaxBundle(
    env=env,
    env_params_train=env_params_train,
    env_params_test=env_params_test,
    agent=agent,
    params=params,
    config=config,
    checkpoint_path=sf_path,
    signature=sig,
  )


# ────────── Rollouts ──────────


@dataclass
class Rollout:
  positions: np.ndarray  # (T, 2) world coords (y, x) of player_position each step
  actions: np.ndarray  # (T,)
  rewards: np.ndarray  # (T,)
  states: list  # env states per step (for re-rendering any frame)
  timesteps: list  # timesteps per step
  rnn_states: list  # rnn carries per step (for preplay snapshot)
  reached_goal: bool
  length: int

  def __len__(self) -> int:
    return self.length


# Pickle stability: pin Rollout's apparent module to a stable name so caches
# round-trip between `python plots/make_promo_video.py` (this file is __main__)
# and `import make_promo_video as m` from a notebook. Without this, pickles
# written by the script are tagged `__main__.Rollout` and unpickling from a
# notebook fails with "Can't get attribute 'Rollout' on <module '__main__'>".
# Paired with the sys.modules alias in the __main__ guard at the bottom of
# the file (which makes `make_promo_video` resolve to the running __main__
# when run as a script) and the Unpickler rewrite in `rollout_cached` (for
# legacy caches that still say __main__).
Rollout.__module__ = "make_promo_video"


def _agent_greedy_action(agent, params, rnn_state, timestep, rng):
  ts_batch = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], timestep)
  preds, new_rnn_state = agent.apply(params, rnn_state, ts_batch, rng)
  q_vals = preds.q_vals[0]
  action = jnp.argmax(q_vals)
  return action, new_rnn_state


def _player_pos(timestep) -> np.ndarray:
  s = _find_state_with(timestep.state, "player_position")
  return np.asarray(s.player_position)


def _find_state_with(state, field: str):
  """Walk .env_state until we hit an object that has `field`."""
  s = state
  while not hasattr(s, field):
    if not hasattr(s, "env_state"):
      raise AttributeError(f"no nested state has field '{field}'")
    s = s.env_state
  return s


def _replace_nested(state, field: str, value):
  """Rebuild wrapper chain with `field` replaced on the inner struct that owns it."""
  if hasattr(state, field):
    return state.replace(**{field: value})
  if not hasattr(state, "env_state"):
    raise AttributeError(f"no nested state has field '{field}'")
  return state.replace(env_state=_replace_nested(state.env_state, field, value))


def _place_goal_in_timestep(timestep, goal_onehot):
  """Set state.current_goal (walks wrapper chain) and observation.task_w."""
  from craftax_web_env import IDX_to_Achievement

  achievement_goal = jax.lax.dynamic_index_in_dim(
    IDX_to_Achievement, jnp.asarray(goal_onehot).argmax(-1), keepdims=False
  )
  new_state = _replace_nested(timestep.state, "current_goal", achievement_goal)
  new_obs = timestep.observation.replace(task_w=goal_onehot)
  return timestep.replace(state=new_state, observation=new_obs)


def rollout_greedy(
  bundle: CraftaxBundle,
  env_params,
  *,
  rng: jax.random.PRNGKey,
  max_steps: int = 80,
  initial_timestep=None,
  initial_rnn_state=None,
  task_w_override=None,
) -> Rollout:
  """Greedy rollout; explicit RNG, no hidden state.

  task_w_override: if given, swap the goal (both state.current_goal and
  observation.task_w) at every step. Used for the Scene 2 preplay rollout.
  """
  rng, reset_rng, init_rng = jax.random.split(rng, 3)

  if initial_timestep is None:
    timestep = bundle.env.reset(reset_rng, env_params)
  else:
    timestep = initial_timestep

  if task_w_override is not None:
    timestep = _place_goal_in_timestep(timestep, task_w_override)

  if initial_rnn_state is None:
    rnn_state = bundle.agent.apply(
      bundle.params,
      batch_dims=(1,),
      rng=init_rng,
      method=bundle.agent.initialize_carry,
    )
  else:
    rnn_state = initial_rnn_state

  positions, actions, rewards, states, timesteps, rnn_states = [], [], [], [], [], []
  reached = False
  for t in range(max_steps):
    positions.append(_player_pos(timestep))
    timesteps.append(timestep)
    states.append(timestep.state)
    rnn_states.append(rnn_state)

    rng, step_rng, act_rng = jax.random.split(rng, 3)
    action, new_rnn_state = _agent_greedy_action(
      bundle.agent, bundle.params, rnn_state, timestep, act_rng
    )
    actions.append(int(action))

    if bool(timestep.last()):
      reached = bool(timestep.reward > 0.5)
      break

    timestep = bundle.env.step(step_rng, timestep, action, env_params)
    if task_w_override is not None:
      timestep = _place_goal_in_timestep(timestep, task_w_override)
    rewards.append(float(timestep.reward))
    rnn_state = new_rnn_state

  return Rollout(
    positions=np.asarray(positions),
    actions=np.asarray(actions),
    rewards=np.asarray(rewards),
    states=states,
    timesteps=timesteps,
    rnn_states=rnn_states,
    reached_goal=reached,
    length=len(positions),
  )


def _cache_path(scene: str, bundle: CraftaxBundle, seed: int) -> Path:
  CACHE_DIR.mkdir(parents=True, exist_ok=True)
  return CACHE_DIR / f"{scene}__{bundle.signature}__seed{seed}.pkl"


class _RewriteMainUnpickler(pickle.Unpickler):
  """Loads legacy caches that were written when this file ran as __main__.

  Old pickles tagged Rollout as `__main__.Rollout`; the canonical name is now
  `make_promo_video.Rollout` (set on the class above). Rewrite on load so
  caches survive the rename.
  """

  def find_class(self, module, name):
    if module == "__main__" and name == "Rollout":
      module = "make_promo_video"
    return super().find_class(module, name)


def rollout_cached(
  scene: str,
  bundle: CraftaxBundle,
  seed: int,
  produce: Callable[[], Rollout],
  *,
  force: bool = False,
) -> Rollout:
  """Run produce() only if no cached rollout exists for (scene, signature, seed)."""
  path = _cache_path(scene, bundle, seed)
  if path.exists() and not force:
    with open(path, "rb") as f:
      r = _RewriteMainUnpickler(f).load()
    print(f"  [cache] loaded {scene} from {path.name}")
    return r
  r = produce()
  with open(path, "wb") as f:
    pickle.dump(r, f)
  print(f"  [cache] saved {scene} -> {path.name}")
  return r


def make_env_params(
  bundle: CraftaxBundle,
  *,
  world_seed: int,
  goal: str,
  source: str = "train",
  pin_start_yx: tuple[int, int] | None = None,
  force_eval_curriculum: bool = False,
):
  """Build env_params pinned to a specific world + active goal.

  The key trick: CraftaxMultiGoalSymbolicWebEnvNoAutoReset.reset samples a
  random row from params.task_configs and OVERWRITES world_seeds / placed_goals
  / current_goal / etc with that row's values (craftax_web_env.py:740-789).
  So to actually pin (world, goal), we must filter task_configs down to rows
  that already have the requested (world_seed, goal_object) before reset.

  Reset *also* picks the spawn cell from the chosen row's `start_positions`
  array: index 0 is "farthest" (the original eval/train start) and indices 1+
  are waypoints along the cached optimal path. Under `params.training=True`,
  reset flips a coin between farthest and a uniform waypoint; under
  `params.training=False` it always uses farthest. To get a deterministic spawn
  for the promo (so train and test panels share a start), pass
  `pin_start_yx=(y, x)` — this filters task_configs further to the row whose
  `start_positions[0] == (y, x)` — and `force_eval_curriculum=True` to set
  `params.training=False` regardless of source.

  Args:
    world_seed: the Craftax world seed (3/15/20/95 for the configs shipped here).
    goal: 'diamond' / 'sapphire' / 'ruby' — the currently active achievement.
    source: 'train' or 'test' — picks which task_configs pool to filter.
    pin_start_yx: if given, additionally filter to rows whose `start_positions[0]`
      equals this (y, x). Combined with `force_eval_curriculum=True`, this makes
      the spawn cell deterministic.
    force_eval_curriculum: if True, set `training=False` so reset always spawns
      at `start_positions[0]` (no waypoint jitter).
  """
  from craftax.craftax.constants import Achievement

  achievement_value = {
    "diamond": Achievement.COLLECT_DIAMOND.value,
    "sapphire": Achievement.COLLECT_SAPPHIRE.value,
    "ruby": Achievement.COLLECT_RUBY.value,
  }[goal]
  base = bundle.env_params_train if source == "train" else bundle.env_params_test

  ws = np.asarray(base.task_configs.world_seed)
  go = np.asarray(base.task_configs.goal_object)
  mask = (ws == world_seed) & (go == achievement_value)
  idx = np.where(mask)[0]
  if len(idx) == 0:
    available = sorted(set(zip(ws.tolist(), go.tolist())))
    raise ValueError(
      f"no task_config matches world_seed={world_seed} goal={goal} "
      f"(achievement_value={achievement_value}) in {source}_configs; "
      f"available (world_seed, goal_object) = {available}"
    )

  if pin_start_yx is not None:
    starts0 = np.asarray(base.task_configs.start_positions)[idx, 0]  # (n, 2)
    py, px = int(pin_start_yx[0]), int(pin_start_yx[1])
    sub_mask = (starts0[:, 0] == py) & (starts0[:, 1] == px)
    sub_idx = np.where(sub_mask)[0]
    if len(sub_idx) == 0:
      available_starts = sorted({(int(s[0]), int(s[1])) for s in starts0})
      raise ValueError(
        f"no task_config matches world_seed={world_seed} goal={goal} "
        f"start={pin_start_yx} in {source}_configs; available "
        f"start_positions[0] = {available_starts}"
      )
    idx = idx[sub_idx]

  filtered = jax.tree_util.tree_map(lambda x: x[idx], base.task_configs)

  if force_eval_curriculum:
    base = base.replace(training=False)

  return base.replace(
    task_configs=filtered,
    world_seeds=(int(world_seed),),
    current_goal=jnp.asarray(achievement_value, dtype=jnp.int32),
  )


def _eval_start_yx(bundle: CraftaxBundle, world_seed: int) -> tuple[int, int]:
  """Canonical eval start for `world_seed`, read from TEST_CONFIGS.

  TEST_CONFIGS only contains rows built from `start_eval_positions`
  (craftax_simulation_configs.py:141-149), so the first matching row's
  `start_positions[0]` is the eval start by construction.
  """
  ws = np.asarray(bundle.env_params_test.task_configs.world_seed)
  matches = np.where(ws == world_seed)[0]
  if len(matches) == 0:
    raise ValueError(
      f"no test task_config has world_seed={world_seed} "
      f"(available: {sorted(set(ws.tolist()))})"
    )
  starts = np.asarray(bundle.env_params_test.task_configs.start_positions)
  y, x = starts[int(matches[0]), 0]
  return int(y), int(x)


def run_three_rollouts(
  bundle: CraftaxBundle,
  *,
  world_seed: int = 3,
  on_task: str = "sapphire",
  off_task: str = "diamond",
  seed: int = 42,
  force: bool = False,
) -> tuple[Rollout, Rollout, Rollout]:
  """Convenience: chain train → preplay → test with one world + two goals.

  For world=3: on-task = sapphire (trained) or ruby (trained); off-task = diamond (held out).
  The preplay rollout is an ordinary greedy rollout with task_w swapped to the
  off-task one-hot, starting from the train-rollout midpoint.
  """
  train = rollout_train(
    bundle, world_seed=world_seed, goal=on_task, seed=seed, force=force
  )
  preplay = rollout_preplay(
    bundle, train, world_seed=world_seed, off_task_goal=off_task, seed=seed, force=force
  )
  test = rollout_test(
    bundle, world_seed=world_seed, goal=off_task, seed=seed, force=force
  )
  return train, preplay, test


def rollout_train(
  bundle: CraftaxBundle,
  *,
  world_seed: int = 3,
  goal: str = "sapphire",
  seed: int = 42,
  force: bool = False,
  max_steps: int = 80,
) -> Rollout:
  """On-task rollout. `goal` is the active achievement (e.g. 'sapphire').

  Spawn is pinned to the world's eval start (read from TEST_CONFIGS) so panels 1
  and 3 share an origin on the minimap. Without this, TRAIN_CONFIGS contains
  one row per (eval_start ∪ train_starts) and reset would land on a random one.
  """
  start_yx = _eval_start_yx(bundle, world_seed)
  params = make_env_params(
    bundle,
    world_seed=world_seed,
    goal=goal,
    source="train",
    pin_start_yx=start_yx,
    force_eval_curriculum=True,
  )
  scene = f"train_w{world_seed}_{goal}_s{start_yx[0]}_{start_yx[1]}"
  return rollout_cached(
    scene,
    bundle,
    seed,
    lambda: rollout_greedy(
      bundle,
      params,
      rng=jax.random.PRNGKey(seed),
      max_steps=max_steps,
    ),
    force=force,
  )


def rollout_preplay(
  bundle: CraftaxBundle,
  train: Rollout,
  *,
  world_seed: int = 3,
  off_task_goal: str = "diamond",
  seed: int = 42,
  force: bool = False,
  max_steps: int = 60,
  distance_from_goal: int = 6,
  start_index: int | None = None,
) -> Rollout:
  """Off-task rollout starting from near the end of the train rollout.

  By default, snapshots the train rollout `distance_from_goal` steps before it
  reached its on-task goal, then swaps task_w to the off-task one-hot and rolls
  greedy. This puts the agent in a state where it has already perceived the
  region around both goals (train and off-task goals are near each other by
  construction) but hasn't yet committed to the on-task goal.

  Pass `start_index` to override and snapshot at a specific step instead.
  """
  if start_index is not None:
    start = start_index
  else:
    # train.length includes all recorded steps; if train ended by reaching the
    # goal, the last few states are "near goal". We back off by distance_from_goal.
    start = max(0, len(train) - 1 - distance_from_goal)

  onehot = jnp.asarray(np.eye(3)[ACHIEVEMENT_IDX[off_task_goal]], dtype=jnp.float32)
  # The off-task goal (e.g. diamond for world=3) lives in TEST_CONFIGS by design,
  # not TRAIN_CONFIGS. Pull params from test_configs so make_env_params can find
  # a matching row.
  params = make_env_params(
    bundle, world_seed=world_seed, goal=off_task_goal, source="test"
  )
  scene = f"preplay_w{world_seed}_{off_task_goal}_start{start}"
  return rollout_cached(
    scene,
    bundle,
    seed,
    lambda: rollout_greedy(
      bundle,
      params,
      rng=jax.random.PRNGKey(seed + 1),
      max_steps=max_steps,
      initial_timestep=train.timesteps[start],
      initial_rnn_state=train.rnn_states[start],
      task_w_override=onehot,
    ),
    force=force,
  )


def rollout_test(
  bundle: CraftaxBundle,
  *,
  world_seed: int = 3,
  goal: str = "diamond",
  seed: int = 42,
  force: bool = False,
  max_steps: int = 60,
) -> Rollout:
  """Test-time rollout: fresh reset with `goal` as the active task.

  Pin is explicit (matches `rollout_train`) even though TEST_CONFIGS already
  has one row per world at the eval start — keeps the contract obvious and
  insulates us from any future config additions.
  """
  start_yx = _eval_start_yx(bundle, world_seed)
  params = make_env_params(
    bundle,
    world_seed=world_seed,
    goal=goal,
    source="test",
    pin_start_yx=start_yx,
  )
  scene = f"test_w{world_seed}_{goal}_s{start_yx[0]}_{start_yx[1]}"
  return rollout_cached(
    scene,
    bundle,
    seed,
    lambda: rollout_greedy(
      bundle,
      params,
      rng=jax.random.PRNGKey(seed),
      max_steps=max_steps,
    ),
    force=force,
  )


# ────────── Rendering ──────────


def _full_map_rgb(
  state, block_pixel_size: int = 10, show_agent: bool = True
) -> np.ndarray:
  """Full-map render. Expensive on first call (JAX trace).

  Unwrap wrapper state because `render_craftax_pixels` (full map) expects the raw
  craftax EnvState, not the LogEnvState/TimestepWrapper wrapping we carry.
  """
  from craftax_fullmap_renderer import render_craftax_pixels as render_full

  raw = _unwrap_state(state)
  img = render_full(raw, block_pixel_size=block_pixel_size, show_agent=show_agent)
  return np.asarray(img).astype(np.uint8)


def _partial_view_rgb(state) -> np.ndarray:
  """Egocentric pixel view (what the agent sees). Matches reference usage in
  experiments/craftax/craftax_utils.py:create_episode_reaction_times_video."""
  from craftax.craftax.renderer import render_craftax_pixels as partial_render
  from craftax.craftax.constants import BLOCK_PIXEL_SIZE_IMG

  raw = _unwrap_state(state)
  img = partial_render(raw, block_pixel_size=BLOCK_PIXEL_SIZE_IMG)
  return np.asarray(img).astype(np.uint8)


def _unwrap_state(state):
  """Peel TimestepWrapper+LogWrapper layers down to the raw craftax EnvState."""
  s = state
  while hasattr(s, "env_state"):
    s = s.env_state
  return s


def _goal_cell_xy(env_params, goal_idx: int) -> tuple[int, int]:
  """World-coords (y, x) of the placed goal for a given achievement-one-hot index.

  env_params.placed_goals: [N] block-type values. env_params.goal_locations:
  [N, 2] positions. We find the entry in placed_goals whose block type matches
  the BlockType value for the requested achievement (diamond/sapphire/ruby).
  """
  from craftax.craftax.constants import BlockType

  _achievement_to_blocktype = {
    0: BlockType.DIAMOND.value,
    1: BlockType.SAPPHIRE.value,
    2: BlockType.RUBY.value,
  }
  target = _achievement_to_blocktype[goal_idx]
  placed = np.asarray(env_params.placed_goals)
  locs = np.asarray(env_params.goal_locations)
  matches = np.where(placed == target)[0]
  if len(matches) == 0:
    raise KeyError(f"no placed goal with block type {target} (idx={goal_idx})")
  i = int(matches[0])
  return int(locs[i, 0]), int(locs[i, 1])


def _goal_marker(ax, cell_yx, pixel_per_cell, color, *, marker_size=10):
  """Filled circle goal marker, black-edged. Same visual language as
  craftax_multi_envs.py — readable on Craftax pixel art and on cream/peach bg.

  marker_size=10 is the size that lands proportional-to-map at our 4.8" / 100dpi
  minimap (the multi_envs default of 15 was for a 7"/300dpi figure — same
  point size would render ~50% larger relative to the map here)."""
  cy, cx = cell_yx
  cx_px = (cx + 0.5) * pixel_per_cell
  cy_px = (cy + 0.5) * pixel_per_cell
  ax.plot(
    cx_px,
    cy_px,
    "o",
    color=color,
    markersize=marker_size,
    markeredgecolor="black",
    markeredgewidth=1.5,
    zorder=20,
  )


def _start_marker(ax, cell_yx, pixel_per_cell, color="white", *, marker_size=18):
  """Filled star for start position, black-edged. Matches
  craftax_utils.place_start_marker styling, scaled for our minimap size."""
  cy, cx = cell_yx
  cx_px = (cx + 0.5) * pixel_per_cell
  cy_px = (cy + 0.5) * pixel_per_cell
  ax.plot(
    cx_px,
    cy_px,
    "*",
    color=color,
    markersize=marker_size,
    markeredgecolor="black",
    markeredgewidth=1.5,
    zorder=20,
  )


_MAP_LABEL_CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")


def _draw_map_label(
  ax,
  cell_yx,
  pixel_per_cell,
  text,
  color,
  *,
  corner: str = "top-left",
  fontsize: int = 22,
  margin_frac: float = 0.04,
) -> None:
  """Paint a colored pill label at a corner of the map with a leader arrow
  pointing to the goal marker. Corner placement keeps the label clear of the
  Craftax pixel art so the text stays readable on top of the map.

  Args:
    corner: one of "top-left", "top-right", "bottom-left", "bottom-right".
    margin_frac: inset from the corner as a fraction of min(map_w, map_h).
  """
  if corner not in _MAP_LABEL_CORNERS:
    raise ValueError(f"unknown corner {corner!r}; choose from {_MAP_LABEL_CORNERS}")
  cy, cx = cell_yx
  marker_x = (cx + 0.5) * pixel_per_cell
  marker_y = (cy + 0.5) * pixel_per_cell

  # ax.imshow set ylim=(H, 0); use raw limits and resolve the inset from the
  # absolute extent so this works regardless of axis orientation.
  xmin, xmax = ax.get_xlim()
  y0, y1 = ax.get_ylim()
  ytop, ybot = (y1, y0) if y0 > y1 else (y0, y1)  # ytop = small y in image coords
  W = xmax - xmin
  H = ybot - ytop
  margin = margin_frac * min(W, H)

  if corner == "top-left":
    label_x, label_y, ha, va = xmin + margin, ytop + margin, "left", "top"
  elif corner == "top-right":
    label_x, label_y, ha, va = xmax - margin, ytop + margin, "right", "top"
  elif corner == "bottom-left":
    label_x, label_y, ha, va = xmin + margin, ybot - margin, "left", "bottom"
  else:  # bottom-right
    label_x, label_y, ha, va = xmax - margin, ybot - margin, "right", "bottom"

  ax.annotate(
    text,
    xy=(marker_x, marker_y),
    xycoords="data",
    xytext=(label_x, label_y),
    textcoords="data",
    ha=ha,
    va=va,
    color=color,
    fontproperties=_font("semibold", fontsize),
    bbox=dict(
      facecolor=DESIGN["bg"],
      edgecolor=color,
      linewidth=2.0,
      boxstyle="round,pad=0.5",
    ),
    arrowprops=dict(
      arrowstyle="-|>",
      color=color,
      lw=2.2,
      shrinkA=6,
      shrinkB=10,
    ),
    zorder=22,
  )


def _cluster_arrow(ax, target_yx, pixel_per_cell, *, length_cells=10, color="black"):
  """Draw an arrow pointing to (target_yx) from a fixed up-and-to-the-side offset.

  Used on the establishing-shot panel to draw the eye toward the goal cluster.
  Target is in maze coords; the arrow originates `length_cells` north-west of
  the target and curves slightly so it reads as a label, not a path.
  """
  ty, tx = target_yx
  start_y = (ty - length_cells + 0.5) * pixel_per_cell
  start_x = (tx - length_cells * 0.6 + 0.5) * pixel_per_cell
  end_y = (ty + 0.5) * pixel_per_cell - 0.6 * pixel_per_cell
  end_x = (tx + 0.5) * pixel_per_cell - 0.6 * pixel_per_cell
  ax.add_patch(
    FancyArrowPatch(
      (start_x, start_y),
      (end_x, end_y),
      arrowstyle="-|>",
      mutation_scale=22,
      linewidth=3.5,
      color=color,
      shrinkA=0,
      shrinkB=4,
      zorder=22,
      connectionstyle="arc3,rad=-0.18",
    )
  )


def _solid_trail(ax, positions, pixel_per_cell, color, lw=2.5):
  if len(positions) < 2:
    return
  xs = [(p[1] + 0.5) * pixel_per_cell for p in positions]
  ys = [(p[0] + 0.5) * pixel_per_cell for p in positions]
  ax.plot(xs, ys, color=color, linewidth=lw, solid_capstyle="round", zorder=15)


def _dashed_trail(ax, positions, pixel_per_cell, color, lw=2.5):
  if len(positions) < 2:
    return
  xs = [(p[1] + 0.5) * pixel_per_cell for p in positions]
  ys = [(p[0] + 0.5) * pixel_per_cell for p in positions]
  ax.plot(
    xs,
    ys,
    color=color,
    linewidth=lw,
    linestyle=(0, (4, 3)),
    solid_capstyle="round",
    zorder=15,
  )
  # arrowhead on the last segment
  ax.add_patch(
    FancyArrowPatch(
      (xs[-2], ys[-2]),
      (xs[-1], ys[-1]),
      arrowstyle="-|>",
      mutation_scale=14,
      color=color,
      linewidth=0,
      zorder=16,
    )
  )


def _fig_to_rgb(fig, target_size: tuple[int, int] | None = None) -> np.ndarray:
  """Render fig to uint8 RGB. On retina/HiDPI matplotlib renders at 2×, so
  downsample to target_size when supplied. target_size is (width, height)."""
  fig.canvas.draw()
  rgba = np.asarray(fig.canvas.buffer_rgba())
  rgb = rgba[..., :3].copy()
  plt.close(fig)
  if target_size is not None:
    tw, th = target_size
    if rgb.shape[1] != tw or rgb.shape[0] != th:
      from PIL import Image

      rgb = np.asarray(Image.fromarray(rgb).resize((tw, th), Image.LANCZOS))
  return rgb


def _make_title_card(
  text: str | None = None,
  *,
  eyebrow: str | None = None,
  subtitle: str | None = None,
  size=(1280, 720),
  fps=20,
  duration_s=1.4,
  fade_s=0.25,
) -> list[np.ndarray]:
  """Three-tier title card with eased fade in/out.

  Defaults pull from TITLE_EYEBROW / TITLE_TEXT / TITLE_SUBTITLE so callers can
  pass nothing for the standard launch card. `text` (positional) overrides the
  title; pass eyebrow=None / subtitle=None to suppress those tiers (e.g. when
  using this card mid-video as a beat).
  """
  return title_card_frames(
    eyebrow=TITLE_EYEBROW if eyebrow is None and text is None else eyebrow,
    title=text if text is not None else TITLE_TEXT,
    subtitle=TITLE_SUBTITLE if subtitle is None and text is None else subtitle,
    size=size,
    fps=fps,
    duration_s=duration_s,
    fade_s=fade_s,
    fonts=_fonts(),
  )


_CAPTION_BOLD_RE = None  # lazy compile


def _parse_caption_segments(text: str) -> list[tuple[str, bool]]:
  """Split a single caption line on **bold** markers.

  Returns a list of (segment_text, is_bold) tuples. Empty segments are dropped.
  Multi-line captions should be split on '\\n' before calling this."""
  global _CAPTION_BOLD_RE
  if _CAPTION_BOLD_RE is None:
    import re

    _CAPTION_BOLD_RE = re.compile(r"(\*\*[^*]+\*\*)")
  out: list[tuple[str, bool]] = []
  for part in _CAPTION_BOLD_RE.split(text):
    if not part:
      continue
    if part.startswith("**") and part.endswith("**"):
      out.append((part[2:-2], True))
    else:
      out.append((part, False))
  return out


def _render_caption(
  fig,
  ax_cap,
  caption: str,
  *,
  fonts: dict,
  color: str,
  fontsize: int,
) -> None:
  """Render a caption with optional **bold** spans, centered on ax_cap.

  Fast path (no bold markers anywhere): a single ax_cap.text call, identical
  to the legacy behavior including matplotlib's native '\\n' handling.

  Bold path: each line is split into (text, is_bold) segments, segment widths
  are measured via the renderer, then segments are placed left-to-right
  starting at x = 0.5 - total_width/2 so the line stays centered. Multi-line
  captions are stacked vertically around y=0.42 (the legacy single-line center)."""
  if "**" not in caption:
    ax_cap.text(
      0.5,
      0.42,
      caption,
      ha="center",
      va="center",
      color=color,
      fontsize=fontsize,
      fontproperties=fonts["semibold"],
    )
    return

  lines = [_parse_caption_segments(line) for line in caption.split("\n")]
  lines = [seg for seg in lines if seg]  # drop empty lines
  if not lines:
    return

  # Force renderer creation so get_window_extent has something to measure with.
  fig.canvas.draw()
  renderer = fig.canvas.get_renderer()

  # ax_cap occupies the bottom 18% of the figure; line spacing is in axes
  # fraction (0..1 of ax_cap height). 0.32 ≈ one 28pt line at 100 dpi.
  line_h = 0.32
  n = len(lines)
  y_top = 0.42 + (n - 1) * line_h / 2

  ax_pos = ax_cap.get_position()
  fig_w_in, _ = fig.get_size_inches()
  ax_w_px = ax_pos.width * fig_w_in * fig.dpi

  for i, segments in enumerate(lines):
    widths = []
    temp = []
    for seg, is_bold in segments:
      fp = fonts["bold" if is_bold else "semibold"].copy()
      fp.set_size(fontsize)
      t = ax_cap.text(
        0,
        -10,
        seg,
        fontproperties=fp,
        color=color,
        transform=ax_cap.transAxes,
      )
      widths.append(t.get_window_extent(renderer=renderer).width / ax_w_px)
      temp.append(t)
    for t in temp:
      t.remove()

    x = 0.5 - sum(widths) / 2
    y = y_top - i * line_h
    for (seg, is_bold), w in zip(segments, widths):
      fp = fonts["bold" if is_bold else "semibold"].copy()
      fp.set_size(fontsize)
      ax_cap.text(
        x,
        y,
        seg,
        ha="left",
        va="center",
        color=color,
        fontproperties=fp,
        transform=ax_cap.transAxes,
      )
      x += w


def _compose_panel(
  main_rgb: np.ndarray,
  caption: str | None,
  *,
  size=(1280, 720),
  minimap_rgb: np.ndarray | None = None,
  ken_burns: tuple[int, int] | None = None,
  ken_burns_pan: tuple[float, float] = (0.0, 0.0),
  main_title: str | None = None,
  reserve_header: bool = False,
  partial_view_labels: list[tuple[str, str, str]] | None = None,
  goal_arrow_targets: list[tuple[float, float] | None] | None = None,
) -> np.ndarray:
  """Stitch the main panel + optional minimap + lower-third caption.

  Three layout modes are auto-selected:
    - Dual (main_rgb + minimap_rgb): left/right split, both axes get titles
      (default "What the agent sees" / "Full map", overridable via main_title).
    - Single-centered with header (main_rgb only, main_title set OR
      reserve_header=True): the main panel is enlarged and centered, with
      space reserved for a title above. Used for panel 2 (preplay) — "What
      the agent imagines during replay" — where the minimap is hidden.
    - Single-left no header (main_rgb only, no title and no reservation):
      legacy panel 0 establishing shot.

  Args:
    ken_burns: optional (t, total) for slow zoom on the main panel.
    ken_burns_pan: (dy_frac, dx_frac) pan during the zoom.
    main_title: header above main panel. None defaults to "What the agent sees"
      when minimap is present; None means "no title" otherwise.
    reserve_header: when True, use the with-header geometry even if no title is
      drawn. Use this for panel-2 phases where the title hasn't appeared yet
      but the image must stay in the same axes box as the rest of the panel,
      so it doesn't visibly jump when the title fades in.
    partial_view_labels: list of (text, color_hex, position) overlay badges to
      paint on top of the main panel — e.g. [("on-task goal", "#56B4E9",
      "top-left"), ("counterfactual goal", "#FD0000", "top-right")]. Position
      ∈ {"top-left", "top-right"}.
  """
  fonts = _fonts()
  fig = plt.figure(
    figsize=(size[0] / 100, size[1] / 100), dpi=100, facecolor=DESIGN["bg"]
  )

  has_minimap = minimap_rgb is not None
  has_header = main_title is not None or has_minimap or reserve_header
  if main_title is None and has_minimap:
    main_title = "What the agent sees"

  # Layout constants — same for every panel regardless of caption length, so
  # all four beats sit at the same vertical position. The bottom (MAIN_Y) and
  # height (MAIN_H) are shared by every branch (dual / single-centered /
  # single no-header), and the minimap re-uses the same y/h, so the main
  # panel and minimap have identical top *and* bottom edges in the dual
  # layout. Sized so a 2-line caption clears the image bottom; 1-line
  # captions get extra breathing room.
  MAIN_Y = 0.18
  MAIN_H = 0.66  # all layouts; top stays at 0.84
  MINI_Y = MAIN_Y  # match main bottom
  MINI_H = MAIN_H
  CAP_FONTSIZE = 28

  if has_minimap:
    # Dual layout (panels 1, 3): left half + right half, titles align at top.
    ax_main = fig.add_axes([0.02, MAIN_Y, 0.54, MAIN_H])
  elif has_header:
    # Single-centered with header (panel 2): wider, centered.
    ax_main = fig.add_axes([0.20, MAIN_Y, 0.60, MAIN_H])
  else:
    # Single-centered, no header (panel 0 establishing shot). Same height as
    # every other layout so the rendered image (1:1, fits to box height)
    # matches the size in the rest of the storyboard — no jump between beats.
    ax_main = fig.add_axes([0.23, MAIN_Y, 0.54, MAIN_H])

  if ken_burns is not None:
    t_kb, total_kb = ken_burns
    ken_burns_imshow(
      ax_main,
      main_rgb,
      t_kb,
      total_kb,
      zoom_start=1.0,
      zoom_end=1.045,
      pan_fraction=ken_burns_pan,
    )
  else:
    ax_main.imshow(main_rgb)
    ax_main.axis("off")

  if has_header and main_title:
    ax_main.set_title(
      main_title,
      color=DESIGN["muted"],
      fontproperties=_font("regular", 28),
      pad=10,
    )

  if partial_view_labels:
    _draw_partial_view_labels(
      ax_main, partial_view_labels, arrow_targets=goal_arrow_targets
    )

  if has_minimap:
    ax_mini = fig.add_axes([0.59, MINI_Y, 0.39, MINI_H])  # top stays at y=0.84
    ax_mini.imshow(minimap_rgb)
    ax_mini.axis("off")
    ax_mini.set_title(
      "Full map",
      color=DESIGN["muted"],
      fontproperties=_font("regular", 28),
      pad=10,
    )

  # Lower-third gradient + caption text. Skip both when caption is None so
  # the bottom of the frame is fully clear (used for panel 2's image-only
  # opening beat — let the partial view land on its own before the caption
  # narrates it).
  if caption is not None:
    add_caption_gradient(fig, size=size, height_frac=0.22)
    ax_cap = fig.add_axes([0.0, 0.0, 1.0, 0.18], zorder=10)
    ax_cap.set_facecolor("none")
    ax_cap.axis("off")
    _render_caption(
      fig,
      ax_cap,
      caption,
      fonts=fonts,
      color=DESIGN["fg"],
      fontsize=CAP_FONTSIZE,
    )

  frame = _fig_to_rgb(fig, target_size=size)
  if frame.shape[0] % 2 or frame.shape[1] % 2:
    frame = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]
  return frame


def render_minimap_with_trails(
  state,
  bundle: CraftaxBundle,
  *,
  train_positions_so_far: np.ndarray | None = None,
  preplay_positions_so_far: np.ndarray | None = None,
  test_positions_so_far: np.ndarray | None = None,
  show_goals: tuple[str, ...] = ("sapphire", "diamond"),
  goal_labels: dict[str, str | tuple[str, str]] | None = None,
  scene: str = "train",  # controls which env_params for goal cells
  trail_subdivisions: int = 3,
  show_start_marker: bool = False,
  start_marker_yx: tuple[int, int] | None = None,
  cluster_arrow: bool = False,
) -> np.ndarray:
  """Render a full map with goal markers + (optionally) any of the three trails overlaid.

  Args:
    show_goals: tuple of goal names to draw markers for. Empty tuple draws no
      markers; default ("sapphire", "diamond") matches the legacy behavior.
      Used by panel 0's phased reveal to add goals one at a time.
    goal_labels: optional `{goal_name: text}` or `{goal_name: (text, corner)}`.
      Paints a colored pill label at the named map corner with a leader arrow
      to the marker. Corner ∈ {"top-left", "top-right", "bottom-left",
      "bottom-right"}. Default corner is "top-left" for sapphire and
      "top-right" for diamond when only `text` is given.
    show_start_marker: draw a white star marker. Without `start_marker_yx`,
      uses the agent's current player position (panel-0 behavior — the agent
      hasn't moved yet so this IS the start). For per-step animations pass
      `start_marker_yx` so the star pins to the rollout's origin instead of
      following the agent.
    start_marker_yx: explicit (y, x) cell for the start marker. Overrides
      the player-position derivation. Implies `show_start_marker=True`.
    cluster_arrow: draw an arrow pointing to the goal cluster centroid. Used
      on the establishing-shot panel only.
  """
  # block_pixel_size must be a key in TEXTURES (see experiments/craftax/
  # craftax_fullmap_renderer.py). 6 is NOT — use 10 (BLOCK_PIXEL_SIZE_IMG).
  img = _full_map_rgb(state, block_pixel_size=10, show_agent=False)
  H, W = img.shape[:2]
  raw_state = _unwrap_state(state)
  map_h, map_w = raw_state.map.shape[1], raw_state.map.shape[2]
  ppc = W / map_w  # pixels per cell (assume square cells)

  fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
  ax = fig.add_axes([0, 0, 1, 1])
  ax.imshow(img)
  ax.set_xlim(0, W)
  ax.set_ylim(H, 0)
  ax.axis("off")

  # Resolve theme colors at call time so use_theme() / theme= switches take
  # effect for this render.
  c_train = train_color()
  c_test = test_color()

  goal_cells: list[tuple[int, int]] = []
  goal_labels = goal_labels or {}
  goal_color = {"sapphire": c_train, "diamond": c_test}
  default_corner = {"sapphire": "top-left", "diamond": "top-right"}
  env_params = bundle.env_params_train if scene == "train" else bundle.env_params_test
  for name in ("sapphire", "diamond"):
    if name not in show_goals:
      continue
    try:
      cell = _goal_cell_xy(env_params, ACHIEVEMENT_IDX[name])
    except Exception:
      continue
    color = goal_color[name]
    _goal_marker(ax, cell, ppc, color)
    goal_cells.append(cell)
    spec = goal_labels.get(name)
    if spec is not None:
      if isinstance(spec, tuple):
        text, corner = spec
      else:
        text, corner = spec, default_corner[name]
      _draw_map_label(ax, cell, ppc, text, color, corner=corner)

  # Explicit start_marker_yx wins; otherwise legacy show_start_marker uses the
  # current player position (only correct for panel 0 where the agent hasn't
  # moved). For per-step animations callers must pass start_marker_yx so the
  # star pins to the rollout's origin instead of following the agent.
  if start_marker_yx is not None:
    try:
      _start_marker(ax, (int(start_marker_yx[0]), int(start_marker_yx[1])), ppc)
    except Exception:
      pass
  elif show_start_marker:
    try:
      start_yx = (int(raw_state.player_position[0]), int(raw_state.player_position[1]))
      _start_marker(ax, start_yx, ppc)
    except Exception:
      pass

  if cluster_arrow and goal_cells:
    cy = sum(c[0] for c in goal_cells) / len(goal_cells)
    cx = sum(c[1] for c in goal_cells) / len(goal_cells)
    _cluster_arrow(ax, (cy, cx), ppc, length_cells=10, color="black")

  # Subdivide trails so growth is smooth at 20 fps instead of cell-by-cell.
  if train_positions_so_far is not None and len(train_positions_so_far) > 1:
    _solid_trail(
      ax,
      interp_trail(train_positions_so_far, n_subdivisions=trail_subdivisions),
      ppc,
      c_train,
      lw=3.0,
    )
  if preplay_positions_so_far is not None and len(preplay_positions_so_far) > 1:
    _dashed_trail(
      ax,
      interp_trail(preplay_positions_so_far, n_subdivisions=trail_subdivisions),
      ppc,
      c_test,
      lw=3.0,
    )
  if test_positions_so_far is not None and len(test_positions_so_far) > 1:
    _solid_trail(
      ax,
      interp_trail(test_positions_so_far, n_subdivisions=trail_subdivisions),
      ppc,
      c_test,
      lw=3.0,
    )

  # Pin output to the unscaled (W, H) the upstream caller expects.
  return _fig_to_rgb(fig, target_size=(W, H))


# ────────── Per-rollout preview (debugging / sanity check) ──────────


def preview_rollout(
  rollout: Rollout,
  out_path: Path,
  *,
  title: str = "",
  fps: int = 4,
) -> Path:
  """Simple 2-panel MP4 for eyeballing a single rollout. Shape matches
  experiments/craftax/craftax_utils.py::create_episode_reaction_times_video:
  left = full map with the agent's path baked in as arrows, right = per-step
  partial (egocentric) view. Use this to sanity-check each rollout before
  committing to the composite storyboard.
  """
  from craftax.craftax.constants import Action

  out_path = Path(out_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  # Bake the path once onto the full map — much cheaper than re-rendering.
  state0 = rollout.states[0]
  full_map = _full_map_rgb(state0, block_pixel_size=10, show_agent=False)
  raw_state = _unwrap_state(state0)
  map_h, map_w = raw_state.map.shape[1], raw_state.map.shape[2]

  # Each per-step partial view
  partial_frames = [_partial_view_rgb(s) for s in rollout.states]

  frames = []
  for t in range(len(rollout)):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="black")
    ax_map = fig.add_axes([0.02, 0.12, 0.46, 0.82])
    ax_view = fig.add_axes([0.52, 0.12, 0.46, 0.82])
    ax_cap = fig.add_axes([0, 0, 1, 0.1])

    # Left: full map + path-so-far (solid trail via our helper, which draws
    # via matplotlib — easier than calling place_arrows_on_image for partial paths).
    H, W = full_map.shape[:2]
    ppc = W / map_w
    ax_map.imshow(full_map)
    ax_map.set_xlim(0, W)
    ax_map.set_ylim(H, 0)
    ax_map.axis("off")
    _solid_trail(ax_map, rollout.positions[: t + 1], ppc, "red", lw=2.5)
    # Mark current position
    cy, cx = rollout.positions[t]
    ax_map.plot(
      [(cx + 0.5) * ppc],
      [(cy + 0.5) * ppc],
      "o",
      color="red",
      markersize=10,
      markeredgecolor="white",
      zorder=20,
    )
    ax_map.set_title(f"Full map (not given to agent)", color="white", fontsize=14)

    # Right: per-step partial view
    ax_view.imshow(partial_frames[t])
    ax_view.axis("off")
    ax_view.set_title("Agent's partial view", color="white", fontsize=14)

    # Caption
    ax_cap.set_facecolor("black")
    ax_cap.axis("off")
    action_name = (
      Action(int(rollout.actions[t])).name if t < len(rollout.actions) else "-"
    )
    cap = f"{title}   step {t}   action={action_name}   reached={rollout.reached_goal}"
    ax_cap.text(
      0.5,
      0.5,
      cap,
      ha="center",
      va="center",
      color="white",
      fontsize=20,
      family="monospace",
    )

    frames.append(_fig_to_rgb(fig))

  encode(frames, out_path, fps=fps)
  return out_path


# ────────── Storyboard ──────────


def render_panel1_only(
  bundle: CraftaxBundle,
  train: Rollout,
  *,
  fps: int = 20,
  size=(720, 720),
) -> list[np.ndarray]:
  """Prototype gate: just the on-task approach + mini-map trail.

  Returns one frame per step of the train rollout, for a Panel-1-only preview.
  """
  frames = []
  for t in range(len(train)):
    partial = _partial_view_rgb(train.states[t])
    # upscale to square target size
    partial = _upsample(partial, size)
    minimap = render_minimap_with_trails(
      train.states[t],
      bundle,
      train_positions_so_far=train.positions[: t + 1],
      scene="train",
    )
    frame = _compose_panel(
      partial, "On-task experience.", size=(1280, 720), minimap_rgb=minimap
    )
    frames.append(frame)
  return frames


def _upsample(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
  """Nearest-neighbor upscale to target (W, H). Matplotlib-friendly."""
  from PIL import Image

  im = Image.fromarray(img)
  im = im.resize(size, Image.NEAREST)
  return np.asarray(im)


# ────────── Per-block builders ──────────
#
# Each builder returns a raw `list[np.ndarray]` for one scene of the storyboard
# at `size` resolution. They DO NOT apply polish (vignette/grain) — that step
# is centralized in `polish_frames` and applied at composition (or by
# `preview_block`) so a block can be inspected as it will appear in the final
# cut without being double-grained.
#
# Tuning knobs (caption text, hold duration, Ken Burns pan) live in the
# top-of-file `STORYBOARD` dict — edit there, re-run the corresponding cell.


def _size(size: tuple[int, int] | None) -> tuple[int, int]:
  return size if size is not None else STORYBOARD["size"]


def _maybe_set_theme(theme: str | None) -> None:
  """If a theme name is given, swap DESIGN before render. Helpers all read
  DESIGN at call time so subsequent imshow/text/trail calls pick up the swap."""
  if theme is not None:
    use_theme(theme)


def _repeat_to_fps(
  frames: list[np.ndarray], source_fps: int, target_fps: int | None
) -> list[np.ndarray]:
  """Repeat each frame so a list built at `source_fps` plays the same wall-clock
  duration when encoded at `target_fps`. No-op when target is None or ≤ source.

  Used to mix panels with different perceived speeds in one MP4 — the container
  can only have one framerate, so we hit it via repetition."""
  if target_fps is None or target_fps <= source_fps:
    return frames
  repeat = max(1, int(round(target_fps / source_fps)))
  return [f for f in frames for _ in range(repeat)]


def _panel_fps(panel_name: str, fps_arg: int | None) -> int:
  """Resolve the perceived fps for a panel: explicit kwarg > STORYBOARD entry
  > storyboard encode_fps default (20)."""
  if fps_arg is not None:
    return fps_arg
  return STORYBOARD[panel_name].get("fps", STORYBOARD.get("encode_fps", 20))


def build_title_block(
  *,
  opening: bool = True,
  fps: int | None = None,
  encode_fps: int | None = None,
  size: tuple[int, int] | None = None,
  theme: str | None = None,
  apply_polish: bool = True,
) -> list[np.ndarray]:
  """Eased fade-in/hold/fade-out three-tier title card.

  Args:
    fps: perceived frame rate (unique frames per second). None → STORYBOARD
      "title"/"title_reprise" entry's "fps" key, default 20.
    encode_fps: when given and > fps, returned frames are repeated so a
      downstream encoder at `encode_fps` plays at the same wall-clock speed.
      Used by `render_full_storyboard` to mix per-panel fps under one MP4.
  """
  _maybe_set_theme(theme)
  name = "title" if opening else "title_reprise"
  cfg = STORYBOARD[name]
  fps = _panel_fps(name, fps)
  out = _make_title_card(
    size=_size(size),
    fps=fps,
    duration_s=cfg["duration_s"],
    fade_s=cfg["fade_s"],
  )
  out = _repeat_to_fps(out, fps, encode_fps)
  return polish_frames(out) if apply_polish else out


def build_panel0_block(
  bundle: CraftaxBundle,
  train: Rollout,
  *,
  fps: int | None = None,
  encode_fps: int | None = None,
  size: tuple[int, int] | None = None,
  theme: str | None = None,
  apply_polish: bool = True,
) -> list[np.ndarray]:
  """Establishing shot: full-map render with goals revealed one at a time.

  Drives the reveal off `STORYBOARD["panel0"]["phases"]`. Each phase has its
  own (goals, labels, duration_s); the Ken-Burns clock runs continuously
  across all phases so the slow zoom keeps pulling in even as goals appear.
  """
  _maybe_set_theme(theme)
  size = _size(size)
  cfg = STORYBOARD["panel0"]
  fps = _panel_fps("panel0", fps)
  panel0_state = train.states[0]
  phases = cfg["phases"]
  phase_n = [max(int(p["duration_s"] * fps), 1) for p in phases]

  out: list[np.ndarray] = []
  for phase, n in zip(phases, phase_n):
    # Re-render the minimap per phase (cheap: <1s) so we can swap goal markers
    # and adjacent labels in/out without touching the main panel.
    minimap = render_minimap_with_trails(
      panel0_state,
      bundle,
      scene="train",
      trail_subdivisions=1,
      show_start_marker=True,
      cluster_arrow=False,  # labeled balls do the eye-drawing now
      show_goals=tuple(phase["goals"]),
      goal_labels=phase["labels"],
    )
    main = _upsample(minimap, (720, 720))
    # Static panel — no Ken Burns. Compose once per phase, then repeat the
    # frame n times (cheap memory copies vs n full renders).
    frame = _compose_panel(
      main,
      cfg["caption"],
      size=size,
      minimap_rgb=None,
      ken_burns=None,
    )
    out.extend([frame] * n)

  out = _repeat_to_fps(out, fps, encode_fps)
  return polish_frames(out) if apply_polish else out


def _panel1_labels(names: tuple[str, ...]) -> list[tuple[str, str, str]]:
  """Resolve panel-1 label-name keys into (text, color_hex, position) specs.

  Colors are resolved at call time so use_theme() takes effect."""
  spec_map = {
    "current": ("current goal", train_color(), "top-left"),
    "accessible": ("accessible goal", test_color(), "top-right"),
  }
  return [spec_map[n] for n in names]


def _panel2_labels(names: tuple[str, ...]) -> list[tuple[str, str, str]]:
  spec_map = {
    "on_task": ("on-task goal", train_color(), "top-left"),
    "counterfactual": ("counterfactual goal", test_color(), "top-right"),
  }
  return [spec_map[n] for n in names]


def build_panel1_block(
  bundle: CraftaxBundle,
  train: Rollout,
  *,
  fps: int | None = None,
  encode_fps: int | None = None,
  size: tuple[int, int] | None = None,
  theme: str | None = None,
  apply_polish: bool = True,
  include_final_pickup: bool = False,
) -> list[np.ndarray]:
  """On-task train rollout, with sequential pre-intro reveals + end labels.

  Phases:
    0a. Partial-only — agent's partial view alone, no minimap. Lets the
        first-person view land before the orientation aid arrives.
    0b. Minimap (no star) — full-map appears next to the partial view, but
        without the start marker yet. "Here's the world."
    0c. Minimap + star — start marker appears at the rollout origin.
        "And here's where the agent starts."
    1.  Animation — per-step partial view + minimap with star + growing blue
        trail. NO labels during animation.
    2.  Sequential end labels — for each entry in cfg["label_phases"], hold
        the last animation frame with the named labels visible (default:
        "current goal" alone for 2 s, then both labels for 2 s).

  fps defaults to STORYBOARD["panel1"]["fps"]; encode_fps (when set higher
  than fps) repeats frames so the playback speed matches at the higher rate.
  """
  _maybe_set_theme(theme)
  size = _size(size)
  cfg = STORYBOARD["panel1"]
  fps = _panel_fps("panel1", fps)
  n_steps = len(train) if include_final_pickup else max(len(train) - 1, 1)
  n_panel1 = max(int(cfg["duration_s"] * fps), n_steps)
  indices = _resample_indices(n_steps, n_panel1)

  # Pin start cell from the rollout's first position so the star stays at the
  # origin across every minimap render (panels 1/3/animation re-render per
  # step, so the player moves but the star must not).
  start_yx = (int(train.positions[0][0]), int(train.positions[0][1]))

  out: list[np.ndarray] = []

  # Frame 0 partial view is reused across the pre-intro beats.
  state0 = train.states[0]
  partial0 = _upsample(_partial_view_rgb(state0), (720, 720))

  # Phase 0b — minimap appears, no star, no trail yet.
  minimap_no_star_n = int(cfg.get("minimap_no_star_s", 0.0) * fps)
  if minimap_no_star_n > 0:
    minimap_no_star = render_minimap_with_trails(state0, bundle, scene="train")
    frame = _compose_panel(
      partial0,
      cfg["caption"],
      size=size,
      minimap_rgb=minimap_no_star,
    )
    out.extend([frame] * minimap_no_star_n)

  # Phase 0c — star appears at the rollout origin.
  minimap_star_n = int(cfg.get("minimap_star_s", 0.0) * fps)
  if minimap_star_n > 0:
    minimap_with_star = render_minimap_with_trails(
      state0, bundle, scene="train", start_marker_yx=start_yx
    )
    frame = _compose_panel(
      partial0,
      cfg["caption"],
      size=size,
      minimap_rgb=minimap_with_star,
    )
    out.extend([frame] * minimap_star_n)

  # Phase 1 — animation. Star pinned to start_yx so it doesn't follow agent.
  for t in indices:
    state = train.states[t]
    partial = _upsample(_partial_view_rgb(state), (720, 720))
    minimap = render_minimap_with_trails(
      state,
      bundle,
      train_positions_so_far=train.positions[: t + 1],
      scene="train",
      start_marker_yx=start_yx,
    )
    out.append(
      _compose_panel(
        partial,
        cfg["caption"],
        size=size,
        minimap_rgb=minimap,
      )
    )

  # Sequential end labels: hold the last animation frame, recomposing it once
  # per label phase with the named labels visible. The minimap is identical
  # across label phases so we render it once and reuse.
  label_phases = cfg.get("label_phases", [])
  if label_phases and indices:
    last_t = indices[-1]
    last_state = train.states[last_t]
    last_partial = _upsample(_partial_view_rgb(last_state), (720, 720))
    last_minimap = render_minimap_with_trails(
      last_state,
      bundle,
      train_positions_so_far=train.positions[: last_t + 1],
      scene="train",
      start_marker_yx=start_yx,
    )
    arrow_targets = _resolve_goal_arrow_targets(
      bundle, last_state, bundle.env_params_train
    )
    for phase in label_phases:
      labels = _panel1_labels(tuple(phase["labels"]))
      # Arrows only on labels that have an arrow target — if "accessible" is
      # absent, drop its (right-side) target so we don't draw a dangling arrow.
      n_labels = len(labels)
      arrows_for_phase = arrow_targets[:n_labels]
      frame = _compose_panel(
        last_partial,
        cfg["caption"],
        size=size,
        minimap_rgb=last_minimap,
        partial_view_labels=labels,
        goal_arrow_targets=arrows_for_phase,
      )
      out.extend([frame] * max(int(phase["duration_s"] * fps), 1))

  out = _repeat_to_fps(out, fps, encode_fps)
  return polish_frames(out) if apply_polish else out


def build_panel2_block(
  bundle: CraftaxBundle,
  train: Rollout,  # noqa: ARG001 — kept for API parity
  preplay: Rollout,
  *,
  fps: int | None = None,
  encode_fps: int | None = None,
  size: tuple[int, int] | None = None,
  theme: str | None = None,
  apply_polish: bool = True,
  include_final_pickup: bool | None = None,
) -> list[np.ndarray]:
  """Off-task preplay rollout: agent's first-person view of the imagined
  trajectory toward the counterfactual goal. Minimap is hidden — the title
  ("What the agent imagines during replay") cues mental simulation.

  Phases (all on frame 0 except the animation):
    0a. Image-only — partial view alone, no caption / no title / no labels,
        for cfg["image_only_s"] seconds. Lets the image land.
    0b. Caption-only — caption + title appear, still no labels, for
        cfg["caption_only_s"] seconds.
    1.  Label intro — sequential label reveals from cfg["intro_phases"]
        (default: "on-task goal" alone for 2 s, then both labels for 2 s).
    2.  Animation — partial-view per step with NO labels. cfg["drop_pickup"]
        removes the final pickup step so the rollout ends on the agent next
        to the goal, not standing on it.
    3.  Final hold — repeat the last animation frame for cfg["final_pause_s"]
        seconds with both labels back.

  fps defaults to STORYBOARD["panel2"]["fps"] (intentionally low — 2 fps —
  for a deliberate / contemplative beat). encode_fps repeats frames if set.
  """
  _maybe_set_theme(theme)
  size = _size(size)
  cfg = STORYBOARD["panel2"]
  fps = _panel_fps("panel2", fps)
  drop_pickup = (
    cfg.get("drop_pickup", False)
    if include_final_pickup is None
    else not include_final_pickup
  )
  n_steps = max(len(preplay) - 1, 1) if drop_pickup else len(preplay)
  n_panel2 = max(int(cfg["duration_s"] * fps), n_steps)
  indices = _resample_indices(n_steps, n_panel2)

  env_params = bundle.env_params_test
  title = cfg.get("title", 'What the agent "imagines" during replay')

  out: list[np.ndarray] = []

  # Frame 0 is reused across phases 0a, 0b, and the label intros below.
  state0 = preplay.states[0]
  partial0 = _upsample(_partial_view_rgb(state0), (720, 720))
  arrow_targets0 = _resolve_goal_arrow_targets(bundle, state0, env_params)

  # Phase 0a — image-only beat. caption=None suppresses the bottom gradient
  # and main_title=None suppresses the title text. reserve_header=True keeps
  # the with-header axes geometry so the partial view doesn't jump when the
  # title appears in phase 0b.
  image_only_n = int(cfg.get("image_only_s", 0.0) * fps)
  if image_only_n > 0:
    frame = _compose_panel(
      partial0,
      None,
      size=size,
      minimap_rgb=None,
      main_title=None,
      reserve_header=True,
    )
    out.extend([frame] * image_only_n)

  # Phase 0b — caption + title appear, still no labels.
  caption_only_n = int(cfg.get("caption_only_s", 0.0) * fps)
  if caption_only_n > 0:
    frame = _compose_panel(
      partial0,
      cfg["caption"],
      size=size,
      minimap_rgb=None,
      main_title=title,
    )
    out.extend([frame] * caption_only_n)

  # Phase 1 — intro labels on frame 0 (no rollout movement).
  intro_phases = cfg.get("intro_phases", [])
  if intro_phases:
    for phase in intro_phases:
      labels = _panel2_labels(tuple(phase["labels"]))
      arrows_for_phase = arrow_targets0[: len(labels)]
      frame = _compose_panel(
        partial0,
        cfg["caption"],
        size=size,
        minimap_rgb=None,
        main_title=title,
        partial_view_labels=labels,
        goal_arrow_targets=arrows_for_phase,
      )
      out.extend([frame] * max(int(phase["duration_s"] * fps), 1))

  # Phase 2 — animation with no labels.
  for t in indices:
    state = preplay.states[t]
    partial = _upsample(_partial_view_rgb(state), (720, 720))
    out.append(
      _compose_panel(
        partial,
        cfg["caption"],
        size=size,
        minimap_rgb=None,
        main_title=title,
      )
    )

  # Phase 3 — final hold with both labels back, on the last animation frame.
  final_pause_s = cfg.get("final_pause_s", 0.0)
  pause_n = int(final_pause_s * fps)
  if pause_n > 0 and indices:
    last_t = indices[-1]
    last_state = preplay.states[last_t]
    last_partial = _upsample(_partial_view_rgb(last_state), (720, 720))
    arrow_targets = _resolve_goal_arrow_targets(bundle, last_state, env_params)
    labels = _panel2_labels(("on_task", "counterfactual"))
    final = _compose_panel(
      last_partial,
      cfg["caption"],
      size=size,
      minimap_rgb=None,
      main_title=title,
      partial_view_labels=labels,
      goal_arrow_targets=arrow_targets,
    )
    out.extend([final] * pause_n)

  out = _repeat_to_fps(out, fps, encode_fps)
  return polish_frames(out) if apply_polish else out


def build_panel3_block(
  bundle: CraftaxBundle,
  train: Rollout,
  test: Rollout,
  *,
  fps: int | None = None,
  encode_fps: int | None = None,
  size: tuple[int, int] | None = None,
  theme: str | None = None,
  apply_polish: bool = True,
  include_final_pickup: bool | None = None,
) -> list[np.ndarray]:
  """Test rollout: per-step partial view + minimap with full pale train history
  + growing amber test trail.

  Two phases:
    1. Animation — per-step partial-view + minimap with growing amber trail.
       cfg["drop_pickup"] removes the final pickup step.
    2. Final hold — repeat the last animation frame for cfg["final_pause_s"]
       seconds with a single "previous preplay goal" label (test color)
       pointing at the diamond.
  """
  _maybe_set_theme(theme)
  size = _size(size)
  cfg = STORYBOARD["panel3"]
  fps = _panel_fps("panel3", fps)
  drop_pickup = (
    cfg.get("drop_pickup", False)
    if include_final_pickup is None
    else not include_final_pickup
  )
  n_steps = max(len(test) - 1, 1) if drop_pickup else len(test)
  n_panel3 = max(int(cfg["duration_s"] * fps), n_steps)
  indices = _resample_indices(n_steps, n_panel3)

  # Pin the start star to the test rollout's origin so it stays put across
  # per-step renders. (Train and test both spawn at _eval_start_yx, so this
  # also matches the panel-1 star location.)
  start_yx = (int(test.positions[0][0]), int(test.positions[0][1]))

  out: list[np.ndarray] = []
  for t in indices:
    partial = _upsample(_partial_view_rgb(test.states[t]), (720, 720))
    minimap = render_minimap_with_trails(
      test.states[t],
      bundle,
      train_positions_so_far=train.positions,  # full history (pale)
      test_positions_so_far=test.positions[: t + 1],
      scene="test",
      start_marker_yx=start_yx,
    )
    out.append(_compose_panel(partial, cfg["caption"], size=size, minimap_rgb=minimap))

  # Final hold — recompose the last frame with the "previous preplay goal"
  # label and an arrow pointing at the diamond (resolved against test params).
  final_pause_s = cfg.get("final_pause_s", 0.0)
  pause_n = int(final_pause_s * fps)
  if pause_n > 0 and indices:
    last_t = indices[-1]
    last_state = test.states[last_t]
    last_partial = _upsample(_partial_view_rgb(last_state), (720, 720))
    last_minimap = render_minimap_with_trails(
      last_state,
      bundle,
      train_positions_so_far=train.positions,
      test_positions_so_far=test.positions[: last_t + 1],
      scene="test",
      start_marker_yx=start_yx,
    )
    # _resolve_goal_arrow_targets returns [sapphire, diamond]; we want the
    # diamond (preplay goal) for the single label.
    arrow_targets = _resolve_goal_arrow_targets(
      bundle, last_state, bundle.env_params_test
    )
    diamond_target = arrow_targets[1] if len(arrow_targets) > 1 else None
    final = _compose_panel(
      last_partial,
      cfg["caption"],
      size=size,
      minimap_rgb=last_minimap,
      partial_view_labels=[
        ("previous preplay goal", test_color(), "top-left"),
      ],
      goal_arrow_targets=[diamond_target],
    )
    out.extend([final] * pause_n)

  out = _repeat_to_fps(out, fps, encode_fps)
  return polish_frames(out) if apply_polish else out


# ────────── Polish + preview helpers ──────────


def _is_light_theme() -> bool:
  """Rec. 709 luminance test on the active background. Used to skip the
  edge-darkening vignette when the theme bg is light (would just gray the
  cream/peach/ivory)."""
  h = DESIGN["bg"].lstrip("#")
  r, g, b = (
    int(h[0:2], 16) / 255.0,
    int(h[2:4], 16) / 255.0,
    int(h[4:6], 16) / 255.0,
  )
  return (0.2126 * r + 0.7152 * g + 0.0722 * b) > 0.5


def polish_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
  """Theme-aware film polish. Always applies subtle grain (paper-texture feel
  on cream, film grain on dark). On dark themes also applies a corner vignette
  to focus the eye; on light themes the vignette would just dirty the cream so
  we skip it."""
  if _is_light_theme():
    return [add_grain(f, intensity=0.018) for f in frames]
  return [add_vignette(add_grain(f, intensity=0.022), strength=0.12) for f in frames]


def preview_block(
  frames: list[np.ndarray],
  name: str,
  *,
  fps: int = 20,
  out_dir: Path | None = None,
) -> Path:
  """Encode one scene block to `<out_dir>/_block_{name}.mp4` for notebook review.

  Builders apply polish by default — pass `build_*_block(..., apply_polish=False)`
  if you need to encode raw frames here.
  """
  out = (Path(out_dir) if out_dir is not None else OUTPUT_DIR) / f"_block_{name}.mp4"
  encode(frames, out, fps=fps)
  return out


def preview_bridge(
  blocks: list[list[np.ndarray] | tuple[list[np.ndarray], int]],
  name: str,
  *,
  encode_fps: int | None = None,
  fps: int | None = None,  # deprecated alias for encode_fps
  crossfade_n: int | None = None,
  out_dir: Path | None = None,
) -> Path:
  """Bridge a sequence of blocks via eased crossfades and write the result.

  Each block can be either:
    - `list[np.ndarray]` — assumed to already be at `encode_fps`
    - `(frames, source_fps)` tuple — frames are repeated to `encode_fps` first

  Use the tuple form to stitch panels that were built at different perceived
  fps (e.g. panel1 at 5 fps + panel2 at 2 fps) without playback drift.

  Args:
    encode_fps: container framerate of the bridged MP4. None →
      STORYBOARD["encode_fps"] (default 20). `fps=` is accepted as a
      deprecated alias for backward compat.
  """
  if encode_fps is None:
    encode_fps = fps if fps is not None else STORYBOARD.get("encode_fps", 20)
  if crossfade_n is None:
    crossfade_n = STORYBOARD["crossfade_n"]

  repeated: list[list[np.ndarray]] = []
  for b in blocks:
    if isinstance(b, tuple):
      block_frames, src_fps = b
      repeated.append(_repeat_to_fps(block_frames, src_fps, encode_fps))
    else:
      repeated.append(b)

  bridged = bridge_scenes(*repeated, crossfade_n=crossfade_n)
  out = (Path(out_dir) if out_dir is not None else OUTPUT_DIR) / f"_bridge_{name}.mp4"
  encode(bridged, out, fps=encode_fps)
  return out


# ────────── Peek helpers (single-frame previews, ~instant) ──────────
#
# Each peek_* returns one polished frame as a uint8 RGB ndarray. They render a
# single representative composition (held / mid-rollout) so you can iterate on
# typography, palette, captions, and pan settings in <1s — no full block render
# and no MP4 encode. In the notebook:
#
#     img = m.peek_panel0(bundle, train, theme="cream")
#     plt.imshow(img); plt.axis("off"); plt.show()


def peek_title(
  *,
  opening: bool = True,
  theme: str | None = None,
  size: tuple[int, int] | None = None,
) -> np.ndarray:
  """Single fully-faded-in title-card frame."""
  _maybe_set_theme(theme)
  # duration=0.05s @ 20fps → 1 frame; fade=0 → frame is at α=1 (held).
  frames = title_card_frames(
    eyebrow=TITLE_EYEBROW,
    title=TITLE_TEXT,
    subtitle=TITLE_SUBTITLE,
    size=_size(size),
    fps=20,
    duration_s=0.05,
    fade_s=0.0,
    fonts=_fonts(),
  )
  return polish_frames([frames[0]])[0]


def peek_panel0(
  bundle: CraftaxBundle,
  train: Rollout,
  *,
  theme: str | None = None,
  size: tuple[int, int] | None = None,
) -> np.ndarray:
  """Static establishing-shot frame at the densest reveal phase (both goal
  balls + corner labels visible). No Ken Burns — matches build output."""
  _maybe_set_theme(theme)
  cfg = STORYBOARD["panel0"]
  last_phase = cfg["phases"][-1]
  panel0_mini = render_minimap_with_trails(
    train.states[0],
    bundle,
    scene="train",
    trail_subdivisions=1,
    show_start_marker=True,
    cluster_arrow=False,
    show_goals=tuple(last_phase["goals"]),
    goal_labels=last_phase["labels"],
  )
  panel0_main = _upsample(panel0_mini, (720, 720))
  frame = _compose_panel(
    panel0_main,
    cfg["caption"],
    size=_size(size),
    minimap_rgb=None,
    ken_burns=None,
  )
  return polish_frames([frame])[0]


def peek_panel1(
  bundle: CraftaxBundle,
  train: Rollout,
  *,
  theme: str | None = None,
  size: tuple[int, int] | None = None,
) -> np.ndarray:
  """Held end-state frame of the on-task beat with both labels visible
  (last label_phase). Matches the final hold the viewer dwells on."""
  _maybe_set_theme(theme)
  cfg = STORYBOARD["panel1"]
  # Held state = the last animation step (drop the pickup, same as the
  # builder when include_final_pickup=False / default).
  last_t = max(len(train) - 2, 0)
  state = train.states[last_t]
  partial = _upsample(_partial_view_rgb(state), (720, 720))
  start_yx = (int(train.positions[0][0]), int(train.positions[0][1]))
  minimap = render_minimap_with_trails(
    state,
    bundle,
    train_positions_so_far=train.positions[: last_t + 1],
    scene="train",
    start_marker_yx=start_yx,
  )
  arrow_targets = _resolve_goal_arrow_targets(bundle, state, bundle.env_params_train)
  labels = _panel1_labels(("current", "accessible"))
  frame = _compose_panel(
    partial,
    cfg["caption"],
    size=_size(size),
    minimap_rgb=minimap,
    partial_view_labels=labels,
    goal_arrow_targets=arrow_targets[: len(labels)],
  )
  return polish_frames([frame])[0]


def peek_panel2(
  bundle: CraftaxBundle,
  train: Rollout,  # noqa: ARG001 — kept for API parity with build_panel2_block
  preplay: Rollout,
  *,
  theme: str | None = None,
  size: tuple[int, int] | None = None,
) -> np.ndarray:
  """Held end-state frame of the preplay beat (both labels, last animation
  frame after dropping pickup). Matches the final 3 s hold."""
  _maybe_set_theme(theme)
  cfg = STORYBOARD["panel2"]
  drop_pickup = cfg.get("drop_pickup", False)
  last_t = max(len(preplay) - 2, 0) if drop_pickup else max(len(preplay) - 1, 0)
  state = preplay.states[last_t]
  partial = _upsample(_partial_view_rgb(state), (720, 720))
  arrow_targets = _resolve_goal_arrow_targets(bundle, state, bundle.env_params_test)
  title = cfg.get("title", 'What the agent "imagines" during replay')
  labels = _panel2_labels(("on_task", "counterfactual"))
  frame = _compose_panel(
    partial,
    cfg["caption"],
    size=_size(size),
    minimap_rgb=None,
    main_title=title,
    partial_view_labels=labels,
    goal_arrow_targets=arrow_targets[: len(labels)],
  )
  return polish_frames([frame])[0]


def peek_panel3(
  bundle: CraftaxBundle,
  train: Rollout,
  test: Rollout,
  *,
  theme: str | None = None,
  size: tuple[int, int] | None = None,
) -> np.ndarray:
  """Held end-state frame of the test beat with the 'previous preplay goal'
  label visible. Matches the final 3 s hold after dropping pickup."""
  _maybe_set_theme(theme)
  cfg = STORYBOARD["panel3"]
  drop_pickup = cfg.get("drop_pickup", False)
  last_t = max(len(test) - 2, 0) if drop_pickup else max(len(test) - 1, 0)
  state = test.states[last_t]
  partial = _upsample(_partial_view_rgb(state), (720, 720))
  start_yx = (int(test.positions[0][0]), int(test.positions[0][1]))
  minimap = render_minimap_with_trails(
    state,
    bundle,
    train_positions_so_far=train.positions,
    test_positions_so_far=test.positions[: last_t + 1],
    scene="test",
    start_marker_yx=start_yx,
  )
  arrow_targets = _resolve_goal_arrow_targets(bundle, state, bundle.env_params_test)
  diamond_target = arrow_targets[1] if len(arrow_targets) > 1 else None
  frame = _compose_panel(
    partial,
    cfg["caption"],
    size=_size(size),
    minimap_rgb=minimap,
    partial_view_labels=[("previous preplay goal", test_color(), "top-left")],
    goal_arrow_targets=[diamond_target],
  )
  return polish_frames([frame])[0]


# ────────── Storyboard composition (thin orchestrator) ──────────


def render_full_storyboard(
  bundle: CraftaxBundle,
  train: Rollout,
  preplay: Rollout,
  test: Rollout,
  *,
  encode_fps: int | None = None,
  fps_overrides: dict[str, int] | None = None,
  size: tuple[int, int] | None = None,
  crossfade_n: int | None = None,
  theme: str | None = None,
  apply_polish: bool = True,
) -> list[np.ndarray]:
  """Compose the full storyboard from the per-block builders.

  Each panel builds at its own STORYBOARD[name]["fps"] (its perceived
  playback speed). Every panel's frames are repeated to `encode_fps` so the
  joined MP4 plays at one container framerate while preserving each panel's
  perceived speed.

  Args:
    encode_fps: final MP4 container framerate. None → STORYBOARD["encode_fps"]
      (default 20).
    fps_overrides: per-panel fps override map, e.g. `{"panel1": 10}`. Useful
      for one-off tweaks without editing STORYBOARD.

  Wall-clock timing follows the per-panel `duration_s` + `pause_s` config in
  STORYBOARD. Crossfades (0.5s default) sit between panels.

  Pass `theme="dark"|"cream"|"peach"|"ivory"` to swap the active palette.
  """
  _maybe_set_theme(theme)
  size = _size(size)
  if encode_fps is None:
    encode_fps = STORYBOARD.get("encode_fps", 20)
  if crossfade_n is None:
    crossfade_n = STORYBOARD["crossfade_n"]
  fps_overrides = fps_overrides or {}

  def panel_fps(name: str) -> int:
    return fps_overrides.get(name, STORYBOARD[name].get("fps", encode_fps))

  # Each builder builds at its perceived fps and repeats internally to
  # encode_fps. apply_polish=False so polish runs exactly once on the
  # bridged result (otherwise we'd double-grain).
  blocks = [
    build_title_block(
      opening=True,
      fps=panel_fps("title"),
      encode_fps=encode_fps,
      size=size,
      apply_polish=False,
    ),
    build_panel0_block(
      bundle,
      train,
      fps=panel_fps("panel0"),
      encode_fps=encode_fps,
      size=size,
      apply_polish=False,
    ),
    build_panel1_block(
      bundle,
      train,
      fps=panel_fps("panel1"),
      encode_fps=encode_fps,
      size=size,
      apply_polish=False,
    ),
    build_panel2_block(
      bundle,
      train,
      preplay,
      fps=panel_fps("panel2"),
      encode_fps=encode_fps,
      size=size,
      apply_polish=False,
    ),
    build_panel3_block(
      bundle,
      train,
      test,
      fps=panel_fps("panel3"),
      encode_fps=encode_fps,
      size=size,
      apply_polish=False,
    ),
    build_title_block(
      opening=False,
      fps=panel_fps("title_reprise"),
      encode_fps=encode_fps,
      size=size,
      apply_polish=False,
    ),
  ]
  frames = bridge_scenes(*blocks, crossfade_n=crossfade_n)
  return polish_frames(frames) if apply_polish else frames


def _resample_indices(n_source: int, n_target: int) -> list[int]:
  """Pick n_target indices from range(n_source) so the scene fits the time budget."""
  if n_source == 0:
    return []
  if n_target >= n_source:
    # hold the last frame to pad out
    return list(range(n_source)) + [n_source - 1] * (n_target - n_source)
  return [int(round(i * (n_source - 1) / (n_target - 1))) for i in range(n_target)]


# ────────── Encoding ──────────


_MIN_CONTAINER_FPS = 12  # browsers / Jupyter Video widget play sub-10-fps
# MP4s unreliably (often upsample to 30 fps and play faster than the file's
# declared duration). For low-fps previews we encode at a higher container
# fps and repeat each input frame, preserving the wall-clock duration.


def encode(
  frames: list[np.ndarray],
  out_mp4: Path,
  *,
  out_gif: Path | None = None,
  fps: int = 20,
) -> None:
  """Encode `frames` to MP4 (and optionally GIF). At low `fps` the requested
  framerate is achieved by repeating frames at a browser-safe container fps —
  ffprobe will show the higher container rate but each unique image still
  visible for 1/fps seconds in any HTML5 player."""
  out_mp4 = Path(out_mp4)
  out_mp4.parent.mkdir(parents=True, exist_ok=True)

  # libx264 + yuv420p for Twitter compat; enforce even dims.
  frames = [_ensure_even(f) for f in frames]

  if fps < _MIN_CONTAINER_FPS:
    repeat = max(1, int(round(_MIN_CONTAINER_FPS / fps)))
    frames_for_mp4 = [f for f in frames for _ in range(repeat)]
    encode_fps = fps * repeat
  else:
    frames_for_mp4 = frames
    encode_fps = fps

  imageio.mimwrite(
    str(out_mp4),
    frames_for_mp4,
    fps=encode_fps,
    codec="libx264",
    macro_block_size=1,
    pixelformat="yuv420p",
    ffmpeg_params=ENCODE_FFMPEG_PARAMS,
  )
  print(
    f"  wrote {out_mp4}  ({len(frames)} unique frames @ effective {fps} fps,"
    f" container {encode_fps} fps)"
  )

  if out_gif is not None:
    out_gif = Path(out_gif)
    # GIF: halve fps + every other source frame to keep filesize down. GIFs
    # don't have the same low-fps player issues as MP4 so we don't repeat.
    gif_fps = max(1, fps // 2)
    imageio.mimwrite(str(out_gif), frames[::2], fps=gif_fps)
    print(f"  wrote {out_gif}")


def _ensure_even(frame: np.ndarray) -> np.ndarray:
  h, w = frame.shape[:2]
  nh = h - (h % 2)
  nw = w - (w % 2)
  if nh == h and nw == w:
    return frame
  return frame[:nh, :nw]


# ────────── Script entry: render every theme ──────────

# Defaults match the notebook driver (make_promo_video.ipynb) so running this
# file as a script reproduces the same scene + rollouts the notebook builds.
_DEFAULT_CKPT = Path("~/Desktop/videos/preplay_promo/craftax_seed9").expanduser()
_DEFAULT_WORLD_SEED = 3
_DEFAULT_ON_TASK = "sapphire"
_DEFAULT_OFF_TASK = "diamond"
_DEFAULT_SEED = 42
_DEFAULT_DISTANCE_FROM_GOAL = 4  # matches notebook cell 6


def render_all_themes(
  *,
  checkpoint_dir: Path = _DEFAULT_CKPT,
  world_seed: int = _DEFAULT_WORLD_SEED,
  on_task: str = _DEFAULT_ON_TASK,
  off_task: str = _DEFAULT_OFF_TASK,
  seed: int = _DEFAULT_SEED,
  distance_from_goal: int = _DEFAULT_DISTANCE_FROM_GOAL,
  themes: tuple[str, ...] = ("dark", "cream", "peach", "ivory"),
  write_gif: bool = True,
  out_dir: Path | None = None,
) -> dict[str, Path]:
  """Render the full storyboard once per theme. Returns {theme: mp4_path}.

  Bundle load and rollouts happen ONCE — they are theme-independent and the
  rollouts are pickled to disk by `rollout_cached`. Per-theme work is just
  the frame composition + encode."""
  import time

  out_dir = out_dir or OUTPUT_DIR
  out_dir.mkdir(parents=True, exist_ok=True)

  print(f"Loading bundle from {checkpoint_dir}...")
  bundle = load_craftax(checkpoint_dir)
  print(f"  signature: {bundle.signature}")

  print(
    f"Running rollouts (world={world_seed}, on={on_task}, off={off_task}, seed={seed})..."
  )
  train = rollout_train(bundle, world_seed=world_seed, goal=on_task, seed=seed)
  preplay = rollout_preplay(
    bundle,
    train,
    world_seed=world_seed,
    off_task_goal=off_task,
    seed=seed,
    distance_from_goal=distance_from_goal,
  )
  test = rollout_test(bundle, world_seed=world_seed, goal=off_task, seed=seed)
  print(f"  train len={len(train)}  preplay len={len(preplay)}  test len={len(test)}")

  results: dict[str, Path] = {}
  for theme in themes:
    print(f"\n=== Theme: {theme} ===")
    t0 = time.time()
    use_theme(theme)
    frames = render_full_storyboard(bundle, train, preplay, test, theme=theme)
    mp4_path = out_dir / f"make_promo_video_craftax_{theme}.mp4"
    gif_path = out_dir / f"make_promo_video_craftax_{theme}.gif" if write_gif else None
    encode(frames, mp4_path, out_gif=gif_path, fps=STORYBOARD["encode_fps"])
    print(f"  rendered + encoded in {time.time() - t0:.1f}s")
    results[theme] = mp4_path

  print("\nDone.")
  for theme, p in results.items():
    print(f"  {theme}: {p}")
  return results


def _main() -> None:
  import argparse

  parser = argparse.ArgumentParser(
    description="Render the preplay promo MP4 for one or more themes.",
  )
  parser.add_argument(
    "--checkpoint",
    type=Path,
    default=_DEFAULT_CKPT,
    help=f"Directory with preplay.safetensors / preplay.config (default: {_DEFAULT_CKPT})",
  )
  parser.add_argument("--world-seed", type=int, default=_DEFAULT_WORLD_SEED)
  parser.add_argument("--on-task", default=_DEFAULT_ON_TASK)
  parser.add_argument("--off-task", default=_DEFAULT_OFF_TASK)
  parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
  parser.add_argument(
    "--distance-from-goal", type=int, default=_DEFAULT_DISTANCE_FROM_GOAL
  )
  parser.add_argument(
    "--themes",
    nargs="+",
    default=list(THEMES.keys()),
    choices=list(THEMES.keys()),
    help="Themes to render (default: all four).",
  )
  parser.add_argument(
    "--no-gif", action="store_true", help="Skip GIF output (MP4 only)."
  )
  args = parser.parse_args()

  render_all_themes(
    checkpoint_dir=args.checkpoint,
    world_seed=args.world_seed,
    on_task=args.on_task,
    off_task=args.off_task,
    seed=args.seed,
    distance_from_goal=args.distance_from_goal,
    themes=tuple(args.themes),
    write_gif=not args.no_gif,
  )


if __name__ == "__main__":
  # Make `import make_promo_video` resolve to this running module so
  # pickle.find_class("make_promo_video", "Rollout") returns the same class
  # object that produced the data — keeps caches interchangeable between
  # CLI runs and notebook sessions. See the comment next to the
  # `Rollout.__module__` assignment above.
  import sys as _sys

  _sys.modules.setdefault("make_promo_video", _sys.modules["__main__"])
  _main()
