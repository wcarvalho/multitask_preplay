"""Side-by-side juncture-condition videos for tweet Asset 7.

Generates ~5 candidate mp4s showing a single participant on the juncture
manipulation, structured as a 5-stage pedagogical reveal:

  A. Caption — title card framing the comparison.
  B. Training path — animated draw of the participant's REAL training path on
     the left, right panel ghosted (alpha 0.15).
  C. Left novel trial — agent respawns at c1 (familiar-juncture) start,
     walks the novel path FPS-paced by reaction times, ends with a green
     time pill.
  D. Right novel trial — same on the right for c2 (novel-juncture) start,
     ends with a red time pill.
  E. Side-by-side w/ strobing freeze — both panels animate from t=0; when
     LEFT reaches goal first, both panels freeze for 2s while left's time
     strobes (the punctum that drives the speed comparison home), then
     right resumes; both end with their colored pills.

Filters: manipulation="juncture", block_name="reverse(Y=False,X=False)",
setting="short", tell_reuse=0 (unannounced test goal — strongest effect).

Usage:
    uv run python plots/jaxmaze_juncture_video.py
"""

import os
import re
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from housemaze import renderer
from housemaze import utils as housemaze_utils
from housemaze.human_dyna import mazes
from matplotlib.patches import FancyArrowPatch, Rectangle

import data_configs
import plot_configs  # noqa: F401
from analysis import analysis_utils, vis_utils
from promo_video_design import (
  DESIGN,
  ease_in_out,
  load_fonts,
  title_card_frames,
  use_theme,
)

# Adopt the peach palette used by the paper-launch promo. Helpers read DESIGN
# at call time, so swapping here once at import is enough.
use_theme("peach")

OUTPUT_DIR = os.path.join(
  os.path.dirname(os.path.abspath(__file__)), "output", "jaxmaze_juncture_video"
)

# --- Filters ---
BLOCK_NAME = "reverse(Y=False,X=False)"
SETTING = "short"
TELL_REUSE = 0
N_CANDIDATES = 5
# When set, generate only for this user id and ignore N_CANDIDATES ranking.
USER_ID_OVERRIDE = 2834726446

# --- Frame size + framerate ---
FPS = 30  # frame generation rate (also mp4 playback fps for real-time)
# Gif fps. Pick an integer divisor of FPS so stride = FPS // GIF_FPS lands on
# a whole number — otherwise gif plays at the wrong speed (the GIF format
# rounds per-frame delays to centiseconds and clamps to 10cs min, so
# non-divisor fps drift fast). 10 → stride 3 → real-time. 15 → stride 2 →
# real-time but ~50% larger file.
GIF_FPS = 10
VIDEO_W, VIDEO_H = 1200, 600

# --- Stage timing (seconds) ---
TITLE_DURATION_S = 3.0
TITLE_FADE_S = 0.4
STAGE_B_DURATION_S = 6.0  # training-path draw is synthetic; pace for readability
STAGE_B_HOLD_S = 3.0  # hold the completed training path so the viewer can study it
RESPAWN_HOLD_S = 0.3  # Stage C: training path fades 1.0 -> FADED_TRAIN_ALPHA
STAGE_D_HOLD_S = 0.3  # Stage D: brief hold at empty start before animation
GHOST_RAMP_S = 0.3
FREEZE_HOLD_S = 4.0
FINAL_HOLD_SEC = 1.5  # legacy: extra hold appended by _write_video at the end

# --- Per-step pacing (seconds) ---
MIN_STEP_SEC = 0.08
MAX_STEP_SEC = 3.0

# --- Strobe / pulse ---
PULSE_HZ = 2.5
PULSE_PEAK = 1.15
PULSE_CYCLES_END_TRIAL = 3  # Stage C/D end freeze: 3 pulses + settle
PULSE_CYCLES_STAGE_E_FREEZE = 5  # Stage E freeze: 5 pulses (no settle)

# --- Goal flash ---
GOAL_FLASH_FRAMES = 3

# --- Color tokens ---
# Was DESIGN["amber"] (peach palette renders this as pure red, which connoted
# "wrong"). Switched to neutral sky-blue per redesign request — both panels'
# novel paths use this so the color carries no goodness signal.
NOVEL_PATH_COLOR = DESIGN["blue"]
# DESIGN["muted"] in peach is amber-900 brown which can blend into the maze
# tile palette. Use a neutral gray for the training path overlay.
TRAIN_PATH_COLOR = "#8A8A8A"
FAST_GREEN = "#16A34A"  # left/fast time emphasis (Stages C, E)
SLOW_RED = "#DC2626"  # right/slow time emphasis (Stages D, E)
INACTIVE_PANEL_ALPHA = 0.15  # ghost panel alpha
FADED_TRAIN_ALPHA = 0.3  # persistent training path during Stages C / E left

# --- Title card text ---
TITLE_EYEBROW = "JAXMAZE · JUNCTURE MANIPULATION"
TITLE_TEXT = "Familiar start. Novel start."
TITLE_SUBTITLE = "Both pursuing a new goal for the first time."
TITLE_FONTSIZE = 54

# Footer caption painted on every trial frame (Stages B-E).
FOOTER_CAPTION = (
  "Multitask Preplay explains why people are faster at "
  "taking novel paths from familiar places."
)

_FONTS_CACHE: dict | None = None


def _fonts() -> dict:
  global _FONTS_CACHE
  if _FONTS_CACHE is None:
    _FONTS_CACHE = load_fonts()
  return _FONTS_CACHE


# Keyboard action enum from jaxmaze.env.KeyboardActions:
#   right=0, down=1, left=2, up=3, done=4
_ACTION_TO_DELTA = {
  0: (0, 1),
  1: (1, 0),
  2: (0, -1),
  3: (-1, 0),
}

_image_dict = housemaze_utils.load_image_dict()
_char2key, _, _ = mazes.get_group_set()


# === Data helpers ===


def _add_setting(df: pl.DataFrame) -> pl.DataFrame:
  return df.with_columns(
    setting=pl.when(pl.col("world").str.contains("short"))
    .then(pl.lit("short"))
    .otherwise(pl.lit("long"))
  )


def _parse_row(row: dict):
  positions = vis_utils.parse_positions_string(row["positions"])
  actions = vis_utils.parse_jax_array_string(row["actions"]).astype(int)
  rts = vis_utils.parse_jax_array_string(row["reaction_times"])
  return positions, actions, rts


def _find_training_row(user_df: pl.DataFrame, eval_row: dict):
  train_idx = eval_row.get("corresponding_train_episode_idx")
  if train_idx is not None and int(train_idx) >= 0:
    matches = user_df.filter(
      user_id=eval_row["user_id"],
      global_episode_idx=int(train_idx),
    )
    if len(matches) > 0:
      return matches.row(0, named=True)

  base_world = eval_row["world"].replace("_eval_same", "").replace("_eval_diff", "")
  matches = user_df.filter(
    user_id=eval_row["user_id"],
    world=base_world,
    block_name=eval_row["block_name"],
    eval=False,
    success=1,
  )
  if len(matches) == 0:
    return None
  return matches.sort("global_episode_idx", descending=True).row(0, named=True)


def _get_grid(world: str, block_name: str):
  match = re.match(r"reverse\(Y=(True|False),X=(True|False)\)", block_name)
  vertical = match.group(1) == "True" if match else False
  horizontal = match.group(2) == "True" if match else False
  maze_str = getattr(mazes, world)
  maze_str = housemaze_utils.reverse(maze_str, horizontal, vertical)
  level_init = housemaze_utils.from_str(maze_str, _char2key, return_map_init=False)
  grid = level_init[0]
  return np.asarray(grid)


# === Render primitives ===


def _render_state(grid, agent_pos, agent_dir):
  img = renderer.create_image_from_grid(
    grid,
    tuple(int(x) for x in agent_pos),
    int(agent_dir),
    _image_dict,
  )
  return np.asarray(img)


def _direction_from_delta(prev_pos, curr_pos):
  """Infer keyboard-action direction from a position delta. Returns 0-3."""
  dy = float(curr_pos[0]) - float(prev_pos[0])
  dx = float(curr_pos[1]) - float(prev_pos[1])
  if abs(dx) >= abs(dy):
    return 0 if dx > 0 else 2
  return 1 if dy > 0 else 3


# === Drawing sub-helpers ===


def _draw_panel_background(ax, grid, *, alpha=1.0, agent_pos=None, agent_dir=0):
  """Render the maze frame onto ax. Returns extent dict for path/label coords.

  When `agent_pos is None` we render with the agent at (0, 0) — the corner —
  so it's clipped to a barely-visible nub. Use this for ghost panels.
  """
  pos = agent_pos if agent_pos is not None else (0, 0)
  frame = _render_state(grid, pos, agent_dir)
  ax.imshow(frame, alpha=alpha)
  img_h, img_w = frame.shape[:2]
  maze_h, maze_w = grid.shape[:2]
  sy = img_h / (maze_h + 2)
  sx = img_w / (maze_w + 2)
  return {
    "sy": sy,
    "sx": sx,
    "off_y": sy,
    "off_x": sx,
    "img_h": img_h,
    "img_w": img_w,
  }


def _path_xy(extent, positions):
  xs = [extent["off_x"] + (float(x) + 0.5) * extent["sx"] for (y, x) in positions]
  ys = [extent["off_y"] + (float(y) + 0.5) * extent["sy"] for (y, x) in positions]
  return xs, ys


def _draw_polyline(ax, extent, positions, *, color, alpha, linewidth, with_arrow=True):
  if positions is None or len(positions) < 2:
    return
  xs, ys = _path_xy(extent, positions)
  ax.plot(xs, ys, color=color, alpha=alpha, linewidth=linewidth, solid_capstyle="round")
  if with_arrow:
    ax.add_patch(
      FancyArrowPatch(
        (xs[-2], ys[-2]),
        (xs[-1], ys[-1]),
        arrowstyle="-|>",
        mutation_scale=22,
        color=color,
        alpha=alpha,
        linewidth=0,
        zorder=5,
      )
    )


def _draw_train_callout(ax, extent, training_positions):
  if training_positions is None or len(training_positions) <= 1:
    return
  mid_idx = max(1, len(training_positions) // 2)
  mid_y, mid_x = training_positions[mid_idx]
  tip_x = extent["off_x"] + (float(mid_x) + 0.5) * extent["sx"]
  tip_y = extent["off_y"] + (float(mid_y) + 0.5) * extent["sy"]
  ax.annotate(
    "example training path",
    xy=(tip_x, tip_y),
    xycoords="data",
    xytext=(0.02, 0.04),
    textcoords="axes fraction",
    fontsize=15,
    fontweight="bold",
    color=TRAIN_PATH_COLOR,
    ha="left",
    va="bottom",
    bbox=dict(
      facecolor=DESIGN["surface"],
      alpha=0.95,
      edgecolor=TRAIN_PATH_COLOR,
      linewidth=1.0,
      pad=3,
    ),
    arrowprops=dict(
      arrowstyle="-|>",
      color=TRAIN_PATH_COLOR,
      lw=1.8,
      shrinkA=3,
      shrinkB=6,
      connectionstyle="arc3,rad=0.2",
      mutation_scale=14,
    ),
  )


def _draw_novel_callout(ax, extent, anchor_pos, *, color):
  if anchor_pos is None:
    return
  start_y, start_x = anchor_pos
  tip_x = extent["off_x"] + (float(start_x) + 0.5) * extent["sx"]
  tip_y = extent["off_y"] + (float(start_y) + 0.5) * extent["sy"]
  ax.annotate(
    "novel path",
    xy=(tip_x, tip_y),
    xycoords="data",
    xytext=(0.98, 0.95),
    textcoords="axes fraction",
    fontsize=15,
    fontweight="bold",
    color=color,
    ha="right",
    va="top",
    bbox=dict(
      facecolor=DESIGN["surface"],
      alpha=0.95,
      edgecolor=color,
      linewidth=1.0,
      pad=3,
    ),
    arrowprops=dict(
      arrowstyle="-|>",
      color=color,
      lw=1.8,
      shrinkA=3,
      shrinkB=6,
      connectionstyle="arc3,rad=0.2",
      mutation_scale=14,
    ),
  )


def _draw_title_box(ax, title):
  ax.set_title(
    title,
    fontsize=15,
    fontweight="bold",
    color=DESIGN["fg"],
    pad=8,
    fontproperties=_fonts()["bold"],
    bbox=dict(facecolor=DESIGN["surface"], alpha=0.9, edgecolor="none", pad=4),
  )


def _draw_time_pill(ax, time_s, *, color=None, emphasized=False, scale=1.0):
  if emphasized:
    fill_color = color if color is not None else DESIGN["fg"]
    fontsize = 22 * scale
    pad = 8 * scale
    ax.text(
      0.02,
      0.98,
      f"t = {time_s:.1f}s",
      transform=ax.transAxes,
      fontsize=fontsize,
      va="top",
      ha="left",
      color="#FFFFFF",
      fontproperties=_fonts()["bold"],
      bbox=dict(facecolor=fill_color, alpha=1.0, edgecolor="none", pad=pad),
    )
  else:
    text_color = color if color is not None else DESIGN["fg"]
    ax.text(
      0.02,
      0.98,
      f"t = {time_s:.1f}s",
      transform=ax.transAxes,
      fontsize=16,
      va="top",
      ha="left",
      color=text_color,
      fontproperties=_fonts()["semibold"],
      bbox=dict(facecolor=DESIGN["surface"], alpha=0.9, edgecolor="none", pad=4),
    )


def _draw_paused_indicator(ax, time_s):
  ax.text(
    0.98,
    0.04,
    f"paused at {time_s:.1f}s",
    transform=ax.transAxes,
    fontsize=13,
    va="bottom",
    ha="right",
    color=DESIGN["muted"],
    fontproperties=_fonts()["semibold"],
    bbox=dict(facecolor=DESIGN["surface"], alpha=0.85, edgecolor="none", pad=3),
  )


def _draw_goal_flash(ax):
  """Thick white border around the panel as a 'goal reached' flash."""
  rect = Rectangle(
    (0, 0),
    1,
    1,
    transform=ax.transAxes,
    fill=False,
    edgecolor="#FFFFFF",
    linewidth=12,
    zorder=20,
  )
  ax.add_patch(rect)


def _decorate_axis(ax):
  ax.set_xticks([])
  ax.set_yticks([])
  for spine in ax.spines.values():
    spine.set_visible(False)


def _pulse_scale(frame_in_pulse, frames_per_pulse, peak=PULSE_PEAK):
  """Half-sine pulse: 1.0 at endpoints, peak at midpoint."""
  if frames_per_pulse <= 0:
    return 1.0
  phase = frame_in_pulse / frames_per_pulse
  s = np.sin(np.pi * float(np.clip(phase, 0.0, 1.0)))
  return 1.0 + (peak - 1.0) * float(s)


# === Figure / layout helpers ===


def _make_figure():
  fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100, facecolor=DESIGN["bg"])
  for ax in axes:
    ax.set_facecolor(DESIGN["bg"])
  return fig, axes


def _draw_footer(fig):
  fig.text(
    0.5,
    0.045,
    FOOTER_CAPTION,
    ha="center",
    va="bottom",
    fontsize=17,
    color=DESIGN["fg"],
    fontproperties=_fonts()["semibold"],
  )


def _finalize_layout(fig):
  fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.11, wspace=0.05)


def _fig_to_rgb(fig) -> np.ndarray:
  fig.canvas.draw()
  buf = np.asarray(fig.canvas.buffer_rgba())
  return buf[..., :3].copy()


# === Per-frame panel orchestrator ===


def _render_panel(
  ax,
  grid,
  *,
  panel_alpha=1.0,
  agent_pos=None,
  agent_dir=0,
  training_positions=None,
  training_alpha=1.0,
  agent_path=None,
  novel_color=NOVEL_PATH_COLOR,
  train_color=TRAIN_PATH_COLOR,
  title=None,
  show_train_callout=False,
  show_novel_callout=False,
  novel_callout_anchor=None,
  time_s=None,
  time_color=None,
  time_emphasized=False,
  time_scale=1.0,
  paused_at_s=None,
  goal_flash=False,
):
  """Compose a single panel-frame from focused sub-helpers.

  Different stages turn different knobs on/off; this function just
  dispatches to the appropriate _draw_* helpers.
  """
  extent = _draw_panel_background(
    ax, grid, alpha=panel_alpha, agent_pos=agent_pos, agent_dir=agent_dir
  )

  if training_positions is not None and len(training_positions) > 1:
    _draw_polyline(
      ax,
      extent,
      training_positions,
      color=train_color,
      alpha=training_alpha * panel_alpha,
      linewidth=4,
      with_arrow=True,
    )

  if agent_path is not None and len(agent_path) > 1:
    _draw_polyline(
      ax,
      extent,
      agent_path,
      color=novel_color,
      alpha=0.95 * panel_alpha,
      linewidth=5,
      with_arrow=False,
    )

  if show_train_callout:
    _draw_train_callout(ax, extent, training_positions)

  if show_novel_callout:
    _draw_novel_callout(ax, extent, novel_callout_anchor, color=novel_color)

  if title:
    _draw_title_box(ax, title)

  if time_s is not None:
    _draw_time_pill(
      ax,
      time_s,
      color=time_color,
      emphasized=time_emphasized,
      scale=time_scale,
    )

  if paused_at_s is not None:
    _draw_paused_indicator(ax, paused_at_s)

  if goal_flash:
    _draw_goal_flash(ax)

  _decorate_axis(ax)


def _ghost_panel(ax, grid):
  """Convenience: render an inactive panel as a faint grid (no agent visible)."""
  _render_panel(ax, grid, panel_alpha=INACTIVE_PANEL_ALPHA)


def _clipped_cumsum(rts: np.ndarray) -> np.ndarray:
  clipped = np.clip(rts.astype(float), MIN_STEP_SEC, MAX_STEP_SEC)
  return np.cumsum(clipped)


# === Stage builders ===


def _build_stage_a_caption() -> list:
  """Stage A — three-tier title card framing the comparison."""
  return title_card_frames(
    eyebrow=TITLE_EYEBROW,
    title=TITLE_TEXT,
    subtitle=TITLE_SUBTITLE,
    size=(VIDEO_W, VIDEO_H),
    fps=FPS,
    duration_s=TITLE_DURATION_S,
    fade_s=TITLE_FADE_S,
    fonts=_fonts(),
    title_fontsize=TITLE_FONTSIZE,
  )


def _build_stage_b_training_path(c1_row, train_row) -> list:
  """Stage B — animated draw of the participant's training path on the left.

  Walks the original integer cells one at a time, paced so the full path takes
  STAGE_B_DURATION_S. Polyline endpoint and rendered agent always sit on the
  same integer cell — no fractional smoothing — so the arrow tip stays under
  the agent triangle every frame (matching the novel-path stages).
  """
  if train_row is None:
    return []

  train_p, _train_a, _ = _parse_row(train_row)
  grid_left = _get_grid(c1_row["world"], c1_row["block_name"])
  grid_right = grid_left  # same maze, ghosted on right

  n_steps = len(train_p)
  n_anim_frames = int(round(STAGE_B_DURATION_S * FPS))
  n_hold_frames = int(round(STAGE_B_HOLD_S * FPS))

  frames = []
  for f in range(n_anim_frames + n_hold_frames):
    progress = (f + 1) / max(n_anim_frames, 1) if f < n_anim_frames else 1.0
    step_idx = min(int(np.floor(progress * n_steps)), n_steps - 1)

    agent_pos = tuple(train_p[step_idx])
    prev_pos = train_p[max(step_idx - 1, 0)]
    direction = _direction_from_delta(prev_pos, agent_pos)

    visible_path = train_p[: step_idx + 1]
    show_callout = f >= n_anim_frames

    fig, axes = _make_figure()
    _render_panel(
      axes[0],
      grid_left,
      agent_pos=agent_pos,
      agent_dir=direction,
      training_positions=visible_path if len(visible_path) > 1 else None,
      train_color=TRAIN_PATH_COLOR,
      title="Familiar part of map (training)",
      show_train_callout=show_callout,
    )
    _ghost_panel(axes[1], grid_right)
    _draw_footer(fig)
    _finalize_layout(fig)
    frames.append(_fig_to_rgb(fig))
    plt.close(fig)

  return frames


def _build_stage_c_left_novel(c1_row, train_row) -> list:
  """Stage C — left novel trial: respawn fade, animation, green time pill."""
  p1, a1, rt1 = _parse_row(c1_row)
  train_p = None
  if train_row is not None:
    train_p, _, _ = _parse_row(train_row)
  grid_left = _get_grid(c1_row["world"], c1_row["block_name"])
  grid_right = grid_left

  t_events = _clipped_cumsum(rt1)
  n_anim_frames = int(np.ceil(float(t_events[-1]) * FPS))
  n_respawn_frames = int(round(RESPAWN_HOLD_S * FPS))
  n_freeze_frames = int(round(FREEZE_HOLD_S * FPS))

  start_pos = tuple(p1[0])
  start_dir = int(a1[0]) if len(a1) > 0 and int(a1[0]) in _ACTION_TO_DELTA else 0

  frames = []

  # --- Respawn hold: training path alpha ramps 1.0 -> FADED_TRAIN_ALPHA ---
  for f in range(n_respawn_frames):
    alpha_t = ease_in_out((f + 1) / max(n_respawn_frames, 1))
    train_alpha = 1.0 + alpha_t * (FADED_TRAIN_ALPHA - 1.0)

    fig, axes = _make_figure()
    _render_panel(
      axes[0],
      grid_left,
      agent_pos=start_pos,
      agent_dir=start_dir,
      training_positions=train_p,
      training_alpha=train_alpha,
      train_color=TRAIN_PATH_COLOR,
      title="New task from familiar part of map",
      show_novel_callout=True,
      novel_callout_anchor=start_pos,
    )
    _ghost_panel(axes[1], grid_right)
    _draw_footer(fig)
    _finalize_layout(fig)
    frames.append(_fig_to_rgb(fig))
    plt.close(fig)

  # --- Animation ---
  for f in range(n_anim_frames):
    t = (f + 1) / FPS
    step_idx = min(int(np.searchsorted(t_events, t, side="right")), len(p1) - 1)
    t_clipped = min(t, float(t_events[-1]))
    last_action = int(a1[max(step_idx - 1, 0)]) if len(a1) > 0 else 0
    direction = last_action if last_action in _ACTION_TO_DELTA else 0

    fig, axes = _make_figure()
    _render_panel(
      axes[0],
      grid_left,
      agent_pos=tuple(p1[step_idx]),
      agent_dir=direction,
      training_positions=train_p,
      training_alpha=FADED_TRAIN_ALPHA,
      train_color=TRAIN_PATH_COLOR,
      agent_path=p1[: step_idx + 1] if step_idx > 0 else None,
      novel_color=NOVEL_PATH_COLOR,
      title="New task from familiar part of map",
      show_novel_callout=True,
      novel_callout_anchor=start_pos,
      time_s=t_clipped,
    )
    _ghost_panel(axes[1], grid_right)
    _draw_footer(fig)
    _finalize_layout(fig)
    frames.append(_fig_to_rgb(fig))
    plt.close(fig)

  final_t = float(t_events[-1])
  final_pos = tuple(p1[-1])
  final_dir = int(a1[-1]) if len(a1) > 0 and int(a1[-1]) in _ACTION_TO_DELTA else 0

  # --- Goal flash ---
  for _ in range(GOAL_FLASH_FRAMES):
    fig, axes = _make_figure()
    _render_panel(
      axes[0],
      grid_left,
      agent_pos=final_pos,
      agent_dir=final_dir,
      training_positions=train_p,
      training_alpha=FADED_TRAIN_ALPHA,
      train_color=TRAIN_PATH_COLOR,
      agent_path=p1,
      novel_color=NOVEL_PATH_COLOR,
      title="New task from familiar part of map",
      show_novel_callout=True,
      novel_callout_anchor=start_pos,
      time_s=final_t,
      goal_flash=True,
    )
    _ghost_panel(axes[1], grid_right)
    _draw_footer(fig)
    _finalize_layout(fig)
    frames.append(_fig_to_rgb(fig))
    plt.close(fig)

  # --- Freeze: green pill, 3 pulses + settle ---
  frames_per_pulse = max(int(round(FPS / PULSE_HZ)), 1)
  n_pulse_phase = PULSE_CYCLES_END_TRIAL * frames_per_pulse
  for f in range(n_freeze_frames):
    if f < n_pulse_phase:
      cycle_f = f % frames_per_pulse
      scale = _pulse_scale(cycle_f, frames_per_pulse)
    else:
      scale = 1.0

    fig, axes = _make_figure()
    _render_panel(
      axes[0],
      grid_left,
      agent_pos=final_pos,
      agent_dir=final_dir,
      training_positions=train_p,
      training_alpha=FADED_TRAIN_ALPHA,
      train_color=TRAIN_PATH_COLOR,
      agent_path=p1,
      novel_color=NOVEL_PATH_COLOR,
      title="New task from familiar part of map",
      show_novel_callout=True,
      novel_callout_anchor=start_pos,
      time_s=final_t,
      time_color=FAST_GREEN,
      time_emphasized=True,
      time_scale=scale,
    )
    _ghost_panel(axes[1], grid_right)
    _draw_footer(fig)
    _finalize_layout(fig)
    frames.append(_fig_to_rgb(fig))
    plt.close(fig)

  return frames


def _build_stage_d_right_novel(c2_row) -> list:
  """Stage D — right novel trial: animation, red time pill. No training overlay."""
  p2, a2, rt2 = _parse_row(c2_row)
  grid_left = _get_grid(c2_row["world"], c2_row["block_name"])
  grid_right = grid_left

  t_events = _clipped_cumsum(rt2)
  n_anim_frames = int(np.ceil(float(t_events[-1]) * FPS))
  n_hold_frames = int(round(STAGE_D_HOLD_S * FPS))
  n_freeze_frames = int(round(FREEZE_HOLD_S * FPS))

  start_pos = tuple(p2[0])
  start_dir = int(a2[0]) if len(a2) > 0 and int(a2[0]) in _ACTION_TO_DELTA else 0

  frames = []

  # --- Hold: agent at c2 start, no path drawn yet ---
  for _ in range(n_hold_frames):
    fig, axes = _make_figure()
    _ghost_panel(axes[0], grid_left)
    _render_panel(
      axes[1],
      grid_right,
      agent_pos=start_pos,
      agent_dir=start_dir,
      novel_color=NOVEL_PATH_COLOR,
      title="New task from new part of map",
      show_novel_callout=True,
      novel_callout_anchor=start_pos,
    )
    _draw_footer(fig)
    _finalize_layout(fig)
    frames.append(_fig_to_rgb(fig))
    plt.close(fig)

  # --- Animation ---
  for f in range(n_anim_frames):
    t = (f + 1) / FPS
    step_idx = min(int(np.searchsorted(t_events, t, side="right")), len(p2) - 1)
    t_clipped = min(t, float(t_events[-1]))
    last_action = int(a2[max(step_idx - 1, 0)]) if len(a2) > 0 else 0
    direction = last_action if last_action in _ACTION_TO_DELTA else 0

    fig, axes = _make_figure()
    _ghost_panel(axes[0], grid_left)
    _render_panel(
      axes[1],
      grid_right,
      agent_pos=tuple(p2[step_idx]),
      agent_dir=direction,
      agent_path=p2[: step_idx + 1] if step_idx > 0 else None,
      novel_color=NOVEL_PATH_COLOR,
      title="New task from new part of map",
      show_novel_callout=True,
      novel_callout_anchor=start_pos,
      time_s=t_clipped,
    )
    _draw_footer(fig)
    _finalize_layout(fig)
    frames.append(_fig_to_rgb(fig))
    plt.close(fig)

  final_t = float(t_events[-1])
  final_pos = tuple(p2[-1])
  final_dir = int(a2[-1]) if len(a2) > 0 and int(a2[-1]) in _ACTION_TO_DELTA else 0

  # --- Goal flash ---
  for _ in range(GOAL_FLASH_FRAMES):
    fig, axes = _make_figure()
    _ghost_panel(axes[0], grid_left)
    _render_panel(
      axes[1],
      grid_right,
      agent_pos=final_pos,
      agent_dir=final_dir,
      agent_path=p2,
      novel_color=NOVEL_PATH_COLOR,
      title="New task from new part of map",
      show_novel_callout=True,
      novel_callout_anchor=start_pos,
      time_s=final_t,
      goal_flash=True,
    )
    _draw_footer(fig)
    _finalize_layout(fig)
    frames.append(_fig_to_rgb(fig))
    plt.close(fig)

  # --- Freeze: red pill, 3 pulses + settle ---
  frames_per_pulse = max(int(round(FPS / PULSE_HZ)), 1)
  n_pulse_phase = PULSE_CYCLES_END_TRIAL * frames_per_pulse
  for f in range(n_freeze_frames):
    if f < n_pulse_phase:
      cycle_f = f % frames_per_pulse
      scale = _pulse_scale(cycle_f, frames_per_pulse)
    else:
      scale = 1.0

    fig, axes = _make_figure()
    _ghost_panel(axes[0], grid_left)
    _render_panel(
      axes[1],
      grid_right,
      agent_pos=final_pos,
      agent_dir=final_dir,
      agent_path=p2,
      novel_color=NOVEL_PATH_COLOR,
      title="New task from new part of map",
      show_novel_callout=True,
      novel_callout_anchor=start_pos,
      time_s=final_t,
      time_color=SLOW_RED,
      time_emphasized=True,
      time_scale=scale,
    )
    _draw_footer(fig)
    _finalize_layout(fig)
    frames.append(_fig_to_rgb(fig))
    plt.close(fig)

  return frames


def _build_stage_e_side_by_side(c1_row, c2_row, train_row) -> tuple[list, dict]:
  """Stage E — side-by-side w/ strobing freeze when left finishes.

  Returns (frames, markers) where markers are int indices into frames marking
  the start of named phases (freeze_start, right_resume_start,
  right_goal_flash_start, final_hold_start). The trailer gif uses these to
  splice together the punchline.
  """
  p1, a1, rt1 = _parse_row(c1_row)
  p2, a2, rt2 = _parse_row(c2_row)
  train_p = None
  if train_row is not None:
    train_p, _, _ = _parse_row(train_row)
  grid_left = _get_grid(c1_row["world"], c1_row["block_name"])
  grid_right = _get_grid(c2_row["world"], c2_row["block_name"])

  t_events1 = _clipped_cumsum(rt1)
  t_events2 = _clipped_cumsum(rt2)
  total_dur1 = float(t_events1[-1])
  total_dur2 = float(t_events2[-1])

  n_ramp = int(round(GHOST_RAMP_S * FPS))
  n_freeze_frames = int(round(FREEZE_HOLD_S * FPS))
  frames_per_pulse = max(int(round(FPS / PULSE_HZ)), 1)

  start_pos1 = tuple(p1[0])
  start_pos2 = tuple(p2[0])
  start_dir1 = int(a1[0]) if len(a1) > 0 and int(a1[0]) in _ACTION_TO_DELTA else 0
  start_dir2 = int(a2[0]) if len(a2) > 0 and int(a2[0]) in _ACTION_TO_DELTA else 0

  frames: list = []
  markers: dict = {}

  # --- Ghost ramp on left (was inactive in Stage D); right at full alpha ---
  for f in range(n_ramp):
    alpha_t = ease_in_out((f + 1) / max(n_ramp, 1))
    left_alpha = INACTIVE_PANEL_ALPHA + alpha_t * (1.0 - INACTIVE_PANEL_ALPHA)
    show_left_overlays = alpha_t > 0.7

    fig, axes = _make_figure()
    _render_panel(
      axes[0],
      grid_left,
      panel_alpha=left_alpha,
      agent_pos=start_pos1,
      agent_dir=start_dir1,
      training_positions=train_p,
      training_alpha=FADED_TRAIN_ALPHA,
      train_color=TRAIN_PATH_COLOR,
      title="New task from familiar part of map" if show_left_overlays else None,
      show_novel_callout=show_left_overlays,
      novel_callout_anchor=start_pos1,
    )
    _render_panel(
      axes[1],
      grid_right,
      agent_pos=start_pos2,
      agent_dir=start_dir2,
      novel_color=NOVEL_PATH_COLOR,
      title="New task from new part of map",
      show_novel_callout=True,
      novel_callout_anchor=start_pos2,
    )
    _draw_footer(fig)
    _finalize_layout(fig)
    frames.append(_fig_to_rgb(fig))
    plt.close(fig)

  markers["animate_start"] = len(frames)

  # --- State machine: animate -> left_goal_flash -> freeze_strobe ->
  #     right_resume -> right_goal_flash -> final_hold ---
  state = "animating"
  state_frames_remaining = 0
  t_left = 0.0
  t_right = 0.0
  left_finished = False
  right_finished = False
  left_finish_handled = False
  right_finish_handled = False
  left_emphasis_active = False
  right_emphasis_active = False

  max_iter = int((total_dur1 + total_dur2 + 4 * FREEZE_HOLD_S + 2) * FPS) + 200
  for _iter in range(max_iter):
    # 1. Advance time per state
    if state == "animating":
      if not left_finished:
        t_left += 1 / FPS
      if not right_finished:
        t_right += 1 / FPS
    elif state == "right_resume":
      if not right_finished:
        t_right += 1 / FPS

    # 2. Clamp + finish flags
    if t_left >= total_dur1:
      left_finished = True
      t_left = total_dur1
    if t_right >= total_dur2:
      right_finished = True
      t_right = total_dur2

    # 3. Step indices + directions
    step_idx1 = min(int(np.searchsorted(t_events1, t_left, side="right")), len(p1) - 1)
    step_idx2 = min(int(np.searchsorted(t_events2, t_right, side="right")), len(p2) - 1)
    last_action1 = int(a1[max(step_idx1 - 1, 0)]) if len(a1) > 0 else 0
    last_action2 = int(a2[max(step_idx2 - 1, 0)]) if len(a2) > 0 else 0
    dir1 = last_action1 if last_action1 in _ACTION_TO_DELTA else 0
    dir2 = last_action2 if last_action2 in _ACTION_TO_DELTA else 0

    # 4. Per-state visual flags
    left_goal_flash_now = state == "left_goal_flash"
    right_goal_flash_now = state == "right_goal_flash"
    paused_at_right = state == "freeze_strobe"

    # Left strobe scale (only during freeze_strobe)
    if state == "freeze_strobe":
      elapsed_in_freeze = n_freeze_frames - state_frames_remaining
      cycles_done = elapsed_in_freeze // frames_per_pulse
      if cycles_done < PULSE_CYCLES_STAGE_E_FREEZE:
        cycle_f = elapsed_in_freeze % frames_per_pulse
        left_scale = _pulse_scale(cycle_f, frames_per_pulse)
      else:
        left_scale = 1.0
    else:
      left_scale = 1.0

    # 5. Render frame
    fig, axes = _make_figure()
    _render_panel(
      axes[0],
      grid_left,
      agent_pos=tuple(p1[step_idx1]),
      agent_dir=dir1,
      training_positions=train_p,
      training_alpha=FADED_TRAIN_ALPHA,
      train_color=TRAIN_PATH_COLOR,
      agent_path=p1[: step_idx1 + 1] if step_idx1 > 0 else None,
      novel_color=NOVEL_PATH_COLOR,
      title="New task from familiar part of map",
      show_novel_callout=True,
      novel_callout_anchor=start_pos1,
      time_s=t_left,
      time_color=FAST_GREEN if left_emphasis_active else None,
      time_emphasized=left_emphasis_active,
      time_scale=left_scale,
      goal_flash=left_goal_flash_now,
    )
    _render_panel(
      axes[1],
      grid_right,
      agent_pos=tuple(p2[step_idx2]),
      agent_dir=dir2,
      agent_path=p2[: step_idx2 + 1] if step_idx2 > 0 else None,
      novel_color=NOVEL_PATH_COLOR,
      title="New task from new part of map",
      show_novel_callout=True,
      novel_callout_anchor=start_pos2,
      time_s=t_right if not paused_at_right else None,
      time_color=SLOW_RED if right_emphasis_active else None,
      time_emphasized=right_emphasis_active,
      paused_at_s=t_right if paused_at_right else None,
      goal_flash=right_goal_flash_now,
    )
    _draw_footer(fig)
    _finalize_layout(fig)
    frames.append(_fig_to_rgb(fig))
    plt.close(fig)

    # 6. State transitions (after rendering)
    if state == "animating":
      if left_finished and not left_finish_handled:
        left_finish_handled = True
        state = "left_goal_flash"
        state_frames_remaining = GOAL_FLASH_FRAMES
      elif right_finished and not right_finish_handled:
        # Right finished first — atypical for our user but handle by
        # skipping the freeze and going straight to right's flash.
        right_finish_handled = True
        right_emphasis_active = True
        state = "right_goal_flash"
        state_frames_remaining = GOAL_FLASH_FRAMES
        markers["right_goal_flash_start"] = len(frames)
    elif state == "left_goal_flash":
      state_frames_remaining -= 1
      if state_frames_remaining <= 0:
        state = "freeze_strobe"
        state_frames_remaining = n_freeze_frames
        left_emphasis_active = True
        markers["freeze_start"] = len(frames)
    elif state == "freeze_strobe":
      state_frames_remaining -= 1
      if state_frames_remaining <= 0:
        if right_finished:
          right_finish_handled = True
          right_emphasis_active = True
          state = "right_goal_flash"
          state_frames_remaining = GOAL_FLASH_FRAMES
          markers["right_goal_flash_start"] = len(frames)
        else:
          state = "right_resume"
          markers["right_resume_start"] = len(frames)
    elif state == "right_resume":
      if right_finished and not right_finish_handled:
        right_finish_handled = True
        right_emphasis_active = True
        state = "right_goal_flash"
        state_frames_remaining = GOAL_FLASH_FRAMES
        markers["right_goal_flash_start"] = len(frames)
    elif state == "right_goal_flash":
      state_frames_remaining -= 1
      if state_frames_remaining <= 0:
        state = "final_hold"
        state_frames_remaining = n_freeze_frames
        markers["final_hold_start"] = len(frames)
    elif state == "final_hold":
      state_frames_remaining -= 1
      if state_frames_remaining <= 0:
        state = "done"
        break

  return frames, markers


# === Top-level assembly ===


def _build_all_stages(c1_row, c2_row, train_row) -> dict:
  return {
    "A": _build_stage_a_caption(),
    "B": _build_stage_b_training_path(c1_row, train_row),
    "C": _build_stage_c_left_novel(c1_row, train_row),
    "D": _build_stage_d_right_novel(c2_row),
    "E": _build_stage_e_side_by_side(c1_row, c2_row, train_row),  # (frames, markers)
  }


def _flatten_stages(per_stage: dict) -> list:
  e_frames, _ = per_stage["E"]
  return per_stage["A"] + per_stage["B"] + per_stage["C"] + per_stage["D"] + e_frames


# === Writers ===


def _write_video(frames, output_path):
  if frames:
    frames = list(frames) + [frames[-1]] * int(round(FINAL_HOLD_SEC * FPS))

  writer = imageio.get_writer(
    output_path,
    format="FFMPEG",
    fps=FPS,
    codec="libx264",
    quality=8,
    macro_block_size=1,
  )
  for f in frames:
    writer.append_data(f)
  writer.close()


def _write_gif(frames, output_path, gif_fps=GIF_FPS):
  """Write a size-reasonable gif at real-time playback speed.

  Requires gif_fps to divide FPS evenly so stride * gif_fps == FPS — each kept
  frame represents 1/gif_fps of story time and is shown for 1/gif_fps wallclock,
  matching the mp4's playback speed exactly.
  """
  if not frames:
    return
  if FPS % gif_fps != 0:
    raise ValueError(
      f"GIF_FPS={gif_fps} must divide FPS={FPS} evenly to play at real-time."
    )
  stride = FPS // gif_fps
  sampled = list(frames[::stride]) + [frames[-1]] * int(round(FINAL_HOLD_SEC * gif_fps))
  imageio.mimwrite(output_path, sampled, fps=gif_fps, loop=0)


# === Entrypoint ===


def main():
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  from analysis.download_dataframes import download_jaxmaze_data

  download_jaxmaze_data()

  user_df = pl.read_parquet(data_configs.get_dataframe_path("jaxmaze", "human"))

  juncture_df, _ = analysis_utils.filter_users_by_success_and_tell_reuse(
    user_df.filter(manipulation="juncture"),
    analysis_name="juncture_results",
  )
  juncture_df = _add_setting(juncture_df)

  eval_df = juncture_df.filter(
    block_name=BLOCK_NAME,
    setting=SETTING,
    tell_reuse=TELL_REUSE,
    eval=True,
    success=1,
  )

  c1 = eval_df.filter(condition=1)
  c2 = eval_df.filter(condition=2)
  paired = c1.join(c2, on=["user_id", "block_name"], suffix="_c2").with_columns(
    rt_diff=pl.col("first_log_rt_c2") - pl.col("first_log_rt"),
  )
  paired = paired.sort("rt_diff", descending=True)

  print(f"Found {len(paired)} users with both successful conditions.")
  print(f"Top {N_CANDIDATES} by Δfirst_log_rt (cond2 - cond1):")
  print(
    paired.select(["user_id", "rt_diff", "first_log_rt", "first_log_rt_c2"]).head(
      N_CANDIDATES
    )
  )

  if USER_ID_OVERRIDE is not None:
    paired = paired.filter(user_id=USER_ID_OVERRIDE)
    if len(paired) == 0:
      raise ValueError(f"USER_ID_OVERRIDE={USER_ID_OVERRIDE} not in paired candidates.")

  for rank in range(min(N_CANDIDATES, len(paired))):
    uid = paired["user_id"][rank]
    c1_row = eval_df.filter(user_id=uid, condition=1).row(0, named=True)
    c2_row = eval_df.filter(user_id=uid, condition=2).row(0, named=True)
    train_row = _find_training_row(juncture_df, c1_row)

    print(f"[{rank + 1}/{N_CANDIDATES}] user {uid}  Δ={paired['rt_diff'][rank]:.3f}")
    per_stage = _build_all_stages(c1_row, c2_row, train_row)
    full_frames = _flatten_stages(per_stage)

    mp4_path = os.path.join(OUTPUT_DIR, f"jaxmaze_juncture_video_user{uid}.mp4")
    gif_path = os.path.join(OUTPUT_DIR, f"jaxmaze_juncture_video_user{uid}.gif")
    _write_video(full_frames, mp4_path)
    _write_gif(full_frames, gif_path)
    print(f"  saved {mp4_path}")
    print(f"  saved {gif_path}")


if __name__ == "__main__":
  main()
