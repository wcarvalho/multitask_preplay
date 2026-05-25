"""Design helpers for the promo video (see PROMO_VIDEO_DESIGN.md).

Pure helpers — no project imports. Import these into `make_promo_video.py` and
call them from `render_full_storyboard`, `_make_title_card`, `_compose_panel`,
`render_minimap_with_trails`, and `encode`.

Design language summary:
  - Tailwind-grade dark palette (see DESIGN dict).
  - Inter typography (three-tier on title cards: eyebrow / title / subtitle).
  - Cubic ease-in-out on every transition; crossfade between scene blocks.
  - Ken Burns zoom on static frames.
  - Trail subdivision for smooth draw-in.
  - Lower-third gradient instead of flat caption band.

Every helper returns either a numpy uint8 RGB array or a list of them, matching
the conventions in `make_promo_video.py`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np

# ────────── Design tokens ──────────

# Four named palettes. Switch with `use_theme(name)` — every helper reads
# `DESIGN` at call time so the swap takes effect for subsequent renders. To
# preview side-by-side, set the theme, build/preview the block, then set the
# next theme. The make_promo_video.py builders accept `theme=...` directly.
THEMES: dict[str, dict] = {
  # A — original Anthropic-launch dark, Dreamer / SIMA visual lineage.
  "dark": {
    "bg": "#0A0A0F",
    "fg": "#F8FAFC",
    "muted": "#94A3B8",
    "subtitle_fg": "#E2E8F0",
    "blue": "#60A5FA",  # tailwind blue-400
    "amber": "#FBBF24",  # tailwind amber-400
    "surface": "#111827",
  },
  # On the light themes (cream / peach / ivory) the on-task / off-task colors
  # match craftax_multi_envs.py — Wong colorblind-safe sky-blue + bright red,
  # the same palette used in paper Figure 1. They pop against both the cream-
  # family backgrounds and the green-brown Craftax pixel art.
  # B — Wilka's warm-cream / habit-website (slate text on warm cream).
  "cream": {
    "bg": "#FBF6EC",  # warm cream
    "fg": "#1F2937",  # slate-800
    "muted": "#78716C",  # stone-500
    "subtitle_fg": "#374151",  # slate-700
    "blue": "#56B4E9",  # multi_envs sky blue (TRAIN_COLOR)
    "amber": "#FD0000",  # multi_envs red (EVAL_COLOR)
    "surface": "#F4ECDC",
  },
  # C — peach: warmer/more "orange" bg, multi_envs sky-blue + red accents.
  "peach": {
    "bg": "#FDE7CC",  # peachy cream
    "fg": "#1C1917",  # stone-900
    "muted": "#78350F",  # amber-900
    "subtitle_fg": "#44403C",  # stone-700
    "blue": "#56B4E9",  # multi_envs sky blue
    "amber": "#FD0000",  # multi_envs red
    "surface": "#FBE0BC",
  },
  # D — ivory + teal: cream/sage/amber/teal canon, dark-teal headings, but
  # markers/trails follow multi_envs so figure 1 and the video look related.
  "ivory": {
    "bg": "#F5EFE0",  # pale ivory
    "fg": "#134E4A",  # teal-900
    "muted": "#57534E",  # stone-600
    "subtitle_fg": "#115E59",  # teal-800
    "blue": "#56B4E9",  # multi_envs sky blue
    "amber": "#FD0000",  # multi_envs red
    "surface": "#ECE5D3",
  },
}

# Active palette — read by every helper. Mutate via use_theme().
# Default = peach (warm cream/orange) per project preference.
DESIGN: dict = dict(THEMES["peach"])


def use_theme(name: str) -> dict:
  """Swap the active palette in place. Returns the (mutated) DESIGN singleton.

  Helpers read DESIGN at call time, so this affects subsequent render calls
  only. To compare themes side-by-side: set theme → build & preview block →
  set the next theme → build & preview the next block.
  """
  if name not in THEMES:
    raise ValueError(f"unknown theme {name!r}; available: {sorted(THEMES)}")
  DESIGN.clear()
  DESIGN.update(THEMES[name])
  return DESIGN


# ────────── Fonts ──────────


def load_fonts() -> dict:
  """Return {"bold", "regular", "semibold"} FontProperties for Inter.

  Searches common install locations. If Inter isn't found, falls back to
  matplotlib default with a warning printed (still works, just not as pretty).

  One-time install:
      brew install --cask font-inter   # macOS
      # or download from https://rsms.me/inter/ and place the .ttf files in
      # ~/.fonts/ (Linux) or /Library/Fonts/ (macOS).
  """
  from matplotlib import font_manager

  candidates = {
    "bold": ["Inter-Bold.ttf", "Inter Bold.ttf", "Inter-Bold.otf"],
    "semibold": ["Inter-SemiBold.ttf", "Inter SemiBold.ttf", "Inter-SemiBold.otf"],
    "regular": ["Inter-Regular.ttf", "Inter Regular.ttf", "Inter-Regular.otf"],
  }
  search_dirs = [
    Path.home() / ".fonts",
    Path.home() / "Library" / "Fonts",
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts"),
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
  ]

  fonts = {}
  for weight, names in candidates.items():
    found = None
    for d in search_dirs:
      if not d.exists():
        continue
      for name in names:
        for match in d.rglob(name):
          found = str(match)
          break
        if found:
          break
      if found:
        break
    if found:
      fonts[weight] = font_manager.FontProperties(fname=found)
    else:
      print(
        f"[promo_video_design] Inter {weight} not found — using default. "
        f"Install Inter from https://rsms.me/inter/ for best results."
      )
      fonts[weight] = font_manager.FontProperties(
        family="sans-serif",
        weight={"bold": "bold", "semibold": "semibold", "regular": "normal"}[weight],
      )
  return fonts


# ────────── Easing ──────────


def ease_in_out(t: float | np.ndarray) -> float | np.ndarray:
  """Cubic ease-in-out. Maps [0, 1] → [0, 1] with smooth start and stop.

  Use for every alpha ramp, zoom, and position interpolation. Linear transitions
  read as robotic; eased transitions read as designed.
  """
  t = np.clip(np.asarray(t), 0.0, 1.0)
  return 3 * t**2 - 2 * t**3


# ────────── Crossfade ──────────


def crossfade(
  frame_a: np.ndarray, frame_b: np.ndarray, n: int = 10
) -> list[np.ndarray]:
  """Eased crossfade from frame_a to frame_b over n frames.

  Both frames must be uint8 RGB arrays of matching shape. Returns a list of
  n frames (not including a or b themselves — insert this between two scene
  blocks so the last frame of A and first frame of B bracket the crossfade).
  """
  assert frame_a.shape == frame_b.shape, (
    f"crossfade shape mismatch: {frame_a.shape} vs {frame_b.shape}"
  )
  a = frame_a.astype(np.float32)
  b = frame_b.astype(np.float32)
  alphas = ease_in_out(np.linspace(0, 1, n))
  return [((1 - α) * a + α * b).astype(np.uint8) for α in alphas]


def bridge_scenes(
  *scene_blocks: list[np.ndarray], crossfade_n: int = 10
) -> list[np.ndarray]:
  """Concatenate scene blocks with crossfades between each pair.

  Example:
      frames = bridge_scenes(title, panel0, panel1, panel2, panel3, title_reprise)

  Each scene block is a list of uint8 RGB frames (all same shape). Returns a
  single flat list of frames with n-frame crossfades woven between each pair.
  """
  out: list[np.ndarray] = []
  for block in scene_blocks:
    if not block:
      continue
    if out:
      out.extend(crossfade(out[-1], block[0], n=crossfade_n))
    out.extend(block)
  return out


# ────────── Ken Burns ──────────


def ken_burns_imshow(
  ax,
  img: np.ndarray,
  t: int,
  total: int,
  *,
  zoom_start: float = 1.0,
  zoom_end: float = 1.04,
  pan_fraction: tuple[float, float] = (0.0, 0.0),
) -> None:
  """Render img onto ax with progressive zoom (and optional pan).

  Use inside a per-frame loop when a nominally-static shot is held for more
  than ~0.5s. At t=0, the image is shown at zoom_start; at t=total-1, at
  zoom_end. The motion is eased (cubic ease-in-out).

  Args:
      ax: matplotlib Axes (should have axis("off") already).
      img: HxWx3 uint8 RGB image.
      t: current frame index (0-indexed).
      total: total number of frames this shot spans.
      zoom_start: zoom factor at frame 0 (1.0 = no zoom).
      zoom_end: zoom factor at frame total-1.
      pan_fraction: (dy_frac, dx_frac) — how much to pan across the total
          duration, as a fraction of image size. (0, 0) = pure zoom, no pan.
          Positive dy pans downward, positive dx pans rightward.
  """
  α = ease_in_out(t / max(total - 1, 1))
  zoom = zoom_start + α * (zoom_end - zoom_start)
  h, w = img.shape[:2]

  # Center-crop to (h/zoom, w/zoom)
  vh = int(h / zoom)
  vw = int(w / zoom)
  cy = (h - vh) // 2
  cx = (w - vw) // 2

  # Apply pan
  dy = int(α * pan_fraction[0] * h)
  dx = int(α * pan_fraction[1] * w)
  cy = int(np.clip(cy + dy, 0, h - vh))
  cx = int(np.clip(cx + dx, 0, w - vw))

  cropped = img[cy : cy + vh, cx : cx + vw]
  ax.imshow(cropped)
  ax.set_xlim(0, vw)
  ax.set_ylim(vh, 0)
  ax.axis("off")


# ────────── Trail subdivision ──────────


def interp_trail(positions: np.ndarray, n_subdivisions: int = 3) -> np.ndarray:
  """Insert n_subdivisions-1 interpolated points between each consecutive pair.

  Used to smooth trail draw-in in `render_minimap_with_trails`. Without this,
  trails grow one grid-cell per frame, which feels jerky at 20 fps on short
  rollouts.

  Example:
      positions.shape == (T, 2), n_subdivisions=3
      → returns shape ((T-1)*3 + 1, 2)

  Pass n_subdivisions=1 to disable (identity transform).
  """
  if n_subdivisions <= 1 or len(positions) < 2:
    return np.asarray(positions, dtype=np.float32)

  out = [np.asarray(positions[0], dtype=np.float32)]
  for a, b in zip(positions[:-1], positions[1:]):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    for i in range(1, n_subdivisions + 1):
      α = i / n_subdivisions
      out.append((1 - α) * a + α * b)
  return np.asarray(out)


# ────────── Lower-third gradient ──────────


def add_caption_gradient(
  fig,
  *,
  size: tuple[int, int] = (1280, 720),
  height_frac: float = 0.22,
  color: str | None = None,
) -> None:
  """Add a bottom-up gradient axis to `fig` that fades the image into the panel
  bg color. Caption text reads cleanly on the (near-opaque) bottom of the
  gradient while the image stays visible above.

  Args:
      fig: matplotlib Figure.
      size: (width_px, height_px) of the figure at 100 dpi.
      height_frac: fraction of the frame height the gradient occupies (0-1).
      color: hex string for the fade-to color. Defaults to DESIGN["bg"], so
          a dark theme fades to near-black and a cream theme fades to cream.
  """
  h_px = int(size[1] * height_frac)
  w_px = size[0]

  # Quadratic fall-off: opaque at the bottom, transparent at the top.
  ramp = np.linspace(0, 1, h_px) ** 2  # 0 at top of band, 1 at bottom
  r, g, b = _hex_to_rgb01(color if color is not None else DESIGN["bg"])
  img = np.zeros((h_px, w_px, 4), dtype=np.float32)
  img[..., 0] = r
  img[..., 1] = g
  img[..., 2] = b
  img[..., 3] = ramp[:, None]  # alpha channel

  ax_grad = fig.add_axes([0.0, 0.0, 1.0, height_frac], zorder=5)
  ax_grad.imshow(img, aspect="auto", origin="upper", extent=[0, w_px, 0, h_px])
  ax_grad.set_xlim(0, w_px)
  ax_grad.set_ylim(0, h_px)
  ax_grad.axis("off")


# ────────── Title card with three-tier hierarchy + fade ──────────


def title_card_frames(
  *,
  eyebrow: str | None,
  title: str,
  subtitle: str | None,
  size: tuple[int, int] = (1280, 720),
  fps: int = 20,
  duration_s: float = 1.4,
  fade_s: float = 0.25,
  fonts: dict | None = None,
  title_fontsize: int = 54,
) -> list[np.ndarray]:
  """Three-tier title card with eased fade-in and fade-out.

  Returns a list of uint8 RGB frames. Duration includes both fades.

  Hierarchy:
      eyebrow  — 14pt, uppercase, tracked out ~3px, muted color.
      title    — 68pt, bold, primary foreground.
      subtitle — 22pt, regular, secondary foreground.

  Any of the three can be None to hide that tier.

  Typical usage:
      frames = title_card_frames(
          eyebrow="MULTITASK PREPLAY · PNAS · 2026",
          title="One mechanism. Two puzzles.",
          subtitle="A single algorithm for RL generalization and hippocampal preplay.",
          size=(1280, 720), fps=20, duration_s=1.4, fade_s=0.25,
      )
  """
  import matplotlib.pyplot as plt

  if fonts is None:
    fonts = load_fonts()

  n_total = int(round(duration_s * fps))
  n_fade = int(round(fade_s * fps))
  n_hold = max(n_total - 2 * n_fade, 1)

  # Per-frame alpha schedule: fade in → hold → fade out
  alphas = np.concatenate(
    [
      ease_in_out(np.linspace(0, 1, n_fade)),
      np.ones(n_hold),
      ease_in_out(np.linspace(1, 0, n_fade)),
    ]
  )
  # Renormalize length to n_total
  if len(alphas) != n_total:
    alphas = np.interp(
      np.linspace(0, len(alphas) - 1, n_total),
      np.arange(len(alphas)),
      alphas,
    )

  frames: list[np.ndarray] = []
  for α in alphas:
    fig = plt.figure(
      figsize=(size[0] / 100, size[1] / 100), dpi=100, facecolor=DESIGN["bg"]
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(DESIGN["bg"])
    ax.axis("off")

    # Eyebrow (top), tracked-out caps. Matplotlib's Text doesn't expose a
    # letter-spacing property, so we emulate tracking by inserting two
    # spaces between every character — same visual effect, no plugin.
    # 18pt (was 14): 14pt at 1280px reads ~7pt on a 400px-wide phone preview,
    # below the visual-specs.md mobile floor.
    if eyebrow is not None:
      tracked = "  ".join(list(eyebrow.upper()))
      ax.text(
        0.5,
        0.68,
        tracked,
        ha="center",
        va="center",
        color=_rgba(DESIGN["muted"], α),
        fontsize=18,
        fontproperties=fonts["regular"],
      )

    # Title (center). Sized so the longest expected title clears the
    # 90% safe area at 1280px width (X/Bluesky crop the edges). When the
    # subtitle is omitted we vertically re-center the title under the eyebrow.
    title_y = 0.45 if subtitle is None else 0.55
    ax.text(
      0.5,
      title_y,
      title,
      ha="center",
      va="center",
      color=_rgba(DESIGN["fg"], α),
      fontsize=title_fontsize,
      fontproperties=fonts["bold"],
    )

    # Subtitle (below title)
    if subtitle is not None:
      ax.text(
        0.5,
        0.42,
        subtitle,
        ha="center",
        va="center",
        color=_rgba(DESIGN["subtitle_fg"], α),
        fontsize=22,
        fontproperties=fonts["regular"],
      )

    frames.append(_fig_to_rgb(fig, target_size=size))
    plt.close(fig)
  return frames


def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
  """Parse "#RRGGBB" → (r, g, b) in [0, 1]."""
  hex_color = hex_color.lstrip("#")
  return (
    int(hex_color[0:2], 16) / 255.0,
    int(hex_color[2:4], 16) / 255.0,
    int(hex_color[4:6], 16) / 255.0,
  )


def _rgba(hex_color: str, alpha: float) -> tuple[float, float, float, float]:
  """Convert "#RRGGBB" + alpha to matplotlib RGBA tuple."""
  r, g, b = _hex_to_rgb01(hex_color)
  return (r, g, b, float(np.clip(alpha, 0.0, 1.0)))


def _fig_to_rgb(fig, target_size: tuple[int, int] | None = None) -> np.ndarray:
  """Render a matplotlib Figure to uint8 RGB array.

  Reads physical pixel dims from the rgba buffer itself — `get_width_height()`
  returns logical (DPI-scaled) dims and would mis-shape on retina backends
  where the figure renders at 2× internally.

  If `target_size=(W, H)` is given and the rendered buffer is larger,
  downsample with PIL so the final frame is exactly the requested size.
  Necessary because libx264 + Twitter assume the encoded resolution is what
  the viewer sees, not 2× upscaled artwork.
  """
  fig.canvas.draw()
  rgba = np.asarray(fig.canvas.buffer_rgba())
  rgb = rgba[..., :3].copy()
  if target_size is not None:
    tw, th = target_size
    if rgb.shape[1] != tw or rgb.shape[0] != th:
      from PIL import Image

      rgb = np.asarray(Image.fromarray(rgb).resize((tw, th), Image.LANCZOS))
  if rgb.shape[0] % 2 or rgb.shape[1] % 2:
    rgb = rgb[: rgb.shape[0] // 2 * 2, : rgb.shape[1] // 2 * 2]
  return rgb


# ────────── Optional polish: film grain + vignette ──────────


def add_grain(frame: np.ndarray, intensity: float = 0.03, seed: int = 42) -> np.ndarray:
  """Add subtle monochrome film grain. intensity 0.02-0.04 works well."""
  rng = np.random.default_rng(seed)
  h, w = frame.shape[:2]
  noise = rng.normal(0, 1, size=(h, w))
  noise = (noise * intensity * 255).astype(np.int16)
  out = np.clip(frame.astype(np.int16) + noise[:, :, None], 0, 255).astype(np.uint8)
  return out


def add_vignette(frame: np.ndarray, strength: float = 0.15) -> np.ndarray:
  """Radial vignette. strength 0.10-0.20 feels premium without being heavy."""
  h, w = frame.shape[:2]
  y, x = np.ogrid[:h, :w]
  cy, cx = h / 2, w / 2
  # Normalized radial distance from center (0 at center, ~1 at corners)
  r = np.sqrt(((y - cy) / cy) ** 2 + ((x - cx) / cx) ** 2) / np.sqrt(2)
  mask = 1.0 - strength * ease_in_out(np.clip(r, 0, 1))
  out = (frame.astype(np.float32) * mask[:, :, None]).astype(np.uint8)
  return out


# ────────── Encode (Twitter-optimized ffmpeg params) ──────────

ENCODE_FFMPEG_PARAMS = [
  "-movflags",
  "+faststart",  # fast first-frame preview in X's player
  "-crf",
  "20",  # 18-23 is the sweet spot; 20 balances size/quality
  "-preset",
  "slow",  # ~30% smaller file at same quality
  "-profile:v",
  "high",
  "-level",
  "4.0",
]
"""Use as ffmpeg_params= kwarg to imageio.mimwrite. Example:

    imageio.mimwrite(
        "out.mp4", frames, fps=20, codec="libx264",
        macro_block_size=1, pixelformat="yuv420p",
        ffmpeg_params=ENCODE_FFMPEG_PARAMS,
    )
"""
