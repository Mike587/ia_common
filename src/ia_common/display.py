"""
Scale bars and image saving — every saved microscopy image gets a scale bar.

  add_scalebar(ax, pixel_size_m)   — on any matplotlib axes showing an image
  save_png(path, img, pixel_size_m) — image at native resolution, always with
                                      a scale bar; optional overlay callback
  nice_length_m / format_length    — round bar lengths (1-2-5 series) and labels

pixel_size_m is the size of one pixel OF THE SHOWN IMAGE, in metres — for a
downsampled mosaic read that is frame.image_pixel_size_m, not the full-res size.

Replaces the three ia_Marc variants (common.add_scalebar, make_poster_faz_views
add_scalebar_no_label, make_poster_overview draw_scalebar).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np


def nice_length_m(target_m: float) -> float:
    """The 1-2-5 series value (…, 10, 20, 50, 100 µm, …) closest to target_m
    (linear distance, as the old ia_Marc scale bars)."""
    if not target_m > 0:
        raise ValueError(f"target length must be > 0, got {target_m}")
    e = math.floor(math.log10(target_m))
    candidates = [m * 10.0 ** k for k in (e - 1, e, e + 1) for m in (1, 2, 5)]
    return min(candidates, key=lambda v: abs(v - target_m))


def format_length(length_m: float) -> str:
    """'500 nm', '50 µm', '2 mm'."""
    for unit, f in (("mm", 1e-3), ("µm", 1e-6), ("nm", 1e-9)):
        if length_m >= f * (1 - 1e-9):
            return f"{length_m / f:g} {unit}"
    return f"{length_m / 1e-9:g} nm"


def add_scalebar(ax, pixel_size_m: float, *, length_m: float | None = None, frac: float = 0.2,
                 label: bool = True, color: str = "white", box: bool = True,
                 loc: str = "lower right", fontsize: float | None = None):
    """Add a scale bar to a matplotlib axes that shows an image (imshow).

    pixel_size_m — physical size of one pixel of the shown image (m)
    length_m     — bar length; default: the 1-2-5 value closest to frac x image width
    label        — write the length ('50 µm') above the bar
    box          — semi-transparent dark box behind it (readable on any image)

    Works with the default imshow extent (pixels) and with a custom extent
    (e.g. µm): the bar is converted into the axes' data units. Returns the
    AnchoredSizeBar artist and the bar length in metres.
    """
    from matplotlib import font_manager
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    if not ax.images:
        raise ValueError("add_scalebar needs an image on the axes (call imshow first)")
    im = ax.images[-1]
    rows, cols = im.get_array().shape[:2]
    left, right, bottom, top = im.get_extent()
    data_per_px_x = abs(right - left) / cols
    data_per_px_y = abs(top - bottom) / rows
    width_m = cols * pixel_size_m
    length_m = nice_length_m(frac * width_m) if length_m is None else length_m
    size = length_m / pixel_size_m * data_per_px_x
    thickness = max(rows * 0.008, 2) * data_per_px_y
    if fontsize is None:                      # scale with the figure's pixel height
        fig_h_px = ax.figure.get_size_inches()[1] * ax.figure.dpi
        fontsize = max(8.0, min(24.0, fig_h_px * 0.025 * 72 / ax.figure.dpi))
    bar = AnchoredSizeBar(ax.transData, size, format_length(length_m) if label else "",
                          loc, pad=0.4, borderpad=0.6, sep=4, color=color,
                          frameon=box, size_vertical=thickness, label_top=True,
                          fontproperties=font_manager.FontProperties(size=fontsize, weight="bold"))
    if box:
        bar.patch.set(facecolor="black", alpha=0.45, edgecolor="none")
    ax.add_artist(bar)
    return bar, length_m


def _auto_limits(img: np.ndarray, pct=(0.5, 99.8)) -> tuple[float, float]:
    finite = img[np.isfinite(img)] if np.issubdtype(img.dtype, np.floating) else img
    lo, hi = np.percentile(finite, pct) if finite.size else (0.0, 1.0)
    return float(lo), float(hi if hi > lo else lo + 1)


def save_png(path: str | Path, img: np.ndarray, pixel_size_m: float, *,
             cmap: str = "gray", clim: tuple[float, float] | None = None,
             overlay: Callable | None = None, title: str | None = None,
             scalebar_kw: dict | None = None, dpi: int = 100) -> Path:
    """Save a 2D (or RGB) image as PNG at native resolution, WITH a scale bar.

    One image pixel = one PNG pixel. clim: display range (default: 0.5–99.8
    percentiles; ignored for RGB). overlay(ax): draw markers, outlines etc. in
    image pixel coordinates before saving. scalebar_kw: passed to add_scalebar.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    path = Path(path)
    img = np.asarray(img)
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    if img.ndim == 2:
        lo, hi = clim if clim is not None else _auto_limits(img)
        ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    else:
        ax.imshow(img, interpolation="nearest")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    if overlay is not None:
        overlay(ax)
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
    if title:
        ax.text(0.01, 0.99, title, transform=ax.transAxes, va="top", ha="left",
                color="white", fontsize=max(8, h * 0.02 * 72 / dpi),
                bbox=dict(facecolor="black", alpha=0.45, edgecolor="none"))
    add_scalebar(ax, pixel_size_m, **(scalebar_kw or {}))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
