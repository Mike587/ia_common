"""
Pixel <-> stage coordinates — the ONE convention for all ia projects.

Verified on the CD7 (2026-09-30) for the 5x OV mosaics (4 files, 22 scenes)
and the 40x camera, LSM and Airyscan images:

  * A CZI's GLOBAL pixel frame is the stage frame:
        stage_m = global_px * pixel_size_m          (no offset)
    OV tiles: fitted offset within 0.2 µm of zero; single images: bounding-box
    centre * pixel size = commanded stage position within 0.1 µm.
  * X follows the column, Y follows the row.
  * Pixels are areas: pixel i covers [i, i + 1), so its CENTRE is at i + 0.5.
    (Older ia code used `col - width/2`, i.e. half a pixel off.)
  * Everything physical is in METRES.

An image read from a CZI is described by an ImageFrame: where its pixel (0, 0)
sits in the global frame and how many global pixels one image pixel spans
(> 1 for a downsampled read).

Rotated LSM scans (2026-10-14): the image axes are turned against the stage
axes by -RoiRotation (= -LaserScanInfo SampleRotation), about the image centre.
ImageFrame.rotation_deg carries that; it is 0 for everything else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _rotate(u, v, deg: float):
    """(u, v) turned by deg, from +X towards +Y."""
    if not deg:
        return u, v
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return c * u - s * v, s * u + c * v


@dataclass(frozen=True)
class ImageFrame:
    """Maps pixels of one read image to stage XY.

    origin_x_px, origin_y_px — global pixel coords of the image's top-left corner
    step_px                  — global pixels per image pixel (1 = full resolution)
    pixel_size_m             — size of one GLOBAL (full-resolution) pixel
    rotation_deg             — image axes against the stage axes, about the
                               top-left corner; + turns from +X towards +Y.
                               0 except for rotated LSM scans.
    """
    origin_x_px: float
    origin_y_px: float
    step_px: float
    pixel_size_m: float
    rotation_deg: float = 0.0

    @classmethod
    def centred_on(cls, x_m: float, y_m: float, width: int, height: int, pixel_size_m: float,
                   rotation_deg: float = 0.0) -> "ImageFrame":
        """Frame of a width x height full-resolution image whose centre is at
        stage (x_m, y_m)."""
        du, dv = _rotate(width / 2, height / 2, rotation_deg)
        return cls(x_m / pixel_size_m - du, y_m / pixel_size_m - dv, 1.0, pixel_size_m, rotation_deg)

    @property
    def image_pixel_size_m(self) -> float:
        """Physical size of one pixel of the read image."""
        return self.step_px * self.pixel_size_m

    def to_stage_m(self, row, col):
        """Image pixel (row, col) -> stage (x_m, y_m). Accepts scalars or arrays."""
        row, col = np.asarray(row, float), np.asarray(col, float)
        u, v = _rotate((col + 0.5) * self.step_px, (row + 0.5) * self.step_px, self.rotation_deg)
        x = (self.origin_x_px + u) * self.pixel_size_m
        y = (self.origin_y_px + v) * self.pixel_size_m
        return (float(x), float(y)) if x.ndim == 0 else (x, y)

    def to_pixel(self, x_m, y_m):
        """Stage (x_m, y_m) -> image pixel (row, col), continuous (inverse of to_stage_m)."""
        x_m, y_m = np.asarray(x_m, float), np.asarray(y_m, float)
        u, v = _rotate(x_m / self.pixel_size_m - self.origin_x_px,
                       y_m / self.pixel_size_m - self.origin_y_px, -self.rotation_deg)
        col = u / self.step_px - 0.5
        row = v / self.step_px - 0.5
        return (float(row), float(col)) if row.ndim == 0 else (row, col)


def offset_from_centre_m(row, col, shape: tuple[int, int], pixel_size_m: float,
                         rotation_deg: float = 0.0):
    """Pixel (row, col) -> (dx_m, dy_m) from the image centre, in stage axes.

    For single images this is the move that brings (row, col) to where the
    image centre was. rotation_deg: SingleImage.rotation_deg of a rotated scan.
    """
    h, w = shape[-2:]
    row, col = np.asarray(row, float), np.asarray(col, float)
    dx, dy = _rotate((col + 0.5 - w / 2) * pixel_size_m, (row + 0.5 - h / 2) * pixel_size_m,
                     rotation_deg)
    return (float(dx), float(dy)) if np.ndim(dx) == 0 else (dx, dy)
