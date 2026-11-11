"""Thin helpers around pylibCZIrw (Zeiss' CZI library) shared by image / mosaic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pylibCZIrw import czi as pyczi

from .meta import MetadataError


@dataclass(frozen=True)
class Rect:
    """Rectangle in the GLOBAL pixel frame (x, y = top-left corner)."""
    x: int
    y: int
    w: int
    h: int

    @classmethod
    def of(cls, r) -> "Rect":
        return cls(int(r.x), int(r.y), int(r.w), int(r.h))

    @property
    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def centre_px(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2


@dataclass(frozen=True)
class SubBlock:
    coord: dict            # e.g. {'C': 0, 'Z': 3, 'H': 0, 'S': 1}
    rect: Rect
    m: int | None          # mosaic tile index, if valid


def subblocks(reader: pyczi.CziReader, full_resolution_only: bool = True) -> list[SubBlock]:
    """All subblocks (header info only — no pixels). Pyramid levels are skipped
    unless full_resolution_only=False."""
    out: list[SubBlock] = []

    def cb(_i, info):
        if full_resolution_only and abs(info.get_zoom() - 1.0) > 1e-6:
            return True
        m = info.mIndex if info.is_mindex_valid() else None
        out.append(SubBlock(info.coordinate.to_dict(), Rect.of(info.logicalRect), m))
        return True

    reader.enumerate_subblocks(cb)
    return out


def read_plane(reader: pyczi.CziReader, rect: Rect, plane: dict, zoom: float = 1.0) -> np.ndarray:
    """One 2D plane (native dtype). Areas without data read as 0."""
    a = reader.read(roi=rect.as_tuple, plane=plane, zoom=zoom)
    if a.ndim == 3:
        if a.shape[2] != 1:
            raise MetadataError(f"colour pixel type ({a.shape[2]} components) is not supported")
        a = a[:, :, 0]
    return a
