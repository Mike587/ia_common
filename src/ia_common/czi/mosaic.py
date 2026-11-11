"""
Tile-region mosaics (ZEN tile experiments): one CZI, one scene per tile
region (scene name = region name, e.g. the well), each scene made of tiles.

Positions come from the pixel frame (ia_common.coords): stage = global px *
pixel size. Verified on the CD7 5x OVs (2026-09-30: 4 files, 22 scenes, each
tile's own recorded stage position within 0.2 µm of this). So tiles need no
stitching and no per-tile metadata: every tile already sits at its stage
position in the global frame.

Downsampling: `downsample=k` turns every k x k block of pixels into one
pixel (image k times smaller per axis, pixels k times larger). It is EXACT by
default: full resolution read in bands, blocks averaged here. libCZI's own
shrinking ("zoom") misplaces content by up to ~1 output pixel (see
read_region) and is only an opt-in. Areas not covered by a tile read as 0.

Replaces: ia_Marc_002 czi_mosaic.MosaicCzi and the downsampled readers of
ia_Marc spheroid_analysis / ia_NfS tissue_analysis (open_mosaic,
scale_for_target, mosaic_z_count, read_dapi_downsampled[_proj]).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pylibCZIrw import czi as pyczi

from ..coords import ImageFrame
from . import meta
from ._pyczi import Rect, read_plane, subblocks
from ._subblocks import read_subblock_metadata
from .image import _um, check_geometry


@dataclass
class Tile:
    rect: Rect                       # global pixel frame
    m: int | None                    # mosaic index, if the file has one


@dataclass
class Scene:
    index: int
    name: str
    rect: Rect                       # bounding box of all its tiles (global px)
    tiles: list[Tile] = field(default_factory=list)


class Mosaic:
    """Open a mosaic CZI. Use as a context manager, or call close()."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._reader = pyczi.CziReader(str(self.path))
        self.metadata = meta.parse(self._reader.raw_metadata)
        self.pixel_size_m = meta.pixel_size_m(self.metadata)
        self.channel_names = meta.channel_names(self.metadata)
        bb = self._reader.total_bounding_box_no_pyramid   # pyramid levels pad the box
        self.n_c = bb.get("C", (0, 1))[1] - bb.get("C", (0, 1))[0]
        self.n_z = bb.get("Z", (0, 1))[1] - bb.get("Z", (0, 1))[0]
        self.rect = Rect(bb["X"][0], bb["Y"][0], bb["X"][1] - bb["X"][0], bb["Y"][1] - bb["Y"][0])
        self._check_geometry()
        self.scenes = self._scenes()

    def _check_geometry(self) -> None:
        """Tile positions are verified for full-chip, unrotated tiles only
        (README): an ROI / scan offset or a rotation raises."""
        sbs = [s for s in read_subblock_metadata(self.path) if not s.pyramid]
        offsets = sorted({s.roi_offset_m for s in sbs} - {(0.0, 0.0)})
        rotations = sorted({s.roi_rotation for s in sbs} - {0.0})
        if offsets or rotations:
            self.close()
            raise meta.MetadataError(
                f"{self.path.name}: tiles with an ROI / scan offset {[_um(o) for o in offsets]} or "
                f"rotation {rotations} — tile placement is not verified for that. Test file wanted.")
        try:
            check_geometry(self.path.name, self.metadata, sbs,
                           [ci.acquisition_mode for ci in meta.channel_info(self.metadata)])
        except meta.MetadataError:
            self.close()
            raise

    # -- context manager ---------------------------------------------------
    def close(self) -> None:
        self._reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- geometry ----------------------------------------------------------
    def _scenes(self) -> list[Scene]:
        names = meta.scene_names(self.metadata)
        rects = {int(k): Rect.of(v) for k, v in self._reader.scenes_bounding_rectangle_no_pyramid.items()}
        if not rects:                                  # no S dimension: one scene
            rects = {0: self.rect}
        scenes = {i: Scene(i, names.get(i, f"S{i}"), r) for i, r in sorted(rects.items())}
        # One plane's subblocks = the tiles (first channel, first Z, H0).
        sbs = subblocks(self._reader)
        c0 = min((sb.coord.get("C", 0) for sb in sbs), default=0)
        for sb in sbs:
            if sb.coord.get("C", 0) != c0 or sb.coord.get("Z", 0) != 0 or sb.coord.get("H", 0) != 0:
                continue
            s = sb.coord.get("S", 0) if len(scenes) > 1 else next(iter(scenes))
            if s in scenes:
                scenes[s].tiles.append(Tile(sb.rect, sb.m))
        for sc in scenes.values():
            sc.tiles.sort(key=lambda t: (t.m is None, t.m, t.rect.y, t.rect.x))
        return list(scenes.values())

    def scene(self, name_or_index) -> Scene:
        for sc in self.scenes:
            if name_or_index in (sc.name, sc.index):
                return sc
        raise KeyError(f"no scene {name_or_index!r}; scenes: {[s.name for s in self.scenes]}")

    def centre_stage_m(self, rect: Rect) -> tuple[float, float]:
        """Stage XY (m) of a rectangle's centre (scene or tile)."""
        cx, cy = rect.centre_px()
        return cx * self.pixel_size_m, cy * self.pixel_size_m

    def downsample_for(self, target_pixel_size_m: float) -> int:
        """Downsample factor k (>= 1) whose pixel size (k x full-res pixel size)
        is closest to the target."""
        return max(1, round(target_pixel_size_m / self.pixel_size_m))

    # -- pixels ------------------------------------------------------------
    def read_region(self, rect: Rect, downsample: int = 1, c: int = 0,
                    z: int | None = None, method: str = "exact") -> tuple[np.ndarray, ImageFrame]:
        """One plane of a global-pixel rectangle -> (2D image, its ImageFrame).

        downsample=k (integer >= 1): every k x k block of pixels becomes one
            pixel, e.g. from downsample_for(). 1 = full resolution.
        method="exact" (default): read at full resolution in bands and average
            the k x k blocks (float32). The frame is exact: image pixel (r, c) is
            the mean of global pixels [x + c*k, x + (c+1)*k) x [y + r*k, y + (r+1)*k).
            The rectangle is cropped to a multiple of k.
        method="libczi": libCZI's own shrinking (its "zoom", = 1/k; fast, native
            dtype, uses the file's pyramid). Its content is misplaced by up to ~1
            OUTPUT pixel, varying across the image (measured 2026-09-30 on a CD7
            OV: up to 1.1 µm at k=2, 1.8 µm at k=4, 5.8 µm at k=8, 8.3 µm at k=10;
            aicspylibczi gives the identical result — both use libCZI). Fine for
            display, not for positions.
        """
        plane = {"C": c}
        if self.n_z > 1:
            if z is None:
                raise ValueError(f"{self.path.name} has {self.n_z} Z planes — pass z "
                                 f"(or use read_max_projection)")
            plane["Z"] = z
        k = downsample
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError(f"downsample must be an integer >= 1, got {downsample!r}")
        k = int(k)
        if k == 1:
            return read_plane(self._reader, rect, plane), ImageFrame(rect.x, rect.y, 1.0, self.pixel_size_m)
        if method == "libczi":
            img = read_plane(self._reader, rect, plane, zoom=1.0 / k)
            return img, ImageFrame(rect.x, rect.y, rect.w / img.shape[1], self.pixel_size_m)
        if method != "exact":
            raise ValueError(f"method must be 'exact' or 'libczi', got {method!r}")
        nw, nh = rect.w // k, rect.h // k
        if nw == 0 or nh == 0:
            raise ValueError(f"rectangle {rect} is smaller than one {k} x {k} block")
        out = np.empty((nh, nw), np.float32)
        band = max(1, 2048 // k)                       # output rows per full-res read
        for r0 in range(0, nh, band):
            nr = min(band, nh - r0)
            a = read_plane(self._reader, Rect(rect.x, rect.y + r0 * k, nw * k, nr * k), plane)
            out[r0:r0 + nr] = a.reshape(nr, k, nw, k).mean(axis=(1, 3), dtype=np.float32)
        return out, ImageFrame(rect.x, rect.y, float(k), self.pixel_size_m)

    def read_scene(self, scene: Scene | str | int, downsample: int = 1, c: int = 0,
                   z: int | None = None, method: str = "exact") -> tuple[np.ndarray, ImageFrame]:
        sc = scene if isinstance(scene, Scene) else self.scene(scene)
        return self.read_region(sc.rect, downsample, c, z, method)

    def read_all(self, downsample: int = 1, c: int = 0,
                 z: int | None = None, method: str = "exact") -> tuple[np.ndarray, ImageFrame]:
        """The whole mosaic (all scenes) in one image."""
        return self.read_region(self.rect, downsample, c, z, method)

    def read_max_projection(self, scene: Scene | str | int | None = None, downsample: int = 1,
                            c: int = 0, method: str = "exact") -> tuple[np.ndarray, ImageFrame]:
        """Max projection over Z (a single plane for 2D mosaics) — robust to
        where focus landed. scene=None: whole mosaic."""
        rect = self.rect if scene is None else (
            scene.rect if isinstance(scene, Scene) else self.scene(scene).rect)
        if self.n_z <= 1:
            return self.read_region(rect, downsample, c, method=method)
        frame, out = None, None
        for z in range(self.n_z):
            img, frame = self.read_region(rect, downsample, c, z, method)
            out = img if out is None else np.maximum(out, img)
        return out, frame
