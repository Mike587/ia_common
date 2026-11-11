"""
Single-position CZIs: widefield snaps, LSM / Airyscan images and z-stacks.

POSITION — where the image is (SingleImage.frame, .stage_xy_m). Measured on
the CD7 (README "Decisions and measurements"):
  * The image centre is at the tile's own stage position (subblock metadata,
    StageXPosition/Y) + its RoiCenterOffset. That is the camera readout
    region's offset from the chip centre, or the LSM scan offset (=
    LaserScanInfo SampleOffset); 0 for a full chip / centred scan. It holds
    for Snap (the stage stays put, an off-centre ROI or scan images
    off-centre) and for camera experiments with a position (ZEN moves the
    stage by -offset so that the ROI centre lands on the position).
  * A rotated LSM scan: the image axes are turned by -RoiRotation (=
    -SampleRotation) against the stage axes, about the image centre; the
    offset is in stage axes (SingleImage.rotation_deg, ImageFrame.rotation_deg).
  * A real pixel box (API-run experiments) is used as it is, after checking
    that its centre agrees with that rule (within BOX_TOLERANCE_PX). It always
    did so far; a disagreement raises (e.g. a camera ROI in an API-run
    experiment, not seen yet).
  * Pixel box at (0, 0) (Snap, experiments started in ZEN, processed
    Airyscan): the frame is built from the rule (frame_source 'tile_stage').
    Without a stage position in the subblocks: the recorded scene centre
    ('scene_centre').
  * Not verified, so they raise: a camera ROI together with binning, a
    rotated camera image, a rotated scan with a real pixel box, and LSM tile
    values that disagree with LaserScanInfo.

AIRYSCAN — several data layouts exist and they are NOT interchangeable. Every
channel records its layout in ChannelType and its mode in AiryscanSettings/Mode
(channel_info / SingleImage.channel_types / .airyscan_modes).

Known and supported (CD7, ZEN 3.13, mode SuperResolution; Marc_SM_002):
  * UNPROCESSED, ZEN's default "4-channel format" (ZEN stores Airyscan data
    like this unless "Keep Airyscan Raw Data" is on):
      'AiryScanRingSheppardSum' — the un-# channel, 4 parts along the CZI "H"
          dimension = detector rings R0..R3 (signal falls from R0 to R3).
      'AiryScanSum'             — the '#' channel: sum of all detector
          elements, one H part (a confocal-like image with a wide pinhole).
    8-bit. This is the INPUT to ZEN's Airyscan processing, not a
    super-resolution image. SingleImage.airyscan_unprocessed is True.
  * The H x C grid is SPARSE (4 parts for the un-# channel, 1 for #). That
    breaks aicspylibczi 4.0.1 (it returns memory for planes that don't exist →
    segfault) and hence bioio-czi 3.x; pylibCZIrw reads it correctly.
  * read_single(h=0) gives ring R0 of the un-# channel plus the # channel —
    the data bioio-czi 2.x returned and Marc_SM_002 analysed (it used DAPI#).

  * PROCESSED ("… -Airyscan Processing.czi", ZEN Airyscan processing,
    seen 2026-09-30): no H dimension, one channel per dye (the # channels are
    dropped), 16-bit, no ChannelType (AiryscanSettings/Mode is kept), and a
    Processing/Airyscan block (deconvolution type, strength per channel).
    SingleImage.airyscan_processed is True.
    !! Processing crops a 12-px border (962 -> 938) AND resets the pixel box to
    (0, 0), so the pixel frame no longer carries the stage position. The frame
    is then built from the tile's stage position, which processing keeps
    (frame_source = 'tile_stage'): the crop is symmetric (DAPI: exactly
    raw[12:950, 12:950]; AF568 one row different = 0.07 µm), so the image
    centre is still where the stage was.

Not yet seen — need a test file before they are supported:
  * The 32-channel raw format ("Keep Airyscan Raw Data", needs the Airyscan
    processing licence): presumably 32 H parts = detector elements.
  * Multiplex modes (SR-4Y / SR-8Y / CO-2Y / CO-8Y): several image lines per
    sweep, so the raw data differs from SR mode.
  * Sheppard-sum export: an extra file with one summed channel.
  A file WITH an H dimension whose ChannelType is not a known layout raises
  MetadataError instead of guessing what H means.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pylibCZIrw import czi as pyczi

from ..coords import ImageFrame
from . import meta
from ._pyczi import Rect, read_plane, subblocks
from ._subblocks import SubBlockMeta, read_subblock_metadata

# A real pixel box must agree with tile stage position + ROI offset within this.
# Measured 2026-10-14: 0.27-0.50 px on the CD7 camera, LSM and Airyscan files.
BOX_TOLERANCE_PX = 1.0
# The subblocks of one image share one stage position (spread 0 on all test files).
STAGE_SPREAD_M = 0.1e-6


# ChannelType values of the unprocessed 4-channel format (see module docstring).
UNPROCESSED_AIRYSCAN_TYPES = {
    "AiryScanRingSheppardSum": "detector rings R0..R3 along H",
    "AiryScanSum": "sum of all detector elements (the '#' channel)",
}
# Layouts whose H dimension we understand.
KNOWN_H_LAYOUTS = set(UNPROCESSED_AIRYSCAN_TYPES)


@dataclass
class SingleImage:
    path: Path
    data: np.ndarray                 # (C, Z, Y, X), native dtype (uint8 / uint16)
    channel_names: list[str]
    pixel_size_m: float
    z_rel_m: np.ndarray              # per plane, relative to the stack centre
    frame: ImageFrame                # image pixel -> stage XY
    h: int                           # which H part was read (Airyscan ring)
    n_h: int                         # H parts in the file (1 = no H dimension)
    channel_types: list[str | None] = field(default_factory=list)   # ChannelType per channel
    airyscan_modes: list[str | None] = field(default_factory=list)  # e.g. 'SuperResolution'; None = not Airyscan
    airyscan_processed: bool = False  # result of ZEN Airyscan processing
    frame_source: str = "pixel_frame"  # 'pixel_frame', 'tile_stage' or 'scene_centre' (module docstring)
    tile_stage_xy_m: tuple[float, float] | None = None   # where the stage was (subblock metadata)
    roi_offset_m: tuple[float, float] = (0.0, 0.0)       # ROI / scan centre offset (RoiCenterOffset)
    rotation_deg: float = 0.0        # image axes vs. stage axes (+: +X towards +Y); -RoiRotation
    geometry_verified: bool = True   # False: read with allow_unverified, positions not trustworthy
    recorded_stage_xy_m: tuple[float, float] | None = None
    recorded_focus_m: float | None = None
    metadata: object = field(default=None, repr=False)   # parsed XML root

    @property
    def stage_xy_m(self) -> tuple[float, float]:
        """Stage XY of the image CENTRE. For a snapped off-centre camera ROI this
        is not where the stage was (that is tile_stage_xy_m)."""
        _, _, h, w = self.data.shape
        return self.frame.to_stage_m((h - 1) / 2, (w - 1) / 2)

    @property
    def airyscan_unprocessed(self) -> bool:
        """True if any channel is unprocessed Airyscan data (ring format / sum) —
        not a super-resolution image yet."""
        return any(t in UNPROCESSED_AIRYSCAN_TYPES for t in self.channel_types)

    def channel(self, name: str) -> np.ndarray:
        """(Z, Y, X) data of the channel with this name."""
        try:
            return self.data[self.channel_names.index(name)]
        except ValueError:
            raise KeyError(f"no channel {name!r}; channels: {self.channel_names}") from None


def read_single(path: str | Path, h: int = 0, allow_unverified: bool = False) -> SingleImage:
    """Read a single-position CZI (all channels and Z planes).

    h: which part along the H dimension to read (Airyscan ring R0..R3 in the
    4-channel format). Channels that have no data at this H are left out
    (their names are dropped too).
    allow_unverified: read an image whose geometry is not verified (camera ROI
    + binning, rotated camera image, LSM tile values that disagree with
    LaserScanInfo; see the module docstring) instead of raising. It is placed by
    the rule anyway and geometry_verified is False: use the pixels, not the
    positions.
    Raises MetadataError for multi-scene files (use czi.mosaic), for an H
    dimension whose Airyscan layout is not known, and for an unverified
    geometry unless allowed.
    """
    path = Path(path)
    with pyczi.open_czi(str(path)) as reader:
        root = meta.parse(reader.raw_metadata)
        if len(reader.scenes_bounding_rectangle_no_pyramid) > 1:
            raise meta.MetadataError(f"{path.name}: {len(reader.scenes_bounding_rectangle_no_pyramid)} "
                                     f"scenes — not a single-position image (use czi.mosaic)")
        bb = reader.total_bounding_box_no_pyramid   # pyramid levels pad the box
        rect = Rect(bb["X"][0], bb["Y"][0], bb["X"][1] - bb["X"][0], bb["Y"][1] - bb["Y"][0])
        n_c = bb.get("C", (0, 1))[1] - bb.get("C", (0, 1))[0]
        n_z = bb.get("Z", (0, 1))[1] - bb.get("Z", (0, 1))[0]
        has_h = "H" in bb
        n_h = bb["H"][1] - bb["H"][0] if has_h else 1
        if not 0 <= h < n_h:
            raise ValueError(f"{path.name}: h={h} but the file has {n_h} H part(s)")

        info = meta.channel_info(root)
        processed = root.find(".//Processing/Airyscan") is not None and not has_h
        if n_h > 1:     # a single H part (plain LSM images on the CD7) is unambiguous
            unknown = sorted({str(ci.channel_type) for ci in info} - KNOWN_H_LAYOUTS)
            if unknown:
                raise meta.MetadataError(
                    f"{path.name}: H dimension ({n_h} parts) with ChannelType {unknown} — "
                    f"an Airyscan layout ia_common does not know yet (known: "
                    f"{sorted(KNOWN_H_LAYOUTS)}). Not guessing what H means; add a test "
                    f"file and support it in czi/image.py.")
        present = {(sb.coord.get("H", 0), sb.coord.get("C", 0)) for sb in subblocks(reader)}
        names_all = [ci.name for ci in info]
        chans = [c for c in range(n_c) if (h, c) in present]
        if not chans:
            raise meta.MetadataError(f"{path.name}: no channel has data at H={h}")

        planes = []
        for c in chans:
            stack = []
            for z in range(n_z):
                plane = {"C": c, "Z": z}
                if has_h:
                    plane["H"] = h
                stack.append(read_plane(reader, rect, plane))
            planes.append(np.stack(stack))
        data = np.stack(planes)

    ps = meta.pixel_size_m(root)
    sbs = [s for s in read_subblock_metadata(path) if not s.pyramid]
    modes = [info[c].acquisition_mode if c < len(info) else None for c in chans]
    verified = True
    try:
        check_geometry(path.name, root, sbs, modes)
    except meta.MetadataError:
        if not allow_unverified:
            raise
        verified = False
    frame, source, tile_stage, roi_offset, rotation = _place(path.name, root, rect, ps, sbs)
    return SingleImage(
        path=path,
        data=data,
        channel_names=[names_all[c] if c < len(names_all) else f"C{c}" for c in chans],
        pixel_size_m=ps,
        z_rel_m=meta.z_relative_m(root, n_z),
        frame=frame,
        frame_source=source,
        tile_stage_xy_m=tile_stage,
        roi_offset_m=roi_offset,
        rotation_deg=rotation,
        geometry_verified=verified,
        airyscan_processed=processed,
        h=h,
        n_h=n_h,
        channel_types=[info[c].channel_type if c < len(info) else None for c in chans],
        airyscan_modes=[info[c].airyscan_mode if c < len(info) else None for c in chans],
        recorded_stage_xy_m=meta.recorded_stage_xy_m(root),
        recorded_focus_m=meta.recorded_focus_m(root),
        metadata=root,
    )


def _um(xy) -> str:
    return f"({xy[0] * 1e6:.3f}, {xy[1] * 1e6:.3f}) µm"


def _same_geometry(a, b) -> bool:
    """(offset x m, offset y m, rotation deg) equal up to their rounding in the
    file (tile tags: 0.1 nm and 0.0001 deg)."""
    return abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9 and abs(a[2] - b[2]) < 1e-3


def check_geometry(name: str, root, sbs: list[SubBlockMeta], modes: list[str | None]) -> None:
    """Raise MetadataError for an acquisition geometry whose placement is not
    verified (module docstring): a camera ROI with binning, a rotated camera
    image, LSM tile values that disagree with LaserScanInfo. Also used by
    czi.mosaic."""
    offsets = sorted({s.roi_offset_m for s in sbs} - {(0.0, 0.0)})
    rotations = sorted({s.roi_rotation for s in sbs} - {0.0})
    widefield = bool(modes) and all(m == "WideField" for m in modes)
    if not widefield:
        # The tile tags and LaserScanInfo record the same scan offset and
        # rotation (2026-10-14); if they don't, the tags may mean something else.
        tiles = {(s.roi_offset_m[0], s.roi_offset_m[1], s.roi_rotation) for s in sbs}
        lsm = {g for g in meta.lsm_scan_geometry(root) if any(g)}
        if not all(any(_same_geometry(a, b) for b in tiles) for a in lsm) or \
                not all(any(_same_geometry(a, b) for b in lsm) for a in tiles if any(a)):
            raise meta.MetadataError(
                f"{name}: tile scan offset/rotation {sorted(tiles)} disagree with LaserScanInfo "
                f"{sorted(lsm)} (m, m, deg) — not verified for this case")
        return
    if rotations:
        raise meta.MetadataError(f"{name}: rotated camera image (RoiRotation {rotations}) — not "
                                 f"verified yet. Test file wanted.")
    for o in offsets:
        binning = meta.camera_binning(root, o)
        if binning is None:
            raise meta.MetadataError(f"{name}: ROI offset {_um(o)} is in no camera's settings")
        if binning.replace(" ", "") != "1,1":
            raise meta.MetadataError(f"{name}: camera ROI with binning {binning} — not verified yet. "
                                     f"Test file wanted.")


def _place(name: str, root, rect: Rect, ps: float,
           sbs: list[SubBlockMeta]) -> tuple[ImageFrame, str, tuple | None, tuple, float]:
    """-> (frame, frame_source, tile_stage_xy_m, roi_offset_m, rotation_deg); see
    the module docstring. The caller has run check_geometry."""
    stages = np.array([s.stage_xy_m for s in sbs if s.stage_xy_m is not None])
    if len(stages) not in (0, len(sbs)):
        raise meta.MetadataError(f"{name}: only some subblocks record a stage position")
    if len(stages) and np.ptp(stages, axis=0).max() > STAGE_SPREAD_M:
        raise meta.MetadataError(f"{name}: subblocks taken at different stage positions (spread "
                                 f"{np.ptp(stages, axis=0).max() * 1e6:.2f} µm) — not a single position")
    offsets = {s.roi_offset_m for s in sbs}
    rotations = {s.roi_rotation for s in sbs}
    if len(offsets) > 1 or len(rotations) > 1:
        raise meta.MetadataError(f"{name}: subblocks with different ROI offsets "
                                 f"{[_um(o) for o in offsets]} / rotations {sorted(rotations)}")
    offset = next(iter(offsets), (0.0, 0.0))
    # image axes -> stage axes: minus the recorded scan rotation (measured 2026-10-14)
    rotation = -next(iter(rotations), 0.0) + 0.0
    stage = (float(stages[:, 0].mean()), float(stages[:, 1].mean())) if len(stages) else None
    if stage is None and (offset != (0.0, 0.0) or rotation):
        raise meta.MetadataError(f"{name}: ROI offset {_um(offset)} / rotation {-rotation} deg "
                                 f"but no stage position in the subblocks")

    if stage is not None:
        cx, cy = stage[0] + offset[0], stage[1] + offset[1]          # the image centre
    if (rect.x, rect.y) != (0, 0):                  # a real pixel box: used as it is, once checked
        if rotation:
            raise meta.MetadataError(f"{name}: rotated scan ({-rotation} deg) with a real pixel box "
                                     f"— not verified yet. Test file wanted.")
        if stage is not None:
            d = math.hypot((rect.x + rect.w / 2) * ps - cx, (rect.y + rect.h / 2) * ps - cy)
            if d > BOX_TOLERANCE_PX * ps:
                raise meta.MetadataError(
                    f"{name}: the pixel box centre is {d * 1e6:.2f} µm from the tile's stage position "
                    f"+ ROI offset {_um(offset)} — a case ia_common has not verified (e.g. a camera "
                    f"ROI in an API-run experiment). Not guessing which one is right: check it with "
                    f"tools/check_placement.py and add a test file.")
        return ImageFrame(rect.x, rect.y, 1.0, ps), "pixel_frame", stage, offset, 0.0
    if stage is not None:
        return (ImageFrame.centred_on(cx, cy, rect.w, rect.h, ps, rotation), "tile_stage",
                stage, offset, rotation)
    # No stage position in the subblocks: the recorded scene centre.
    centres = meta.planned_scene_centres_m(root)
    if len(centres) != 1:
        raise meta.MetadataError(f"{name}: pixel box starts at (0, 0), no stage position in the "
                                 f"subblocks and no single scene centre — the position is unknown")
    sx, sy = next(iter(centres.values()))
    return ImageFrame.centred_on(sx, sy, rect.w, rect.h, ps), "scene_centre", None, offset, 0.0
