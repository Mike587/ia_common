"""
Metadata from a CZI's XML (parse it with `parse(raw_xml)`; pylibCZIrw gives
the raw XML as `CziReader.raw_metadata`).

Rules:
  * Everything returned is in METRES (the XML stores µm for positions and
    metres for Scaling).
  * No silent guesses. A value that is needed but missing raises
    MetadataError; a value that is merely informative returns None.

Positions — which one to trust (checked on the CD7, 2026-09-30):
  * Where an image or tile is: the pixel frame (see ia_common.coords). The
    readers in czi.image / czi.mosaic use it.
  * recorded_stage_xy_m (MTBStageAxisX/Y): the encoder reading stored with the
    image. It agreed with the commanded position to ~0.1 µm, but it is ONE value
    per file (useless for mosaics) and flagged IsPrecise=false. Informative only.
  * planned_scene_centres_m (Scene/CenterPosition): the PLANNED centre. For a
    single position it is the commanded XY; for a tile region it is the region
    centre from the experiment, not necessarily where the tiles landed. The
    Marc_SM project found single-tile scene centres unreliable. Informative only.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np


class MetadataError(ValueError):
    """Needed metadata is missing or inconsistent."""


def parse(raw_xml: str) -> ET.Element:
    return ET.fromstring(raw_xml)


def pixel_size_m(root: ET.Element) -> float:
    """Square XY pixel size (m) of the full-resolution image."""
    vals = {}
    for d in root.findall(".//Scaling/Items/Distance"):
        v = d.findtext("Value")
        if v:
            vals[d.get("Id")] = float(v)          # Scaling is stored in metres
    if "X" not in vals or "Y" not in vals:
        raise MetadataError("no X/Y scaling in the metadata")
    if abs(vals["X"] - vals["Y"]) > 1e-12:
        raise MetadataError(f"non-square pixels: X={vals['X']} m, Y={vals['Y']} m")
    return vals["X"]


def channel_names(root: ET.Element) -> list[str]:
    return [c.get("Name") for c in root.findall(".//Information/Image/Dimensions/Channels/Channel")]


@dataclass(frozen=True)
class ChannelInfo:
    name: str
    channel_type: str | None       # e.g. 'AiryScanRingSheppardSum', 'AiryScanSum'; None = not given
    airyscan_mode: str | None      # e.g. 'SuperResolution'; None = not an Airyscan channel
    bits: int | None
    acquisition_mode: str | None = None   # 'WideField' (camera), 'LaserScanningConfocalMicroscopy', 'LSM', ...


def channel_info(root: ET.Element) -> list[ChannelInfo]:
    """Per channel: name, ChannelType, Airyscan mode (AiryscanSettings/Mode), bit
    depth, acquisition mode."""
    out = []
    for c in root.findall(".//Information/Image/Dimensions/Channels/Channel"):
        bits = c.findtext("ComponentBitCount")
        out.append(ChannelInfo(c.get("Name"), c.findtext("ChannelType"),
                               c.findtext("AiryscanSettings/Mode"), int(bits) if bits else None,
                               c.findtext("AcquisitionMode")))
    return out


def camera_binning(root: ET.Element, roi_offset_m: tuple[float, float]) -> str | None:
    """Binning (e.g. '1,1') of the camera whose settings record this ROI offset
    (RoiCenterOffsetX/Y, µm, rounded to 1 nm); None if no camera matches.
    The hardware settings list every camera of the scope, used or not."""
    for p in root.iter("ParameterCollection"):
        x, y = p.findtext("RoiCenterOffsetX"), p.findtext("RoiCenterOffsetY")
        if x is None or y is None or p.findtext("Binning") is None:
            continue
        if (abs(float(x) * 1e-6 - roi_offset_m[0]) < 1e-9
                and abs(float(y) * 1e-6 - roi_offset_m[1]) < 1e-9):
            return p.findtext("Binning")
    return None


def lsm_scan_geometry(root: ET.Element) -> list[tuple[float, float, float]]:
    """Per LSM channel: (SampleOffsetX, SampleOffsetY, SampleRotation) from its
    LaserScanInfo, as recorded. Only ever tested for == 0 (the placement of
    offset or rotated scans is not verified)."""
    out = []
    for c in root.findall(".//Information/Image/Dimensions/Channels/Channel/LaserScanInfo"):
        out.append(tuple(float(c.findtext(t) or 0) for t in
                         ("SampleOffsetX", "SampleOffsetY", "SampleRotation")))
    return out


def scene_names(root: ET.Element) -> dict[int, str]:
    """{scene index: name}; tile-region scenes carry the region name (e.g. the well)."""
    return {int(s.get("Index")): s.get("Name")
            for s in root.findall(".//Information/Image/Dimensions/S/Scenes/Scene")}


def planned_scene_centres_m(root: ET.Element) -> dict[int, tuple[float, float]]:
    """{scene index: planned (x, y)} from Scene/CenterPosition. Informative only."""
    out = {}
    for s in root.findall(".//Information/Image/Dimensions/S/Scenes/Scene"):
        cp = s.findtext("CenterPosition")
        if cp:
            x, y = (float(v) * 1e-6 for v in cp.split(","))
            out[int(s.get("Index"))] = (x, y)
    return out


def _recorded(root: ET.Element, pid: str) -> float | None:
    for p in root.iter("ParameterCollection"):
        if p.get("Id") == pid and p.findtext("Position"):
            return float(p.findtext("Position")) * 1e-6
    return None


def recorded_stage_xy_m(root: ET.Element) -> tuple[float, float] | None:
    """Encoder XY (m) stored with the image (MTBStageAxisX/Y). Informative only."""
    x, y = _recorded(root, "MTBStageAxisX"), _recorded(root, "MTBStageAxisY")
    return None if x is None or y is None else (x, y)


def recorded_focus_m(root: ET.Element) -> float | None:
    """Focus-drive Z (m) stored with the image (MTBFocus). Informative only.

    For a z-stack this is ONE value for the file; which plane it belongs to
    depends on the stack mode — use z_relative_m for plane positions.
    """
    return _recorded(root, "MTBFocus")


def size(root: ET.Element, dim: str) -> int:
    """Information/Image/Size<dim> (1 if absent, e.g. no Z dimension)."""
    v = root.findtext(f".//Information/Image/Size{dim}")
    return int(v) if v else 1


def z_relative_m(root: ET.Element, n_z: int | None = None) -> np.ndarray:
    """Z of each plane (m), relative to the stack centre.

    From Dimensions/Z/Positions/Interval Start + Increment. Relative positions
    need no knowledge of which Z frame (drive / focus / sample) the stack was
    defined in: for a CENTER-mode stack the centre is the focus at acquisition
    time, so an offset from here is directly the focus correction.

    Raises MetadataError for a stack whose spacing is not in the metadata —
    older code silently assumed 3 µm steps, which gives a wrong Z.
    """
    n = size(root, "Z") if n_z is None else n_z
    if n <= 1:
        return np.zeros(max(n, 1))
    inc = root.findtext(".//Information/Image/Dimensions/Z/Positions/Interval/Increment")
    if inc is None:
        raise MetadataError(f"{n}-plane stack without Z/Positions/Interval/Increment — "
                            f"plane spacing unknown (not guessing)")
    z = np.arange(n) * float(inc) * 1e-6
    return z - z.mean()


def z_step_m(root: ET.Element) -> float | None:
    inc = root.findtext(".//Information/Image/Dimensions/Z/Positions/Interval/Increment")
    return None if inc is None else float(inc) * 1e-6


# --------------------------------------------------------------------------
# Sample carrier (moved from SmartMic MS_metadata, 2026-11-11)
# --------------------------------------------------------------------------
_CARRIER_TEMPLATE_PARENTS = ("SampleHolder", "ScanAreaSampleHolderInfo")


def _strip_czsht(name: str) -> str:
    return name[:-6] if name.lower().endswith(".czsht") else name


def sample_carrier(root: ET.Element) -> dict | None:
    """The sample carrier the CZI was acquired with (``SampleHolder/Template``).

    Prefers the experiment's own ``SampleHolder/Template``, else the
    ``ScanAreaSampleHolderInfo`` copy (``.czsht`` suffix dropped); logs a
    warning if the file names more than one carrier. Informative: any value may
    be None; None if there is no Template.

    Keys: ``name``, ``category``, ``shape_type``, ``rows``, ``columns``,
    ``well_width_m``, ``well_height_m``, ``pitch_x_m``/``pitch_y_m`` (centre to
    centre), ``bounds_size_m`` / ``anchor_point_m`` as ``(x, y)``,
    ``bottom_material``, ``bottom_thickness_m``, ``insert_type``
    (``"Default"``/``"PartOfInsert"``), ``part_of_insert_name``,
    ``is_part_of_insert``. Lengths in metres (µm in the file).

    CAUTION, inserts: for an insert holding slides the file records only the
    ONE SLIDE (``is_part_of_insert`` True), neither the insert's name nor the
    slot, so ``name`` matches ZEN's carrier name only for a plate. SmartMic's
    ``MS_metadata.check_sample_carrier`` compares such a dict with the expected
    carrier.
    """
    found = [(parent.tag, child) for parent in root.iter() for child in parent
             if child.tag == "Template" and parent.tag in _CARRIER_TEMPLATE_PARENTS]
    if not found:
        return None
    tmpl = next((t for p, t in found if p == "SampleHolder"), found[0][1])
    names = {_strip_czsht(t.get("Name", "")) for _, t in found}
    if len(names) > 1:
        logging.getLogger(__name__).warning(
            f"more than one sample carrier ({', '.join(sorted(names))}); "
            f"using '{_strip_czsht(tmpl.get('Name', ''))}'.")

    def _float(tag):
        try:
            v = float(tmpl.findtext(tag))
        except (TypeError, ValueError):
            return None
        return None if v != v else v            # NaN → None

    def _m(tag):                                 # µm → m
        v = _float(tag)
        return None if v is None else v * 1e-6

    def _pair_m(tag):                            # "x,y" µm → m
        try:
            x, y = (float(v) * 1e-6 for v in tmpl.findtext(tag).split(","))
        except (AttributeError, TypeError, ValueError):
            return None
        return (x, y)

    def _text(tag):
        t = tmpl.findtext(tag)
        return t.strip() if t and t.strip() else None

    rows, cols = _float("ShapeRows"), _float("ShapeColumns")
    insert_type = _text("InsertType")
    return {
        "name": _strip_czsht(tmpl.get("Name", "")),
        "category": _text("Category"),
        "shape_type": _text("ShapeType"),
        "rows": None if rows is None else int(rows),
        "columns": None if cols is None else int(cols),
        "well_width_m": _m("ShapeWidth"),
        "well_height_m": _m("ShapeHeight"),
        "pitch_x_m": _m("ShapeDistanceX"),
        "pitch_y_m": _m("ShapeDistanceY"),
        "bounds_size_m": _pair_m("BoundsSize"),
        "anchor_point_m": _pair_m("AnchorPoint"),
        "bottom_material": _text("BottomMaterial"),
        "bottom_thickness_m": _m("BottomThickness"),
        "insert_type": insert_type,
        "part_of_insert_name": _text("PartOfInsertName"),
        "is_part_of_insert": insert_type == "PartOfInsert",
    }
