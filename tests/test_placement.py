"""Where an image is: tile stage position + camera ROI offset, and the guards
for geometries that are not verified (see czi/image.py, README)."""

import dataclasses
import math

import pytest

from ia_common.czi import Mosaic, read_single
from ia_common.czi import image as image_mod
from ia_common.czi import meta
from ia_common.czi._subblocks import read_subblock_metadata

UM = 1e-6
# Mike's dirt images, 2026-10-14, 5x + 1x, all at this stage position (µm).
POSITION = (12985.496, 38470.159)
# Content position of each ROI image relative to POSITION, measured against a
# full-chip image of the same field with tools/check_placement.py (µm).
SNAP_MEASURED = (198.81, 220.17)
EXP_MEASURED = (0.73, -0.33)


def test_subblock_metadata(data):
    sbs = read_subblock_metadata(data / "EXP_WF_PartialChip_off_centered.czi")
    assert len(sbs) == 1 and not sbs[0].pyramid
    assert sbs[0].stage_xy_m == pytest.approx((12676.083 * UM, 38403.856 * UM), abs=1e-12)
    assert sbs[0].roi_offset_m == pytest.approx((309.413 * UM, 66.303 * UM), abs=1e-12)
    assert sbs[0].focus_m == pytest.approx(1123.934 * UM)
    stack = read_subblock_metadata(data / "WT_A1_cell001_midplane.czi")
    assert len(stack) == 100 and {s.start["Z"] for s in stack} == set(range(20))
    assert {s.roi_rotation for s in stack} == {0.0}


def test_snapped_roi_is_off_centre(data):
    """Snap: the stage stays at the position, the ROI images an area off-centre."""
    im = read_single(data / "WF_PartialChip_off_centered.czi")
    assert im.frame_source == "tile_stage"
    assert im.tile_stage_xy_m == pytest.approx((POSITION[0] * UM, POSITION[1] * UM), abs=1e-12)
    assert im.roi_offset_m == pytest.approx((198.908 * UM, 220.318 * UM), abs=1e-12)
    cx, cy = im.stage_xy_m
    assert ((cx / UM - POSITION[0]), (cy / UM - POSITION[1])) == pytest.approx(SNAP_MEASURED, abs=0.3)


def test_experiment_roi_is_centred_on_the_position(data):
    """Start Experiment with a position: ZEN moves the stage by -offset."""
    im = read_single(data / "EXP_WF_PartialChip_off_centered.czi")
    assert im.frame_source == "tile_stage"
    assert im.tile_stage_xy_m == pytest.approx(((POSITION[0] - 309.413) * UM,
                                                (POSITION[1] - 66.303) * UM), abs=1e-12)
    cx, cy = im.stage_xy_m
    assert (cx / UM, cy / UM) == pytest.approx(POSITION, abs=1e-6)             # the rule
    # measured: within the stage's repeat accuracy after a 310 µm move (1.3 µm)
    assert ((cx / UM - POSITION[0]), (cy / UM - POSITION[1])) == pytest.approx(EXP_MEASURED, abs=1.0)


def test_pipeline_images_keep_their_pixel_box(data):
    for f in ("WT_A1_cell001_recentre.czi", "WT_A1_cell001_midplane.czi", "WT_A1_cell001_airyscan.czi"):
        im = read_single(data / f)
        assert im.frame_source == "pixel_frame" and im.roi_offset_m == (0.0, 0.0)
        assert im.stage_xy_m == pytest.approx(im.tile_stage_xy_m, abs=0.1 * UM)


def patch_tags(monkeypatch, module, **tags):
    """Pretend every subblock recorded these tags (µm values as in the file)."""
    real = read_subblock_metadata

    def fake(path):
        return [dataclasses.replace(s, tags={**s.tags, **tags}) for s in real(path)]
    monkeypatch.setattr(module, "read_subblock_metadata", fake)


def test_real_pixel_box_disagreeing_with_the_rule_raises(data, monkeypatch):
    """A camera ROI in an API-run experiment (real pixel box) is not verified:
    if the box doesn't fit tile stage + offset, refuse instead of guessing."""
    patch_tags(monkeypatch, image_mod, RoiCenterOffsetX="50.0", RoiCenterOffsetY="0")
    monkeypatch.setattr(meta, "camera_binning", lambda root, o: "1,1")
    with pytest.raises(meta.MetadataError, match=r"pixel box centre is (49\.9\d|50\.0\d) µm"):
        read_single(data / "WT_A1_cell001_recentre.czi")


# LSM, same session and stage position: content measured against a zoom-1 scan
# with tools/check_placement.py (centre relative to the tile stage, µm; deg).
LSM_OFFSET_MEASURED = (94.30, -77.08)
LSM_ROTATION_MEASURED = -30.94


def test_lsm_scan_offset(data):
    """The LSM scan offset follows the camera rule: centre = tile stage + RoiCenterOffset."""
    im = read_single(data / "LSM_off-centered_zoom4.czi")          # also: an H dimension of size 1
    assert im.frame_source == "tile_stage" and im.geometry_verified and im.rotation_deg == 0
    assert im.roi_offset_m == pytest.approx((94.4169 * UM, -77.2502 * UM), abs=1e-12)
    (cx, cy), (tx, ty) = im.stage_xy_m, im.tile_stage_xy_m
    assert ((cx - tx) / UM, (cy - ty) / UM) == pytest.approx(LSM_OFFSET_MEASURED, abs=0.5)


def test_lsm_scan_rotation(data):
    """Rotated scan: image axes turned by -RoiRotation about the image centre."""
    im = read_single(data / "LSM_centered_zoom4_rotated.czi")
    assert im.rotation_deg == pytest.approx(-30.924, abs=1e-4)                 # the rule
    assert im.rotation_deg == pytest.approx(LSM_ROTATION_MEASURED, abs=0.2)   # measured
    assert im.stage_xy_m == pytest.approx(im.tile_stage_xy_m, abs=0.5 * UM)
    # a pixel 100 px right of the centre lies 100 px along the rotated X axis
    _, _, h, w = im.data.shape
    x0, y0 = im.frame.to_stage_m((h - 1) / 2, (w - 1) / 2)
    x1, y1 = im.frame.to_stage_m((h - 1) / 2, (w - 1) / 2 + 100)
    assert math.degrees(math.atan2(y1 - y0, x1 - x0)) == pytest.approx(-30.924, abs=1e-6)
    assert math.hypot(x1 - x0, y1 - y0) == pytest.approx(100 * im.pixel_size_m)


@pytest.mark.parametrize("tags", [{"RoiCenterOffsetX": "5.0"}, {"RoiRotation": "12.5"}])
def test_lsm_tags_disagreeing_with_laserscaninfo_raise(data, monkeypatch, tags):
    patch_tags(monkeypatch, image_mod, **tags)
    with pytest.raises(meta.MetadataError, match="disagree with LaserScanInfo"):
        read_single(data / "WT_A1_cell001_airyscan.czi")


def test_laserscaninfo_without_matching_tags_raises(data, monkeypatch):
    monkeypatch.setattr(meta, "lsm_scan_geometry", lambda root: [(3e-6, 0.0, 0.0)])
    with pytest.raises(meta.MetadataError, match="disagree with LaserScanInfo"):
        read_single(data / "WT_A1_cell001_midplane.czi")


def test_rotated_scan_with_real_pixel_box_raises(data, monkeypatch):
    patch_tags(monkeypatch, image_mod, RoiRotation="12.5")
    monkeypatch.setattr(meta, "lsm_scan_geometry", lambda root: [(0.0, 0.0, 12.5)])
    with pytest.raises(meta.MetadataError, match="real pixel box"):
        read_single(data / "WT_A1_cell001_airyscan.czi")


def test_rotated_camera_image_raises(data, monkeypatch):
    patch_tags(monkeypatch, image_mod, RoiRotation="5.0")
    with pytest.raises(meta.MetadataError, match="rotated camera image"):
        read_single(data / "WF_PartialChip_off_centered.czi")
    # the opt-out reads it, flagged (a real pixel box with a rotation would still raise)
    im = read_single(data / "WF_PartialChip_off_centered.czi", allow_unverified=True)
    assert not im.geometry_verified and im.rotation_deg == -5.0


def test_camera_roi_with_binning_raises(data, monkeypatch):
    monkeypatch.setattr(meta, "camera_binning", lambda root, o: "2,2")
    with pytest.raises(meta.MetadataError, match="binning 2,2"):
        read_single(data / "WF_PartialChip_off_centered.czi")


def test_mosaic_with_camera_roi_raises(data):
    with pytest.raises(meta.MetadataError, match="tiles with an ROI"):
        Mosaic(data / "WF_PartialChip_off_centered.czi")


def test_scene_centre_fallback_without_tile_stage(data, monkeypatch):
    """Subblocks without a stage position (not seen yet): the recorded scene centre."""
    real = read_subblock_metadata
    monkeypatch.setattr(image_mod, "read_subblock_metadata",
                        lambda path: [dataclasses.replace(s, tags={}) for s in real(path)])
    im = read_single(data / "WT_A1_cell001_airyscan-Airyscan Processing.czi")
    assert im.frame_source == "scene_centre" and im.tile_stage_xy_m is None
