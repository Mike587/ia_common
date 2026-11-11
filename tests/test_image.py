"""Single-position readers on real CD7 files (see tests/data/README.md)."""

import numpy as np
import pytest

from ia_common.coords import offset_from_centre_m
from ia_common.czi import read_single

# Commanded stage positions from the Marc_SM_002 cells log (WT_A1_cell001).
RECENTRE_XY = (0.006994146987159908, 0.018914637865063436)   # camera image
LSM_XY = (0.006994502038145527, 0.01891646086708145)         # mid-plane + final


def test_widefield_recentre(data):
    im = read_single(data / "WT_A1_cell001_recentre.czi")
    assert im.data.shape == (1, 1, 3008, 4096) and im.data.dtype == np.uint16
    assert im.channel_names == ["DAPI"]
    assert int(im.data.sum(dtype=np.int64)) == 5450749222          # bioio reference
    assert im.pixel_size_m == pytest.approx(0.08643832750605382e-6)
    # image centre = commanded stage position (pixel frame = stage frame)
    assert im.stage_xy_m == pytest.approx(RECENTRE_XY, abs=0.1e-6)
    assert im.recorded_stage_xy_m == pytest.approx(RECENTRE_XY, abs=0.2e-6)


def test_airyscan_midplane_stack_sparse_h(data):
    im = read_single(data / "WT_A1_cell001_midplane.czi")
    assert im.n_h == 4 and im.h == 0
    assert im.channel_names == ["DAPI", "DAPI#"]
    assert im.data.shape == (2, 20, 256, 256) and im.data.dtype == np.uint8
    # same pixels bioio-czi 2.x (pylibczirw mode) returned
    assert [int(im.data[c].sum(dtype=np.int64)) for c in range(2)] == [57332, 129187]
    assert im.z_rel_m[0] == pytest.approx(-19e-6) and im.z_rel_m[-1] == pytest.approx(19e-6)
    assert im.stage_xy_m == pytest.approx(LSM_XY, abs=0.1e-6)
    assert im.channel("DAPI#").shape == (20, 256, 256)


def test_airyscan_other_h_parts(data):
    # H1..H3 exist only for the un-# channel; the # channel is left out
    im = read_single(data / "WT_A1_cell001_midplane.czi", h=1)
    assert im.channel_names == ["DAPI"] and im.data.shape == (1, 20, 256, 256)
    with pytest.raises(ValueError):
        read_single(data / "WT_A1_cell001_midplane.czi", h=4)


def test_airyscan_final(data):
    im = read_single(data / "WT_A1_cell001_airyscan.czi")
    assert im.channel_names == ["AF568-T3", "AF568#-T3", "DAPI-T4", "DAPI#-T4"]
    assert im.data.shape == (4, 1, 962, 962)
    assert [int(im.data[c].sum(dtype=np.int64)) for c in range(4)] == [286174, 684777, 236146, 625423]
    assert im.stage_xy_m == pytest.approx(LSM_XY, abs=0.1e-6)


def test_frame_and_offset_agree(data):
    im = read_single(data / "WT_A1_cell001_airyscan.czi")
    _, _, h, w = im.data.shape
    r, c = 100.0, 800.0
    x, y = im.frame.to_stage_m(r, c)
    dx, dy = offset_from_centre_m(r, c, (h, w), im.pixel_size_m)
    cx, cy = im.stage_xy_m
    assert (x - cx, y - cy) == pytest.approx((dx, dy), abs=1e-12)


def test_airyscan_layout_is_reported(data):
    wf = read_single(data / "WT_A1_cell001_recentre.czi")
    assert wf.channel_types == [None] and wf.airyscan_modes == [None] and not wf.airyscan_unprocessed
    im = read_single(data / "WT_A1_cell001_airyscan.czi")
    assert im.channel_types == ["AiryScanRingSheppardSum", "AiryScanSum"] * 2
    assert im.airyscan_modes == ["SuperResolution"] * 4
    assert im.airyscan_unprocessed                      # ring format, not a SR image


def test_unknown_airyscan_layout_with_h_raises(data, monkeypatch):
    """An H dimension with an unknown ChannelType (e.g. 32-channel raw, Multiplex)
    must not be read as if it were the ring format."""
    from ia_common.czi import meta
    real = meta.channel_info

    def fake(root):
        return [meta.ChannelInfo(ci.name, "AiryScanRaw32" if ci.channel_type else None,
                                 ci.airyscan_mode, ci.bits) for ci in real(root)]

    monkeypatch.setattr(meta, "channel_info", fake)
    with pytest.raises(meta.MetadataError, match="does not know"):
        read_single(data / "WT_A1_cell001_midplane.czi")
    # no H dimension -> no ambiguity, still readable
    assert read_single(data / "WT_A1_cell001_recentre.czi").data.shape[0] == 1


PROCESSED = "WT_A1_cell001_airyscan-Airyscan Processing.czi"


def test_airyscan_processed(data):
    im = read_single(data / PROCESSED)
    assert im.channel_names == ["AF568-T3", "DAPI-T4"]           # the # channels are dropped
    assert im.data.shape == (2, 1, 938, 938) and im.data.dtype == np.uint16
    assert im.airyscan_processed and not im.airyscan_unprocessed
    assert im.airyscan_modes == ["SuperResolution"] * 2 and im.channel_types == [None, None]
    # pixel box was reset to (0, 0): position from the tile's stage position,
    # which processing keeps
    assert im.frame_source == "tile_stage"
    assert im.stage_xy_m == pytest.approx(LSM_XY, abs=0.1e-6)


def test_processed_and_raw_agree_on_positions(data):
    """Processing crops 12 px per side: processed pixel (r, c) = raw pixel
    (r + 12, c + 12) (checked by registering the DAPI channels)."""
    raw = read_single(data / "WT_A1_cell001_airyscan.czi")
    proc = read_single(data / PROCESSED)
    assert raw.frame_source == "pixel_frame" and not raw.airyscan_processed
    for r, c in [(0, 0), (400, 700), (937, 937)]:
        # within half a pixel: the raw pixel box is whole pixels, the recorded centre is not
        assert proc.frame.to_stage_m(r, c) == pytest.approx(raw.frame.to_stage_m(r + 12, c + 12),
                                                            abs=0.5 * raw.pixel_size_m)
    # and the image content really is that crop (DAPI: raw '#' sum vs processed)
    a = raw.channel("DAPI#-T4")[0, 12:950, 12:950].astype(float)
    b = proc.channel("DAPI-T4")[0].astype(float)
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.95
