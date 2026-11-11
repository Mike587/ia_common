"""Mosaic reader on the WT OV (not committed; skipped when F: is not there)."""

import numpy as np
import pytest

from ia_common.czi import Mosaic

# Tile-region centres in Marc_SM_002_5x_OV_6Wells_slide2.czexp (µm).
REGIONS_UM = {"A1": (7300, 19100), "A2": (18200, 19100), "B1": (7300, 31200),
              "B2": (18200, 31200), "C1": (7300, 43300), "C2": (18200, 43300)}


def test_scenes_and_tiles(ov_wt):
    with Mosaic(ov_wt) as m:
        assert m.pixel_size_m == pytest.approx(0.69065e-6, rel=1e-4)
        assert sorted(s.name for s in m.scenes) == sorted(REGIONS_UM)
        assert sum(len(s.tiles) for s in m.scenes) == 72
        for s in m.scenes:
            assert len(s.tiles) == 12                          # 3 x 4 tiles per well
            # the imaged area sits on its tile region (the OV check of Marc_SM_002)
            x, y = m.centre_stage_m(s.rect)
            rx, ry = REGIONS_UM[s.name]
            assert abs(x * 1e6 - rx) < 1.0 and abs(y * 1e6 - ry) < 1.0


def test_downsampled_read_frame(ov_wt):
    with Mosaic(ov_wt) as m:
        k = m.downsample_for(8e-6)                        # closest to 8 µm pixels
        assert k == 12
        img, frame = m.read_scene("B2", downsample=k)
        sc = m.scene("B2")
        assert img.shape == (sc.rect.h // k, sc.rect.w // k) and img.dtype == np.float32
        assert frame.step_px == k and frame.image_pixel_size_m == pytest.approx(k * m.pixel_size_m)
        # the image centre maps to the scene centre (up to the crop to a multiple of k)
        x, y = frame.to_stage_m((img.shape[0] - 1) / 2, (img.shape[1] - 1) / 2)
        assert (x, y) == pytest.approx(m.centre_stage_m(sc.rect), abs=frame.image_pixel_size_m)


def test_exact_downsampling_is_block_mean_of_full_res(ov_wt):
    """Downsampled pixel (r, c) is exactly the mean of its k x k full-res block —
    so a position found downsampled is right (libCZI's own shrinking is off by up
    to ~1 output pixel, see Mosaic.read_region)."""
    with Mosaic(ov_wt) as m:
        small, fs = m.read_scene("A1", downsample=8)
        rect = m.scene("A1").rect
        for r0, c0 in [(3, 5), (small.shape[0] // 2, small.shape[1] // 2), (small.shape[0] - 30, small.shape[1] - 30)]:
            n = 20
            big, fb = m.read_region(type(rect)(rect.x + c0 * 8, rect.y + r0 * 8, n * 8, n * 8))
            assert fb.to_stage_m(0, 0) == pytest.approx(fs.to_stage_m(r0 - 0.5 + 0.5 / 8, c0 - 0.5 + 0.5 / 8))
            block = big.astype(np.float32).reshape(n, 8, n, 8).mean(axis=(1, 3))
            assert np.allclose(block, small[r0:r0 + n, c0:c0 + n], rtol=1e-5)


def test_libczi_shrinking_available(ov_wt):
    with Mosaic(ov_wt) as m:
        img, frame = m.read_scene("A1", downsample=8, method="libczi")
        assert img.dtype == np.uint16 and frame.step_px == pytest.approx(8, rel=1e-3)
        for bad in (0, 2.5, 0.125):
            with pytest.raises(ValueError):
                m.read_scene("A1", downsample=bad)
