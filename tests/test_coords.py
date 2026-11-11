import numpy as np
import pytest

from ia_common.coords import ImageFrame, offset_from_centre_m


def test_pixel_centre_convention():
    f = ImageFrame(origin_x_px=100, origin_y_px=200, step_px=1.0, pixel_size_m=1e-6)
    # pixel (0, 0) covers global [100, 101) x [200, 201): centre at 100.5 / 200.5 px
    assert f.to_stage_m(0, 0) == pytest.approx((100.5e-6, 200.5e-6))
    # X follows the column, Y the row
    x, y = f.to_stage_m(row=10, col=3)
    assert x == pytest.approx(103.5e-6) and y == pytest.approx(210.5e-6)


def test_downsampled_frame_and_inverse():
    f = ImageFrame(origin_x_px=4835, origin_y_px=22090, step_px=8.0, pixel_size_m=0.69e-6)
    assert f.image_pixel_size_m == pytest.approx(5.52e-6)
    rows, cols = np.array([0.0, 12.3, 500]), np.array([7.0, 0.0, 99.5])
    x, y = f.to_stage_m(rows, cols)
    r2, c2 = f.to_pixel(x, y)
    assert np.allclose(r2, rows) and np.allclose(c2, cols)


def test_offset_from_centre():
    # 4 x 4 image: the centre lies between pixels 1 and 2 -> pixel 2 is +0.5 px right/down
    assert offset_from_centre_m(2, 2, (4, 4), 1e-6) == pytest.approx((0.5e-6, 0.5e-6))
    assert offset_from_centre_m(1.5, 1.5, (4, 4), 1e-6) == pytest.approx((0.0, 0.0))
    # works on arrays and takes the last two dims of the shape
    dx, dy = offset_from_centre_m(np.array([0, 3]), np.array([0, 3]), (2, 20, 4, 4), 2e-6)
    assert np.allclose(dx, [-3e-6, 3e-6]) and np.allclose(dy, [-3e-6, 3e-6])


def test_rotated_frame():
    """Rotated LSM scan: image axes turned against the stage axes, + from +X towards +Y."""
    f = ImageFrame.centred_on(1e-3, 2e-3, 400, 300, 1e-6, rotation_deg=90)
    assert f.to_stage_m(149.5, 199.5) == pytest.approx((1e-3, 2e-3))        # the centre
    # one pixel along the image's X (column) axis = +Y on the stage after 90 deg
    x0, y0 = f.to_stage_m(149.5, 199.5)
    x1, y1 = f.to_stage_m(149.5, 200.5)
    assert (x1 - x0, y1 - y0) == pytest.approx((0.0, 1e-6), abs=1e-15)
    rows, cols = np.array([0.0, 12.3, 299]), np.array([7.0, 0.0, 399.5])
    r2, c2 = f.to_pixel(*f.to_stage_m(rows, cols))
    assert np.allclose(r2, rows) and np.allclose(c2, cols)
    # rotation 0 is the plain frame
    g = ImageFrame.centred_on(1e-3, 2e-3, 400, 300, 1e-6)
    assert g == ImageFrame(1e-3 / 1e-6 - 200, 2e-3 / 1e-6 - 150, 1.0, 1e-6)


def test_offset_from_centre_rotated():
    dx, dy = offset_from_centre_m(1.5, 2.5, (4, 4), 1e-6, rotation_deg=90)   # +1 px along columns
    assert (dx, dy) == pytest.approx((0.0, 1e-6), abs=1e-15)
