import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ia_common.display import add_scalebar, format_length, nice_length_m, save_png


def test_nice_length_1_2_5():
    assert nice_length_m(0.2 * 354e-6) == pytest.approx(50e-6)       # 70.8 µm -> 50 (linearly closest)
    assert nice_length_m(18e-6) == pytest.approx(20e-6)
    assert nice_length_m(7e-6) == pytest.approx(5e-6)
    assert nice_length_m(1.4e-3) == pytest.approx(1e-3)
    assert nice_length_m(3.3e-3) == pytest.approx(2e-3)
    with pytest.raises(ValueError):
        nice_length_m(0)


def test_format_length():
    assert format_length(50e-6) == "50 µm"
    assert format_length(2e-3) == "2 mm"
    assert format_length(500e-9) == "500 nm"
    assert format_length(1e-6) == "1 µm"


def test_bar_size_in_data_units_pixel_and_um_extent():
    img = np.zeros((300, 400))
    fig, (a1, a2) = plt.subplots(1, 2)
    a1.imshow(img)                                        # extent in pixels
    bar, length = add_scalebar(a1, 0.5e-6, length_m=50e-6)
    assert bar.size_bar.get_children()[0].get_width() == pytest.approx(100)   # 50 µm / 0.5 µm
    a2.imshow(img, extent=(0, 200, 150, 0))               # extent in µm
    bar, _ = add_scalebar(a2, 0.5e-6, length_m=50e-6)
    assert bar.size_bar.get_children()[0].get_width() == pytest.approx(50)
    plt.close(fig)


def test_needs_an_image():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        add_scalebar(ax, 1e-6)
    plt.close(fig)


def test_save_png_native_size_and_bar_length(tmp_path):
    img = np.zeros((400, 600), np.uint16)
    p = save_png(tmp_path / "x.png", img, 0.5e-6, clim=(0, 1),
                 scalebar_kw={"length_m": 100e-6, "box": False, "label": False})
    out = plt.imread(p)
    assert out.shape[:2] == (400, 600)
    white = out[..., :3].mean(axis=2) > 0.9
    cols = np.where(white.any(axis=0))[0]
    rows = np.where(white.any(axis=1))[0]
    assert cols.max() - cols.min() + 1 == pytest.approx(200, abs=2)   # 100 µm / 0.5 µm
    assert cols.min() > 300 and rows.min() > 300                      # lower right


def test_save_png_with_overlay_and_rgb(tmp_path):
    rgb = np.zeros((100, 120, 3), np.uint8)
    p = save_png(tmp_path / "rgb.png", rgb, 1e-6, title="A1",
                 overlay=lambda ax: ax.plot([10, 50], [10, 50], "r-"))
    assert plt.imread(p).shape[:2] == (100, 120)
