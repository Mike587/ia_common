"""
analyze_template.py - starting point for an analysis script in a SmartMic
pipeline. Copy it into the new ia project as analyze_<what>.py, set SUFFIX,
and replace add_args() and analyse() with the real analysis.

    pixi run python analyze_template.py <image.czi> <output_dir> --prefix TAG
        [--channel 0] [--bin 4] [--k-sigma 10]

SmartMic (MS_image_analysis.run_analysis) runs the script like this and reads
<output_dir>/<TAG>_<SUFFIX>.json if the exit code is 0. The fixed part
(arguments, log, result file, exit code) is ia_common.cli.run_analysis_script;
this file only holds the analysis.

Example analysis: the brightest spot of one channel (max over Z, then a
k x k block mean), as a stage position and as an offset from the image centre.
Found only if it stands k_sigma noise sigmas above the background.

Writes to <output_dir>:
    <TAG>_brightest.json   found, x_m, y_m, dx_m, dy_m, snr, peak + provenance
    <TAG>_brightest.log    run log (also on stdout -> SmartMic's run log)
    <TAG>_brightest.png    QC: the searched image, spot (red), centre (cyan)
"""

import numpy as np

from ia_common.cli import run_analysis_script
from ia_common.coords import ImageFrame
from ia_common.czi import read_single
from ia_common.display import save_png

# Outputs are <prefix>_<SUFFIX>.json / .log: the pipeline reads <prefix>_<SUFFIX>.json.
SUFFIX = "brightest"


def add_args(parser):
    """The script's own options (the pipeline passes them as extra_args)."""
    parser.add_argument("--channel", type=int, default=0, help="channel index")
    parser.add_argument("--bin", type=int, default=4, help="k x k block mean before the search")
    parser.add_argument("--k-sigma", type=float, default=10.0,
                        help="found only if the peak is this many noise sigmas above the background")


def analyse(run):
    """The analysis -> result dict. Nothing found is a valid result (found=False,
    exit 0); raise only if the analysis cannot be done."""
    a = run.args
    im = read_single(run.image)
    cx, cy = im.stage_xy_m
    run.log.info(f"{im.data.shape} (C, Z, Y, X), channels {im.channel_names}, "
                 f"pixel {im.pixel_size_m * 1e6:.4f} um, stage ({cx * 1e6:.1f}, {cy * 1e6:.1f}) um")
    img = im.data[a.channel].max(axis=0).astype(np.float32)          # (Y, X), max over Z

    # k x k block mean (cropped to a multiple of k from the top left, so the
    # binned image keeps the same origin and its frame is exact)
    k = a.bin
    h, w = img.shape[0] // k, img.shape[1] // k
    small = img[:h * k, :w * k].reshape(h, k, w, k).mean(axis=(1, 3))
    frame = ImageFrame(im.frame.origin_x_px, im.frame.origin_y_px, k * im.frame.step_px, im.pixel_size_m)

    bg = float(np.median(small))
    noise = float(np.median(np.abs(small - bg)) * 1.4826) or 1.0     # robust sigma (MAD); 0 on an empty 8-bit image
    r, c = np.unravel_index(int(np.argmax(small)), small.shape)
    peak = float(small[r, c]) - bg
    snr = peak / noise
    found = snr >= a.k_sigma
    x, y = frame.to_stage_m(r, c)

    if found:
        run.log.info(f"Spot at ({(x - cx) * 1e6:+.1f}, {(y - cy) * 1e6:+.1f}) um from the centre, SNR {snr:.0f}")
    else:
        run.log.info(f"No spot: brightest SNR {snr:.1f} < {a.k_sigma}")

    def overlay(ax):
        ax.plot((w - 1) / 2, (h - 1) / 2, "c+", ms=20)
        if found:
            ax.plot(c, r, "o", mfc="none", mec="red", ms=20, mew=2)
    run.qc(save_png, run.output_path(f"{SUFFIX}.png"), small, frame.image_pixel_size_m, overlay=overlay)

    return {
        "found": bool(found),
        "x_m": x if found else None,          # stage position of the spot
        "y_m": y if found else None,
        "dx_m": x - cx if found else None,    # offset from the image centre, stage axes
        "dy_m": y - cy if found else None,
        "snr": snr,
        "peak": peak,
    }


if __name__ == "__main__":
    raise SystemExit(run_analysis_script(analyse, SUFFIX, add_args=add_args))
