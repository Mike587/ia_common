"""
check_placement.py - does ia_common place an image where ZEN imaged it?

For test images taken at the SAME stage position as a reference image, with
the same objective, stage and focus untouched. Examples: camera images with a
partial readout region (ROI) vs. a full-chip reference; LSM scans at zoom 4
with a scan offset or rotation vs. a zoom-1 scan.

    pixi run python tools/check_placement.py <reference.czi> <test.czi> [<test.czi> ...]

Per test image:
  measured   where its content lies in the reference: normalised
             cross-correlation of the patterns (dirt, beads, ...), sub-pixel.
             The test image is resampled to the reference's pixel size, and
             for an image with a recorded rotation, its rotation is measured:
             0 and +-the recorded angle are tried, then a finer search around
             the best (--search-deg). Both images are smoothed slightly first
             (see smooth()).
             A central square of the test image is matched, so it may lie
             partly outside the reference.
  predicted  where ia_common places it (read with allow_unverified, so
             LSM offsets and rotated scans can be checked too)
  error      measured - predicted, in µm. ~0 (well under 1 µm): ia_common
             places the image correctly.
A second full-chip image at the end of the series, given as a test image,
measures stage/focus drift (its error should be ~0). If the stage moved
between the images (e.g. ZEN centring an ROI on a position), that image's
error is the stage's repeat accuracy, and no image can be placed better:
set --ok-um accordingly (2026-10-14: 0.1 µm without moves, 1.3 µm after one).

Also prints each file's pixel box and the metadata that differs from the
reference (which fields ZEN changes for an ROI, a scan offset, ...).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ia_common.czi import meta, read_single
from ia_common.czi._subblocks import read_subblock_metadata

# Metadata paths that change between any two acquisitions: not shown in the diff.
NOISE = ("Time", "Date", "Guid", "Id", "Name", "Uri", "Filename", "Path", "Temperature",
         "Statistics", "Duration", "Hash", "MTBStageAxis", "MTBFocus", "Version")


def flatten(root) -> dict[str, str]:
    """{element path: text} for every element with text, e.g. '.../Frame[1]': '0,0,4096,3008'."""
    out = {}

    def walk(el, path):
        counts = {}
        for ch in el:
            counts[ch.tag] = counts.get(ch.tag, 0) + 1
        seen = {}
        for ch in el:
            seen[ch.tag] = seen.get(ch.tag, 0) + 1
            p = f"{path}/{ch.tag}" + (f"[{seen[ch.tag]}]" if counts[ch.tag] > 1 else "")
            if ch.text and ch.text.strip():
                out[p] = ch.text.strip()
            walk(ch, p)

    walk(root, "")
    return out


def smooth(a: np.ndarray) -> np.ndarray:
    """Binomial blur [1 4 6 4 1]/16 along both axes (sigma ~1 px). Applied to
    both images before matching, so that the extra smoothing of bilinear
    interpolation (rotated or rescaled templates) does not raise their NCC
    over the unrotated one - that biased the angle search (2026-10-14)."""
    k = np.array([1, 4, 6, 4, 1], float) / 16
    a = a.astype(np.float64)
    for axis in (0, 1):
        p = np.pad(a, [(2, 2) if ax == axis else (0, 0) for ax in (0, 1)], mode="edge")
        a = sum(k[i] * np.take(p, np.arange(i, i + a.shape[axis]), axis=axis) for i in range(5))
    return a


def window_sum(a: np.ndarray, h: int, w: int) -> np.ndarray:
    """Sum over every h x w window (valid positions)."""
    c = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return c[h:, w:] - c[:-h, w:] - c[h:, :-w] + c[:-h, :-w]


class Matcher:
    """Normalised cross-correlation (NCC) of n x n templates against one image;
    the image's FFT and window sums are computed once."""

    def __init__(self, img: np.ndarray, n: int):
        img = img.astype(np.float64)
        img = (img - img.mean()) / (img.std() or 1.0)     # keeps the window sums precise
        self.shape, self.n = img.shape, n
        self.fft = np.fft.rfft2(img)
        s1, s2 = window_sum(img, n, n), window_sum(img * img, n, n)
        self.norm = np.sqrt(np.maximum(s2 - s1 * s1 / (n * n), 1e-12))

    def match(self, tpl: np.ndarray) -> tuple[float, float, float, float]:
        """(row, col, ncc, next_best): where the template's CENTRE lies in the
        image (sub-pixel), the NCC there (1 = identical patterns) and the best
        NCC further away than a tenth of the template (at least 10 px), i.e.
        outside the shoulder of the peak: a match is ambiguous if that is close."""
        t = tpl - tpl.mean()
        corr = np.fft.irfft2(self.fft * np.conj(np.fft.rfft2(t, s=self.shape)), s=self.shape)
        n = self.n
        ncc = corr[:self.shape[0] - n + 1, :self.shape[1] - n + 1] / (self.norm * np.sqrt((t * t).sum()))
        r, c = np.unravel_index(int(np.argmax(ncc)), ncc.shape)

        def refine(m1, m0, p1):                            # parabola through 3 points
            den = m1 - 2 * m0 + p1
            return 0.5 * (m1 - p1) / den if den < 0 else 0.0

        dr = refine(ncc[r - 1, c], ncc[r, c], ncc[r + 1, c]) if 0 < r < ncc.shape[0] - 1 else 0.0
        dc = refine(ncc[r, c - 1], ncc[r, c], ncc[r, c + 1]) if 0 < c < ncc.shape[1] - 1 else 0.0
        best = float(ncc[r, c])
        k = max(10, n // 10)
        ncc[max(0, r - k):r + k + 1, max(0, c - k):c + k + 1] = -np.inf
        next_best = float(ncc.max()) if np.isfinite(ncc).any() else -1.0
        return r + dr + (n - 1) / 2, c + dc + (n - 1) / 2, best, next_best


def template(img: np.ndarray, n: int, step: float, phi_deg: float) -> np.ndarray:
    """n x n template in the reference's pixel grid, centred on the centre of
    img, for test content rotated by phi relative to the reference: template
    pixel offset (u, v) (columns, rows; reference pixels) samples img at
    R(-phi) (u, v) * step from its centre, bilinear. step = reference pixel /
    test pixel. phi > 0 turns from +X (columns) towards +Y (rows)."""
    k = np.arange(n) - (n - 1) / 2
    u, v = np.meshgrid(k, k)
    c, s = np.cos(np.radians(phi_deg)), np.sin(np.radians(phi_deg))
    h, w = img.shape
    x = (c * u + s * v) * step + (w - 1) / 2
    y = (-s * u + c * v) * step + (h - 1) / 2
    x0 = np.clip(np.floor(x).astype(int), 0, w - 2)
    y0 = np.clip(np.floor(y).astype(int), 0, h - 2)
    fx, fy = np.clip(x - x0, 0, 1), np.clip(y - y0, 0, 1)
    a = img.astype(np.float64)
    return (a[y0, x0] * (1 - fx) * (1 - fy) + a[y0, x0 + 1] * fx * (1 - fy)
            + a[y0 + 1, x0] * (1 - fx) * fy + a[y0 + 1, x0 + 1] * fx * fy)


def recorded_rotation_deg(im) -> float:
    """Scan rotation as recorded (tile RoiRotation, else LaserScanInfo); 0 if none."""
    rot = {s.roi_rotation for s in read_subblock_metadata(im.path) if not s.pyramid} - {0.0}
    if rot:
        return float(next(iter(rot)))
    return next((g[2] for g in meta.lsm_scan_geometry(im.metadata) if g[2]), 0.0)


def best_match(matcher: Matcher, img: np.ndarray, step: float, candidates, search_deg: float):
    """(phi, (row, col, ncc, next_best)) of the best-matching rotation: the
    candidates, then a finer search around the best one."""
    def try_(phis):
        return max(((phi, matcher.match(template(img, matcher.n, step, phi))) for phi in phis),
                   key=lambda pm: pm[1][2])

    phi, m = try_(candidates)
    for width, d in ((search_deg, search_deg / 5), (search_deg / 5, search_deg / 50)):
        if width > 0:
            phi, m = try_(np.arange(phi - width, phi + width + d / 2, d))
    return phi, m


def describe(im) -> str:
    _, _, h, w = im.data.shape
    x0, y0 = im.frame.origin_x_px, im.frame.origin_y_px
    sx, sy = im.stage_xy_m
    rec = im.recorded_stage_xy_m
    rec_s = f"({rec[0] * 1e6:.2f}, {rec[1] * 1e6:.2f})" if rec else "n/a"
    return (f"{im.path.name}\n"
            f"  {w} x {h} px of {im.pixel_size_m * 1e6:.5f} um; pixel box x={x0:g} y={y0:g} "
            f"({im.frame_source})\n"
            f"  image centre at stage ({sx * 1e6:.2f}, {sy * 1e6:.2f}) um; "
            f"recorded stage {rec_s} um")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", type=Path)
    ap.add_argument("tests", type=Path, nargs="+")
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--diff-lines", type=int, default=40, help="max. metadata differences shown")
    ap.add_argument("--ok-um", type=float, default=0.5, help="largest error still reported as OK (µm)")
    ap.add_argument("--search-deg", type=float, default=1.0,
                    help="rotation search around the best candidate angle, +- this (0 = none); "
                         "only for images with a recorded rotation, unless --search-all")
    ap.add_argument("--search-all", action="store_true",
                    help="search the rotation of unrotated images too (adds ~0.3 µm noise on sparse "
                         "fields: a noise-level angle error shifts the centre)")
    args = ap.parse_args(argv)

    ref = read_single(args.reference)
    ref_img = smooth(ref.data[args.channel].max(axis=0))
    ref_meta = flatten(ref.metadata)
    print("REFERENCE " + describe(ref))
    matchers = {}

    for path in args.tests:
        im = read_single(path, allow_unverified=True)
        print("\nTEST " + describe(im))
        img = smooth(im.data[args.channel].max(axis=0))
        step = ref.pixel_size_m / im.pixel_size_m          # test pixels per reference pixel
        theta = recorded_rotation_deg(im)
        candidates = sorted({0.0, theta, -theta})
        search = args.search_deg if (theta or args.search_all) else 0.0
        widest = max(abs(np.cos(np.radians(a))) + abs(np.sin(np.radians(a)))
                     for p in candidates for a in (p - search, p, p + search))
        # the largest square inside the rotated test image, 80 % of it: the test
        # may then sit partly outside the reference (a same-size drift image)
        n = int(0.8 * min(img.shape) / step / widest)
        if n > min(ref_img.shape) or n < 32:
            print(f"  !! cannot match: template {n} px vs reference {ref_img.shape}")
            continue
        matcher = matchers.setdefault(n, Matcher(ref_img, n))
        phi, (r, c, ncc, next_best) = best_match(matcher, img, step, candidates, search)

        mx, my = ref.frame.to_stage_m(r, c)                 # where the test image's centre really is
        px, py = im.stage_xy_m                              # where ia_common places it
        ex, ey = mx - px, my - py
        print(f"  match: NCC {ncc:.3f} (next best elsewhere {next_best:.3f}); rotation measured "
              f"{phi:+.2f} deg, ia_common {im.rotation_deg:+.3f} deg (+ = from +X towards +Y; "
              f"recorded RoiRotation {theta:+.3f})")
        if im.tile_stage_xy_m is not None:
            tx, ty = im.tile_stage_xy_m
            ox, oy = im.roi_offset_m
            print(f"  content centre - tile stage: ({(mx - tx) * 1e6:+.2f}, {(my - ty) * 1e6:+.2f}) um; "
                  f"recorded RoiCenterOffset ({ox * 1e6:+.2f}, {oy * 1e6:+.2f}) um")
        verdict = "OK" if np.hypot(ex, ey) < args.ok_um * 1e-6 else "MISPLACED"
        # the angle is only as precise as the structure allows (±1° for a few large blobs)
        rot_ok = abs(phi - im.rotation_deg) < 1.0
        print(f"  centre error (measured - ia_common): dx {ex * 1e6:+.2f} um, dy {ey * 1e6:+.2f} um  "
              f"-> {verdict}" + ("" if rot_ok else
                                 f"; ROTATION differs by {phi - im.rotation_deg:+.2f} deg"))
        if next_best > 0.8 * ncc or ncc < 0.3:
            print("  !! ambiguous match: too little structure, or not the same field")

        meta_t = flatten(im.metadata)
        diff = [(p, ref_meta.get(p), meta_t.get(p)) for p in sorted(set(ref_meta) | set(meta_t))
                if ref_meta.get(p) != meta_t.get(p) and not any(k in p for k in NOISE)]
        print(f"  metadata differing from the reference: {len(diff)}")
        for p, a, b in diff[:args.diff_lines]:
            print(f"    {p}: {a!r} -> {b!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
