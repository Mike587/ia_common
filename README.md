# ia_common

Shared basis for **new** image-analysis (`ia_*`) projects: CZI reading, the
pixel ↔ stage convention, logging and JSON output. The existing ia repos are
not adapted to it; they were only sources to copy from. SmartMic does not
depend on it.

## Use in a new project

In the project's `pixi.toml`:

```toml
[pypi-dependencies]
ia_common = { path = "../ia_common", editable = true }
```

A frozen project pins a commit instead.

```python
from ia_common.czi import read_single, Mosaic
from ia_common.coords import offset_from_centre_m
from ia_common.logs import setup_logging, write_json
from ia_common.display import save_png, add_scalebar

im = read_single("cell_recentre.czi")        # (C, Z, Y, X), metres everywhere
dx, dy = offset_from_centre_m(row, col, im.data.shape, im.pixel_size_m)

with Mosaic("overview.czi") as m:            # one scene per tile region (well)
    img, frame = m.read_scene("A1", downsample=m.downsample_for(2e-6))
    x_m, y_m = frame.to_stage_m(row, col)    # pixel of img -> stage XY
    save_png("A1_overview.png", img, frame.image_pixel_size_m,   # always with a scale bar
             overlay=lambda ax: ax.plot(cols, rows, "r+"))
```

**Every saved image gets a scale bar.**
- `save_png` requires the pixel size, so the bar can't be forgotten.
- For your own matplotlib figures, call `add_scalebar(ax, pixel_size_m)`
  after `imshow`.
- The pixel size is that of the image *as shown*: for a downsampled read,
  `frame.image_pixel_size_m`.
- Bar lengths follow the 1-2-5 series (… 10, 20, 50 µm … 1, 2, 5 mm), and
  the label sits on a semi-transparent box.

## Writing an analysis script for a SmartMic pipeline

**Start from `templates/analyze_template.py`:** copy it into the project as
`analyze_<what>.py`, set `SUFFIX`, and replace `add_args()` and `analyse()`.
The fixed part is `ia_common.cli.run_analysis_script`.

SmartMic (`MS_image_analysis.run_analysis`) runs
`pixi run python <script> <image> <output_dir> --prefix TAG [extra args]`.
It reads `<output_dir>/<TAG>_<suffix>.json` only if the exit code is 0, and
copies the script's stdout into its run log. `run_analysis_script` handles
that part:
- **Arguments:** `<image> <output_dir> [--prefix TAG]` plus the script's own
  options. Without a prefix, the image file name is used.
- **Log:** `<TAG>_<suffix>.log`, started afresh on each run, and also
  printed to stdout.
- **Old results:** an old `<TAG>_<suffix>.json` is deleted first, so a stale
  result can't be taken for the new one.
- **Result:** written atomically, so it is never half-written. A dict result
  gets a `provenance` entry: script, image, all options, the git commit of
  the script and of ia_common (`-dirty` if they have uncommitted changes),
  start time and duration. A list result (the ia_PoC_002 `targets.json`
  contract) is written unchanged, and its provenance goes to the log.
- **Exit code:** 0 on success. Any exception, or a missing image, is logged
  with the traceback and gives exit 1 with no result file.

Rules for `analyse(run)`:
- **"Nothing found" is a valid result:** return e.g. `{"found": False}`, and
  the exit code is 0. Raise only when the analysis can't be done. The
  pipeline can then tell a broken analysis from an empty field.
- **QC images go through `run.qc(fn, …)`:** a failing QC step is logged and
  recorded in `provenance.qc_errors`, but the result is kept.
- **Output files:** name them with `run.output_path("<name>.png")`, which
  gives `<TAG>_<name>.png`.

**Run scripts with `pixi run`, not the env's `python.exe` directly.** Without
pixi's activation, the env's DLL folders are not on PATH. Saving a
matplotlib PNG then crashes the process with exit code 0xC06D007E
(-1066598273), and no Python error handling can catch that.

## Modules

| Module | What it does |
|---|---|
| `cli` | `run_analysis_script`: the script contract with SmartMic (arguments, log, result file, exit code) |
| `coords` | `ImageFrame` (image pixel ↔ stage XY, with the rotation of a rotated LSM scan) and `offset_from_centre_m` |
| `czi.meta` | Metadata from the XML: pixel size, channels, scene names, Z of each plane, and the recorded stage/focus position |
| `czi.image` | `read_single`: widefield, LSM and Airyscan images and z-stacks, including 8-bit Airyscan; placed by the tile's stage position + camera ROI / scan offset (with the scan rotation), and raises for geometries not verified yet |
| `czi.mosaic` | `Mosaic`: tile-region mosaics with scenes, tiles and exact downsampled reads |
| `display` | `add_scalebar` (any matplotlib axes), `save_png` (native resolution, always with a scale bar, optional overlay) |
| `logs` | `setup_logging` (file + stdout), `write_json` (atomic, handles numpy types) |

Run the tests with `pixi run test`. The mosaic tests need the WT OV on F:
and are skipped without it (`tests/data/README.md`).

## Decisions and measurements (2026-09-30, CD7, Marc_SM_002 files)

**The global pixel frame of a CZI is the stage frame:**
`stage_m = global_px × pixel_size_m`, with no offset.
- **OV mosaics (5x):** 4 files, 22 scenes. Each tile's own recorded stage
  position agrees with this within 0.2 µm.
- **40x camera, LSM and Airyscan images:** the centre of the pixel box
  equals the commanded stage position within 0.1 µm.
- So positions come from the pixel boxes; the per-tile XML is not needed.
- The encoder read-out (`MTBStageAxisX/Y`) and the planned scene centre
  (`CenterPosition`) are available in `czi.meta`, as information only.

**Camera readout region (ROI): the image centre is at the tile's own stage
position + `RoiCenterOffset`** (2026-10-14; CD7, 5x + 1x, Axiocam 712m;
dirt in DAPI, taken in ZEN by Mike; measured with `tools/check_placement.py`).
- **Where ZEN records the ROI:** in the camera's settings.
  - `Frame`: x, y, w, h in chip pixels. The chip is `AcquisitionFrame`
    (0, 0, 4096, 3008).
  - `RoiCenterOffsetX/Y`: ROI centre minus chip centre, in µm. X runs along
    the columns, Y along the rows.
- **ZEN behaves differently in two modes**, and the file-level positions
  (scene `CenterPosition`, the `MTBStageAxis` reading) show the position in
  both, so they can't tell the modes apart:
  - **Snap:** the stage stays at the position, so the ROI images an area
    off the centre by `RoiCenterOffset`. Off-centre ROI 1376 × 1090 at
    (1648, 1278): content offset (198.82, 220.20) µm, while
    `RoiCenterOffset` = (198.908, 220.318) µm.
  - **Start Experiment with a position:** ZEN moves the stage by
    −`RoiCenterOffset`, so the ROI centre images the position, and then
    moves back. ROI 2080 × 1532 at (1456, 834), `RoiCenterOffset` =
    (309.413, 66.303) µm: content centred on the position within 0.8 µm.
    The tile's own stage position was (12676.083, 38403.856) µm, i.e. the
    position minus the offset exactly.
- **The tile's own stage position** (`StageXPosition/StageYPosition` in the
  subblock metadata) is where the stage really was, in both modes. Tile
  position + `RoiCenterOffset` gives the image centre in both modes. Every
  subblock also records its `RoiCenterOffset` (and LSM tiles their
  `RoiRotation`).
- **`read_single` places images by this rule** (since 2026-10-14). Errors
  against the content measurement: 0.08–0.15 µm in the snap series. In the
  experiment series it was 0.8 µm, which is within the stage's repeat
  accuracy (the full-chip repeat image there: 1.3 µm). pylibCZIrw 6.1 can't
  read subblock metadata, so `czi/_subblocks.py` reads it directly from the
  file. Checked against aicspylibczi on 12 files (193 subblocks):
  identical.
- **A real pixel box** (API-run pipeline images) is checked against the
  rule. On the test files it agrees within 0.27–0.50 px; a disagreement of
  more than 1 px raises.
- **Still raises, as not verified:** a camera ROI in a mosaic, a camera ROI
  with binning, and a rotated camera image. For the LSM, see below.
- **Even a "centred" ROI is off:** ZEN puts a 1408 × 1090 ROI one row above
  the centre (y = 958, not 959): −0.69 µm at 5x.
- **Chip vs. image:** the matched positions agree with `Frame` within
  0.2 px, so the image is not flipped relative to the chip.
- **Repeat accuracy:** 0.1 µm over the snap series, where nothing moved.
  1.3 µm over the experiment series, where the stage moved about 310 µm and
  back.
- **No real pixel box:** all 7 images (snap and experiment) start at (0, 0),
  unlike the pipeline's API-run images. Processed Airyscan files too.
- **Not yet known:** an ROI image with a real pixel box (an experiment run
  via the API). Does that box include the offset? Until a file shows it,
  the check above refuses such an image if it doesn't fit the rule.

**LSM scan offset and rotation: the same rule, plus a rotation about the
image centre** (2026-10-14; CD7 LSM 900, confocal detector, dirt in DAPI,
Snap; a zoom-1 2022² reference vs. zoom-4 506² scans, 0.395 µm pixels).
- **Where ZEN records them:** the scan offset in three places that always
  agreed: the tile's `RoiCenterOffsetX/Y` (µm),
  `LaserScanInfo/SampleOffsetX/Y` (metres!) and the LSM device settings.
  The rotation is in the tile's `RoiRotation` and in
  `LaserScanInfo/SampleRotation` (degrees).
  - The device settings (`MTBLSMImagingDevice` `Frame` and
    `RoiCenterOffset`) were stale in 2 of 7 files, still showing the
    previous scan. Don't use them.
- **Offset:** image centre = tile stage position + `RoiCenterOffset`, the
  same as for the camera. The stage did not move.
  - (94.42, −77.25) µm recorded: measured within 0.2 µm.
  - (−77.25, +85.83) µm recorded: measured within 0.4 µm (1 px).
- **Rotation:** the image axes are turned by −`RoiRotation` against the
  stage axes (positive = from +X towards +Y), about the image centre.
  - `RoiRotation` +30.924°: measured −30.94°, centre within 0.4 µm.
  - Rotation + offset: the offset stays in stage axes (it is not rotated
    with the scan); centre within 0.6 µm. The angle there is only defined to
    ±1° by the few dirt blobs.
- **`read_single` now handles both** (`SingleImage.rotation_deg`,
  `ImageFrame.rotation_deg`). It still raises in three cases:
  - the tile values disagree with `LaserScanInfo`;
  - a rotated scan has a real pixel box;
  - a mosaic has a scan offset or rotation.
- **Plain LSM images** carry an H dimension of size 1. `read_single`
  refused them as an unknown Airyscan layout before 2026-10-14; it now
  accepts a single H part.
- **Repeat accuracy** over the series: 0.2 µm, with the stage not moved.

**Checking a new case:** take images at one stage position, without moving
the stage or the focus (dirt on a slide is enough):
1. a full-field reference;
2. the variants to test;
3. the reference again, which measures drift.

`tools/check_placement.py` then measures where each variant's content lies
in the reference and compares that with where ia_common places it.

**Pixel convention:**
- Pixel *i* covers [*i*, *i* + 1), so its centre is at *i* + 0.5.
- X follows the column, Y follows the row.
- All units are metres.
- The older ia code (`pixel_to_abs_xy`) was half a pixel off.

**Z of each plane:** relative to the stack centre, from
`Dimensions/Z/Positions/Interval` (Start + Increment).
- If a stack's spacing is missing, the code raises `MetadataError`. The
  older code silently assumed 3 µm steps.
- The recorded focus-drive Z is also available, as one value per file.

**CZI library: pylibCZIrw (Zeiss).**
- **aicspylibczi 4.0.1** returns an array with memory for planes that don't
  exist when it reads 8-bit Airyscan files. Their H × C grid is sparse: the
  un-# channel has H0..H3, the # channel only H0. Touching the array then
  segfaults.
- **bioio-czi 3.x** made aicspylibczi its engine and deprecated its
  pylibczirw mode (3.0.1), so it crashes on the same files.
- **pylibCZIrw 6.1** reads them correctly.

**Downsampling is exact by default.** `downsample=k` turns every k × k
block of pixels into one pixel: the image gets k times smaller per axis and
the pixels k times larger. On the 5x OV (0.69 µm pixels), k = 8 gives 5.5 µm
pixels.
- libCZI's own shrinking (used by pylibCZIrw *and* aicspylibczi, with
  identical results) misplaces content by up to about 1 output pixel, and
  the error varies across the image. Measured on the WT OV: 1.1 µm at k = 2,
  1.8 µm at k = 4, 5.8 µm at k = 8, 8.3 µm at k = 10.
- `Mosaic` therefore reads at full resolution in bands and averages the
  k × k blocks itself. That takes about 2 s per well.
- `method="libczi"` gives the fast shrinking, for display only.
- k must be a whole number.

**Airyscan: several data layouts, not interchangeable.** Each channel
records its layout (`ChannelType`) and its mode (`AiryscanSettings/Mode`).
`read_single` reports both (`channel_types`, `airyscan_modes`).
- **Supported: ZEN's default unprocessed "4-channel format"** (seen on the
  CD7, SR mode). This is the input to ZEN's Airyscan processing, *not* a
  super-resolution image; `airyscan_unprocessed` is True. It holds:
  - `AiryScanRingSheppardSum`: 4 detector rings R0–R3 along the CZI "H"
    dimension;
  - `AiryScanSum`: the "#" channel, the sum of all detector elements (a
    confocal-like image).
  - `read_single(h=0)` gives ring R0 plus the # channel; `h=1..3` gives the
    other rings.
- **Supported: processed images** ("… -Airyscan Processing.czi"):
  `airyscan_processed` is True.
  - No H dimension, one channel per dye (the # channels are dropped),
    16-bit.
  - **Processing crops a 12-px border and resets the pixel box to (0, 0),**
    so the position comes from the recorded scene centre
    (`frame_source = "scene_centre"`). The crop is symmetric, and processed
    and raw pixel positions agree within half a pixel.
- **Not seen yet, test files wanted:**
  - the 32-channel raw format ("Keep Airyscan Raw Data");
  - Multiplex modes (SR-4Y, SR-8Y, CO-2Y);
  - the Sheppard-sum export.
- **A file with an H dimension and an unknown `ChannelType` raises**
  `MetadataError`, instead of guessing what H means.
