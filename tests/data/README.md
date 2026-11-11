# Test data

**The `WT_A1_cell001_*` files are not published** (collaborator sample data). They
live in `tests/data_private/` (git-ignored) on the acquisition PC; elsewhere the
tests that need them are skipped.

Real CZIs from the Marc_SM_002 run on the CD7, slide "WT", well A1, cell 1
(2026-09-29). Reference values in the tests come from independent sources:

- **Commanded stage positions:** the pipeline's cells log
  (`WT_cells.jsonl`, record `WT_A1_cell001`).
- **Pixel sums:** read with bioio-czi 2.x in pylibczirw mode, the reader
  Marc_SM_002 used.

| File | What | Size |
|---|---|---|
| `WT_A1_cell001_recentre.czi` | widefield DAPI, 20x/0.95 + 2x, Gray16, 4096 × 3008 | 18.6 MB |
| `WT_A1_cell001_midplane.czi` | 8-bit Airyscan DAPI z-stack (CENTER mode, ±19 µm / 2 µm, 20 planes), channels DAPI / DAPI#, sparse H × C grid | 2.1 MB |
| `WT_A1_cell001_airyscan.czi` | 8-bit Airyscan final image, AF568 / AF568# / DAPI / DAPI#, 962 × 962 | 9.5 MB |
| `WT_A1_cell001_airyscan-Airyscan Processing.czi` | the same image after Airyscan processing in ZEN (2D, automatic strength; processed by Mike): AF568 / DAPI, 16-bit, 938 × 938, pixel box reset to (0, 0) | 4.4 MB |

Camera readout region (ROI): dirt on a slide in DAPI, 5x + 1x, Axiocam
712m, all taken by Mike at the stage position (12985.496, 38470.159) µm on
2026-10-14. The full series (with the full-chip references) is in
`Example_Images/`, which is not committed. The content positions in the
tests were measured against the full-chip images with
`tools/check_placement.py`.

| File | What | Size |
|---|---|---|
| `WF_PartialChip_off_centered.czi` | Snap, ROI 1376 × 1090 at (1648, 1278): content offset (198.82, 220.20) µm | 9.7 MB |
| `EXP_WF_PartialChip_off_centered.czi` | Start Experiment with the current position, ROI 2080 × 1532 at (1456, 834): ZEN moved the stage by −`RoiCenterOffset`, content centred on the position | 11.8 MB |

LSM scan offset and rotation: same slide, same stage position, LSM 900
confocal detector, Snap, zoom 4, 506 × 506 px of 0.395 µm (2026-10-14). They
were compared with a zoom-1 scan in `Example_Images/`. About 6.9 MB of each
file are attachments ZEN stores; the image itself is 0.26 MB.

| File | What | Size |
|---|---|---|
| `LSM_off-centered_zoom4.czi` | scan offset (94.42, −77.25) µm: content offset measured (94.30, −77.08) µm | 8.0 MB |
| `LSM_centered_zoom4_rotated.czi` | scan rotation `RoiRotation` +30.924°: content rotated by −30.94° (measured) | 8.0 MB |

Not committed (too large): the WT OV mosaic (6 scenes = wells, 72 tiles). The
mosaic tests read it from `F:\UserData\mike\Marc_SM_002\...` and are skipped
when it is not there. **Wanted:** a small tile-region mosaic (e.g. 2 regions
× 2 × 2 tiles) to commit instead.
