"""
ia_common — shared basis for new image-analysis (ia_*) projects.

Modules:
  cli       — run_analysis_script: the analysis-script contract with SmartMic
  logs      — setup_logging, write_json
  display   — scale bars, save_png
  coords    — THE pixel <-> stage convention (ImageFrame, offset_from_centre_m)
  czi.meta  — metadata from a CZI's XML (pixel size, channels, Z, recorded positions)
  czi.image — single-position images (widefield, LSM, Airyscan incl. 8-bit)
  czi.mosaic — tile-region mosaics (scenes, tiles, downsampled reads)

Everything returned in physical units is in METRES.
"""

__version__ = "0.1.0"
