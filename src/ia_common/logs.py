"""Logging and JSON output for analysis scripts (from ia_Marc/ia_NfS common.py)."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np


def setup_logging(log_path: str | Path, name: str, mode: str = "a") -> logging.Logger:
    """Logger writing to both a file and stdout.

    SmartMic's run_analysis forwards the script's stdout into the run log, so
    stdout is part of the contract, not just convenience.
    mode: 'a' appends to an existing log file, 'w' starts it afresh.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    close_logging(logger)  # avoid duplicate handlers on re-entry
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%S")
    for h in (logging.FileHandler(log_path, mode=mode, encoding="utf-8"),
              logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def close_logging(logger: logging.Logger) -> None:
    """Close and remove the logger's handlers (Windows keeps an open log file locked)."""
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)


def _to_builtin(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def write_json(obj, json_path: str | Path, log: logging.Logger | None = None) -> None:
    """Write a list/dict to JSON (numpy scalars/arrays and Paths are converted).

    The write is atomic: the file appears complete or not at all, and an
    existing file is left untouched if the write fails. Non-ASCII characters
    are escaped (json's default), so the file reads back correctly in any
    code page (SmartMic uses read_text() without an encoding).
    """
    json_path = Path(json_path)
    if log:
        log.info(f"Writing results to {json_path}")
    tmp = json_path.with_name(json_path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=_to_builtin)
        os.replace(tmp, json_path)
    finally:
        tmp.unlink(missing_ok=True)
