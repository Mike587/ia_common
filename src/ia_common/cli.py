"""
The analysis-script contract with SmartMic, implemented once.

SmartMic's MS_image_analysis.run_analysis launches an analysis as

    pixi run python <script> <image> <output_dir> --prefix TAG [extra args...]

and the pipeline then reads <output_dir>/<TAG>_<suffix>.json, but only if the
exit code is 0. The script's stdout is copied into SmartMic's run log.

run_analysis_script does that fixed part, so a script only holds the
analysis (see templates/analyze_template.py):

    def analyse(run):                  # run.image, run.args, run.log, ...
        ...
        return {"found": True, "dx_m": dx, "dy_m": dy}

    if __name__ == "__main__":
        raise SystemExit(run_analysis_script(analyse, "recentre", add_args=add_args))

What it does, in order:
  1. arguments: <image> <output_dir> [--prefix TAG] + the script's own options
     (add_args). No prefix -> the image file name without extension.
  2. log: <prefix>_<suffix>.log (started afresh) and stdout.
  3. deletes an old <prefix>_<suffix>.json, so a stale result can never be
     taken for the result of this run.
  4. a missing image -> logged error, exit 1.
  5. result = analyse(run); written atomically to <prefix>_<suffix>.json.
     A dict result gets a "provenance" entry (script, image, options, code
     versions, time). A list result (the ia_PoC_002 targets.json contract) is
     written unchanged; its provenance is in the log only.
  6. exit 0; any exception -> logged with traceback, exit 1, no result file.

Conventions for analyse():
  * "Nothing found" is a valid result: return e.g. {"found": False, ...} and
    exit 0. Raise only when the analysis could not be done (unreadable file,
    missing metadata, a bug): the pipeline then knows the analysis is broken,
    not that the field is empty.
  * Returning None is an error (a forgotten return).
  * QC images and plots go through run.qc(...): a failing QC step is logged
    and recorded in the provenance, but does not cost the result.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import __version__
from .logs import close_logging, setup_logging, write_json


@dataclass
class AnalysisRun:
    """What analyse(run) gets: the inputs, the options, the log, output naming."""
    image: Path
    output_dir: Path
    prefix: str
    suffix: str
    args: argparse.Namespace         # all options, including the script's own
    log: logging.Logger
    qc_errors: list[str] = field(default_factory=list)

    def output_path(self, name: str) -> Path:
        """<output_dir>/<prefix>_<name>, e.g. output_path('recentre.png')."""
        return self.output_dir / f"{self.prefix}_{name}"

    @property
    def result_path(self) -> Path:
        """The result file the pipeline reads: <prefix>_<suffix>.json."""
        return self.output_path(f"{self.suffix}.json")

    def qc(self, fn: Callable, *args, **kwargs):
        """Run a QC step (overlay image, plot, ...) and return its value.

        A failure is logged with its traceback and recorded in the provenance,
        but does not fail the analysis: QC is for people, the result file is
        what the pipeline needs. Returns None if the step failed.
        """
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            name = getattr(fn, "__name__", repr(fn))
            self.log.warning(f"QC step {name} failed: {e!r}", exc_info=True)
            self.qc_errors.append(f"{name}: {e!r}")
            return None


def _git_commit(path: Path) -> str | None:
    """Short commit of the git repo containing `path`, with '-dirty' if tracked
    files have uncommitted changes; None if git or the repo is not available."""
    def git(*a):
        return subprocess.run(["git", "-C", str(path), *a], capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip()
    try:
        commit = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return None
    return commit + ("-dirty" if dirty else "")


def _source_file(fn: Callable) -> Path | None:
    try:
        return Path(inspect.getfile(fn)).resolve()
    except (TypeError, OSError):
        return None


def run_analysis_script(analyse: Callable[[AnalysisRun], dict | list], suffix: str, *,
                        add_args: Callable[[argparse.ArgumentParser], None] | None = None,
                        description: str | None = None, argv: list[str] | None = None) -> int:
    """Run an analysis under the SmartMic script contract; returns the exit code.

    analyse     -- analyse(run) -> result (dict, or list); see the module docstring
    suffix      -- names the outputs: <prefix>_<suffix>.json / .log
    add_args    -- add_args(parser): the script's own options
    description -- --help text (default: the docstring of analyse's module)
    argv        -- arguments (default: the command line)
    """
    if description is None:
        description = inspect.getdoc(inspect.getmodule(analyse))
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="input image (e.g. a .czi file)")
    parser.add_argument("output_dir", type=Path, help="output folder (created if absent)")
    parser.add_argument("--prefix", default=None,
                        help="prefix of all output file names "
                             "(default: the image file name without extension)")
    if add_args is not None:
        add_args(parser)
    args = parser.parse_args(argv)

    prefix = args.prefix or args.image.stem
    log = setup_logging(args.output_dir / f"{prefix}_{suffix}.log", f"ia_common.run.{suffix}", mode="w")
    run = AnalysisRun(args.image, args.output_dir, prefix, suffix, args, log)
    try:
        run.result_path.unlink(missing_ok=True)
        script = _source_file(analyse)
        log.info(f"{suffix} analysis: {args.image}")
        log.info(f"Output: {run.result_path}")
        log.info(f"Options: {json.dumps(vars(args), default=str)}")
        if not args.image.exists():
            log.error(f"Image not found: {args.image}")
            return 1

        started, t0 = datetime.now(), time.perf_counter()
        result = analyse(run)
        if result is None:
            raise TypeError("analyse() returned None; return the result, "
                            "e.g. {'found': False} if nothing was found")
        if not isinstance(result, (dict, list)):
            raise TypeError(f"analyse() must return a dict or a list, not {type(result).__name__}")

        ia_dir = Path(__file__).resolve().parent
        provenance = {
            "script": script,
            "script_commit": _git_commit(script.parent) if script else None,
            "image": args.image.resolve(),
            "options": dict(vars(args)),
            "ia_common_version": __version__,
            "ia_common_commit": _git_commit(ia_dir),
            "started": started.isoformat(timespec="seconds"),
            "duration_s": round(time.perf_counter() - t0, 3),
            "qc_errors": run.qc_errors,
        }
        if isinstance(result, dict):
            if "provenance" in result:
                raise ValueError("the result already has a 'provenance' entry; rename it")
            result = {**result, "provenance": provenance}
        else:
            log.info(f"Provenance: {json.dumps(provenance, default=str)}")
        write_json(result, run.result_path)
        log.info(f"Result written: {run.result_path.name} ({provenance['duration_s']:.1f} s)")
        return 0
    except Exception:
        log.exception(f"{suffix} analysis failed; no result written")
        return 1
    finally:
        close_logging(log)
