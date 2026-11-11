"""The analysis-script contract (ia_common.cli) and the template script."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ia_common.cli import run_analysis_script

TEMPLATE = Path(__file__).parents[1] / "templates" / "analyze_template.py"


@pytest.fixture
def image(tmp_path):
    p = tmp_path / "cell007.czi"          # the runner only checks it exists; analyse() reads it
    p.write_bytes(b"not read")
    return p


def add_k(parser):
    parser.add_argument("--k", type=float, default=1.0)


def found_k(run):
    run.log.info("analysing")
    return {"found": True, "k": run.args.k}


def test_result_log_and_provenance(image, tmp_path):
    out = tmp_path / "out"
    # SmartMic's call: <image> <output_dir> --prefix TAG, then extra_args
    assert run_analysis_script(found_k, "spot", add_args=add_k,
                               argv=[str(image), str(out), "--prefix", "A1_c3", "--k", "2.5"]) == 0
    res = json.loads((out / "A1_c3_spot.json").read_text())
    assert res["found"] is True and res["k"] == 2.5
    prov = res["provenance"]
    assert prov["image"] == str(image.resolve())
    assert prov["options"]["k"] == 2.5 and prov["options"]["prefix"] == "A1_c3"
    assert prov["script"] == str(Path(__file__).resolve())
    assert prov["qc_errors"] == [] and prov["duration_s"] >= 0
    assert {"script_commit", "ia_common_version", "ia_common_commit", "started"} <= set(prov)
    log = (out / "A1_c3_spot.log").read_text(encoding="utf-8")
    assert "analysing" in log and "Result written" in log


def test_prefix_defaults_to_image_name(image, tmp_path):
    assert run_analysis_script(found_k, "spot", add_args=add_k, argv=[str(image), str(tmp_path)]) == 0
    assert (tmp_path / "cell007_spot.json").exists()


def test_log_starts_afresh(image, tmp_path):
    for _ in range(2):
        run_analysis_script(found_k, "spot", add_args=add_k, argv=[str(image), str(tmp_path)])
    assert (tmp_path / "cell007_spot.log").read_text(encoding="utf-8").count("analysing") == 1


def test_missing_image_exits_1(tmp_path):
    assert run_analysis_script(found_k, "spot", argv=[str(tmp_path / "nope.czi"), str(tmp_path)]) == 1
    assert not (tmp_path / "nope_spot.json").exists()
    assert "Image not found" in (tmp_path / "nope_spot.log").read_text(encoding="utf-8")


def test_failure_exits_1_and_removes_the_stale_result(image, tmp_path):
    stale = tmp_path / "cell007_spot.json"
    stale.write_text('{"found": true}')           # left by an earlier run with the same prefix

    def broken(run):
        raise RuntimeError("no Z spacing in the metadata")

    assert run_analysis_script(broken, "spot", argv=[str(image), str(tmp_path)]) == 1
    assert not stale.exists()
    log = (tmp_path / "cell007_spot.log").read_text(encoding="utf-8")
    assert "no Z spacing in the metadata" in log and "Traceback" in log


@pytest.mark.parametrize("bad", [None, 3.0, {"provenance": "mine"}])
def test_invalid_results_exit_1(image, tmp_path, bad):
    assert run_analysis_script(lambda run: bad, "spot", argv=[str(image), str(tmp_path)]) == 1
    assert not (tmp_path / "cell007_spot.json").exists()


def test_list_result_is_written_unchanged(image, tmp_path):
    """The ia_PoC_002 targets.json contract: a plain JSON array."""
    targets = [{"id": 1, "abs_x_m": 0.01, "abs_y_m": 0.02}]
    assert run_analysis_script(lambda run: targets, "targets", argv=[str(image), str(tmp_path)]) == 0
    assert json.loads((tmp_path / "cell007_targets.json").read_text()) == targets
    assert "Provenance" in (tmp_path / "cell007_targets.log").read_text(encoding="utf-8")


def test_failed_qc_step_keeps_the_result(image, tmp_path):
    def draw_overlay(path):
        raise ValueError("bad colour map")

    def analyse(run):
        assert run.qc(draw_overlay, run.output_path("spot.png")) is None
        return {"found": False}

    assert run_analysis_script(analyse, "spot", argv=[str(image), str(tmp_path)]) == 0
    res = json.loads((tmp_path / "cell007_spot.json").read_text())
    assert res["found"] is False
    assert res["provenance"]["qc_errors"] == ["draw_overlay: ValueError('bad colour map')"]
    assert "QC step draw_overlay failed" in (tmp_path / "cell007_spot.log").read_text(encoding="utf-8")


def run_template(*args):
    """Run the template as SmartMic does: a subprocess, stdout captured."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PIXI_")}
    return subprocess.run([sys.executable, str(TEMPLATE), *map(str, args)],
                          capture_output=True, text=True, env=env, timeout=120)


def test_template_end_to_end(data, tmp_path):
    czi = data / "WT_A1_cell001_recentre.czi"           # widefield DAPI, nuclei
    p = run_template(czi, tmp_path, "--prefix", "WT_A1_cell001", "--bin", "8")
    assert p.returncode == 0, p.stdout + p.stderr
    res = json.loads((tmp_path / "WT_A1_cell001_brightest.json").read_text())
    assert res["found"] and res["snr"] > 10
    # the spot is inside the field of view: 4096 x 3008 px of 0.0864 um
    assert abs(res["dx_m"]) < 177e-6 and abs(res["dy_m"]) < 130e-6
    assert res["provenance"]["options"]["bin"] == 8
    assert (tmp_path / "WT_A1_cell001_brightest.png").exists()
    assert "Result written" in p.stdout                   # SmartMic copies stdout into its run log


def test_template_nothing_found_is_exit_0(data, tmp_path):
    p = run_template(data / "WT_A1_cell001_recentre.czi", tmp_path, "--k-sigma", "1e9")
    assert p.returncode == 0, p.stdout + p.stderr
    res = json.loads((tmp_path / "WT_A1_cell001_recentre_brightest.json").read_text())
    assert res["found"] is False and res["x_m"] is None
