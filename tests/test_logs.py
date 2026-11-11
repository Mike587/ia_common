import json
from pathlib import Path

import numpy as np
import pytest

from ia_common.logs import setup_logging, write_json


def test_write_json_converts_numpy(tmp_path):
    p = tmp_path / "r.json"
    write_json({"n": np.int64(3), "x": np.float32(0.5), "ok": np.bool_(True),
                "a": np.arange(3), "p": Path("a/b")}, p)
    assert json.loads(p.read_text(encoding="utf-8")) == {"n": 3, "x": 0.5, "ok": True,
                                                         "a": [0, 1, 2], "p": str(Path("a/b"))}


def test_write_json_is_atomic(tmp_path):
    p = tmp_path / "r.json"
    write_json({"found": True}, p)
    with pytest.raises(TypeError):
        write_json({"found": object()}, p)            # fails half-way through the dump
    assert json.loads(p.read_text()) == {"found": True}   # old file untouched
    assert [f.name for f in tmp_path.iterdir()] == ["r.json"]   # no temp file left


def test_setup_logging_file_and_no_duplicate_handlers(tmp_path):
    log = setup_logging(tmp_path / "sub" / "run.log", "t")
    log = setup_logging(tmp_path / "sub" / "run.log", "t")
    log.info("hello")
    assert len(log.handlers) == 2
    for h in log.handlers:
        h.flush()
    assert "hello" in (tmp_path / "sub" / "run.log").read_text(encoding="utf-8")
