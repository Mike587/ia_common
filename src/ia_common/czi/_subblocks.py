"""
Read-only access to the metadata stored with every subblock (image tile) of a
CZI, which pylibCZIrw 6.1 does not expose. Per subblock it holds, among
others (µm in the file, metres here):

  StageXPosition / StageYPosition   where the stage was when the tile was taken
  FocusPosition                     focus drive Z
  RoiCenterOffsetX / Y              camera ROI centre minus chip centre
                                    (LSM: presumably the scan offset, not verified)
  RoiRotation                       LSM scan rotation

The file layout (ZISRAW, Zeiss' CZI file-format specification):
  * the file header segment ZISRAWFILE holds the position of the subblock
    directory;
  * the directory segment ZISRAWDIRECTORY lists one entry per subblock: its
    file position and its dimension indices (C, Z, S, M, X, Y, ...);
  * each subblock segment ZISRAWSUBBLOCK starts with the sizes of its parts
    and a copy of its directory entry, padded to at least 240 bytes; the
    metadata XML follows.
Every segment starts with a 32-byte header: id (16 bytes), allocated size,
used size. Checked against aicspylibczi's read_subblock_metadata on the CD7
test files and example images (2026-10-14).
"""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .meta import MetadataError

_SEGMENT_HEADER = struct.Struct("<16sqq")    # id, allocated size, used size
# schema ('DV'), pixel type, file position, file part, compression, pyramid type,
# 5 spare bytes, dimension count; then 20 bytes per dimension
_ENTRY = struct.Struct("<2siqiiB5xi")
_DIMENSION = struct.Struct("<4siifi")       # name, start, size, start coordinate, stored size


@dataclass(frozen=True)
class SubBlockMeta:
    start: dict[str, int]            # index per dimension, e.g. {'C': 0, 'Z': 3, 'X': 1234, 'Y': 5678}
    size: dict[str, int]             # logical size per dimension (X, Y: pixels in the global frame)
    pyramid: bool                    # a downscaled pyramid tile, not full resolution
    tags: dict[str, str]             # METADATA/Tags as text, e.g. {'StageXPosition': '+000000012676.0830'}

    def _um(self, tag: str) -> float | None:
        v = self.tags.get(tag)
        return None if v is None else float(v) * 1e-6

    @property
    def stage_xy_m(self) -> tuple[float, float] | None:
        x, y = self._um("StageXPosition"), self._um("StageYPosition")
        return None if x is None or y is None else (x, y)

    @property
    def focus_m(self) -> float | None:
        return self._um("FocusPosition")

    @property
    def roi_offset_m(self) -> tuple[float, float]:
        """ROI centre minus chip centre (0, 0 if not recorded)."""
        return self._um("RoiCenterOffsetX") or 0.0, self._um("RoiCenterOffsetY") or 0.0

    @property
    def roi_rotation(self) -> float:
        """Scan rotation as recorded (0 if not recorded); only tested for == 0."""
        return float(self.tags.get("RoiRotation", 0.0))


def _segment(f, expected: bytes) -> None:
    sid, _, _ = _SEGMENT_HEADER.unpack(f.read(_SEGMENT_HEADER.size))
    if sid.rstrip(b"\0") != expected:
        raise MetadataError(f"expected a {expected.decode()} segment, found {sid!r}")


def _entry(f) -> tuple[int, int, int, list[tuple]]:
    _schema, _ptype, pos, part, _comp, pyramid, n = _ENTRY.unpack(f.read(_ENTRY.size))
    dims = [_DIMENSION.unpack(f.read(_DIMENSION.size)) for _ in range(n)]
    return pos, part, pyramid, dims


def read_subblock_metadata(path: str | Path) -> list[SubBlockMeta]:
    """The metadata of every subblock, in directory order (pyramid tiles included,
    flagged)."""
    out = []
    with open(path, "rb") as f:
        _segment(f, b"ZISRAWFILE")
        part, directory = struct.unpack_from("<iq", f.read(80), 48)   # FilePart, DirectoryPosition
        if part != 0:
            raise MetadataError(f"{Path(path).name}: part of a multi-file CZI - not supported")
        f.seek(directory)
        _segment(f, b"ZISRAWDIRECTORY")
        (n,) = struct.unpack("<i", f.read(4))
        f.seek(124, 1)                                                 # reserved
        entries = [_entry(f) for _ in range(n)]

        for pos, part, pyramid, dims in entries:
            if part != 0:
                raise MetadataError(f"{Path(path).name}: subblock in another file part - not supported")
            f.seek(pos)
            _segment(f, b"ZISRAWSUBBLOCK")
            meta_size, _att_size, _data_size = struct.unpack("<iiq", f.read(16))
            n_dims = _ENTRY.unpack(f.read(_ENTRY.size))[-1]
            f.seek(pos + _SEGMENT_HEADER.size + 16 + max(240, _ENTRY.size + n_dims * _DIMENSION.size))
            raw = f.read(meta_size).rstrip(b"\0")
            tags = {}
            if raw:
                for el in ET.fromstring(raw).findall("Tags/*"):
                    tags[el.tag] = (el.text or "").strip()
            start = {d[0].rstrip(b"\0 ").decode(): d[1] for d in dims}
            size = {d[0].rstrip(b"\0 ").decode(): d[2] for d in dims}
            stored_x = next((d[4] for d in dims if d[0].rstrip(b"\0 ") == b"X"), None)
            out.append(SubBlockMeta(start, size, bool(pyramid) or stored_x != size.get("X"), tags))
    return out
