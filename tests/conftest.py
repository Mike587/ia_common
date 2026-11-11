from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"
# Collaborator sample images (Marc_SM_002, WT_A1_cell001_*): NOT published, kept only
# on the acquisition PC (git-ignored). Tests that need one are skipped elsewhere.
PRIVATE = Path(__file__).parent / "data_private"

# Large OV mosaic, not committed: read from the acquisition drive, skipped if absent.
OV_WT = Path("F:/UserData/mike/Marc_SM_002/SM_Output/2026-09-29/WT/WT_20260929_150834_ov.czi")


class _Data:
    """``data / name``: tests/data, else tests/data_private, else skip the test."""

    def __truediv__(self, name):
        for folder in (DATA, PRIVATE):
            if (folder / name).exists():
                return folder / name
        pytest.skip(f"test file not available: {name} (unpublished data, tests/data_private/)")


@pytest.fixture
def data():
    return _Data()


@pytest.fixture
def ov_wt():
    if not OV_WT.exists():
        pytest.skip(f"OV mosaic not available: {OV_WT}")
    return OV_WT
