import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def examples() -> pathlib.Path:
    return REPO / "examples"
