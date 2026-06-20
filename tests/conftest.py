"""Shared pytest fixtures.

The synthetic test data is generated on demand so the suite is self-contained and does
not depend on large binary blobs being checked in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import make_fixtures

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session", autouse=True)
def _ensure_fixtures() -> None:
    """Regenerate fixtures before the test session if missing."""
    if not (DATA / "alphafold_db").exists():
        make_fixtures.main()


@pytest.fixture
def colabfold_dir() -> Path:
    return DATA / "colabfold"


@pytest.fixture
def af3_dir() -> Path:
    return DATA / "af3_server"


@pytest.fixture
def boltz_dir() -> Path:
    return DATA / "boltz"


@pytest.fixture
def openfold3_dir() -> Path:
    return DATA / "openfold3"


@pytest.fixture
def single_chain_dir() -> Path:
    return DATA / "single_chain"


@pytest.fixture
def malformed_dir() -> Path:
    return DATA / "malformed"


@pytest.fixture
def empty_dir() -> Path:
    return DATA / "empty"


@pytest.fixture
def alphafold_db_dir() -> Path:
    return DATA / "alphafold_db"
