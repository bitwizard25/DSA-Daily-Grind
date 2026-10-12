from datetime import datetime

import pytest

from google_prep.db import get_session

NOW = datetime(2026, 10, 12, 7, 30)


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """Every test gets its own database and a fixed clock (Monday 07:30)."""
    monkeypatch.setenv("GOOGLE_PREP_HOME", str(tmp_path))
    monkeypatch.setenv("GOOGLE_PREP_NOW", NOW.isoformat())
    monkeypatch.setenv("GOOGLE_PREP_NO_EDITOR", "1")
    return tmp_path


@pytest.fixture
def session(home):
    with get_session() as s:
        yield s
