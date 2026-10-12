from datetime import datetime, timedelta

import pytest

from google_prep.core import scheduler
from google_prep.db import get_session

NOW = datetime(2026, 10, 12, 7, 30)


PLAN_START = NOW.date() - timedelta(days=2)  # tests run on plan Day 3: Phase 1, after the fast-track


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch, request):
    """Every test gets its own database and a fixed clock (Monday 07:30). The plan starts two days
    earlier so most tests exercise Phase 1; tests marked `fasttrack` start on Day 1 (Phase 0)."""
    monkeypatch.setenv("GOOGLE_PREP_HOME", str(tmp_path))
    monkeypatch.setenv("GOOGLE_PREP_NOW", NOW.isoformat())
    monkeypatch.setenv("GOOGLE_PREP_NO_EDITOR", "1")
    if "fasttrack" not in request.keywords:
        with get_session() as s:
            scheduler.set_setting(s, "start_date", PLAN_START.isoformat())
    return tmp_path


@pytest.fixture
def session(home):
    with get_session() as s:
        yield s
