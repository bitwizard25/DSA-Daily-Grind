import pytest

from google_swe_prep import engine
from google_swe_prep.models import connect


@pytest.fixture
def db_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GSP_DB", str(tmp_path / "t.db"))
    return tmp_path / "t.db"


@pytest.fixture
def conn(db_env):
    c = connect()
    engine.seed(c)
    return c
