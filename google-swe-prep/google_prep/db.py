"""SQLite engine, schema creation and seeding from the bundled JSON data."""
import json
import os
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

from google_prep.models import Problem, RecallDrill


def resource_dir() -> Path:
    """Folder holding data/ and patterns_vault/ (inside the PyInstaller bundle when frozen)."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


def load_json(name: str):
    return json.loads((resource_dir() / "data" / name).read_text(encoding="utf-8"))


def home_dir() -> Path:
    path = Path(os.environ.get("GOOGLE_PREP_HOME") or Path.home() / ".google_prep")
    path.mkdir(parents=True, exist_ok=True)
    return path


def migrate(engine) -> list[str]:
    """Add columns that newer app versions define but an existing prep.db lacks, so an
    upgrade never crashes on (or wipes) your progress. SQLite can only add nullable columns."""
    added = []
    with engine.begin() as conn:
        for table in SQLModel.metadata.sorted_tables:
            existing = {row[1] for row in conn.exec_driver_sql(f'PRAGMA table_info("{table.name}")')}
            if not existing:
                continue  # missing tables are created by create_all
            for column in table.columns:
                if column.name not in existing:
                    ddl_type = column.type.compile(engine.dialect)
                    conn.exec_driver_sql(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {ddl_type}')
                    added.append(f"{table.name}.{column.name}")
    return added


@lru_cache(maxsize=None)
def _engine(url: str):
    engine = create_engine(url)
    SQLModel.metadata.create_all(engine)
    migrate(engine)
    with Session(engine) as session:
        seed(session)
    return engine


def get_engine():
    return _engine(f"sqlite:///{home_dir() / 'prep.db'}")


@contextmanager
def get_session():
    with Session(get_engine(), expire_on_commit=False) as session:
        yield session


def seed(session: Session) -> None:
    """Insert or refresh bundled problems and drills. User progress is never touched."""
    for row in load_json("problems.json"):
        session.merge(Problem(**row))
    for row in load_json("recall_drills.json"):
        session.merge(RecallDrill(**row))
    session.commit()
