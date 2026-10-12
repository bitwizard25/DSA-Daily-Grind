"""SQLite engine, schema creation and seeding from the bundled JSON data."""
import json
import os
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine, select

from google_prep.models import Attempt, DrillProgress, MockSession, Problem, RecallDrill, UserProgress

CONTENT_TABLES = {"problem", "recalldrill"}  # rebuilt from bundled data; never hold user progress


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
    """Bring an existing prep.db up to the current schema so an upgrade never crashes on (or
    wipes) your progress: add new columns everywhere (nullable, as SQLite requires), and drop
    columns that left the schema, but only in content tables rebuilt from bundled data."""
    changes = []
    with engine.begin() as conn:
        for table in SQLModel.metadata.sorted_tables:
            existing = {row[1] for row in conn.exec_driver_sql(f'PRAGMA table_info("{table.name}")')}
            if not existing:
                continue  # missing tables are created by create_all
            for column in table.columns:
                if column.name not in existing:
                    ddl_type = column.type.compile(engine.dialect)
                    conn.exec_driver_sql(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {ddl_type}')
                    changes.append(f"+{table.name}.{column.name}")
            if table.name in CONTENT_TABLES:
                for name in existing - {c.name for c in table.columns}:
                    conn.exec_driver_sql(f'ALTER TABLE "{table.name}" DROP COLUMN "{name}"')
                    changes.append(f"-{table.name}.{name}")
    return changes


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
    """Insert or refresh bundled problems and drills, and retire ones that left the bank.
    Progress on a retired problem moves to the bank problem with the same LeetCode slug;
    problems you imported with `add` (X- ids) are yours and are never touched."""
    problems = load_json("problems.json")
    drills = load_json("recall_drills.json")
    by_slug = {p["slug"]: p["id"] for p in problems}
    keep_ids = {p["id"] for p in problems}
    for row in problems:
        session.merge(Problem(**row))
    for row in drills:
        session.merge(RecallDrill(**row))
    session.flush()
    for old in session.exec(select(Problem)).all():
        if old.id in keep_ids or old.id.startswith("X-"):
            continue
        _move_history(session, old.id, by_slug.get(old.slug))
        session.delete(old)
    keep_drills = {d["id"] for d in drills}
    for old in session.exec(select(RecallDrill)).all():
        if old.id not in keep_drills:
            progress = session.get(DrillProgress, old.id)
            if progress:
                session.delete(progress)
            session.delete(old)
    session.commit()


def _move_history(session: Session, old_id: str, new_id: str | None) -> None:
    progress = session.exec(select(UserProgress).where(UserProgress.problem_id == old_id)).first()
    if progress:
        taken = new_id and session.exec(select(UserProgress).where(UserProgress.problem_id == new_id)).first()
        if new_id and not taken:
            progress.problem_id = new_id
            session.add(progress)
        else:
            session.delete(progress)
    if new_id:  # attempts and mocks are real practice history: keep them, re-pointed
        for model in (Attempt, MockSession):
            for row in session.exec(select(model).where(model.problem_id == old_id)).all():
                row.problem_id = new_id
                session.add(row)
