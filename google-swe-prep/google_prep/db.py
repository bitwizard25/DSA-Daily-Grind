"""SQLite engine, schema creation and seeding from the bundled JSON data."""
import json
import os
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine, select

from google_prep.models import Attempt, DrillProgress, MockSession, Problem, RecallDrill, Setting, UserProgress

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


RENAMES = {  # (table, old column) -> new column, applied before anything else so data is kept
    ("dailyschedule", "date"): "date_str",
    ("dailyschedule", "total_minutes_spent"): "total_minutes",
    ("starstory", "updated_at"): "created_at",
    ("mocksession", "day"): "date",
    ("mocksession", "used_seconds"): "duration_seconds",
}
LEGACY_RUBRIC = {"algorithms", "coding", "communication", "problem_solving", "checklist_done"}


def _columns(conn, table: str) -> set[str]:
    return {row[1] for row in conn.exec_driver_sql(f'PRAGMA table_info("{table}")')}


def migrate(engine) -> list[str]:
    """Bring an existing prep.db up to the current schema without losing progress:
    1. rename columns whose names changed (data kept),
    2. add new columns (nullable, as SQLite requires),
    3. convert the old 1-4 rubric scores into the 4 pass/fail pillars,
    4. drop columns that left the schema: content tables (rebuilt from bundled data) and the
       converted rubric columns, never other user data."""
    changes = []
    with engine.begin() as conn:
        for (table, old), new in RENAMES.items():
            cols = _columns(conn, table)
            if old in cols and new not in cols:
                conn.exec_driver_sql(f'ALTER TABLE "{table}" RENAME COLUMN "{old}" TO "{new}"')
                changes.append(f"{table}.{old}->{new}")
        for table in SQLModel.metadata.sorted_tables:
            existing = _columns(conn, table.name)
            if not existing:
                continue  # missing tables are created by create_all
            for column in table.columns:
                if column.name not in existing:
                    ddl_type = column.type.compile(engine.dialect)
                    conn.exec_driver_sql(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {ddl_type}')
                    changes.append(f"+{table.name}.{column.name}")
        mock_cols = _columns(conn, "mocksession")
        if {"algorithms", "coding", "problem_solving"} <= mock_cols:
            conn.exec_driver_sql(
                "UPDATE mocksession SET clarified_constraints = algorithms >= 3, stated_complexity = algorithms >= 3,"
                " clean_code = coding >= 3, follow_up_handled = problem_solving >= 3,"
                " dry_run_passed = COALESCE(dry_run_passed, 0),"
                " score = 12.5 * ((algorithms >= 3) * 2) + 25 * (coding >= 3) + 25 * (problem_solving >= 3)")
            changes.append("mocksession: rubric 1-4 -> pillars")
        if "date" in mock_cols:  # old rows stored YYYY-MM-DD; DateTime needs a time part
            conn.exec_driver_sql("UPDATE mocksession SET date = date || ' 00:00:00.000000' WHERE length(date) = 10")
        if _columns(conn, "starstory"):  # spec v2 renamed the dimension
            conn.exec_driver_sql("UPDATE starstory SET dimension = 'Disagreement' WHERE dimension = 'Collaboration'")
        for table in SQLModel.metadata.sorted_tables:
            existing = _columns(conn, table.name)
            obsolete = existing - {c.name for c in table.columns}
            if table.name not in CONTENT_TABLES:
                obsolete &= LEGACY_RUBRIC if table.name == "mocksession" else set()
            for name in obsolete:
                conn.exec_driver_sql(f'ALTER TABLE "{table.name}" DROP COLUMN "{name}"')
                changes.append(f"-{table.name}.{name}")
    return changes


def bank_problems() -> list[dict]:
    """The Google bank plus the Foundations set that is served before it."""
    return load_json("problems.json") + load_json("foundations.json")["problems"]


def bank_drills() -> list[dict]:
    return load_json("recall_drills.json") + load_json("foundations.json")["drills"]


def data_hash() -> str:
    """Fingerprint of the bundled problem and drill data."""
    import hashlib

    digest = hashlib.sha1()
    for name in ("problems.json", "recall_drills.json", "foundations.json"):
        digest.update((resource_dir() / "data" / name).read_bytes())
    return digest.hexdigest()


@lru_cache(maxsize=None)
def _engine(url: str):
    engine = create_engine(url)
    SQLModel.metadata.create_all(engine)
    migrate(engine)
    with Session(engine) as session:
        fingerprint = data_hash()
        stored = session.get(Setting, "data_hash")
        if stored is None or stored.value != fingerprint:  # re-seed only when the bundled data changed
            seed(session)
            session.merge(Setting(key="data_hash", value=fingerprint))
            session.commit()
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
    problems = bank_problems()
    drills = bank_drills()
    by_slug = {p["slug"]: p["id"] for p in problems}
    keep_ids = {p["id"] for p in problems}
    for row in problems:
        session.merge(Problem(**{**row, "edge_cases": json.dumps(row["edge_cases"])}))
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
