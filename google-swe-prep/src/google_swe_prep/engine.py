"""Business logic: seeding, scheduling, spaced repetition, readiness index."""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta

from google_swe_prep.models import Drill, Problem
from google_swe_prep.seed_data import COMPETENCIES, DRILLS, PROBLEMS, WEEKS

BOX_DAYS = [0, 1, 3, 7, 14, 30]
MAX_BOX = len(BOX_DAYS) - 1
DIFF_ORDER = {"Easy": 0, "Medium": 1, "Hard": 2}

WEIGHT_MASTERY = 0.5
WEIGHT_MOCK = 0.3
WEIGHT_GOOGLY = 0.2


def seed(conn: sqlite3.Connection) -> None:
    conn.executemany(
        "INSERT OR IGNORE INTO problems(title, topic, difficulty, phase, google_freq, pattern)"
        " VALUES (?,?,?,?,?,?)",
        PROBLEMS,
    )
    conn.executemany(
        "INSERT OR IGNORE INTO recall_drills(title, topic, prompt, answer) VALUES (?,?,?,?)",
        DRILLS,
    )
    conn.commit()


# ---- schedule ---------------------------------------------------------------

def start_date(conn: sqlite3.Connection) -> date:
    row = conn.execute("SELECT value FROM meta WHERE key='start_date'").fetchone()
    return date.fromisoformat(row["value"])


def reset_start(conn: sqlite3.Connection, day: date) -> None:
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('start_date', ?)", (day.isoformat(),))
    conn.commit()


def week_of(conn: sqlite3.Connection, today: date | None = None) -> int:
    today = today or date.today()
    days = max(0, (today - start_date(conn)).days)
    return min(len(WEEKS), days // 7 + 1)


def phase_of_week(week: int) -> int:
    return WEEKS[week - 1][1]


# ---- problems ---------------------------------------------------------------

def _problem(row: sqlite3.Row) -> Problem:
    return Problem(**{k: row[k] for k in Problem.__dataclass_fields__})


def get_problem(conn: sqlite3.Connection, pid: int) -> Problem | None:
    row = conn.execute("SELECT * FROM problems WHERE id=?", (pid,)).fetchone()
    return _problem(row) if row else None


def unsolved_problems(conn: sqlite3.Connection) -> list[Problem]:
    rows = conn.execute(
        "SELECT * FROM problems WHERE id NOT IN (SELECT problem_id FROM attempts WHERE solved=1)"
    ).fetchall()
    return [_problem(r) for r in rows]


def pick_problems(conn: sqlite3.Connection, phase: int, n: int, easy_only: bool = False) -> list[Problem]:
    pool = unsolved_problems(conn)
    if easy_only:
        pool = sorted(pool, key=lambda p: (DIFF_ORDER[p.difficulty], abs(p.phase - phase), -p.google_freq, p.id))
    else:
        # current phase first, then earlier phases (catch-up), then later ones
        def rank(p: Problem) -> tuple:
            bucket = 0 if p.phase == phase else (1 if p.phase < phase else 2)
            return (bucket, -p.google_freq, p.id)
        pool = sorted(pool, key=rank)
    return pool[:n]


def log_attempt(conn: sqlite3.Connection, pid: int, minutes: int, solved: bool,
                mode: str = "deep", day: date | None = None) -> None:
    conn.execute(
        "INSERT INTO attempts(problem_id, day, minutes, solved, mode) VALUES (?,?,?,?,?)",
        (pid, (day or date.today()).isoformat(), minutes, int(solved), mode),
    )
    conn.commit()


# ---- recall (Leitner boxes) -----------------------------------------------

def _drill(row: sqlite3.Row) -> Drill:
    return Drill(**{k: row[k] for k in Drill.__dataclass_fields__})


def due_drills(conn: sqlite3.Connection, limit: int | None = None, today: date | None = None) -> list[Drill]:
    today_s = (today or date.today()).isoformat()
    rows = conn.execute(
        """SELECT d.* FROM recall_drills d LEFT JOIN recall_state s ON s.drill_id = d.id
           WHERE s.drill_id IS NULL OR s.due <= ?
           ORDER BY COALESCE(s.box, -1), d.id""",
        (today_s,),
    ).fetchall()
    drills = [_drill(r) for r in rows]
    return drills[:limit] if limit else drills


def grade_drill(conn: sqlite3.Connection, drill_id: int, correct: bool, today: date | None = None) -> int:
    """Move the drill between boxes; returns the new box."""
    today = today or date.today()
    row = conn.execute("SELECT box FROM recall_state WHERE drill_id=?", (drill_id,)).fetchone()
    box = row["box"] if row else 0
    box = min(MAX_BOX, box + 1) if correct else 0
    due = today + timedelta(days=BOX_DAYS[box] if correct else 1)
    conn.execute(
        "INSERT INTO recall_state(drill_id, box, due) VALUES (?,?,?) "
        "ON CONFLICT(drill_id) DO UPDATE SET box=excluded.box, due=excluded.due",
        (drill_id, box, due.isoformat()),
    )
    conn.commit()
    return box


# ---- mocks ------------------------------------------------------------------

def pick_mock_problem(conn: sqlite3.Connection, phase: int) -> Problem | None:
    """Prefer high-frequency problems the user already attempted but failed, then unsolved ones."""
    failed = conn.execute(
        """SELECT p.* FROM problems p
           WHERE p.id IN (SELECT problem_id FROM attempts WHERE solved=0)
             AND p.id NOT IN (SELECT problem_id FROM attempts WHERE solved=1)
           ORDER BY p.google_freq DESC, p.id LIMIT 1"""
    ).fetchone()
    if failed:
        return _problem(failed)
    picks = pick_problems(conn, phase, 1)
    return picks[0] if picks else None


def record_mock(conn: sqlite3.Connection, pid: int, limit_min: int, used_min: int,
                solved: bool, communication: int, day: date | None = None) -> None:
    conn.execute(
        "INSERT INTO mocks(problem_id, day, limit_min, used_min, solved, communication) VALUES (?,?,?,?,?,?)",
        (pid, (day or date.today()).isoformat(), limit_min, used_min, int(solved), communication),
    )
    conn.execute(
        "INSERT INTO attempts(problem_id, day, minutes, solved, mode) VALUES (?,?,?,?,'mock')",
        (pid, (day or date.today()).isoformat(), used_min, int(solved)),
    )
    conn.commit()


# ---- STAR -------------------------------------------------------------------

def add_story(conn: sqlite3.Connection, competency: str, title: str, situation: str,
              task: str, action: str, result: str) -> int:
    cur = conn.execute(
        "INSERT INTO star_stories(competency, title, situation, task, action, result, created)"
        " VALUES (?,?,?,?,?,?,?)",
        (competency, title, situation, task, action, result, date.today().isoformat()),
    )
    conn.commit()
    return cur.lastrowid


def uncovered_competencies(conn: sqlite3.Connection) -> list[str]:
    have = {r["competency"] for r in conn.execute("SELECT DISTINCT competency FROM star_stories")}
    return [c for c in COMPETENCIES if c not in have]


# ---- readiness --------------------------------------------------------------

def mastery_score(conn: sqlite3.Connection) -> float:
    total = conn.execute("SELECT COALESCE(SUM(google_freq),0) t FROM problems").fetchone()["t"]
    solved = conn.execute(
        "SELECT COALESCE(SUM(google_freq),0) t FROM problems "
        "WHERE id IN (SELECT problem_id FROM attempts WHERE solved=1)"
    ).fetchone()["t"]
    problem_part = solved / total if total else 0.0
    drills = conn.execute("SELECT COUNT(*) c FROM recall_drills").fetchone()["c"]
    boxes = conn.execute("SELECT COALESCE(SUM(box),0) b FROM recall_state").fetchone()["b"]
    recall_part = boxes / (drills * MAX_BOX) if drills else 0.0
    return 0.7 * problem_part + 0.3 * recall_part


def mock_speed_score(conn: sqlite3.Connection) -> float:
    rows = conn.execute(
        "SELECT limit_min, used_min, solved, communication FROM mocks ORDER BY id DESC LIMIT 5"
    ).fetchall()
    if not rows:
        return 0.0
    scores = []
    for r in rows:
        if not r["solved"]:
            scores.append(0.0)
            continue
        speed = min(1.0, r["limit_min"] / max(1, r["used_min"]))
        scores.append(0.7 * speed + 0.3 * r["communication"] / 5)
    return sum(scores) / len(scores)


def googlyness_score(conn: sqlite3.Connection) -> float:
    return 1 - len(uncovered_competencies(conn)) / len(COMPETENCIES)


def readiness(conn: sqlite3.Connection) -> dict[str, float]:
    m, s, g = mastery_score(conn), mock_speed_score(conn), googlyness_score(conn)
    index = 100 * (WEIGHT_MASTERY * m + WEIGHT_MOCK * s + WEIGHT_GOOGLY * g)
    return {"mastery": m, "mock_speed": s, "googlyness": g, "index": index}
