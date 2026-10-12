from datetime import datetime

from google_prep import db
from google_prep.core import scheduler

PROBLEM_FIELDS = {"id", "title", "leetcode_id", "slug", "difficulty", "category", "phase", "key_pattern",
                  "key_insight", "time_target", "space_target", "google_frequency", "recency", "recent_frequency",
                  "statement", "edge_cases", "google_follow_up"}
SIX = {"graphs", "trees", "heaps", "intervals", "dp", "monotonic"}


def test_problem_bank():
    problems = db.load_json("problems.json")
    assert len(problems) == 75
    for key in ("id", "slug", "leetcode_id", "title"):
        assert len({p[key] for p in problems}) == 75, key
    for p in problems:
        assert set(p) == PROBLEM_FIELDS, p["id"]
        assert p["id"] == f"G-{p['leetcode_id']}"  # stable ids: the LeetCode number never changes
        assert p["difficulty"] in ("Easy", "Medium", "Hard")
        assert p["category"] in scheduler.CATEGORIES
        assert p["phase"] in (1, 2, 3)
        assert p["recency"] in ("30d", "3m", "6m")  # every bank problem was asked in the last 6 months
        assert p["statement"] and p["google_follow_up"] and p["key_insight"]
        assert isinstance(p["edge_cases"], list) and p["edge_cases"] and all(p["edge_cases"])
    assert {p["category"] for p in problems} == set(scheduler.CATEGORIES) == SIX
    share = {c: sum(p["category"] == c for p in problems) / 75 for c in SIX}
    # spec section 4 weights: Graphs 35%, Trees 20%, DP 20%, Concurrency/Queues/Intervals 25%
    assert round(share["graphs"], 2) == 0.35 and round(share["trees"], 2) == 0.20 and round(share["dp"], 2) == 0.20
    assert round(share["heaps"] + share["intervals"] + share["monotonic"], 2) == 0.25
    for named in ("Course Schedule", "Course Schedule II", "Evaluate Division", "Network Delay Time", "Alien Dictionary",
                  "Meeting Rooms II", "Merge Intervals", "Sliding Window Maximum", "Task Scheduler",
                  "Lowest Common Ancestor of a Binary Tree", "Word Search II", "Binary Tree Maximum Path Sum",
                  "Word Break", "Longest Increasing Subsequence", "Trapping Rain Water"):
        assert named in {p["title"] for p in problems}, named  # every problem the spec names is in the bank
    assert {p["phase"] for p in problems} == {1, 2, 3}


def test_recall_drills():
    drills = db.load_json("recall_drills.json")
    problem_ids = {p["id"] for p in db.load_json("problems.json")}
    assert len(drills) == 50 and len({d["id"] for d in drills}) == 50
    assert sum(d["type"] == "pattern" for d in drills) == 40
    assert sum(d["type"] == "bug_hunt" for d in drills) == 10
    assert {d["category"] for d in drills if d["type"] == "bug_hunt"} == SIX  # a bug-hunt for every category
    for d in drills:
        assert d["category"] in scheduler.CATEGORIES
        assert d["prompt"] and d["pattern"] and d["complexity"] and d["edge_case"]
        assert d["problem_id"] is None or d["problem_id"] in problem_ids


def test_patterns_and_vault():
    patterns = db.load_json("patterns.json")
    assert len(patterns) == 15
    vault = db.resource_dir() / "patterns_vault"
    assert len(list(vault.glob("*.md"))) == 9
    for p in patterns:
        assert p["vault_file"] is None or (vault / p["vault_file"]).exists()


def test_googlyness_prompts_cover_dimensions():
    prompts = db.load_json("googlyness_prompts.json")
    assert set(prompts) == {"Ambiguity", "Ownership", "Disagreement", "Humility"}
    assert all(len(v) >= 3 for v in prompts.values())


def test_seed_is_idempotent_and_keeps_progress(session):
    scheduler.grade_drill(session, "R-200", 5, scheduler.clock.now())
    db.seed(session)
    db.seed(session)
    assert scheduler.count(session, scheduler.Problem) == 75
    assert scheduler.count(session, scheduler.RecallDrill) == 50
    assert scheduler.count(session, scheduler.DrillProgress) == 1


def test_migration_adds_columns_to_an_old_database(home):
    import sqlite3

    from sqlmodel import create_engine

    old = home / "old.db"
    con = sqlite3.connect(old)
    con.execute("CREATE TABLE attempt (id INTEGER PRIMARY KEY, problem_id VARCHAR, day VARCHAR, minutes INTEGER,"
                " solved BOOLEAN, source VARCHAR)")
    con.execute("INSERT INTO attempt VALUES (1, 'G-01', '2026-10-01', 30, 1, 'solve')")
    con.commit()
    con.close()
    added = db.migrate(create_engine(f"sqlite:///{old}"))
    assert {"+attempt.dry_run_ok", "+attempt.mistake"} <= set(added)
    con = sqlite3.connect(old)
    assert con.execute("SELECT problem_id, mistake FROM attempt").fetchone() == ("G-01", None)


def test_upgrade_moves_progress_from_retired_ids(session):
    """A bank change must never strand progress: old ids map to new ones by LeetCode slug."""
    from google_prep.models import Attempt, DrillProgress, Problem, RecallDrill, UserProgress

    islands = session.get(Problem, "G-200")
    old = Problem(**{**islands.model_dump(), "id": "G-47"})  # how Number of Islands was keyed before
    gone = Problem(**{**islands.model_dump(), "id": "G-02", "slug": "product-of-array-except-self"})
    session.add_all([old, gone, RecallDrill(id="R-01", type="pattern", category="graphs", prompt="p", pattern="p",
                                            complexity="c", edge_case="e")])
    session.add_all([UserProgress(problem_id="G-47", status="learning", reps=2, next_review_date=datetime(2026, 10, 1)),
                     UserProgress(problem_id="G-02", status="learning", next_review_date=datetime(2026, 10, 1)),
                     Attempt(problem_id="G-47", day="2026-10-01", minutes=30, solved=True),
                     Attempt(problem_id="G-02", day="2026-10-01", minutes=30, solved=False, mistake="pattern"),
                     DrillProgress(drill_id="R-01", next_review_date=datetime(2026, 10, 1))])
    session.commit()
    db.seed(session)
    assert session.get(Problem, "G-47") is None and session.get(Problem, "G-02") is None
    moved = session.exec(scheduler.select(UserProgress)).all()
    assert [(p.problem_id, p.reps) for p in moved] == [("G-200", 2)]
    history = sorted(a.problem_id for a in session.exec(scheduler.select(Attempt)).all())
    assert history == ["G-02", "G-200"]  # retired problem's attempt stays as history for the mistake profile
    assert session.get(RecallDrill, "R-01") is None and session.get(DrillProgress, "R-01") is None
    assert [p.id for p in scheduler.attempts_on(session, datetime(2026, 10, 1).date())] == ["G-200"]


def test_migration_drops_renamed_content_columns(home):
    import sqlite3

    from sqlmodel import create_engine

    old = home / "renamed.db"
    con = sqlite3.connect(old)
    con.execute("CREATE TABLE problem (id VARCHAR PRIMARY KEY, title VARCHAR, follow_up_prompt VARCHAR NOT NULL)")
    con.execute("CREATE TABLE starstory (id INTEGER PRIMARY KEY, dimension VARCHAR, legacy_note VARCHAR)")
    con.commit()
    con.close()
    changes = db.migrate(create_engine(f"sqlite:///{old}"))
    assert "-problem.follow_up_prompt" in changes and "+problem.google_follow_up" in changes
    assert "-starstory.legacy_note" not in changes  # user tables are never stripped


def test_upgrade_from_previous_release_keeps_history(home):
    """A prep.db written by the 6-category release opens cleanly and keeps its history."""
    import sqlite3

    from sqlmodel import Session, create_engine

    from google_prep.models import DailySchedule, MockSession, StarStory

    old = home / "v11.db"
    con = sqlite3.connect(old)
    con.executescript("""
        CREATE TABLE dailyschedule (date VARCHAR PRIMARY KEY, mode VARCHAR NOT NULL, morning_completed BOOLEAN NOT NULL,
            evening_completed BOOLEAN NOT NULL, total_minutes_spent INTEGER NOT NULL);
        INSERT INTO dailyschedule VALUES ('2026-10-11', 'normal', 1, 0, 25);
        CREATE TABLE starstory (id INTEGER PRIMARY KEY, dimension VARCHAR NOT NULL, title VARCHAR NOT NULL,
            situation VARCHAR NOT NULL, task VARCHAR NOT NULL, action VARCHAR NOT NULL, result VARCHAR NOT NULL,
            updated_at DATETIME NOT NULL);
        INSERT INTO starstory VALUES (1, 'Collaboration', 't', 's', 't', 'a', 'r', '2026-10-11 09:00:00.000000');
        CREATE TABLE mocksession (id INTEGER PRIMARY KEY, problem_id VARCHAR NOT NULL, day VARCHAR NOT NULL,
            limit_minutes INTEGER NOT NULL, used_seconds INTEGER NOT NULL, solved BOOLEAN NOT NULL,
            follow_up VARCHAR NOT NULL, checklist_done INTEGER NOT NULL, algorithms INTEGER NOT NULL,
            coding INTEGER NOT NULL, communication INTEGER NOT NULL, problem_solving INTEGER NOT NULL);
        INSERT INTO mocksession VALUES (1, 'G-200', '2026-10-11', 45, 1800, 1, 'stream it', 2, 4, 3, 2, 1);
    """)
    con.commit()
    con.close()
    engine = create_engine(f"sqlite:///{old}")
    from sqlmodel import SQLModel
    SQLModel.metadata.create_all(engine)
    changes = db.migrate(engine)
    assert "dailyschedule.date->date_str" in changes and "mocksession.used_seconds->duration_seconds" in changes
    with Session(engine) as s:
        day = s.get(DailySchedule, "2026-10-11")
        assert day.total_minutes == 25 and day.morning_completed
        story = s.get(StarStory, 1)
        assert story.dimension == "Disagreement" and story.created_at.day == 11
        m = s.get(MockSession, 1)
        assert m.duration_seconds == 1800 and m.date.year == 2026 and m.solved
        assert (m.clarified_constraints, m.clean_code, m.follow_up_handled) == (True, True, False)
        assert m.score == 50
        s.add(MockSession(problem_id="G-207", duration_seconds=60))  # new rows insert fine after the upgrade
        s.commit()
