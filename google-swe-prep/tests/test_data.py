from google_prep import db
from google_prep.core import scheduler

PROBLEM_FIELDS = {"id", "title", "leetcode_id", "slug", "difficulty", "category", "phase", "key_pattern",
                  "time_target", "space_target", "google_frequency", "recency", "recent_frequency",
                  "statement", "edge_case", "follow_up_prompt"}


def test_problem_bank():
    problems = db.load_json("problems.json")
    assert len(problems) == 75
    for key in ("id", "slug", "leetcode_id", "title"):
        assert len({p[key] for p in problems}) == 75, key
    for p in problems:
        assert set(p) == PROBLEM_FIELDS, p["id"]
        assert p["difficulty"] in ("Easy", "Medium", "Hard")
        assert p["category"] in scheduler.CATEGORIES
        assert p["phase"] in (1, 2, 3)
        assert p["recency"] in ("30d", "3m", "6m")  # every bank problem was asked in the last 6 months
        assert p["statement"] and p["follow_up_prompt"] and p["edge_case"]
    assert {p["category"] for p in problems} == set(scheduler.CATEGORIES)
    assert {p["phase"] for p in problems} == {1, 2, 3}


def test_recall_drills():
    drills = db.load_json("recall_drills.json")
    problem_ids = {p["id"] for p in db.load_json("problems.json")}
    assert len(drills) == 50 and len({d["id"] for d in drills}) == 50
    assert sum(d["type"] == "pattern" for d in drills) == 40
    assert sum(d["type"] == "bug_hunt" for d in drills) == 10
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
    assert set(prompts) == {"Ambiguity", "Ownership", "Collaboration", "Humility"}
    assert all(len(v) >= 3 for v in prompts.values())


def test_seed_is_idempotent_and_keeps_progress(session):
    scheduler.grade_drill(session, "R-01", 5, scheduler.clock.now())
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
    assert {"attempt.dry_run_ok", "attempt.mistake"} <= set(added)
    con = sqlite3.connect(old)
    assert con.execute("SELECT problem_id, mistake FROM attempt").fetchone() == ("G-01", None)
