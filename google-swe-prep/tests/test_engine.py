from datetime import date, timedelta

from google_swe_prep import engine
from google_swe_prep.models import connect
from google_swe_prep.seed_data import COMPETENCIES, DRILLS, PROBLEMS


def test_seed_is_idempotent(conn):
    engine.seed(conn)
    assert conn.execute("SELECT COUNT(*) FROM problems").fetchone()[0] == len(PROBLEMS)
    assert conn.execute("SELECT COUNT(*) FROM recall_drills").fetchone()[0] == len(DRILLS)


def test_problem_bank_covers_all_phases(conn):
    phases = {r[0] for r in conn.execute("SELECT DISTINCT phase FROM problems")}
    assert phases == {1, 2, 3}


def test_week_progression(conn):
    engine.reset_start(conn, date.today() - timedelta(days=15))
    assert engine.week_of(conn) == 3
    engine.reset_start(conn, date.today() - timedelta(days=400))
    assert engine.week_of(conn) == 8
    assert engine.phase_of_week(8) == 3


def test_pick_prefers_current_phase_then_freq(conn):
    picks = engine.pick_problems(conn, 2, 3)
    assert all(p.phase == 2 for p in picks)
    assert picks[0].google_freq >= picks[-1].google_freq


def test_solved_problem_not_repicked(conn):
    first = engine.pick_problems(conn, 1, 1)[0]
    engine.log_attempt(conn, first.id, 30, True)
    assert first.id not in {p.id for p in engine.pick_problems(conn, 1, 50)}


def test_micro_picks_easy(conn):
    assert engine.pick_problems(conn, 1, 1, easy_only=True)[0].difficulty == "Easy"


def test_leitner_promotion_and_reset(conn):
    d = engine.due_drills(conn, limit=1)[0]
    assert engine.grade_drill(conn, d.id, True) == 1
    assert engine.grade_drill(conn, d.id, True) == 2
    assert engine.grade_drill(conn, d.id, False) == 0
    # promoted drills leave the due list; failed one is due tomorrow, not today
    assert d.id not in {x.id for x in engine.due_drills(conn)}
    tomorrow = date.today() + timedelta(days=1)
    assert d.id in {x.id for x in engine.due_drills(conn, today=tomorrow)}


def test_box_caps_at_max(conn):
    d = engine.due_drills(conn, limit=1)[0]
    for _ in range(10):
        box = engine.grade_drill(conn, d.id, True)
    assert box == engine.MAX_BOX


def test_readiness_starts_at_zero_and_grows(conn):
    assert engine.readiness(conn)["index"] == 0
    p = engine.pick_problems(conn, 1, 1)[0]
    engine.record_mock(conn, p.id, 45, 30, True, 5)
    r = engine.readiness(conn)
    assert r["mock_speed"] == 1.0
    assert 0 < r["index"] < 100


def test_mock_unsolved_scores_zero_and_slow_penalised(conn):
    p = engine.pick_problems(conn, 1, 1)[0]
    engine.record_mock(conn, p.id, 45, 45, False, 5)
    assert engine.mock_speed_score(conn) == 0
    engine.record_mock(conn, p.id, 30, 60, True, 5)
    assert 0 < engine.mock_speed_score(conn) < 0.5


def test_mock_picks_failed_problem_first(conn):
    p = engine.pick_problems(conn, 3, 1)[-1]
    engine.log_attempt(conn, p.id, 40, False)
    assert engine.pick_mock_problem(conn, 1).id == p.id


def test_googlyness_coverage(conn):
    assert engine.googlyness_score(conn) == 0
    for c in COMPETENCIES:
        engine.add_story(conn, c, "t", "s", "t", "a", "r")
    assert engine.googlyness_score(conn) == 1
    assert engine.uncovered_competencies(conn) == []


def test_perfect_readiness_is_100(conn):
    for p in engine.unsolved_problems(conn):
        engine.log_attempt(conn, p.id, 20, True)
    for d in engine.due_drills(conn):
        for _ in range(engine.MAX_BOX):
            engine.grade_drill(conn, d.id, True)
    for c in COMPETENCIES:
        engine.add_story(conn, c, "t", "s", "t", "a", "r")
    p = engine.get_problem(conn, 1)
    engine.record_mock(conn, p.id, 45, 20, True, 5)
    assert round(engine.readiness(conn)["index"]) == 100


def test_db_persists(db_env):
    c = connect()
    engine.seed(c)
    engine.log_attempt(c, 1, 10, True)
    c.close()
    assert connect().execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1
