from datetime import timedelta

from google_prep.core import readiness, scheduler
from google_prep.models import MockSession, StarStory

from conftest import NOW


def add_mock(session, used_seconds, solved=True, limit=45):
    session.add(MockSession(problem_id="G-200", date=NOW, limit_minutes=limit, duration_seconds=used_seconds,
                            solved=solved, clarified_constraints=True, clean_code=True))
    session.commit()


def test_weights_match_spec():
    assert readiness.WEIGHTS == {"pattern_mastery": 0.40, "mock_speed": 0.30, "dry_run": 0.15, "star": 0.15}


def test_starts_at_zero(session):
    assert readiness.compute(session).index == 0


def test_mock_speed(session):
    add_mock(session, 30 * 60)  # inside the 35-minute coding target
    assert readiness.mock_speed(session) == 1.0
    add_mock(session, 70 * 60)  # slow: 35/70
    assert readiness.mock_speed(session) == (1.0 + 0.5) / 2
    add_mock(session, 10 * 60, solved=False)
    assert readiness.mock_speed(session) == 1.5 / 3


def test_dry_run_accuracy_uses_recent_attempts(session):
    for ok in (True, True, False, None):
        scheduler.log_attempt(session, "G-200", 30, True, ok, 4, NOW)
    assert readiness.dry_run_accuracy(session) == 2 / 3


def test_star_coverage_needs_two_per_dimension(session):
    for dim in readiness.STAR_DIMENSIONS:
        session.add(StarStory(dimension=dim, title="t", situation="s", task="t", action="a", result="r"))
    session.commit()
    assert readiness.star_coverage(session) == 0.5


def test_everything_maxed_is_100(session):
    for prob in session.exec(scheduler.select(scheduler.Problem)).all():
        for _ in range(3):
            scheduler.record_problem_review(session, prob.id, 5, True, NOW)
    for drill in session.exec(scheduler.select(scheduler.RecallDrill)).all():
        for _ in range(3):
            scheduler.grade_drill(session, drill.id, 5, NOW)
    for dim in readiness.STAR_DIMENSIONS * 2:
        session.add(StarStory(dimension=dim, title="t", situation="s", task="t", action="a", result="r"))
    session.commit()
    add_mock(session, 20 * 60)
    scheduler.log_attempt(session, "G-200", 20, True, True, 5, NOW)
    assert round(readiness.compute(session).index) == 100


def test_snapshots_trend_and_weekly_change(session):
    today = NOW.date()
    assert readiness.weekly_change(session, today) is None
    readiness.record_snapshot(session, today - timedelta(days=7), readiness.Readiness(0.1, 0, 0, 0))
    r = readiness.record_snapshot(session, today, readiness.Readiness(0.5, 0, 0, 0))
    assert readiness.weekly_change(session, today) == r.index - 4
    assert [round(v) for v in readiness.history(session, today)] == [4, 20]
    assert readiness.sparkline([0, 50, 100]) == "▁▅█"
