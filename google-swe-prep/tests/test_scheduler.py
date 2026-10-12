from datetime import date, datetime, timedelta

import pytest

from google_prep.core import scheduler
from google_prep.models import DrillProgress, Problem, TrendingProblem

from conftest import NOW


def test_phase_boundaries():
    assert [scheduler.phase_for_day(d) for d in (1, 21, 22, 42, 43, 56, 80)] == [1, 1, 2, 2, 3, 3, 3]


@pytest.mark.parametrize("hour,slot", [(7, "morning"), (11, "morning"), (12, "midday"), (16, "midday"),
                                       (17, "evening"), (23, "evening")])
def test_time_slot(hour, slot):
    assert scheduler.time_slot(datetime(2026, 1, 1, hour)) == slot


def test_day_number_counts_from_start(session):
    scheduler.set_setting(session, "start_date", "2026-10-01")
    assert scheduler.day_number(session, date(2026, 10, 1)) == 1
    assert scheduler.day_number(session, date(2026, 10, 12)) == 12
    assert scheduler.interview_date(session) == date(2026, 11, 26)
    scheduler.set_setting(session, "interview_date", "2026-11-30")
    assert scheduler.interview_date(session) == date(2026, 11, 30)


def test_start_date_defaults_to_first_run(session):
    assert scheduler.start_date(session) == NOW.date()


def test_streak(session):
    today = NOW.date()
    for back in (1, 2, 3, 5):
        scheduler.mark_done(session, today - timedelta(days=back), "morning", 20)
    assert scheduler.streak(session, today) == 3  # today not done yet still keeps yesterday's streak
    scheduler.mark_done(session, today, "evening", 60)
    assert scheduler.streak(session, today) == 4
    assert scheduler.schedule_for(session, today).total_minutes_spent == 60


def test_needs_micro():
    assert not scheduler.needs_micro(None)
    assert not scheduler.needs_micro(9.5)
    assert scheduler.needs_micro(10)


def test_target_prefers_recent_non_easy_in_weakest_topic(session):
    p = scheduler.pick_target_problem(session, 1, NOW)
    assert p.category == "arrays"  # all tied at 0: plan order
    assert p.difficulty != "Easy" and p.recency == "30d"


def test_target_moves_to_next_weakest_topic(session):
    for prob in session.exec(scheduler.select(Problem).where(Problem.category == "arrays")).all()[:6]:
        scheduler.log_attempt(session, prob.id, 30, True, True, 5, NOW)
    assert scheduler.pick_target_problem(session, 1, NOW).category != "arrays"


def test_topic_filter_and_failed_problem_comes_back_first(session):
    p = scheduler.pick_target_problem(session, 1, NOW, ["graphs"])
    assert p.category == "graphs"
    scheduler.log_attempt(session, p.id, 45, False, False, 2, NOW, mistake="pattern")
    later = NOW + timedelta(days=2)
    assert scheduler.pick_target_problem(session, 1, later, ["graphs"]).id == p.id
    assert p.id in [x.id for x in scheduler.due_problems(session, later)]


def test_sync_overrides_bundled_recency(session):
    target = scheduler.pick_target_problem(session, 1, NOW, ["arrays"])
    other = session.get(Problem, "G-03")  # Subarray Sum Equals K
    session.add(TrendingProblem(slug=other.slug, title=other.title, difficulty="Medium", window="30d", frequency=99))
    session.commit()
    assert target.id != "G-03"
    assert scheduler.pick_target_problem(session, 1, NOW, ["arrays"]).id == "G-03"


def test_pick_drills_due_first_then_round_robin(session):
    first = scheduler.pick_drills(session, NOW, 8)
    assert len({d.category for d in first}) == 8  # one per category before repeating
    scheduler.grade_drill(session, "R-07", 2, NOW - timedelta(days=3))
    assert scheduler.pick_drills(session, NOW, 1)[0].id == "R-07"


def test_graded_drill_leaves_queue_until_due(session):
    scheduler.grade_drill(session, "R-01", 5, NOW)
    assert "R-01" not in [d.id for d in scheduler.pick_drills(session, NOW, 50)]
    assert session.get(DrillProgress, "R-01").next_review_date == NOW + timedelta(days=1)


def test_warmup_adapts_to_mistakes(session):
    kinds = lambda drills: sum(d.type == "bug_hunt" for d in drills)  # noqa: E731
    assert kinds(scheduler.pick_warmup(session, NOW)) == 1
    for pid in ("G-01", "G-02"):
        scheduler.log_attempt(session, pid, 30, False, False, 2, NOW, mistake="bug_edge")
    assert scheduler.dominant_mistake(session) == "bug_edge"
    assert kinds(scheduler.pick_warmup(session, NOW)) == 2
    for pid in ("G-03", "G-04", "G-05"):
        scheduler.log_attempt(session, pid, 30, False, False, 2, NOW, mistake="pattern")
    assert kinds(scheduler.pick_warmup(session, NOW)) == 0
    assert scheduler.mistake_profile(session) == {"pattern": 3, "implementation": 0, "bug_edge": 2,
                                                  "slow": 0, "follow_up": 0}


def test_pace(session):
    scheduler.set_setting(session, "start_date", (NOW.date() - timedelta(days=13)).isoformat())  # day 14
    pace = scheduler.pace(session, NOW.date())
    assert pace["expected"] == round(75 * 13 / 56) and pace["behind"] == pace["expected"]
    scheduler.set_setting(session, "start_date", NOW.date().isoformat())
    assert scheduler.pace(session, NOW.date())["behind"] == 0
