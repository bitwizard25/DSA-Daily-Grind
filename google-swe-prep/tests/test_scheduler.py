from datetime import date, datetime, timedelta

import pytest

from google_prep.core import scheduler
from google_prep.models import DrillProgress, Problem, TrendingProblem, UserProgress

from conftest import NOW


def close_foundations(session):
    """Mark every Foundations problem as tried once (not due), as after the first two weeks."""
    tried = set(session.exec(scheduler.select(UserProgress.problem_id)).all())
    for p in session.exec(scheduler.select(Problem).where(Problem.category == scheduler.FOUNDATIONS)).all():
        if p.id not in tried:
            session.add(UserProgress(problem_id=p.id, status="learning", reps=1,
                                     next_review_date=NOW + timedelta(days=5)))
    session.commit()


def test_phase_boundaries():
    days = (1, 2, 3, 21, 22, 42, 43, 56, 80)
    assert [scheduler.phase_for_day(d) for d in days] == [0, 0, 1, 1, 2, 2, 3, 3, 3]


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
    session.delete(session.get(scheduler.Setting, "start_date"))
    session.commit()
    assert scheduler.start_date(session) == NOW.date()


def test_streak(session):
    today = NOW.date()
    for back in (1, 2, 3, 5):
        scheduler.mark_done(session, today - timedelta(days=back), "morning", 20)
    assert scheduler.streak(session, today) == 3  # today not done yet still keeps yesterday's streak
    scheduler.mark_done(session, today, "evening", 60)
    assert scheduler.streak(session, today) == 4
    assert scheduler.schedule_for(session, today).total_minutes == 60


def test_needs_micro():
    assert not scheduler.needs_micro(None)
    assert not scheduler.needs_micro(9.5)
    assert scheduler.needs_micro(10)


def test_foundations_come_first_in_order_then_the_google_bank(session):
    first = scheduler.pick_target_problem(session, 1, NOW)
    assert (first.id, first.category, first.difficulty) == ("G-1", "foundations", "Easy")  # Two Sum opens the set
    scheduler.log_attempt(session, first.id, 30, True, True, 4, NOW, min_days=3)
    assert scheduler.pick_target_problem(session, 1, NOW).sequence == 2  # then the next building block
    close_foundations(session)
    assert not scheduler.foundations_open(session)
    p = scheduler.pick_target_problem(session, 1, NOW)
    assert p.category == "arrays"  # Google problems, all tied at 0: plan order
    assert p.difficulty != "Easy" and p.recency == "30d"
    assert scheduler.pick_target_problem(session, 1, NOW, ["graphs"]).recency == "30d"


def test_foundations_set_starts_easy_and_covers_the_basics(session):
    rows = session.exec(scheduler.select(Problem).where(Problem.category == scheduler.FOUNDATIONS)
                        .order_by(Problem.sequence)).all()
    assert [p.sequence for p in rows] == list(range(1, len(rows) + 1))
    assert rows[0].difficulty == "Easy" and all(p.difficulty != "Hard" for p in rows)
    titles = {p.title for p in rows}
    for must in ("Two Sum", "Valid Palindrome", "Longest Substring Without Repeating Characters", "Binary Search",
                 "Reverse Linked List", "Same Tree", "Subsets", "Permutations", "Valid Parentheses", "LRU Cache"):
        assert must in titles, must


def test_target_moves_to_next_weakest_topic(session):
    close_foundations(session)
    for prob in session.exec(scheduler.select(Problem).where(Problem.category == "graphs")).all()[:7]:
        scheduler.log_attempt(session, prob.id, 30, True, True, 5, NOW)
    assert scheduler.pick_target_problem(session, 1, NOW, ["graphs", "trees"]).category == "trees"


def test_topic_filter_and_failed_problem_comes_back_first(session):
    p = scheduler.pick_target_problem(session, 1, NOW, ["graphs"])
    assert p.category == "graphs"
    scheduler.log_attempt(session, p.id, 45, False, False, 2, NOW, mistake="pattern")
    later = NOW + timedelta(days=2)
    assert scheduler.pick_target_problem(session, 1, later, ["graphs"]).id == p.id
    assert p.id in [x.id for x in scheduler.due_problems(session, later)]


def test_sync_overrides_bundled_recency(session):
    target = scheduler.pick_target_problem(session, 3, NOW, ["heaps"])
    other = session.get(Problem, "G-1834")  # Single-Threaded CPU: only in the 6-month data
    session.add(TrendingProblem(slug=other.slug, title=other.title, difficulty="Medium", window="30d", frequency=99))
    session.commit()
    assert target.id != "G-1834"
    assert scheduler.pick_target_problem(session, 3, NOW, ["heaps"]).id == "G-1834"


def test_pick_drills_due_first_then_round_robin(session):
    assert {d.category for d in scheduler.pick_drills(session, NOW, 8)} == {"foundations"}  # basics first
    close_foundations(session)
    first = scheduler.pick_drills(session, NOW, 9)
    assert len({d.category for d in first[:8]}) == 8  # then one per category before repeating
    scheduler.grade_drill(session, "R-127", 2, NOW - timedelta(days=3))
    assert scheduler.pick_drills(session, NOW, 1)[0].id == "R-127"


def test_graded_drill_leaves_queue_until_due(session):
    scheduler.grade_drill(session, "R-200", 5, NOW)
    assert "R-200" not in [d.id for d in scheduler.pick_drills(session, NOW, 50)]
    assert session.get(DrillProgress, "R-200").next_review_date == NOW + timedelta(days=1)


def test_warmup_adapts_to_mistakes(session):
    kinds = lambda drills: sum(d.type == "bug_hunt" for d in drills)  # noqa: E731
    assert kinds(scheduler.pick_warmup(session, NOW)) == 1
    for pid in ("G-200", "G-207"):
        scheduler.log_attempt(session, pid, 30, False, False, 2, NOW, mistake="bug_edge")
    assert scheduler.dominant_mistake(session) == "bug_edge"
    assert kinds(scheduler.pick_warmup(session, NOW)) == 2
    for pid in ("G-994", "G-547", "G-210"):
        scheduler.log_attempt(session, pid, 30, False, False, 2, NOW, mistake="pattern")
    assert kinds(scheduler.pick_warmup(session, NOW)) == 0
    assert scheduler.mistake_profile(session) == {"pattern": 3, "implementation": 0, "bug_edge": 2,
                                                  "slow": 0, "follow_up": 0, "python_syntax": 0,
                                                  "implementation_logic": 0}


def test_pace(session):
    scheduler.set_setting(session, "start_date", (NOW.date() - timedelta(days=13)).isoformat())  # day 14
    pace = scheduler.pace(session, NOW.date())
    total = scheduler.count(session, Problem)  # the Google bank plus the Foundations set
    assert total == 75 + 36 + 25
    assert pace["expected"] == round(total * (14 - 3) / 54) and pace["behind"] == pace["expected"]  # starts Day 3
    scheduler.set_setting(session, "start_date", NOW.date().isoformat())
    assert scheduler.pace(session, NOW.date())["behind"] == 0
