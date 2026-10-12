from datetime import timedelta

import pytest

from google_prep.core import rubric, scheduler
from google_prep.models import DrillProgress
from google_prep.modules import recall
from google_prep.ui.formatters import gauge

from conftest import NOW


@pytest.mark.parametrize("choice,days", [("1", 0), ("2", 3), ("3", 7)])
def test_three_tier_recall_reschedules_today_3_days_next_week(session, choice, days):
    q, min_days, note = recall.quality(choice, seconds=5, timed_out=False)
    scheduler.grade_drill(session, "R-200", q, NOW, min_days)
    assert session.get(DrillProgress, "R-200").next_review_date == NOW + timedelta(days=days)
    assert note == ""


def test_missed_drill_is_due_again_today(session):
    q, min_days, _ = recall.quality("1", 5, False)
    scheduler.grade_drill(session, "R-200", q, NOW, min_days)
    assert scheduler.pick_drills(session, NOW, 1)[0].id == "R-200"


def test_measured_time_overrides_flattering_grades():
    assert recall.quality("3", 35, False)[:2] == (recall.SM2_QUALITY["2"], 3)  # slow "instant" -> effort
    assert recall.quality("3", 10, True)[1] == 3  # after the timer ran out -> effort
    assert recall.quality("1", 2, False)[1] == 0  # a miss is a miss


def test_instant_reflex_intervals_keep_growing(session):
    for _ in range(3):
        q, min_days, _ = recall.quality("3", 5, False)
        prog = scheduler.grade_drill(session, "R-200", q, NOW, min_days)
    assert prog.interval_days > 7  # SM-2 stretches the gap beyond the one-week floor


def test_four_pillars_are_25_percent_each():
    none = rubric.from_ticks([False] * 4)
    assert rubric.score(none) == 0
    assert rubric.score({**none, "clarified_constraints": True}) == 12.5  # half of Analytical Ability
    full = {field: True for field, _, _ in rubric.CHECKS}
    assert rubric.score(full) == 100 and set(rubric.pillar_scores(full)) == set(rubric.PILLARS)
    assert rubric.from_ticks([True, False, True, True]) == {
        "clarified_constraints": True, "stated_complexity": False, "clean_code": True, "dry_run_passed": True,
        "follow_up_handled": False}


def test_hire_signal_needs_a_solution():
    assert rubric.hire_signal(100, solved=True) == "Strong Hire"
    assert rubric.hire_signal(75, solved=True) == "Lean Hire"
    assert rubric.hire_signal(100, solved=False) == "Lean No Hire"
    assert rubric.hire_signal(25, solved=False) == "No Hire"


def test_gauge_is_a_ring_with_the_percentage_inside():
    empty, half, full = (gauge(f).plain for f in (0, 0.5, 1))
    assert "0%" in empty and "50%" in half and "100%" in full and "READY" in half
    assert "●" not in empty and "·" not in full
    assert 0 < half.count("●") < full.count("●")
    assert len(half.splitlines()) == 9  # a ring, not a bar


def test_mock_outcome_reschedules_the_problem(session):
    from google_prep.modules import mock

    p = session.get(scheduler.Problem, "G-207")
    checks = rubric.from_ticks([True] * 4)
    mock.record(session, p, 45, 600, "1", checks, "", False, "pattern", NOW)  # Missed it
    assert p.id in [x.id for x in scheduler.due_problems(session, NOW)]  # back in today's re-solve queue
    out = mock.record(session, p, 45, 600, "3", checks, "", False, None, NOW)  # Instant Reflex
    prog = session.exec(scheduler.select(scheduler.UserProgress).where(scheduler.UserProgress.problem_id == p.id)).one()
    assert out["outcome"] == "3" and prog.next_review_date >= NOW + timedelta(days=7)


def test_an_over_time_solve_is_never_instant():
    assert rubric.outcome_for("3", over_time=True) == "2"
    assert rubric.outcome_for("3", over_time=False) == "3"
    assert rubric.outcome_for("1", over_time=True) == "1"


def test_breadcrumb_has_the_app_phase_and_bold_current_step(session):
    from google_prep.ui.formatters import crumb_text

    path = scheduler.breadcrumb(session, NOW.date(), "Mock", "Graphs", "Course Schedule II")
    assert path == "[Google SWE II OS] Phase 1 > Week 1 > Day 1 > Mock > Graphs > Course Schedule II"
    text = crumb_text(path)
    assert text.spans[-1].style.endswith("#FFFFFF") and "bold" in text.spans[-1].style


def test_text_colours_pass_wcag_aaa():
    from google_prep.ui import theme

    def lum(hex_colour):
        rgb = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    bg = lum(theme.SURFACE)
    for colour in (theme.TEXT_BLUE, theme.TEXT_GREEN, theme.TEXT_YELLOW, theme.TEXT_RED, theme.BODY, theme.MUTED,
                   theme.HEADER):
        assert (lum(colour) + 0.05) / (bg + 0.05) >= 7, colour
