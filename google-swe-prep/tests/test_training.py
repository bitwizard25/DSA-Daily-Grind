"""Interview habits: persistent intentions, truthful evidence, and local-day boundaries."""
from datetime import datetime, timedelta

import pytest
from sqlmodel import Session, create_engine, select

from google_prep import db
from google_prep.core import scheduler, training
from google_prep.gui.api import Api, ApiError, RPC, call
from google_prep.models import Attempt, MockSession, TrainingCheckin

NOW = datetime(2026, 10, 12, 7, 30)


def test_rotating_habit_uses_plan_day_and_keeps_saved_intention(session):
    # The shared fixture starts on plan Day 3; restarting the app keeps its habit.
    assert training.habit_for_day(session, NOW.date())["id"] == "reset"
    assert training.habit_for_day(session, NOW.date() + timedelta(days=1))["id"] == "dry_run"
    training.save_checkin(session, NOW, NOW.date().isoformat(), "reset", intention="Name the obstacle.")
    scheduler.set_setting(session, "start_date", NOW.date().isoformat())
    assert training.habit_for_day(session, NOW.date())["id"] == "reset"
    assert training.habit_for_day(session, NOW.date() + timedelta(days=1))["id"] == "explain"


def test_fresh_state_reports_missing_evidence_instead_of_zero_scores():
    data = Api().training_state()
    assert data["today"]["checkin"] is None
    assert data["week"]["practice_attempts"] == data["week"]["mock_sessions"] == 0
    assert len(data["week"]["days"]) == 7
    evidence = {item["id"]: item for item in data["week"]["habits"]}
    assert evidence["clarify"]["observed"] is None
    assert evidence["dry_run"]["observed"] is None
    assert evidence["follow_up"]["observed"] is None
    assert all(item["status"] == "no_evidence" for item in evidence.values())
    assert "score" not in data and "probability" not in data


def test_intention_reflection_round_trip_preserves_omitted_fields():
    api = Api()
    assert {"training_state", "training_checkin"} <= RPC
    first = call(api, "training_checkin", {"date": NOW.date().isoformat(), "habit_id": "reset",
                                          "intention": "  Name the obstacle.  ", "confidence_before": 2})
    assert first["today"]["checkin"]["intention"] == "Name the obstacle."
    assert not first["today"]["checkin"]["completed"]
    second = api.training_checkin(NOW.date().isoformat(), "reset",
                                  reflection="A smaller example helped.", confidence_after=3)
    saved = second["today"]["checkin"]
    assert saved["intention"] == "Name the obstacle."
    assert saved["confidence_before"] == 2 and saved["confidence_after"] == 3
    assert saved["completed"] and second["week"]["completed_checkins"] == 1
    assert Api().training_state()["today"]["checkin"] == saved
    with db.get_session() as session:
        assert len(session.exec(select(TrainingCheckin)).all()) == 1


def test_repeat_submission_updates_one_row_instead_of_inflating_history():
    api = Api()
    payload = {"date": NOW.date().isoformat(), "habit_id": "reset", "reflection": "I named the obstacle."}
    first, second = api.training_checkin(**payload), api.training_checkin(**payload)
    assert first["week"]["checkins"] == second["week"]["checkins"] == 1
    assert second["week"]["completed_checkins"] == 1


@pytest.mark.parametrize("field,value", [
    ("date", "20261012"), ("date", "2026-10-13"), ("date", "2026-10-11"),
    ("date", None), ("habit_id", "unknown"), ("habit_id", []),
    ("intention", " "), ("intention", "x" * 501), ("intention", 3),
    ("reflection", ""), ("reflection", "x" * 2001),
    ("confidence_before", 0), ("confidence_before", 6), ("confidence_before", True),
    ("confidence_after", "3"), ("confidence_after", 2.5),
])
def test_invalid_checkin_does_not_write(field, value):
    payload = {"date": NOW.date().isoformat(), "habit_id": "reset", "intention": "Take one step."}
    payload[field] = value
    with pytest.raises(ApiError):
        Api().training_checkin(**payload)
    assert Api().training_state()["today"]["checkin"] is None


def test_empty_payload_and_habit_change_are_rejected():
    api = Api()
    with pytest.raises(ApiError):
        api.training_checkin(NOW.date().isoformat(), "reset")
    api.training_checkin(NOW.date().isoformat(), "reset", intention="Try a smaller example.")
    with pytest.raises(ApiError):
        api.training_checkin(NOW.date().isoformat(), "dry_run", reflection="Changed habit.")
    assert api.training_state()["today"]["checkin"]["habit_id"] == "reset"


def test_midnight_stale_page_cannot_overwrite_previous_day():
    before = Api(now=lambda: datetime(2026, 10, 12, 23, 59))
    after = Api(now=lambda: datetime(2026, 10, 13, 0, 1))
    before.training_checkin("2026-10-12", "reset", intention="Stay with the example.")
    with pytest.raises(ApiError, match="day has changed"):
        after.training_checkin("2026-10-12", "reset", reflection="Stale page text.")
    result = after.training_checkin("2026-10-13", "dry_run", intention="Trace a boundary case.")
    assert result["today"]["habit"]["id"] == "dry_run"
    assert result["week"]["checkins"] == 2
    previous = result["week"]["days"][-2]["checkin"]
    assert previous["intention"] == "Stay with the example." and not previous["reflection"]


def test_week_evidence_uses_saved_records_and_excludes_outside_window(session):
    first = NOW - timedelta(days=6)
    for when in (first, NOW, NOW + timedelta(days=1), first - timedelta(days=1)):
        session.add(Attempt(problem_id="G-1", day=when.date().isoformat(), minutes=20,
                            solved=True, dry_run_ok=True, source="mock"))
        session.add(MockSession(problem_id="G-1", date=when, duration_seconds=1200,
                                clarified_constraints=True, stated_complexity=True, dry_run_passed=True,
                                follow_up="What if the input is a stream?", follow_up_handled=True))
    # A failed solve counts as useful practice. An ungraded trace stays missing.
    session.add(Attempt(problem_id="G-1", day=NOW.date().isoformat(), minutes=8,
                        solved=False, dry_run_ok=None, source="solve"))
    session.add(Attempt(problem_id="G-1", day=NOW.date().isoformat(), minutes=11,
                        solved=False, dry_run_ok=False, source="solve"))
    session.add(MockSession(problem_id="G-1", date=NOW, duration_seconds=1200,
                            clarified_constraints=False, stated_complexity=False, follow_up=""))
    session.add(TrainingCheckin(date_str=first.date().isoformat(), habit_id="clarify", intention="Ask first.",
                               reflection="Input size changed my plan.", created_at=first, updated_at=first))
    session.add(TrainingCheckin(date_str=(first - timedelta(days=1)).date().isoformat(), habit_id="reset",
                               reflection="Outside the window.", created_at=first, updated_at=first))
    session.commit()
    data = training.state(session, NOW)
    week = data["week"]
    assert week["practice_attempts"] == 4 and week["practice_solves"] == 2
    assert week["mock_sessions"] == 3
    assert week["checkins"] == week["completed_checkins"] == 1
    evidence = {item["id"]: item for item in week["habits"]}
    assert (evidence["clarify"]["observed"], evidence["clarify"]["total"]) == (2, 3)
    assert (evidence["explain"]["observed"], evidence["explain"]["total"]) == (2, 3)
    assert (evidence["dry_run"]["observed"], evidence["dry_run"]["total"]) == (2, 3)
    assert (evidence["follow_up"]["observed"], evidence["follow_up"]["total"]) == (2, 2)
    assert evidence["reset"]["status"] == "no_evidence"
    assert evidence["reflect"]["observed"] == 1
    assert week["days"][0]["attempts"] == 1 and week["days"][-1]["attempts"] == 3


def test_existing_database_gains_training_table_and_keeps_attempts(session):
    session.add(Attempt(problem_id="G-1", day=NOW.date().isoformat(), minutes=10, solved=False))
    session.commit()
    # Simulate the previous release, which had no training table, in this test's temporary DB.
    TrainingCheckin.__table__.drop(session.get_bind())
    db._engine.cache_clear()
    result = Api().training_checkin(NOW.date().isoformat(), "reset", intention="Ask one focused question.")
    assert result["week"]["practice_attempts"] == 1
    assert result["today"]["checkin"]["intention"] == "Ask one focused question."


def test_legacy_missing_mock_fields_are_missing_evidence_not_failures(home):
    url = f"sqlite:///{home / 'legacy.db'}"
    engine = create_engine(url)
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE mocksession (id INTEGER PRIMARY KEY, problem_id TEXT, date TEXT, "
            "duration_seconds INTEGER, solved INTEGER)")
        connection.exec_driver_sql(
            "INSERT INTO mocksession VALUES (1, 'G-1', '2026-10-12 07:00:00.000000', 1200, 1)")
    with Session(db._engine(url)) as session:
        data = training.state(session, NOW)
    assert data["week"]["mock_sessions"] == 1
    evidence = {item["id"]: item for item in data["week"]["habits"]}
    for habit_id in ("clarify", "explain", "follow_up"):
        assert evidence[habit_id]["observed"] is None and evidence[habit_id]["total"] == 0


def test_new_mistake_choices_preserve_legacy_history(session):
    api = Api()
    scorecard = api.mock_scorecard("G-1", True, 60, 600, [False] * 4, -1)
    choices = [item["key"] for item in scorecard["mistakes"]]
    assert choices == ["pattern", "python_syntax", "implementation_logic", "bug_edge", "slow", "follow_up"]
    for mistake in ("implementation", "python_syntax", "implementation_logic"):
        session.add(Attempt(problem_id="G-1", day=NOW.date().isoformat(), minutes=10, solved=False,
                            mistake=mistake))
    session.commit()
    profile = scheduler.mistake_profile(session)
    assert profile["implementation"] == profile["python_syntax"] == profile["implementation_logic"] == 1
