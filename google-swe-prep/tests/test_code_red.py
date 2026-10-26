"""Code Red: the Medium/Hard Google pool, a fixed daily mission, the re-solve queue and honest progress."""
from datetime import datetime, timedelta

import pytest
from sqlmodel import select

from google_prep.core import code_red, scheduler
from google_prep.models import Problem, RecallDrill, StarStory

NOW = datetime(2026, 10, 12, 7, 30)


def _enter(session, interview_in_days=21):
    scheduler.set_setting(session, "preparation_mode", code_red.MODE)
    scheduler.set_setting(session, "interview_date", (NOW.date() + timedelta(days=interview_in_days)).isoformat())


def _fail(session, problem_id, when=NOW, help="none"):  # noqa: A002
    scheduler.log_attempt(session, problem_id, 45, False, False, 1, when, source="mock", min_days=0, help=help)


def _solve(session, problem_id, when=NOW, help="none"):  # noqa: A002
    scheduler.log_attempt(session, problem_id, 30, True, True, 5, when, source="mock", min_days=7, help=help)


def test_pool_is_the_80_medium_hard_problems_from_the_last_three_months(session):
    pool = code_red.pool(session)
    assert len(pool) == 80
    assert all(p.category != scheduler.FOUNDATIONS for p in pool)
    assert all(p.difficulty in ("Medium", "Hard") for p in pool)
    assert all(p.recency in ("30d", "3m") for p in pool)


def test_fresh_day_plan_serves_a_new_strike_two_mocks_and_a_story_first(session):
    _enter(session)
    data = code_red.today(session, NOW)
    ids = [b["id"] for b in data["blocks"]]
    assert ids == ["warmup", "strike", "mock1", "mock2", "behavioral"]
    coding = [b for b in data["blocks"] if b.get("problem")]
    assert coding[0]["kind"] == "new"  # nothing due on day 1
    problem_ids = [b["problem"]["id"] for b in coding]
    assert len(set(problem_ids)) == 3
    pool = {p.id for p in code_red.pool(session)}
    assert set(problem_ids) <= pool
    topics = [b["problem"]["category"] for b in coding]
    assert len(set(topics)) == 3  # mixed topics, not three of the same pattern
    assert data["blocks"][-1]["title"] == "Write one STAR story"  # no stories yet
    assert data["completed"] == 0 and data["phase"] == "sprint"


def test_the_plan_is_fixed_for_the_day_and_tracks_completion(session):
    _enter(session)
    first = code_red.today(session, NOW)
    strike = first["blocks"][1]["problem"]["id"]
    _solve(session, strike, NOW + timedelta(minutes=50))
    again = code_red.today(session, NOW + timedelta(hours=1))
    assert [b.get("problem", {}).get("id") for b in again["blocks"]] == [b.get("problem", {}).get("id") for b in first["blocks"]]
    assert again["blocks"][1]["done"] is True
    assert again["completed"] == 1


def test_due_failures_come_first_and_a_long_queue_replaces_the_second_mock(session):
    _enter(session)
    yesterday = NOW - timedelta(days=1)
    failed = [p.id for p in code_red.pool(session)[:5]]
    for pid in failed:
        _fail(session, pid, yesterday, help="hint")
    data = code_red.today(session, NOW)
    strike, mock1, mock2 = data["blocks"][1:4]
    assert strike["kind"] == "resolve" and strike["problem"]["id"] in failed
    assert mock1["kind"] == "mock" and mock1["problem"]["id"] not in failed
    assert mock2["kind"] == "resolve" and mock2["problem"]["id"] in failed
    assert data["queue_due"] == 5


def test_taper_serves_no_new_problems_and_interview_day_is_warmup_only(session):
    _enter(session, interview_in_days=2)
    _solve(session, code_red.pool(session)[0].id, NOW - timedelta(days=3))
    taper = code_red.today(session, NOW)
    assert taper["phase"] == "taper"
    assert all(b["kind"] == "resolve" for b in taper["blocks"] if b.get("problem"))
    later = NOW + timedelta(days=1)
    scheduler.set_setting(session, "interview_date", later.date().isoformat())
    day = code_red.today(session, later)
    assert day["phase"] == "interview_day"
    assert [b["id"] for b in day["blocks"]] == ["warmup", "behavioral"]


def test_behavioral_block_switches_to_the_two_minute_drill_once_eight_stories_exist(session):
    _enter(session)
    for i in range(code_red.STAR_TARGET):
        session.add(StarStory(dimension="Ownership", title=f"s{i}", situation="s", task="t", action="a", result="r",
                              created_at=NOW - timedelta(days=2)))
    session.commit()
    block = code_red.today(session, NOW)["blocks"][-1]
    assert block["title"] == "2-minute Googleyness drill" and block["done"] is False
    code_red.mark_behavioral(session, NOW.date())
    assert code_red.today(session, NOW)["blocks"][-1]["done"] is True


def test_warmup_skips_foundations_even_before_they_are_tried(session):
    assert scheduler.foundations_open(session)
    drills = code_red.warmup(session, NOW)
    assert len(drills) == code_red.WARMUP_CARDS
    assert all(d.category != scheduler.FOUNDATIONS for d in drills)


def test_queue_lists_what_beat_you_most_urgent_first_and_ignores_clean_solves(session):
    pool = [p.id for p in code_red.pool(session)]
    _fail(session, pool[0], NOW - timedelta(days=2))
    _fail(session, pool[1], NOW - timedelta(days=1), help="answer")
    _solve(session, pool[2], NOW - timedelta(days=1))
    foundation = session.exec(select(Problem).where(Problem.category == scheduler.FOUNDATIONS)).first()
    _fail(session, foundation.id, NOW - timedelta(days=1))
    rows = code_red.queue(session, NOW)["rows"]
    assert [r["id"] for r in rows] == [pool[0], pool[1]]
    assert rows[0]["reason"] == "Missed it" and rows[1]["reason"] == "Saw the answer"
    assert rows[0]["overdue_days"] == 2


def test_solved_solo_counts_only_pool_problems_without_help(session):
    pool = [p.id for p in code_red.pool(session)]
    _solve(session, pool[0])
    _solve(session, pool[1], help="hint")
    _solve(session, pool[2], help="answer")
    data = code_red.progress(session, NOW)
    assert data["solved_solo"] == 1 and data["pool_total"] == 80
    assert len(data["trend"]) == code_red.TREND_DAYS


def test_api_mode_switch_accepts_code_red_and_pauses_a_general_focus_block():
    from google_prep.gui.api import Api, ApiError
    api = Api(now=lambda: NOW)
    api.preparation_mode("general")
    api.support_start(NOW.date().isoformat(), 5, "Explain one edge case")
    state = api.preparation_mode("code_red")
    assert state["settings"]["preparation_mode"] == "code_red"
    assert state["code_red"] == {"queue_due": 0}
    assert api.support_state()["active"]["status"] == "paused"
    pool = {p["id"] for p in api.problems()["problems"] if p["code_red"]}
    assert len(pool) == 80
    assert api.mock_target()["problem"]["id"] in pool
    drills = api.recall_start(mode="code_red")["drills"]
    assert len(drills) == code_red.WARMUP_CARDS
    with pytest.raises(ApiError):
        api.preparation_mode("red")


def test_drill_categories_exist_for_every_pool_topic(session):
    cats = set(code_red.pool_categories(session))
    drill_cats = {d.category for d in session.exec(select(RecallDrill)).all()}
    assert cats <= drill_cats


def test_practice_never_runs_dry_once_every_pool_problem_was_tried(session):
    pool = code_red.pool(session)
    for i, p in enumerate(pool):
        _solve(session, p.id, NOW - timedelta(days=1, minutes=i))
    assert code_red.pick_new(session, NOW) is None
    assert code_red.pick(session, NOW).id in {p.id for p in pool}


def test_switching_to_code_red_and_back_keeps_history_notes_and_the_google_plan():
    from google_prep.db import get_session
    from google_prep.gui.api import Api
    api = Api(now=lambda: NOW)
    api.settings_save(target="Google SWE II", daily_problems="3", interview_date="2026-11-30")
    with get_session() as s:
        _fail(s, code_red.pool(s)[0].id, NOW - timedelta(days=1))
    api.star_save("Ownership", "Outage", "s", "t", "I fixed it", "p99 -40%")
    api.role_prep_save("recruiter", done=True, notes="Ask about the ML round")
    before = api.state()
    api.preparation_mode("code_red")
    assert api.code_red_queue()["due_count"] == 1  # Google-mode history feeds the Code Red queue
    after = api.preparation_mode("google")
    for key in ("day", "week", "plan_days", "streak", "interview_date", "days_left"):
        assert after[key] == before[key], key
    assert after["settings"] == {**before["settings"], "preparation_mode": "google"}
    assert after["code_red"] is None
    assert [x["title"] for x in api.star_list()["stories"]] == ["Outage"]
    item = next(x for x in api.role_prep()["items"] if x["id"] == "recruiter")
    assert item["done"] is True and item["notes"] == "Ask about the ML round"
