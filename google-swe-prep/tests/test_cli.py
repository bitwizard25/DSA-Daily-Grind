from datetime import datetime

from typer.testing import CliRunner

import google_prep
from google_prep.cli import app
from google_prep.core import scheduler, session_store
from google_prep.db import get_session
from google_prep.models import Attempt, DrillProgress, MockSession

runner = CliRunner()


def run(*args, input=None, at=None, monkeypatch=None):
    if at:
        monkeypatch.setenv("GOOGLE_PREP_NOW", at)
    return runner.invoke(app, list(args), input=input, catch_exceptions=False)


def count(model):
    with get_session() as s:
        return scheduler.count(s, model)


def test_zero_args_in_the_morning_starts_the_warmup():
    r = run(input="\n3\n\n2\nq\n")  # reveal, Instant, reveal, Effort, stop
    assert r.exit_code == 0, r.output
    assert "RECALL DRILL" in r.output and "[Google SWE II OS] Phase 1 > Week 1 > Day 3 > Recall > Drill 1/5" in r.output
    assert "Graphs" not in r.output.split("Drill 1/5")[1].split("\n")[0]  # unlabeled: no topic in the breadcrumb
    assert "DRILLS LOGGED" in r.output and "Instant 1" in r.output
    assert count(DrillProgress) == 2
    with get_session() as s:
        assert scheduler.schedule_for(s, datetime(2026, 10, 12).date()).morning_completed


def test_zero_args_in_the_evening_starts_the_solve(monkeypatch):
    r = run(input="1\ns\n2\ny\n", at="2026-10-12T19:30", monkeypatch=monkeypatch)  # Solved with Effort, trace ok
    assert r.exit_code == 0, r.output
    assert "TIMED SOLVE" in r.output and "RESULT" in r.output
    assert count(Attempt) == 1 and count(MockSession) == 0  # practice solves are not mocks


def test_zero_args_after_both_sessions_shows_today(monkeypatch):
    with get_session() as s:
        scheduler.mark_done(s, datetime(2026, 10, 12).date(), "morning", 20)
    r = run(at="2026-10-12T09:00", monkeypatch=monkeypatch)
    assert "NEXT: EVENING DEEP WORK" in r.output and "Done" in r.output and "Morning warmup" in r.output


def test_today_screen_has_breadcrumb_and_no_greeting():  # Day 3: Phase 1
    r = run("today")
    assert r.exit_code == 0, r.output
    assert "[Google SWE II OS] Phase 1 > Week 1 > Day 3 > Today" in r.output
    assert "[t] Today" in r.output and "[w] Mock" in r.output and "[s] STAR" in r.output  # section bar
    assert "Good morning" not in r.output and "Welcome" not in r.output
    assert "NEXT: MORNING WARMUP" in r.output and "[Enter] Start Today's 20-min Warmup" in r.output
    assert "EVENING DEEP WORK" not in r.output and "Later" in r.output  # one dominant card, the rest is a status line


def test_worked_hours_switch_to_micro_for_the_day():
    r = run("today", "--worked", "11")
    assert "micro-mode on" in r.output and "MICRO-MODE" in r.output
    assert "MICRO-MODE" in run("today").output  # stays on for the rest of the day


def test_m_starts_micro_mode_in_one_key():
    r = run("today", input="m\n\n2\n\n1\n")
    assert r.exit_code == 0, r.output
    assert "BUG HUNT" in r.output and "DRILLS LOGGED" in r.output
    with get_session() as s:
        assert scheduler.schedule_for(s, datetime(2026, 10, 12).date()).mode == "micro"


def test_esc_goes_back_from_the_roadmap():
    r = run("roadmap", input="\x1b\nq\n")
    assert "ROADMAP" in r.output and "NEXT: MORNING WARMUP" in r.output


def test_micro_mode_runs_two_drills():
    r = run("today", "--mode", "micro", input="\n\n2\n\n1\n")
    assert r.exit_code == 0, r.output
    assert "BUG HUNT" in r.output and "DRILLS LOGGED" in r.output
    assert count(DrillProgress) == 2


def test_mock_records_rubric_follow_up_and_mistake():
    # tick 1-2, follow-up, submit | [1] Missed it | pillars: clarified y, complexity y, clean n, trace n, follow-up y | mistake
    keys = "1\n2\nf\ns\n" + "1\n" + "y\ny\nn\nn\ny\n" + "1\n"
    r = run("mock", "--topic", "graphs", "--no-editor", input=keys)
    assert r.exit_code == 0, r.output
    assert "Phase 1 > Week 1 > Day 3 > Mock > Graphs" in r.output and "GOOGLE FOLLOW-UP" in r.output
    with get_session() as s:
        m = s.exec(scheduler.select(MockSession)).one()
        a = s.exec(scheduler.select(Attempt)).one()
    assert (m.solved, m.clarified_constraints, m.stated_complexity, m.clean_code, m.dry_run_passed,
            m.follow_up_handled) == (False, True, True, False, False, True)
    assert m.score == 50 and "Lean No Hire" in r.output and "SCORECARD · GOOGLE 4 PILLARS" in r.output
    assert "back in today's re-solve queue" in r.output
    assert a.mistake == "pattern" and a.source == "mock" and a.dry_run_ok is False


def test_mock_abandon_logs_nothing():
    r = run("mock", "--no-editor", input="q\n")
    assert "Abandoned" in r.output and count(Attempt) == 0
    assert session_store.load() is None


def test_crashed_mock_resumes_with_saved_state():
    session_store.save({"kind": "mock", "problem_id": "G-200", "minutes": 45, "practice": False,
                        "elapsed": 600.0, "ticks": [True, False, False, False], "shown": -1, "submitted": False})
    r = run(input="\ns\n2\n" + "y\n" * 5)  # zero-arg launch offers the resume first; Effort, all checks pass
    assert r.exit_code == 0, r.output
    assert "Session restored: Number of Islands (35m remaining)." in r.output and "Press [Enter] to resume." in r.output
    with get_session() as s:
        m = s.exec(scheduler.select(MockSession)).one()
    assert m.problem_id == "G-200" and m.duration_seconds >= 600 and m.score == 100 and "Strong Hire" in r.output
    assert session_store.load() is None


def test_crashed_recall_resumes_remaining_drills():
    session_store.save({"kind": "recall", "mode": "warmup", "drill_ids": ["R-200", "R-207", "R-994"], "index": 2,
                        "counts": {"1": 0, "2": 2, "3": 0}, "seconds": 120, "day": "2026-10-12"})
    r = run("resume", input="\n\n3\n")
    assert r.exit_code == 0, r.output
    assert "Session restored: Pattern recall (1 drill left)" in r.output and "Effort 2 · Instant 1" in r.output
    assert count(DrillProgress) == 1


def test_discarding_a_saved_session():
    session_store.save({"kind": "recall", "mode": "warmup", "drill_ids": ["R-200"], "index": 0,
                        "counts": {"1": 0, "2": 0, "3": 0}, "seconds": 0, "day": "2026-10-12"})
    r = run("resume", input="x\n")
    assert session_store.load() is None and "NEXT: MORNING WARMUP" in r.output


def test_solve_specific_problem():
    r = run("solve", "G-5", "--no-editor", input="s\n2\nn\n5\n")
    assert r.exit_code == 0, r.output
    with get_session() as s:
        a = s.exec(scheduler.select(Attempt)).one()
    assert a.problem_id == "G-5" and a.mistake == "follow_up" and a.dry_run_ok is False


def test_roadmap():
    r = run("roadmap")
    assert r.exit_code == 0, r.output
    assert "Readiness = 0.40×0% + 0.30×0% + 0.15×0% + 0.15×0% = 0%" in r.output  # transparent formula
    for text in ("Week 1 > Day 3 > Roadmap", "READINESS INDEX COMPONENTS", "Dry-Run Accuracy", "MILESTONE PHASES",
                 "Monotonic Queues", "Googlyness STAR Stories", "0/8 completed"):
        assert text in r.output, text


def test_problems_patterns_and_bad_topic():
    assert "Bus Routes" in run("problems", "--topic", "graphs").output
    assert "Sliding Window" in run("patterns").output
    assert "Sliding window: variable size" in run("patterns", "2").output  # pattern 2 opens its vault note
    assert runner.invoke(app, ["mock", "--topic", "nope"]).exit_code == 2


def test_star_add_list_review_show():
    r = run("star", "add", "-d", "own", "-t", "Fixed the 2am pager", "-s", "Prod outage", "--task", "Restore",
            "-a", "Rolled back", "-r", "MTTR 4h -> 20m")
    assert r.exit_code == 0, r.output and "Ownership" in r.output
    interactive = run("star", "add", input="humility\nLearned Rust\nNew team\nShip in Rust\nPaired daily\nShipped\n")
    assert "Saved story #2" in interactive.output
    assert "Ownership 1/2" in run("star", "list").output
    review = run("star", "review", "-d", "Ownership", input="\n")
    assert "STAR > Ownership" in review.output and "Fixed the 2am pager" in review.output
    assert "Disagreement 0/2" in run("star", "list").output
    assert "MTTR" in run("star", "show", "1").output
    assert runner.invoke(app, ["star", "add", "-d", "nope"]).exit_code == 2


def test_setup_remind_version_hot_add():
    out = run("setup", "--name", "Raj", "--interview-date", "2026-11-30", "--target", "Google Hyderabad SWE II").output
    assert "Interview: 2026-11-30" in out
    assert "49 days to interview" in run("today").output
    reminders = run("remind").output  # Windows quotes the exe path: cmd /k "google-prep" today
    assert "google-prep" in reminders and " today" in reminders
    assert runner.invoke(app, ["remind", "--morning", "7am"]).exit_code == 2
    assert google_prep.__version__ in run("--version").output
    assert "Run: google-prep sync" in run("hot").output
    assert "Already in your bank: G-410" in run("add", "split-array-largest-sum").output
    assert runner.invoke(app, ["add", "generate-parentheses"]).exit_code == 1  # not synced yet


def test_remind_commands_per_platform():
    from google_prep.modules import remind
    win = remind.commands("07:30", "19:30", "windows")
    assert win[0].startswith('schtasks /Create /F /SC DAILY /TN "GooglePrep Morning" /ST 07:30')
    assert remind.commands("07:05", "19:30", "mac")[0].startswith("5 7 * * * osascript")
    assert remind.commands("07:30", "21:15", "linux")[1].startswith("15 21 * * *")


def test_no_manual_log_command():
    assert runner.invoke(app, ["log", "G-200", "-m", "10"]).exit_code != 0
