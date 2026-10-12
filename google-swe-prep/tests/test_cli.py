from datetime import datetime

from typer.testing import CliRunner

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
    assert "PATTERN RECALL" in r.output and "Week 1 > Day 1 > Recall" in r.output
    assert "DRILLS LOGGED" in r.output and "Instant 1" in r.output
    assert count(DrillProgress) == 2
    with get_session() as s:
        assert scheduler.schedule_for(s, datetime(2026, 10, 12).date()).morning_completed


def test_zero_args_in_the_evening_starts_the_solve(monkeypatch):
    r = run(input="1\ns\ny\ny\n", at="2026-10-12T19:30", monkeypatch=monkeypatch)
    assert r.exit_code == 0, r.output
    assert "TIMED SOLVE" in r.output and "RESULT" in r.output
    assert count(Attempt) == 1 and count(MockSession) == 0  # practice solves are not mocks


def test_zero_args_after_both_sessions_shows_today(monkeypatch):
    with get_session() as s:
        scheduler.mark_done(s, datetime(2026, 10, 12).date(), "morning", 20)
    r = run(at="2026-10-12T09:00", monkeypatch=monkeypatch)
    assert "GOOGLE SWE II OS" in r.output and "✓ Completed" in r.output


def test_today_screen_has_breadcrumb_and_no_greeting():
    r = run("today")
    assert r.exit_code == 0, r.output
    assert "Week 1 > Day 1 > Today" in r.output
    assert "Good morning" not in r.output and "MORNING WARMUP" in r.output and "EVENING DEEP WORK" in r.output


def test_worked_hours_switch_to_micro_for_the_day():
    r = run("today", "--worked", "11")
    assert "micro-mode on" in r.output and "MICRO-MODE" in r.output
    assert "MICRO-MODE" in run("today").output  # stays on for the rest of the day


def test_micro_mode_runs_two_drills():
    r = run("today", "--mode", "micro", input="\n\n2\n\n1\n")
    assert r.exit_code == 0, r.output
    assert "BUG HUNT" in r.output and "DRILLS LOGGED" in r.output
    assert count(DrillProgress) == 2


def test_mock_records_rubric_follow_up_and_mistake():
    keys = "1\n2\nf\ns\nn\nn\n4\n3\n2\n3\n1\n"  # tick 1-2, follow-up, submit, not solved, no dry run, rubric, mistake
    r = run("mock", "--topic", "graphs", "--no-editor", input=keys)
    assert r.exit_code == 0, r.output
    assert "Week 1 > Day 1 > Mock > Graphs & Union-Find" in r.output and "GOOGLE FOLLOW-UP" in r.output
    with get_session() as s:
        m = s.exec(scheduler.select(MockSession)).one()
        a = s.exec(scheduler.select(Attempt)).one()
    assert (m.solved, m.checklist_done, m.algorithms, m.problem_solving) == (False, 2, 4, 3)
    assert a.mistake == "pattern" and a.source == "mock" and a.dry_run_ok is False


def test_mock_abandon_logs_nothing():
    r = run("mock", "--no-editor", input="q\n")
    assert "Abandoned" in r.output and count(Attempt) == 0
    assert session_store.load() is None


def test_crashed_mock_resumes_with_saved_state():
    session_store.save({"kind": "mock", "problem_id": "G-47", "minutes": 45, "practice": False,
                        "elapsed": 600.0, "ticks": [True, False, False, False], "shown": -1, "submitted": False})
    r = run(input="\ns\ny\ny\n4\n4\n4\n4\n")  # zero-arg launch offers the resume first
    assert r.exit_code == 0, r.output
    assert "UNFINISHED SESSION" in r.output and "35:00 left" in r.output
    with get_session() as s:
        m = s.exec(scheduler.select(MockSession)).one()
    assert m.problem_id == "G-47" and m.used_seconds >= 600 and m.checklist_done == 1
    assert session_store.load() is None


def test_crashed_recall_resumes_remaining_drills():
    session_store.save({"kind": "recall", "mode": "warmup", "drill_ids": ["R-01", "R-02", "R-03"], "index": 2,
                        "counts": {"1": 0, "2": 2, "3": 0}, "seconds": 120, "day": "2026-10-12"})
    r = run("resume", input="\n\n3\n")
    assert r.exit_code == 0, r.output
    assert "Recall drills · 1 left" in r.output and "Effort 2 · Instant 1" in r.output
    assert count(DrillProgress) == 1


def test_discarding_a_saved_session():
    session_store.save({"kind": "recall", "mode": "warmup", "drill_ids": ["R-01"], "index": 0,
                        "counts": {"1": 0, "2": 0, "3": 0}, "seconds": 0, "day": "2026-10-12"})
    r = run("resume", input="d\n")
    assert session_store.load() is None and "GOOGLE SWE II OS" in r.output


def test_solve_specific_problem():
    r = run("solve", "G-05", "--no-editor", input="s\ny\nn\n5\n")
    assert r.exit_code == 0, r.output
    with get_session() as s:
        a = s.exec(scheduler.select(Attempt)).one()
    assert a.problem_id == "G-05" and a.mistake == "follow_up" and a.dry_run_ok is False


def test_roadmap():
    r = run("roadmap")
    assert r.exit_code == 0, r.output
    for text in ("Week 1 > Day 1 > Roadmap", "READINESS INDEX COMPONENTS", "Dry-Run Accuracy", "MILESTONE PHASES",
                 "Graphs & Union-Find", "Googlyness STAR Stories", "0/8 completed"):
        assert text in r.output, text


def test_problems_patterns_and_bad_topic():
    assert "Bus Routes" in run("problems", "--topic", "graphs").output
    assert "Sliding Window" in run("patterns").output
    assert "left = 0" in run("patterns", "2").output
    assert runner.invoke(app, ["mock", "--topic", "nope"]).exit_code == 2


def test_star_add_list_review_show():
    r = run("star", "add", "-d", "own", "-t", "Fixed the 2am pager", "-s", "Prod outage", "--task", "Restore",
            "-a", "Rolled back", "-r", "MTTR 4h -> 20m")
    assert r.exit_code == 0, r.output and "Ownership" in r.output
    interactive = run("star", "add", input="humility\nLearned Rust\nNew team\nShip in Rust\nPaired daily\nShipped\n")
    assert "Saved story #2" in interactive.output
    assert "Ownership 1/2" in run("star", "list").output
    review = run("star", "review", "-d", "Ownership", input="\n")
    assert "STAR Vault > Ownership" in review.output and "Fixed the 2am pager" in review.output
    assert "MTTR" in run("star", "show", "1").output
    assert runner.invoke(app, ["star", "add", "-d", "nope"]).exit_code == 2


def test_setup_remind_version_hot_add():
    out = run("setup", "--name", "Raj", "--interview-date", "2026-11-30", "--target", "Google Hyderabad SWE II").output
    assert "Interview: 2026-11-30" in out
    assert "49 days to interview" in run("today").output
    assert "google-prep today" in run("remind").output
    assert runner.invoke(app, ["remind", "--morning", "7am"]).exit_code == 2
    assert "1.0.0" in run("--version").output
    assert "Run: google-prep sync" in run("hot").output
    assert runner.invoke(app, ["add", "split-array-largest-sum"]).exit_code == 1


def test_remind_commands_per_platform():
    from google_prep.modules import remind
    win = remind.commands("07:30", "19:30", "windows")
    assert win[0].startswith('schtasks /Create /F /SC DAILY /TN "GooglePrep Morning" /ST 07:30')
    assert remind.commands("07:05", "19:30", "mac")[0].startswith("5 7 * * * osascript")
    assert remind.commands("07:30", "21:15", "linux")[1].startswith("15 21 * * *")


def test_no_manual_log_command():
    assert runner.invoke(app, ["log", "G-01", "-m", "10"]).exit_code != 0
