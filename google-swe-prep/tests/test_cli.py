from typer.testing import CliRunner

from google_swe_prep.cli import app

runner = CliRunner()


def test_today(db_env):
    r = runner.invoke(app, ["today"])
    assert r.exit_code == 0, r.output
    assert "Morning Warmup" in r.output and "Evening Deep Solve" in r.output
    assert "Phase 1" in r.output


def test_today_micro(db_env):
    r = runner.invoke(app, ["today", "--micro"])
    assert r.exit_code == 0, r.output
    assert "Micro-Mode" in r.output


def test_log_and_unknown_problem(db_env):
    assert runner.invoke(app, ["log", "1", "-m", "20"]).exit_code == 0
    assert runner.invoke(app, ["log", "9999", "-m", "20"]).exit_code == 1


def test_recall_session(db_env):
    r = runner.invoke(app, ["recall", "-n", "2"], input="\ny\n\nn\n")
    assert r.exit_code == 0, r.output
    assert "Score: 1/2" in r.output


def test_mock(db_env):
    r = runner.invoke(app, ["mock", "--limit", "30"], input="\ny\n4\n")
    assert r.exit_code == 0, r.output
    assert "Recorded" in r.output


def test_roadmap(db_env):
    r = runner.invoke(app, ["roadmap"])
    assert r.exit_code == 0, r.output
    assert "Foundation" in r.output and "Googlyness" in r.output


def test_star_flow(db_env):
    args = ["star", "add", "-c", "leadership", "-t", "Led migration", "-s", "S", "--task", "T", "-a", "A", "-r", "R"]
    assert runner.invoke(app, args).exit_code == 0
    assert "Led migration" in runner.invoke(app, ["star", "list"]).output
    assert "SITUATION" in runner.invoke(app, ["star", "show", "1"]).output
    bad = runner.invoke(app, ["star", "add", "-c", "nope", "-t", "x", "-s", "s", "--task", "t", "-a", "a", "-r", "r"])
    assert bad.exit_code == 1
    assert runner.invoke(app, ["star", "show", "99"]).exit_code == 1
