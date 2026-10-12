"""Desktop window: the JSON API, the local server's guards, and the launcher."""
import http.client
import json

import pytest

from google_prep.gui import launcher
from google_prep.gui.api import RPC, Api, ApiError, call
from google_prep.gui.server import serve


@pytest.fixture
def api():
    return Api()


# ---- API --------------------------------------------------------------------------------------
def test_state_and_crumbs(api, home):
    st = api.state()
    assert st["day"] == 3 and st["phase"]["number"] == 1
    assert st["fasttrack"]["pending"] is None
    assert st["data_dir"].endswith(home.name)  # shown in Settings: where progress is saved
    crumbs = api.today()["crumbs"]
    assert crumbs == ["Google SWE II OS", "Phase 1", "Week 1", "Day 3", "Today"]


@pytest.mark.fasttrack
def test_day1_today_serves_fasttrack_and_hides_answers(api):
    d = api.today()
    assert d["kind"] == "fasttrack" and d["action"] == "fasttrack" and d["day"] == 1
    ft = api.fasttrack()
    assert ft["current"] == 1 and [x["day"] for x in ft["days"]] == [1, 2]
    for ex in ft["days"][0]["exercises"]:
        assert "answer" not in ex and "explanation" not in ex and ex["result"] is None


@pytest.mark.fasttrack
def test_fasttrack_answer_and_complete(api):
    ex = api.fasttrack()["days"][0]["exercises"][0]
    from google_prep.modules import fasttrack

    right = fasttrack.exercise(fasttrack.load_day(1), ex["id"])["answer"]
    res = api.fasttrack_answer(ex["id"], right)
    assert res["correct"] and res["answer"] == right and res["output"]
    wrong = api.fasttrack_answer(ex["id"], (right + 1) % 3)
    assert not wrong["correct"]
    assert api.fasttrack()["days"][0]["exercises"][0]["result"] is False  # latest answer wins
    with pytest.raises(ApiError):
        api.fasttrack_answer("d1-e999", 0)
    assert api.fasttrack_complete(1, 30)["streak"] == 1
    assert api.today()["kind"] == "done"
    assert api.state()["fasttrack"]["unfinished"] == [2]


def test_recall_round_trip(api):
    start = api.recall_start("warmup", 3)
    assert len(start["drills"]) == 3 and [g["key"] for g in start["grades"]] == ["1", "2", "3"]
    first = start["drills"][0]
    reveal = api.recall_reveal(first["id"])
    assert reveal["pattern"] and reveal["complexity"]
    slow = api.recall_grade(first["id"], "3", 45.0)  # "instant" after 45 s counts as effort
    assert slow["counted_as"] == "2" and slow["note"]
    assert api.state()["session"]["kind"] == "recall"
    done = api.recall_finish("warmup", {"1": 0, "2": 1, "3": 0}, 70)
    assert done["done"] == 1 and done["minutes"] == 1
    assert api.state()["session"] is None


def test_mock_round_trip(api, home):
    target = api.mock_target()["problem"]
    assert target["time_target"] and target["url"].startswith("https://leetcode.com/problems/")
    board = api.mock_start(target["id"])
    assert board["minutes"] == 45 and "time_target" not in board["problem"]  # the board hides the pattern
    api.mock_save(target["id"], False, 45, 120.0, [True, False, False, False], -1, code="x = 1\n", trace="1 | a | b\n")
    resumed = api.mock_start(resume=True)
    assert resumed["elapsed"] == 120.0 and resumed["ticks"][0] and resumed["code"] == "x = 1\n"
    assert "remaining" in api.state()["session"]["text"]
    sc = api.mock_scorecard(target["id"], False, 45, 1500.0, [True, True, False, False], 0)
    assert [q["field"] for q in sc["questions"]][0] == "clarified_constraints" and sc["questions"][0]["ticked"]
    assert sc["trace_rows"] == 1 and sc["follow_up"]
    assert api.state()["session"]["submitted"]
    res = api.mock_grade(target["id"], False, 45, 1500.0, "2",
                         {"clarified_constraints": True, "stated_complexity": True, "clean_code": True,
                          "dry_run_passed": True, "follow_up_handled": False}, None, 0)
    assert res["solved"] and res["score"] == 75 and res["again"] == "Re-solve in 3 days"
    assert api.state()["session"] is None


def test_problem_card_and_bank(api):
    rows = api.problems()["problems"]
    assert len(rows) == 75
    assert api.problem_card(rows[0]["id"])["problem"]["title"] == rows[0]["title"]
    with pytest.raises(ApiError):
        api.problem_card("nope")


def test_star_save_edit_delete(api):
    with pytest.raises(ApiError):
        api.star_save("Ambiguity", "t", "", "x", "x", "x")
    data = api.star_save("ambiguity", "Title", "S", "T", "A", "R")
    story = data["stories"][0]
    assert story["dimension"] == "Ambiguity"
    edited = api.star_save("Ownership", "New", "S", "T", "A", "R", id=story["id"])
    assert edited["stories"][0]["title"] == "New" and edited["stories"][0]["dimension"] == "Ownership"
    assert api.star_delete(story["id"])["stories"] == []
    assert api.star_practice("Humility")["prompt"]


def test_vault_lists_and_renders_safely(api):
    notes = api.vault_list()["notes"]
    assert notes and all(n["file"].endswith(".md") for n in notes)
    note = api.vault_note(notes[0]["file"])
    assert note["html"].startswith("<h1>")
    for bad in ("../README.md", "../../etc/passwd", "x.txt", "missing.md"):
        with pytest.raises(ApiError):
            api.vault_note(bad)


def test_settings_validation(api):
    with pytest.raises(ApiError):
        api.settings_save(interview_date="12/01/2026")
    st = api.settings_save(name="Ravi", interview_date="2026-12-20")
    assert st["settings"]["name"] == "Ravi" and st["interview_date"] == "2026-12-20"


def test_open_link_only_https(api, monkeypatch):
    opened = []
    monkeypatch.setattr("webbrowser.open", opened.append)
    with pytest.raises(ApiError):
        api.open_link("file:///etc/passwd")
    api.open_link("https://leetcode.com/problems/two-sum/")
    assert opened == ["https://leetcode.com/problems/two-sum/"]


def test_dispatch_whitelist(api):
    assert "state" in RPC and "_session_summary" not in RPC and "_problem_card" not in RPC
    with pytest.raises(ApiError):
        call(api, "_session_summary", {})
    with pytest.raises(ApiError):
        call(api, "__init__", {})
    with pytest.raises(ApiError):
        call(api, "today", {"bogus": 1})


# ---- HTTP server ------------------------------------------------------------------------------
@pytest.fixture
def server():
    srv = serve(Api())
    yield srv
    srv.shutdown()


def _request(server, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    conn.request(method, path, body=body, headers=headers or {})
    res = conn.getresponse()
    data = res.read()
    conn.close()
    return res, data


def test_page_served_with_csp(server):
    res, body = _request(server, "GET", "/")
    assert res.status == 200 and b"<title>Google SWE II OS</title>" in body
    assert "script-src 'self'" in res.getheader("Content-Security-Policy")
    for asset in ("/app.js", "/app.css", "/icon.svg", "/icon.png"):
        assert _request(server, "GET", asset)[0].status == 200
    assert _request(server, "GET", "/../api.py")[0].status == 404
    assert _request(server, "GET", "/%2e%2e/api.py")[0].status == 404


def test_api_requires_token_and_host(server):
    ok = {"X-Prep-Token": server.token, "Content-Type": "application/json"}
    res, body = _request(server, "POST", "/api/state", b"{}", ok)
    assert res.status == 200 and json.loads(body)["day"] == 3
    assert _request(server, "POST", "/api/state", b"{}", {"X-Prep-Token": "wrong"})[0].status == 403
    assert _request(server, "POST", "/api/state", b"{}", {})[0].status == 403
    evil = {**ok, "Host": "evil.example:80"}  # DNS rebinding
    assert _request(server, "POST", "/api/state", b"{}", evil)[0].status == 403
    assert _request(server, "GET", "/", headers={"Host": "evil.example"})[0].status == 403
    res, body = _request(server, "POST", "/api/nope", b"{}", ok)
    assert res.status == 400 and "Unknown action" in json.loads(body)["error"]
    assert _request(server, "POST", "/api/state", b"not json", ok)[0].status == 400
    big = {**ok, "Content-Length": str(3_000_000)}
    assert _request(server, "POST", "/api/state", b"", big)[0].status == 413


# ---- launcher ---------------------------------------------------------------------------------
def test_selftest_and_version(capsys):
    assert launcher.main(["--selftest"]) == 0
    assert launcher.main(["--version"]) == 0
    assert "google-prep" in capsys.readouterr().out


def test_browser_override(monkeypatch):
    monkeypatch.setenv("GOOGLE_PREP_BROWSER", "/opt/my-chrome")
    assert launcher.find_browser() == "/opt/my-chrome"


def test_window_handoff_falls_back_to_heartbeat(monkeypatch):
    """A browser that hands the page to an already-open copy exits at once; the server must stay up
    until the page stops sending heartbeats instead of quitting under the user."""
    calls = []

    class Quick:
        def wait(self):
            calls.append("wait")
            return 0

    monkeypatch.setattr(launcher, "open_window", lambda url: Quick())
    monkeypatch.setattr(launcher, "HEARTBEAT_TIMEOUT", 0)
    assert launcher.main([]) == 0
    assert calls == ["wait"]


# ---- review fixes: sessions are never silently lost ---------------------------------------------
def _submitted_mock(api):
    pid = api.mock_target()["problem"]["id"]
    api.mock_start(pid, practice=True)
    api.mock_scorecard(pid, True, 60, 1500.0, [False] * 4, -1)
    return pid


def test_starting_drills_never_overwrites_an_unfinished_mock(api):
    pid = _submitted_mock(api)
    res = api.recall_start("warmup", 3)  # e.g. Enter on Today's warmup card
    assert "scorecard not finished" in res["conflict"]["text"]
    assert api.state()["session"]["problem_id"] == pid  # untouched
    assert api.mock_start(resume=True)["submitted"]  # still resumable straight into the scorecard
    started = api.recall_start("warmup", 3, discard=True)  # the user chose to discard it
    assert started["drills"] and api.state()["session"]["kind"] == "recall"


def test_starting_a_mock_never_overwrites_unfinished_drills(api):
    first = api.recall_start("warmup", 3)["drills"][0]["id"]
    api.recall_grade(first, "2", 10.0)
    assert "conflict" in api.mock_start(api.mock_target()["problem"]["id"])
    assert api.recall_start(resume=True)["index"] == 1


def test_same_problem_resumes_instead_of_conflicting(api):
    pid = api.mock_target()["problem"]["id"]
    api.mock_start(pid)
    api.mock_save(pid, False, 45, 300.0, [True, False, False, False], -1)
    again = api.mock_start(pid)  # picked the same problem again from the bank
    assert again["elapsed"] == 300.0 and again["ticks"][0]


def test_grading_a_drill_credits_the_day_at_once(api, session):
    from google_prep.core import scheduler
    from conftest import NOW

    drill = api.recall_start("warmup", 3)["drills"][0]["id"]
    api.recall_grade(drill, "3", 5.0)  # then the window is closed: no recall_finish
    assert scheduler.schedule_for(session, NOW.date()).morning_completed


def test_discard_keeps_credit_for_graded_drills(api, session):
    from google_prep.core import scheduler
    from conftest import NOW

    drill = api.recall_start("warmup", 3)["drills"][0]["id"]
    api.recall_grade(drill, "2", 30.0)
    api.session_discard()
    assert api.state()["session"] is None
    assert scheduler.schedule_for(session, NOW.date()).morning_completed


def test_micro_mode_does_not_stick_for_the_whole_day(api):
    from conftest import NOW

    late = Api(now=lambda: NOW.replace(hour=19, minute=30))
    start = late.recall_start("micro")
    for d in start["drills"]:
        late.recall_grade(d["id"], "2", 10.0)
    late.recall_finish("micro", {"1": 0, "2": len(start["drills"]), "3": 0}, 300)
    assert late.today()["kind"] == "deep"  # the evening solve is still offered
    assert late.today("full")["kind"] == "deep"


def test_a_mock_that_runs_past_midnight_counts_for_the_day_it_began(home):
    from datetime import datetime

    from google_prep.core import scheduler
    from google_prep.db import get_session
    from google_prep.models import Attempt
    from sqlmodel import select

    clock = {"t": datetime(2026, 10, 12, 23, 20)}
    api = Api(now=lambda: clock["t"])
    pid = api.mock_target()["problem"]["id"]
    api.mock_start(pid, practice=True)
    clock["t"] = datetime(2026, 10, 13, 0, 25)
    api.mock_scorecard(pid, True, 60, 3600.0, [False] * 4, -1)
    api.mock_grade(pid, True, 60, 3600.0, "2", {"dry_run_passed": True}, None, -1)
    with get_session() as s:
        assert scheduler.schedule_for(s, datetime(2026, 10, 12).date()).evening_completed
        assert not scheduler.schedule_for(s, datetime(2026, 10, 13).date()).evening_completed
        assert {a.day for a in s.exec(select(Attempt)).all()} == {"2026-10-12"}


def test_mock_grade_rejects_an_unknown_outcome(api):
    pid = api.mock_target()["problem"]["id"]
    with pytest.raises(ApiError):
        api.mock_grade(pid, False, 45, 100.0, "9", {}, None, -1)


# ---- review fixes: server and launcher ----------------------------------------------------------
def test_static_types_do_not_depend_on_the_os_registry(server, monkeypatch):
    import mimetypes

    monkeypatch.setattr(mimetypes, "guess_type", lambda *a, **k: ("text/plain", None))  # a broken Windows registry
    res, _ = _request(server, "GET", "/app.js")
    assert res.getheader("Content-Type") == "text/javascript; charset=utf-8"
    assert _request(server, "GET", "/app.css")[0].getheader("Content-Type") == "text/css; charset=utf-8"


def test_old_commands_open_the_window_on_their_screen(monkeypatch):
    urls = []

    class Quick:
        def wait(self):
            return 0

    monkeypatch.setattr(launcher, "open_window", lambda url: urls.append(url) or Quick())
    monkeypatch.setattr(launcher, "HEARTBEAT_TIMEOUT", 0)
    assert launcher.main(["today"]) == 0  # what old reminders run
    assert launcher.main(["recall"]) == 0
    assert "#/" not in urls[0] and urls[1].endswith("#/recall")
    with pytest.raises(SystemExit) as exc:
        launcher.main(["add", "two-sum"])  # terminal-only commands get a clear message
    assert exc.value.code == 2


def test_selftest_reports_failure_instead_of_raising(monkeypatch, tmp_path, home):
    from google_prep.gui import server as server_mod

    monkeypatch.setattr(server_mod, "STATIC", tmp_path)  # a bundle without its static files
    assert launcher.selftest() == 1
    assert "Traceback" in (home / "selftest.log").read_text()


def test_reminders_open_the_window_when_installed(monkeypatch, tmp_path):
    import sys

    from google_prep.modules import remind

    cli = tmp_path / "google-prep-cli.exe"
    gui = tmp_path / "google-prep.exe"
    cli.write_bytes(b"")
    gui.write_bytes(b"")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(cli))
    line = remind.commands("07:30", "19:30", "windows")[0]
    assert str(gui) in line and "cmd /k" not in line
