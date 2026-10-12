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
