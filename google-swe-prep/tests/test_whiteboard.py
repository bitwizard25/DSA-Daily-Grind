import asyncio

from textual.widgets import Static, TextArea

from google_prep.core.timer import Countdown
from google_prep.db import get_session
from google_prep.models import Problem
from google_prep.modules import mock
from google_prep.ui.whiteboard import Whiteboard


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def board(practice=False, minutes=45):
    with get_session() as s:
        problem = s.get(Problem, "G-200")
    clock = Clock()
    timer = Countdown(minutes * 60, clock)
    saved = []
    app = Whiteboard(problem, minutes, practice, "Week 1 > Day 1 > Mock > Graphs > Number of Islands", timer,
                     [False] * 4, -1, mock.follow_ups(problem), mock.CHECKLIST, mock.scratchpad(problem),
                     saved.append, clock)
    return app, clock, saved, problem


def test_plain_editor_checklist_follow_up_and_rubric_handover():
    app, clock, saved, problem = board()

    async def scenario():
        async with app.run_test(size=(140, 45)) as pilot:
            editor = app.query_one("#editor", TextArea)
            assert editor.language is None  # no syntax highlighting / checking
            assert "Week 1 > Day 1 > Mock > Graphs" in str(app.query_one("#crumbs", Static).render())
            await pilot.press("f1", "f4")
            assert app.ticks == [True, False, False, True]
            editor.move_cursor(editor.document.end)
            await pilot.press(*"seen = set()")
            assert app.query_one("#followup", Static).display is False
            clock.t = 36 * 60  # 35:00: the Google follow-up appears by itself
            app.tick()
            assert app.shown == 0 and app.query_one("#followup", Static).display is True
            assert problem.google_follow_up in str(app.query_one("#followup", Static).render())
            await pilot.press("ctrl+f")
            assert app.shown == 1
            clock.t = 40 * 60  # 40:00: coding ends, rubric takes over
            app.tick()
            await pilot.pause()
        return app.return_value

    result = asyncio.run(scenario())
    assert result["outcome"] == "time" and result["ticks"] == [True, False, False, True] and result["shown"] == 1
    assert "seen = set()" in mock.scratchpad(problem).read_text(encoding="utf-8")  # code persisted
    assert saved and saved[-1]["elapsed"] == 40 * 60


def test_abandon_needs_two_presses_and_pause_freezes_editing():
    app, clock, _, _ = board(practice=True, minutes=60)

    async def scenario():
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.press("ctrl+t")
            assert app.timer.paused and app.query_one("#editor", TextArea).read_only
            await pilot.press("ctrl+t")
            await pilot.press("ctrl+q")
            assert app.return_value is None and not app.done  # one press only warns
            clock.t = 10.0  # second press too late: re-arms
            await pilot.press("ctrl+q")
            assert not app.done
            clock.t = 11.0
            await pilot.press("ctrl+q")
        return app.return_value

    assert asyncio.run(scenario())["outcome"] == "q"


def test_submit_and_scratchpad_is_plain_text():
    app, _, _, problem = board()

    async def scenario():
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.press("ctrl+s")
        return app.return_value

    assert asyncio.run(scenario())["outcome"] == "s"
    assert mock.scratchpad(problem).suffix == ".txt"  # never opens in an IDE with autocomplete
