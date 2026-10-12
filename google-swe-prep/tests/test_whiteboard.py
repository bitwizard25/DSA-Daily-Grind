import asyncio

from textual.widgets import Static, TabbedContent, TextArea

from google_prep.core.timer import Countdown
from google_prep.db import get_session
from google_prep.models import Problem
from google_prep.modules import mock
from google_prep.ui.whiteboard import MockApp, RubricScreen, WhiteboardScreen

PATH = "[Google SWE II OS] Phase 1 > Week 1 > Day 1 > Mock > Graphs > Number of Islands"


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def problem():
    with get_session() as s:
        return s.get(Problem, "G-200")


def board(p, clock, practice=False, minutes=45):
    timer = Countdown(minutes * 60, clock)
    return mock.board_screen(p, minutes, practice, timer, [False] * 4, -1, PATH, clock), timer


def test_split_screen_follow_up_banner_trace_and_scorecard_handover():
    p, clock = problem(), Clock()
    screen, timer = board(p, clock)
    app = MockApp(screen, lambda b: mock.rubric_screen(p, 45, False, timer, b, PATH))

    async def scenario():
        async with app.run_test(size=(150, 50)) as pilot:
            wb = app.screen
            assert isinstance(wb, WhiteboardScreen)
            assert wb.query_one("#editor", TextArea).language is None  # no highlighting, nothing runs
            assert "[Google SWE II OS] Phase 1 > Week 1 > Day 1 > Mock > Graphs" in str(wb.query_one(".crumbs", Static).render())
            assert wb.query_one("#banner", Static).display is False
            await pilot.press("f1", "f2")
            editor = wb.query_one("#editor", TextArea)
            editor.move_cursor(editor.document.end)
            await pilot.press(*"seen = set()")
            await pilot.press("f6")  # trace table tab
            assert wb.query_one(TabbedContent).active == "trace"
            tracer = wb.query_one("#tracer", TextArea)
            tracer.text = tracer.text.replace("1    |      |", "1    | 3    | i=0 grid=[[1]]", 1)
            clock.t = 36 * 60  # minute 35: the follow-up banner is injected
            wb.tick()
            banner = wb.query_one("#banner", Static)
            assert banner.display is True and p.google_follow_up in str(banner.render())
            clock.t = 40 * 60  # minute 40: the scorecard takes over
            wb.tick()
            await pilot.pause()
            card = app.screen
            assert isinstance(card, RubricScreen)
            shown = lambda: str(card.query_one("#card", Static).render())  # noqa: E731
            assert "Missed it" in shown() and "Instant Reflex" in shown() and "will count as Effort" in shown()
            await pilot.press("2")  # Solved with Effort
            assert "You ticked this during the round" in shown()  # F1 pre-noted
            await pilot.press("y", "y", "y")
            assert "Trace table rows you filled: 1" in shown()
            await pilot.press("y")
            assert p.google_follow_up in shown()
            await pilot.press("b")  # one step back is always possible
            assert "Trace table rows you filled: 1" in shown()
            await pilot.press("y", "y")
            # 40 minutes of coding is past the 35-minute target, so the scorecard asks why
            assert "What hurt most" in shown()
            await pilot.press("4")
        return app.return_value

    result = asyncio.run(scenario())
    assert result["outcome"] == "time" and result["elapsed"] == 40 * 60
    graded = result["graded"]
    assert graded["outcome"] == "2" and all(graded["checks"].values()) and graded["mistake"] == "slow"
    assert "seen = set()" in mock.scratchpad(p).read_text(encoding="utf-8")
    assert mock.scratchpad(p).parent.name == "sessions"  # code auto-saves next to the session
    assert "i=0 grid=[[1]]" in mock.trace_pad(p).read_text(encoding="utf-8")


def test_scorecard_asks_for_the_mistake_when_it_went_badly():
    p, clock = problem(), Clock()
    screen, timer = board(p, clock)
    app = MockApp(screen, lambda b: mock.rubric_screen(p, 45, False, timer, b, PATH))

    async def scenario():
        async with app.run_test(size=(150, 50)) as pilot:
            await pilot.press("ctrl+s")
            await pilot.pause()
            await pilot.press("1", "n", "n", "n", "n", "n")  # Missed it, five no
            assert "What hurt most" in str(app.screen.query_one("#card", Static).render())
            await pilot.press("3")
        return app.return_value

    graded = asyncio.run(scenario())["graded"]
    assert graded["solved"] is False and graded["outcome"] == "1" and graded["mistake"] == "bug_edge"


def test_abandon_needs_two_presses_and_pause_freezes_both_editors():
    p, clock = problem(), Clock()
    screen, timer = board(p, clock, practice=True, minutes=60)
    app = MockApp(screen, lambda b: mock.rubric_screen(p, 60, True, timer, b, PATH))

    async def scenario():
        async with app.run_test(size=(150, 50)) as pilot:
            wb = app.screen
            await pilot.press("ctrl+t")
            assert timer.paused and all(a.read_only for a in wb.query(TextArea))
            await pilot.press("ctrl+t")
            await pilot.press("ctrl+q")
            assert not wb.done  # one press only warns
            clock.t = 10.0
            await pilot.press("ctrl+q")  # too late: re-arms
            assert not wb.done
            clock.t = 11.0
            await pilot.press("ctrl+q")
        return app.return_value

    assert asyncio.run(scenario())["outcome"] == "q"


def test_trace_rows_counts_only_filled_steps():
    assert mock.trace_rows(mock.TRACE_TEMPLATE) == 0
    filled = mock.TRACE_TEMPLATE.replace("2    |      |", "2    | 7    | lo=0 hi=4", 1)
    assert mock.trace_rows(filled) == 1
