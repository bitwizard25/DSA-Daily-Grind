import asyncio

from textual.widgets import Input, Select, Static

from google_prep.db import get_session
from google_prep.core import scheduler
from google_prep.models import MockSession, StarStory
from google_prep.ui.tui_app import PrepUI, StarForm, StarStudio
from google_prep.ui.whiteboard import RubricScreen, WhiteboardScreen


def test_ui_views_and_star_studio_share_the_cli_database():
    async def scenario():
        app = PrepUI()
        async with app.run_test(size=(140, 50)) as pilot:
            assert app.view == "today" and "enter" in app.actions
            await pilot.press("r")
            assert app.view == "roadmap" and "READY" in str(app.query_one("#body", Static).render())  # the gauge
            await pilot.press("s")
            assert isinstance(app.screen, StarStudio)
            await pilot.press("a")
            form = app.screen
            assert isinstance(form, StarForm)
            form.query_one("#dimension", Select).value = "Disagreement"
            for field, value in [("title", "Hapminds: queue vs. cron debate"), ("situation", "Two designs"),
                                 ("task", "Decide"), ("action", "Wrote a doc with load numbers"),
                                 ("result", "Queue shipped, p99 -40%")]:
                form.query_one(f"#{field}", Input).value = value
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert isinstance(app.screen, StarStudio)
            await pilot.press("v")
            assert "PRACTICE" in str(app.screen.query_one("#detail", Static).render())
            await pilot.press("escape")
            assert not isinstance(app.screen, StarStudio)

    asyncio.run(scenario())
    with get_session() as s:
        story = s.exec(scheduler.select(StarStory)).one()
    assert story.dimension == "Disagreement" and story.result == "Queue shipped, p99 -40%"


def test_mock_runs_inside_the_ui_and_is_logged():
    async def scenario():
        app = PrepUI()
        async with app.run_test(size=(150, 50)) as pilot:
            await pilot.press("w")
            assert isinstance(app.screen, WhiteboardScreen)
            await pilot.press("f1", "ctrl+s")
            await pilot.pause()
            assert isinstance(app.screen, RubricScreen)
            await pilot.press("3", "y", "y", "y", "y", "y")  # Instant Reflex, then five yes
            await pilot.pause()
            assert app.view == "today"

    asyncio.run(scenario())
    with get_session() as s:
        m = s.exec(scheduler.select(MockSession)).one()
    assert m.solved and m.score == 100 and m.clarified_constraints


def test_ui_keys_match_the_cli_section_bar():
    async def scenario(key):
        app = PrepUI()
        async with app.run_test() as pilot:
            await pilot.press(key)
        return app.return_value

    assert asyncio.run(scenario("d")) == "warmup"
    assert asyncio.run(scenario("m")) == "micro_start"


def test_enter_hands_recall_drills_to_the_terminal():
    async def scenario():
        app = PrepUI()
        async with app.run_test() as pilot:
            await pilot.press("enter")
        return app.return_value

    assert asyncio.run(scenario()) == "warmup"
