import asyncio

from google_prep.ui.tui_app import PrepDashboard


def test_dashboard_switches_views_and_launches_mock():
    async def scenario():
        app = PrepDashboard()
        async with app.run_test(size=(110, 50)) as pilot:
            assert app.view == "today" and "enter" in app.actions
            await pilot.press("r")
            assert app.view == "roadmap"
            await pilot.press("s")
            assert app.view == "star"
            await pilot.press("t")
            assert app.view == "today"
            await pilot.press("m")
        return app.return_value

    assert asyncio.run(scenario()) == "mock"


def test_dashboard_enter_starts_the_warmup():
    async def scenario():
        app = PrepDashboard()
        async with app.run_test() as pilot:
            await pilot.press("enter")
        return app.return_value

    assert asyncio.run(scenario()) == "warmup"
