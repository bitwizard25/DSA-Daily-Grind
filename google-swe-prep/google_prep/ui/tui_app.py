"""Full-screen Textual dashboard: today / roadmap / STAR vault with single-key navigation.
Timed drills and mocks run in the normal terminal, so launching one exits the dashboard
and returns the action for the CLI to run."""
from datetime import datetime
from typing import Callable

from rich.console import Console, RenderableType
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Static

from google_prep.core import clock
from google_prep.db import get_session
from google_prep.modules import roadmap, star, today
from google_prep.ui.theme import BODY, SURFACE, THEME


def to_text(renderable: RenderableType, width: int) -> Text:
    """Render with the app theme, so theme style names survive inside Textual."""
    console = Console(theme=THEME, width=max(40, width), force_terminal=True, color_system="truecolor",
                      highlight=False)
    with console.capture() as cap:
        console.print(renderable)
    return Text.from_ansi(cap.get())


class PrepDashboard(App[str | None]):
    TITLE = "Google SWE II OS"
    CSS = f"""
    Screen {{ background: {SURFACE}; color: {BODY}; }}
    #body {{ padding: 1 2; }}
    """
    BINDINGS = [
        Binding("t", "show('today')", "Today"),
        Binding("r", "show('roadmap')", "Roadmap"),
        Binding("s", "show('star')", "STAR Vault"),
        Binding("m", "launch('mock')", "Mock"),
        Binding("enter", "launch('start')", "Start"),
        Binding("q", "quit", "Exit"),
    ]

    def __init__(self, now: Callable[[], datetime] = clock.now):
        super().__init__()
        self.now = now
        self.view = "today"
        self.actions: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield VerticalScroll(Static(id="body"))
        yield Footer()

    def on_mount(self) -> None:
        self.action_show("today")

    def action_show(self, view: str) -> None:
        with get_session() as session:
            if view == "roadmap":
                renderable, self.actions = roadmap.build(session, self.now().date())
            elif view == "star":
                renderable, self.actions = star.list_view(session), {}
            else:
                renderable, self.actions = today.build(session, self.now())
        self.view = view
        self.query_one("#body", Static).update(to_text(renderable, self.size.width - 6))

    def action_launch(self, what: str) -> None:
        if what == "start":
            target = self.actions.get("enter")
            if target:
                self.exit(target)
        else:
            self.exit(what)
