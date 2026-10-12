"""`google-prep ui`: the full-screen interview cockpit. Same database as the CLI.

Views: Today, Roadmap (with the circular readiness gauge), STAR Studio, and the 45-minute
whiteboard + 4-pillar scorecard running inside the app. Recall drills are terminal-first
(single keys, 60-second timer), so starting them closes the app and runs them in the shell."""
from typing import Callable

from rich.console import Console, RenderableType
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import DataTable, Footer, Input, Select, Static

from google_prep.core import clock, readiness, scheduler
from google_prep.core.timer import Countdown
from google_prep.db import get_session
from google_prep.models import StarStory
from google_prep.modules import mock, roadmap, star, today
from google_prep.ui.formatters import crumb_text, nav
from google_prep.ui.theme import BODY, MUTED, SURFACE, TEXT_BLUE, TEXT_GREEN, TEXT_YELLOW, THEME
from google_prep.ui.whiteboard import CSS as BOARD_CSS
from google_prep.ui.whiteboard import WhiteboardScreen


def to_text(renderable: RenderableType, width: int) -> Text:
    """Render with the app theme, so theme style names survive inside Textual."""
    console = Console(theme=THEME, width=max(40, width), force_terminal=True, color_system="truecolor",
                      highlight=False)
    with console.capture() as cap:
        console.print(renderable)
    return Text.from_ansi(cap.get())


class StarForm(ModalScreen[dict | None]):
    BINDINGS = [Binding("ctrl+s", "save", "Save", priority=True), Binding("escape", "cancel", "Cancel")]
    FIELDS = [("title", "Title, e.g. 'Hapminds: cache migration under a hard deadline'"),
              ("situation", "Situation: context in 1-2 sentences"), ("task", "Task: what YOU were responsible for"),
              ("action", "Action: what you did, specifically"), ("result", "Result: numbers (latency, cost, time saved)")]

    def __init__(self, dimension: str | None = None):
        super().__init__()
        self.dimension = dimension

    def compose(self) -> ComposeResult:
        with Vertical(id="card"):
            yield Static(Text("NEW STAR STORY", style=f"bold {TEXT_BLUE}"))
            yield Select([(label, key) for key, label in star.DIMENSIONS.items()], prompt="Google dimension",
                         value=self.dimension or Select.NULL, id="dimension")
            for key, placeholder in self.FIELDS:
                yield Input(placeholder=placeholder, id=key)
            yield Static(Text("Ctrl+S save · Esc cancel", style=MUTED))

    def action_save(self) -> None:
        dim = self.query_one("#dimension", Select).value
        values = {key: self.query_one(f"#{key}", Input).value.strip() for key, _ in self.FIELDS}
        if dim is Select.NULL or not all(values.values()):
            self.notify("Pick a dimension and fill all five fields.", severity="error")
            return
        self.dismiss({"dimension": dim, **values})

    def action_cancel(self) -> None:
        self.dismiss(None)


class StarStudio(Screen):
    BINDINGS = [Binding("a", "add", "Add"), Binding("v", "review", "Practice"),
                Binding("b", "back", "Back"), Binding("escape", "back", "Back", show=False),
                Binding("q", "app.quit", "Quit")]

    def compose(self) -> ComposeResult:
        with get_session() as session:
            path = scheduler.breadcrumb(session, clock.now().date(), "STAR Studio")
        yield Static(crumb_text(path), classes="crumbs")
        yield Static(nav("STAR"), classes="crumbs")
        yield Static(Text("STAR STORIES", style=f"bold {TEXT_BLUE}"), classes="crumbs")
        yield Static(id="coverage")
        yield DataTable(id="stories", cursor_type="row")
        yield VerticalScroll(Static(id="detail"))
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#stories", DataTable)
        table.add_columns("ID", "Dimension", "Title", "Created")
        self.load()

    def load(self) -> None:
        with get_session() as session:
            stories = session.exec(scheduler.select(StarStory).order_by(StarStory.dimension, StarStory.id)).all()
            counts = readiness.star_counts(session)
        need = readiness.STARS_PER_DIMENSION
        cov = Text("Coverage  ", style="bold")
        for dim in star.DIMENSIONS:
            cov.append(f"{dim} {min(counts[dim], need)}/{need}   ", style=TEXT_GREEN if counts[dim] >= need else TEXT_YELLOW)
        self.query_one("#coverage", Static).update(cov)
        table = self.query_one("#stories", DataTable)
        table.clear()
        self.stories = {str(s.id): s for s in stories}
        for s in stories:
            table.add_row(str(s.id), s.dimension, s.title, f"{s.created_at:%Y-%m-%d}", key=str(s.id))
        if not stories:
            self.query_one("#detail", Static).update(Text("No stories yet. Press a to add one.", style=MUTED))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        story = self.stories.get(event.row_key.value) if event.row_key else None
        if story:
            self.query_one("#detail", Static).update(Text.assemble(
                (f"{story.title}\n", "bold"), (f"{star.DIMENSIONS.get(story.dimension, story.dimension)}\n\n", MUTED),
                ("SITUATION ", TEXT_BLUE), f"{story.situation}\n", ("TASK      ", TEXT_BLUE), f"{story.task}\n",
                ("ACTION    ", TEXT_BLUE), f"{story.action}\n", ("RESULT    ", TEXT_BLUE), f"{story.result}"))

    def action_add(self) -> None:
        def saved(values: dict | None) -> None:
            if values:
                with get_session() as session:
                    star.add(session, **values)
                self.notify(f"Saved under {values['dimension']}.")
                self.load()
        self.app.push_screen(StarForm(), saved)

    def action_review(self) -> None:
        with get_session() as session:
            dim, prompt = star.review_prompt(session)
            titles = [s.title for s in star.stories_for(session, dim)]
        body = Text.assemble((f"PRACTICE · {star.DIMENSIONS[dim]}\n\n", f"bold {TEXT_BLUE}"), f'"{prompt}"\n\n',
                             ("Out loud, under 2 min: Situation > Task > Action > Result. Quantify the result.\n\n", MUTED))
        body.append("Your stories for this dimension: " + (", ".join(titles) if titles else "none yet (press a)"))
        self.query_one("#detail", Static).update(body)

    def action_back(self) -> None:
        self.app.pop_screen()


class PrepUI(App[str | None]):
    TITLE = "Google SWE II OS"
    ENABLE_COMMAND_PALETTE = False
    CSS = BOARD_CSS + f"""
    #body {{ padding: 1 2; }}
    StarForm {{ align: center middle; }}
    StarForm #card {{ width: 90; background: {SURFACE}; }}
    StarForm Input, StarForm Select {{ margin-top: 1; }}
    DataTable {{ height: auto; max-height: 14; color: {BODY}; }}
    #coverage {{ padding: 0 1; }}
    """
    BINDINGS = [  # same keys as the CLI section bar: t d w r s, plus m for micro-mode
        Binding("enter", "start", "Start"),
        Binding("t", "show('today')", "Today"),
        Binding("d", "drills", "Recall"),
        Binding("w", "mock", "Mock"),
        Binding("r", "show('roadmap')", "Roadmap"),
        Binding("s", "studio", "STAR"),
        Binding("m", "micro", "Micro-Mode (10 min)"),
        Binding("escape", "show('today')", "Back", show=False),
        Binding("q", "quit", "Quit"),
        Binding("ctrl+q", "guarded_quit", show=False, priority=True),
    ]

    def action_guarded_quit(self) -> None:
        """Ctrl+Q quits, except during a round, where it is the whiteboard's press-twice abandon."""
        if isinstance(self.screen, WhiteboardScreen):
            self.screen.action_abandon()
        else:
            self.exit()

    def __init__(self, now: Callable = clock.now):
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
            else:
                renderable, self.actions = today.build(session, self.now())
        self.view = view
        self.query_one("#body", Static).update(to_text(renderable, self.size.width - 6))

    def action_studio(self) -> None:
        self.push_screen(StarStudio())

    def action_drills(self) -> None:
        self.exit("warmup")  # 60-second drills run in the terminal

    def action_micro(self) -> None:
        self.exit("micro_start")

    def action_start(self) -> None:
        action = self.actions.get("enter")
        if action == "deep":
            self.run_round(mock_round=False)
        elif action:
            self.exit(action)  # recall drills and re-solves run in the terminal

    def action_mock(self) -> None:
        self.run_round(mock_round=True)

    def run_round(self, mock_round: bool) -> None:
        """Whiteboard -> scorecard -> auto-log, all inside the app."""
        when = self.now()
        with get_session() as session:
            phase = scheduler.phase_for_day(scheduler.day_number(session, when.date()))
            problem = scheduler.pick_target_problem(session, phase, when)
            practice = not mock_round and phase < 3
            minutes = 60 if practice else 45
            path = scheduler.breadcrumb(session, when.date(), "Solve" if practice else "Mock",
                                        scheduler.CATEGORIES[problem.category], problem.title)
        timer = Countdown(minutes * 60)
        board = mock.board_screen(problem, minutes, practice, timer, [False] * 4, -1, path)

        def graded(result: dict, grade: dict | None) -> None:
            if not grade:
                return
            with get_session() as session:
                out = mock.save_graded(session, problem, minutes, practice, result, grade, when)
            msg = (f"Logged {problem.title}: " + ("solved" if out["solved"] else "not solved")
                   + ("" if practice else f" · score {out['score']:.0f}/100 · {out['signal']}"))
            self.notify(msg, timeout=8)
            self.action_show("today")

        def after_board(result: dict | None) -> None:
            if not result or result["outcome"] == "q":
                mock.session_store.clear()
                self.notify("Abandoned. Not logged.")
                return
            self.push_screen(mock.rubric_screen(problem, minutes, practice, timer, result, path),
                             lambda grade: graded(result, grade))

        self.push_screen(board, after_board)


PrepDashboard = PrepUI  # v1 name
