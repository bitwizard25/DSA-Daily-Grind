"""Full-screen 45-minute Google whiteboard: problem, plain-text editor (no syntax
highlighting, no autocomplete), live timer, phase timeline, checklist and follow-up.
The rubric (40:00-45:00) runs after this screen closes."""
import time
from pathlib import Path
from typing import Callable

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Static, TextArea

from google_prep.core.timer import FOLLOW_UP_AT, Countdown, mmss
from google_prep.ui.theme import BLUE, BODY, GREEN, MUTED, RED, SURFACE, YELLOW

AUTOSAVE_SECONDS = 5
RUBRIC_AT = 40 / 45  # mocks hand over to the rubric at 40:00


class Whiteboard(App[dict]):
    ENABLE_COMMAND_PALETTE = False
    CSS = f"""
    Screen {{ background: {SURFACE}; color: {BODY}; }}
    #crumbs {{ color: {BLUE}; text-style: bold; padding: 0 1; }}
    #head {{ border: round {BLUE}; padding: 0 1; height: 3; }}
    #problem {{ padding: 0 1; height: auto; max-height: 6; }}
    #editor {{ border: solid {MUTED}; }}
    #editor:focus {{ border: solid {BLUE}; }}
    #side {{ width: 42; padding: 0 1; }}
    .panel {{ border: solid {BLUE}; padding: 0 1; height: auto; margin-bottom: 1; }}
    #followup {{ border: solid {YELLOW}; }}
    """
    BINDINGS = [
        Binding("ctrl+s", "submit", "Submit & grade", priority=True),
        Binding("ctrl+f", "follow_up", "Follow-up", priority=True),
        Binding("ctrl+t", "pause", "Pause", priority=True),
        Binding("f1", "tick(0)", "✓1", priority=True),
        Binding("f2", "tick(1)", "✓2", priority=True),
        Binding("f3", "tick(2)", "✓3", priority=True),
        Binding("f4", "tick(3)", "✓4", priority=True),
        Binding("ctrl+q", "abandon", "Abandon (x2)", priority=True),
    ]

    def __init__(self, problem, minutes: int, practice: bool, path: str, timer: Countdown,
                 ticks: list[bool], shown: int, follow_ups: list[str], checklist: list[str], pad: Path,
                 autosave: Callable[[dict], None], clock: Callable[[], float] = time.monotonic):
        super().__init__()
        self.checklist = checklist
        self.problem, self.minutes, self.practice, self.path = problem, minutes, practice, path
        self.timer, self.ticks, self.shown, self.follow_ups = timer, list(ticks), shown, follow_ups
        self.pad, self.autosave_cb, self.clock = pad, autosave, clock
        self.last_save = -AUTOSAVE_SECONDS
        self.abandon_armed_at: float | None = None
        self.done = False

    # ---- layout -------------------------------------------------------------
    def compose(self) -> ComposeResult:
        text = self.pad.read_text(encoding="utf-8")
        yield Static(self.path, id="crumbs")
        yield Static(id="head")
        with Horizontal():
            with Vertical():
                yield Static(self.problem.statement, id="problem")
                yield TextArea(text, id="editor", language=None, soft_wrap=True, tab_behavior="indent",
                               show_line_numbers=False, highlight_cursor_line=False)
            with Vertical(id="side"):
                yield Static(id="timeline", classes="panel")
                yield Static(id="checklist", classes="panel")
                yield Static(id="followup", classes="panel")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#editor", TextArea).focus()
        self.set_interval(0.5, self.tick)
        self.tick()

    # ---- state ----------------------------------------------------------------
    def state(self) -> dict:
        return {"elapsed": self.timer.elapsed(), "ticks": self.ticks, "shown": self.shown}

    def save(self, force: bool = False) -> None:
        if force or self.clock() - self.last_save >= AUTOSAVE_SECONDS:
            self.pad.write_text(self.query_one("#editor", TextArea).text, encoding="utf-8")
            self.autosave_cb(self.state())
            self.last_save = self.clock()

    def finish(self, outcome: str) -> None:
        if self.done:
            return
        self.done = True
        self.save(force=True)
        self.exit({"outcome": outcome, **self.state()})

    # ---- rendering --------------------------------------------------------------
    def boundaries(self) -> list[float]:
        total = self.minutes * 60
        return [0, total * 5 / 45, total * 35 / 45, total * 40 / 45, total]

    def tick(self) -> None:
        if self.done:
            return
        frac = self.timer.fraction()
        if self.shown < 0 and frac >= FOLLOW_UP_AT:
            self.shown = 0
            self.bell()
        if not self.practice and frac >= RUBRIC_AT:
            return self.finish("time")
        if self.practice and self.timer.expired():
            return self.finish("time")
        rem = self.timer.remaining()
        colour = RED if rem <= 300 else YELLOW
        title = "TIMED SOLVE" if self.practice else "MOCK INTERVIEW · Google Coding Round"
        head = Text.assemble((title, f"bold {BLUE}"), "   ", (f"{self.problem.title} ({self.problem.difficulty})", "bold"),
                             "   ", (f"⏱ {mmss(rem)} left", f"bold {colour}"),
                             ("  (paused)" if self.timer.paused else "", MUTED))
        self.query_one("#head", Static).update(head)

        b = self.boundaries()
        labels = ["Scoping & constraints", "Code: plain text, no autocomplete", "Google follow-up",
                  "Rubric self-evaluation"]
        elapsed = self.timer.elapsed()
        lines = Text("TIMELINE\n", style="bold")
        for i, label in enumerate(labels):
            active = b[i] <= elapsed < b[i + 1]
            past = elapsed >= b[i + 1]
            style = f"bold {BLUE}" if active else (GREEN if past else MUTED)
            mark = "▶" if active else ("✓" if past else " ")
            lines.append(f"{mark} {mmss(b[i])}–{mmss(b[i + 1])} {label}\n", style=style)
        self.query_one("#timeline", Static).update(lines)

        check = Text("CHECKLIST (F1-F4)\n", style="bold")
        for i, (done, item) in enumerate(zip(self.ticks, self.checklist)):
            check.append(f"{'[✓]' if done else '[ ]'} {i + 1}. {item}\n", style=GREEN if done else MUTED)
        self.query_one("#checklist", Static).update(check)

        follow = self.query_one("#followup", Static)
        if self.shown >= 0:
            follow.display = True
            follow.update(Text.assemble(("GOOGLE FOLLOW-UP\n", f"bold {YELLOW}"),
                                        self.follow_ups[self.shown % len(self.follow_ups)],
                                        ("\nCtrl+F: next constraint", MUTED)))
        else:
            follow.display = False
        self.save()

    # ---- actions ------------------------------------------------------------------
    def action_submit(self) -> None:
        self.finish("s")

    def action_follow_up(self) -> None:
        self.shown = (self.shown + 1) % len(self.follow_ups)
        self.tick()

    def action_pause(self) -> None:
        self.timer.toggle_pause()
        self.query_one("#editor", TextArea).read_only = self.timer.paused
        self.tick()

    def action_tick(self, index: int) -> None:
        self.ticks[index] = not self.ticks[index]
        self.tick()

    def action_abandon(self) -> None:
        now = self.clock()
        if self.abandon_armed_at is not None and now - self.abandon_armed_at <= 3:
            self.done = True
            self.save(force=True)
            self.exit({"outcome": "q", **self.state()})
            return
        self.abandon_armed_at = now
        self.notify("Press Ctrl+Q again within 3 seconds to abandon. Nothing will be logged.", severity="warning")
