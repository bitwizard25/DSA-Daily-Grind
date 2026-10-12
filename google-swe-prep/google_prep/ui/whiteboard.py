"""Full-screen 45-minute Google whiteboard and the 4-pillar scorecard.

Layout (spec section 5, feature 3): timer bar on top, problem on the left, a plain-text
editor on the right (no highlighting, no autocomplete, nothing runs), a hand-filled trace
table next to the code, a follow-up banner injected at 35:00, and the rubric at 40:00.
The screens are reused by `google-prep mock` (MockApp) and by `google-prep ui`."""
import time
from pathlib import Path
from typing import Callable

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Static, TabbedContent, TabPane, TextArea

from google_prep.core import rubric
from google_prep.core.timer import FOLLOW_UP_AT, Countdown, mmss
from google_prep.ui.formatters import crumb_text, nav
from google_prep.ui.theme import BLUE, BODY, MUTED, RED, SURFACE, YELLOW
from google_prep.ui.theme import TEXT_BLUE, TEXT_GREEN, TEXT_RED, TEXT_YELLOW


def nav_text(active: str) -> Text:
    """Section bar without keys: you cannot jump away in the middle of a round."""
    return nav(active, with_keys=False)

RUBRIC_AT = 40 / 45  # mocks hand over to the scorecard at 40:00

CSS = f"""
Screen {{ background: {SURFACE}; color: {BODY}; }}
.crumbs {{ padding: 0 1; height: 1; }}
#timerbar {{ border: round {BLUE}; padding: 0 1; height: 3; }}
#banner {{ border: heavy {YELLOW}; padding: 0 1; height: auto; color: {TEXT_YELLOW}; }}
#left {{ width: 2fr; padding: 0 1; }}
#right {{ width: 3fr; }}
#problem {{ border: solid {BLUE}; padding: 0 1; height: auto; }}
.panel {{ border: solid {MUTED}; padding: 0 1; height: auto; margin-top: 1; }}
TextArea {{ border: solid {MUTED}; height: 1fr; }}
TextArea:focus {{ border: solid {BLUE}; }}
#card {{ border: round {BLUE}; padding: 1 2; margin: 1 2; height: auto; }}
"""


class WhiteboardScreen(Screen[dict]):
    BINDINGS = [
        Binding("ctrl+s", "submit", "Submit", priority=True),
        Binding("f1", "tick(0)", "✓1", priority=True),
        Binding("f2", "tick(1)", "✓2", priority=True),
        Binding("f3", "tick(2)", "✓3", priority=True),
        Binding("f4", "tick(3)", "✓4", priority=True),
        Binding("f5", "tab('code')", "Code", priority=True),
        Binding("f6", "tab('trace')", "Trace", priority=True),
        Binding("ctrl+f", "follow_up", "Follow-up", priority=True),
        Binding("ctrl+t", "pause", "Pause", priority=True),
        Binding("ctrl+q", "abandon", "Abandon x2"),
    ]

    def __init__(self, problem, minutes: int, practice: bool, path: str, timer: Countdown, ticks: list[bool],
                 shown: int, follow_ups: list[str], checklist: list[str], pad: Path, trace: Path,
                 autosave: Callable[[dict], None], clock: Callable[[], float] = time.monotonic):
        super().__init__()
        self.problem, self.minutes, self.practice, self.path = problem, minutes, practice, path
        self.timer, self.ticks, self.shown = timer, list(ticks), shown
        self.follow_ups, self.checklist = follow_ups, checklist
        self.pad, self.trace, self.autosave_cb, self.clock = pad, trace, autosave, clock
        self.abandon_armed_at: float | None = None
        self.done = False

    def compose(self) -> ComposeResult:
        yield Static(crumb_text(self.path), classes="crumbs")
        yield Static(nav_text("Mock"), classes="crumbs")
        yield Static(id="timerbar")
        yield Static(id="banner")
        with Horizontal():
            with Vertical(id="left"):
                p = self.problem
                yield Static(Text.assemble((f"{p.title}\n", "bold"), (f"{p.difficulty}\n\n", MUTED), p.statement),
                             id="problem")
                yield Static(id="timeline", classes="panel")
                yield Static(id="checklist", classes="panel")
            with TabbedContent(initial="code", id="right"):
                with TabPane("Code (F5)", id="code"):
                    yield TextArea(self.pad.read_text(encoding="utf-8"), id="editor", language=None,
                                   soft_wrap=True, tab_behavior="indent", show_line_numbers=False,
                                   highlight_cursor_line=False)
                with TabPane("Trace table (F6)", id="trace"):
                    yield TextArea(self.trace.read_text(encoding="utf-8"), id="tracer", language=None,
                                   soft_wrap=False, tab_behavior="indent", show_line_numbers=False,
                                   highlight_cursor_line=False)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#editor", TextArea).focus()
        self.set_interval(0.5, self.tick)
        self.tick()

    # ---- state ----------------------------------------------------------------
    def code(self) -> str:
        return self.query_one("#editor", TextArea).text

    def trace_text(self) -> str:
        return self.query_one("#tracer", TextArea).text

    def state(self) -> dict:
        return {"elapsed": self.timer.elapsed(), "ticks": self.ticks, "shown": self.shown}

    def save(self, force: bool = False) -> None:
        """Every timer tick saves the session state (Krug: never punish an interruption)."""
        self.autosave_cb(self.state())
        if force:
            self.save_text()

    def save_text(self) -> None:
        self.pad.write_text(self.code(), encoding="utf-8")
        self.trace.write_text(self.trace_text(), encoding="utf-8")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Every keystroke in the code or the trace table is written to disk."""
        self.save_text()

    def finish(self, outcome: str) -> None:
        if self.done:
            return
        self.done = True
        self.save(force=True)
        self.dismiss({"outcome": outcome, **self.state(), "trace_text": self.trace_text()})

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
            self.app.bell()
        if (not self.practice and frac >= RUBRIC_AT) or (self.practice and self.timer.expired()):
            return self.finish("time")
        rem = self.timer.remaining()
        colour = TEXT_RED if rem <= 300 else TEXT_YELLOW
        title = "TIMED SOLVE" if self.practice else "MOCK INTERVIEW · WHITEBOARD SIMULATOR"
        self.query_one("#timerbar", Static).update(Text.assemble(
            (title, f"bold {TEXT_BLUE}"), "   ", (f"⏱ {mmss(rem)} left", f"bold {colour}"),
            ("  (paused)" if self.timer.paused else "", MUTED), "   ", (self.phase_label(), MUTED)))

        banner = self.query_one("#banner", Static)
        banner.display = self.shown >= 0
        if self.shown >= 0:
            banner.update(Text.assemble(("⚠ GOOGLE FOLLOW-UP (minute 35)  ", f"bold {TEXT_YELLOW}"),
                                        self.follow_ups[self.shown % len(self.follow_ups)],
                                        ("   Ctrl+F: another constraint", MUTED)))

        b, elapsed = self.boundaries(), self.timer.elapsed()
        labels = ["Scoping & constraints", "Code: plain text only", "Google follow-up", "4-pillar scorecard"]
        lines = Text("TIMELINE\n", style="bold")
        for i, label in enumerate(labels):
            active, past = b[i] <= elapsed < b[i + 1], elapsed >= b[i + 1]
            style = f"bold {TEXT_BLUE}" if active else (TEXT_GREEN if past else MUTED)
            lines.append(f"{'▶' if active else '✓' if past else ' '} {mmss(b[i])}–{mmss(b[i + 1])} {label}\n",
                         style=style)
        self.query_one("#timeline", Static).update(lines)

        check = Text("CHECKLIST (F1-F4)\n", style="bold")
        for i, (done, item) in enumerate(zip(self.ticks, self.checklist)):
            check.append(f"{'[✓]' if done else '[ ]'} {i + 1}. {item}\n", style=TEXT_GREEN if done else MUTED)
        self.query_one("#checklist", Static).update(check)
        self.save()

    def phase_label(self) -> str:
        b, elapsed = self.boundaries(), self.timer.elapsed()
        names = ["Scoping", "Coding", "Follow-up", "Scorecard"]
        return next((names[i] for i in range(4) if elapsed < b[i + 1]), names[-1])

    # ---- actions ------------------------------------------------------------------
    def action_submit(self) -> None:
        self.finish("s")

    def action_tab(self, tab: str) -> None:
        self.query_one(TabbedContent).active = tab
        self.query_one("#editor" if tab == "code" else "#tracer", TextArea).focus()

    def action_follow_up(self) -> None:
        self.shown = (self.shown + 1) % len(self.follow_ups)
        self.tick()

    def action_pause(self) -> None:
        self.timer.toggle_pause()
        for area in self.query(TextArea):
            area.read_only = self.timer.paused
        self.tick()

    def action_tick(self, index: int) -> None:
        self.ticks[index] = not self.ticks[index]
        self.tick()

    def action_abandon(self) -> None:
        now = self.clock()
        if self.abandon_armed_at is not None and now - self.abandon_armed_at <= 3:
            self.done = True
            self.save(force=True)
            self.dismiss({"outcome": "q", **self.state()})
            return
        self.abandon_armed_at = now
        self.notify("Press Ctrl+Q again within 3 seconds to abandon. Nothing will be logged.", severity="warning")


class RubricScreen(Screen[dict]):
    """40:00-45:00 scorecard as a short run of mindless choices: the 3-tier outcome, then one
    yes/no per pillar check (your checklist ticks are pre-noted), then the mistake if needed."""

    BINDINGS = [Binding(k, f"key('{k}')", show=False) for k in "12345yn"] + [Binding("b", "back", "Back a step")]

    def __init__(self, problem, path: str, practice: bool, used: int, ticks: list[bool], follow_up: str,
                 trace_rows: int, needs_mistake: Callable[[bool, bool], bool], mistakes: dict[str, str],
                 over_time: bool = False):
        super().__init__()
        self.problem, self.path, self.practice, self.used = problem, path, practice, used
        self.follow_up, self.trace_rows, self.over_time = follow_up, trace_rows, over_time
        self.ticked = rubric.from_ticks(ticks)
        self.checks = dict(self.ticked)
        self.needs_mistake, self.mistakes = needs_mistake, mistakes
        self.outcome: str | None = None
        self.questions = [(f, p, q) for f, p, q in rubric.CHECKS if not practice or f in rubric.PRACTICE_CHECKS]
        self.step = 0  # 0 = outcome, 1..n = checks, n+1 = mistake

    def compose(self) -> ComposeResult:
        yield Static(crumb_text(f"{self.path} > {'Self-grade' if self.practice else 'Scorecard'}"), classes="crumbs")
        yield Static(nav_text("Mock"), classes="crumbs")
        yield Static(id="card")
        yield Footer()

    def on_mount(self) -> None:
        self.render_card()

    def render_card(self) -> None:
        title = "SELF-GRADE" if self.practice else "SCORECARD · GOOGLE 4 PILLARS · 40:00–45:00"
        out = Text.assemble((title + "\n", f"bold {TEXT_BLUE}"),
                            (f"{self.problem.title} · coding time {mmss(self.used)}\n\n", MUTED))
        n = len(self.questions)
        if self.step == 0:
            out.append("How did it go?\n\n", style="bold")
            again = {"1": "back in today's re-solve queue", "2": "re-solve in 3 days", "3": "re-solve next week"}
            for key, label in rubric.OUTCOMES.items():
                out.append(f"[{key}] {label}", style="bold")
                out.append(f"  → {again[key]}\n", style=MUTED)
            if self.over_time:
                out.append("\nOver the 35-min coding target: Instant Reflex will count as Effort.", style=TEXT_YELLOW)
        elif self.step <= n:
            field, pillar, question = self.questions[self.step - 1]
            out.append(f"Check {self.step}/{n}" + ("" if self.practice else f" · {rubric.PILLARS[pillar]}") + "\n\n",
                       style=MUTED)
            out.append(question + "?\n", style="bold")
            if field == "dry_run_passed":
                out.append(f"Trace table rows you filled: {self.trace_rows}\n", style=MUTED)
            if field == "follow_up_handled" and self.follow_up:
                out.append(f'Follow-up: "{self.follow_up}"\n', style=MUTED)
            if self.ticked.get(field):
                out.append("You ticked this during the round.\n", style=TEXT_GREEN)
            out.append("\n[y] Yes    [n] No", style="bold")
        else:
            out.append("What hurt most? (tunes tomorrow's warmup)\n\n", style="bold")
            for i, label in enumerate(self.mistakes.values(), 1):
                out.append(f"[{i}] {label}\n")
        if not self.practice and self.step > 0:
            points = rubric.score(self.checks)
            out.append(f"\n\nScore so far {points:.0f}/100 · {rubric.hire_signal(points, self.outcome != '1')}",
                       style=MUTED)
        out.append("\n[b] Back a step", style=MUTED)
        self.query_one("#card", Static).update(out)

    def action_key(self, key: str) -> None:
        n = len(self.questions)
        if self.step == 0 and key in rubric.OUTCOMES:
            self.outcome = key
            self.step = 1
        elif 1 <= self.step <= n and key in "yn":
            self.checks[self.questions[self.step - 1][0]] = key == "y"
            self.step += 1
            if self.step > n and not self.needs_mistake(self.outcome != "1", self.checks["dry_run_passed"]):
                return self.finish(None)
        elif self.step > n and key in "12345":
            return self.finish(list(self.mistakes)[int(key) - 1])
        self.render_card()

    def action_back(self) -> None:
        self.step = max(0, self.step - 1)
        self.render_card()

    def finish(self, mistake: str | None) -> None:
        self.dismiss({"outcome": self.outcome, "solved": self.outcome != "1", "checks": self.checks,
                      "mistake": mistake})


class MockApp(App[dict]):
    """Hosts the whiteboard then the scorecard for `google-prep mock` / `solve`."""

    ENABLE_COMMAND_PALETTE = False
    CSS = CSS
    # Textual's own Ctrl+Q quits instantly; during a round it must be the press-twice abandon.
    BINDINGS = [Binding("ctrl+q", "guarded_quit", show=False, priority=True)]

    def action_guarded_quit(self) -> None:
        if isinstance(self.screen, WhiteboardScreen):
            self.screen.action_abandon()

    def __init__(self, board: WhiteboardScreen | None, make_rubric: Callable[[dict], RubricScreen],
                 start: dict | None = None):
        """`board` None with `start` given resumes straight into the scorecard."""
        super().__init__()
        self.board, self.make_rubric, self.start = board, make_rubric, start

    def on_mount(self) -> None:
        if self.board is None:
            self.after_board(self.start)
        else:
            self.push_screen(self.board, self.after_board)

    def after_board(self, result: dict | None) -> None:
        if not result or result["outcome"] == "q":
            self.exit(result)
            return
        screen = self.make_rubric(result)
        self.push_screen(screen, lambda graded: self.exit({**result, "graded": graded}))
