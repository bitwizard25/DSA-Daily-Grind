"""45-minute Google whiteboard simulator (spec Feature 3). In practice mode the same
screen runs a 60-minute evening deep-work solve without the rubric."""
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Group
from rich.live import Live
from rich.text import Text

from google_prep import db
from google_prep.core import clock, readiness, rubric, scheduler, session_store
from google_prep.core.timer import FOLLOW_UP_AT, Countdown, mmss
from google_prep.models import MockSession, Problem
from google_prep.ui import keys
from google_prep.ui.formatters import banner, card, console, crumbs, keys as key_bar, nav, show
from google_prep.ui.theme import DIFFICULTY_STYLE, RED, YELLOW

TOPICS = {"all": None, **{c: [c] for c in scheduler.CATEGORIES}}
CHECKLIST = [  # short labels for the rubric's first four checks (same order)
    "Clarified constraints & edge cases",
    "Stated time/space complexity",
    "Idiomatic, modular code",
    "Traced a test case by hand",
]
GENERIC_FOLLOW_UPS = [
    "What if the input is a stream of 10^9 events and does not fit in memory?",
    "What if memory is constrained to 64MB?",
    "What if the data is sharded across 100 machines?",
    "What if this function is called 10^5 times per second? What would you precompute?",
    "What if the input is modified concurrently by other threads?",
]
AUTOSAVE_SECONDS = 5
# "Struggle first": the app's own hints stay locked this long, because the struggle is the practice.
HINT_AFTER_MINUTES = {"solve": 25, "mock": 15}  # a mock interviewer nudges sooner: the code is due by 35:00
STUCK_TIPS = [
    "Say the problem back in your own words. What exactly goes in, and what comes out?",
    "Solve a tiny example by hand (3 to 5 items). Write down every step you take.",
    "Write the slow, obvious answer first. Then ask: what work am I repeating?",
    "Which tool fits? dict or set, two pointers, sliding window, binary search, stack, heap, BFS or DFS.",
    "Still stuck? Write what you tried in the code pad. Explaining it often shows the gap.",
]


def hint_unlock_seconds(practice: bool) -> int:
    return HINT_AFTER_MINUTES["solve" if practice else "mock"] * 60


def hints(problem: Problem, key_insight: str | None = None) -> list[str]:
    """Two levels: the pattern's name, then the key idea. Never code."""
    return [f"Pattern: {problem.key_pattern}.", f"Key idea: {key_insight or problem.key_insight}"]


def follow_ups(problem: Problem) -> list[str]:
    return [problem.google_follow_up] + GENERIC_FOLLOW_UPS


TRACE_TEMPLATE = (
    "Dry-run trace: pick a non-trivial input and step through YOUR code by hand. No running it.\n"
    "Input:\n\n"
    "step | line | variables (name=value, ...)              | note\n"
    "-----+------+------------------------------------------+----------------\n"
    "1    |      |                                          |\n"
    "2    |      |                                          |\n"
    "3    |      |                                          |\n")


def _pad_name(problem: Problem) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", problem.title.lower()).strip("_")
    return f"{problem.id}_{slug}"


def pad_language(problem: Problem, language: str | None = None) -> str:
    from google_prep.modules import fasttrack

    if language is not None:
        if language not in fasttrack.CODING_LANGUAGES:
            raise ValueError("Choose Python or C++.")
        return language
    active = session_store.load() or {}
    if active.get("kind") == "mock" and active.get("problem_id") == problem.id:
        return active.get("coding_language", "python")
    with db.get_session() as s:
        return fasttrack.coding_language(s)


def scratchpad(problem: Problem, language: str | None = None) -> Path:
    """Plain .txt on purpose: a .py file can open in an IDE with autocomplete and linting.
    Saved under ~/.google_prep/sessions/ with the rest of the auto-saved session."""
    folder = db.home_dir() / "sessions"
    folder.mkdir(exist_ok=True)
    language = pad_language(problem, language)
    suffix = ".cpp" if language == "cpp" else ""
    path = folder / f"{_pad_name(problem)}{suffix}.txt"
    legacy = db.home_dir() / "scratchpads" / path.name
    if not path.exists() and legacy.exists():  # v1.0 kept scratchpads in their own folder
        legacy.replace(path)
    if not path.exists():
        template = (
            f"# {problem.id} {problem.title} ({problem.difficulty})\n# {problem.statement}\n#\n"
            "# 1) Clarifying questions and constraints:\n\n"
            "# 2) Brute force + complexity:\n\n"
            "# 3) Optimal approach + complexity:\n\n"
            "# 4) Code:\n\n\n"
            "# 5) Dry-run one test case line by line:\n\n")
        if language == "cpp":
            template = re.sub(r"^#", "//", template, flags=re.MULTILINE)
        path.write_text(template, encoding="utf-8")
    return path


def trace_pad(problem: Problem, language: str | None = None) -> Path:
    pad = scratchpad(problem, language)
    path = pad.with_name(f"{pad.stem}.trace.txt")
    if not path.exists():
        path.write_text(TRACE_TEMPLATE, encoding="utf-8")
    return path


def fresh_scratchpads(problem: Problem, language: str) -> None:
    """Back up both previous pads before clearing them for an independent retry."""
    from uuid import uuid4
    pad, trace = scratchpad(problem, language), trace_pad(problem, language)
    backup = db.home_dir() / 'retry_backups' / uuid4().hex
    backup.mkdir(parents=True)
    for source in (pad, trace):
        shutil.copy2(source, backup/source.name)
    prefix = '//' if language == 'cpp' else '#'
    for path, text in ((pad, f'{prefix} Fresh attempt: explain your approach, then write your solution.\n'),
                       (trace, TRACE_TEMPLATE)):
        temporary = path.with_name(path.name + '.retry.tmp')
        temporary.write_text(text, encoding='utf-8')
        temporary.replace(path)


def trace_rows(text: str) -> int:
    """Filled rows of the hand trace table: evidence for Verification & Rigor."""
    rows = 0
    for line in text.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) >= 3 and cells[0].isdigit() and any(cells[1:]):
            rows += 1
    return rows


def open_editor(path: Path) -> str:
    """Open the scratchpad in a plain editor without blocking the timer."""
    if os.environ.get("GOOGLE_PREP_NO_EDITOR"):
        return "editor launch disabled"
    custom = os.environ.get("GOOGLE_PREP_EDITOR")
    try:
        if custom:
            subprocess.Popen(shlex.split(custom) + [str(path)])
        elif os.name == "nt":
            subprocess.Popen(["notepad.exe", str(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-e", str(path)])
        elif shutil.which("xdg-open") and os.environ.get("DISPLAY"):
            subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            return "Open it in any plain editor (set GOOGLE_PREP_EDITOR to auto-launch)"
    except OSError as exc:
        return f"Could not launch editor ({exc}); open the file manually"
    return "External Editor Launched"


def mock_screen(problem: Problem, timer: Countdown, ticks: list[bool], follow_up: str | None,
                pad: Path, editor_msg: str, practice: bool, path: str = "") -> Group:
    rem = timer.remaining()
    style = "danger" if rem <= 300 else "warn"
    clock = f"[{style}]⏱ {mmss(rem)} Remaining[/{style}]" + ("  [muted](paused)[/muted]" if timer.paused else "")
    title = "TIMED SOLVE" if practice else "MOCK INTERVIEW · WHITEBOARD SIMULATOR"
    dstyle = DIFFICULTY_STYLE[problem.difficulty]
    head = banner(title, f"Problem: [header]{problem.title}[/header] ([{dstyle}]{problem.difficulty}[/{dstyle}])",
                  f"{clock}   [muted]Focus: {scheduler.CATEGORIES[problem.category]}[/muted]")
    phase = Text(f"  [{timer.mock_phase()}]", style="accent")
    statement = Text(f"  {problem.statement}", style="body")
    env = Text.from_markup(
        f"  Editing: {pad.name} ({editor_msg})\n  [muted]{pad}[/muted]\n"
        "  * Autocomplete: [danger]DISABLED[/danger]\n"
        "  * Compiler / Test Runner: [danger]DISABLED[/danger] (Manual mental dry-run required)")
    check = card("LIVE INTERVIEW CHECKLIST",
                 [f"[{'ok' if t else 'muted'}]{'[✓]' if t else '[ ]'}[/] {i + 1}. {item}"
                  for i, (t, item) in enumerate(zip(ticks, CHECKLIST))])
    parts = [crumbs(path), nav("Mock", with_keys=False), head, phase, statement, Text(""), env, check]
    if follow_up:
        parts.append(card("GOOGLE FOLLOW-UP CONSTRAINT", [follow_up], style=YELLOW))
    if timer.expired():
        parts.append(card("TIME EXPIRED", ["Stop coding. Grade now."], style=RED))
    parts.append(key_bar(("1-4", "Tick checklist"), ("p", "Pause Timer"),
                         ("f", "Trigger Google Follow-up Constraint"), ("s", "Submit & Grade"), ("q", "Abandon")))
    return Group(*parts)


def _ask(question: str, options: list[tuple[str, str]]) -> str:
    show(Text(""), Text(f"  {question}", style="header"), key_bar(*options))
    return keys.choose({k for k, _ in options})


def _state(problem: Problem, minutes: int, practice: bool, timer: Countdown, ticks: list[bool],
           shown: int, submitted: bool = False) -> dict:
    return {"kind": "mock", "problem_id": problem.id, "minutes": minutes, "practice": practice,
            "coding_language": pad_language(problem),
            "elapsed": timer.elapsed(), "ticks": ticks, "shown": shown, "submitted": submitted}


def run(session, problem: Problem, minutes: int = 45, practice: bool = False, editor: bool = True,
        now: datetime | None = None, clock_fn=time.monotonic, resume: dict | None = None,
        external: bool = False) -> dict | None:
    """Timed round. In a terminal it opens the full-screen whiteboard with a built-in plain
    editor; `external` keeps the timer screen and opens Notepad/TextEdit instead. State is
    auto-saved every few seconds, so a crash or closed window resumes where it stopped."""
    resume = resume or {}
    when = now or clock.now()
    timer = Countdown(minutes * 60, clock_fn, elapsed=resume.get("elapsed", 0.0))
    pad = scratchpad(problem)
    inline = keys.interactive() and not external
    editor_msg = ("built-in editor" if inline else
                  open_editor(pad) if editor else "editor launch skipped")
    ticks = list(resume.get("ticks", [False] * len(CHECKLIST)))
    fups = follow_ups(problem)
    shown = resume.get("shown", -1)
    path = scheduler.breadcrumb(session, when.date(), "Mock" if not practice else "Solve",
                                scheduler.CATEGORIES[problem.category], problem.title)
    last_save = [0.0]

    def autosave(force: bool = False) -> None:
        if force or clock_fn() - last_save[0] >= AUTOSAVE_SECONDS:
            session_store.save(_state(problem, minutes, practice, timer, ticks, shown))
            last_save[0] = clock_fn()

    def screen():
        return mock_screen(problem, timer, ticks, fups[shown] if shown >= 0 else None, pad, editor_msg,
                           practice, path)

    def handle(key: str | None) -> str | None:
        nonlocal shown
        if key and len(key) == 1 and key in "1234":
            ticks[int(key) - 1] = not ticks[int(key) - 1]
        elif key == "p":
            timer.toggle_pause()
        elif key == "f":
            shown = (shown + 1) % len(fups)
        elif key in ("s", "q"):
            return key
        if key:
            autosave(force=True)
        return None

    outcome = "s"
    if inline and resume.get("submitted"):
        return _run_inline(session, problem, minutes, practice, timer, ticks, shown, fups, path, when, graded_only=True)
    if not resume.get("submitted"):
        autosave(force=True)
        if inline:
            return _run_inline(session, problem, minutes, practice, timer, ticks, shown, fups, path, when)
        elif keys.interactive():
            with keys.held(), Live(screen(), console=console, refresh_per_second=2) as live:
                while True:
                    if shown < 0 and timer.fraction() >= FOLLOW_UP_AT:
                        shown = 0
                    if timer.expired():
                        break
                    result = handle(keys.read_key(timeout=0.5))
                    if result:
                        outcome = result
                        break
                    autosave()
                    live.update(screen())
                live.update(screen())
        else:
            show(screen())
            while not timer.expired():
                result = handle(keys.read_key())
                if result:
                    outcome = result
                    break
                show(screen())
    if outcome == "q":
        session_store.clear()
        show(Text("  Abandoned. Not logged.", style="muted"))
        return None
    if timer.paused:
        timer.toggle_pause()
    session_store.save(_state(problem, minutes, practice, timer, ticks, shown, submitted=True))
    with keys.held():
        result = evaluate(session, problem, minutes, timer, ticks, fups[shown] if shown >= 0 else "", practice, when,
                          path)
    session_store.clear()
    return result


def board_screen(problem: Problem, minutes: int, practice: bool, timer: Countdown, ticks: list[bool], shown: int,
                 path: str, clock_fn=time.monotonic):
    """The full-screen whiteboard for this problem (used by `mock`, `solve` and `ui`)."""
    from google_prep.ui.whiteboard import WhiteboardScreen  # textual loads only when needed

    def save_state(st: dict) -> None:
        session_store.save(_state(problem, minutes, practice, timer, st["ticks"], st["shown"]))

    return WhiteboardScreen(problem, minutes, practice, path, timer, ticks, shown, follow_ups(problem), CHECKLIST,
                            scratchpad(problem), trace_pad(problem), save_state, clock_fn)


def rubric_screen(problem: Problem, minutes: int, practice: bool, timer: Countdown, board: dict, path: str):
    """The 40:00 scorecard. Saves a 'submitted' checkpoint so a crash resumes straight into it."""
    from google_prep.ui.whiteboard import RubricScreen

    fups = follow_ups(problem)
    session_store.save({**_state(problem, minutes, practice, timer, board["ticks"], board["shown"], submitted=True),
                        "elapsed": board["elapsed"]})
    used = max(1, int(board["elapsed"]))
    follow_up = fups[board["shown"] % len(fups)] if board["shown"] >= 0 else ""
    trace = board.get("trace_text") or trace_pad(problem).read_text(encoding="utf-8")
    return RubricScreen(problem, path, practice, used, board["ticks"], follow_up, trace_rows(trace),
                        lambda solved, dry: needs_mistake(solved, dry, used, minutes),
                        {key: scheduler.MISTAKES[key] for key in scheduler.MISTAKE_CHOICES},
                        over_time=used > minutes * 60 * readiness.CODING_SHARE)


def save_graded(session, problem: Problem, minutes: int, practice: bool, board: dict, graded: dict,
                now: datetime) -> dict:
    fups = follow_ups(problem)
    follow_up = fups[board["shown"] % len(fups)] if board["shown"] >= 0 else ""
    result = record(session, problem, minutes, max(1, int(board["elapsed"])), graded["outcome"], graded["checks"],
                    follow_up, practice, graded["mistake"], now, graded.get("help", "none"))
    session_store.clear()
    return result


def _run_inline(session, problem: Problem, minutes: int, practice: bool, timer: Countdown, ticks: list[bool],
                shown: int, fups: list[str], path: str, when: datetime, graded_only: bool = False) -> dict | None:
    from google_prep.ui.whiteboard import MockApp

    resumed = {"outcome": "s", "elapsed": timer.elapsed(), "ticks": ticks, "shown": shown}
    board = None if graded_only else board_screen(problem, minutes, practice, timer, ticks, shown, path)
    res = MockApp(board, lambda b: rubric_screen(problem, minutes, practice, timer, b, path),
                  start=resumed if graded_only else None).run()
    if res is None or (res["outcome"] != "q" and not res.get("graded")):
        show(Text("  Stopped. Session auto-saved: run google-prep to resume.", style="muted"))
        return None
    if res["outcome"] == "q":
        session_store.clear()
        show(Text("  Abandoned. Not logged.", style="muted"))
        return None
    result = save_graded(session, problem, minutes, practice, res, res["graded"], when)
    show(crumbs(f"{path} > Result"), card("RESULT · LOGGED", result_lines(result, minutes, practice)))
    return result


def needs_mistake(solved: bool, dry: bool, used: int, minutes: int) -> bool:
    return not solved or not dry or used > minutes * 60 * readiness.CODING_SHARE


def record(session, problem: Problem, minutes: int, used: int, outcome: str, checks: dict[str, bool],
           follow_up: str, practice: bool, mistake: str | None, now: datetime, help: str = "none") -> dict:  # noqa: A002
    """Log the attempt (and the mock with its pillar score) and reschedule the problem.
    `outcome` is the 3-tier choice: 1 Missed it, 2 Solved with Effort, 3 Instant Reflex.
    `help` is the honest answer to "did you use help?": none, hint or answer."""
    help = help if help in rubric.HELP else "none"  # noqa: A001
    target = minutes * 60 * readiness.CODING_SHARE
    outcome = rubric.outcome_for(outcome, used > target, help)
    solved = outcome != "1"
    points = rubric.score(checks)
    scheduler.log_attempt(session, problem.id, max(1, round(used / 60)), solved, checks["dry_run_passed"],
                          rubric.OUTCOME_GRADE[outcome], now, source="solve" if practice else "mock", mistake=mistake,
                          min_days=rubric.OUTCOME_DAYS[outcome], help=help)
    result = {"problem": problem.id, "solved": solved, "outcome": outcome, "dry_run_ok": checks["dry_run_passed"],
              "used_seconds": used, "mistake": mistake, "checks": checks, "score": points,
              "signal": rubric.hire_signal(points, solved), "over_time": used > target, "help": help,
              "again": rubric.again_for(outcome, help)}
    if not practice:
        session.add(MockSession(problem_id=problem.id, date=now, duration_seconds=used, limit_minutes=minutes,
                                solved=solved, follow_up=follow_up, score=points, **checks))
        session.commit()
    return result


def result_lines(result: dict, minutes: int, practice: bool) -> list[str]:
    lines = [f"Coding time: {mmss(result['used_seconds'])} of {minutes:02d}:00"
             + (" [danger](over the 35-min coding target)[/danger]" if result["over_time"] and result["solved"] else ""),
             f"{rubric.OUTCOMES[result['outcome']]} · {result['again']}   "
             f"Dry-run matched: {'[ok]yes[/ok]' if result['dry_run_ok'] else '[danger]no[/danger]'}"]
    if not practice:
        pillars = rubric.pillar_scores(result["checks"])
        lines.append("   ".join(f"{rubric.PILLARS[p]} {round(v * 25):>2}/25" for p, v in pillars.items()))
        lines.append(f"Score {result['score']:.0f}/100  ->  [accent]{result['signal']}[/accent]")
    return lines


def evaluate(session, problem: Problem, minutes: int, timer: Countdown, ticks: list[bool],
             follow_up: str, practice: bool, now: datetime, path: str = "") -> dict:
    """Terminal scorecard (piped input or --external). The full-screen UI uses RubricScreen."""
    used = max(1, int(timer.elapsed()))
    stage = "Self-grade" if practice else "Rubric"
    show(crumbs(f"{path} > {stage}" if path else stage), nav("Mock", with_keys=False),
         banner("SELF-GRADE" if practice else "SCORECARD · GOOGLE 4 PILLARS",
                f"{problem.title} · coding time used [warn]{mmss(used)}[/warn]",
                "" if practice else "[muted]40:00–45:00[/muted]"))
    help = rubric.HELP_KEYS[_ask("Did you use any help? Be honest: a seen answer is not your solve yet.",  # noqa: A001
                                 [(k, rubric.HELP[v]) for k, v in rubric.HELP_KEYS.items()])]
    outcome = "1" if help == "answer" else _ask("How did it go?", list(rubric.OUTCOMES.items()))
    checks = rubric.from_ticks(ticks)
    asked = rubric.PRACTICE_CHECKS if practice else [f for f, _, _ in rubric.CHECKS]
    for field, pillar, question in rubric.CHECKS:
        if field in asked:
            hint = " (ticked during the round)" if checks[field] else ""
            label = "" if practice else f"[{rubric.PILLARS[pillar]}] "
            checks[field] = _ask(f"{label}{question}?{hint}", [("y", "Yes"), ("n", "No")]) == "y"
    mistake = None
    if needs_mistake(outcome != "1", checks["dry_run_passed"], used, minutes):
        options = [(str(i), scheduler.MISTAKES[key]) for i, key in enumerate(scheduler.MISTAKE_CHOICES, 1)]
        mistake = scheduler.MISTAKE_CHOICES[int(_ask("What hurt most? (tunes tomorrow's warmup)", options)) - 1]
    result = record(session, problem, minutes, used, outcome, checks, follow_up, practice, mistake, now, help)
    show(Text(""), card("RESULT", result_lines(result, minutes, practice)))
    return result
