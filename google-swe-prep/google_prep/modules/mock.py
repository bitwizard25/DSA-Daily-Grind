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
from google_prep.core import clock, readiness, scheduler, session_store
from google_prep.core.timer import FOLLOW_UP_AT, Countdown, mmss
from google_prep.models import MockSession, Problem
from google_prep.ui import keys
from google_prep.ui.formatters import banner, card, console, crumbs, keys as key_bar, show
from google_prep.ui.theme import DIFFICULTY_STYLE, RED, YELLOW

TOPICS = {"all": None, **{c: [c] for c in scheduler.CATEGORIES}}
CHECKLIST = [
    "Clarified constraints and input size",
    "Stated brute force vs. optimal approach with complexity",
    "Wrote clean, modular code with descriptive names",
    "Traced a test case line-by-line manually",
]
GENERIC_FOLLOW_UPS = [
    "What if the input is a stream of 10^9 events and does not fit in memory?",
    "What if memory is constrained to 64MB?",
    "What if the data is sharded across 100 machines?",
    "What if this function is called 10^5 times per second? What would you precompute?",
    "What if the input is modified concurrently by other threads?",
]
RUBRIC = [
    ("algorithms", "Algorithms", "chose the optimal approach and justified its complexity"),
    ("coding", "Coding", "clean, correct, idiomatic code with no compiler help"),
    ("communication", "Communication", "explained thinking out loud the whole time"),
    ("problem_solving", "Problem Solving", "handled edge cases, hints and the follow-up"),
]
RUBRIC_SCALE = [("1", "Weak"), ("2", "Mixed"), ("3", "Solid"), ("4", "Strong")]
AUTOSAVE_SECONDS = 5


def follow_ups(problem: Problem) -> list[str]:
    return [problem.follow_up_prompt] + GENERIC_FOLLOW_UPS


def hire_signal(avg: float) -> str:
    if avg >= 3.5:
        return "Strong Hire"
    if avg >= 2.75:
        return "Lean Hire"
    if avg >= 2.0:
        return "Lean No Hire"
    return "No Hire"


def scratchpad(problem: Problem) -> Path:
    folder = db.home_dir() / "scratchpads"
    folder.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", problem.title.lower()).strip("_")
    path = folder / f"{problem.id}_{slug}.py"
    if not path.exists():
        path.write_text(
            f"# {problem.title} ({problem.difficulty})\n#\n# {problem.statement}\n#\n"
            "# Rules: no compiler, no autocomplete, no running it.\n"
            "# 1) Clarify constraints  2) Brute force + complexity  3) Optimal + complexity\n"
            "# 4) Code  5) Dry-run one test case line by line in comments below.\n\n",
            encoding="utf-8",
        )
    return path


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
    title = "TIMED SOLVE" if practice else "MOCK INTERVIEW: Google Coding Round"
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
    parts = [crumbs(path), head, phase, statement, Text(""), env, check]
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
            "elapsed": timer.elapsed(), "ticks": ticks, "shown": shown, "submitted": submitted}


def run(session, problem: Problem, minutes: int = 45, practice: bool = False, editor: bool = True,
        now: datetime | None = None, clock_fn=time.monotonic, resume: dict | None = None) -> dict | None:
    """Timed round. State is auto-saved every few seconds and on every key, so a crash or a
    closed window resumes with the same time left, checklist and follow-up."""
    resume = resume or {}
    when = now or clock.now()
    timer = Countdown(minutes * 60, clock_fn, elapsed=resume.get("elapsed", 0.0))
    pad = scratchpad(problem)
    editor_msg = open_editor(pad) if editor else "editor launch skipped"
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
    if not resume.get("submitted"):
        autosave(force=True)
        if keys.interactive():
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
        result = evaluate(session, problem, minutes, timer, ticks, fups[shown] if shown >= 0 else "", practice, when)
    session_store.clear()
    return result


def evaluate(session, problem: Problem, minutes: int, timer: Countdown, ticks: list[bool],
             follow_up: str, practice: bool, now: datetime) -> dict:
    used = max(1, int(timer.elapsed()))
    solved = _ask("Did you reach a correct, complete solution?", [("y", "Yes"), ("n", "No")]) == "y"
    dry = _ask("Dry-run: did your hand trace of a test case match the expected output?",
               [("y", "Yes, matched"), ("n", "No / skipped")]) == "y"
    rubric = {}
    if not practice:
        for field, name, hint in RUBRIC:
            rubric[field] = int(_ask(f"{name}: {hint}", RUBRIC_SCALE))
    target = minutes * 60 * readiness.CODING_SHARE
    mistake = None
    if not solved or not dry or used > target:
        options = [(str(i), label) for i, label in enumerate(scheduler.MISTAKES.values(), 1)]
        pick = _ask("What hurt most? (tunes tomorrow's warmup)", options)
        mistake = list(scheduler.MISTAKES)[int(pick) - 1]
    if not solved:
        grade = 2
    elif used > target:
        grade = 3
    elif practice:
        grade = 5 if dry else 4
    else:
        grade = 5 if sum(rubric.values()) / 4 >= 3.5 else 4
    used_min = max(1, round(used / 60))
    scheduler.log_attempt(session, problem.id, used_min, solved, dry, grade, now,
                          source="solve" if practice else "mock", mistake=mistake)
    result = {"problem": problem.id, "solved": solved, "dry_run_ok": dry, "used_seconds": used,
              "grade": grade, "rubric": rubric, "mistake": mistake}
    lines = [f"Time used: {mmss(used)} of {minutes:02d}:00"
             + (" [danger](over the 35-min coding target)[/danger]" if used > target and solved else ""),
             f"Solved: {'[ok]yes[/ok]' if solved else '[danger]no[/danger]'}   "
             f"Dry-run matched: {'[ok]yes[/ok]' if dry else '[danger]no[/danger]'}   "
             f"Checklist: {sum(ticks)}/{len(ticks)}"]
    if not practice:
        session.add(MockSession(problem_id=problem.id, day=now.date().isoformat(), limit_minutes=minutes,
                                used_seconds=used, solved=solved, follow_up=follow_up,
                                checklist_done=sum(ticks), **rubric))
        session.commit()
        avg = sum(rubric.values()) / 4
        result["signal"] = hire_signal(avg)
        lines.append(f"Rubric average: {avg:.2f}/4  ->  [accent]{result['signal']}[/accent]")
    show(Text(""), card("RESULT", lines))
    return result
