"""60-second pattern recall drill (spec Feature 2) with SM-2 grading.

Three grades only: Missed, Solved with Effort, Instant Reflex. Each grade is saved the
moment it is given, and the session position is auto-saved so a crash can resume."""
import time
from datetime import datetime

from rich.console import Group
from rich.live import Live
from rich.text import Text

from google_prep.core import clock, scheduler, session_store
from google_prep.core.timer import Countdown, mmss
from google_prep.models import Problem, RecallDrill
from google_prep.ui import keys
from google_prep.ui.formatters import banner, console, crumbs, keys as key_bar, show

DRILL_SECONDS = 60
INSTANT_SECONDS = 20  # "Instant Reflex" must actually be instant; slower reveals count as effort
GRADES = [("1", "Missed"), ("2", "Solved with Effort"), ("3", "Instant Reflex")]
SM2_QUALITY = {"1": 1, "2": 4, "3": 5}  # choice -> SM-2 quality (below 3 resets the card)
TIMED_OUT_QUALITY = 3  # correct, but only after the 60 seconds ran out


def quality(choice: str, seconds: float, timed_out: bool) -> tuple[int, str]:
    """SM-2 quality for a choice, downgraded when the measured time contradicts it."""
    if choice == "1":
        return SM2_QUALITY["1"], ""
    if timed_out:
        return TIMED_OUT_QUALITY, "over 60s: counted as effort"
    if choice == "3" and seconds > INSTANT_SECONDS:
        return SM2_QUALITY["2"], f"{seconds:.0f}s: counted as effort"
    return SM2_QUALITY[choice], ""


def drill_screen(drill: RecallDrill, remaining: float, n: int, total: int, path: str) -> Group:
    bug = drill.type == "bug_hunt"
    clock_style = "danger" if remaining <= 10 else "warn"
    title = "BUG HUNT" if bug else "PATTERN RECALL"
    sub = "find the bug, state the fix" if bug else "title hidden"
    head = banner(title, f"#{drill.id} ({sub})   [muted]{n}/{total}[/muted]",
                  f"[{clock_style}]⏱ {mmss(remaining)}[/{clock_style}]")
    body = Text(drill.prompt if bug else f'"{drill.prompt}"', style="body")
    questions = (["1. Which line is wrong?", "2. Which input breaks it?", "3. What is the fix?"] if bug else
                 ["1. Pattern / data structure?", "2. Time and space complexity?", "3. Edge case to clarify?"])
    think = Text("─" * 70 + "\n" + "\n".join(questions) + "\n" + "─" * 70, style="muted")
    return Group(crumbs(path), head, Text(""), body, Text(""), think, Text(""),
                 key_bar(("Space", "Reveal answer"), ("q", "Stop")))


def reveal_lines(session, drill: RecallDrill) -> list[str]:
    label = "BUG & FIX" if drill.type == "bug_hunt" else "PATTERN"
    lines = [f"  ✅ [ok]{label}:[/ok] {drill.pattern}",
             f"  ⏱  [warn]COMPLEXITY:[/warn] {drill.complexity}",
             f"  ⚠️  [danger]EDGE CASE:[/danger] {drill.edge_case}"]
    if drill.problem_id:
        p = session.get(Problem, drill.problem_id)
        lines.append(f"  [muted]{p.id} · {p.title} (LeetCode {p.leetcode_id})[/muted]")
    return lines


def _wait_for_reveal(drill: RecallDrill, n: int, total: int, path: str) -> tuple[str, bool, float]:
    """Returns (key, timed_out, seconds_to_reveal). key is 'reveal' or 'q'."""
    timer = Countdown(DRILL_SECONDS)
    if not keys.interactive():
        show(drill_screen(drill, DRILL_SECONDS, n, total, path))
        key = keys.read_key()
        return ("q" if key == "q" else "reveal"), False, timer.elapsed()
    with keys.held(), Live(drill_screen(drill, timer.remaining(), n, total, path), console=console,
                           refresh_per_second=4, transient=False) as live:
        while not timer.expired():
            key = keys.read_key(timeout=0.25)
            if key == "q":
                return "q", False, timer.elapsed()
            if key in ("space", "enter"):
                return "reveal", False, timer.elapsed()
            live.update(drill_screen(drill, timer.remaining(), n, total, path))
    return "reveal", True, timer.elapsed()


def run_session(session, drills: list[RecallDrill], now: datetime | None = None, kind: str = "warmup",
                resume: dict | None = None) -> dict:
    """Run drills; every grade is written immediately. Returns counts per grade."""
    resume = resume or {}
    ids = [d.id for d in drills]
    start_index = resume.get("index", 0)
    carried = resume.get("seconds", 0.0)
    counts = dict(resume.get("counts", {"1": 0, "2": 0, "3": 0}))
    started = time.monotonic()
    when = now or clock.now()
    quit_early = False
    with keys.held():
        quit_early = _drill_loop(session, drills, ids, start_index, counts, carried, started, when, kind)
    session_store.clear()
    done = sum(counts.values())
    minutes = max(1, round((carried + time.monotonic() - started) / 60)) if done else 0
    return {"done": done, "counts": counts, "minutes": minutes, "quit": quit_early}


def _drill_loop(session, drills, ids, start_index, counts, carried, started, when, kind) -> bool:
    """Returns True if the user quit early."""
    for n in range(start_index, len(drills)):
        drill = drills[n]
        session_store.save({"kind": "recall", "mode": kind, "drill_ids": ids, "index": n, "counts": counts,
                            "seconds": carried + time.monotonic() - started, "day": when.date().isoformat()})
        path = scheduler.breadcrumb(session, when.date(), "Recall", scheduler.CATEGORIES[drill.category])
        key, timed_out, seconds = _wait_for_reveal(drill, n + 1, len(drills), path)
        if key == "q":
            return True
        show(Text.from_markup("\n".join(reveal_lines(session, drill))), Text(""), key_bar(*GRADES))
        choice = keys.choose({g for g, _ in GRADES} | {"q"})
        if choice == "q":
            return True
        q, note = quality(choice, seconds, timed_out)
        prog = scheduler.grade_drill(session, drill.id, q, when)
        counts[choice] += 1
        show(Text(f"  Saved · next review {prog.next_review_date:%a %d %b}" + (f" · {note}" if note else "") + "\n",
                  style="muted"))
    return False
