"""Phase 0: the 2-day C++ -> Python fast-track (spec section 3). Days 1-2 serve these modules
instead of the problem bank. Content lives in data/phase0_day1.json and data/phase0_day2.json."""
import contextlib
import io
from datetime import datetime

from sqlmodel import Session, select

from google_prep import db
from google_prep.core import scheduler
from google_prep.models import LessonProgress

DAYS = (1, 2)


def load_day(day: int) -> dict:
    return db.load_json(f"phase0_day{day}.json")


def load_days() -> list[dict]:
    return [load_day(d) for d in DAYS]


def exercise(day: dict, exercise_id: str) -> dict:
    found = next((e for e in day["exercises"] if e["id"] == exercise_id), None)
    if found is None:
        raise KeyError(exercise_id)
    return found


def run_snippet(code: str) -> str:
    """The real interpreter output of an exercise (shown after answering, never before)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(code, "<exercise>", "exec"), {"__name__": "__fasttrack__"})  # noqa: S102 - bundled content
    return buffer.getvalue().strip()


def answer(session: Session, exercise_id: str, choice: int, now: datetime) -> dict:
    day = load_day(int(exercise_id.split("-")[0][1:]))
    ex = exercise(day, exercise_id)
    correct = choice == ex["answer"]
    session.merge(LessonProgress(item_id=exercise_id, kind="exercise", correct=correct, updated_at=now))
    session.commit()
    return {"correct": correct, "answer": ex["answer"], "explanation": ex["explanation"],
            "output": run_snippet(ex["code"])}


def progress(session: Session, day: dict) -> dict:
    rows = {r.item_id: r for r in session.exec(select(LessonProgress)).all()}
    ids = [e["id"] for e in day["exercises"]]
    answered = [rows[i] for i in ids if i in rows]
    return {"answered": len(answered), "correct": sum(1 for r in answered if r.correct), "total": len(ids),
            "done": f"day-{day['day']}" in rows}


def is_done(session: Session, day: int) -> bool:
    return session.get(LessonProgress, f"day-{day}") is not None


def complete(session: Session, day: int, now: datetime, minutes: int = 0) -> None:
    """Finishing a fast-track day counts as a full study day for the streak."""
    session.merge(LessonProgress(item_id=f"day-{day}", kind="day", correct=True, updated_at=now))
    session.commit()
    scheduler.mark_done(session, now.date(), "morning", minutes)
    scheduler.mark_done(session, now.date(), "evening")


def pending_day(session: Session, today_day: int) -> int | None:
    """The fast-track day to serve now: during Phase 0 the earliest unfinished day up to today."""
    if today_day > DAYS[-1]:
        return None
    for d in DAYS:
        if d <= today_day and not is_done(session, d):
            return d
    return None


def unfinished(session: Session) -> list[int]:
    return [d for d in DAYS if not is_done(session, d)]


# ---- terminal runner (the GUI has its own view) ------------------------------------

def inline_markup(text: str) -> str:
    """**bold** and `code` from the content files -> rich markup (everything else escaped)."""
    import re

    from rich.markup import escape

    out, pos = [], 0
    for m in re.finditer(r"\*\*(.+?)\*\*|`(.+?)`", text):
        out.append(escape(text[pos:m.start()]))
        out.append(f"[header]{escape(m.group(1))}[/header]" if m.group(1) else f"[accent]{escape(m.group(2))}[/accent]")
        pos = m.end()
    out.append(escape(text[pos:]))
    return "".join(out)


def run_terminal(session: Session, day_no: int, now: datetime) -> dict:
    """One section per screen, then the exercises; Enter moves on, 1-3 answers, q stops."""
    import time

    from rich.console import Group
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text

    from google_prep.ui import keys
    from google_prep.ui.formatters import banner, card, crumbs, keys as key_bar, nav, show

    lesson = load_day(day_no)
    started = time.monotonic()
    base = scheduler.breadcrumb(session, now.date(), f"Fast-Track Day {day_no}")

    def code_pair(cpp: str, python: str) -> Table:
        grid = Table.grid(expand=True, padding=(0, 2))
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(Text("C++", style="muted"), Text("Python", style="accent"))
        grid.add_row(Syntax(cpp, "cpp", theme="ansi_dark", word_wrap=True),
                     Syntax(python, "python", theme="ansi_dark", word_wrap=True))
        return grid

    sections = lesson["sections"]
    for i, s in enumerate(sections, 1):
        show(Text(""), crumbs(f"{base} > {s['title']}"), nav("Fast-Track", with_keys=False),
             banner(f"PYTHON FAST-TRACK · DAY {day_no}", f"Section {i}/{len(sections)} · {s['title']}",
                    lesson["title"]),
             Text.from_markup("\n".join(f"  • {inline_markup(p)}" for p in s["points"])), Text(""),
             code_pair(s["cpp"], s["python"]),
             card("GOTCHA", [inline_markup(s["gotcha"])], style="#FBBC05"),
             key_bar(("Enter", "Next"), ("q", "Stop")))
        if keys.choose({"enter", "space", "q"}) == "q":
            return {"done": False}

    right = 0
    exercises = lesson["exercises"]
    for i, e in enumerate(exercises, 1):
        show(Text(""), crumbs(f"{base} > Exercise {i}/{len(exercises)}"), nav("Fast-Track", with_keys=False),
             banner(f"PYTHON FAST-TRACK · DAY {day_no}", f"Exercise {i}/{len(exercises)} · {e['topic']}",
                    f"{right} correct"),
             Text.from_markup(f"  [header]{inline_markup(e['prompt'])}[/header]"),
             Syntax(e["code"], "python", theme="ansi_dark", line_numbers=False, word_wrap=True),
             key_bar(*[(str(k + 1), opt) for k, opt in enumerate(e["options"])], ("q", "Stop")))
        pick = keys.choose({"1", "2", "3", "q"})
        if pick == "q":
            return {"done": False}
        result = answer(session, e["id"], int(pick) - 1, now)
        right += result["correct"]
        verdict = "[ok]✓ Correct[/ok]" if result["correct"] else \
            f"[danger]✗ Not quite[/danger] · answer: [header]{e['options'][result['answer']]}[/header]"
        show(Group(Text.from_markup(f"  {verdict}   [muted]real output:[/muted] {inline_markup(result['output'])}"),
                   Text.from_markup(f"  {inline_markup(result['explanation'])}")),
             key_bar(("Enter", "Next")))
        keys.choose({"enter", "space", "q"})
    minutes = max(1, round((time.monotonic() - started) / 60))
    complete(session, day_no, now, minutes)
    show(card(f"DAY {day_no} COMPLETE", [f"{right}/{len(exercises)} correct · {minutes} min · counted for the streak",
                                         "Day 3: the Google problem bank starts." if day_no == 2 else
                                         "Tomorrow: Day 2, data structures."], style="#34A853"))
    return {"done": True, "correct": right, "total": len(exercises)}
