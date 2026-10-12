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
LANGUAGES = ("standard", "simple")  # standard: technical English, explained; simple: plain English + Hindi hints


def load_day(day: int) -> dict:
    return db.load_json(f"phase0_day{day}.json")


def load_days() -> list[dict]:
    return [load_day(d) for d in DAYS]


def load_hints() -> dict[str, str]:
    """English term -> short Hinglish hint, shown in simple mode the first time a term appears on a page."""
    try:
        return db.load_json("hindi_hints.json").get("hints", {})
    except (OSError, ValueError):
        return {}


def text(item: dict, key: str, language: str):
    """A lesson field in the learner's language mode (simple falls back to standard)."""
    if language == "simple" and item.get(f"{key}_simple"):
        return item[f"{key}_simple"]
    return item.get(key, "")


def language(session: Session) -> str:
    mode = scheduler.get_setting(session, "language", "standard")
    return mode if mode in LANGUAGES else "standard"


def mark_unclear(session: Session, section_id: str, unclear: bool, now: datetime) -> None:
    """'I did not get this step': kept as a list to review again."""
    item_id = f"{section_id}-unclear"
    row = session.get(LessonProgress, item_id)
    if unclear and row is None:
        session.add(LessonProgress(item_id=item_id, kind="unclear", correct=None, updated_at=now))
    elif not unclear and row is not None:
        session.delete(row)
    session.commit()


def unclear_steps(session: Session) -> list[str]:
    rows = session.exec(select(LessonProgress).where(LessonProgress.kind == "unclear")).all()
    return sorted(r.item_id.removesuffix("-unclear") for r in rows)


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
            "explanation_simple": ex.get("explanation_simple") or ex["explanation"],
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
    """One step per screen: example, what happened, remember, C++ hint, watch out, quick check.
    Enter moves on, 1-3 answers, q stops. Uses the learner's language mode (standard or simple)."""
    import time

    from rich.syntax import Syntax
    from rich.text import Text

    from google_prep.ui import keys
    from google_prep.ui.formatters import banner, card, crumbs, keys as key_bar, nav, show

    lesson = load_day(day_no)
    mode = language(session)
    glossary = {g["key"]: g for g in lesson.get("glossary", [])}
    ex_for = {e.get("section"): e for e in lesson["exercises"]}
    started = time.monotonic()
    base = scheduler.breadcrumb(session, now.date(), f"Fast-Track Day {day_no}")
    core = [s for s in lesson["sections"] if not s.get("optional")]
    right = 0
    for i, s in enumerate(lesson["sections"], 1):
        label = "Optional extra" if s.get("optional") else f"Step {i}/{len(core)}"
        words = [f"[accent]{glossary[w]['term']}[/accent] = {inline_markup(text(glossary[w], 'meaning', mode))}"
                 for w in s.get("new_words", []) if w in glossary]
        parts = [Text(""), crumbs(f"{base} > {text(s, 'title', mode)}"), nav("Fast-Track", with_keys=False),
                 banner(f"PYTHON FAST-TRACK · DAY {day_no}", f"{label} · {text(s, 'title', mode)}",
                        text(lesson, "title", mode))]
        if words:
            parts.append(Text.from_markup("  New words: " + "   ".join(words)))
        parts += [Text.from_markup("\n  [header]Try it[/header]"),
                  Syntax(s["python"], "python", theme="ansi_dark", word_wrap=True),
                  Text.from_markup("  [header]What happened[/header]\n"
                                   + "\n".join(f"  • {inline_markup(p)}" for p in text(s, "points", mode))),
                  card("REMEMBER", [inline_markup(text(s, "summary", mode))])]
        if s.get("bridge"):
            parts.append(Text.from_markup("  [muted]Coming from C++:[/muted]\n" + "\n".join(
                f"    {b['cpp']}  →  [accent]{b['python']}[/accent]" for b in s["bridge"])))
        parts += [card("WATCH OUT", [inline_markup(text(s, "gotcha", mode))], style="#FBBC05"),
                  key_bar(("Enter", "Quick check" if s["id"] in ex_for else "Next"), ("a", "Explain another way"),
                          ("q", "Stop"))]
        show(*parts)
        key = keys.choose({"enter", "space", "a", "q"})
        if key == "q":
            return {"done": False}
        if key == "a":
            show(card("ANOTHER WAY TO SEE IT", [inline_markup(text(s, "analogy", mode))], style="#81C995"),
                 key_bar(("Enter", "Next")))
            if keys.choose({"enter", "space", "q"}) == "q":
                return {"done": False}
        e = ex_for.get(s["id"])
        if e is None:
            continue
        show(Text.from_markup(f"\n  [header]Quick check · {inline_markup(e['prompt'])}[/header]"),
             Syntax(e["code"], "python", theme="ansi_dark", line_numbers=False, word_wrap=True),
             key_bar(*[(str(k + 1), opt) for k, opt in enumerate(e["options"])], ("q", "Stop")))
        pick = keys.choose({"1", "2", "3", "q"})
        if pick == "q":
            return {"done": False}
        result = answer(session, e["id"], int(pick) - 1, now)
        right += result["correct"]
        verdict = "[ok]✓ Correct[/ok]" if result["correct"] else \
            f"[danger]✗ Not quite[/danger] · answer: [header]{e['options'][result['answer']]}[/header]"
        explanation = result["explanation_simple"] if mode == "simple" else result["explanation"]
        show(Text.from_markup(f"  {verdict}   [muted]Python prints:[/muted] {inline_markup(result['output'])}"),
             Text.from_markup(f"  {inline_markup(explanation)}"), key_bar(("Enter", "Next step")))
        if keys.choose({"enter", "space", "q"}) == "q":
            return {"done": False}
    exercises = lesson["exercises"]
    minutes = max(1, round((time.monotonic() - started) / 60))
    complete(session, day_no, now, minutes)
    show(card(f"DAY {day_no} COMPLETE", [f"{right}/{len(exercises)} correct · {minutes} min · counted for the streak",
                                         "Day 3: the Google problem bank starts." if day_no == 2 else
                                         "Tomorrow: Day 2, data structures."], style="#34A853"))
    return {"done": True, "correct": right, "total": len(exercises)}
