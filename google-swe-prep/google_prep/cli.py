"""google-prep: Typer router. Running it with no command starts the next best action."""
import sys
from datetime import date, datetime
from typing import Optional

import typer
from rich.markdown import Markdown
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from google_prep import __version__, db
from google_prep.core import scheduler, session_store, trending
from google_prep.core.clock import now
from google_prep.db import get_session
from google_prep.models import Problem, RecallDrill, StarStory
from google_prep.modules import fasttrack as fasttrack_mod
from google_prep.modules import mock as mock_mod
from google_prep.modules import recall as recall_mod
from google_prep.modules import remind as remind_mod
from google_prep.modules import roadmap as roadmap_mod
from google_prep.modules import star as star_mod
from google_prep.modules import today as today_mod
from google_prep.ui import keys
from google_prep.ui.formatters import banner, card, console, crumbs, keys as key_bar, nav, show
from google_prep.ui.theme import DIFFICULTY_STYLE

app = typer.Typer(help="Google SWE II interview OS. Run with no command to start the next best action.",
                  add_completion=False, invoke_without_command=True, rich_markup_mode=None,
                  pretty_exceptions_show_locals=False)
star_app = typer.Typer(help="Googlyness STAR story vault.", no_args_is_help=True, rich_markup_mode=None)
app.add_typer(star_app, name="star")


def fail(message: str, code: int = 1):
    console.print(f"[danger]{message}[/danger]")
    raise typer.Exit(code)


def wait_for(actions: dict[str, str]) -> str | None:
    """Wait for a screen's key. Skipped when output is piped but input is a terminal."""
    if not keys.interactive() and sys.stdin.isatty():
        return None
    key = keys.choose(set(actions) | {"q", "esc"})
    action = actions.get(key) or (actions.get("b") if key == "esc" else None)
    return None if action == "quit" else action


# ---- screens / actions ---------------------------------------------------------

def screen_today(mode: str = "normal") -> str | None:
    with get_session() as session:
        renderable, actions = today_mod.build(session, now(), mode)
    show(renderable)
    actions.update({"t": "today", "s": "star_list"})
    return wait_for(actions)


def run_drills(kind: str, count: int = today_mod.WARMUP_DRILLS, resume: dict | None = None) -> None:
    t = now()
    with get_session() as session:
        if resume:
            drills = [session.get(RecallDrill, i) for i in resume["drill_ids"]]
            drills = [d for d in drills if d is not None]
        elif kind == "micro":
            drills = today_mod.plan(session, t, "micro")["micro"]
        else:
            drills = scheduler.pick_warmup(session, t, count)
        if not drills:
            show(Text("No drills due.", style="muted"))
            return
        result = recall_mod.run_session(session, drills, t, kind, resume)
        if result["done"]:
            scheduler.mark_done(session, t.date(), "morning", result["minutes"],
                                mode="micro" if kind == "micro" else None)
            c = result["counts"]
            streak = scheduler.streak(session, t.date())
            show(card("DRILLS LOGGED", [f"Missed {c['1']} · Effort {c['2']} · Instant {c['3']}"
                                        f"   │   {result['minutes']} min   │   Streak: {streak} "
                                        f"day{'s' if streak != 1 else ''}"]))


def run_problem(problem: Problem, minutes: int, practice: bool, editor: bool = True,
                resume: dict | None = None, external: bool = False) -> None:
    with get_session() as session:
        mock_mod.run(session, problem, minutes=minutes, practice=practice, editor=editor, now=now(), resume=resume,
                     external=external)


def describe(state: dict) -> str:
    if state.get("kind") == "recall":
        left = len(state["drill_ids"]) - state["index"]
        return f"Session restored: Pattern recall ({left} drill{'s' if left != 1 else ''} left)."
    with get_session() as session:
        p = session.get(Problem, state["problem_id"])
    title = p.title if p else state["problem_id"]
    if state.get("submitted"):
        return f"Session restored: {title} (scorecard not finished)."
    left = max(0, state["minutes"] * 60 - state["elapsed"])
    return f"Session restored: {title} ({round(left / 60)}m remaining)."


def screen_resume() -> str | None:
    state = session_store.load()
    if not state:
        return "today"
    show(card("SESSION RESTORED", [escape(describe(state)), escape("Press [Enter] to resume.")]),
         key_bar(("Enter", "Resume"), ("x", "Discard it"), ("q", "Quit")))
    key = keys.choose({"enter", "x", "q", "esc"}) if (keys.interactive() or not sys.stdin.isatty()) else "q"
    if key == "x":
        session_store.clear()
        return "today"
    if key != "enter":
        return None
    if state["kind"] == "recall":
        run_drills(state.get("mode", "warmup"), resume=state)
    else:
        with get_session() as session:
            problem = session.get(Problem, state["problem_id"])
        if problem is None:
            session_store.clear()
            return "today"
        run_problem(problem, state["minutes"], state["practice"], editor=False, resume=state)
    return None


def next_action() -> str:
    """Zero-argument launch: the single next best thing to do right now."""
    if session_store.load():
        return "resume"
    t = now()
    with get_session() as session:
        sched = scheduler.schedule_for(session, t.date())
        if fasttrack_mod.pending_day(session, scheduler.day_number(session, t.date())) is not None:
            return "fasttrack"  # Days 1-2: the C++ -> Python fast-track comes first
    if sched.mode == "micro":
        return "micro_start" if not (sched.morning_completed or sched.evening_completed) else "today"
    if t.hour < 12:
        return "warmup" if not sched.morning_completed else "today"
    return "deep" if not sched.evening_completed else "today"


def action_deep() -> None:
    t = now()
    with get_session() as session:
        phase = scheduler.phase_for_day(scheduler.day_number(session, t.date()))
        target = scheduler.pick_target_problem(session, phase, t)
    if target is None:
        fail("No problem available.")
    run_problem(target, 45 if phase == 3 else 60, practice=phase < 3)


def action_review() -> None:
    with get_session() as session:
        due = scheduler.due_problems(session, now())
    if due:
        run_problem(due[0], 45, practice=True)


def screen_roadmap() -> str | None:
    with get_session() as session:
        renderable, actions = roadmap_mod.build(session, now().date())
    show(renderable)
    return wait_for(actions)


def screen_star_list() -> str | None:
    with get_session() as session:
        show(crumbs(scheduler.breadcrumb(session, now().date(), "STAR")), nav("STAR"),
             banner("STAR STORIES", "Ambiguity · Ownership · Disagreement · Humility", "2 per dimension"),
             star_mod.list_view(session))
    show(key_bar(("a", "Add"), ("v", "Practice"), ("b", "Back"), ("q", "Quit")))
    return wait_for({"a": "star_add", "v": "star_review", "b": "today", "t": "today", "f": "fasttrack", "d": "warmup",
                     "w": "mock",
                     "r": "roadmap", "m": "micro_start"})


def action_star_add() -> None:
    dims = list(star_mod.DIMENSIONS)
    show(Text("Which Google dimension does this story show?", style="header"),
         key_bar(*[(str(i), star_mod.DIMENSIONS[d]) for i, d in enumerate(dims, 1)]))
    dim = dims[int(keys.choose({str(i) for i in range(1, len(dims) + 1)})) - 1]
    star_add(dimension=dim, title=None, situation=None, task=None, action=None, result=None)


def action_star_review() -> None:
    star_review(dimension=None)


def action_fasttrack(day: int | None = None) -> str | None:
    t = now()
    with get_session() as session:
        day_no = scheduler.day_number(session, t.date())
        day = day or fasttrack_mod.pending_day(session, day_no) or (fasttrack_mod.unfinished(session) or [1])[0]
        fasttrack_mod.run_terminal(session, day, t)
    return None


ACTIONS = {
    "resume": screen_resume,
    "fasttrack": action_fasttrack,
    "today": lambda: screen_today(),
    "micro": lambda: screen_today("micro"),
    "review": lambda: screen_today("review"),
    "warmup": lambda: run_drills("warmup"),
    "micro_start": lambda: run_drills("micro"),
    "deep": action_deep,
    "review_start": action_review,
    "roadmap": screen_roadmap,
    "mock": lambda: mock(topic="all", minutes=45, problem=None, no_editor=False, external=False),
    "star_list": screen_star_list,
    "star_add": action_star_add,
    "star_review": action_star_review,
}


def navigate(action: str | None) -> None:
    while action and action != "quit":
        action = ACTIONS[action]()


# ---- commands -------------------------------------------------------------------

@app.callback()
def root(ctx: typer.Context,
         version: bool = typer.Option(False, "--version", help="Show the version and exit.")):
    if version:
        console.print(f"google-prep {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        navigate(next_action())


@app.command()
def today(
    mode: Optional[str] = typer.Option(None, "--mode", help="normal | micro | review"),
    worked: Optional[float] = typer.Option(None, "--worked", min=0,
                                           help=f"Hours worked today; {scheduler.MICRO_HOURS}+ switches to micro-mode."),
):
    """Daily command center: morning warmup, evening deep work, or micro-mode."""
    if mode not in (None, "normal", "micro", "review"):
        fail("--mode must be normal, micro or review", 2)
    with get_session() as session:
        sched = scheduler.schedule_for(session, now().date())
        if scheduler.needs_micro(worked):
            sched.mode = "micro"
            session.add(sched)
            session.commit()
            console.print(f"[warn]{worked:g}h workday: micro-mode on for today.[/warn]")
        mode = mode or ("micro" if sched.mode == "micro" else "normal")
    navigate({"micro": "micro", "review": "review"}.get(mode, "today"))


@app.command()
def fasttrack(day: Optional[int] = typer.Option(None, "--day", min=1, max=2, help="1 = basics, 2 = data structures")):
    """Phase 0: the 2-day C++ -> Python fast-track (served automatically on Days 1-2)."""
    action_fasttrack(day)


@app.command()
def recall(count: int = typer.Option(5, "--count", "-n", min=1, max=50, help="Drills this session.")):
    """60-second pattern recall drills with SM-2 spaced repetition."""
    run_drills("warmup", count)


@app.command()
def mock(
    topic: str = typer.Option("all", "--topic", help="all | " + " | ".join(scheduler.CATEGORIES)),
    minutes: int = typer.Option(45, "--minutes", min=10, max=120),
    problem: Optional[str] = typer.Option(None, "--problem", "-p", help="Problem id, e.g. G-200."),
    no_editor: bool = typer.Option(False, "--no-editor", help="With --external: do not auto-open the file."),
    external: bool = typer.Option(False, "--external", help="Use Notepad/TextEdit instead of the built-in editor."),
):
    """45-minute Google whiteboard: scoping, plain-text coding, follow-up, rubric."""
    if topic not in mock_mod.TOPICS:
        fail(f"Unknown topic '{topic}'. Choose: all, {', '.join(scheduler.CATEGORIES)}", 2)
    t = now()
    with get_session() as session:
        if problem:
            target = session.get(Problem, problem.upper())
            if target is None:
                fail(f"No problem with id {problem}")
        else:
            phase = scheduler.phase_for_day(scheduler.day_number(session, t.date()))
            target = scheduler.pick_target_problem(session, phase, t, mock_mod.TOPICS[topic])
            if target is None:
                fail("No problem available for that topic.")
    run_problem(target, minutes, practice=False, editor=not no_editor, external=external)


@app.command()
def solve(
    problem: Optional[str] = typer.Argument(None, help="Problem id; default is today's target."),
    minutes: int = typer.Option(60, "--minutes", min=10, max=120),
    no_editor: bool = typer.Option(False, "--no-editor"),
    external: bool = typer.Option(False, "--external", help="Use Notepad/TextEdit instead of the built-in editor."),
):
    """Timed deep-work solve (no rubric): today's evening problem or any problem id."""
    t = now()
    with get_session() as session:
        if problem:
            target = session.get(Problem, problem.upper())
        else:
            phase = scheduler.phase_for_day(scheduler.day_number(session, t.date()))
            target = scheduler.pick_target_problem(session, phase, t)
    if target is None:
        fail("No such problem.")
    run_problem(target, minutes, practice=True, editor=not no_editor, external=external)


@app.command()
def resume():
    """Resume an auto-saved session after a crash or closed window."""
    if not session_store.load():
        console.print("[muted]No unfinished session.[/muted]")
        return
    navigate("resume")


@app.command()
def roadmap():
    """8-week North Star roadmap and readiness breakdown."""
    navigate("roadmap")


@app.command()
def problems(topic: str = typer.Option("all", "--topic")):
    """List the problem bank with status and Google recency."""
    if topic not in mock_mod.TOPICS:
        fail(f"Unknown topic '{topic}'", 2)
    from google_prep.models import UserProgress
    from sqlmodel import select
    with get_session() as session:
        progress = {p.problem_id: p for p in session.exec(select(UserProgress)).all()}
        overrides = scheduler.recency_overrides(session)
        rows = [p for p in session.exec(select(Problem).order_by(Problem.id)).all()
                if topic == "all" or p.category == topic]
        table = Table(header_style="accent")
        for col in ("ID", "Problem", "Diff", "Topic", "Recent at Google", "Status"):
            table.add_column(col)
        for p in rows:
            window, _ = scheduler.recency(p, overrides)
            status = progress[p.id].status if p.id in progress else "unseen"
            sstyle = {"mastered": "ok", "learning": "warn"}.get(status, "muted")
            table.add_row(p.id, p.title, f"[{DIFFICULTY_STYLE[p.difficulty]}]{p.difficulty}[/]",
                          scheduler.CATEGORIES[p.category], window or "-", f"[{sstyle}]{status}[/]")
    console.print(table)


@app.command()
def patterns(name: Optional[str] = typer.Argument(None, help="Pattern name or number to open its note.")):
    """Pattern cheat sheet (15 core patterns) and the reference vault notes."""
    items = db.load_json("patterns.json")
    if name is None:
        table = Table(header_style="accent")
        for col in ("#", "Pattern", "Use when", "Cost"):
            table.add_column(col)
        for i, p in enumerate(items, 1):
            table.add_row(str(i), p["name"], p["cue"], p["complexity"])
        console.print(table)
        console.print("[muted]Open one: google-prep patterns 2[/muted]")
        return
    match = next((p for i, p in enumerate(items, 1) if name == str(i) or name.lower() in p["name"].lower()), None)
    if match is None:
        fail(f"No pattern matching '{name}'")
    if match["vault_file"]:
        console.print(Markdown((db.resource_dir() / "patterns_vault" / match["vault_file"]).read_text(encoding="utf-8")))
    else:
        console.print(card(match["name"], [f"Use when: {match['cue']}", "", match["template"], "",
                                           f"Cost: {match['complexity']}"]))


@app.command()
def sync(company: str = typer.Option("Google", "--company"),
         source: str = typer.Option(trending.DEFAULT_SOURCE, "--source", help="Base URL of the dataset.")):
    """Download recent interview frequency from GitHub (needs internet; everything else is offline)."""
    with get_session() as session:
        try:
            counts = trending.sync(session, company, source)
        except Exception as exc:  # network, HTTP or parse errors: keep the old data
            fail(f"Sync failed: {exc}. Your existing data is unchanged.")
        console.print(f"[ok]Synced {company}:[/ok] " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    hot(limit=5, easy=False)


@app.command()
def hot(limit: int = typer.Option(15, "--limit", min=1, max=100),
        easy: bool = typer.Option(False, "--easy", help="Include Easy problems.")):
    """Most recently asked problems that are not in your bank yet (run sync first)."""
    with get_session() as session:
        rows = trending.hot(session, limit, easy)
    if not rows:
        console.print("[muted]No synced data yet. Run: google-prep sync[/muted]")
        return
    table = Table(title="Trending at the company, not in your bank", header_style="accent")
    for col in ("Window", "Freq", "Diff"):
        table.add_column(col, no_wrap=True)
    table.add_column("Problem")
    table.add_column("Slug", overflow="fold")
    for t in rows:
        table.add_row(t.window, f"{t.frequency:.0f}", f"[{DIFFICULTY_STYLE.get(t.difficulty, 'body')}]{t.difficulty}[/]",
                      t.title, t.slug)
    console.print(table)
    console.print("[muted]Import one: google-prep add <slug>. Frequency is noisy (LeetCode #1-#10 rank high "
                  "everywhere): prefer problems that fill a weak topic.[/muted]")


@app.command()
def add(problem: str = typer.Argument(..., help="LeetCode slug or URL from `hot`.")):
    """Import a trending problem into your bank."""
    with get_session() as session:
        phase = scheduler.phase_for_day(scheduler.day_number(session, now().date()))
        existing = session.exec(scheduler.select(Problem).where(Problem.slug == trending.slug_of(problem))).first()
        if existing:
            console.print(f"[muted]Already in your bank: {existing.id} {existing.title}[/muted]")
            return
        try:
            p = trending.add_problem(session, problem, phase)
        except KeyError as exc:
            fail(str(exc.args[0]))
    console.print(f"[ok]Added {p.id}: {p.title}[/ok] ({scheduler.CATEGORIES[p.category]})")


@app.command()
def remind(morning: str = typer.Option("07:30", "--morning", help="HH:MM, 24h."),
           evening: str = typer.Option("19:30", "--evening", help="HH:MM, 24h.")):
    """Print commands that open today's plan automatically at your anchor times."""
    for value in (morning, evening):
        try:
            datetime.strptime(value, "%H:%M")
        except ValueError:
            fail(f"Time must be HH:MM (24h), got '{value}'", 2)
    console.print("[header]Run once to schedule both anchors:[/header]")
    for line in remind_mod.commands(morning, evening):
        console.print(Text(line, style="accent"), soft_wrap=True)


@app.command()
def setup(
    name: Optional[str] = typer.Option(None, "--name"),
    target: Optional[str] = typer.Option(None, "--target", help='e.g. "Google Hyderabad SWE II"'),
    interview_date: Optional[str] = typer.Option(None, "--interview-date", help="YYYY-MM-DD"),
    start_date: Optional[str] = typer.Option(None, "--start-date", help="YYYY-MM-DD (Day 1 of the plan)"),
    evening_time: Optional[str] = typer.Option(None, "--evening-time", help='e.g. "7:30 PM"'),
):
    """Set your name, target role, start date and interview date."""
    with get_session() as session:
        for key, value in (("interview_date", interview_date), ("start_date", start_date)):
            if value:
                try:
                    date.fromisoformat(value)
                except ValueError:
                    fail(f"{key} must be YYYY-MM-DD", 2)
        for key, value in (("name", name), ("target", target), ("interview_date", interview_date),
                           ("start_date", start_date), ("evening_time", evening_time)):
            if value:
                scheduler.set_setting(session, key, value)
        start = scheduler.start_date(session, now().date())
        console.print(f"Name: {scheduler.get_setting(session, 'name', '-')}   "
                      f"Target: {scheduler.get_setting(session, 'target', 'Google SWE II')}")
        console.print(f"Plan start: {start}   Interview: {scheduler.interview_date(session)}   "
                      f"Today is Day {scheduler.day_number(session, now().date())}")


@app.command()
def ui(
    no_browser: bool = typer.Option(False, "--no-browser", help="Only start the local server and print its address"),
    port: int = typer.Option(0, "--port", help="Fixed port (default: any free port)"),
):
    """Desktop window: Today, Python fast-track, drills, whiteboard mock, roadmap, STAR, vault (same database)."""
    from google_prep.gui.launcher import main as gui_main

    raise typer.Exit(gui_main((["--no-browser"] if no_browser else []) + (["--port", str(port)] if port else [])))


@app.command()
def tui():
    """Full-screen terminal cockpit, for when you want no window (same database)."""
    from google_prep.ui.tui_app import PrepUI

    navigate(PrepUI(now=now).run())


@star_app.command("add")
def star_add(
    dimension: Optional[str] = typer.Option(None, "--dimension", "-d",
                                            help="Ambiguity | Ownership | Disagreement | Humility"),
    title: Optional[str] = typer.Option(None, "--title", "-t"),
    situation: Optional[str] = typer.Option(None, "--situation", "-s"),
    task: Optional[str] = typer.Option(None, "--task"),
    action: Optional[str] = typer.Option(None, "--action", "-a"),
    result: Optional[str] = typer.Option(None, "--result", "-r"),
):
    """Draft a STAR story (prompts for anything not passed as an option)."""
    dim = star_mod.resolve_dimension(dimension or typer.prompt("Dimension (Ambiguity/Ownership/Disagreement/Humility)"))
    if dim is None:
        fail(f"Unknown dimension. Choose: {', '.join(star_mod.DIMENSIONS)}", 2)
    fields = {
        "title": title or typer.prompt("Title (e.g. 'Hapminds: cache migration under deadline')"),
        "situation": situation or typer.prompt("Situation (context, 1-2 sentences)"),
        "task": task or typer.prompt("Task (your responsibility)"),
        "action": action or typer.prompt("Action (what YOU did, specifically)"),
        "result": result or typer.prompt("Result (numbers: latency, revenue, time saved)"),
    }
    with get_session() as session:
        story = star_mod.add(session, dim, **fields)
        console.print(f"[ok]Saved story #{story.id}[/ok] under {star_mod.DIMENSIONS[dim]}.")
        show(star_mod.list_view(session))


@star_app.command("list")
def star_list():
    """List stories and coverage per Google dimension."""
    with get_session() as session:
        show(star_mod.list_view(session))


@star_app.command("review")
def star_review(dimension: Optional[str] = typer.Option(None, "--dimension", "-d")):
    """Practise a Googlyness question; weakest dimension first."""
    with get_session() as session:
        dim = star_mod.resolve_dimension(dimension) if dimension else None
        if dimension and dim is None:
            fail(f"Unknown dimension. Choose: {', '.join(star_mod.DIMENSIONS)}", 2)
        dim, prompt = star_mod.review_prompt(session, dim)
        show(crumbs(scheduler.breadcrumb(session, now().date(), "STAR", dim)), nav("STAR"),
             star_mod.review_view(session, dim, prompt))
        if keys.read_key() == "q":
            return
        stories = star_mod.stories_for(session, dim)
        if not stories:
            console.print(f"[warn]No {dim} stories yet.[/warn] Add one: google-prep star add -d {dim}")
        for s in stories:
            show(star_mod.story_view(s))


@star_app.command("show")
def star_show(story_id: int):
    """Print one story in full."""
    with get_session() as session:
        s = session.get(StarStory, story_id)
        if s is None:
            fail(f"No story {story_id}")
        show(star_mod.story_view(s))


def _utf8_output() -> None:
    """Box characters and emoji crash cp1252 pipes on Windows (e.g. `google-prep roadmap > out.txt`)."""
    for stream in (sys.stdout, sys.stderr):
        if (getattr(stream, "encoding", "") or "").lower().replace("-", "") != "utf8":
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass


def main() -> None:
    _utf8_output()
    try:
        app()
    except SystemExit as exc:  # Typer turns Ctrl+C into a silent exit code 130
        if exc.code == 130:
            hint = " Session auto-saved: run google-prep to resume." if session_store.load() else ""
            console.print(f"\n[muted]Stopped.{hint}[/muted]")
        raise


if __name__ == "__main__":
    main()
