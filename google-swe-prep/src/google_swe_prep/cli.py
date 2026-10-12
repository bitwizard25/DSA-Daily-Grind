"""Typer CLI: today, recall, mock, roadmap, star (+ log)."""
from __future__ import annotations

import time
from typing import Optional

import click
import typer
from rich.console import Console
from rich.table import Table

from google_swe_prep import engine
from google_swe_prep.models import connect
from google_swe_prep.seed_data import COMPETENCIES, PHASES, WEEKS

app = typer.Typer(help="Google SWE II prep coach.", no_args_is_help=True, add_completion=False)
star_app = typer.Typer(help="Googlyness / behavioural STAR stories.", no_args_is_help=True)
app.add_typer(star_app, name="star")
console = Console()


def _db():
    conn = connect()
    engine.seed(conn)
    return conn


def _readiness_line(conn) -> str:
    r = engine.readiness(conn)
    return (f"Readiness {r['index']:.0f}%  |  Mastery {r['mastery']*100:.0f}%  "
            f"Mock speed {r['mock_speed']*100:.0f}%  Googlyness {r['googlyness']*100:.0f}%")


@app.command()
def today(
    micro: bool = typer.Option(False, "--micro", help="10-minute emergency mode for stressful days."),
):
    """Show today's plan: 20m morning warmup + 60-90m evening block (or 10m micro mode)."""
    conn = _db()
    week = engine.week_of(conn)
    phase = engine.phase_of_week(week)
    pname, _ = PHASES[phase]
    console.print(f"[bold]Week {week}/8 - Phase {phase}: {pname}[/bold]")
    console.print(_readiness_line(conn))

    if micro:
        console.print("\n[bold yellow]Emergency Micro-Mode (10 min)[/bold yellow]")
        drills = engine.due_drills(conn, limit=1)
        probs = engine.pick_problems(conn, phase, 1, easy_only=True)
        console.print(f"  1. (4 min) Recall: {drills[0].title}  -> run `recall --count 1`" if drills
                      else "  1. (4 min) No recall drills due.")
        if probs:
            p = probs[0]
            console.print(f"  2. (6 min) Sketch the approach only: [{p.id}] {p.title} ({p.difficulty}, {p.pattern})")
        return

    console.print("\n[bold]Morning Warmup (20 min)[/bold]")
    drills = engine.due_drills(conn, limit=3)
    if drills:
        for d in drills:
            console.print(f"  - Recall: {d.title} [{d.topic}]")
        console.print("  Run: google-swe-prep recall")
    else:
        console.print("  No drills due. Re-read one pattern note.")

    console.print("\n[bold]Evening Deep Solve (60-90 min)[/bold]")
    if phase == 3:
        console.print("  Run a timed mock: google-swe-prep mock")
    picks = engine.pick_problems(conn, phase, 2)
    if not picks:
        console.print("  Problem bank cleared. Do a mock on your weakest topic.")
    for p in picks:
        console.print(f"  - [{p.id}] {p.title} - {p.difficulty}, {p.topic}, Google freq {p.google_freq}/5")
        console.print(f"      pattern: {p.pattern}")
    console.print("\nAfter solving: google-swe-prep log <id> --minutes N [--no-solved]")


@app.command()
def log(
    problem_id: int = typer.Argument(..., help="Problem id from `today`."),
    minutes: int = typer.Option(..., "--minutes", "-m", min=1),
    solved: bool = typer.Option(True, "--solved/--no-solved"),
):
    """Log a practice attempt."""
    conn = _db()
    p = engine.get_problem(conn, problem_id)
    if p is None:
        console.print(f"[red]No problem with id {problem_id}[/red]")
        raise typer.Exit(1)
    engine.log_attempt(conn, problem_id, minutes, solved)
    console.print(f"Logged {p.title}: {minutes} min, {'solved' if solved else 'not solved'}.")
    console.print(_readiness_line(conn))


@app.command()
def recall(
    count: int = typer.Option(5, "--count", "-n", min=1, help="Max drills this session."),
):
    """Spaced-repetition recall drills (Leitner boxes)."""
    conn = _db()
    drills = engine.due_drills(conn, limit=count)
    if not drills:
        console.print("Nothing due. Come back tomorrow.")
        return
    right = 0
    for i, d in enumerate(drills, 1):
        console.print(f"\n[bold]({i}/{len(drills)}) {d.title}[/bold]  [{d.topic}]")
        console.print(d.prompt)
        typer.prompt("Answer in your head, press Enter to reveal", default="", show_default=False)
        console.print(f"[green]{d.answer}[/green]")
        ok = typer.confirm("Did you recall it correctly?")
        box = engine.grade_drill(conn, d.id, ok)
        right += ok
        console.print(f"  -> box {box}")
    console.print(f"\nScore: {right}/{len(drills)}")
    console.print(_readiness_line(conn))


@app.command()
def mock(
    limit: int = typer.Option(45, "--limit", "-l", min=5, help="Time limit in minutes."),
    problem_id: Optional[int] = typer.Option(None, "--problem", "-p", help="Specific problem id."),
):
    """Run a timed mock interview round and record speed + communication."""
    conn = _db()
    phase = engine.phase_of_week(engine.week_of(conn))
    p = engine.get_problem(conn, problem_id) if problem_id else engine.pick_mock_problem(conn, phase)
    if p is None:
        console.print("[red]No problem available.[/red]")
        raise typer.Exit(1)
    console.print(f"[bold]MOCK[/bold] [{p.id}] {p.title} ({p.difficulty}, {p.topic})")
    console.print(f"Limit: {limit} min. Talk out loud, state complexity before coding.")
    start = time.monotonic()
    typer.prompt("Press Enter when you are done (or give up)", default="", show_default=False)
    used = max(1, round((time.monotonic() - start) / 60))
    solved = typer.confirm("Did you reach a correct, complete solution?")
    comm = typer.prompt("Communication score 1-5", type=click.IntRange(1, 5))
    engine.record_mock(conn, p.id, limit, used, solved, comm)
    over = " (over time)" if used > limit else ""
    console.print(f"Recorded: {used}/{limit} min{over}, solved={solved}, communication={comm}")
    console.print(_readiness_line(conn))


@app.command()
def roadmap():
    """Show the 8-week plan and current position."""
    conn = _db()
    cur = engine.week_of(conn)
    t = Table(title="Google SWE II - 8 week roadmap")
    for c in ("", "Week", "Phase", "Focus"):
        t.add_column(c)
    for week, phase, focus in WEEKS:
        t.add_row(">>" if week == cur else "", str(week), f"{phase}: {PHASES[phase][0]}", focus)
    console.print(t)
    solved = conn.execute("SELECT COUNT(DISTINCT problem_id) c FROM attempts WHERE solved=1").fetchone()["c"]
    total = conn.execute("SELECT COUNT(*) c FROM problems").fetchone()["c"]
    console.print(f"Problems solved: {solved}/{total}")
    console.print(_readiness_line(conn))


@star_app.command("add")
def star_add(
    competency: str = typer.Option(..., "--competency", "-c", help=f"One of: {', '.join(COMPETENCIES)}"),
    title: str = typer.Option(..., "--title", "-t"),
    situation: str = typer.Option(..., "--situation", "-s"),
    task: str = typer.Option(..., "--task"),
    action: str = typer.Option(..., "--action", "-a"),
    result: str = typer.Option(..., "--result", "-r"),
):
    """Save a STAR story."""
    match = {c.lower(): c for c in COMPETENCIES}.get(competency.lower())
    if match is None:
        console.print(f"[red]Unknown competency. Choose from: {', '.join(COMPETENCIES)}[/red]")
        raise typer.Exit(1)
    conn = _db()
    sid = engine.add_story(conn, match, title, situation, task, action, result)
    console.print(f"Saved story #{sid} under {match}.")
    missing = engine.uncovered_competencies(conn)
    console.print(f"Still uncovered: {', '.join(missing)}" if missing else "All competencies covered.")


@star_app.command("list")
def star_list():
    """List saved STAR stories and uncovered competencies."""
    conn = _db()
    rows = conn.execute("SELECT id, competency, title, created FROM star_stories ORDER BY id").fetchall()
    t = Table(title="STAR stories")
    for c in ("ID", "Competency", "Title", "Created"):
        t.add_column(c)
    for r in rows:
        t.add_row(str(r["id"]), r["competency"], r["title"], r["created"])
    console.print(t)
    missing = engine.uncovered_competencies(conn)
    if missing:
        console.print(f"Uncovered: {', '.join(missing)}")


@star_app.command("show")
def star_show(story_id: int):
    """Print a story in full for rehearsal."""
    conn = _db()
    r = conn.execute("SELECT * FROM star_stories WHERE id=?", (story_id,)).fetchone()
    if r is None:
        console.print(f"[red]No story {story_id}[/red]")
        raise typer.Exit(1)
    console.print(f"[bold]{r['title']}[/bold] ({r['competency']})")
    for k in ("situation", "task", "action", "result"):
        console.print(f"[cyan]{k.upper()}[/cyan]: {r[k]}")
