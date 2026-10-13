"""Daily Command Center (spec Feature 1): morning warmup, evening deep work, micro-mode."""
from datetime import datetime, timedelta

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from google_prep.core import readiness, scheduler
from google_prep.ui.formatters import banner, bar, crumbs, keys as key_bar, nav
from google_prep.ui.theme import BLUE, GREEN, YELLOW

SHORT_PHASE = {0: "Python Fast-Track", 1: "Foundation", 2: "Core Patterns", 3: "Onsite Loop"}
WARMUP_DRILLS = 5


def header(session, now: datetime, title: str = "TODAY") -> Group:
    day = scheduler.day_number(session, now.date())
    phase = scheduler.phase_for_day(day)
    last = scheduler.PHASES[phase][2]
    day_txt = f"Day {day}/{last}" if day <= scheduler.PLAN_DAYS else f"Day {day}, past the 8-week plan"
    r = readiness.record_snapshot(session, now.date()).index
    change = readiness.weekly_change(session, now.date())
    delta = "" if change is None else f" [{'ok' if change >= 0 else 'danger'}]({change:+.0f} this week)[/]"
    s = scheduler.streak(session, now.date())
    return banner(title,
                  f"Phase {phase}: {SHORT_PHASE[phase]} · {day_txt} │ [ok]🔥 {s}-day streak[/ok]",
                  f"Readiness [accent]{r:.0f}%[/accent] {bar(r / 100, 6)}{delta}")


def status_line(session, now: datetime) -> Text:
    days_left = (scheduler.interview_date(session) - now.date()).days
    p = scheduler.pace(session, now.date())
    if p["solved"] >= p["total"]:
        pace = "[ok]bank done[/ok]"
    elif p["behind"]:
        pace = f"[danger]behind by {p['behind']}[/danger] · need {p['per_day']:.1f}/day"
    else:
        pace = "[ok]on track[/ok]"
    return Text.from_markup(f"  {now:%a %d %b %H:%M} │ [warn]{days_left} days to interview[/warn] │ "
                            f"Pace {p['solved']}/{p['total']} · {pace}")


def recency_text(session, problem) -> str:
    if problem.category == scheduler.FOUNDATIONS:
        total = len(session.exec(scheduler.select(scheduler.Problem.id)
                                 .where(scheduler.Problem.category == scheduler.FOUNDATIONS)).all())
        return f"foundations {problem.sequence} of {total}: a building block the Google problems reuse"
    window, freq = scheduler.recency(problem, scheduler.recency_overrides(session))
    return scheduler.WINDOW_LABEL.get(window, "") + (f" (freq {freq:.0f})" if freq else "")


def plan(session, now: datetime, mode: str = "normal") -> dict:
    """Everything the screen and the TUI need, without rendering."""
    day = scheduler.day_number(session, now.date())
    phase = scheduler.phase_for_day(day)
    sched = scheduler.schedule_for(session, now.date())
    drills = scheduler.pick_warmup(session, now, WARMUP_DRILLS)
    target = scheduler.pick_target_problem(session, phase, now)
    yesterday = scheduler.attempts_on(session, now.date() - timedelta(days=1))
    return {"phase": phase, "slot": scheduler.time_slot(now), "schedule": sched, "drills": drills,
            "target": target, "yesterday": yesterday, "mock": phase == 3, "mode": mode,
            "micro": [d for d in (scheduler.pick_drills(session, now, 1, "pattern")
                                  + scheduler.pick_drills(session, now, 1, "bug_hunt"))],
            "due": scheduler.due_problems(session, now)}


def next_card(title: str, bullets: list[str], start_label: str | None, style: str = BLUE) -> Panel:
    """The one dominant action on the home screen (Krug's 'Where do I start?')."""
    body = Text.from_markup("\n".join(f"• {b}" for b in bullets), style="body")
    if start_label:
        body.append("\n\n")
        body.append(f" [Enter] {start_label} ", style=f"reverse bold {style}")
    return Panel(body, title=f"[header]▶ NEXT: {title}[/header]", title_align="left", box=box.HEAVY,
                 border_style=style, padding=(1, 2))


def decide(session, now: datetime, mode: str = "normal") -> dict:
    """The single next action for today, as plain data shared by the CLI and the GUI.
    Bullets use **bold** / `code` inline markup. `action` is what Enter / the main button does."""
    from google_prep.modules import fasttrack  # Phase 0: Days 1-2 serve the C++ -> Python modules

    p = plan(session, now, mode)
    out = {"mode": mode, "kind": "done", "title": "", "bullets": [], "cta": None, "action": None, "tone": "blue",
           "status": []}
    if mode == "micro":
        bullets = [f"{'Bug-hunt' if d.type == 'bug_hunt' else 'Pattern drill'} · ~5 min" for d in p["micro"]]
        return {**out, "kind": "micro", "title": "Micro-Mode · 10 min", "tone": "yellow",
                "bullets": bullets + ["Counts toward your streak"], "cta": "Start 10-min Micro-Mode",
                "action": "micro_start"}
    if mode == "review":
        due = p["due"][:8]
        return {**out, "kind": "review", "title": "Re-solve queue",
                "bullets": [f"{x.id} **{x.title}** ({x.difficulty})" for x in due] or ["Nothing due for a re-solve."],
                "cta": f"Re-solve {due[0].title}" if due else None, "action": "review_start" if due else None,
                "problem_id": due[0].id if due else None}

    sched, target = p["schedule"], p["target"]
    evening_time = scheduler.get_setting(session, "evening_time", "7:30 PM")
    day_no = scheduler.day_number(session, now.date())
    fast_day = fasttrack.pending_day(session, day_no)
    if fast_day is not None:
        lesson = fasttrack.load_day(fast_day)
        return {**out, "kind": "fasttrack", "day": fast_day, "minutes": lesson["minutes"],
                "title": f"Python fast-track · Day {fast_day} of 2 · {lesson['minutes']} min",
                "bullets": [f"{sum(not s.get('optional') for s in lesson['sections'])} short steps: "
                            "a tiny example, what happened, one line to remember",
                            "One quick check after each step · new words come a few at a time",
                            lesson["subtitle"]],
                "cta": f"Start Day {fast_day}: {lesson['title']}", "action": "fasttrack"}
    if day_no <= fasttrack.DAYS[-1]:
        return {**out, "kind": "done", "title": "Done for today", "tone": "green",
                "bullets": ["Fast-track module finished. The streak is safe.",
                            "The Google problem bank starts on Day 3."]}

    morning_next = not sched.morning_completed and p["slot"] != "evening"
    evening_next = not morning_next and not sched.evening_completed and target is not None
    kind = "45-min mock" if p["mock"] else "60-min timed solve"
    if morning_next:
        bugs = sum(d.type == "bug_hunt" for d in p["drills"])
        bullets = [f"{len(p['drills'])} unlabeled drills, mixed topics" + (f" (incl. {bugs} bug-hunt)" if bugs else ""),
                   "Name the **pattern** + **complexity** in under 60s each"]
        mistake = scheduler.dominant_mistake(session)
        if mistake:
            bullets.append(f"Tuned to your top mistake: {scheduler.MISTAKES[mistake]}")
        if p["yesterday"]:
            bullets.append("Re-read yesterday's code: " + ", ".join(x.title for x in p["yesterday"]))
        out.update(kind="warmup", title="Morning warmup · 20 min", bullets=bullets,
                   cta="Start Today's 20-min Warmup", action="warmup", minutes=20)
    elif evening_next:
        signal = recency_text(session, target)
        out.update(kind="deep", title=f"Evening deep work · {kind}", minutes=45 if p["mock"] else 60,
                   bullets=[f"{target.id} **{target.title}** · {target.difficulty} · "
                            + (scheduler.CATEGORIES[target.category] if target.category == scheduler.FOUNDATIONS
                               else f"{scheduler.CATEGORIES[target.category]} (weakest topic)"),
                            f"Target **{target.time_target}** time · **{target.space_target}** space",
                            signal[:1].upper() + signal[1:]],
                   cta=f"Start {kind.title()}", action="deep", problem_id=target.id, practice=not p["mock"])
    else:
        out.update(kind="done", title="Done for today", tone="green",
                   bullets=["Both sessions logged. The streak is safe.", "Extra rep? Mock or drills."])

    for d in fasttrack.unfinished(session):
        out["status"].append(("Open", f"Python fast-track Day {d} not finished"))
    if not morning_next:
        out["status"].append(("Done" if sched.morning_completed else "Not done", "Morning warmup"))
    if not evening_next and target is not None:
        out["status"].append(("Done" if sched.evening_completed else "Later",
                              f"Evening · {evening_time} · {kind} · {target.title}"))
    return out


TONE = {"blue": BLUE, "green": GREEN, "yellow": YELLOW}


def build(session, now: datetime, mode: str = "normal") -> tuple[Group, dict[str, str]]:
    """Home screen: one dominant next-action card; everything else is a one-line status."""
    from google_prep.modules.fasttrack import inline_markup

    label = {"micro": "Micro-Mode", "review": "Re-solve Queue"}.get(mode, "Today")
    parts = [crumbs(scheduler.breadcrumb(session, now.date(), label)), nav("Today"),
             header(session, now, label.upper()), status_line(session, now), Text("")]
    actions = {"m": "micro_start", "t": "today", "f": "fasttrack", "d": "warmup", "w": "mock", "r": "roadmap",
               "s": "star_list", "q": "quit"}
    footer = [("m", "Micro-Mode (10 min)"), ("q", "Quit")]
    d = decide(session, now, mode)
    if d["action"]:
        actions["enter"] = d["action"]
    parts.append(next_card(d["title"].upper(), [inline_markup(b) for b in d["bullets"]], d["cta"], TONE[d["tone"]]))
    for tag, text in d["status"]:
        style = "ok" if tag == "Done" else "warn" if tag == "Open" else "muted"
        hint = " · [f] open it" if tag == "Open" else ""
        parts.append(Text.from_markup(f"  [{style}]{tag:<8}[/{style}] [muted]{inline_markup(text)}{hint}[/muted]"))
    if mode == "micro":
        footer = [("q", "Quit")]
    parts += [Text(""), key_bar(*([("Enter", "Start")] if d["action"] else []), *footer)]
    return Group(*parts), actions
