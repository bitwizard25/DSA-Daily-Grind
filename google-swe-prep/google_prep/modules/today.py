"""Daily Command Center (spec Feature 1): morning warmup, evening deep work, micro-mode."""
from datetime import datetime, timedelta

from rich.console import Group
from rich.text import Text

from google_prep.core import readiness, scheduler
from google_prep.ui.formatters import banner, bar, card, crumbs, keys as key_bar
from google_prep.ui.theme import GREEN, YELLOW

SHORT_PHASE = {1: "Foundation", 2: "Core Patterns", 3: "Onsite Loop"}
WARMUP_DRILLS = 5


def header(session, now: datetime) -> Group:
    day = scheduler.day_number(session, now.date())
    phase = scheduler.phase_for_day(day)
    last = scheduler.PHASES[phase][2]
    day_txt = f"Day {day}/{last}" if day <= scheduler.PLAN_DAYS else f"Day {day}, past the 8-week plan"
    r = readiness.record_snapshot(session, now.date()).index
    change = readiness.weekly_change(session, now.date())
    delta = "" if change is None else f" [{'ok' if change >= 0 else 'danger'}]({change:+.0f} this week)[/]"
    s = scheduler.streak(session, now.date())
    return banner("GOOGLE SWE II OS",
                  f"Phase {phase}: {SHORT_PHASE[phase]} ({day_txt}) │ Streak: [ok]🔥 {s} Day{'s' if s != 1 else ''}[/ok]",
                  f"Readiness: [accent]{r:.0f}%[/accent] {bar(r / 100, 6)}{delta}")


def pace_line(session, now: datetime) -> Text:
    p = scheduler.pace(session, now.date())
    if p["solved"] >= p["total"]:
        status = "[ok]bank complete: mocks and re-solves only[/ok]"
    elif p["behind"]:
        status = f"[danger]behind by {p['behind']}[/danger] · need {p['per_day']:.1f}/day"
    else:
        status = f"[ok]on track[/ok] · need {p['per_day']:.1f}/day"
    return Text.from_markup(f"  Pace: {p['solved']}/{p['total']} solved · {status}")


def status_line(session, now: datetime) -> Text:
    target = scheduler.get_setting(session, "target", "Google SWE II")
    days_left = (scheduler.interview_date(session) - now.date()).days
    return Text.from_markup(f"  {now:%a %d %b · %I:%M %p}  │  {target}  │  "
                            f"[warn]{days_left} days to interview[/warn]")


def recency_text(session, problem) -> str:
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


def build(session, now: datetime, mode: str = "normal") -> tuple[Group, dict[str, str]]:
    """Return the screen and its key -> action map."""
    p = plan(session, now, mode)
    label = {"micro": "Micro-Mode", "review": "Re-solve Queue"}.get(mode, "Today")
    parts = [crumbs(scheduler.breadcrumb(session, now.date(), label)), header(session, now),
             status_line(session, now), pace_line(session, now), Text("")]
    actions: dict[str, str] = {}

    if mode == "micro":
        lines = []
        for i, d in enumerate(p["micro"], 1):
            kind = "Bug-hunt drill" if d.type == "bug_hunt" else "Pattern recall drill"
            lines.append(f"{i}. {kind} ({scheduler.CATEGORIES[d.category]}) - ~5-7 min")
        parts.append(card("⚡ EMERGENCY MICRO-MODE [15 Min]", lines, style=YELLOW))
        parts += [Text(""), key_bar(("Enter", "Start Micro-Mode"), ("r", "View Roadmap"), ("q", "Exit"))]
        return Group(*parts), {"enter": "micro_start", "r": "roadmap", "q": "quit"}

    if mode == "review":
        due = p["due"][:8]
        lines = [f"{x.id}  {x.title} ({x.difficulty})" for x in due] or ["Nothing due."]
        parts.append(card("🔁 DUE FOR RE-SOLVE (spaced repetition)", lines))
        parts += [Text("")]
        if due:
            actions["enter"] = "review_start"
            parts.append(key_bar(("Enter", f"Re-solve {due[0].id}"), ("r", "View Roadmap"), ("q", "Exit")))
        else:
            parts.append(key_bar(("r", "View Roadmap"), ("q", "Exit")))
        actions.update({"r": "roadmap", "q": "quit"})
        return Group(*parts), actions

    sched = p["schedule"]
    first = p["drills"][0] if p["drills"] else None
    topic = scheduler.CATEGORIES[first.category] if first else "-"
    if sched.morning_completed:
        mlines = ["[ok]✓ Completed[/ok]"]
    else:
        bugs = sum(d.type == "bug_hunt" for d in p["drills"])
        mix = f" incl. {bugs} bug-hunt{'s' if bugs != 1 else ''}" if bugs else ""
        mlines = [f"Task: {len(p['drills'])} recall drills{mix} · first topic: {topic}",
                  "Goal: pattern + complexity in under 60s each"]
        mistake = scheduler.dominant_mistake(session)
        if mistake:
            mlines.append(f"[warn]Tuned to top mistake: {scheduler.MISTAKES[mistake]}[/warn]")
        if p["yesterday"]:
            mlines.append("Re-read yesterday's code: " + ", ".join(x.title for x in p["yesterday"]))
    parts += [card("🌅 MORNING WARMUP [20 Min]", mlines, style=GREEN if sched.morning_completed else "#4285F4"),
              Text("")]

    target = p["target"]
    evening_time = scheduler.get_setting(session, "evening_time", "7:30 PM")
    if target is None:
        elines = ["Bank complete. Run a mock on any topic."]
    else:
        kind = "45-min mock" if p["mock"] else "60-min timed solve"
        status = ("[ok]✓ Completed[/ok]" if sched.evening_completed else
                  "[warn]Ready now[/warn]" if p["slot"] == "evening" else "[muted]Pending evening[/muted]")
        elines = [f"Task: {kind} · {target.id} {target.title}",
                  f"Topic: {scheduler.CATEGORIES[target.category]} (your weakest)",
                  f"Target: {target.time_target} time, {target.space_target} space",
                  f"Signal: {recency_text(session, target)}",
                  f"Status: {status}"]
    parts += [card(f"🌙 EVENING DEEP WORK [Scheduled for {evening_time}]", elines,
                   style=GREEN if sched.evening_completed else "#4285F4"), Text("")]

    if p["slot"] == "evening" or sched.morning_completed:
        if target is not None and not sched.evening_completed:
            actions["enter"] = "deep"
            label = "Start Deep Work"
        else:
            label = None
    else:
        actions["enter"] = "warmup"
        label = "Start Warmup"
    hints = ([("Enter", label)] if label else []) + [("m", "Switch to Micro-Mode"), ("r", "View Roadmap"), ("q", "Exit")]
    parts.append(key_bar(*hints))
    actions.update({"m": "micro", "r": "roadmap", "q": "quit"})
    return Group(*parts), actions
