"""8-Week North Star tracker (spec Feature 4)."""
from datetime import date

from rich.console import Group
from rich.table import Table
from rich.text import Text

from google_prep.core import readiness, scheduler
from google_prep.ui.formatters import banner, bar, crumbs, gauge, keys as key_bar, nav, pct, rule


def build(session, today: date) -> tuple[Group, dict[str, str]]:
    r = readiness.record_snapshot(session, today)
    day = scheduler.day_number(session, today)
    head = banner("ROADMAP · 8-WEEK NORTH STAR",
                  f"Readiness [accent]{r.index:.0f}%[/accent]",
                  f"Interview [header]{scheduler.interview_date(session):%b %d, %Y}[/header]")

    parts = [crumbs(scheduler.breadcrumb(session, today, "Roadmap")), nav("Roadmap"), head, Text("")]
    comp = Table.grid(padding=(0, 3))
    for name, value, weight in [("Pattern Mastery", r.pattern_mastery, 40), ("Mock Speed", r.mock_speed, 30),
                                ("Dry-Run Accuracy", r.dry_run, 15), ("STAR Stories", r.star, 15)]:
        comp.add_row(f"[body]{name}[/body] [muted]({weight}%)[/muted]", f"[accent]{bar(value)}[/accent] {pct(value)}")
    hist = readiness.history(session, today)
    change = readiness.weekly_change(session, today)
    if len(hist) > 1:
        delta = "" if change is None else f"   [{'ok' if change >= 0 else 'danger'}]{change:+.0f} vs. last week[/]"
        comp.add_row("[body]Trend (14 days)[/body]", f"[accent]{readiness.sparkline(hist)}[/accent]{delta}")
    pace = scheduler.pace(session, today)
    pace_txt = (f"[danger]behind by {pace['behind']}[/danger]" if pace["behind"] else "[ok]on track[/ok]")
    comp.add_row("[body]Problem pace[/body]",
                 f"{pace['solved']}/{pace['total']} solved, plan says {pace['expected']} by today: {pace_txt}")
    side = Table.grid(padding=(0, 4))
    side.add_row(gauge(r.index / 100), comp)
    # Krug ch 13: no false precision; show exactly how the number is made
    terms = [(w / 100, getattr(r, k)) for k, w in (("pattern_mastery", 40), ("mock_speed", 30), ("dry_run", 15),
                                                    ("star", 15))]
    formula = " + ".join(f"{w:.2f}×{v * 100:.0f}%" for w, v in terms)
    parts += [rule("  READINESS INDEX COMPONENTS"), side,
              Text.from_markup(f"  [muted]Readiness = {formula} = [/muted][accent]{r.index:.0f}%[/accent]"), Text("")]

    phases = Table.grid(padding=(0, 2))
    for phase, (name, first, last) in scheduler.PHASES.items():
        span = last - first + 1
        progress = min(1.0, max(0.0, (day - first + 1) / span))
        state = ("[ok]DONE[/ok]" if day > last else "[warn]IN PROGRESS[/warn]" if day >= first
                 else "[muted]UPCOMING[/muted]")
        phases.add_row(f"[accent]{bar(progress)}[/accent]", f"Phase {phase}: {name}",
                       f"[muted](Days {first}–{last})[/muted]", state)
    parts += [rule("  MILESTONE PHASES"), phases, Text("")]

    topics = Table.grid(padding=(0, 2))
    for cat, row in scheduler.category_mastery(session).items():
        topics.add_row(scheduler.CATEGORIES[cat], f"[accent]{bar(row['score'], 20)}[/accent]",
                       f"{pct(row['score'])} ({row['solved']}/{row['total']} solved)")
    counts = readiness.star_counts(session)
    target = readiness.STARS_PER_DIMENSION * len(counts)
    have = sum(min(c, readiness.STARS_PER_DIMENSION) for c in counts.values())
    topics.add_row("Googlyness STAR Stories", f"[accent]{bar(r.star, 20)}[/accent]",
                   f"{pct(r.star)} ({have}/{target} completed)")
    parts += [rule("  TOPIC MASTERY BREAKDOWN"), topics, Text("")]

    mistakes = scheduler.mistake_profile(session)
    if any(mistakes.values()):
        worst = max(mistakes.values())
        table = Table.grid(padding=(0, 2))
        for key, label in scheduler.MISTAKES.items():
            n = mistakes[key]
            table.add_row(label, f"[{'danger' if n == worst else 'muted'}]{'■' * n or '·'} {n}[/]")
        parts += [rule("  MISTAKE PROFILE (last 20 logged)"), table, Text("")]
    parts += [key_bar(("b", "Back"), ("w", "Mock"), ("s", "STAR"), ("q", "Quit"))]
    return Group(*parts), {"b": "today", "t": "today", "f": "fasttrack", "d": "warmup", "w": "mock", "r": "roadmap",
                           "s": "star_list",
                           "m": "micro_start", "q": "quit"}
