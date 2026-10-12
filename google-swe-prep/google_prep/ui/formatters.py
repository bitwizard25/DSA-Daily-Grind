"""Rich building blocks shared by every screen and the Textual dashboard."""
from rich import box
from rich.console import Console, Group, RenderableType
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from google_prep.ui.theme import BLUE, HEADER, MUTED, SURFACE, TEXT_BLUE, THEME

console = Console(theme=THEME, highlight=False)


def bar(fraction: float, width: int = 10) -> str:
    filled = round(max(0.0, min(1.0, fraction)) * width)
    return "[" + "=" * filled + "-" * (width - filled) + "]"


def pct(fraction: float) -> str:
    return f"{round(fraction * 100)}%"


def banner(title: str, left: str, right: str = "", subtitle: str | None = None) -> Panel:
    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(ratio=1)
    grid.add_column(justify="right")
    grid.add_row(Text.from_markup(left), Text.from_markup(right))
    return Panel(grid, title=f"[header]{title}[/header]", subtitle=subtitle, box=box.ROUNDED,
                 border_style=BLUE)


def card(title: str, lines: list[str], style: str = BLUE) -> Panel:
    body = Text.from_markup("\n".join(lines), style="body")
    return Panel(body, title=f"[header]{title}[/header]", title_align="left", box=box.SQUARE,
                 border_style=style, padding=(0, 2))


def keys(*pairs: tuple[str, str]) -> Text:
    """Key hint bar, e.g. keys(("Enter", "Start"), ("q", "Exit"))."""
    return Text.from_markup("    ".join(f"[key]{escape(f'[{k}]')}[/key] [muted]{escape(label)}[/muted]"
                                       for k, label in pairs))


def rule(label: str) -> Group:
    return Group(Text(label, style="header"), Text("─" * 70, style="muted"))


def show(*items: RenderableType) -> None:
    for item in items:
        console.print(item)


def crumb_text(path: str) -> Text:
    """Breadcrumb with the current (non-clickable) step in bold white (Krug ch 6)."""
    head, sep, last = path.rpartition(" > ")
    out = Text(head + sep, style=TEXT_BLUE)
    out.append(last, style=f"bold {HEADER}")
    return out


def crumbs(path: str) -> Text:
    """Breadcrumb line shown at the top of every screen, e.g. Phase 1 > Week 2 > Day 3 > Graphs."""
    return crumb_text(path)


def gauge(fraction: float, label: str = "READY", radius: int = 4) -> Text:
    """Circular readiness gauge: a ring of segments filled clockwise from 12 o'clock with
    the percentage in the middle. Terminal cells are ~2x taller than wide, so x is doubled."""
    import math

    fraction = max(0.0, min(1.0, fraction))
    rows, cols, segments = 2 * radius + 1, 4 * radius + 3, 28
    cells: dict[tuple[int, int], bool] = {}
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        r = round(radius - radius * math.cos(angle))
        c = round(2 * radius + 1 + 2 * radius * math.sin(angle))
        cells.setdefault((r, c), i < round(fraction * segments))
    colour = "danger" if fraction < 0.4 else "warn" if fraction < 0.7 else "ok"
    centre = {radius - 1: f"{round(fraction * 100)}%", radius + 1: label}
    out = Text()
    for r in range(rows):
        line = [" "] * cols
        styles = [""] * cols
        for (rr, c), filled in cells.items():
            if rr == r:
                line[c], styles[c] = ("●", colour) if filled else ("·", "muted")
        if r in centre:
            word = centre[r]
            start = (cols - len(word)) // 2
            for k, ch in enumerate(word):
                line[start + k], styles[start + k] = ch, "header"
        for ch, st in zip(line, styles):
            out.append(ch, style=st or None)
        out.append("\n")
    out.rstrip()
    return out


SECTIONS = [("t", "Today"), ("d", "Recall"), ("w", "Mock"), ("r", "Roadmap"), ("s", "STAR")]


def nav(active: str, with_keys: bool = True) -> Text:
    """Persistent section bar. The current section is highlighted; inside a timed round the
    keys are hidden because jumping away is not possible there."""
    out = Text()
    for key, name in SECTIONS:
        label = f" [{key}] {name} " if with_keys else f" {name} "
        out.append(label, style=f"bold {SURFACE} on {TEXT_BLUE}" if name == active else MUTED)
        out.append(" ")
    return out
