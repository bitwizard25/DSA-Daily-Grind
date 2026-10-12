"""Rich building blocks shared by every screen and the Textual dashboard."""
from rich import box
from rich.console import Console, Group, RenderableType
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from google_prep.ui.theme import BLUE, THEME

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


def crumbs(path: str) -> Text:
    """Breadcrumb line shown at the top of every screen, e.g. Week 2 > Day 3 > Graphs."""
    return Text(path, style="accent")
