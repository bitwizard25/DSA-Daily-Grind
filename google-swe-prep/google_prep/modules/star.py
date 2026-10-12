"""Googlyness & behavioural vault (spec Feature 5)."""
import random

from rich.console import Group
from rich.table import Table
from rich.text import Text

from google_prep import db
from google_prep.core import clock, readiness
from google_prep.models import StarStory
from google_prep.ui.formatters import card, keys as key_bar
from sqlmodel import select

DIMENSIONS = {
    "Ambiguity": "Thriving in Ambiguity",
    "Ownership": "Ownership & Bias for Action",
    "Collaboration": "Collaboration & Navigating Disagreement",
    "Humility": "Technical Humility & Continuous Learning",
}


def resolve_dimension(text: str) -> str | None:
    t = text.strip().lower()
    for key, label in DIMENSIONS.items():
        if t in (key.lower(), label.lower()) or (len(t) >= 3 and key.lower().startswith(t)):
            return key
    return None


def add(session, dimension: str, title: str, situation: str, task: str, action: str, result: str) -> StarStory:
    story = StarStory(dimension=dimension, title=title, situation=situation, task=task, action=action,
                      result=result, updated_at=clock.now())
    session.add(story)
    session.commit()
    return story


def list_view(session) -> Group:
    stories = session.exec(select(StarStory).order_by(StarStory.dimension, StarStory.id)).all()
    table = Table(title="STAR Story Vault", header_style="accent", expand=False)
    for c in ("ID", "Dimension", "Title", "Updated"):
        table.add_column(c)
    for s in stories:
        table.add_row(str(s.id), DIMENSIONS.get(s.dimension, s.dimension), s.title, f"{s.updated_at:%Y-%m-%d}")
    counts = readiness.star_counts(session)
    need = readiness.STARS_PER_DIMENSION
    cov = "   ".join(f"[{'ok' if counts[d] >= need else 'warn'}]{d} {min(counts[d], need)}/{need}[/]" for d in DIMENSIONS)
    return Group(table, Text.from_markup(f"Coverage: {cov}"))


def story_view(s: StarStory) -> Group:
    return card(f"{s.title}  [{DIMENSIONS.get(s.dimension, s.dimension)}]",
                [f"[accent]SITUATION[/accent] {s.situation}", f"[accent]TASK[/accent]      {s.task}",
                 f"[accent]ACTION[/accent]    {s.action}", f"[accent]RESULT[/accent]    {s.result}"])


def weakest_dimension(session) -> str:
    counts = readiness.star_counts(session)
    return min(DIMENSIONS, key=lambda d: (counts[d], list(DIMENSIONS).index(d)))


def review_prompt(session, dimension: str | None = None, rng: random.Random | None = None) -> tuple[str, str]:
    dim = dimension or weakest_dimension(session)
    prompts = db.load_json("googlyness_prompts.json")[dim]
    return dim, (rng or random).choice(prompts)


def review_view(session, dim: str, prompt: str) -> Group:
    return Group(card(f"GOOGLYNESS QUESTION · {DIMENSIONS[dim]}",
                      [f'"{prompt}"', "", "Out loud, under 2 min: Situation > Task > Action > Result.",
                       "Quantify the result. Name what you would change."]),
                 key_bar(("Enter", "Show my stories"), ("q", "Exit")))


def stories_for(session, dim: str) -> list[StarStory]:
    return list(session.exec(select(StarStory).where(StarStory.dimension == dim)).all())
