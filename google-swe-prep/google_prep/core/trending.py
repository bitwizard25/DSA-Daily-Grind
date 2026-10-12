"""Recent company interview frequency from a public GitHub dataset.

Source: github.com/liquidslr/leetcode-company-wise-problems (one CSV per time window,
scraped from LeetCode company tags). It shows what people *reported* being asked, so treat
it as a priority signal, not a list to memorise: Google rotates and retires leaked questions.
`sync` is the only feature that needs internet; everything else works offline."""
import csv
import io
import re
import urllib.parse
import urllib.request
from typing import Callable

from sqlmodel import Session, delete, select

from google_prep.core import scheduler
from google_prep.models import Problem, TrendingProblem

DEFAULT_SOURCE = "https://raw.githubusercontent.com/liquidslr/leetcode-company-wise-problems/main"
WINDOWS = [("30d", "1. Thirty Days"), ("3m", "2. Three Months"), ("6m", "3. Six Months"),
           ("6m+", "4. More Than Six Months")]
TOPIC_TO_CATEGORY = [  # first match wins; anything else is outside the six Google categories
    (("monotonic",), "monotonic"),
    (("union-find", "graph", "breadth-first", "topological", "shortest path"), "graphs"),
    (("tree", "trie"), "trees"),
    (("heap", "priority queue"), "heaps"),
    (("line sweep", "interval"), "intervals"),
    (("dynamic programming",), "dp"),
]


def fetch_url(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - fixed https source
        return resp.read().decode("utf-8")


def slug_of(link_or_slug: str) -> str:
    text = link_or_slug.strip().rstrip("/")
    match = re.search(r"leetcode\.com/problems/([^/?#]+)", text)
    return match.group(1) if match else text.lower()


def parse(csv_text: str) -> list[dict]:
    return [r for r in csv.DictReader(io.StringIO(csv_text)) if r.get("Link")]


def sync(session: Session, company: str = "Google", source: str = DEFAULT_SOURCE,
         fetch: Callable[[str], str] = fetch_url) -> dict[str, int]:
    """Download every window, keep each problem's most recent window, replace the cache."""
    best: dict[str, TrendingProblem] = {}
    counts = {}
    for label, name in reversed(WINDOWS):  # oldest first, newer windows overwrite
        url = f"{source}/{urllib.parse.quote(company)}/{urllib.parse.quote(name)}.csv"
        rows = parse(fetch(url))
        counts[label] = len(rows)
        for r in rows:
            slug = slug_of(r["Link"])
            best[slug] = TrendingProblem(slug=slug, title=r["Title"], difficulty=r["Difficulty"].title(),
                                         topics=r.get("Topics", ""), window=label,
                                         frequency=float(r.get("Frequency") or 0))
    if not best:
        raise ValueError(f"No data found for company '{company}'")
    session.exec(delete(TrendingProblem))
    for t in best.values():
        session.add(t)
    session.commit()
    scheduler.set_setting(session, "trending_company", company)
    return counts


def hot(session: Session, limit: int = 15, include_easy: bool = False) -> list[TrendingProblem]:
    """Most recently asked problems that fit the six categories and are not in your bank yet."""
    in_bank = {p.slug for p in session.exec(select(Problem)).all()}
    rows = [t for t in session.exec(select(TrendingProblem)).all()
            if t.slug not in in_bank and (include_easy or t.difficulty != "Easy") and category_for(t.topics)]
    rows.sort(key=lambda t: (scheduler.WINDOW_RANK.get(t.window, 4), -t.frequency, t.title))
    return rows[:limit]


def category_for(topics: str) -> str | None:
    low = topics.lower()
    for keywords, cat in TOPIC_TO_CATEGORY:
        if any(k in low for k in keywords):
            return cat
    return None


def add_problem(session: Session, link_or_slug: str, phase: int) -> Problem:
    """Import a synced problem into your bank so it can be picked, logged and mocked."""
    slug = slug_of(link_or_slug)
    existing = session.exec(select(Problem).where(Problem.slug == slug)).first()
    if existing:
        return existing
    t = session.get(TrendingProblem, slug)
    if t is None:
        raise KeyError(f"'{slug}' is not in the synced data. Run `google-prep sync` first.")
    category = category_for(t.topics)
    if category is None:
        raise KeyError(f"'{t.title}' ({t.topics or 'no topics'}) is outside the six Google categories.")
    n = len([p for p in session.exec(select(Problem)).all() if p.id.startswith("X-")]) + 1
    topics = ", ".join(t.topics.split(", ")[:3]) or "Unknown"
    problem = Problem(
        id=f"X-{n:02d}", title=t.title, leetcode_id=0, slug=slug,
        difficulty=t.difficulty if t.difficulty in ("Easy", "Medium", "Hard") else "Medium",
        category=category, phase=phase, key_pattern=topics,
        time_target="state it before coding", space_target="state it before coding",
        google_frequency="Very High" if t.window == "30d" else "High", recency=t.window,
        recent_frequency=t.frequency, statement=f"Read it on https://leetcode.com/problems/{slug}/",
        edge_case="", google_follow_up="How does your solution change if the input does not fit in memory?",
    )
    session.add(problem)
    session.commit()
    return problem
