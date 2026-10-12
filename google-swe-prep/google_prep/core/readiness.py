"""Readiness Index = 0.40 Pattern Mastery + 0.30 Mock Speed + 0.15 Dry-Run Accuracy + 0.15 STAR."""
from dataclasses import dataclass
from datetime import date, timedelta

from sqlmodel import Session, col, select

from google_prep.core import scheduler
from google_prep.models import Attempt, MockSession, ReadinessSnapshot, StarStory

WEIGHTS = {"pattern_mastery": 0.40, "mock_speed": 0.30, "dry_run": 0.15, "star": 0.15}
STAR_DIMENSIONS = ["Ambiguity", "Ownership", "Disagreement", "Humility"]
STARS_PER_DIMENSION = 2  # two stories per dimension so follow-up questions do not force a repeat
MOCK_WINDOW = 5
DRY_RUN_WINDOW = 10
CODING_SHARE = 35 / 45  # code should be done by 35:00 to leave time for the follow-up


@dataclass
class Readiness:
    pattern_mastery: float
    mock_speed: float
    dry_run: float
    star: float

    @property
    def index(self) -> float:
        return 100 * sum(WEIGHTS[k] * getattr(self, k) for k in WEIGHTS)


def pattern_mastery(session: Session) -> float:
    mastery = scheduler.category_mastery(session)
    problems = sum(r["problem"] for r in mastery.values())
    total = sum(r["total"] for r in mastery.values())
    drills = sum(r["drill"] for r in mastery.values())
    drill_total = sum(r["drills"] for r in mastery.values())
    p = problems / total if total else 0.0
    d = drills / drill_total if drill_total else 0.0
    return 0.7 * p + 0.3 * d


def mock_score(m: MockSession) -> float:
    if not m.solved:
        return 0.0
    target = m.limit_minutes * 60 * CODING_SHARE
    return min(1.0, target / max(1, m.duration_seconds))


def mock_speed(session: Session) -> float:
    rows = session.exec(select(MockSession).order_by(col(MockSession.id).desc()).limit(MOCK_WINDOW)).all()
    return sum(mock_score(m) for m in rows) / len(rows) if rows else 0.0


def dry_run_accuracy(session: Session) -> float:
    rows = session.exec(
        select(Attempt).where(col(Attempt.dry_run_ok).is_not(None))
        .order_by(col(Attempt.id).desc()).limit(DRY_RUN_WINDOW)
    ).all()
    return sum(1 for a in rows if a.dry_run_ok) / len(rows) if rows else 0.0


def star_counts(session: Session) -> dict[str, int]:
    counts = {d: 0 for d in STAR_DIMENSIONS}
    for s in session.exec(select(StarStory)).all():
        if s.dimension in counts:
            counts[s.dimension] += 1
    return counts


def star_coverage(session: Session) -> float:
    counts = star_counts(session)
    target = STARS_PER_DIMENSION * len(STAR_DIMENSIONS)
    return sum(min(c, STARS_PER_DIMENSION) for c in counts.values()) / target


def compute(session: Session) -> Readiness:
    return Readiness(pattern_mastery(session), mock_speed(session),
                     dry_run_accuracy(session), star_coverage(session))


# ---- trend ------------------------------------------------------------------

SPARK = "▁▂▃▄▅▆▇█"


def record_snapshot(session: Session, day: date, r: Readiness | None = None) -> Readiness:
    """Store today's readiness (one row per day, last write wins)."""
    r = r or compute(session)
    session.merge(ReadinessSnapshot(date=day.isoformat(), score=r.index, pattern_mastery=r.pattern_mastery,
                                    mock_speed=r.mock_speed, dry_run=r.dry_run, star=r.star))
    session.commit()
    return r


def history(session: Session, today: date, days: int = 14) -> list[float]:
    start = (today - timedelta(days=days - 1)).isoformat()
    rows = session.exec(select(ReadinessSnapshot).where(ReadinessSnapshot.date >= start)
                        .where(ReadinessSnapshot.date <= today.isoformat())
                        .order_by(ReadinessSnapshot.date)).all()
    return [row.score for row in rows]


def weekly_change(session: Session, today: date) -> float | None:
    """Change vs. the latest snapshot at least 7 days old."""
    old = session.exec(select(ReadinessSnapshot)
                       .where(ReadinessSnapshot.date <= (today - timedelta(days=7)).isoformat())
                       .order_by(col(ReadinessSnapshot.date).desc())).first()
    now = session.get(ReadinessSnapshot, today.isoformat())
    if old is None or now is None:
        return None
    return now.score - old.score


def sparkline(values: list[float]) -> str:
    return "".join(SPARK[min(len(SPARK) - 1, int(v / 100 * len(SPARK)))] for v in values)
