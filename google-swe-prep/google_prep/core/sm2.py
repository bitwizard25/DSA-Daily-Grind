"""SuperMemo-2 spaced repetition. Grades follow the recall screen:
1 blanked, 2 wrong pattern, 3 slow, 4 good, 5 instant. Grades below 3 reset the card."""
from dataclasses import dataclass
from datetime import datetime, timedelta

MIN_EF = 1.3
MASTERED_REPS = 3


@dataclass
class CardState:
    reps: int = 0
    easiness_factor: float = 2.5
    interval_days: int = 1


def review(state: CardState, grade: int) -> CardState:
    if not 1 <= grade <= 5:
        raise ValueError("grade must be 1-5")
    ef = state.easiness_factor + (0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02))
    ef = max(MIN_EF, ef)
    if grade < 3:
        return CardState(reps=0, easiness_factor=ef, interval_days=1)
    reps = state.reps + 1
    if reps == 1:
        interval = 1
    elif reps == 2:
        interval = 6
    else:
        interval = round(state.interval_days * ef)
    return CardState(reps=reps, easiness_factor=ef, interval_days=interval)


def status_for(reps: int) -> str:
    return "mastered" if reps >= MASTERED_REPS else "learning"


def next_date(now: datetime, state: CardState) -> datetime:
    return now + timedelta(days=state.interval_days)


def score(reps: int) -> float:
    """0-1 mastery contribution of one card."""
    return min(reps, MASTERED_REPS) / MASTERED_REPS
