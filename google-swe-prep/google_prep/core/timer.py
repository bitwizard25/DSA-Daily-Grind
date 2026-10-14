"""Pausable countdown used by the recall drill and the 45-minute mock."""
import time
from typing import Callable

# Mock timeline from the spec, as fractions of the 45-minute round.
MOCK_PHASES = [
    (5 / 45, "Phase 1: Scoping & Clarification"),
    (35 / 45, "Phase 2: Live Plain-Text Implementation"),
    (40 / 45, "Phase 3: Google Follow-up Constraint"),
    (1.0, "Phase 4: Rubric Self-Evaluation"),
]
FOLLOW_UP_AT = 35 / 45


class Countdown:
    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic, elapsed: float = 0.0):
        """`elapsed` restores a recovered session; time while the app was closed is not counted."""
        self.total = seconds
        self.clock = clock
        self.started = clock() - elapsed
        self.paused_at: float | None = None
        self.paused_total = 0.0

    @property
    def paused(self) -> bool:
        return self.paused_at is not None

    def toggle_pause(self) -> None:
        if self.paused_at is None:
            self.paused_at = self.clock()
        else:
            self.paused_total += self.clock() - self.paused_at
            self.paused_at = None

    def elapsed(self) -> float:
        now = self.paused_at if self.paused_at is not None else self.clock()
        return now - self.started - self.paused_total

    def remaining(self) -> float:
        return max(0.0, self.total - self.elapsed())

    def expired(self) -> bool:
        return self.remaining() <= 0

    def fraction(self) -> float:
        return min(1.0, self.elapsed() / self.total) if self.total else 1.0

    def mock_phase(self) -> str:
        f = self.fraction()
        for end, name in MOCK_PHASES:
            if f < end:
                return name
        return MOCK_PHASES[-1][1]


def mmss(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"
